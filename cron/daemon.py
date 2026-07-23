"""
Standalone cron ticker daemon.

Runs `cron.scheduler.tick()` every 60 seconds in a loop.
Can be used independently of the gateway.

Usage:
    python -m cron.daemon               # foreground
    python -m cron.daemon --daemon      # background (detach)
    python -m cron.daemon --stop        # stop background instance
"""

import os
import sys
import time
import atexit
import signal
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root
_HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_HERE))


def _get_pid_path() -> Path:
    """Path to the daemon PID file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cron" / ".daemon.pid"


def _get_lock_path() -> Path:
    """Path to the tick lock file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cron" / ".tick.lock"


def daemon_alive() -> bool:
    """Check if cron is running, supporting both the standalone daemon AND the
    in-process ticker used by the gateway.

    Detection order:
      1. Standalone daemon PID file (``~/.hermes/cron/.daemon.pid``) — if
         the process is alive, return ``True`` immediately.
      2. In-process ticker heartbeat (``~/.hermes/cron/ticker_heartbeat``) —
         if the file exists and its epoch is <= 120s old (2x tick interval),
         the gateway's cron scheduler thread is alive, return ``True``.
      3. Neither -> return ``False``.

    The heartbeat fallback closes the gap introduced when the gateway moved
    cron execution into an in-process thread (``InProcessCronScheduler``)
    which never writes a PID file.
    """
    pid_path = _get_pid_path()
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            # Check if process exists (signal 0)
            os.kill(pid, 0)
            return True
        except (ValueError, OSError, ProcessLookupError):
            pid_path.unlink(missing_ok=True)

    # Fallback: check the in-process ticker heartbeat written by
    # InProcessCronScheduler on every tick loop iteration.
    try:
        from cron.jobs import get_ticker_heartbeat_age

        age = get_ticker_heartbeat_age()
        # 120s = 2x the default 60s tick interval — tolerate one missed
        # tick without falsely reporting dead.
        if age is not None and age <= 120.0:
            return True
    except Exception:
        pass

    return False


def stop_daemon() -> bool:
    """Stop a running daemon."""
    pid_path = _get_pid_path()
    if not pid_path.exists():
        print("No daemon PID file found.")
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped cron daemon (PID {pid}).")
        # Wait briefly for cleanup
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.3)
            except ProcessLookupError:
                break
        pid_path.unlink(missing_ok=True)
        return True
    except (ValueError, OSError, ProcessLookupError) as e:
        print(f"Could not stop daemon: {e}")
        pid_path.unlink(missing_ok=True)
        return False


def run_ticker(interval: int = 60, verbose: bool = True):
    """Run the cron ticker loop."""
    from cron.scheduler import tick

    logger.info("Cron ticker started (interval=%ds)", interval)
    last_log: float = 0

    try:
        while True:
            try:
                tick(verbose=False)
            except Exception as e:
                logger.debug("Cron tick error: %s", e)

            # Log heartbeat once per minute (keep-alive signal)
            now = time.monotonic()
            if now - last_log >= 60:
                logger.debug("Cron ticker heartbeat")
                last_log = now

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Cron ticker stopped by signal.")
        _cleanup()
        raise


def _cleanup():
    """Remove PID file on exit."""
    try:
        _get_pid_path().unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_pid():
    """Remove PID file on exit (catches SIGTERM/SIGINT via atexit)."""
    try:
        _get_pid_path().unlink(missing_ok=True)
    except OSError:
        pass


def _signal_handler(signum, frame):
    """Handle termination signals -- raise SystemExit so atexit runs."""
    sys.exit(128 + signum)


def _daemonize():
    """Detach from terminal and run as background daemon."""
    pid = os.fork()
    if pid > 0:
        # Parent: exit so the shell gets control back
        print(f"Cron daemon started (PID {pid}).")
        sys.exit(0)

    # Child: become session leader, detach from terminal
    os.setsid()
    os.umask(0)

    # Second fork to prevent re-acquiring terminal
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Write PID file BEFORE closing FDs (the file is already written+closed)
    pid_path = _get_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    # Register cleanup + signal handlers so PID file is removed on exit
    atexit.register(_cleanup_pid)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Close all file descriptors
    import resource
    maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if maxfd == resource.RLIM_INFINITY:
        maxfd = 1024
    for fd in range(maxfd):
        try:
            os.close(fd)
        except OSError:
            pass

    # Redirect stdio to /dev/null
    os.open("/dev/null", os.O_RDONLY)  # stdin
    os.open("/dev/null", os.O_WRONLY)  # stdout
    os.open("/dev/null", os.O_WRONLY)  # stderr

    # Set up logging to file
    from hermes_logging import setup_logging
    setup_logging(mode="cli")

    # Run ticker
    run_ticker()


def main():
    parser = argparse.ArgumentParser(description="Standalone cron ticker daemon")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--stop", action="store_true", help="Stop background daemon")
    parser.add_argument("--interval", type=int, default=60, help="Tick interval in seconds")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground (default)")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
        return

    if args.daemon:
        if daemon_alive():
            pid_path = _get_pid_path()
            if pid_path.exists():
                pid = pid_path.read_text().strip()
                print(f"Cron daemon already running (PID {pid}).")
            else:
                print("Cron scheduler is already running (in-process gateway ticker).")
            return
        _daemonize()
    else:
        # Foreground mode
        if daemon_alive():
            pid_path = _get_pid_path()
            if pid_path.exists():
                pid = pid_path.read_text().strip()
                print(f"Cron daemon already running (PID {pid}) -- foreground ticker may conflict.")
            else:
                print("Cron scheduler is already running (in-process gateway ticker).")
            print("Stop it first: python -m cron.daemon --stop")
            return

        pid_path = _get_pid_path()
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

        atexit.register(_cleanup)
        run_ticker(interval=args.interval)


if __name__ == "__main__":
    main()