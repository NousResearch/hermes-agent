import os
import signal
import subprocess
import threading
import time
import atexit
import logging
from pathlib import Path

# Logger
logger = logging.getLogger(__name__)

# Paths
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
LISTENER_SCRIPT = HERMES_HOME / "citadel_listener.py"
VENV_PYTHON = HERMES_HOME / "wakeword-venv" / "bin" / "python"

# Global state
_listener_process = None
_plugin_context = None
_cli_ref = None
_watcher_thread = None
_watcher_running = False


def _listener_watcher():
    """Background thread: restart citadel listener after agent responds."""
    logger.info("Listener watcher started")
    while True:
        try:
            # Access globals via dict to avoid declaration issues
            proc = globals().get('_listener_process')
            running = globals().get('_watcher_running', False)
            if not running:
                break
            # Check if listener process is dead
            if proc is None or (hasattr(proc, 'poll') and proc.poll() is not None):
                # Listener died (e.g., after wake word). Wait for agent to finish response.
                cli = globals().get('_cli_ref')
                if cli is not None:
                    # Wait until voice processing is done (agent finished responding)
                    for _ in range(60):  # max 60s wait
                        if not getattr(cli, '_voice_processing', False):
                            break
                        time.sleep(1)
                    # Also wait for TTS if enabled
                    if getattr(cli, '_voice_tts', False):
                        cli._voice_tts_done.wait(timeout=30)
                        time.sleep(0.5)
                # Restart listener
                logger.info("Agent response done, restarting listener...")
                start_listener()
                # Avoid rapid restarts if start_listener fails immediately
                time.sleep(3)
            else:
                # Listener alive, just sleep
                time.sleep(2)
        except Exception as e:
            logger.error(f"Watcher error: {e}")
            time.sleep(5)
    logger.info("Listener watcher stopped")


def start_watcher():
    """Start the listener watcher thread."""
    global _watcher_thread
    global _watcher_running
    if _watcher_running:
        return
    _watcher_running = True
    _watcher_thread = threading.Thread(target=_listener_watcher, daemon=True)
    _watcher_thread.start()
    logger.info("Listener watcher thread started")


def stop_watcher():
    """Stop the listener watcher thread."""
    global _watcher_running
    _watcher_running = False
    # Wait briefly for thread to exit
    if _watcher_thread and _watcher_thread.is_alive():
        _watcher_thread.join(timeout=5)
    logger.info("Listener watcher stopped")


def start_listener():
    """Start the citadel listener as a child process of Hermes."""
    global _listener_process

    if _listener_process and _listener_process.poll() is None:
        logger.info(f"Already running (PID: {_listener_process.pid})")
        return

    if not LISTENER_SCRIPT.exists():
        logger.error(f"Listener script not found at {LISTENER_SCRIPT}")
        return

    if not VENV_PYTHON.exists():
        logger.error(f"Virtual environment Python not found at {VENV_PYTHON}")
        return

    try:
        hermes_pid = os.getpid()
        _listener_process = subprocess.Popen(
            [str(VENV_PYTHON), str(LISTENER_SCRIPT), "--pid", str(hermes_pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Started listener (PID: {_listener_process.pid}) targeting Hermes PID {hermes_pid}")
    except Exception as e:
        logger.error(f"Failed to start listener: {e}")


def stop_listener():
    """Stop the citadel listener child process."""
    global _listener_process

    # Stop watcher first
    stop_watcher()

    if not _listener_process:
        return

    pid = _listener_process.pid
    try:
        if _listener_process.poll() is None:
            os.kill(_listener_process.pid, signal.SIGTERM)
            for _ in range(5):
                if _listener_process.poll() is not None:
                    break
                time.sleep(0.5)
            else:
                try:
                    os.kill(_listener_process.pid, signal.SIGKILL)
                except OSError:
                    pass
        logger.info(f"Stopped listener (PID: {pid})")
    except (OSError, ProcessLookupError):
        pass
    finally:
        _listener_process = None


def _ensure_voice_attrs(cli):
    """Dynamically add wake word attributes if they don't exist."""
    if not hasattr(cli, '_wake_word_pending'):
        cli._wake_word_pending = False
    if not hasattr(cli, '_wake_word_one_shot'):
        cli._wake_word_one_shot = False


def _sigusr1_handler(signum, frame):
    """Handle SIGUSR1: toggle voice recording."""
    global _plugin_context, _cli_ref

    if _cli_ref is None:
        try:
            _cli_ref = _plugin_context._manager._cli_ref
        except Exception:
            return

    cli = _cli_ref
    if cli is None:
        return

    _ensure_voice_attrs(cli)

    # If voice mode not enabled, enable it first
    if not getattr(cli, '_voice_mode', False):
        def _enable_and_start():
            try:
                cli._enable_voice_mode()
                if not getattr(cli, '_voice_mode', False):
                    return
                # Start recording immediately on wake word, even if agent is busy
                if getattr(cli, '_voice_processing', False):
                    return
                with getattr(cli, '_voice_lock', threading.Lock()):
                    cli._voice_continuous = False
                cli._wake_word_one_shot = True
                threading.Thread(target=cli._voice_start_recording, daemon=True).start()
            except Exception as e:
                logger.error(f"Enable voice failed: {e}")

        threading.Thread(target=_enable_and_start, daemon=True).start()
        return

    # Voice mode is enabled - toggle recording
    if getattr(cli, '_voice_recording', False):
        # Stop recording
        def _stop_recording():
            try:
                cli._voice_stop_and_transcribe()
            except Exception as e:
                logger.error(f"Stop recording failed: {e}")

        threading.Thread(target=_stop_recording, daemon=True).start()
    else:
        # Start recording
        if getattr(cli, '_voice_processing', False):
            return

        def _start_recording():
            try:
                with getattr(cli, '_voice_lock', threading.Lock()):
                    cli._voice_continuous = False
                cli._wake_word_one_shot = True
                cli._voice_start_recording()
            except Exception as e:
                logger.error(f"Start recording failed: {e}")

        threading.Thread(target=_start_recording, daemon=True).start()


def register(plugin_context):
    """Register the citadel listener plugin with Hermes."""
    global _plugin_context, _cli_ref

    _plugin_context = plugin_context

    # Get CLI reference
    try:
        _cli_ref = plugin_context._manager._cli_ref
    except Exception:
        pass

    logger.info("Plugin registered")

    # Register SIGUSR1 handler
    try:
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, _sigusr1_handler)
            logger.info("SIGUSR1 handler registered")
    except Exception as e:
        logger.error(f"Failed to register SIGUSR1 handler: {e}")

    # Start listener process
    start_listener()

    # Start watcher thread
    start_watcher()

    # Register cleanup hooks
    plugin_context.register_hook("on_session_end", stop_listener)

    # Handle exit
    atexit.register(stop_listener)
    atexit.register(stop_watcher)
