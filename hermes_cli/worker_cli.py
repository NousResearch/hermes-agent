#!/usr/bin/env python3
"""
Command-line interface for managing the Hermes queue worker daemon.

This provides commands to start, stop, and manage standalone queue workers
that execute cron jobs independently of the gateway process.
"""

import os
import sys
import json
import signal
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# Add the parent directory to sys.path so we can import hermes modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_cli.config import get_hermes_home


def get_worker_pid_file(worker_id: str = "default") -> Path:
    """Get the path to the worker PID file."""
    return get_hermes_home() / "workers" / f"{worker_id}.pid"


def get_worker_log_file(worker_id: str = "default") -> Path:
    """Get the path to the worker log file."""
    return get_hermes_home() / "workers" / f"{worker_id}.log"


def is_worker_running(worker_id: str = "default") -> bool:
    """Check if a worker with the given ID is currently running."""
    pid_file = get_worker_pid_file(worker_id)
    
    if not pid_file.exists():
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if the process is actually running
        os.kill(pid, 0)  # This will raise if process doesn't exist
        return True
        
    except (ValueError, OSError, ProcessLookupError):
        # PID file is invalid or process is dead
        # Clean up stale PID file
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False


def start_worker(
    worker_id: str = "default",
    profile: Optional[str] = None,
    poll_interval: int = 30,
    max_concurrent: int = 5,
    daemon: bool = True,
    verbose: bool = False
) -> bool:
    """
    Start a queue worker daemon.
    
    Returns:
        True if worker started successfully, False otherwise
    """
    if is_worker_running(worker_id):
        print(f"Worker '{worker_id}' is already running")
        return False
    
    # Prepare worker directories
    worker_dir = get_hermes_home() / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    
    pid_file = get_worker_pid_file(worker_id)
    log_file = get_worker_log_file(worker_id)
    
    # Build command
    worker_script = Path(__file__).parent / "worker_daemon.py"
    cmd = [
        sys.executable, "-m", "cron.worker_daemon",
        "--worker-id", worker_id,
        "--poll-interval", str(poll_interval),
        "--max-concurrent", str(max_concurrent)
    ]
    
    if profile:
        cmd.extend(["--profile", profile])
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        if daemon:
            # Start as daemon process
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Detach from current session
                    cwd=Path(__file__).parent.parent  # Run from hermes-agent root
                )
            
            # Write PID file
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            
            print(f"Worker '{worker_id}' started (PID: {process.pid})")
            print(f"Logs: {log_file}")
            
        else:
            # Run in foreground
            print(f"Starting worker '{worker_id}' in foreground...")
            os.chdir(Path(__file__).parent.parent)  # Run from hermes-agent root
            subprocess.run(cmd)
        
        return True
        
    except Exception as e:
        print(f"Failed to start worker '{worker_id}': {e}")
        return False


def stop_worker(worker_id: str = "default", timeout: int = 30) -> bool:
    """
    Stop a running queue worker.
    
    Args:
        worker_id: Worker ID to stop
        timeout: Seconds to wait for graceful shutdown
        
    Returns:
        True if worker stopped successfully, False otherwise
    """
    pid_file = get_worker_pid_file(worker_id)
    
    if not is_worker_running(worker_id):
        print(f"Worker '{worker_id}' is not running")
        return True
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        print(f"Stopping worker '{worker_id}' (PID: {pid})")
        
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to exit
        import time
        for _ in range(timeout):
            try:
                os.kill(pid, 0)  # Check if still running
                time.sleep(1)
            except ProcessLookupError:
                # Process is dead
                break
        else:
            # Timeout reached, force kill
            print(f"Worker '{worker_id}' didn't stop gracefully, forcing shutdown")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        
        # Clean up PID file
        try:
            pid_file.unlink()
        except OSError:
            pass
        
        print(f"Worker '{worker_id}' stopped")
        return True
        
    except Exception as e:
        print(f"Failed to stop worker '{worker_id}': {e}")
        return False


def list_workers() -> None:
    """List all registered workers and their status."""
    worker_dir = get_hermes_home() / "workers"
    
    if not worker_dir.exists():
        print("No workers directory found")
        return
    
    pid_files = list(worker_dir.glob("*.pid"))
    
    if not pid_files:
        print("No workers configured")
        return
    
    print("Workers:")
    print("-" * 50)
    
    for pid_file in pid_files:
        worker_id = pid_file.stem
        running = is_worker_running(worker_id)
        status = "RUNNING" if running else "STOPPED"
        
        log_file = get_worker_log_file(worker_id)
        log_size = ""
        if log_file.exists():
            size_bytes = log_file.stat().st_size
            if size_bytes > 1024 * 1024:
                log_size = f" ({size_bytes // (1024 * 1024)}MB)"
            elif size_bytes > 1024:
                log_size = f" ({size_bytes // 1024}KB)"
        
        print(f"  {worker_id:15} {status:8} {log_file}{log_size}")


def show_worker_logs(worker_id: str = "default", tail_lines: int = 50) -> None:
    """Show recent log entries for a worker."""
    log_file = get_worker_log_file(worker_id)
    
    if not log_file.exists():
        print(f"No log file found for worker '{worker_id}'")
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Show last N lines
        recent_lines = lines[-tail_lines:] if len(lines) > tail_lines else lines
        
        print(f"Recent logs for worker '{worker_id}' ({len(recent_lines)} lines):")
        print("-" * 60)
        for line in recent_lines:
            print(line.rstrip())
            
    except Exception as e:
        print(f"Failed to read logs for worker '{worker_id}': {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hermes Queue Worker Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start default worker
  %(prog)s start --worker-id bg     # Start background worker
  %(prog)s start --foreground       # Run in foreground (no daemon)
  %(prog)s stop                     # Stop default worker
  %(prog)s list                     # List all workers
  %(prog)s logs --worker-id bg      # Show logs for background worker
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a queue worker')
    start_parser.add_argument('--worker-id', default='default', help='Worker ID')
    start_parser.add_argument('--profile', help='Hermes profile to use')
    start_parser.add_argument('--poll-interval', type=int, default=30, help='Poll interval in seconds')
    start_parser.add_argument('--max-concurrent', type=int, default=5, help='Max concurrent jobs')
    start_parser.add_argument('--foreground', action='store_true', help='Run in foreground (not daemon)')
    start_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a queue worker')
    stop_parser.add_argument('--worker-id', default='default', help='Worker ID')
    stop_parser.add_argument('--timeout', type=int, default=30, help='Shutdown timeout in seconds')
    
    # List command
    subparsers.add_parser('list', help='List all workers')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show worker logs')
    logs_parser.add_argument('--worker-id', default='default', help='Worker ID')
    logs_parser.add_argument('--tail', type=int, default=50, help='Number of recent lines to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'start':
        success = start_worker(
            worker_id=args.worker_id,
            profile=args.profile,
            poll_interval=args.poll_interval,
            max_concurrent=args.max_concurrent,
            daemon=not args.foreground,
            verbose=args.verbose
        )
        sys.exit(0 if success else 1)
        
    elif args.command == 'stop':
        success = stop_worker(args.worker_id, args.timeout)
        sys.exit(0 if success else 1)
        
    elif args.command == 'list':
        list_workers()
        
    elif args.command == 'logs':
        show_worker_logs(args.worker_id, args.tail)


if __name__ == '__main__':
    main()