#!/usr/bin/env python3
"""
Argus CLI - Main entry point for the Agent Resource Guardian.

Usage:
    argus                          # Show help
    argus setup                    # Interactive setup
    argus setup quick              # Quick setup (essential settings only)
    argus setup core               # Core watcher settings
    argus setup modules            # Monitoring modules
    argus setup alerts             # Notifications/alerting
    argus setup full               # Full reconfiguration
    
    argus status                   # Show ARGUS status
    argus start                    # Start ARGUS daemon
    argus stop                     # Stop ARGUS daemon
    argus restart                  # Restart ARGUS daemon
    argus logs                     # View ARGUS logs
    
    argus service install          # Install launchd service (macOS)
    argus service uninstall        # Uninstall launchd service
    argus service status           # Check service status
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
_ARGUS_HOME = Path(os.path.expanduser("~/hermes"))


def _require_tty(command_name: str) -> None:
    """Exit if stdin is not a terminal."""
    if not sys.stdin.isatty():
        print(
            f"Error: 'argus {command_name}' requires an interactive terminal.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_banner():
    """Print the ARGUS CLI banner."""
    print("\033[95m" + "  ┌─────────────────────────────────────────────────────────────┐" + "\033[0m")
    print("\033[95m" + "  │           ⚔ ARGUS - Agent Resource Guardian                 │" + "\033[0m")
    print("\033[95m" + "  │     Unified Supervisor for Hermes Agent Sessions            │" + "\033[0m")
    print("\033[95m" + "  └─────────────────────────────────────────────────────────────┘" + "\033[0m")
    print()


def _print_status():
    """Print ARGUS daemon status."""
    try:
        from argus import is_argus_running, get_argus_running_pid, argus_launchd_status
        
        _print_banner()
        
        running_pid = get_argus_running_pid()
        launchd_status = argus_launchd_status()
        
        print("\033[96m  Status:\033[0m")
        print()
        
        if running_pid:
            print(f"  \033[92m● Running\033[0m (PID: {running_pid})")
        else:
            print(f"  \033[91m● Not running\033[0m")
        
        print()
        print("\033[96m  Launchd Service:\033[0m")
        print(f"    Label: {launchd_status['label']}")
        print(f"    Plist: {'Installed' if launchd_status['plist_exists'] else 'Not installed'}")
        print(f"    PID file: {'Present' if launchd_status['pid_file_exists'] else 'Absent'}")
        
        # Config status
        config_path = _ARGUS_HOME / "config" / "argus.yaml"
        print()
        print("\033[96m  Configuration:\033[0m")
        print(f"    Path: {config_path}")
        print(f"    Status: {'Present' if config_path.exists() else 'Not configured'}")
        
    except Exception as e:
        print(f"\033[91m  Error checking status: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_setup(args):
    """Run interactive setup."""
    _require_tty("setup")
    
    try:
        from argus.setup import (
            run_quick_setup,
            run_full_setup,
            setup_core_settings,
            setup_monitoring_modules,
            setup_notifications,
            load_argus_config,
            save_argus_config,
            print_banner,
            print_summary,
        )
        
        config = load_argus_config()
        
        # Determine which setup to run
        section = getattr(args, "section", None)
        
        if section == "quick":
            print_banner()
            run_quick_setup(config)
            save_argus_config(config)
            print_summary(config)
            
        elif section == "core":
            print_banner()
            setup_core_settings(config)
            save_argus_config(config)
            print_summary(config)
            
        elif section == "modules":
            print_banner()
            setup_monitoring_modules(config)
            save_argus_config(config)
            print_summary(config)
            
        elif section == "alerts":
            print_banner()
            setup_notifications(config)
            save_argus_config(config)
            print_summary(config)
            
        elif section == "full":
            print_banner()
            run_full_setup(config)
            save_argus_config(config)
            
        else:
            # No section specified - check if config exists
            from argus.setup import get_argus_config_path
            config_exists = get_argus_config_path().exists()
            
            if config_exists:
                # Show menu for existing install
                print_banner()
                run_full_setup(config)
                save_argus_config(config)
            else:
                # First-time setup
                print_banner()
                run_quick_setup(config)
                save_argus_config(config)
                print_summary(config)
                
    except KeyboardInterrupt:
        print()
        print("\033[93m  Setup cancelled.\033[0m")
        sys.exit(0)
    except Exception as e:
        print(f"\033[91m  Setup error: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Show ARGUS status."""
    _print_status()


def cmd_start(args):
    """Start ARGUS daemon."""
    try:
        from argus import Argus
        from argus.daemon_mgmt import write_argus_pid_file, is_argus_running
        
        if is_argus_running():
            print("\033[93m  ARGUS is already running.\033[0m")
            return
        
        print("\033[96m  Starting ARGUS...\033[0m")
        argus = Argus()
        
        # Handle signals
        import signal
        def _signal_handler(signum, frame):
            print("\n\033[93m  Received signal, stopping...\033[0m")
            argus.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        
        argus.run()
        
    except Exception as e:
        print(f"\033[91m  Error starting ARGUS: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_stop(args):
    """Stop ARGUS daemon."""
    try:
        from argus import get_argus_running_pid
        import os
        
        pid = get_argus_running_pid()
        if not pid:
            print("\033[93m  ARGUS is not running.\033[0m")
            return
        
        print(f"\033[96m  Stopping ARGUS (PID: {pid})...\033[0m")
        os.kill(pid, 15)  # SIGTERM
        
        # Wait for process to exit
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print("\033[92m  ARGUS stopped.\033[0m")
                return
        
        # Force kill if still running
        print("\033[93m  Force killing...\033[0m")
        os.kill(pid, 9)  # SIGKILL
        print("\033[92m  ARGUS killed.\033[0m")
        
    except Exception as e:
        print(f"\033[91m  Error stopping ARGUS: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_restart(args):
    """Restart ARGUS daemon."""
    cmd_stop(args)
    time.sleep(1)
    cmd_start(args)


def cmd_logs(args):
    """View ARGUS logs."""
    import subprocess
    
    log_dir = _ARGUS_HOME / "logs" / "argus"
    stdout_log = log_dir / "argus.stdout.log"
    stderr_log = log_dir / "argus.stderr.log"
    
    if not stdout_log.exists() and not stderr_log.exists():
        print("\033[93m  No logs found.\033[0m")
        return
    
    # Use tail or cat depending on what's available
    if args.follow:
        cmd = ["tail", "-f", str(stderr_log)]
    else:
        cmd = ["cat", str(stderr_log)]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print()


def cmd_service_install(args):
    """Install ARGUS as launchd service."""
    try:
        from argus import argus_launchd_install
        
        print("\033[96m  Installing ARGUS launchd service...\033[0m")
        if argus_launchd_install():
            print("\033[92m  Service installed successfully.\033[0m")
            print("\033[96m  Start with: launchctl start com.hermes.argus\033[0m")
        else:
            print("\033[91m  Service installation failed.\033[0m")
            sys.exit(1)
    except Exception as e:
        print(f"\033[91m  Error: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_service_uninstall(args):
    """Uninstall ARGUS launchd service."""
    try:
        from argus import argus_launchd_uninstall
        
        print("\033[96m  Uninstalling ARGUS launchd service...\033[0m")
        if argus_launchd_uninstall():
            print("\033[92m  Service uninstalled.\033[0m")
        else:
            print("\033[91m  Service uninstall failed.\033[0m")
            sys.exit(1)
    except Exception as e:
        print(f"\033[91m  Error: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_service_status(args):
    """Check ARGUS service status."""
    try:
        from argus import argus_launchd_status
        
        status = argus_launchd_status()
        
        print("\033[96m  Launchd Service Status:\033[0m")
        print()
        print(f"    Label:       {status['label']}")
        print(f"    Plist:       {status['plist_path']}")
        installed_status = "\033[92mYes\033[0m" if status['plist_exists'] else "\033[91mNo\033[0m"
        pid_status = "\033[92mPresent\033[0m" if status['pid_file_exists'] else "\033[91mAbsent\033[0m"
        print(f"    Installed:   {installed_status}")
        print(f"    PID file:    {pid_status}")
        
        if status['running_pid']:
            print(f"    Running PID: \033[92m{status['running_pid']}\033[0m")
        else:
            print(f"    Status:      \033[91mNot running\033[0m")
            
    except Exception as e:
        print(f"\033[91m  Error: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARGUS - Agent Resource Guardian & Unified Supervisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    argus setup                    # Interactive setup (quick if first time)
    argus setup quick              # Quick setup (essential settings)
    argus setup core               # Core watcher settings only
    argus setup modules            # Monitoring modules only
    argus setup alerts             # Notification settings only
    argus setup full               # Full reconfiguration
    
    argus status                   # Show ARGUS status
    argus start                    # Start ARGUS daemon (foreground)
    argus stop                     # Stop ARGUS daemon
    argus logs                     # View ARGUS logs
    argus logs --follow            # Follow ARGUS logs
    
    argus service install          # Install launchd service (macOS)
    argus service uninstall        # Uninstall launchd service
    argus service status           # Check service status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Interactive setup and configuration",
        description="Configure ARGUS monitoring and recovery settings"
    )
    setup_parser.add_argument(
        "section",
        nargs="?",
        choices=["quick", "core", "modules", "alerts", "full"],
        help="Setup section to run (default: interactive menu)"
    )
    setup_parser.set_defaults(func=cmd_setup)
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show ARGUS daemon status"
    )
    status_parser.set_defaults(func=cmd_status)
    
    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start ARGUS daemon (foreground)"
    )
    start_parser.set_defaults(func=cmd_start)
    
    # Stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop ARGUS daemon"
    )
    stop_parser.set_defaults(func=cmd_stop)
    
    # Restart command
    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart ARGUS daemon"
    )
    restart_parser.set_defaults(func=cmd_restart)
    
    # Logs command
    logs_parser = subparsers.add_parser(
        "logs",
        help="View ARGUS logs"
    )
    logs_parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Follow log output (like tail -f)"
    )
    logs_parser.set_defaults(func=cmd_logs)
    
    # Service subcommand
    service_parser = subparsers.add_parser(
        "service",
        help="Manage ARGUS launchd service (macOS)"
    )
    service_subparsers = service_parser.add_subparsers(dest="service_command", help="Service commands")
    
    # Service install
    service_install_parser = service_subparsers.add_parser("install", help="Install launchd service")
    service_install_parser.set_defaults(func=cmd_service_install)
    
    # Service uninstall
    service_uninstall_parser = service_subparsers.add_parser("uninstall", help="Uninstall launchd service")
    service_uninstall_parser.set_defaults(func=cmd_service_uninstall)
    
    # Service status
    service_status_parser = service_subparsers.add_parser("status", help="Check service status")
    service_status_parser.set_defaults(func=cmd_service_status)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to handler
    args.func(args)


if __name__ == "__main__":
    main()
