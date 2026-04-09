#!/usr/bin/env python3
"""
Agathos CLI - Main entry point for the Agent Guardian & Health Oversight System.

Usage:
    agathos                          # Show help
    agathos setup                    # Interactive setup
    agathos setup quick              # Quick setup (essential settings only)
    agathos setup core               # Core watcher settings
    agathos setup modules            # Monitoring modules
    agathos setup alerts             # Notifications/alerting
    agathos setup full               # Full reconfiguration

    agathos status                   # Show Agathos status
    agathos start                    # Start Agathos daemon
    agathos stop                     # Stop Agathos daemon
    agathos restart                  # Restart Agathos daemon
    agathos logs                     # View Agathos logs

    agathos service install          # Install system service (macOS only)
    agathos service uninstall        # Uninstall system service
    agathos service status           # Check service status
    agathos service list             # List running services

Platform Support:
- Service management: macOS (launchd) - Linux/Windows planned
- PID file operations: All platforms
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
_AGATHOS_HOME = Path(os.path.expanduser("~/hermes"))

# Platform detection
_IS_MACOS = sys.platform == 'darwin'
_IS_LINUX = sys.platform.startswith('linux')
_IS_WINDOWS = os.name == 'nt' or sys.platform == 'win32'


def _require_tty(command_name: str) -> None:
    """Exit if stdin is not a terminal.

    Args:
        command_name: Name of the command requiring TTY (for error message)
    """
    if not sys.stdin.isatty():
        print(
            f"Error: 'agathos {command_name}' requires an interactive terminal.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_banner() -> None:
    """Print the Agathos CLI banner."""
    print("\033[95m" + "  ┌─────────────────────────────────────────────────────────────┐" + "\033[0m")
    print("\033[95m" + "  │           AGATHOS - Agent Guardian & Health System          │" + "\033[0m")
    print("\033[95m" + "  │     Unified Supervisor for Hermes Agent Sessions            │" + "\033[0m")
    print("\033[95m" + "  └─────────────────────────────────────────────────────────────┘" + "\033[0m")
    print()


def _print_status():
    """Print ARGUS daemon status."""
    try:
        from agathos import is_agathos_running, get_agathos_running_pid, agathos_launchd_status
        
        _print_banner()
        
        running_pid = get_agathos_running_pid()
        launchd_status = agathos_launchd_status()
        
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
        config_path = _AGATHOS_HOME / "config" / "argus.yaml"
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
        from agathos.setup import (
            run_quick_setup,
            run_full_setup,
            setup_core_settings,
            setup_monitoring_modules,
            setup_notifications,
            load_agathos_config,
            save_agathos_config,
            print_banner,
            print_summary,
        )
        
        config = load_agathos_config()
        
        # Determine which setup to run
        section = getattr(args, "section", None)
        
        if section == "quick":
            print_banner()
            run_quick_setup(config)
            save_agathos_config(config)
            print_summary(config)
            
        elif section == "core":
            print_banner()
            setup_core_settings(config)
            save_agathos_config(config)
            print_summary(config)
            
        elif section == "modules":
            print_banner()
            setup_monitoring_modules(config)
            save_agathos_config(config)
            print_summary(config)
            
        elif section == "alerts":
            print_banner()
            setup_notifications(config)
            save_agathos_config(config)
            print_summary(config)
            
        elif section == "full":
            print_banner()
            run_full_setup(config)
            save_agathos_config(config)
            
        else:
            # No section specified - check if config exists
            from agathos.setup import get_agathos_config_path
            config_exists = get_agathos_config_path().exists()
            
            if config_exists:
                # Show menu for existing install
                print_banner()
                run_full_setup(config)
                save_agathos_config(config)
            else:
                # First-time setup
                print_banner()
                run_quick_setup(config)
                save_agathos_config(config)
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
        from agathos import Agathos
        from agathos.daemon_mgmt import write_agathos_pid_file, is_agathos_running
        
        if is_agathos_running():
            print("\033[93m  ARGUS is already running.\033[0m")
            return
        
        print("\033[96m  Starting ARGUS...\033[0m")
        agathos_daemon = Agathos()
        
        # Handle signals
        import signal
        def _signal_handler(signum, frame):
            print("\n\033[93m  Received signal, stopping...\033[0m")
            agathos_daemon.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        
        agathos_daemon.run()
        
    except Exception as e:
        print(f"\033[91m  Error starting ARGUS: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_stop(args):
    """Stop ARGUS daemon."""
    try:
        from agathos import get_agathos_running_pid
        import os
        
        pid = get_agathos_running_pid()
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
    
    log_dir = _AGATHOS_HOME / "logs" / "agathos"
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
    """Install ARGUS as system service."""
    try:
        from agathos import agathos_launchd_install

        # Check platform support
        if not _IS_MACOS:
            print(f"\033[93m  Warning: Service installation is only supported on macOS (darwin).\033[0m")
            print(f"\033[93m  Current platform: {sys.platform}\033[0m")
            print("\033[96m  You can still run Agathos manually using 'agathos start'\033[0m")
            sys.exit(1)

        print("\033[96m  Installing ARGUS system service...\033[0m")
        if agathos_launchd_install():
            print("\033[92m  Service installed successfully.\033[0m")
            print("\033[96m  Start with: launchctl start com.hermes.agathos\033[0m")
        else:
            print("\033[91m  Service installation failed.\033[0m")
            sys.exit(1)
    except Exception as e:
        print(f"\033[91m  Error: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


def cmd_service_uninstall(args):
    """Uninstall ARGUS system service."""
    try:
        from agathos import agathos_launchd_uninstall

        print("\033[96m  Uninstalling ARGUS system service...\033[0m")
        if agathos_launchd_uninstall():
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
        from agathos import agathos_launchd_status

        status = agathos_launchd_status()

        # Use platform-appropriate service name
        service_type = "LaunchAgent" if _IS_MACOS else "System Service"
        print(f"\033[96m  {service_type} Status:\033[0m")
        print()
        print(f"    Label:       {status['label']}")
        print(f"    Config:      {status['plist_path']}")
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


def cmd_service_list(args):
    """List ARGUS service details."""
    try:
        from agathos import agathos_launchd_status, is_agathos_running, get_agathos_running_pid
        import subprocess

        _print_banner()

        print("\033[96m  ARGUS Services:\033[0m")
        print()

        # Get service status
        status = agathos_launchd_status()

        # Service entry with platform-appropriate type
        service_type = "LaunchAgent (user service)" if _IS_MACOS else "System Service"
        print(f"  \033[95mService:\033[0m        com.hermes.agathos")
        print(f"  \033[95mType:\033[0m           {service_type}")
        print(f"  \033[95mPlatform:\033[0m       {sys.platform}")

        # Status with color
        running_pid = get_agathos_running_pid()
        if running_pid:
            print(f"  \033[95mStatus:\033[0m         \033[92mrunning (PID {running_pid})\033[0m")
        else:
            print(f"  \033[95mStatus:\033[0m         \033[91mnot running\033[0m")

        # Installation state
        installed_str = "\033[92myes\033[0m" if status['plist_exists'] else "\033[91mno\033[0m"
        print(f"  \033[95mInstalled:\033[0m      {installed_str}")

        # Paths
        print(f"  \033[95mConfig path:\033[0m    {status['plist_path']}")
        pid_file_str = "\033[92mpresent\033[0m" if status['pid_file_exists'] else "\033[91mabsent\033[0m"
        print(f"  \033[95mPID file:\033[0m       {pid_file_str}")

        # Platform-specific service manager info
        if _IS_MACOS:
            print()
            print("\033[96m  Launchctl Info:\033[0m")
            try:
                result = subprocess.run(
                    ["launchctl", "print", f"gui/{os.getuid()}/{status['label']}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse key info from output
                    for line in result.stdout.split('\n'):
                        if 'state =' in line.lower() or 'pid =' in line or 'last exit' in line.lower():
                            print(f"    {line.strip()}")
                else:
                    print("    Service not loaded in launchctl")
            except Exception:
                print("    Unable to query launchctl")
        elif _IS_LINUX:
            print()
            print("\033[96m  Systemd Info:\033[0m")
            try:
                result = subprocess.run(
                    ["systemctl", "--user", "is-active", status['label']],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"    Service active: {result.stdout.strip()}")
                else:
                    print(f"    Service status: {result.stdout.strip() or 'inactive'}")
            except FileNotFoundError:
                print("    systemctl not found - systemd may not be available")
            except Exception as e:
                print(f"    Unable to query systemd: {e}")

            # Show WSL info if applicable
            if status.get('is_wsl'):
                print()
                print("\033[96m  WSL Detected:\033[0m")
                print(f"    WSL Distro: {os.environ.get('WSL_DISTRO_NAME', 'Unknown')}")
                print("    Windows interoperability enabled")
        elif _IS_WINDOWS:
            print()
            print("\033[96m  Windows Service Info:\033[0m")
            print("    Windows service management is planned.")
            print("    Currently supports manual daemon operation via 'agathos start'.")

        # Next steps hint
        print()
        if not status['plist_exists']:
            if _IS_MACOS:
                print("\033[93m  Run 'agathos service install' to install the service\033[0m")
            else:
                print(f"\033[93m  Service installation not yet supported on {sys.platform}\033[0m")
                print("\033[96m  Run 'agathos start' to start the daemon manually\033[0m")
        elif not running_pid:
            if _IS_MACOS:
                print("\033[93m  Run 'launchctl start com.hermes.agathos' to start the service\033[0m")
            else:
                print("\033[93m  Run 'agathos start' to start the daemon\033[0m")
        else:
            print("\033[96m  Run 'agathos service status' for detailed status\033[0m")

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
    
    argus service list             # List service details
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
        help="Manage ARGUS system service (macOS only)"
    )
    service_subparsers = service_parser.add_subparsers(dest="service_command", help="Service commands")
    
    # Service list
    service_list_parser = service_subparsers.add_parser("list", help="List service details")
    service_list_parser.set_defaults(func=cmd_service_list)
    
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
