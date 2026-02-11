"""
Gateway subcommand for hermes CLI.

Handles: hermes gateway [run|start|stop|restart|status|install|uninstall]
"""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# =============================================================================
# Process Management (for manual gateway runs)
# =============================================================================

def find_gateway_pids() -> list:
    """Find PIDs of running gateway processes."""
    pids = []
    try:
        # Look for gateway processes with multiple patterns
        patterns = [
            "hermes_cli.main gateway",
            "hermes gateway",
            "gateway/run.py",
        ]
        
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            # Skip grep and current process
            if 'grep' in line or str(os.getpid()) in line:
                continue
            
            for pattern in patterns:
                if pattern in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            if pid not in pids:
                                pids.append(pid)
                        except ValueError:
                            continue
                    break
    except Exception:
        pass
    
    return pids


def kill_gateway_processes(force: bool = False) -> int:
    """Kill any running gateway processes. Returns count killed."""
    pids = find_gateway_pids()
    killed = 0
    
    for pid in pids:
        try:
            if force:
                os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            # Process already gone
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
    
    return killed


def is_linux() -> bool:
    return sys.platform.startswith('linux')

def is_macos() -> bool:
    return sys.platform == 'darwin'

def is_windows() -> bool:
    return sys.platform == 'win32'


# =============================================================================
# Service Configuration
# =============================================================================

SERVICE_NAME = "hermes-gateway"
SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"

def get_systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"

def get_launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"

def get_python_path() -> str:
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def get_hermes_cli_path() -> str:
    """Get the path to the hermes CLI."""
    # Check if installed via pip
    import shutil
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return hermes_bin
    
    # Fallback to direct module execution
    return f"{get_python_path()} -m hermes_cli.main"


# =============================================================================
# Systemd (Linux)
# =============================================================================

def generate_systemd_unit() -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main gateway run
WorkingDirectory={working_dir}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""

def systemd_install(force: bool = False):
    unit_path = get_systemd_unit_path()
    
    if unit_path.exists() and not force:
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return
    
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing systemd service to: {unit_path}")
    unit_path.write_text(generate_systemd_unit())
    
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", SERVICE_NAME], check=True)
    
    print()
    print("✓ Service installed and enabled!")
    print()
    print("Next steps:")
    print(f"  hermes gateway start              # Start the service")
    print(f"  hermes gateway status             # Check status")
    print(f"  journalctl --user -u {SERVICE_NAME} -f  # View logs")
    print()
    print("To enable lingering (keeps running after logout):")
    print("  sudo loginctl enable-linger $USER")

def systemd_uninstall():
    subprocess.run(["systemctl", "--user", "stop", SERVICE_NAME], check=False)
    subprocess.run(["systemctl", "--user", "disable", SERVICE_NAME], check=False)
    
    unit_path = get_systemd_unit_path()
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")
    
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print("✓ Service uninstalled")

def systemd_start():
    subprocess.run(["systemctl", "--user", "start", SERVICE_NAME], check=True)
    print("✓ Service started")

def systemd_stop():
    subprocess.run(["systemctl", "--user", "stop", SERVICE_NAME], check=True)
    print("✓ Service stopped")

def systemd_restart():
    subprocess.run(["systemctl", "--user", "restart", SERVICE_NAME], check=True)
    print("✓ Service restarted")

def systemd_status(deep: bool = False):
    # Check if service unit file exists
    unit_path = get_systemd_unit_path()
    if not unit_path.exists():
        print("✗ Gateway service is not installed")
        print("  Run: hermes gateway install")
        return
    
    # Show detailed status first
    subprocess.run(
        ["systemctl", "--user", "status", SERVICE_NAME, "--no-pager"],
        capture_output=False
    )
    
    # Check if service is active
    result = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        capture_output=True,
        text=True
    )
    
    status = result.stdout.strip()
    
    if status == "active":
        print("✓ Gateway service is running")
    else:
        print("✗ Gateway service is stopped")
        print("  Run: hermes gateway start")
    
    if deep:
        print()
        print("Recent logs:")
        subprocess.run([
            "journalctl", "--user", "-u", SERVICE_NAME,
            "-n", "20", "--no-pager"
        ])


# =============================================================================
# Launchd (macOS)
# =============================================================================

def generate_launchd_plist() -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    log_dir = Path.home() / ".hermes" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.hermes.gateway</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hermes_cli.main</string>
        <string>gateway</string>
        <string>run</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/gateway.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/gateway.error.log</string>
</dict>
</plist>
"""

def launchd_install(force: bool = False):
    plist_path = get_launchd_plist_path()
    
    if plist_path.exists() and not force:
        print(f"Service already installed at: {plist_path}")
        print("Use --force to reinstall")
        return
    
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing launchd service to: {plist_path}")
    plist_path.write_text(generate_launchd_plist())
    
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    
    print()
    print("✓ Service installed and loaded!")
    print()
    print("Next steps:")
    print("  hermes gateway status             # Check status")
    print("  tail -f ~/.hermes/logs/gateway.log  # View logs")

def launchd_uninstall():
    plist_path = get_launchd_plist_path()
    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    
    if plist_path.exists():
        plist_path.unlink()
        print(f"✓ Removed {plist_path}")
    
    print("✓ Service uninstalled")

def launchd_start():
    subprocess.run(["launchctl", "start", "ai.hermes.gateway"], check=True)
    print("✓ Service started")

def launchd_stop():
    subprocess.run(["launchctl", "stop", "ai.hermes.gateway"], check=True)
    print("✓ Service stopped")

def launchd_restart():
    launchd_stop()
    launchd_start()

def launchd_status(deep: bool = False):
    result = subprocess.run(
        ["launchctl", "list", "ai.hermes.gateway"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Gateway service is loaded")
        print(result.stdout)
    else:
        print("✗ Gateway service is not loaded")
    
    if deep:
        log_file = Path.home() / ".hermes" / "logs" / "gateway.log"
        if log_file.exists():
            print()
            print("Recent logs:")
            subprocess.run(["tail", "-20", str(log_file)])


# =============================================================================
# Gateway Runner
# =============================================================================

def run_gateway(verbose: bool = False):
    """Run the gateway in foreground."""
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from gateway.run import start_gateway
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│           🦋 Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    # Exit with code 1 if gateway fails to connect any platform,
    # so systemd Restart=on-failure will retry on transient errors
    success = asyncio.run(start_gateway())
    if not success:
        sys.exit(1)


# =============================================================================
# Main Command Handler
# =============================================================================

def gateway_command(args):
    """Handle gateway subcommands."""
    subcmd = getattr(args, 'gateway_command', None)
    
    # Default to run if no subcommand
    if subcmd is None or subcmd == "run":
        verbose = getattr(args, 'verbose', False)
        run_gateway(verbose)
        return
    
    # Service management commands
    if subcmd == "install":
        force = getattr(args, 'force', False)
        if is_linux():
            systemd_install(force)
        elif is_macos():
            launchd_install(force)
        else:
            print("Service installation not supported on this platform.")
            print("Run manually: hermes gateway run")
            sys.exit(1)
    
    elif subcmd == "uninstall":
        if is_linux():
            systemd_uninstall()
        elif is_macos():
            launchd_uninstall()
        else:
            print("Not supported on this platform.")
            sys.exit(1)
    
    elif subcmd == "start":
        if is_linux():
            systemd_start()
        elif is_macos():
            launchd_start()
        else:
            print("Not supported on this platform.")
            sys.exit(1)
    
    elif subcmd == "stop":
        # Try service first, fall back to killing processes directly
        service_available = False
        
        if is_linux() and get_systemd_unit_path().exists():
            try:
                systemd_stop()
                service_available = True
            except subprocess.CalledProcessError:
                pass  # Fall through to process kill
        elif is_macos() and get_launchd_plist_path().exists():
            try:
                launchd_stop()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        
        if not service_available:
            # Kill gateway processes directly
            killed = kill_gateway_processes()
            if killed:
                print(f"✓ Stopped {killed} gateway process(es)")
            else:
                print("✗ No gateway processes found")
    
    elif subcmd == "restart":
        # Try service first, fall back to killing and restarting
        service_available = False
        
        if is_linux() and get_systemd_unit_path().exists():
            try:
                systemd_restart()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        elif is_macos() and get_launchd_plist_path().exists():
            try:
                launchd_restart()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        
        if not service_available:
            # Manual restart: kill existing processes
            killed = kill_gateway_processes()
            if killed:
                print(f"✓ Stopped {killed} gateway process(es)")
            
            import time
            time.sleep(2)
            
            # Start fresh
            print("Starting gateway...")
            run_gateway(verbose=False)
    
    elif subcmd == "status":
        deep = getattr(args, 'deep', False)
        
        # Check for service first
        if is_linux() and get_systemd_unit_path().exists():
            systemd_status(deep)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep)
        else:
            # Check for manually running processes
            pids = find_gateway_pids()
            if pids:
                print(f"✓ Gateway is running (PID: {', '.join(map(str, pids))})")
                print("  (Running manually, not as a system service)")
                print()
                print("To install as a service:")
                print("  hermes gateway install")
            else:
                print("✗ Gateway is not running")
                print()
                print("To start:")
                print("  hermes gateway          # Run in foreground")
                print("  hermes gateway install  # Install as service")
