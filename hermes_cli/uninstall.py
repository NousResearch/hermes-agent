"""
Hermes Agent Uninstaller.

Provides options for:
- Full uninstall: Remove everything including configs and data
- Keep data: Remove code but keep ~/.hermes/ (configs, sessions, logs)
"""

import os
import shutil
import subprocess
from pathlib import Path

from hermes_constants import get_hermes_home

from hermes_cli.colors import Colors, color

def log_info(msg: str):
    print(f"{color('→', Colors.CYAN)} {msg}")

def log_success(msg: str):
    print(f"{color('✓', Colors.GREEN)} {msg}")

def log_warn(msg: str):
    print(f"{color('⚠', Colors.YELLOW)} {msg}")

def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


def find_shell_configs() -> list:
    """Find shell configuration files that might have PATH entries."""
    home = Path.home()
    configs = []
    
    candidates = [
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
        home / ".zshrc",
        home / ".zprofile",
    ]
    
    for config in candidates:
        if config.exists():
            configs.append(config)
    
    return configs


def remove_path_from_shell_configs():
    """Remove Hermes PATH entries from shell configuration files."""
    configs = find_shell_configs()
    removed_from = []
    
    for config_path in configs:
        try:
            content = config_path.read_text()
            original_content = content
            
            # Remove lines containing hermes-agent or hermes PATH entries
            new_lines = []
            skip_next = False
            
            for line in content.split('\n'):
                # Skip the "# Hermes Agent" comment and following line
                if '# Hermes Agent' in line or '# hermes-agent' in line:
                    skip_next = True
                    continue
                if skip_next and ('hermes' in line.lower() and 'PATH' in line):
                    skip_next = False
                    continue
                skip_next = False
                
                # Remove any PATH line containing hermes
                if 'hermes' in line.lower() and ('PATH=' in line or 'path=' in line.lower()):
                    continue
                    
                new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            # Clean up multiple blank lines
            while '\n\n\n' in new_content:
                new_content = new_content.replace('\n\n\n', '\n\n')
            
            if new_content != original_content:
                config_path.write_text(new_content)
                removed_from.append(config_path)
                
        except Exception as e:
            log_warn(f"Could not update {config_path}: {e}")
    
    return removed_from


def remove_wrapper_script():
    """Remove the hermes wrapper script if it exists."""
    wrapper_paths = [
        Path.home() / ".local" / "bin" / "hermes",
        Path("/usr/local/bin/hermes"),
    ]
    
    removed = []
    for wrapper in wrapper_paths:
        if wrapper.exists():
            try:
                # Check if it's our wrapper (contains hermes_cli reference)
                content = wrapper.read_text()
                if 'hermes_cli' in content or 'hermes-agent' in content:
                    wrapper.unlink()
                    removed.append(wrapper)
            except Exception as e:
                log_warn(f"Could not remove {wrapper}: {e}")
    
    return removed


def uninstall_gateway_service():
    """Stop and uninstall the gateway service (systemd, launchd) and kill any
    standalone gateway processes.

    Delegates to the gateway module which handles:
    - Linux: user + system systemd services (with proper DBUS env setup)
    - macOS: launchd plists
    - All platforms: standalone ``hermes gateway run`` processes
    - Termux/Android: skips systemd (no systemd on Android), still kills standalone processes
    """
    import platform
    stopped_something = False

    # 1. Kill any standalone gateway processes (all platforms, including Termux)
    try:
        from hermes_cli.gateway import kill_gateway_processes, find_gateway_pids
        pids = find_gateway_pids()
        if pids:
            killed = kill_gateway_processes()
            if killed:
                log_success(f"Killed {killed} running gateway process(es)")
                stopped_something = True
    except Exception as e:
        log_warn(f"Could not check for gateway processes: {e}")

    system = platform.system()

    # Termux/Android has no systemd and no launchd — nothing left to do.
    prefix = os.getenv("PREFIX", "")
    is_termux = bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)
    if is_termux:
        return stopped_something

    # 2. Linux: uninstall systemd services (both user and system scopes)
    if system == "Linux":
        try:
            from hermes_cli.gateway import (
                get_systemd_unit_path,
                get_service_name,
                _systemctl_cmd,
            )
            svc_name = get_service_name()

            for is_system in (False, True):
                unit_path = get_systemd_unit_path(system=is_system)
                if not unit_path.exists():
                    continue

                scope = "system" if is_system else "user"
                try:
                    if is_system and os.geteuid() != 0:  # windows-footgun: ok — Linux systemd uninstall path, guarded by `if system == "Linux"` above
                        log_warn(f"System gateway service exists at {unit_path} "
                                 f"but needs sudo to remove")
                        continue

                    cmd = _systemctl_cmd(is_system)
                    subprocess.run(cmd + ["stop", svc_name],
                                   capture_output=True, check=False)
                    subprocess.run(cmd + ["disable", svc_name],
                                   capture_output=True, check=False)
                    unit_path.unlink()
                    subprocess.run(cmd + ["daemon-reload"],
                                   capture_output=True, check=False)
                    log_success(f"Removed {scope} gateway service ({unit_path})")
                    stopped_something = True
                except Exception as e:
                    log_warn(f"Could not remove {scope} gateway service: {e}")
        except Exception as e:
            log_warn(f"Could not check systemd gateway services: {e}")

    # 3. macOS: uninstall launchd plist
    elif system == "Darwin":
        try:
            from hermes_cli.gateway import get_launchd_plist_path
            plist_path = get_launchd_plist_path()
            if plist_path.exists():
                subprocess.run(["launchctl", "unload", str(plist_path)],
                               capture_output=True, check=False)
                plist_path.unlink()
                log_success(f"Removed macOS gateway service ({plist_path})")
                stopped_something = True
        except Exception as e:
            log_warn(f"Could not remove launchd gateway service: {e}")

    return stopped_something


def _is_default_hermes_home(hermes_home: Path) -> bool:
    """Return True when ``hermes_home`` points at the default (non-profile) root."""
    try:
        from hermes_constants import get_default_hermes_root
        return hermes_home.resolve() == get_default_hermes_root().resolve()
    except Exception:
        return False


def _discover_named_profiles():
    """Return a list of ``ProfileInfo`` for every non-default profile, or ``[]``
    if profile support is unavailable or nothing is installed beyond the
    default root."""
    try:
        from hermes_cli.profiles import list_profiles
    except Exception:
        return []
    try:
        return [p for p in list_profiles() if not getattr(p, "is_default", False)]
    except Exception as e:
        log_warn(f"Could not enumerate profiles: {e}")
        return []


def _uninstall_profile(profile) -> None:
    """Fully uninstall a single named profile: stop its gateway service,
    remove its alias wrapper, and wipe its HERMES_HOME directory.

    We shell out to ``hermes -p <name> gateway stop|uninstall`` because
    service names, unit paths, and plist paths are all derived from the
    current HERMES_HOME and can't be easily switched in-process.
    """
    import sys as _sys
    name = profile.name
    profile_home = profile.path

    log_info(f"Uninstalling profile '{name}'...")

    # 1. Stop and remove this profile's gateway service.
    #    Use `python -m hermes_cli.main` so we don't depend on a `hermes`
    #    wrapper that may be half-removed mid-uninstall.
    hermes_invocation = [_sys.executable, "-m", "hermes_cli.main", "--profile", name]
    for subcmd in ("stop", "uninstall"):
        try:
            subprocess.run(
                hermes_invocation + ["gateway", subcmd],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_warn(f"  Gateway {subcmd} timed out for '{name}'")
        except Exception as e:
            log_warn(f"  Could not run gateway {subcmd} for '{name}': {e}")

    # 2. Remove the wrapper alias script at ~/.local/bin/<name> (if any).
    alias_path = getattr(profile, "alias_path", None)
    if alias_path and alias_path.exists():
        try:
            alias_path.unlink()
            log_success(f"  Removed alias {alias_path}")
        except Exception as e:
            log_warn(f"  Could not remove alias {alias_path}: {e}")

    # 3. Wipe the profile's HERMES_HOME directory.
    try:
        if profile_home.exists():
            shutil.rmtree(profile_home)
            log_success(f"  Removed {profile_home}")
    except Exception as e:
        log_warn(f"  Could not remove {profile_home}: {e}")


def run_uninstall(args):
    """
    Run the uninstall process.
    
    Options:
    - Full uninstall: removes code + ~/.hermes/ (configs, data, logs)
    - Keep data: removes code but keeps ~/.hermes/ for future reinstall
    """
    project_root = get_project_root()
    hermes_home = get_hermes_home()

    # Detect named profiles when uninstalling from the default root —
    # offer to clean them up too instead of leaving zombie HERMES_HOMEs
    # and systemd units behind.
    is_default_profile = _is_default_hermes_home(hermes_home)
    named_profiles = _discover_named_profiles() if is_default_profile else []

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA, Colors.BOLD))
    print(color("│            ⚕ Hermes Agent Uninstaller                  │", Colors.MAGENTA, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA, Colors.BOLD))
    print()
    
    # Show what will be affected
    print(color("Current Installation:", Colors.CYAN, Colors.BOLD))
    print(f"  Code:    {project_root}")
    print(f"  Config:  {hermes_home / 'config.yaml'}")
    print(f"  Secrets: {hermes_home / '.env'}")
    print(f"  Data:    {hermes_home / 'cron/'}, {hermes_home / 'sessions/'}, {hermes_home / 'logs/'}")
    print()

    if named_profiles:
        print(color("Other profiles detected:", Colors.CYAN, Colors.BOLD))
        for p in named_profiles:
            running = " (gateway running)" if getattr(p, "gateway_running", False) else ""
            print(f"  • {p.name}{running}: {p.path}")
        print()
    
    # Ask for confirmation
    print(color("Uninstall Options:", Colors.YELLOW, Colors.BOLD))
    print()
    print("  1) " + color("Keep data", Colors.GREEN) + " - Remove code only, keep configs/sessions/logs")
    print("     (Recommended - you can reinstall later with your settings intact)")
    print()
    print("  2) " + color("Full uninstall", Colors.RED) + " - Remove everything including all data")
    print("     (Warning: This deletes all configs, sessions, and logs permanently)")
    print()
    print("  3) " + color("Cancel", Colors.CYAN) + " - Don't uninstall")
    print()
    
    try:
        choice = input(color("Select option [1/2/3]: ", Colors.BOLD)).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return
    
    if choice == "3" or choice.lower() in ("c", "cancel", "q", "quit", "n", "no"):
        print()
        print("Uninstall cancelled.")
        return
    
    full_uninstall = (choice == "2")

    # When doing a full uninstall from the default profile, also offer to
    # remove any named profiles — stopping their gateway services, unlinking
    # their alias wrappers, and wiping their HERMES_HOME dirs. Otherwise
    # those leave zombie services and data behind.
    remove_profiles = False
    if full_uninstall and named_profiles:
        print()
        print(color("Other profiles will NOT be removed by default.", Colors.YELLOW))
        print(f"Found {len(named_profiles)} named profile(s): " +
              ", ".join(p.name for p in named_profiles))
        print()
        try:
            resp = input(color(
                f"Also stop and remove these {len(named_profiles)} profile(s)? [y/N]: ",
                Colors.BOLD
            )).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            print("Cancelled.")
            return
        remove_profiles = resp in ("y", "yes")

    # Final confirmation
    print()
    if full_uninstall:
        print(color("⚠️  WARNING: This will permanently delete ALL Hermes data!", Colors.RED, Colors.BOLD))
        print(color("   Including: configs, API keys, sessions, scheduled jobs, logs", Colors.RED))
        if remove_profiles:
            print(color(
                f"   Plus {len(named_profiles)} profile(s): " +
                ", ".join(p.name for p in named_profiles),
                Colors.RED
            ))
    else:
        print("This will remove the Hermes code but keep your configuration and data.")
    
    print()
    try:
        confirm = input(f"Type '{color('yes', Colors.YELLOW)}' to confirm: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return
    
    if confirm != "yes":
        print()
        print("Uninstall cancelled.")
        return
    
    print()
    print(color("Uninstalling...", Colors.CYAN, Colors.BOLD))
    print()
    
    # 1. Stop and uninstall gateway service + kill standalone processes
    log_info("Checking for running gateway...")
    if not uninstall_gateway_service():
        log_info("No gateway service or processes found")
    
    # 2. Remove PATH entries from shell configs
    log_info("Removing PATH entries from shell configs...")
    removed_configs = remove_path_from_shell_configs()
    if removed_configs:
        for config in removed_configs:
            log_success(f"Updated {config}")
    else:
        log_info("No PATH entries found to remove")
    
    # 3. Remove wrapper script
    log_info("Removing hermes command...")
    removed_wrappers = remove_wrapper_script()
    if removed_wrappers:
        for wrapper in removed_wrappers:
            log_success(f"Removed {wrapper}")
    else:
        log_info("No wrapper script found")
    
    # 4. Remove installation directory (code)
    log_info("Removing installation directory...")
    
    # Check if we're running from within the install dir
    # We need to be careful here
    try:
        if project_root.exists():
            # If the install is inside ~/.hermes/, just remove the hermes-agent subdir
            if hermes_home in project_root.parents or project_root.parent == hermes_home:
                shutil.rmtree(project_root)
                log_success(f"Removed {project_root}")
            else:
                # Installation is somewhere else entirely
                shutil.rmtree(project_root)
                log_success(f"Removed {project_root}")
    except Exception as e:
        log_warn(f"Could not fully remove {project_root}: {e}")
        log_info("You may need to manually remove it")
    
    # 5. Optionally remove ~/.hermes/ data directory (and named profiles)
    if full_uninstall:
        # 5a. Stop and remove each named profile's gateway service and
        #     alias wrapper. The profile HERMES_HOME dirs live under
        #     ``<default>/profiles/<name>/`` and will be swept away by the
        #     rmtree below, but services + alias scripts live OUTSIDE the
        #     default root and have to be cleaned up explicitly.
        if remove_profiles and named_profiles:
            for prof in named_profiles:
                _uninstall_profile(prof)

        log_info("Removing configuration and data...")
        try:
            if hermes_home.exists():
                shutil.rmtree(hermes_home)
                log_success(f"Removed {hermes_home}")
        except Exception as e:
            log_warn(f"Could not fully remove {hermes_home}: {e}")
            log_info("You may need to manually remove it")
    else:
        log_info(f"Keeping configuration and data in {hermes_home}")
    
    # Done
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN, Colors.BOLD))
    print(color("│              ✓ Uninstall Complete!                      │", Colors.GREEN, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN, Colors.BOLD))
    print()
    
    if not full_uninstall:
        print(color("Your configuration and data have been preserved:", Colors.CYAN))
        print(f"  {hermes_home}/")
        print()
        print("To reinstall later with your existing settings:")
        print(color("  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash", Colors.DIM))
        print()
    
    print(color("Reload your shell to complete the process:", Colors.YELLOW))
    print("  source ~/.bashrc  # or ~/.zshrc")
    print()
    print("Thank you for using Hermes Agent! ⚕")
    print()
