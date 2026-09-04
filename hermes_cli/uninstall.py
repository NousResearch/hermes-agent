"""Hermes Agent Uninstaller."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home

from hermes_cli.colors import Colors, color

def _logger(mark: str, col: str):
    return lambda msg: print(f"{color(mark, col)} {msg}")


log_info = _logger("→", Colors.CYAN)
log_success = _logger("✓", Colors.GREEN)
log_warn = _logger("⚠", Colors.YELLOW)


def _print_box(middle: str, col: str) -> None:
    """Print the 3-line framed heading used by the uninstall screens."""
    print(color("┌─────────────────────────────────────────────────────────┐", col, Colors.BOLD))
    print(color(middle, col, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", col, Colors.BOLD))


def _prompt(text: str):
    """``input(text).strip().lower()``; None (after printing "Cancelled.") on Ctrl-C/EOF."""
    try:
        return input(text).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return None


def _cancelled() -> None:
    print()
    print("Uninstall cancelled.")


def _confirm_yes(text: str) -> bool:
    """Ask the user to type ``yes``; False (after the cancel line, unless Ctrl-C/EOF) otherwise."""
    confirm = _prompt(f"Type '{color('yes', Colors.YELLOW)}' {text}: ")
    if confirm != "yes" and confirm is not None:
        _cancelled()
    return confirm == "yes"


def _remove_each(candidates, remove) -> list:
    """Run ``remove(path)`` per candidate, collecting the ones it reports removed; failures become
    the shared ``Could not remove <path>: <err>`` warning."""
    removed = []
    for path in candidates:
        try:
            if remove(path):
                removed.append(path)
        except Exception as e:
            log_warn(f"Could not remove {path}: {e}")
    return removed


def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


def code_removal_refusal() -> "str | None":
    """Why the uninstaller must not remove this tree's code, or ``None``.

    A git checkout (the install.sh / install.ps1 / `hermes desktop` layout)
    is ours to remove. A sealed tree (no ``.git``) belongs to a steward —
    the Nix store, the bundled desktop app, a Docker image — and only the
    steward removes it. Returns the user-facing refusal text for sealed
    trees; the text always points at ``hermes uninstall --data`` for the
    user-data cleanup that IS allowed everywhere.
    """
    from hermes_cli.steward import sealed_steward, steward_uninstall_message

    steward = sealed_steward(get_project_root())
    if steward is None:
        return None
    return steward_uninstall_message(steward)


def _refuse_if_steward_owned() -> None:
    """Exit with the steward's instructions when code removal is forbidden."""
    refusal = code_removal_refusal()
    if refusal is None:
        return
    print()
    print(refusal)
    print()
    raise SystemExit(1)


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
    removed_from = []
    for config_path in (c for c in (Path.home() / n for n in _SHELL_RC_NAMES) if c.exists()):
        try:
            content = config_path.read_text(encoding="utf-8-sig")
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
                from utils import atomic_write_text
                # The user's own rc, never backed up: a bare write_text() truncates before the new
                # content lands and a crash mid-write would leave an empty ~/.zshrc. Atomic replace
                # also follows a symlinked rc; preserve_mode keeps its bits/owner (sudo-run uninstalls).
                atomic_write_text(config_path, new_content, preserve_mode=True)
                removed_from.append(config_path)
        except Exception as e:
            log_warn(f"Could not update {config_path}: {e}")
    return removed_from


def remove_wrapper_script():
    """Remove the hermes wrapper script if it exists."""
    wrapper_paths = [
        Path.home() / ".local" / "bin" / "hermes",
        Path.home() / ".local" / "bin" / "hermes-acp",
        Path.home() / ".local" / "bin" / "hermes-agent",
        Path("/usr/local/bin/hermes"),
        Path("/usr/local/bin/hermes-acp"),
        Path("/usr/local/bin/hermes-agent"),
    ]
    
    removed = []
    for wrapper in wrapper_paths:
        if wrapper.exists():
            try:
                # Check if it's our wrapper (contains hermes_cli reference)
                content = wrapper.read_text(encoding="utf-8-sig")
                if 'hermes_cli' in content or 'hermes-agent' in content:
                    wrapper.unlink()
                    removed.append(wrapper)
            except Exception as e:
                log_warn(f"Could not remove {wrapper}: {e}")
    
    return removed


def _node_symlink_candidate_dirs() -> "list[Path]":
    """Directories where the installer may have placed node/npm/npx symlinks."""
    dirs: list[Path] = [Path.home() / ".local" / "bin"]
    if sys.platform == "linux":  # root FHS installs put links in /usr/local/bin
        dirs.append(Path("/usr/local/bin"))
    return dirs


def remove_node_symlinks(hermes_home: Path) -> list:
    """Remove the node/npm/npx symlinks the installer placed on PATH.

    Historical POSIX installs (``scripts/install.sh`` before pm owned the
    Node runtime) symlinked node/npm/npx into the same directory as the
    ``hermes`` command:

    - ``/usr/local/bin/`` on root FHS installs (Linux, uid 0)
    - ``~/.local/bin/`` otherwise (the common non-root case)

    We check all candidate directories so that uninstall works regardless of
    how the install was done (e.g. a root FHS install that placed links in
    ``/usr/local/bin``, or an older install that used ``~/.local/bin`` before
    the FHS fix).  Only symlinks that resolve into this Hermes home's ``node``
    directory are removed — links the user has repointed elsewhere (nvm, fnm,
    etc.) are left untouched.
    """
    node_dir = (hermes_home / "node").resolve()

    def _unlink_ours(link: Path) -> bool:
        # Only act on symlinks — never delete a real binary the user put here.
        if not link.is_symlink():
            return False
        # os.readlink + manual join handles dangling links too (Path.resolve() on a dangling
        # link still returns the target path); the link must point into OUR node dir.
        target = (link.parent / os.readlink(link)).resolve()
        return _unlink_if(target == node_dir or node_dir in target.parents, link)

    candidates = (bin_dir / name for name in ("node", "npm", "npx") for bin_dir in _node_symlink_candidate_dirs())
    return _remove_each(candidates, _unlink_ours)


def uninstall_gateway_service():
    """Stop and uninstall the gateway service (systemd, launchd, Windows
    Scheduled Task / Startup folder) and kill any standalone gateway processes.

    Delegates to the gateway module which handles:
    - Linux: user + system systemd services (with proper DBUS env setup)
    - macOS: launchd plists
    - Windows: Scheduled Task + Startup-folder fallback, via ``gateway_windows``
    - All platforms: standalone ``hermes gateway run`` processes
    """
    import platform
    stopped_something = False

    # 1. Kill any standalone gateway processes (all platforms)
    try:
        from hermes_cli.gateway import kill_gateway_processes, find_gateway_pids
        killed = kill_gateway_processes() if find_gateway_pids() else 0
        if killed:
            log_success(f"Killed {killed} running gateway process(es)")
            stopped_something = True
    except Exception as e:
        log_warn(f"Could not check for gateway processes: {e}")

    system = platform.system()

    # 2. Linux: uninstall systemd services (both user and system scopes)
    if system == "Linux":
        try:
            stopped_something = remover() or stopped_something
        except Exception as e:
            log_warn(f"{warn_label}: {e}")
    return stopped_something


def _remove_systemd_gateway() -> bool:
    """Linux: uninstall systemd services (both user and system scopes)."""
    from hermes_cli.gateway import _systemctl_cmd, get_service_name, get_systemd_unit_path
    svc_name = get_service_name()
    removed_any = False
    for is_system, scope in ((False, "user"), (True, "system")):
        unit_path = get_systemd_unit_path(system=is_system)
        if not unit_path.exists():
            continue
        try:
            if is_system and os.geteuid() != 0:  # windows-footgun: ok — Linux-only systemd path
                log_warn(f"System gateway service exists at {unit_path} but needs sudo to remove")
                continue
            cmd = _systemctl_cmd(is_system)
            for verb in ("stop", "disable"):
                subprocess.run(cmd + [verb, svc_name], capture_output=True, check=False)
            unit_path.unlink()
            subprocess.run(cmd + ["daemon-reload"], capture_output=True, check=False)
            log_success(f"Removed {scope} gateway service ({unit_path})")
            removed_any = True
        except Exception as e:
            log_warn(f"Could not remove {scope} gateway service: {e}")
    return removed_any


def _remove_launchd_gateway() -> bool:
    """macOS: uninstall launchd plist."""
    from hermes_cli.gateway import get_launchd_plist_path
    plist_path = get_launchd_plist_path()
    if not plist_path.exists():
        return False
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, check=False)
    plist_path.unlink()
    log_success(f"Removed macOS gateway service ({plist_path})")
    return True


def _remove_windows_gateway() -> bool:
    """Windows: uninstall Scheduled Task + Startup-folder entry via ``gateway_windows`` (it owns
    schtasks /Delete, the .cmd unlink and stopping the detached pythonw gateway)."""
    from hermes_cli import gateway_windows as gw
    if not any(probe() for probe in (gw.is_installed, gw.is_task_registered, gw.is_startup_entry_installed)):
        return False
    try:
        gw.stop()
    except Exception as e:
        log_warn(f"Could not stop Windows gateway cleanly: {e}")
    try:
        gw.uninstall()
        log_success("Removed Windows gateway (Scheduled Task + Startup entry)")
        return True
    except Exception as e:
        log_warn(f"Could not fully uninstall Windows gateway: {e}")
        return False


# platform.system() -> (remover, warning label when the remover itself blows up)
_GATEWAY_SERVICE_REMOVERS = {
    "Linux": (_remove_systemd_gateway, "Could not check systemd gateway services"),
    "Darwin": (_remove_launchd_gateway, "Could not remove launchd gateway service"),
    "Windows": (_remove_windows_gateway, "Could not check Windows gateway service")}

# Windows helpers. install.ps1 leaves four things no rc file covers: User-scope env vars
# HERMES_HOME / HERMES_GIT_BASH_PATH (HKCU\Environment), User-scope PATH entries
# (%LOCALAPPDATA%\hermes\git\{cmd,bin,usr\bin}, ...\hermes\node), PortableGit + Node copies
# (~200MB) and the gateway-service dir. Direct winreg writes (not PowerShell): no subprocess, and
# they work under Constrained Language Mode; new shells see them without WM_SETTINGCHANGE.


def _hermes_path_markers(hermes_home: Path, *, include_managed_bin: bool = False) -> list[str]:
    """Prefixes identifying Hermes-owned User-PATH entries (prefix match sweeps git\cmd, git\bin,
    node...). ``include_managed_bin`` adds ``<root>\bin`` (launchers + managed uv) — only when that
    dir is about to be deleted, so a keep-data uninstall keeps the working uv resolvable."""
    root = str(hermes_home).rstrip("\\/")
    subs = ("hermes-agent", "git", "node", "venv") + (("bin",) if include_managed_bin else ())
    return [f"{root}\\{sub}" for sub in subs]


def remove_path_from_windows_registry(hermes_home: Path, *, include_managed_bin: bool = False) -> list[str]:
    """Strip Hermes-owned entries from User-scope PATH in the registry (see ``_hermes_path_markers``)."""
    markers = tuple(m.lower() for m in _hermes_path_markers(hermes_home, include_managed_bin=include_managed_bin))

    def edit(winreg, key, removed):
        try:
            path_value, path_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return
        # Preserve REG_EXPAND_SZ vs REG_SZ so unexpanded %VARS% survive.
        kept: list[str] = []
        for entry in (e for e in path_value.split(";") if e):
            is_ours = entry.rstrip("\\/").lower().startswith(markers)
            (removed if is_ours else kept).append(entry)
        if removed:
            winreg.SetValueEx(key, "Path", 0, path_type, ";".join(kept))

    return _edit_user_environment(edit, warn_label="Could not edit User PATH in registry")


def remove_hermes_env_vars_windows() -> list[str]:
    """Delete HERMES_HOME and HERMES_GIT_BASH_PATH from User-scope env vars."""
    def edit(winreg, key, removed):
        for name in ("HERMES_HOME", "HERMES_GIT_BASH_PATH"):
            try:
                winreg.QueryValueEx(key, name)
            except FileNotFoundError:
                continue
            try:
                winreg.DeleteValue(key, name)
                removed.append(name)
            except OSError as e:
                log_warn(f"Could not delete {name} from User env: {e}")

    return _edit_user_environment(edit, warn_label="Could not open User Environment key")


def _edit_user_environment(edit, *, warn_label: str) -> list[str]:
    """Open HKCU\\Environment read/write and run ``edit(winreg, key, removed)``. Returns what
    ``edit`` appended to ``removed`` even if a later registry call raised (callers report exactly
    what was touched); ``[]`` off-Windows (no ``winreg``)."""
    removed: list[str] = []
    try:
        import winreg
    except ImportError:
        return removed  # not on Windows, nothing to do
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ | winreg.KEY_WRITE) as key:
            edit(winreg, key, removed)
    except OSError as e:
        log_warn(f"{warn_label}: {e}")
    return removed


def remove_portable_tooling_windows(hermes_home: Path) -> list[Path]:
    """Delete the PortableGit / Node / gateway-service dirs the Windows installer created under
    ``hermes_home`` (isolated from any system Git / Node, so nothing else breaks)."""
    targets = (hermes_home / sub for sub in ("git", "node", "gateway-service"))
    return _remove_each((t for t in targets if t.exists()), lambda t: shutil.rmtree(t) or True)


def remove_legacy_runtime_trees(hermes_home: Path) -> list[Path]:
    """Delete managed-runtime trees a PRE-SPLIT install left in HERMES_HOME.

    Runtime artifacts are install-scoped now, so the current locations go
    away with ``rmtree(project_root)``. But a checkout OUTSIDE the home
    (the common case: ``~/src/hermes-agent``) used to put its node/uv
    under ``$HERMES_HOME`` — that tree survives removing the checkout and
    survives "keep my data" uninstalls, because it is not data.

    Only the exact managed layout is removed: ``node/`` (a tree the
    installer owned wholesale) and ``bin/uv`` (the single binary, NOT the
    whole ``bin/`` dir — a user's own scripts can live there). Profile
    state is never touched.
    """
    removed: list[Path] = []

    node_tree = hermes_home / "node"
    if node_tree.is_dir():
        try:
            shutil.rmtree(node_tree, ignore_errors=False)
            removed.append(node_tree)
        except Exception as e:
            log_warn(f"Could not remove {node_tree}: {e}")

    for uv_name in ("uv", "uv.exe"):
        uv_binary = hermes_home / "bin" / uv_name
        if uv_binary.is_file():
            try:
                uv_binary.unlink()
                removed.append(uv_binary)
            except Exception as e:
                log_warn(f"Could not remove {uv_binary}: {e}")

    return removed


def remove_windows_bin_launchers(*, windows: bool | None = None) -> list[Path]:
    """Delete the managed binary dir (the default Hermes root's ``bin``).

    The dir holds only hermes-owned launcher copies (the relocatable venv's
    console scripts, staged onto PATH by first-run repair) — pm keeps uv in
    its own store entry, so nothing shared lives here and the whole dir goes.
    Every uninstall mode deletes the code checkout, so a surviving launcher
    would dangle: ``hermes`` in a new terminal resolves and then errors on
    its missing venv target, which reads worse than command-not-found.

    A launcher that IS this process's own trampoline is mandatory-locked
    against deletion but not rename, so removal falls back to renaming it
    aside with a non-executable suffix.

    *windows* is an injectable platform verdict for tests (same pattern as
    ``_install_repair.ensure_windows_bin_launchers``).
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return []
    try:
        from hermes_constants import get_default_hermes_root
        bin_dir = get_default_hermes_root() / "bin"
    except Exception as e:
        log_warn(f"Could not locate the managed binary dir: {e}")
        return []
    if not bin_dir.is_dir():
        return []

    removed: list[Path] = []
    for launcher in sorted(bin_dir.iterdir()):
        if not launcher.is_file():
            continue
        try:
            launcher.unlink()
            removed.append(launcher)
        except OSError:
            aside = launcher.with_name(f"{launcher.name}.uninstalled.{os.getpid()}")
            try:
                os.rename(launcher, aside)
                removed.append(launcher)
            except OSError as e:
                log_warn(f"Could not remove {launcher}: {e}")
    try:
        bin_dir.rmdir()
    except OSError:
        pass  # leftovers (renamed-aside trampolines) keep the dir until next run
    return removed


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_default_hermes_home(hermes_home: Path) -> bool:
    """Return True when ``hermes_home`` points at the default (non-profile) root."""
    try:
        from hermes_constants import get_default_hermes_root
        return hermes_home.resolve() == get_default_hermes_root().resolve()
    except Exception:
        return False


def _discover_named_profiles():
    """``ProfileInfo`` for every non-default profile; ``[]`` when profile support is unavailable."""
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
    """Fully uninstall a named profile: stop its gateway, remove its alias, wipe its home. Shells
    out to ``hermes -p <name> gateway stop|uninstall`` because service names / unit paths derive
    from the current HERMES_HOME and can't be switched in-process."""
    name = profile.name
    log_info(f"Uninstalling profile '{name}'...")

    # 1. Gateway service, via `python -m hermes_cli.main` (the `hermes` wrapper may be half-gone).
    hermes_invocation = [sys.executable, "-m", "hermes_cli.main", "--profile", name]
    for subcmd in ("stop", "uninstall"):
        try:
            subprocess.run(
                hermes_invocation + ["gateway", subcmd], capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=60, check=False)
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
    _rmtree_step(profile.path, indent="  ", fully=False)


def run_data_uninstall(args):
    """Remove Hermes user data only — no code, on any install kind.

    This is the one destructive action that is valid everywhere: source
    checkouts, the bundled desktop app, Nix, Docker. It removes everything
    under ``$HERMES_HOME`` EXCEPT the ``hermes-agent`` checkout (which is
    code, owned by the code-removal modes / the steward), plus the desktop
    app's Electron userData directory.
    """
    hermes_home = get_hermes_home()
    agent_root = hermes_home / "hermes-agent"
    skip_confirm = bool(getattr(args, "yes", False))

    targets = []
    if hermes_home.exists():
        targets = sorted(
            (p for p in hermes_home.iterdir() if p.name != "hermes-agent"),
            key=lambda p: p.name,
        )

    from hermes_cli.gui_uninstall import desktop_userdata_dir

    userdata = desktop_userdata_dir()

    if not targets and not userdata.exists():
        print("No Hermes user data found.")
        print(f"  Checked: {hermes_home}")
        return

    print()
    print(color("This removes your Hermes data — config, chats, secrets, logs.", Colors.YELLOW, Colors.BOLD))
    print(color("Installed code is not touched.", Colors.CYAN))
    print()
    print(color("Will remove:", Colors.YELLOW, Colors.BOLD))
    for p in targets:
        print(f"  • {p}")
    if userdata.exists():
        print(f"  • {userdata}  (desktop app data)")
    if agent_root.exists():
        print()
        print(color("Kept intact:", Colors.GREEN, Colors.BOLD))
        print(f"  • {agent_root}")
    print()

    if not skip_confirm:
        try:
            confirm = input(f"Type '{color('yes', Colors.YELLOW)}' to remove your Hermes data: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            print("Cancelled.")
            return
        if confirm != "yes":
            print()
            print("Uninstall cancelled.")
            return

    print()
    for p in targets:
        try:
            if p.is_symlink() or p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
            log_success(f"Removed {p}")
        except Exception as e:
            log_warn(f"Could not remove {p}: {e}")

    if userdata.exists():
        try:
            shutil.rmtree(userdata)
            log_success(f"Removed {userdata}")
        except Exception as e:
            log_warn(f"Could not remove {userdata}: {e}")

    print()
    print(color("✓ Hermes data removed.", Colors.GREEN, Colors.BOLD))
    print()


def run_gui_uninstall(args):
    """GUI-only uninstall: remove the Chat GUI, leave the agent + data intact.

    Mirrors ``hermes uninstall --gui``. Removes the desktop app's built
    artifacts, the packaged app bundle (best-effort), and the Electron
    userData dir — nothing under ``$HERMES_HOME`` config/sessions/.env, and
    never the Python agent or its venv.
    """
    from hermes_cli.gui_uninstall import (
        agent_is_installed,
        gui_install_summary,
        uninstall_gui,
    )

    _refuse_if_steward_owned()

    hermes_home = get_hermes_home()
    summary = gui_install_summary(hermes_home)
    skip_confirm = bool(getattr(args, "yes", False))

    print()
    _print_box("│         ⚕ Hermes Chat GUI Uninstaller                  │", Colors.MAGENTA)
    print()

    if not summary["gui_installed"]:
        print("No Hermes Chat GUI installation was found.")
        print(f"  Checked: {hermes_home}, and the standard app locations for this OS.")
        return

    print(color("This removes the Chat GUI only. The Hermes agent stays installed.", Colors.CYAN))
    print()
    print(color("Will remove:", Colors.YELLOW, Colors.BOLD))
    for p in (*summary["source_built_artifacts"], *summary["packaged_app_paths"]):
        print(f"  • {p}")
    if summary["userdata_exists"]:
        print(f"  • {summary['userdata_dir']}  (desktop app data)")
    print()
    if agent_is_installed(hermes_home):
        print(color("Kept intact:", Colors.GREEN, Colors.BOLD))
        print(f"  • The Hermes agent at {hermes_home / 'hermes-agent'}")
        print(f"  • Your config, sessions, and secrets under {hermes_home}")
        print()

    if not skip_confirm and not _confirm_yes("to remove the Chat GUI"):
        return

    print()
    print(color("Uninstalling Chat GUI...", Colors.CYAN, Colors.BOLD))
    print()
    uninstall_gui(hermes_home)

    print()
    _print_box("│            ✓ Chat GUI Uninstalled!                      │", Colors.GREEN)
    print()
    print("The Hermes agent is still installed. Run 'hermes' to use the CLI,")
    print("or 'hermes uninstall' to remove the agent too.")
    print()


def run_uninstall(args):
    """
    Run the uninstall process.
    
    Options:
    - Full uninstall: removes code + ~/.hermes/ (configs, data, logs)
    - Keep data: removes code but keeps ~/.hermes/ for future reinstall
    """
    _refuse_if_steward_owned()

    project_root = get_project_root()
    hermes_home = get_hermes_home()

    full_flag = bool(getattr(args, "full", False))
    if bool(getattr(args, "dry_run", False)):
        _print_uninstall_dry_run(
            project_root=project_root, hermes_home=hermes_home, full_uninstall=full_flag)
        return

    # Named profiles (only when uninstalling from the default root) are offered for cleanup too,
    # instead of leaving zombie HERMES_HOMEs and systemd units behind.
    is_default_profile = _is_default_hermes_home(hermes_home)
    named_profiles = _discover_named_profiles() if is_default_profile else []

    # ``--yes`` (the desktop app's detached cleanup script): no prompts; ``--full`` = full wipe.
    # Named profiles are NOT auto-removed here — too destructive a default for an unattended run.
    if bool(getattr(args, "yes", False)):
        _perform_uninstall(
            project_root=project_root, hermes_home=hermes_home, full_uninstall=full_flag,
            remove_profiles=False, named_profiles=named_profiles)
        return

    print()
    _print_box("│            ⚕ Hermes Agent Uninstaller                  │", Colors.MAGENTA)
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
            print(f"  • {p.name}{' (gateway running)' if getattr(p, 'gateway_running', False) else ''}: {p.path}")
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

    choice = _prompt(color("Select option [1/2/3]: ", Colors.BOLD))
    if choice is None:
        return

    if choice in {"3", "c", "cancel", "q", "quit", "n", "no"}:
        _cancelled()
        return

    full_uninstall = (choice == "2")

    # Full uninstall from the default profile: offer to remove named profiles too (gateway
    # services, alias wrappers, HERMES_HOME dirs) — otherwise they leave zombie services behind.
    remove_profiles = False
    n_profiles = len(named_profiles)
    profile_names = ", ".join(p.name for p in named_profiles)
    if full_uninstall and named_profiles:
        print()
        print(color("Other profiles will NOT be removed by default.", Colors.YELLOW))
        print(f"Found {n_profiles} named profile(s): {profile_names}")
        print()
        resp = _prompt(color(f"Also stop and remove these {n_profiles} profile(s)? [y/N]: ", Colors.BOLD))
        if resp is None:
            return
        remove_profiles = resp in {"y", "yes"}

    # Final confirmation
    print()
    if full_uninstall:
        print(color("⚠️  WARNING: This will permanently delete ALL Hermes data!", Colors.RED, Colors.BOLD))
        print(color("   Including: configs, API keys, sessions, scheduled jobs, logs", Colors.RED))
        if remove_profiles:
            print(color(f"   Plus {n_profiles} profile(s): {profile_names}", Colors.RED))
    else:
        print("This will remove the Hermes code but keep your configuration and data.")

    print()
    if not _confirm_yes("to confirm"):
        return

    _perform_uninstall(
        project_root=project_root, hermes_home=hermes_home, full_uninstall=full_uninstall,
        remove_profiles=remove_profiles, named_profiles=named_profiles)


def _print_uninstall_dry_run(*, project_root: Path, hermes_home: Path, full_uninstall: bool) -> None:
    """Print the uninstall plan without stopping services or deleting files."""
    print()
    print(color("Dry run: no files, services, or environment entries will be changed.", Colors.CYAN, Colors.BOLD))
    print()
    print(color("Would inspect/remove:", Colors.YELLOW, Colors.BOLD))
    print("  • Gateway services and standalone gateway processes")
    print("  • Hermes PATH entries from shell configs / Windows User PATH")
    print("  • Hermes wrapper scripts and Hermes-managed node/npm/npx symlinks")
    print("  • Desktop Chat GUI artifacts")
    print(f"  • Code checkout: {project_root}")
    if not full_uninstall:
        print(f"  • Keep Hermes config/data: {hermes_home}")
    else:
        print(f"  • Hermes config/data: {hermes_home}")
        profiles = _discover_named_profiles() if _is_default_hermes_home(hermes_home) else []
        if profiles:
            print("  • Named profiles (interactive uninstall asks before removing):")
            for prof in profiles:
                print(f"    - {prof.name}: {prof.path}")
    print()


def _remove_step(label: str, remove, success_fmt: str, none_msg: str) -> None:
    """Announce ``label``, run ``remove()``, log one success line per removed item (or ``none_msg``)."""
    log_info(label)
    removed = remove()
    for item in removed:
        log_success(success_fmt.format(item))
    if not removed:
        log_info(none_msg)


def _rmtree_step(path: Path, *, indent: str = "", fully: bool = True) -> None:
    """Best-effort ``rmtree`` with the shared success/warning lines."""
    try:
        if path.exists():
            shutil.rmtree(path)
            log_success(f"{indent}Removed {path}")
    except Exception as e:
        log_warn(f"{indent}Could not {'fully ' if fully else ''}remove {path}: {e}")
        if fully:
            log_info("You may need to manually remove it")


def _perform_uninstall(
    *,
    project_root: Path,
    hermes_home: Path,
    full_uninstall: bool,
    remove_profiles: bool,
    named_profiles: list) -> None:
    """The uninstall steps shared by the interactive and ``--yes`` paths: stop gateway -> strip PATH
    (rc files + Windows registry) -> wrapper/launchers/node symlinks -> Chat GUI artifacts -> delete
    the checkout -> (Windows) PortableGit/Node -> optionally ``$HERMES_HOME`` and named profiles."""
    print()
    print(color("Uninstalling...", Colors.CYAN, Colors.BOLD))
    print()
    # 1. Stop and uninstall gateway service + kill standalone processes
    log_info("Checking for running gateway...")
    if not uninstall_gateway_service():
        log_info("No gateway service or processes found")

    # 2-3b. PATH entries, wrapper, Windows launchers, node symlinks. Windows: hermes_home is
    #    %VAR%-expanded because install.ps1 writes literal C:\Users\<u>\...; hermes\bin (launchers +
    #    managed uv) leaves the PATH only when the full wipe below deletes it (keep-data keeps uv
    #    resolvable), while the launchers themselves always go. Symlinks go only when they still
    #    point into this home's node dir (never clobber nvm / user-managed Node).
    windows = _is_windows()
    sweep_managed_bin = windows and full_uninstall and _is_default_hermes_home(hermes_home)
    for on_this_platform, label, remove, success_fmt, none_msg in (
        (True, "Removing PATH entries from shell configs...",
         remove_path_from_shell_configs, "Updated {}", "No PATH entries found to remove in shell rc files"),
        (windows, "Removing PATH entries from Windows User environment...",
         lambda: remove_path_from_windows_registry(
             Path(os.path.expandvars(str(hermes_home))), include_managed_bin=sweep_managed_bin),
         "Removed from User PATH: {}", "No Hermes-owned PATH entries in User environment"),
        (windows, "Removing HERMES_HOME / HERMES_GIT_BASH_PATH User env vars...",
         remove_hermes_env_vars_windows, "Removed User env var: {}", "No Hermes-set User env vars to remove"),
        (True, "Removing hermes command...", remove_wrapper_script, "Removed {}", "No wrapper script found"),
        (windows, "Removing Windows hermes launchers...",
         remove_windows_bin_launchers, "Removed {}", "No Windows hermes launchers found"),
        (True, "Removing Hermes-managed node/npm/npx symlinks...",
         lambda: remove_node_symlinks(hermes_home), "Removed {}", "No Hermes-managed node/npm/npx symlinks found"),
    ):
        if on_this_platform:
            _remove_step(label, remove, success_fmt, none_msg)

    # 3c. Chat GUI artifacts go with the agent code. uninstall_gui() never touches config/sessions/
    #     .env (safe in keep-data mode); the packaged app + Electron userData live OUTSIDE HERMES_HOME.
    log_info("Removing desktop Chat GUI artifacts...")
    try:
        from hermes_cli.gui_uninstall import uninstall_gui
        if not uninstall_gui(hermes_home):
            log_info("No desktop GUI artifacts found")
    except Exception as e:
        log_warn(f"Could not remove desktop GUI artifacts: {e}")

    # 4. Remove installation directory (code) — we may be running from inside it.
    log_info("Removing installation directory...")
    _rmtree_step(project_root)
    # 4b. Windows installer tooling (PortableGit, Node, gateway-service) is not user data:
    #     safe to remove in keep-data mode too.
    if windows:
        _remove_step(
            "Removing Windows installer artifacts (PortableGit, Node, gateway-service)...",
            lambda: remove_portable_tooling_windows(hermes_home), "Removed {}",
            "No Windows installer artifacts to remove")

    # 4b. Remove managed-runtime trees a PRE-SPLIT install left in
    #     HERMES_HOME. Current installs keep these inside the checkout, so
    #     step 4 already removed them — but a checkout outside the home
    #     (~/src/hermes-agent) used to leave its node/uv behind, surviving
    #     both the checkout removal and a "keep my data" uninstall. They
    #     are install tooling, not data, so removing them is correct in
    #     either mode.
    log_info("Removing managed runtime trees...")
    removed_runtimes = remove_legacy_runtime_trees(hermes_home)
    if removed_runtimes:
        for path in removed_runtimes:
            log_success(f"Removed {path}")
    else:
        log_info("No legacy runtime trees to remove")

    # 4c. Remove Windows-only installer artifacts that are NOT user data:
    #     PortableGit, bundled Node, gateway-service dir.  Installer put them
    #     under HERMES_HOME but they're install tooling, not config — safe to
    #     remove even in "keep data" mode.  If we're doing a full uninstall
    #     the step-5 rmtree(hermes_home) would sweep them anyway; calling
    #     this helper there is a no-op since they'll already be gone.
    if _is_windows():
        log_info("Removing Windows installer artifacts (PortableGit, Node, gateway-service)...")
        removed_artifacts = remove_portable_tooling_windows(hermes_home)
        if removed_artifacts:
            for path in removed_artifacts:
                log_success(f"Removed {path}")
        else:
            log_info("No Windows installer artifacts to remove")
    
    # 5. Optionally remove ~/.hermes/ data directory (and named profiles)
    if full_uninstall:
        # 5a. Named profiles' homes live under <default>/profiles/ (swept by the rmtree below),
        #     but their services + alias scripts live OUTSIDE the default root.
        for prof in named_profiles if remove_profiles else ():
            _uninstall_profile(prof)
        log_info("Removing configuration and data...")
        _rmtree_step(hermes_home)
    else:
        log_info(f"Keeping configuration and data in {hermes_home}")

    print()
    _print_box("│              ✓ Uninstall Complete!                      │", Colors.GREEN)
    print()

    if not full_uninstall:
        print(color("Your configuration and data have been preserved:", Colors.CYAN))
        print(f"  {hermes_home}/")
        print()
        print("To reinstall later with your existing settings:")
        print(color(_REINSTALL_HINT[windows], Colors.DIM))
        print()

    for line, col in _RELOAD_HINT[windows]:
        print(color(line, col) if col else line)
    print()
    print("Thank you for using Hermes Agent! ⚕")
    print()


_REINSTALL_HINT = {
    True: "  iex (irm https://hermes-agent.nousresearch.com/install.ps1)",
    False: "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"}
# windows -> [(line, color or None)]
_RELOAD_HINT = {
    True: [("Open a new terminal (PowerShell / Windows Terminal) to pick up", Colors.YELLOW),
           ("the updated User PATH and environment variables.", Colors.YELLOW)],
    False: [("Reload your shell to complete the process:", Colors.YELLOW),
            ("  source ~/.bashrc  # or ~/.zshrc", None)]}


class _UninstallArgs:
    """Lightweight args namespace for the module entrypoint below."""

    def __init__(self, *, mode: str):
        self.gui = mode == "gui"
        self.gui_summary = False
        self.full = mode == "full"
        self.data = mode == "data"
        self.yes = True  # the module entrypoint is always non-interactive


def main(argv=None) -> int:
    """Module entrypoint: ``python -m hermes_cli.uninstall --mode <gui|lite|full|data>``.

    Exists so the desktop app can run the uninstall under a Python interpreter
    OUTSIDE the venv being deleted. On Windows, ``lite``/``full`` rmtree the
    venv that contains the running ``python.exe`` — and a running .exe is
    mandatory-locked, so doing that from the venv's own interpreter half-fails.
    The desktop launches this with the system Python + ``PYTHONPATH=<agentRoot>``
    so ``import hermes_cli`` resolves from source while the venv is torn down.

    ``data`` removes user data only (no code) and is the one mode allowed on
    steward-owned installs (Nix, the bundled desktop app, Docker); the
    code-removing modes hard-fail there with the steward's instructions.

    This module imports only stdlib + ``hermes_constants`` + ``hermes_cli.colors``
    (and lazily ``hermes_cli.gui_uninstall`` / ``hermes_cli.steward``), so it
    runs fine under a bare system Python with no site-packages from the venv.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="python -m hermes_cli.uninstall")
    parser.add_argument(
        "--mode",
        choices=["gui", "lite", "full", "data"],
        required=True,
        help="gui = Chat GUI only; lite = GUI + agent, keep data; "
        "full = everything; data = user data only, keep code",
    )
    ns = parser.parse_args(argv)
    args = _UninstallArgs(mode=ns.mode)

    if args.data:
        run_data_uninstall(args)
    elif args.gui:
        run_gui_uninstall(args)
    else:
        run_uninstall(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----
