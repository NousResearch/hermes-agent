"""Gateway systemd unit, bus, scope, install and lifecycle management.

Extracted from hermes_cli.gateway; function bodies preserve the pinned base.
Collaborators are rebound from that facade per call to preserve its patch surface.
"""

import subprocess
from pathlib import Path


def user_systemd_unit_dir() -> Path:
    """``$XDG_CONFIG_HOME/systemd/user`` (``~/.config`` only as the spec's default).

    Hardcoding ``~/.config`` made every unit probe silently false-negative on a host that moves
    XDG_CONFIG_HOME — doctor then reported no gateway unit while systemd was running ours.
    """
    from hermes_cli.gateway import Path, os
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "systemd" / "user"


def get_systemd_unit_path(system: bool = False) -> Path:
    from hermes_cli.gateway import _SYSTEM_UNIT_DIR, get_service_name, user_systemd_unit_dir
    name = get_service_name()
    if system:
        return _SYSTEM_UNIT_DIR / f"{name}.service"
    return user_systemd_unit_dir() / f"{name}.service"


def _user_runtime_dir() -> Path:
    """``$XDG_RUNTIME_DIR`` or ``/run/user/<uid>`` (regardless of existence)."""
    from hermes_cli.gateway import Path, os
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows


def _user_dbus_socket_path() -> Path:
    """Return the expected per-user D-Bus socket path (regardless of existence)."""
    from hermes_cli.gateway import _user_runtime_dir
    return _user_runtime_dir() / "bus"


def _user_systemd_private_socket_path() -> Path:
    """Return the per-user systemd private socket path (regardless of existence)."""
    from hermes_cli.gateway import _user_runtime_dir
    return _user_runtime_dir() / "systemd" / "private"


def _path_exists_safe(path: Path) -> bool:
    """``Path.exists()`` treating an inaccessible path as absent: a leaked ``XDG_RUNTIME_DIR`` from
    another user (``/run/user/0`` is 0700) would otherwise crash the preflight with EACCES.

    ``Path.exists()`` only swallows a subset of ``OSError`` (ENOENT/ENOTDIR/ EBADF/ELOOP); ``EACCES`` still
    propagates. When ``XDG_RUNTIME_DIR`` leaks from another user — the classic ``su``/``sudo -u`` from a
    root shell case, where ``/run/user/0`` is ``0700 root:root`` — stat-ing a socket underneath it raises
    ``PermissionError`` that escapes the systemd preflight as a raw traceback (#86558). An unreadable path
    is, for our purposes, not reachable.
    """
    try:
        return path.exists()
    except OSError:  # e.g. EACCES on another user's runtime dir
        return False


def _runtime_dir_is_ours(runtime_dir: str) -> bool:
    """True when *runtime_dir* exists and is owned by our uid (a leaked foreign XDG_RUNTIME_DIR must not be trusted)."""
    from hermes_cli.gateway import Path, os
    try:
        return Path(runtime_dir).stat().st_uid == os.getuid()  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    except OSError:
        return False


def _user_systemd_socket_ready() -> bool:
    """True when the user D-Bus socket OR the per-user systemd private socket exists (some distros
    expose only the latter and ``systemctl --user`` still works). Inaccessible counts as not-ready."""
    from hermes_cli.gateway import (
        _path_exists_safe,
        _user_dbus_socket_path,
        _user_systemd_private_socket_path,
    )
    return _path_exists_safe(_user_dbus_socket_path()) or _path_exists_safe(_user_systemd_private_socket_path())


def _ensure_user_systemd_env() -> None:
    """Set XDG_RUNTIME_DIR / DBUS_SESSION_BUS_ADDRESS so ``systemctl --user`` works on headless (SSH)
    hosts; an XDG_RUNTIME_DIR leaked from another user is replaced with our own ``/run/user/{uid}``.

    An ``XDG_RUNTIME_DIR`` that leaked from another user (``su``/``sudo -u`` from root, where the env still
    points at ``/run/user/0``) is dropped in favour of our own ``/run/user/{uid}`` so ``systemctl --user``
    targets the right instance instead of an unreadable foreign socket (#86558).
    """
    from hermes_cli.gateway import Path, _path_exists_safe, _runtime_dir_is_ours, os
    uid = os.getuid()  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if (not xdg or not _runtime_dir_is_ours(xdg)) and _runtime_dir_is_ours(f"/run/user/{uid}"):
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"

    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        bus_path = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")) / "bus"
        if _path_exists_safe(bus_path):
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"


def _wait_for_user_dbus_socket(timeout: float = 3.0) -> bool:
    """Poll up to ``timeout`` s for a user systemd control socket (user@.service takes a moment after enable-linger)."""
    from hermes_cli.gateway import _ensure_user_systemd_env, _user_systemd_socket_ready, time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _user_systemd_socket_ready():
            _ensure_user_systemd_env()
            return True
        time.sleep(0.2)
    return _user_systemd_socket_ready()


def _wait_for_target_user_bus(uid: int, timeout: float = 5.0) -> bool:
    """Poll for ``/run/user/<uid>/bus`` of ANOTHER account (the system unit's ``User=`` while root installs).
    Only the D-Bus socket counts — ``systemd/private`` alone is enough for ``systemctl --user`` but not for
    the ``systemd-run --user`` that restart-safe workers need. Never adopts anything into our env."""
    from hermes_cli.gateway import Path, _path_exists_safe, time
    bus = Path(f"/run/user/{uid}/bus")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _path_exists_safe(bus):
            return True
        time.sleep(0.2)
    return _path_exists_safe(bus)


def _loginctl_enable_linger(username: str) -> subprocess.CompletedProcess:
    """``loginctl enable-linger <username>`` (check=False, 30s); exceptions propagate to the caller."""
    from hermes_cli.gateway import _CAPTURE_TEXT, subprocess
    return subprocess.run(["loginctl", "enable-linger", username], check=False, timeout=30, **_CAPTURE_TEXT)


def _completed_process_detail(result) -> str:
    """stderr, else stdout, else ``exit <rc>`` — stripped."""
    return (result.stderr or result.stdout or f"exit {result.returncode}").strip()


def _preflight_user_systemd(*, auto_enable_linger: bool = True) -> None:
    """Ensure ``systemctl --user`` can reach user-scope systemd; raise UserSystemdUnavailableError otherwise.
    No-op when a control socket exists; else wait briefly if linger is on, or (``auto_enable_linger``)
    try ``loginctl enable-linger`` (non-root works when polkit permits)."""
    from hermes_cli.gateway import (
        _completed_process_detail,
        _ensure_user_systemd_env,
        _loginctl_enable_linger,
        _raise_user_systemd_unavailable,
        _user_systemd_socket_ready,
        _wait_for_user_dbus_socket,
        get_service_name,
        get_systemd_linger_status,
        os,
        shutil,
    )
    _ensure_user_systemd_env()
    if _user_systemd_socket_ready():
        return

    import getpass
    username = getpass.getuser()
    linger_enabled, linger_detail = get_systemd_linger_status()
    sudo_hint = f"  sudo loginctl enable-linger {username}"

    if linger_enabled is True:
        if _wait_for_user_dbus_socket(timeout=3.0):
            return
        # Linger is on but socket still missing — unusual; fall through to error.
        _raise_user_systemd_unavailable(
            username,
            reason="User systemd control sockets are missing even though linger is enabled.",
            fix_hint=(
                f"  systemctl start user@{os.getuid()}.service\n"  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
                "  (may require sudo; try again after the command succeeds)"
            ),
        )

    if auto_enable_linger and shutil.which("loginctl"):
        try:
            result = _loginctl_enable_linger(username)
        except Exception as exc:
            _raise_user_systemd_unavailable(
                username, reason=f"loginctl enable-linger failed ({exc}).", fix_hint=sudo_hint
            )
        else:
            if result.returncode == 0:
                if _wait_for_user_dbus_socket(timeout=5.0):
                    print(f"✓ Enabled linger for {username} — user D-Bus now available")
                    return
                # enable-linger succeeded but the socket never appeared.
                _raise_user_systemd_unavailable(
                    username,
                    reason="Linger was enabled, but the user D-Bus socket did not appear.",
                    fix_hint=(
                        "  Log out and log back in, then re-run the command.\n"
                        f"  Or reboot and run: systemctl --user start {get_service_name()}"
                    ),
                )
            _raise_user_systemd_unavailable(
                username,
                reason=f"loginctl enable-linger was denied: {_completed_process_detail(result)}",
                fix_hint=sudo_hint,
            )

    _raise_user_systemd_unavailable(
        username,
        reason=f"User D-Bus session is not available ({linger_detail or 'linger disabled'}).",
        fix_hint=sudo_hint,
    )


def _raise_user_systemd_unavailable(username: str, *, reason: str, fix_hint: str) -> None:
    """Build a user-facing error message and raise UserSystemdUnavailableError."""
    from hermes_cli.gateway import UserSystemdUnavailableError
    msg = (
        f"{reason}\n"
        "  systemctl --user cannot reach the user D-Bus session in this shell.\n"
        "\n"
        "  To fix:\n"
        f"{fix_hint}\n"
        "\n"
        "  Alternative: run the gateway in the foreground (stays up until\n"
        "  you exit / close the terminal):\n"
        "    hermes gateway run"
    )
    raise UserSystemdUnavailableError(msg)


def _systemctl_cmd(system: bool = False) -> list[str]:
    from hermes_cli.gateway import _ensure_user_systemd_env
    if not system:
        _ensure_user_systemd_env()
    return ["systemctl"] if system else ["systemctl", "--user"]


def _run_systemctl(args: list[str], *, system: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run systemctl; raise RuntimeError (not raw FileNotFoundError) if missing, for callers bypassing
    ``supports_systemd_services()``."""
    from hermes_cli.gateway import _systemctl_cmd, subprocess
    try:
        return subprocess.run(_systemctl_cmd(system) + args, **kwargs)
    except FileNotFoundError:
        from hermes_cli.gateway_command_errors import SystemctlUnavailableError
        raise SystemctlUnavailableError() from None


def _service_scope_label(system: bool = False) -> str:
    return "system" if system else "user"


def get_installed_systemd_scopes() -> list[str]:
    from hermes_cli.gateway import Path, get_systemd_unit_path
    scopes: list[str] = []
    seen_paths: set[Path] = set()
    for system, label in ((False, "user"), (True, "system")):
        unit_path = get_systemd_unit_path(system=system)
        if unit_path not in seen_paths and unit_path.exists():
            scopes.append(label)
            seen_paths.add(unit_path)
    return scopes


def has_conflicting_systemd_units() -> bool:
    from hermes_cli.gateway import get_installed_systemd_scopes
    return len(get_installed_systemd_scopes()) > 1


# Legacy pre-rename names: explicit allowlist (NOT a glob) so profile and third-party units never match.
_LEGACY_SERVICE_NAMES: tuple[str, ...] = ("hermes.service",)

# ExecStart markers identifying a unit as running our gateway; a legacy unit is flagged only if one matches.
_LEGACY_UNIT_EXECSTART_MARKERS: tuple[str, ...] = (
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
)


def _legacy_unit_search_paths() -> list[tuple[bool, Path]]:
    """``[(is_system, base_dir), ...]`` to scan for legacy units; factored out so tests can monkeypatch."""
    from hermes_cli.gateway import _SYSTEM_UNIT_DIR, user_systemd_unit_dir
    return [(False, user_systemd_unit_dir()), (True, _SYSTEM_UNIT_DIR)]


def _find_legacy_hermes_units() -> list[tuple[str, Path, bool]]:
    """``[(unit_name, unit_path, is_system)]`` for legacy gateway units (e.g. ``hermes.service``), which
    fight the current unit for the bot token (SIGTERM flap loop). Explicit name allowlist + ExecStart
    marker check so profile/third-party units never match; no mutation.

    Detects unit files installed by older Hermes versions that used a different service name (e.g. When both
    a legacy unit and the current ``hermes-gateway.service`` are active, they fight over the same bot token
    — the PR #5646 signal-recovery change turns this into a 30-second SIGTERM flap loop.
    """
    from hermes_cli.gateway import (
        Path,
        _LEGACY_SERVICE_NAMES,
        _LEGACY_UNIT_EXECSTART_MARKERS,
        _legacy_unit_search_paths,
    )
    results: list[tuple[str, Path, bool]] = []
    for is_system, base in _legacy_unit_search_paths():
        for name in _LEGACY_SERVICE_NAMES:
            unit_path = base / name
            try:
                if not unit_path.exists():
                    continue
                text = unit_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue
            if any(marker in text for marker in _LEGACY_UNIT_EXECSTART_MARKERS):
                results.append((name, unit_path, is_system))
    return results


def has_legacy_hermes_units() -> bool:
    """Return True when any legacy Hermes gateway unit files exist."""
    from hermes_cli.gateway import _find_legacy_hermes_units
    return bool(_find_legacy_hermes_units())


def print_legacy_unit_warning() -> None:
    """Warn about installed legacy gateway units; prints nothing when there are none."""
    from hermes_cli.gateway import _find_legacy_hermes_units, _service_scope_label, print_info, print_warning
    legacy = _find_legacy_hermes_units()
    if not legacy:
        return
    print_warning("Legacy Hermes gateway unit(s) detected from an older install:")
    for name, path, is_system in legacy:
        print_info(f"    {path}  ({_service_scope_label(is_system)} scope)")
    print_info("  These run alongside the current hermes-gateway service and")
    print_info("  cause SIGTERM flap loops — both try to use the same bot token.")
    print_info("  Remove them with:")
    print_info("    hermes gateway migrate-legacy")


def remove_legacy_hermes_units(interactive: bool = True, dry_run: bool = False) -> tuple[int, list[Path]]:
    """Stop, disable, and remove legacy gateway units. ``interactive=False`` skips the prompt; ``dry_run``
    only lists. Returns ``(removed_count, remaining_paths)`` (remaining: e.g. system-scope when not root)."""
    from hermes_cli.gateway import (
        Path,
        _find_legacy_hermes_units,
        _run_systemctl,
        _service_scope_label,
        contextlib,
        os,
        print_info,
        print_success,
        print_warning,
        prompt_yes_no,
    )
    legacy = _find_legacy_hermes_units()
    if not legacy:
        print("No legacy Hermes gateway units found.")
        return 0, []

    print()
    print("Legacy Hermes gateway unit(s) found:")
    for name, path, is_system in legacy:
        print(f"  {path}  ({_service_scope_label(is_system)} scope)")
    print()

    if dry_run:
        print("(dry-run — nothing removed)")
        return 0, [p for _, p, _ in legacy]

    if interactive and not prompt_yes_no("Remove these legacy units?", True):
        print("Skipped. Run again with: hermes gateway migrate-legacy")
        return 0, [p for _, p, _ in legacy]

    removed = 0
    remaining: list[Path] = []

    def _remove_units(units: list[tuple[str, Path]], *, system: bool) -> None:
        nonlocal removed
        for name, path in units:
            try:
                _run_systemctl(["stop", name], system=system, check=False, timeout=90)
                _run_systemctl(["disable", name], system=system, check=False, timeout=30)
                path.unlink(missing_ok=True)
                print(f"  ✓ Removed {path}")
                removed += 1
            except (OSError, RuntimeError) as e:
                print(f"  ⚠ Could not remove {path}: {e}")
                remaining.append(path)
        with contextlib.suppress(RuntimeError):
            _run_systemctl(["daemon-reload"], system=system, check=False, timeout=30)

    user_units = [(n, p) for n, p, is_sys in legacy if not is_sys]
    system_units = [(n, p) for n, p, is_sys in legacy if is_sys]
    if user_units:
        _remove_units(user_units, system=False)

    # System-scope removal (needs root)
    if system_units:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux systemd removal path, guarded by `if system == "Linux"` / systemd-only branch
            print()
            print_warning("System-scope legacy units require root to remove.")
            print_info("  Re-run with: sudo hermes gateway migrate-legacy")
            remaining.extend(path for _, path in system_units)
        else:
            _remove_units(system_units, system=True)

    print()
    if remaining:
        print_warning(f"{len(remaining)} legacy unit(s) still present — see messages above.")
    else:
        print_success(f"Removed {removed} legacy unit(s).")

    return removed, remaining


def print_systemd_scope_conflict_warning() -> None:
    from hermes_cli.gateway import get_installed_systemd_scopes, print_info, print_warning
    scopes = get_installed_systemd_scopes()
    if len(scopes) < 2:
        return

    print_warning(f"Both user and system gateway services are installed ({' + '.join(scopes)}).")
    print_info("  This is confusing and can make start/stop/status behavior ambiguous.")
    print_info("  Default gateway commands target the user service unless you pass --system.")
    print_info("  Keep one of these:")
    print_info("    hermes gateway uninstall")
    print_info("    sudo hermes gateway uninstall --system")


def refuses_container_user_scope_install(system: bool) -> bool:
    """True (after printing the guidance) when a fresh USER-scope unit was requested inside a container.

    A systemd container passes ``supports_systemd_services()`` on purpose so ``--system`` keeps working,
    but a user unit there is not container-scoped: the unit file and its ``default.target.wants`` symlink
    land in ``~/.config/systemd/user`` — commonly the host's own home bind-mounted in — so the host's
    ``systemd --user`` enables it too and a second gateway polls the same bot token outside the container.
    Callers decide between ``sys.exit(1)`` (CLI) and skipping the install (wizard)."""
    from hermes_cli.gateway import _print_info_lines, is_container, print_error
    if system or not is_container():
        return False
    print_error("Refusing to install a user-scope systemd gateway service inside a container.")
    _print_info_lines(
        "The unit file and its enable symlink would be written to the home directory, which is",
        "commonly the host's own home bind-mounted in — the host's user manager then enables and",
        "starts the same unit, so a second gateway polls the same bot token outside the container",
        "(Telegram: 'Conflict: terminated by other getUpdates request').",
        "",
        "  hermes gateway run                                # run as the container's main process",
        "  docker run --restart unless-stopped ...           # container restart policy",
        "",
        "If systemd manages this container (systemd as PID 1), install an isolated system service instead:",
        "  sudo hermes gateway install --system --run-as-user <user>",
    )
    return True


def _require_root_for_system_service(action: str) -> None:
    from hermes_cli.gateway import SystemScopeRequiresRootError, os
    if os.geteuid() != 0:  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
        raise SystemScopeRequiresRootError(f"System gateway {action} requires root. Re-run with sudo.", action)


def _system_service_identity(run_as_user: str | None = None) -> tuple[str, str, str, int]:
    from hermes_cli.gateway import os, print_info, print_warning
    import getpass
    import grp
    import pwd
    username = (
        run_as_user or os.getenv("SUDO_USER") or os.getenv("USER") or os.getenv("LOGNAME") or getpass.getuser()
    ).strip()
    if not username:
        raise ValueError("Could not determine which user the gateway service should run as")
    if username == "root" and not run_as_user:
        raise ValueError(
            "Refusing to install the gateway system service as root; pass --run-as-user root to override (e.g. in LXC containers)"
        )
    if username == "root":
        print_warning("Installing gateway service to run as root.")
        print_info("  This is fine for LXC/container environments but not recommended on bare-metal hosts.")

    try:
        user_info = pwd.getpwnam(username)
    except KeyError as e:
        raise ValueError(f"Unknown user: {username}") from e
    return username, grp.getgrgid(user_info.pw_gid).gr_name, user_info.pw_dir, user_info.pw_uid


def _read_systemd_user_from_unit(unit_path: Path) -> str | None:
    if not unit_path.exists():
        return None
    for line in unit_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("User="):
            return line.split("=", 1)[1].strip() or None
    return None


def _default_system_service_user() -> str | None:
    from hermes_cli.gateway import os
    for candidate in (os.getenv("SUDO_USER"), os.getenv("USER"), os.getenv("LOGNAME")):
        candidate = (candidate or "").strip()
        if candidate and candidate != "root":
            return candidate
    return None


def prompt_linux_gateway_install_scope() -> str | None:
    # Only root can create a boot-time system service; never hand a non-root user a "re-run under sudo" recipe.
    from hermes_cli.gateway import os, print_info, prompt_choice
    is_root = os.geteuid() == 0  # windows-footgun: ok — Linux systemd install wizard, never invoked on Windows
    options = ["User service (no sudo; best for laptops/dev boxes; may need linger after logout)"]
    values: list[str | None] = ["user"]
    if is_root:
        options.append("System service (starts on boot; runs as your chosen user)")
        values.append("system")
    options.append("Skip service install for now")
    values.append(None)
    choice = prompt_choice("  Choose how the gateway should run in the background:", options, default=0)
    if not is_root and choice == 0:
        print_info("  Tip: for a boot-time system service, re-run setup as root (e.g. from a root shell or `sudo -i`).")
    return values[choice]


def install_linux_gateway_from_setup(force: bool = False, enable_on_startup: bool = True) -> tuple[str | None, bool]:
    from hermes_cli.gateway import (
        _default_system_service_user,
        os,
        print_error,
        print_warning,
        prompt,
        prompt_linux_gateway_install_scope,
        refuses_container_user_scope_install,
        systemd_install,
    )
    scope = prompt_linux_gateway_install_scope()
    if scope is None:
        return None, False

    if scope == "system":
        run_as_user = _default_system_service_user()
        if os.geteuid() != 0:  # windows-footgun: ok — Linux systemd install wizard, never invoked on Windows
            # Unreachable from the wizard (system scope only offered to root); defensive guard for direct callers.
            print_warning(
                "  System service install requires root. Re-run setup from a "
                "root shell, or install a user service instead: hermes gateway install"
            )
            return scope, False

        while not run_as_user:
            run_as_user = (prompt("  Run the system gateway service as which user?", default="") or "").strip()
            if not run_as_user:
                print_error("  Enter a username.")

        systemd_install(force=force, system=True, run_as_user=run_as_user, enable_on_startup=enable_on_startup)
        return scope, True

    if refuses_container_user_scope_install(system=False):
        return scope, False
    systemd_install(force=force, system=False, enable_on_startup=enable_on_startup)
    return scope, True


def ensure_gateway_service(context: str = "setup") -> bool:
    """Install and start a user-scope gateway service without prompting (``hermes setup``/``import``).
    A zero-platform gateway is a supported degraded mode (cron runs), so this never gates on messaging
    config. Never raises; True when a service is installed and running."""
    from hermes_cli.gateway import (
        SystemScopeRequiresRootError,
        UserSystemdUnavailableError,
        _gw_windows,
        _is_service_installed,
        _is_service_running,
        _print_indented,
        _print_system_scope_remediation,
        _served_profile_needs_no_service,
        has_conflicting_systemd_units,
        is_macos,
        is_windows,
        launchd_install,
        launchd_start,
        print_info,
        print_success,
        print_systemd_scope_conflict_warning,
        print_warning,
        supports_systemd_services,
        systemd_install,
        systemd_start,
    )
    from hermes_constants import is_container
    if is_container():
        # Containers use restart policies, not service managers.
        print_info("Start the gateway to bring your bots online:")
        print_info("   hermes gateway run          # Run as container main process")
        print_info("")
        print_info("For automatic restarts, use a Docker restart policy:")
        print_info("   docker run --restart unless-stopped ...")
        return False

    supports_systemd = supports_systemd_services()
    if not (supports_systemd or is_macos() or is_windows()):
        print_info("  No supported service manager found on this host.")
        print_info("  Run the gateway in the foreground with: hermes gateway")
        return False

    try:
        if _is_service_running():
            return True
        if _served_profile_needs_no_service():
            return True
        if not _is_service_installed():
            if supports_systemd and has_conflicting_systemd_units():
                # Both units would fight over bot tokens; don't pile a fresh install onto a conflicted state.
                print_systemd_scope_conflict_warning()
                return False
            print_info("  Installing the gateway background service ...")
            if supports_systemd:
                systemd_install(force=False, non_interactive=True)
            elif is_macos():
                launchd_install(force=False)
            else:
                _gw_windows().install(force=False)  # Registers the Scheduled Task AND starts it.
                print_success("  Gateway service installed and started.")
                return True
        if supports_systemd:
            systemd_start()
        elif is_macos():
            launchd_start()
        else:
            _gw_windows().start()
        print_success("  Gateway service running (cron jobs + messaging platforms).")
        return True
    except UserSystemdUnavailableError as e:
        print_warning("  Could not reach user systemd to start the gateway service:")
        _print_indented(str(e), print_info)
    except SystemScopeRequiresRootError as e:
        print_warning(f"  Gateway service needs root for this scope: {e}")
        _print_system_scope_remediation("start")
    except SystemExit:
        # Some install/start paths sys.exit() on hard failures (temp-HOME guard); never abort setup/import.
        print_warning("  Gateway service install did not complete.")
        print_info("  You can retry manually: hermes gateway install")
    except Exception as e:
        print_warning(f"  Gateway service install failed: {e}")
        print_info("  You can retry manually: hermes gateway install")
    return False


def get_systemd_linger_status(username: str | None = None) -> tuple[bool | None, str]:
    """Linger status for *username* or the current user when omitted.

    System-scope gateway installation runs as root but the service runs as a
    configured target user, so querying the caller would validate the wrong
    user manager.
    """
    from hermes_cli.gateway import (
        _CAPTURE_TEXT,
        _completed_process_detail,
        is_linux,
        is_termux,
        os,
        shutil,
        subprocess,
    )
    if is_termux():
        return None, "not supported in Termux"
    if not is_linux():
        return None, "not supported on this platform"
    if not shutil.which("loginctl"):
        return None, "loginctl not found"

    if username is None:
        username = os.getenv("USER") or os.getenv("LOGNAME")
    if not username:
        try:
            import pwd
            username = pwd.getpwuid(os.getuid()).pw_name  # windows-footgun: ok — POSIX loginctl helper, never invoked on Windows
        except Exception:
            return None, "could not determine current user"

    try:
        result = subprocess.run(
            ["loginctl", "show-user", username, "--property=Linger", "--value"],
            check=False, timeout=10, **_CAPTURE_TEXT,
        )
    except Exception as e:
        return None, str(e)

    if result.returncode != 0:
        return None, _completed_process_detail(result) or "loginctl query failed"

    value = (result.stdout or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return True, ""
    if value in {"no", "false", "0"}:
        return False, ""
    return None, f"unexpected loginctl output: {value or '<empty>'}"


def _detect_venv_dir() -> Path | None:
    """Active virtualenv dir: ``sys.prefix``, then ``VIRTUAL_ENV`` (uv sets it without changing
    sys.prefix), then .venv/venv under PROJECT_ROOT; None if none found."""
    from hermes_cli.gateway import PROJECT_ROOT, Path, os, sys
    candidates: list[Path] = []
    if sys.prefix != sys.base_prefix:
        candidates.append(Path(sys.prefix))
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates += [PROJECT_ROOT / ".venv", PROJECT_ROOT / "venv"]
    return next((venv for venv in candidates if venv.is_dir()), None)


def get_python_path() -> str:
    from hermes_cli.gateway import _detect_venv_dir, is_windows, sys
    venv = _detect_venv_dir()
    if venv is not None:
        try:
            from hermes_constants import venv_python_path
        except ImportError:
            # Update-boundary: a gateway restarted mid-update can hold a stale hermes_constants
            # without this symbol; see _reload_hermes_constants() in hermes_cli/managed_uv.py.
            from hermes_cli.managed_uv import _reload_hermes_constants
            venv_python_path = _reload_hermes_constants().venv_python_path

        venv_python = venv_python_path(venv, windows=is_windows())
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


# =============================================================================
# Systemd (Linux)
# =============================================================================


def _build_user_local_paths(home: Path, path_entries: list[str]) -> list[str]:
    """Return user-local bin dirs that exist and aren't already in *path_entries*."""
    from hermes_cli.gateway import Path
    candidates = [
        str(home / ".local" / "bin"),  # uv, uvx, pip-installed CLIs
        str(home / ".cargo" / "bin"),  # Rust/cargo tools
        str(home / "go" / "bin"),  # Go tools
        str(home / ".npm-global" / "bin"),  # npm global packages
    ]
    return [p for p in candidates if p not in path_entries and Path(p).exists()]


def _build_wsl_interop_paths(path_entries: list[str]) -> list[str]:
    """WSL Windows-interop PATH entries for generated units: systemd services don't inherit the
    Windows PATH (``/mnt/c/WINDOWS/System32``…), so ``powershell.exe``/``cmd.exe`` break unless persisted."""
    from hermes_cli.gateway import Path, is_wsl, os, shutil
    if not is_wsl():
        return []

    candidates = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry.startswith("/mnt/")]
    for executable in ("powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(str(Path(resolved).parent))
    candidates += [
        entry
        for entry in (
            "/mnt/c/WINDOWS/system32",
            "/mnt/c/WINDOWS",
            "/mnt/c/WINDOWS/System32/Wbem",
            "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
            "/mnt/c/WINDOWS/System32/OpenSSH/",
        )
        if Path(entry).exists()
    ]

    result: list[str] = []
    seen = set(path_entries)
    for entry in candidates:
        if entry and entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


def _remap_path_for_user(path: str, target_home_dir: str) -> str:
    """Swap the ``Path.home()`` prefix of *path* for *target_home_dir*; other paths return unchanged.
    Intentionally does NOT resolve symlinks."""
    from hermes_cli.gateway import Path
    current_home = Path.home()
    p = Path(path).expanduser()
    try:
        relative = p.relative_to(current_home)
        return str(Path(target_home_dir) / relative)
    except ValueError:
        return str(p)




def _print_linger_enable_warning(username: str, detail: str | None = None, *, system: bool = False) -> None:
    from hermes_cli.gateway import _systemd_cli_bits, get_service_name
    print()
    if system:
        print(f"⚠ Linger not enabled for {username} — cron and Kanban workers cannot start (no user D-Bus).")
    else:
        print("⚠ Linger not enabled — gateway may stop when you close this terminal.")
    if detail:
        print(f"  Auto-enable failed: {detail}")
    print()
    print("  Enable it manually:" if system else "  On headless servers (VPS, cloud instances) run:")
    print(f"    sudo loginctl enable-linger {username}")
    print()
    print("  Then restart the gateway:")
    sudo, _, user_flag = _systemd_cli_bits(system)
    print(f"    {sudo}systemctl {user_flag}restart {get_service_name()}.service")
    print()


def _ensure_linger_enabled(username: str | None = None, *, system: bool = False) -> bool:
    """Enable linger for *username* (default: the current user) when possible.

    A user unit needs linger so the gateway survives logout. A system unit (``system=True``) needs
    it for its ``User=`` so ``user@<uid>.service`` provides the D-Bus that ``systemd-run --user
    --scope`` — every restart-safe cron/Kanban worker — connects to (#104893). Returns True only
    when linger was enabled by this call.
    """
    from hermes_cli.gateway import (
        Path,
        _completed_process_detail,
        _loginctl_enable_linger,
        _print_linger_enable_warning,
        get_systemd_linger_status,
        is_linux,
        is_termux,
        shutil,
    )
    if is_termux() or not is_linux():
        return False

    if username is None:
        import getpass
        username = getpass.getuser()
    enabled_msg = (
        f"✓ Systemd linger is enabled for {username} (worker D-Bus available)" if system
        else "✓ Systemd linger is enabled (service survives logout)"
    )
    if Path(f"/var/lib/systemd/linger/{username}").exists():
        print(enabled_msg)
        return False

    linger_enabled, linger_detail = get_systemd_linger_status(username)
    if linger_enabled is True:
        print(enabled_msg)
        return False

    if not shutil.which("loginctl"):
        _print_linger_enable_warning(username, linger_detail or "loginctl not found", system=system)
        return False

    if system:
        print(f"Enabling linger for {username} so cron and Kanban workers can reach systemd-run --user...")
    else:
        print("Enabling linger so the gateway survives SSH logout...")
    try:
        result = _loginctl_enable_linger(username)
    except Exception as e:
        _print_linger_enable_warning(username, str(e), system=system)
        return False

    if result.returncode != 0:
        _print_linger_enable_warning(username, _completed_process_detail(result) or linger_detail, system=system)
        return False
    print(f"✓ Enabled linger for {username}" if system else "✓ Linger enabled — gateway will persist after logout")
    return True


def _ensure_system_service_linger(username: str) -> None:
    """Enable linger for the installed unit's ``User=`` (root included: restart-safe workers always cross
    ``systemd-run --user``, so a root gateway needs ``user@0.service`` just the same).

    After a fresh enable, wait for the TARGET user's bus: logind starts ``user@<uid>.service``
    asynchronously and ``--start-now`` boots the gateway immediately. A gateway that was already running
    keeps its bus-less environment and ``systemctl start`` on an active unit is a no-op — say so rather
    than let the repair silently not take."""
    from hermes_cli.gateway import (
        _ensure_linger_enabled,
        _systemd_unit_is_active,
        _wait_for_target_user_bus,
        get_service_name,
    )
    if not _ensure_linger_enabled(username, system=True):
        return
    import pwd
    uid = pwd.getpwnam(username).pw_uid  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    if _wait_for_target_user_bus(uid):
        print(f"✓ /run/user/{uid}/bus is up — cron and Kanban workers can use systemd-run --user")
    else:
        print(f"⚠ /run/user/{uid}/bus did not appear within 5s.")
        print(f"  Start the user manager: sudo systemctl start user@{uid}.service")
    if _systemd_unit_is_active(system=True):
        print("  The running gateway was started without a user D-Bus; restart it to pick one up:")
        print(f"    sudo systemctl restart {get_service_name()}.service")


def _select_systemd_scope(system: bool = False) -> bool:
    from hermes_cli.gateway import get_systemd_unit_path
    return system or (get_systemd_unit_path(system=True).exists() and not get_systemd_unit_path(system=False).exists())


def _system_scope_wizard_would_need_root(system: bool = False) -> bool:
    """True when the wizard would trigger a system-scope operation as non-root — mirrors
    ``_select_systemd_scope`` so the dead-end is detected BEFORE prompting."""
    from hermes_cli.gateway import _select_systemd_scope, os
    if os.geteuid() == 0:  # windows-footgun: ok — systemd scope wizard decision, never invoked on Windows
        return False
    return _select_systemd_scope(system=system)


def _print_system_scope_remediation(action: str) -> None:
    """Print remediation when the wizard skips a system-scope action because the user isn't root."""
    from hermes_cli.gateway import get_service_name, print_info, print_warning
    print_warning(f"Gateway is installed as a system-wide service — {action} requires root.")
    print_info("  Options:")
    print_info(f"    1. {action.capitalize()} it this time:")
    print_info(f"         sudo systemctl {action} {get_service_name()}")
    print_info("    2. Switch to a per-user service (recommended for personal use):")
    print_info("         sudo hermes gateway uninstall --system")
    print_info("         hermes gateway install")
    print_info("         hermes gateway start")


def _get_restart_drain_timeout() -> float:
    """Return the configured gateway restart drain timeout in seconds."""
    from hermes_cli.gateway import (
        DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
        os,
        parse_restart_drain_timeout,
        read_raw_config,
    )
    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        cfg = read_raw_config()
        agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
        raw = str(agent_cfg.get("restart_drain_timeout", DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT))
    return parse_restart_drain_timeout(raw)


def _agent_timeout_setting(env_var: str, key: str, parse) -> float:
    """``parse(env)`` when the env var is non-empty, else ``parse(agent.<key>)`` (None if unset)."""
    from hermes_cli.gateway import os, read_raw_config
    env_raw = os.getenv(env_var)
    if env_raw is not None and str(env_raw).strip() != "":
        return parse(env_raw)
    cfg = read_raw_config()
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    if isinstance(agent_cfg, dict) and key in agent_cfg:
        return parse(agent_cfg.get(key))
    return parse(None)


def _get_cron_drain_timeout() -> float:
    """Return the configured cron-only drain floor in seconds.

    See #82161.
    """
    from hermes_cli.gateway import _agent_timeout_setting, parse_cron_drain_timeout
    return _agent_timeout_setting("HERMES_CRON_DRAIN_TIMEOUT", "cron_drain_timeout", parse_cron_drain_timeout)


def _get_restart_exit_wait_budget() -> float:
    """CLI wait for gateway exit after SIGUSR1 / self-restart (#77184)."""
    from hermes_cli.gateway import (
        _agent_timeout_setting,
        _get_restart_drain_timeout,
        parse_restart_after_turn_timeout,
        resolve_restart_exit_wait_budget,
    )
    return resolve_restart_exit_wait_budget(
        # TimeoutStopSec must cover the full stop budget, not just restart_drain_timeout. Cron work can
        # legally wait cron_drain_timeout plus cleanup reserve before interrupt/teardown, and systemd
        # SIGKILLs if the unit's deadline is shorter (#94759). 30s of post-drain headroom is preserved on
        # top, with a 60s floor.
        _get_restart_drain_timeout(),
        _agent_timeout_setting(
            "HERMES_RESTART_AFTER_TURN_TIMEOUT", "restart_after_turn_timeout", parse_restart_after_turn_timeout
        ),
    )


def systemd_install(
    force: bool = False,
    system: bool = False,
    run_as_user: str | None = None,
    enable_on_startup: bool = True,
    non_interactive: bool = False,
):
    from hermes_cli.gateway import (
        _ensure_linger_enabled,
        _ensure_system_service_linger,
        _read_systemd_user_from_unit,
        _refuse_temp_home_service_write,
        _require_root_for_system_service,
        _run_systemctl,
        _service_scope_label,
        _sync_hermes_home_from_systemd_unit,
        _systemd_cli_bits,
        generate_systemd_unit,
        get_service_name,
        get_systemd_unit_path,
        has_legacy_hermes_units,
        print_legacy_unit_warning,
        print_systemd_scope_conflict_warning,
        prompt_yes_no,
        refresh_systemd_unit_if_needed,
        remove_legacy_hermes_units,
        systemd_unit_is_current,
    )
    if system:
        _require_root_for_system_service("install")

    # Offer to remove legacy units first: alongside the new unit they flap-fight for the bot token.
    if has_legacy_hermes_units():
        print()
        print_legacy_unit_warning()
        print()
        if non_interactive or prompt_yes_no("Remove the legacy unit(s) before installing?", True):
            remove_legacy_hermes_units(interactive=False)
            print()

    unit_path = get_systemd_unit_path(system=system)
    scope_label = _service_scope_label(system)
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)

    # Existing system units already pin HERMES_HOME; adopt it before any regenerate.
    if unit_path.exists():
        _sync_hermes_home_from_systemd_unit(system=system)

    if unit_path.exists() and not force:
        if not systemd_unit_is_current(system=system):
            print(f"↻ Repairing outdated {scope_label} systemd service at: {unit_path}")
            refresh_systemd_unit_if_needed(system=system)
            if enable_on_startup:
                _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
            print(f"✓ {scope_label.capitalize()} service definition updated")
        else:
            print(f"Service already installed at: {unit_path}")
            print("Use --force to reinstall")
        # Same post-install guarantee as a fresh install: a repaired user unit must survive logout too.
        configured_user = _read_systemd_user_from_unit(unit_path) if system else None
        if configured_user:
            _ensure_system_service_linger(configured_user)
        elif not system:
            _ensure_linger_enabled()
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    new_unit = generate_systemd_unit(system=system, run_as_user=run_as_user)
    if _refuse_temp_home_service_write(new_unit, "systemd unit"):
        return
    print(f"Installing {scope_label} systemd service to: {unit_path}")
    unit_path.write_text(new_unit, encoding="utf-8")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    if enable_on_startup:
        _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)

    print()
    print(f"✓ {scope_label.capitalize()} service {'installed and enabled' if enable_on_startup else 'installed'}!")
    print()
    print("Next steps:")
    print(f"  {sudo}hermes gateway start{scope_flag}              # Start the service")
    print(f"  {sudo}hermes gateway status{scope_flag}             # Check status")
    print(f"  journalctl {user_flag}-u {get_service_name()} -f  # View logs")
    print()

    if system:
        configured_user = _read_systemd_user_from_unit(unit_path)
        if configured_user:
            print(f"Configured to run as: {configured_user}")
            _ensure_system_service_linger(configured_user)
    else:
        _ensure_linger_enabled()

    print_systemd_scope_conflict_warning()
    print_legacy_unit_warning()


def _systemd_scope_preamble(
    action: str, system: bool, *, require_installed: bool = True, preflight_user: bool = False
) -> bool:
    """Resolve the effective scope, then enforce root (system) / user D-Bus reachability (user, when
    ``preflight_user``) and — when ``require_installed`` — that the unit exists. Returns the scope."""
    from hermes_cli.gateway import (
        _preflight_user_systemd,
        _require_root_for_system_service,
        _require_service_installed,
        _select_systemd_scope,
    )
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service(action)
    elif preflight_user:
        # Fail fast with guidance when the user D-Bus session is unreachable (raises UserSystemdUnavailableError).
        _preflight_user_systemd()
    if require_installed:
        _require_service_installed(action, system=system)
    return system


def _systemd_unit_belongs_to_current_home(system: bool = False) -> bool:
    """False (with a warning) when the installed unit pins a HERMES_HOME other than this process's: the
    service name then resolved to ANOTHER install's gateway, and stop/disable/unlink would take it down."""
    from hermes_cli.gateway import (
        Path,
        _hermes_home_from_systemd_unit_file,
        _sync_hermes_home_from_systemd_unit,
        get_hermes_home,
        get_systemd_unit_path,
        print_warning,
    )
    _sync_hermes_home_from_systemd_unit(system=system)  # sudo strips HERMES_HOME; adopt the unit's first
    unit_home = _hermes_home_from_systemd_unit_file(system=system)
    if unit_home is None or Path(unit_home).expanduser().resolve() == get_hermes_home().resolve():
        return True
    print_warning(
        f"Refusing to remove {get_systemd_unit_path(system=system)}: it runs HERMES_HOME={unit_home}, "
        f"but this process has HERMES_HOME={get_hermes_home()}"
    )
    return False


def systemd_uninstall(system: bool = False):
    from hermes_cli.gateway import (
        _run_systemctl,
        _service_scope_label,
        _systemd_scope_preamble,
        _systemd_unit_belongs_to_current_home,
        get_service_name,
        get_systemd_unit_path,
    )
    system = _systemd_scope_preamble("uninstall", system, require_installed=False)
    if not _systemd_unit_belongs_to_current_home(system):
        return
    _run_systemctl(["stop", get_service_name()], system=system, check=False, timeout=90)
    _run_systemctl(["disable", get_service_name()], system=system, check=False, timeout=30)

    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service uninstalled")


def _print_service_not_installed(system: bool) -> None:
    from hermes_cli.gateway import _systemd_cli_bits
    sudo, scope_flag, _ = _systemd_cli_bits(system)
    print("✗ Gateway service is not installed")
    print(f"  Run: {sudo}hermes gateway install{scope_flag}")


def _require_service_installed(action: str, system: bool = False) -> None:
    from hermes_cli.gateway import _print_service_not_installed, get_systemd_unit_path, sys
    if not get_systemd_unit_path(system=system).exists():
        _print_service_not_installed(system)
        sys.exit(1)


def systemd_start(system: bool = False):
    from hermes_cli.gateway import (
        _run_systemctl,
        _service_scope_label,
        _systemd_scope_preamble,
        get_service_name,
        refresh_systemd_unit_if_needed,
    )
    system = _systemd_scope_preamble("start", system, preflight_user=True)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; the unit is guaranteed to exist here.
    refresh_systemd_unit_if_needed(system=system)
    _run_systemctl(["start", get_service_name()], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service started")


def systemd_stop(system: bool = False):
    from hermes_cli.gateway import (
        _mark_planned_stop,
        _run_systemctl,
        _service_scope_label,
        _sync_hermes_home_from_systemd_unit,
        _systemd_scope_preamble,
        get_service_name,
        subprocess,
    )
    system = _systemd_scope_preamble("stop", system)
    _sync_hermes_home_from_systemd_unit(system=system)
    _mark_planned_stop()
    try:
        _run_systemctl(["stop", get_service_name()], system=system, check=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(
            f"Gateway {_service_scope_label(system)} service is still stopping after 90s; "
            "check `hermes gateway status` or logs for final shutdown state."
        )
        return
    print(f"✓ {_service_scope_label(system).capitalize()} service stopped")


def systemd_restart(system: bool = False):
    from hermes_cli.gateway import (
        GATEWAY_LOOP_WEDGED,
        _escalate_wedged_gateway,
        _recover_pending_systemd_restart,
        _run_systemctl,
        _systemd_graceful_restart_action,
        _systemd_main_pid,
        _systemd_reset_and_run,
        _systemd_scope_preamble,
        _wait_for_systemd_service_restart,
        get_service_name,
        probe_gateway_loop_liveness,
        refresh_systemd_unit_if_needed,
    )
    system = _systemd_scope_preamble("restart", system, preflight_user=True)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; its os.environ mutation
    # persists for the get_running_pid / drain-timeout reads below.
    refresh_systemd_unit_if_needed(system=system)
    from gateway.status import get_running_pid
    pid = get_running_pid() or _systemd_main_pid(system=system)
    if pid is not None and probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        # Event loop provably dead: SIGUSR1 can't drain it, so bounded SIGTERM → SIGKILL and let systemd relaunch.
        print(
            # Health probe says the event loop is provably dead (#81642): SIGUSR1 can never drain it, so the
            # graceful wait below would burn the full budget. A busy-but-alive gateway (fresh heartbeat)
            # never takes this path — its in-flight work, including the #86684 cron drain floor, keeps the
            # full graceful budget.
            # Health probe says the event loop is provably dead (#81642): the gateway cannot process a
            # graceful shutdown, so waiting the full drain budget only stalls the restart (and `hermes
            # update` behind it) for 180s. Bounded escalation instead: SIGTERM grace → SIGKILL → proceed,
            # ~10s worst case. Never taken for a busy-but-alive gateway — a fresh heartbeat keeps the drain
            # path (and the #86684 cron drain floor) fully intact.
            f"⚠ Gateway PID {pid} event loop is unresponsive — "
            "skipping graceful drain and forcing a bounded stop..."
        )
        _escalate_wedged_gateway(pid)
        svc = get_service_name()
        _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        _run_systemctl(["restart", svc], system=system, check=False, timeout=90)
        _wait_for_systemd_service_restart(system=system, previous_pid=pid)
        return
    if pid is not None:
        service_action = _systemd_graceful_restart_action(system, pid)
        if service_action:
            _systemd_reset_and_run(service_action, system=system, previous_pid=pid)
        return

    if _recover_pending_systemd_restart(system=system, previous_pid=pid):
        return
    _systemd_reset_and_run("restart", system=system, previous_pid=pid)


def _systemd_graceful_restart_action(system: bool, pid: int) -> str | None:
    """SIGUSR1-drain the live gateway ``pid``; return the follow-up ``systemctl`` verb (``"start"`` /
    ``"restart"``) the caller must still issue, or None when systemd already owns the relaunch."""
    from hermes_cli.gateway import (
        _get_restart_exit_wait_budget,
        _graceful_restart_via_sigusr1,
        _read_systemd_unit_properties,
        _service_scope_label,
        _systemd_main_pid_from_props,
        _systemd_service_is_start_limited,
        _wait_for_systemd_service_restart,
    )
    scope_label = _service_scope_label(system).capitalize()
    # Graceful in-band restart, mirroring the systemd branch. Previously this sent a bare SIGTERM and waited
    # ``_get_restart_drain_timeout()`` — which defaults to 0, so the wait could never succeed and every
    # restart fell through to ``kickstart -k``. A bare SIGTERM also leaves ``restart_requested`` False, so
    # the gateway exits 1 instead of 75 and reports itself to chat as "shutting down" rather than
    # "restarting", losing the resume_pending handoff. SIGUSR1 is the drain-aware path: refuse new turns,
    # wait for in-flight work (``agent.restart_after_turn_timeout``), then stop() within
    # ``agent.restart_drain_timeout``. The wait budget must cover BOTH phases plus headroom (#77184) — the
    # raw drain timeout covers only the second. Announce the wait BEFORE it runs: it can last the full
    # budget while the old gateway finishes in-flight agent runs, and it streams into surfaces with no other
    # feedback — the desktop updater's live output most of all, where a silent stop here reads as "update
    # stuck" (#44515).
    wait_budget = _get_restart_exit_wait_budget()
    print(
        f"⏳ {scope_label} service restarting gracefully (PID {pid}) — "
        f"waiting up to {wait_budget:.0f}s for in-flight turns + drain..."
    )
    from hermes_cli.update_cmd_drain_report import drain_progress_reporter
    if not _graceful_restart_via_sigusr1(pid, wait_budget, on_progress=drain_progress_reporter(budget_s=wait_budget)):
        print(f"⚠ Graceful restart did not complete within {int(wait_budget)}s; forcing a service restart...")
        return "restart"

    # Exit 75 hands restart ownership to systemd; observe that replacement rather than restarting again.
    replacement_observed: list[bool] = []
    if _wait_for_systemd_service_restart(system=system, previous_pid=pid, replacement_observed=replacement_observed):
        return None
    if replacement_observed or _systemd_service_is_start_limited(system=system):
        return None

    # A replacement may have started but not reached runtime readiness in time; never stop that generation.
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return None
    replacement_pid = _systemd_main_pid_from_props(props)
    if (
        props.get("ActiveState") in {"active", "activating", "reloading"}
        or props.get("SubState") == "auto-restart"
        or (replacement_pid is not None and replacement_pid != pid)
    ):
        return None

    print("⚠ Systemd did not relaunch the gateway after its graceful exit; starting the inactive service...")
    # ``start`` is intentionally idempotent: a replacement appearing after the snapshot must not be stopped.
    return "start"


def _systemd_reset_and_run(action: str, *, system: bool, previous_pid) -> None:
    """``reset-failed`` then ``systemctl <action>``, then wait for the relaunch. Start-limit
    rejection prints the wait hint instead of raising; a 90s timeout prints where to look."""
    from hermes_cli.gateway import (
        _print_systemd_start_limit_wait,
        _run_systemctl,
        _service_scope_label,
        _systemd_error_indicates_start_limit,
        _systemd_service_is_start_limited,
        _wait_for_systemd_service_restart,
        get_service_name,
        subprocess,
    )
    svc = get_service_name()
    _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
    try:
        _run_systemctl([action, svc], system=system, check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if _systemd_error_indicates_start_limit(exc) or _systemd_service_is_start_limited(system=system):
            _print_systemd_start_limit_wait(system=system)
            return
        raise
    except subprocess.TimeoutExpired:
        print(
            f"Gateway {_service_scope_label(system)} service is still restarting after 90s; "
            "check `hermes gateway status` or logs for final state."
        )
        return
    _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)


def systemd_status(deep: bool = False, system: bool = False, full: bool = False):
    from hermes_cli.gateway import (
        GATEWAY_SERVICE_RESTART_EXIT_CODE,
        _CAPTURE_TEXT,
        _print_runtime_health,
        _print_service_not_installed,
        _read_systemd_unit_properties,
        _read_systemd_user_from_unit,
        _run_systemctl,
        _select_systemd_scope,
        _service_scope_label,
        _systemd_cli_bits,
        _systemd_unit_is_start_limited,
        get_service_name,
        get_systemd_linger_status,
        get_systemd_unit_path,
        has_conflicting_systemd_units,
        has_legacy_hermes_units,
        print_legacy_unit_warning,
        print_systemd_scope_conflict_warning,
        subprocess,
        systemd_unit_is_current,
    )
    system = _select_systemd_scope(system)
    unit_path = get_systemd_unit_path(system=system)
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)

    if not unit_path.exists():
        _print_service_not_installed(system)
        return

    if has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if has_legacy_hermes_units():
        print_legacy_unit_warning()
        print()

    if not systemd_unit_is_current(system=system):
        print("⚠ Installed gateway service definition is outdated")
        print(f"  Run: {sudo}hermes gateway restart{scope_flag}  # auto-refreshes the unit")
        print()

    status_cmd = ["status", svc, "--no-pager"] + (["-l"] if full else [])
    _run_systemctl(status_cmd, system=system, capture_output=False, timeout=10)
    result = _run_systemctl(["is-active", svc], system=system, timeout=10, **_CAPTURE_TEXT)
    if result.stdout.strip() == "active":
        print(f"✓ {scope_label} gateway service is running")
    else:
        print(f"✗ {scope_label} gateway service is stopped")
        print(f"  Run: {sudo}hermes gateway start{scope_flag}")

    configured_user = _read_systemd_user_from_unit(unit_path) if system else None
    if configured_user:
        print(f"Configured to run as: {configured_user}")

    _print_runtime_health()

    unit_props = _read_systemd_unit_properties(system=system)
    active_state = unit_props.get("ActiveState", "")
    result_code = unit_props.get("Result", "")
    if active_state == "activating" and unit_props.get("SubState", "") == "auto-restart":
        print("  ⏳ Restart pending: systemd is waiting to relaunch the gateway")
    elif _systemd_unit_is_start_limited(unit_props):
        print("  ⏳ Restart pending: systemd is temporarily rate-limiting starts")
        print(f"  Run after the start-limit window expires: {sudo}hermes gateway restart{scope_flag}")
        print(f"  Or clear it manually: systemctl {user_flag}reset-failed {svc}")
    elif active_state == "failed" and unit_props.get("ExecMainStatus", "") == str(GATEWAY_SERVICE_RESTART_EXIT_CODE):
        print("  ⚠ Planned restart is stuck in systemd failed state (exit 75)")
        print(f"  Run: systemctl {user_flag}reset-failed {svc} && {sudo}hermes gateway start{scope_flag}")
    elif active_state == "failed" and result_code:
        print(f"  ⚠ Systemd unit result: {result_code}")

    if system:
        print("✓ System service starts at boot without requiring systemd linger")
    else:
        linger_enabled, linger_detail = get_systemd_linger_status()
        if linger_enabled is True:
            print("✓ Systemd linger is enabled (service survives logout)")
        elif linger_enabled is False:
            print("⚠ Systemd linger is disabled (gateway may stop when you log out)")
            print("  Run: sudo loginctl enable-linger $USER")
        elif deep:
            print(f"⚠ Could not verify systemd linger ({linger_detail})")
            print("  If you want the gateway user service to survive logout, run:")
            print("  sudo loginctl enable-linger $USER")

    if deep:
        print()
        print("Recent logs:")
        log_cmd = ["journalctl"] + ([] if system else ["--user"]) + ["-u", svc, "-n", "20", "--no-pager"]
        if full:
            log_cmd.append("-l")
        subprocess.run(log_cmd, timeout=10)
