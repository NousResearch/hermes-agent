"""Profile-scoped native service management for the Telegram Mini App.

The Mini App deliberately has its own narrow service definition.  It does not
inherit the gateway environment and never exposes the listener beyond
loopback; a separately configured HTTPS reverse proxy is the only supported
public ingress.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn


SERVICE_BASENAME = "hermes-telegram-mini-app"


class MiniAppServiceError(RuntimeError):
    """A user-actionable Mini App service lifecycle failure."""


@dataclass(frozen=True)
class MiniAppPaths:
    root: Path
    env: Path
    state: Path
    stdout_log: Path
    stderr_log: Path


def paths_for(hermes_home: Path) -> MiniAppPaths:
    root = hermes_home / "telegram-mini-app"
    return MiniAppPaths(
        root=root,
        env=root / "service.env",
        state=root / "state.json",
        stdout_log=root / "service.log",
        stderr_log=root / "service.error.log",
    )


def ensure_private_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.touch(exist_ok=True, mode=0o600)
    path.chmod(0o600)


def _profile_suffix(hermes_home: Path) -> str:
    """Return a filesystem-safe suffix unique to the active profile home."""
    try:
        from hermes_constants import get_default_hermes_root

        default = get_default_hermes_root().resolve()
        home = hermes_home.resolve()
        if home == default:
            return ""
        profiles = (default / "profiles").resolve()
        rel = home.relative_to(profiles)
        if len(rel.parts) == 1:
            candidate = rel.parts[0]
            if candidate.replace("-", "").replace("_", "").isalnum():
                return candidate
    except (ImportError, ValueError, OSError):
        pass
    return hashlib.sha256(str(hermes_home.resolve()).encode()).hexdigest()[:8]


def service_name(hermes_home: Path) -> str:
    suffix = _profile_suffix(hermes_home)
    return SERVICE_BASENAME if not suffix else f"{SERVICE_BASENAME}-{suffix}"


def systemd_unit_path(hermes_home: Path) -> Path:
    return (
        Path.home()
        / ".config"
        / "systemd"
        / "user"
        / f"{service_name(hermes_home)}.service"
    )


def _env_executable() -> str:
    """Return an absolute env(1) path for the pre-Python clean-room boundary."""
    candidate = shutil.which("env") or "/usr/bin/env"
    return str(Path(candidate).resolve())


def service_command(
    hermes_home: Path, *, python_executable: str | Path | None = None
) -> list[str]:
    """Build the clean runner command used by both systemd and foreground mode.

    ``env -i`` removes the systemd user-manager/interactive-shell environment
    before Python starts, so even package initializers cannot observe provider
    credentials. ``-I`` then excludes the writable cwd, ``PYTHONPATH``, and the
    user site from module resolution.
    """
    # Keep a virtualenv's absolute launcher path intact: resolving its symlink
    # to the base interpreter discards pyvenv.cfg and therefore its site-packages.
    python = Path(python_executable or sys.executable).expanduser().absolute()
    return [
        _env_executable(),
        "-i",
        f"HOME={Path.home().resolve()}",
        f"HERMES_HOME={hermes_home.resolve()}",
        str(python),
        "-I",
        "-m",
        "plugins.platforms.telegram.mini_app.run",
    ]


def exec_clean_runner(hermes_home: Path) -> NoReturn:
    """Replace the current CLI with the same clean runner systemd executes."""
    command = service_command(hermes_home)
    os.chdir("/")
    os.execve(command[0], command, {})


def _default_root(hermes_home: Path) -> Path:
    try:
        from hermes_constants import get_default_hermes_root

        return get_default_hermes_root().expanduser().absolute()
    except (ImportError, OSError):
        return hermes_home.expanduser().absolute()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following any symlink component."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _validated_data_path(
    path: Path,
    *,
    root: Path,
    expected: str,
    optional: bool = True,
) -> Path:
    """Validate one allowlisted data path without resolving away symlinks."""
    lexical = _lexical_absolute(path)
    lexical_root = _lexical_absolute(root)
    try:
        lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise MiniAppServiceError(
            f"Mini App data path escapes its expected root: {lexical}"
        ) from exc

    current = Path(lexical.anchor)
    missing = False
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing = True
            break
        except OSError as exc:
            raise MiniAppServiceError(
                f"Could not validate Mini App data path: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise MiniAppServiceError(
                f"Mini App data paths must not contain symlinks: {current}"
            )
        if current != lexical and not stat.S_ISDIR(metadata.st_mode):
            raise MiniAppServiceError(
                f"Mini App data path has a non-directory parent: {current}"
            )

    if missing:
        if not optional:
            raise MiniAppServiceError(
                f"Required Mini App data path is missing: {lexical}"
            )
        return lexical

    metadata = lexical.lstat()
    if expected == "file" and not stat.S_ISREG(metadata.st_mode):
        raise MiniAppServiceError(
            f"Mini App data path must be a regular file: {lexical}"
        )
    if expected == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise MiniAppServiceError(f"Mini App data path must be a directory: {lexical}")
    try:
        lexical.resolve(strict=True).relative_to(lexical_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise MiniAppServiceError(
            f"Mini App data path does not resolve inside its expected root: {lexical}"
        ) from exc
    return lexical


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _home_namespace_roots() -> tuple[Path, ...]:
    """Roots systemd masks when ProtectHome=tmpfs."""
    return (Path("/home"), Path("/root"), Path("/run/user"))


def _runtime_read_paths(hermes_home: Path) -> list[Path]:
    """Expose installed Python trees only when a filesystem mask would hide them."""
    default_root = _default_root(hermes_home)
    masked_roots = (*_home_namespace_roots(), default_root)
    candidates = {Path(sys.prefix).absolute(), Path(sys.base_prefix).absolute()}
    paths = {
        path.resolve()
        for path in candidates
        if path.exists() and any(_is_within(path, root) for root in masked_roots)
    }
    unsafe = [
        path
        for path in paths
        if _is_within(default_root, path) or _is_within(hermes_home.resolve(), path)
    ]
    if unsafe:
        raise MiniAppServiceError(
            "The Python runtime directory contains the Hermes data root and cannot "
            "be exposed safely to the Mini App service. Install Hermes into a "
            "dedicated venv below, not at, HERMES_HOME."
        )
    module_path = Path(__file__).resolve()
    if any(_is_within(module_path, root) for root in masked_roots) and not any(
        _is_within(module_path, runtime) for runtime in paths
    ):
        raise MiniAppServiceError(
            "The Mini App service requires a wheel/venv installation when Hermes "
            "source is inside the protected home or data root; editable source trees "
            "cannot be exposed without also exposing credential-bearing files."
        )
    return sorted(paths, key=str)


def _selected_kanban_db(default_root: Path) -> Path:
    current = _validated_data_path(
        default_root / "kanban" / "current",
        root=default_root,
        expected="file",
    )
    try:
        slug = current.read_text(encoding="utf-8").strip()
    except OSError:
        slug = ""
    if slug and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug) and slug != "default":
        selected = default_root / "kanban" / "boards" / slug / "kanban.db"
    else:
        selected = default_root / "kanban.db"
    return _validated_data_path(selected, root=default_root, expected="file")


def _skill_manifest_paths(home: Path) -> set[Path]:
    skills_root = _validated_data_path(home / "skills", root=home, expected="directory")
    if not skills_root.exists():
        return set()
    manifests: set[Path] = set()
    try:
        entries = list(skills_root.iterdir())
    except OSError as exc:
        raise MiniAppServiceError(
            "Could not enumerate Mini App skill manifests."
        ) from exc
    for entry in entries:
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise MiniAppServiceError(
                f"Mini App skill directories must not be symlinks: {entry}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        manifest = _validated_data_path(
            entry / "SKILL.md", root=skills_root, expected="file"
        )
        if manifest.exists():
            manifests.add(manifest)
    return manifests


def _data_read_paths(hermes_home: Path) -> list[Path]:
    """Return the complete data allowlist consumed by Mini App projections."""
    home = _lexical_absolute(hermes_home)
    default_root = _default_root(home)
    _validated_data_path(
        default_root, root=default_root, expected="directory", optional=False
    )
    _validated_data_path(home, root=default_root, expected="directory", optional=False)
    state_db = home / "state.db"
    kanban_db = _selected_kanban_db(default_root)
    paths = {
        _validated_data_path(home / "memories" / "USER.md", root=home, expected="file"),
        _validated_data_path(
            home / "memories" / "MEMORY.md", root=home, expected="file"
        ),
        _validated_data_path(home / "gateway_state.json", root=home, expected="file"),
        _validated_data_path(state_db, root=home, expected="file"),
        _validated_data_path(
            state_db.with_name(f"{state_db.name}-wal"), root=home, expected="file"
        ),
        _validated_data_path(
            state_db.with_name(f"{state_db.name}-shm"), root=home, expected="file"
        ),
        _validated_data_path(
            default_root / "kanban" / "current",
            root=default_root,
            expected="file",
        ),
        kanban_db,
        _validated_data_path(
            kanban_db.with_name(f"{kanban_db.name}-wal"),
            root=default_root,
            expected="file",
        ),
        _validated_data_path(
            kanban_db.with_name(f"{kanban_db.name}-shm"),
            root=default_root,
            expected="file",
        ),
    }
    paths.update(_skill_manifest_paths(home))
    return sorted(paths, key=str)


def _custom_data_mask(hermes_home: Path) -> Path | None:
    """Mask custom data roots outside ProtectHome's standard namespaces."""
    root = _default_root(hermes_home)
    if root == Path("/"):
        raise MiniAppServiceError("HERMES_HOME=/ cannot be sandboxed safely.")
    if any(_is_within(root, protected) for protected in _home_namespace_roots()):
        return None
    return root


def _systemd_unit(hermes_home: Path) -> str:
    p = paths_for(hermes_home)
    cmd = " ".join(_systemd_quote(part) for part in service_command(hermes_home))
    read_only = " ".join(
        _systemd_quote(str(path)) for path in _runtime_read_paths(hermes_home)
    )
    optional_data = " ".join(
        f"-{_systemd_quote(str(path))}" for path in _data_read_paths(hermes_home)
    )
    custom_mask = _custom_data_mask(hermes_home)
    temporary_fs = (
        f"TemporaryFileSystem={_systemd_quote(f'{custom_mask}:ro')}\n"
        if custom_mask is not None
        else ""
    )
    return (
        "[Unit]\n"
        "Description=Hermes Telegram Mini App\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={cmd}\n"
        "WorkingDirectory=/\n"
        "Restart=on-failure\n"
        "RestartSec=3\n"
        "UMask=0077\n"
        "NoNewPrivileges=true\n"
        "PrivateTmp=true\n"
        "PrivateDevices=true\n"
        "ProtectSystem=strict\n"
        "ProtectHome=tmpfs\n"
        "ProtectControlGroups=true\n"
        "ProtectKernelModules=true\n"
        "ProtectKernelTunables=true\n"
        "ProtectKernelLogs=true\n"
        "ProtectClock=true\n"
        "ProtectHostname=true\n"
        "RestrictSUIDSGID=true\n"
        "RestrictRealtime=true\n"
        "RestrictNamespaces=true\n"
        "LockPersonality=true\n"
        "RemoveIPC=true\n"
        "ProtectProc=invisible\n"
        "ProcSubset=pid\n"
        "SystemCallArchitectures=native\n"
        "SystemCallFilter=@system-service\n"
        "SystemCallErrorNumber=EPERM\n"
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6\n"
        "IPAddressDeny=any\n"
        "IPAddressAllow=localhost\n"
        f"{temporary_fs}"
        f"BindReadOnlyPaths={read_only} {optional_data}\n"
        f"BindPaths={_systemd_quote(str(p.root.resolve()))}\n"
        f"ReadWritePaths={_systemd_quote(str(p.root.resolve()))}\n"
        f"StandardOutput=append:{p.stdout_log.resolve()}\n"
        f"StandardError=append:{p.stderr_log.resolve()}\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _systemd_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _platform() -> str:
    if sys.platform.startswith("linux"):
        return "systemd"
    return "unsupported"


def require_install_support() -> None:
    """Fail before setup mutates state when native supervision is unavailable."""
    platform = _platform()
    if platform == "systemd" and shutil.which("systemctl"):
        return
    raise MiniAppServiceError(
        "Native Mini App installation currently requires Linux with systemd user services. "
        "On macOS and other POSIX platforms, `hermes gateway mini-app serve` runs unsandboxed in the foreground; "
        "do not expose it directly to the network."
    )


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, check=check, capture_output=True, text=True, timeout=15
        )
    except FileNotFoundError as exc:
        raise MiniAppServiceError(
            f"Required service manager command is unavailable: {argv[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise MiniAppServiceError(
            detail or f"Command failed: {' '.join(argv)}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MiniAppServiceError(f"Service manager timed out: {argv[0]}") from exc


def install(hermes_home: Path) -> None:
    """Install, but do not start, the native per-user service."""
    require_install_support()
    p = paths_for(hermes_home)
    if not p.env.exists():
        raise MiniAppServiceError(
            "Mini App is not configured; run `hermes gateway mini-app setup` first."
        )
    for log in (p.stdout_log, p.stderr_log):
        ensure_private_file(log)

    platform = _platform()
    if platform == "systemd":
        target = systemd_unit_path(hermes_home)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_systemd_unit(hermes_home), encoding="utf-8")
        target.chmod(0o600)
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", service_name(hermes_home)])
        return
    raise AssertionError(f"Unhandled service platform: {platform}")


def start(hermes_home: Path) -> None:
    platform = _platform()
    if platform == "systemd":
        if not systemd_unit_path(hermes_home).exists():
            raise MiniAppServiceError("Mini App service is not installed.")
        _run(["systemctl", "--user", "start", service_name(hermes_home)])
        return
    raise MiniAppServiceError(
        "Native Mini App services are unsupported on this platform; use `serve`."
    )


def stop(hermes_home: Path) -> None:
    platform = _platform()
    if platform == "systemd":
        _run(["systemctl", "--user", "stop", service_name(hermes_home)])
        return
    raise MiniAppServiceError(
        "Native Mini App services are unsupported on this platform; stop the foreground process."
    )


def restart(hermes_home: Path) -> None:
    platform = _platform()
    if platform == "systemd":
        _run(["systemctl", "--user", "restart", service_name(hermes_home)])
        return
    raise MiniAppServiceError(
        "Native Mini App services are unsupported on this platform."
    )


def status(hermes_home: Path) -> tuple[bool, str]:
    platform = _platform()
    if platform == "systemd":
        if not systemd_unit_path(hermes_home).exists():
            return False, "not installed"
        result = _run(
            ["systemctl", "--user", "is-active", service_name(hermes_home)],
            check=False,
        )
        running = result.returncode == 0 and result.stdout.strip() == "active"
        return running, result.stdout.strip() or result.stderr.strip() or "inactive"
    return False, "native service unsupported; foreground serve is unsandboxed"


def uninstall(hermes_home: Path) -> None:
    platform = _platform()
    if platform == "systemd":
        name = service_name(hermes_home)
        disable = _run(["systemctl", "--user", "disable", "--now", name], check=False)
        active = _run(["systemctl", "--user", "is-active", name], check=False)
        state = active.stdout.strip().lower()
        safely_stopped = active.returncode != 0 and state in {
            "inactive",
            "failed",
            "unknown",
            "not-found",
        }
        if not safely_stopped:
            detail = (active.stderr or disable.stderr or disable.stdout or "").strip()
            raise MiniAppServiceError(
                detail
                or "Telegram Mini App service did not stop; its unit and recovery files were preserved."
            )
        systemd_unit_path(hermes_home).unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"])
        return
    # Foreground-only platforms have no native service definition to remove.
    return
