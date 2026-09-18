"""Helper .app so a launchd gateway can obtain macOS Local Network permission.

Bare venv Python launched by launchd has no application ID, so ``nehelper``
denies LAN sockets as ``EHOSTUNREACH`` / ``No route to host`` (#71206).
Terminal children inherit Terminal.app's grant; this bundle gives the launchd
job its own identity and ``NSLocalNetworkUsageDescription`` so macOS can
prompt. Best-effort: install failure leaves ProgramArguments on venv Python.
"""

from __future__ import annotations

import logging
import os
import platform
import plistlib
import shutil
import subprocess
from pathlib import Path

from hermes_constants import get_hermes_home
from utils import atomic_write_text

logger = logging.getLogger(__name__)

BUNDLE_ID = "com.nousresearch.hermes.gateway"
BUNDLE_NAME = "Hermes Gateway"
EXECUTABLE_NAME = "HermesGateway"
LOCAL_NETWORK_USAGE = (
    "Hermes connects to devices on your local network when a plugin or feature "
    "you enable requests it."
)
_MARKER_NAME = ".hermes-gateway-app-source"
_APP_DIRNAME = "HermesGateway.app"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def gateway_app_path(hermes_home: Path | None = None) -> Path:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return home / "macos" / _APP_DIRNAME


def gateway_app_executable(app: Path | None = None) -> Path:
    bundle = app if app is not None else gateway_app_path()
    return bundle / "Contents" / "MacOS" / EXECUTABLE_NAME


def wrap_launchd_python(
    command: list[str],
    python_path: str,
    helper_exe: Path | str | None = None,
) -> list[str]:
    """Replace venv ``python_path`` entries with the helper executable when it is ready."""
    if helper_exe is not None:
        helper = str(helper_exe)
        return [helper if part == python_path else part for part in command]
    if not is_macos():
        return list(command)
    exe = _ready_executable()
    if exe is None or not exe.is_file():
        return list(command)
    return [str(exe) if part == python_path else part for part in command]


def launchd_python_command(command: list[str], python_path: str) -> list[str]:
    """Ensure the helper exists (best-effort) and wrap launchd ProgramArguments."""
    try:
        ensure_gateway_app()
    except Exception:
        logger.debug("macOS gateway helper ensure skipped", exc_info=True)
    return wrap_launchd_python(command, python_path)


def gateway_app_state(hermes_home: Path | None = None) -> tuple[str, str]:
    """``(status, detail)`` for ``hermes doctor``: skip / active / missing / stale."""
    if not is_macos():
        return "skip", "not macOS"
    found = _venv_python_source()
    if found is None:
        return "skip", "no venv interpreter"
    _venv, python = found
    app = gateway_app_path(hermes_home)
    exe = gateway_app_executable(app)
    if not exe.is_file():
        return "missing", str(app)
    if _marker_matches(app, python):
        return "active", str(exe)
    return "stale", str(exe)


def ensure_gateway_app(hermes_home: Path | None = None) -> Path | None:
    """Install or refresh the helper .app. Never raises. None if not applicable or install failed."""
    if not is_macos():
        return None
    found = _venv_python_source()
    if found is None:
        return None
    venv, python = found
    app = gateway_app_path(hermes_home)
    exe = gateway_app_executable(app)
    if exe.is_file() and _marker_matches(app, python) and _passes_boot_gate(exe, venv):
        return exe
    try:
        _install_app(app, python, venv)
    except Exception as exc:
        logger.warning("macOS gateway helper install failed: %s", exc)
        _discard_app(app)
        return None
    if not _passes_boot_gate(exe, venv):
        logger.warning("macOS gateway helper boot-gate refused install at %s", exe)
        _discard_app(app)
        return None
    _sign_app(app)
    return exe if exe.is_file() else None


def remove_gateway_app(hermes_home: Path | None = None) -> None:
    _discard_app(gateway_app_path(hermes_home))


def _venv_python_source() -> tuple[Path, Path] | None:
    """``(venv_dir, resolved interpreter)`` when launchd should wrap, else None."""
    try:
        from hermes_cli.gateway import get_python_path, _detect_venv_dir
    except Exception:
        return None
    venv = _detect_venv_dir()
    if venv is None or not (venv / "pyvenv.cfg").is_file():
        return None
    python = Path(get_python_path())
    try:
        if python.parent.resolve() != (venv / "bin").resolve():
            return None
    except OSError:
        return None
    if not python.is_file() and not python.is_symlink():
        return None
    try:
        resolved = python.resolve(strict=False)
    except OSError:
        return None
    return (venv, resolved) if resolved.is_file() else None


def _marker_path(app: Path) -> Path:
    return app / "Contents" / _MARKER_NAME


def _marker_matches(app: Path, source: Path) -> bool:
    marker = _marker_path(app)
    if not marker.is_file():
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == str(source)
    except OSError:
        return False


def _write_marker(app: Path, source: Path) -> None:
    atomic_write_text(_marker_path(app), str(source) + "\n")


def _info_plist_payload() -> dict:
    return {
        "CFBundleDisplayName": BUNDLE_NAME,
        "CFBundleExecutable": EXECUTABLE_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": BUNDLE_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSUIElement": True,
        "NSLocalNetworkUsageDescription": LOCAL_NETWORK_USAGE,
    }


def _install_app(app: Path, source: Path, venv: Path) -> None:
    staging = app.with_name(f".{_APP_DIRNAME}.staging-{os.getpid()}")
    _discard_app(staging)
    try:
        macos = staging / "Contents" / "MacOS"
        macos.mkdir(parents=True)
        exe = macos / EXECUTABLE_NAME
        shutil.copy2(source, exe)
        os.chmod(exe, source.stat().st_mode | 0o111)
        _link_venv_lib(staging, venv, source)
        cfg = venv / "pyvenv.cfg"
        if cfg.is_file():
            shutil.copy2(cfg, staging / "Contents" / "pyvenv.cfg")
        with (staging / "Contents" / "Info.plist").open("wb") as fh:
            plistlib.dump(_info_plist_payload(), fh)
        _write_marker(staging, source)
        _discard_app(app)
        app.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, app)
    except Exception:
        _discard_app(staging)
        raise


def _link_venv_lib(staging: Path, venv: Path, source: Path) -> None:
    src_lib = venv / "lib"
    if not src_lib.is_dir():
        try:
            from hermes_cli.macos_tcc_anchor import _provision_libpython
            _provision_libpython(venv, source, refresh=False)
        except Exception:
            logger.debug("libpython provision for gateway helper skipped", exc_info=True)
    if not src_lib.is_dir():
        return
    dest = staging / "Contents" / "lib"
    try:
        dest.symlink_to(src_lib.resolve())
    except OSError:
        logger.debug("gateway helper lib symlink skipped", exc_info=True)


def _passes_boot_gate(exe: Path, venv: Path) -> bool:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__")
    }
    env["VIRTUAL_ENV"] = str(venv)
    try:
        proc = subprocess.run(
            [str(exe), "-c", "import encodings, sys; print(sys.prefix)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    printed = (proc.stdout or "").strip()
    if not printed:
        return False
    printed_path = printed.splitlines()[-1]
    venv_s = str(venv.resolve())
    return venv_s in printed_path or str(exe.parent.parent.resolve()) in printed_path


def _sign_app(app: Path) -> None:
    codesign = shutil.which("codesign")
    if not codesign:
        return
    requirement = f'=designated => identifier "{BUNDLE_ID}"'
    cmd = [
        codesign, "--force", "--deep", "--sign", "-", "--timestamp=none",
        "--identifier", BUNDLE_ID, "--requirements", requirement, str(app),
    ]
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        logger.warning(
            "could not sign gateway helper %s: %s",
            app,
            (result.stderr or result.stdout or "codesign failed").strip(),
        )


def _ready_executable(hermes_home: Path | None = None) -> Path | None:
    app = gateway_app_path(hermes_home)
    exe = gateway_app_executable(app)
    if exe.is_file() and _marker_path(app).is_file():
        return exe
    return None


def _discard_app(app: Path) -> None:
    shutil.rmtree(app, ignore_errors=True)
