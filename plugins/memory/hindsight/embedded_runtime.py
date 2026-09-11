"""Build an isolated Hindsight API runtime and start its daemon in that interpreter."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EMBED_VERSION = "0.9.2"
_API_SLIM_VERSION = "0.9.2"
# Marker file written inside each generation recording what it was built from;
# a generation whose marker no longer matches is replaced (new generation) on
# the next ensure — never mutated in place.
_EXPECTED = {"hindsight-embed": _EMBED_VERSION, "hindsight-api-slim": _API_SLIM_VERSION}
_STATE_FILE = ".hermes-sideenv.json"
_ACTIVE_RECORD = "active.json"

# First successful daemon start downloads/loads ML models — minutes, not seconds.
_DEFAULT_DAEMON_START_TIMEOUT = 900.0
_UV_LOCK_TIMEOUT = 900.0
_UV_SYNC_TIMEOUT = 3600.0
_PROBE_TIMEOUT = 300.0

_PORT_HEALTH_GRACE_ENV = "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"
_API_VERSION_ENV = "HINDSIGHT_EMBED_API_VERSION"

# Printed by the bridge as its LAST stdout line; the manager's Rich output and
# the daemon's own logs share the stream, so the marker makes parsing exact.
_BRIDGE_MARKER = "__HERMES_HINDSIGHT_BRIDGE__"

_PROBE_CODE = "import hindsight_embed.daemon_embed_manager, hindsight_api, sentence_transformers"

_DAEMON_BRIDGE_CODE = """\
import json, sys
from hindsight_embed import daemon_client

profile = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "restart":
    daemon_client.stop_daemon(profile)
result = {"ok": False, "url": None, "error": None}
try:
    # {} config: the manager merges the profile's own .env (materialized by the
    # plugin) over it, so the LLM keys live only in the 0600 profile file.
    result["ok"] = bool(daemon_client.ensure_daemon_running({}, profile))
    if result["ok"]:
        result["url"] = daemon_client.get_daemon_url(profile)
    else:
        result["error"] = "daemon did not start (see the hindsight profile log)"
except Exception as exc:  # bridge must always report, never crash the parse
    result["error"] = f"{type(exc).__name__}: {exc}"
print("@MARKER@" + json.dumps(result))
""".replace("@MARKER@", _BRIDGE_MARKER)


def sideenv_root() -> Path:
    """Private side env for the embedded runtime (never the boot-selected venv)."""
    return get_hermes_home() / "profiles" / "Hindsight" / "env"


def _active_generation(root: Path) -> Path | None:
    """The published generation directory, or None when nothing is active."""
    try:
        record = json.loads((root / _ACTIVE_RECORD).read_text(encoding="utf-8-sig"))
        gen = root / str(record.get("generation", ""))
    except Exception:
        return None
    return gen if gen.resolve().is_relative_to(root.resolve()) and gen != root and gen.is_dir() else None


def _publish_generation(root: Path, generation: Path) -> None:
    """Point the selection record at *generation* atomically (file replace —
    never a directory rename: venv launchers embed absolute paths)."""
    from utils import atomic_json_write

    atomic_json_write(root / _ACTIVE_RECORD, {"generation": generation.name, "pins": _EXPECTED})


def sideenv_python(root: Path | None = None) -> Path | None:
    """The active generation's interpreter, or None when the side env is not
    installed. Resolved through the selection record — the generation dir is
    created at its final path and never renamed."""
    root = Path(root) if root is not None else sideenv_root()
    gen = _active_generation(root)
    if gen is None:
        return None
    exe = "python.exe" if os.name == "nt" else "python"
    for scripts in ("Scripts", "bin"):
        candidate = gen / ".venv" / scripts / exe
        if candidate.is_file():
            return candidate
    return None


def sideenv_pyproject() -> str:
    """Exact pinned requirements of the isolated runtime (public PyPI versions,
    frozen into uv.lock at install time; never resolved against the main env)."""
    return (
        '[project]\n'
        'name = "hermes-hindsight-embedded"\n'
        'version = "0.9.2"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = [\n'
        f'    "hindsight-embed=={_EMBED_VERSION}",\n'
        f'    "hindsight-api-slim[all]=={_API_SLIM_VERSION}",\n'
        ']\n'
    )


def _generation_current(generation: Path) -> bool:
    """Installed and matching the pinned versions (marker written at build)."""
    if generation is None:
        return False
    try:
        state = json.loads((generation / _STATE_FILE).read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return state.get("pins") == _EXPECTED


def _uv_bridge(venv: Path) -> tuple[str, dict[str, str]]:
    """The sanctioned pm bridge: pinned uv binary + sanitized env for *venv*."""
    from pm.client import uv as pm_uv

    uv_bin, env = pm_uv(venv=venv)
    if not uv_bin:
        raise RuntimeError(
            "pm could not realize the pinned uv binary; run 'hermes pm install' "
            "before installing the isolated Hindsight runtime"
        )
    return str(uv_bin), env


def _run_uv(uv_bin: str, env: dict[str, str], args: list[str], timeout: float) -> None:
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [uv_bin, *args], env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"uv {' '.join(args)} failed (exit {result.returncode}): {detail}")


def ensure_sideenv() -> Path:
    """Install/update the isolated runtime; returns the active generation dir.

    Each install is an immutable generation built at its final path and
    published via the ``active.json`` selection record. A failed build removes
    only its own unpublished generation; the previous generation (and any
    daemon running from it) stays untouched."""
    root = sideenv_root()
    active = _active_generation(root)
    if _generation_current(active) and sideenv_python(root) is not None:
        return active

    generation = root / f"gen-{uuid.uuid4().hex}"
    generation.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Hindsight embedded runtime: building generation %s (pins=%s)",
                generation, _EXPECTED)
    try:
        generation.mkdir(parents=True)
        (generation / "pyproject.toml").write_text(sideenv_pyproject(), encoding="utf-8")
        uv_bin, env = _uv_bridge(generation / ".venv")
        _run_uv(uv_bin, env, ["lock", "--project", str(generation)], _UV_LOCK_TIMEOUT)
        _run_uv(uv_bin, env, ["sync", "--project", str(generation), "--frozen"], _UV_SYNC_TIMEOUT)
        (generation / _STATE_FILE).write_text(
            json.dumps({"pins": _EXPECTED}, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        # The failed generation was never published; nothing can be running
        # from it, so removing it leaks nothing. Previous generations stay.
        shutil.rmtree(generation, ignore_errors=True)
        raise
    _publish_generation(root, generation)
    logger.info("Hindsight embedded runtime: generation %s published (side env %s)",
                generation.name, root)
    return generation


def _probe_interpreter(python: Path, timeout: float = _PROBE_TIMEOUT) -> tuple[bool, str | None]:
    """Import the embedded stack IN THE SIDE ENV (its own interpreter). Covers the
    old-CPU NumPy failure class and a broken embedding stack the same way the
    legacy in-process probe did — but inside the isolated environment."""
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(python), "-c", _PROBE_CODE], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            env=_daemon_subprocess_env({}),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"side runtime probe failed: {exc}"
    if result.returncode == 0:
        return True, None
    reason = (result.stderr or result.stdout or "").strip()
    return False, reason[-500:] if reason else f"probe exited {result.returncode}"


def check_local_runtime() -> tuple[bool, str | None]:
    """(available, reason) for local_embedded — probes the side env, never the
    boot-selected main environment."""
    root = sideenv_root()
    python = sideenv_python(root)
    if python is None:
        reason = (f"the isolated Hindsight runtime is not installed at {root}; "
                  "run 'hermes memory setup' and choose Local Embedded")
        logger.debug("Hindsight local runtime unavailable: %s", reason)
        return False, reason
    available, reason = _probe_interpreter(python)
    if available:
        logger.debug("Hindsight side runtime probe OK (%s)", python)
    else:
        logger.debug("Hindsight side runtime probe failed: %s", reason)
    return available, reason


def _local_runtime_hint(reason: str | None) -> str:
    """Install/reinstall guidance for an unavailable side runtime. The main
    Hermes environment is never touched, so the fix is always the same one."""
    text = f"{reason or ''}".strip()
    return (
        " The local_embedded runtime lives in an isolated environment"
        f" ({sideenv_root()}; hindsight-embed=={_EMBED_VERSION} +"
        f" hindsight-api-slim[all]=={_API_SLIM_VERSION}), separate from Hermes"
        " itself. Reinstall it with 'hermes memory setup' (Local Embedded), or"
        " switch to cloud / local_external mode."
        + (f" Last probe failure: {text}" if text else "")
    )


def _daemon_subprocess_env(config: dict[str, Any]) -> dict[str, str]:
    """Env for the side-env bridge/manager process. daemon_embed_manager reads
    the health-grace window AT IMPORT TIME of that process, so it rides here —
    not in Hermes' own environment. Main-env interpreter markers are stripped so
    the side python resolves only its own environment."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(key, None)
    raw = config.get("port_health_grace_timeout")
    if raw not in (None, ""):
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid Hindsight port_health_grace_timeout %r; ignoring.", raw)
        else:
            if seconds >= 0:
                env.setdefault(_PORT_HEALTH_GRACE_ENV, repr(seconds))
            else:
                logger.warning("Negative Hindsight port_health_grace_timeout %r; ignoring.", raw)
    # Freeze the daemon/API component version to the pinned pair (the manager
    # reads this as the profile .env override's fallback).
    env[_API_VERSION_ENV] = _API_SLIM_VERSION
    return env


def _parse_bridge_output(stdout: str) -> dict[str, Any]:
    """The bridge's marked JSON line; Rich/log noise is ignored."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_BRIDGE_MARKER):
            try:
                return json.loads(line[len(_BRIDGE_MARKER):])
            except ValueError:
                continue
    return {}


def ensure_daemon_and_url(config: dict[str, Any], *, restart: bool = False) -> str:
    """Start (or reuse) the side-env daemon and return its URL.

    The URL is resolved by the side env's own ProfileManager/get_url — profile
    ``HINDSIGHT_API_PORT`` override, then the manager's metadata allocation —
    and never hardcoded here. The daemon child runs the side env's hindsight-api
    because DaemonEmbedManager resolves the API command from the interpreter it
    runs under (the side python)."""
    python = sideenv_python()
    if python is None:
        raise RuntimeError(_local_runtime_hint("the isolated runtime is not installed"))
    profile = str(config.get("profile", "hermes") or "hermes")
    env = _daemon_subprocess_env(config)
    logger.info("Hindsight embedded: starting side-env daemon manager (profile=%s, side_python=%s)",
                profile, python)
    timeout = float(config.get("daemon_start_timeout") or _DEFAULT_DAEMON_START_TIMEOUT)
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv (python -c bridge), no shell
            [str(python), "-c", _DAEMON_BRIDGE_CODE, profile, "restart" if restart else "start"],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("Hindsight embedded daemon start timed out after %.0fs (profile=%s)", timeout, profile)
        raise RuntimeError(
            f"Hindsight daemon start timed out after {timeout:.0f}s; the first start "
            "can take several minutes — check the hindsight profile log and retry."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"could not run the side-env runtime {python}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        logger.warning("Hindsight embedded daemon bridge failed (exit %s): %s", result.returncode, detail)
        raise RuntimeError(f"Hindsight daemon startup failed (exit {result.returncode}): {detail}")
    payload = _parse_bridge_output(result.stdout)
    url = payload.get("url")
    if payload.get("ok") and url:
        logger.info("Hindsight embedded: daemon ready at %s (profile=%s)", url, profile)
        return str(url)
    error = payload.get("error") or f"no bridge result in side-env output: {(result.stdout or '')[-500:]}"
    logger.warning("Hindsight embedded daemon did not start: %s", error)
    raise RuntimeError(f"Hindsight daemon startup failed: {error}")
