"""Build an isolated Hindsight API runtime and start its daemon in that interpreter."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EMBED_VERSION = "0.9.2"
_API_SLIM_VERSION = "0.9.2"
_REQUIREMENTS = (f"hindsight-embed=={_EMBED_VERSION}", f"hindsight-api-slim[all]=={_API_SLIM_VERSION}")

# First successful daemon start downloads/loads ML models — minutes, not seconds.
_DEFAULT_DAEMON_START_TIMEOUT = 900.0
_INSTALL_TIMEOUT = 3600
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


def sideenv_python(root: Path | None = None) -> Path | None:
    """Read PM's selected side interpreter without creating an environment."""
    import pm

    return pm.environment_python("hindsight", root=root if root is not None else sideenv_root())


def ensure_sideenv() -> Path:
    """Provision the isolated embedded runtime; PM owns generation publication."""
    import pm

    return pm.ensure_environment(
        "hindsight", _REQUIREMENTS, root=sideenv_root(), explicit=True, timeout=_INSTALL_TIMEOUT,
    )


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
