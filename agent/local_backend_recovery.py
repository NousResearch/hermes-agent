"""Opt-in recovery hook for wedged local inference backends."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from typing import Any

from agent.model_metadata import is_local_endpoint

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def maybe_recover_local_backend(
    agent: Any,
    *,
    reason: str,
    model: Any,
    base_url: Any,
    context_tokens: int | None = None,
    elapsed: float | None = None,
    threshold: float | None = None,
) -> bool:
    """Run a configured recovery command after a local backend timeout.

    Hermes cannot safely know how every self-hosted backend should be repaired:
    llama.cpp may support slot cancellation, llama-swap may need its child
    process killed, and production deployments may want a service restart.  The
    durable contract is therefore an opt-in command hook.  The command receives
    sanitized metadata through environment variables and is rate-limited per
    agent process so repeated retries do not flap a shared GPU service.
    """
    base_url_s = str(base_url or "").strip()
    if not is_local_endpoint(base_url_s):
        return False

    command = str(os.getenv("HERMES_LOCAL_BACKEND_RECOVERY_COMMAND") or "").strip()
    if not command:
        return False

    cooldown = _env_float("HERMES_LOCAL_BACKEND_RECOVERY_COOLDOWN", 60.0)
    now = time.monotonic()
    last_at = float(getattr(agent, "_local_backend_recovery_last_at", 0.0) or 0.0)
    if cooldown > 0 and last_at and now - last_at < cooldown:
        logger.info(
            "Skipping local backend recovery for %s: cooldown %.0fs still active",
            reason,
            cooldown,
        )
        return False
    try:
        setattr(agent, "_local_backend_recovery_last_at", now)
    except Exception:
        pass

    timeout = _env_float("HERMES_LOCAL_BACKEND_RECOVERY_TIMEOUT", 20.0)
    env = os.environ.copy()
    env.update(
        {
            "HERMES_RECOVERY_REASON": str(reason),
            "HERMES_RECOVERY_MODEL": str(model or ""),
            "HERMES_RECOVERY_PROVIDER": str(getattr(agent, "provider", "") or ""),
            "HERMES_RECOVERY_BASE_URL": base_url_s,
            "HERMES_RECOVERY_CONTEXT_TOKENS": str(int(context_tokens or 0)),
            "HERMES_RECOVERY_ELAPSED": str(float(elapsed or 0.0)),
            "HERMES_RECOVERY_THRESHOLD": str(float(threshold or 0.0)),
            "HERMES_RECOVERY_SESSION_ID": str(getattr(agent, "session_id", "") or ""),
        }
    )

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        logger.warning("Invalid local backend recovery command: %s", exc)
        return False
    if not argv:
        return False

    logger.warning(
        "Running local backend recovery command after %s: model=%s base_url=%s",
        reason,
        model or "unknown",
        base_url_s,
    )
    try:
        completed = subprocess.run(
            argv,
            env=env,
            timeout=timeout if timeout > 0 else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Local backend recovery command timed out after %.0fs",
            timeout,
        )
        return False
    except Exception as exc:
        logger.warning("Local backend recovery command failed to start: %s", exc)
        return False

    if completed.returncode != 0:
        logger.warning(
            "Local backend recovery command exited rc=%s",
            completed.returncode,
        )
        return False

    logger.info("Local backend recovery command completed successfully")
    return True
