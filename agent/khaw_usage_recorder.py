"""Best-effort KHAW OpenRouter generation ledger integration.

Hermes sees OpenRouter generation IDs on successful model responses. KHAW owns
the per-FDE ledger and later reconciles those IDs via OpenRouter's generation
endpoint. This module records only generation metadata; it never handles,
prints, or forwards API keys or response content.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_GENERATION_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{6,200}")
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def _env_enabled(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def _is_openrouter(provider: Any, base_url: Any) -> bool:
    if str(provider or "").strip().lower() == "openrouter":
        return True
    parsed = urlparse(str(base_url or "").strip())
    host = (parsed.hostname or "").lower()
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _header_get(headers: Any, key: str) -> Any:
    if not headers:
        return None
    try:
        return headers.get(key)
    except Exception:
        pass
    lowered = key.lower()
    try:
        for existing, value in headers.items():
            if str(existing).lower() == lowered:
                return value
    except Exception:
        return None
    return None


def _extract_generation_id(response: Any) -> str | None:
    candidates = [
        getattr(response, "id", None),
        _header_get(getattr(response, "headers", None), "x-openrouter-generation-id"),
        _header_get(getattr(response, "headers", None), "X-OpenRouter-Generation-Id"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        generation_id = candidate.strip()
        if not generation_id:
            continue
        # Ignore Hermes-local synthetic IDs. They are not OpenRouter generation
        # IDs and would poison KHAW's ledger.
        if generation_id.startswith(("stream-", "chatcmpl-local-")):
            continue
        if _GENERATION_ID_RE.fullmatch(generation_id):
            return generation_id
    return None


def _resolve_identity() -> str | None:
    for name in ("KHAW_FDE_IDENTITY", "HERMES_FDE_IDENTITY", "HERMES_KHAW_IDENTITY"):
        value = os.getenv(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def maybe_record_openrouter_generation(
    *,
    provider: Any,
    base_url: Any,
    response: Any,
    identity: str | None = None,
    source: str = "hermes",
) -> bool:
    """Record an OpenRouter generation id through ``khaw`` if all inputs exist.

    This is intentionally best-effort: non-OpenRouter providers, missing
    generation IDs, absent ``khaw``, command failures, and timeouts all return
    ``False`` and never interrupt the model turn. Set
    ``HERMES_KHAW_USAGE_RECORDING=0`` to disable. If no identity env is set,
    KHAW falls back to its local OS-user default.
    """
    if not _env_enabled("HERMES_KHAW_USAGE_RECORDING", default=True):
        return False
    if not _is_openrouter(provider, base_url):
        return False

    resolved_identity = (identity or _resolve_identity() or "").strip()

    generation_id = _extract_generation_id(response)
    if not generation_id:
        return False

    khaw_name = os.getenv("HERMES_KHAW_BIN", "khaw").strip() or "khaw"
    khaw_bin = shutil.which(khaw_name)
    if not khaw_bin:
        logger.debug("KHAW usage recording skipped: %s executable not found", khaw_name)
        return False
    cmd = [
        khaw_bin,
        "fde",
        "usage",
        "record",
        "--generation-id",
        generation_id,
        "--source",
        source,
        "--json",
    ]
    if resolved_identity:
        cmd.extend(["--identity", resolved_identity])

    model = getattr(response, "model", None)
    if isinstance(model, str) and model.strip():
        cmd.extend(["--model", model.strip()])

    session_id = getattr(response, "session_id", None)
    if isinstance(session_id, str) and session_id.strip():
        cmd.extend(["--session-id", session_id.strip()])

    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell, no secrets
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(os.getenv("HERMES_KHAW_USAGE_TIMEOUT", "3")),
        )
    except Exception as exc:
        logger.debug("KHAW usage recording failed: %s", exc)
        return False

    if completed.returncode != 0:
        logger.debug(
            "KHAW usage recording exited %s: %s",
            completed.returncode,
            (completed.stderr or completed.stdout or "").strip()[:300],
        )
        return False

    logger.debug("KHAW recorded OpenRouter generation %s", generation_id)
    return True


def record_openrouter_generation(agent: Any, response: Any) -> bool:
    """Adapter for the conversation loop's successful-response hook."""
    response_for_recording = SimpleNamespace(
        id=getattr(response, "id", None),
        headers=getattr(response, "headers", None),
        model=getattr(response, "model", None) or getattr(agent, "model", None),
        session_id=getattr(agent, "session_id", None),
    )
    return maybe_record_openrouter_generation(
        provider=getattr(agent, "provider", None),
        base_url=getattr(agent, "base_url", None),
        response=response_for_recording,
    )
