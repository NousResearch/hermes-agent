"""Redacted validation-failure telemetry for the kanban / agent surfaces.

Single, lean sink for recording when a tool call fails validation or an
auxiliary JSON parse fails. Everything written here is REDACTED by
construction: only the five fields below are ever stored. No prompts,
secrets, raw user content, reference paths, or raw model output are ever
written (this satisfies the F001 handoff hard boundary).

Output: one JSON object per line in
``$HERMES_HOME/governance/telemetry/invalid-tool-calls.jsonl``

Write failures are non-fatal: the caller's behaviour must not depend on
telemetry, so any exception is swallowed (debug-logged only).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_TELEMETRY_DIRNAME = "invalid-tool-calls.jsonl"


def _telemetry_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "governance", "telemetry", _TELEMETRY_DIRNAME)


def record_validation_failure(
    *,
    profile: str,
    provider: str,
    model: str,
    task_class: str,
    schema_or_version_mismatch: str,
) -> Optional[str]:
    """Append one redacted validation-failure event.

    Args (all redacted, no raw content):
        profile: profile/aux name (e.g. ``octacon``, ``kanban_decomposer``)
        provider: resolved provider slug (e.g. ``openrouter``, ``nous``)
        model: resolved model id
        task_class: one of ``invalid_tool_call``, ``aux_malformed_json``,
            ``aux_empty_response``, ``aux_api_error``
        schema_or_version_mismatch: short redacted reason class, never raw text

    Returns the written line id (``profile|model|ts``) or None on failure.
    Never raises.
    """
    # Structural redaction: build a brand-new dict with ONLY these keys.
    event = {
        "profile": (profile or "unknown")[:64],
        "provider": (provider or "unknown")[:64],
        "model": (model or "unknown")[:128],
        "task_class": (task_class or "unknown")[:48],
        "schema_or_version_mismatch": (schema_or_version_mismatch or "unknown")[:128],
        "ts": int(time.time()),
    }
    line_id = f"{event['profile']}|{event['model']}|{event['ts']}"
    try:
        path = _telemetry_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        return line_id
    except Exception as exc:  # pragma: no cover — telemetry must never break callers
        logger.debug("telemetry write failed (non-fatal): %s", exc)
        return None
