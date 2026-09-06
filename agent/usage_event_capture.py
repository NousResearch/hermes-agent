"""Execution-local Codex observations, persisted only at normalized post_api_request.

No cumulative counters, parent summaries, payloads, credentials, or network calls.
"""
from dataclasses import dataclass
import hashlib
import logging
import time
import uuid
from typing import Any

from hermes_constants import hermes_home_key
from hermes_state_usage_events import UsageEvent, note_recording_failure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Observation:
    response_identity: int
    event: UsageEvent
    db: Any


def _get(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _codex_tokens(usage) -> dict:
    # Unlike CanonicalUsage.input_tokens these are cache-INCLUSIVE inputs. Keep
    # absent details NULL, not CanonicalUsage's default zero. This is the Codex
    # Responses contract, not a heuristic for arbitrary providers/API modes.
    details = _get(usage, "input_tokens_details")
    cache_write = _get(details, "cache_write_tokens")
    if cache_write is None:
        cache_write = _get(details, "cache_creation_tokens")
    raw = {
        "input_tokens": _get(usage, "input_tokens"),
        "output_tokens": _get(usage, "output_tokens"),
        "cache_read_tokens": _get(details, "cached_tokens"),
        "cache_write_tokens": cache_write,
        "reasoning_tokens": _get(_get(usage, "output_tokens_details"), "reasoning_tokens"),
    }
    values = {key: value if type(value) is int and 0 <= value <= 2**63 - 1 else None
              for key, value in raw.items()}
    invalid = any(raw[key] is not None and values[key] is None for key in raw)
    state = "reported"
    if invalid:
        state = "invalid"
    elif all(value is None for value in values.values()):
        state = "missing"
    elif values["input_tokens"] is None or values["output_tokens"] is None:
        state = "partial"
    return dict(values, usage_state=state)


def _failed(agent, db):
    agent._usage_event_recording_incomplete = True
    if db is not None:
        note_recording_failure(db)
    else:
        logger.warning("usage_event_recording_unavailable; local usage coverage is incomplete")


def observe_execution(agent, kwargs, call, *, retry_count):
    """Snapshot attribution BEFORE execution; stamp completion when the call returns.

    The callback is one outer attempt. Hidden transport/SDK failures have no returned
    usage; those attempts cannot be reconstructed here. See usage-events.md.
    """
    agent._usage_event_observation = None
    db = getattr(agent, "_session_db", None)
    provider, mode = agent.provider, agent.api_mode
    if provider != "openai-codex" or mode != "codex_responses":
        return call(kwargs)
    try:
        attempt_id = uuid.uuid4().hex
        model = str(kwargs.get("model") or agent.model)
        profile = hashlib.sha256(hermes_home_key(db.db_path.parent).encode()).hexdigest() if db else ""
    except Exception:
        _failed(agent, db)
        return call(kwargs)
    response = call(kwargs)  # Provider exceptions retain their original behavior.
    completed = time.time_ns() // 1000
    try:
        status = _get(response, "status")
        status = status if status in {"completed", "incomplete", "failed", "cancelled", "in_progress", "queued"} else None
        event = UsageEvent(attempt_id, provider, model, profile, completed,
                           retry_count=retry_count, status=status,
                           **_codex_tokens(_get(response, "usage")))
        agent._usage_event_observation = _Observation(id(response), event, db)
    except Exception:
        _failed(agent, db)
    return response


def record_post_request(agent, response) -> None:
    """Independent of plugin registration; duplicate seam delivery reuses the same UUID."""
    observation = getattr(agent, "_usage_event_observation", None)
    if not isinstance(observation, _Observation):
        return  # Auxiliary / synthetic / unsupported routes did not execute here.
    db = observation.db
    try:
        if observation.response_identity != id(response) or db is None:
            _failed(agent, db)
            return
        if not db.record_usage_event(observation.event):
            agent._usage_event_recording_incomplete = True
    except Exception:
        _failed(agent, db)
