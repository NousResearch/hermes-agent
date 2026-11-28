"""Request-attributed recovery for profile-classified encrypted replay rejections."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any

from agent.error_classifier import FailoverReason
from agent.route_verdicts import route_visit


def is_classified_replay_401(classified: Any, status_code: Any) -> bool:
    return (classified.status_code or status_code) == 401 and classified.reason == FailoverReason.invalid_encrypted_content


def _opaque_key(item: Any) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    kind, blob = item.get("type"), item.get("encrypted_content")
    if kind in ("reasoning", "compaction") and isinstance(blob, str) and blob.strip():
        return kind, blob
    return None


def _history_items(messages: Any) -> list[dict]:
    return [item for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"
            and isinstance(msg.get("codex_reasoning_items"), list)
            for item in msg["codex_reasoning_items"]] if isinstance(messages, list) else []


def _sent_items(error: Exception) -> list:
    """The SDK merges extra_body AFTER preflight; only its serialized request is authoritative."""
    try:
        request = getattr(error, "request", None)
        if request is None:
            request = getattr(getattr(error, "response", None), "request", None)
        if request is None:
            return []
        body = json.loads(request.content)
    except (AttributeError, ValueError, UnicodeError, RuntimeError):
        return []
    items = body.get("input") if isinstance(body, dict) else None
    return items if isinstance(items, list) else []


def _drop_keys(messages: Any, keys: set[tuple[str, str]]) -> None:
    for msg in messages if isinstance(messages, list) else []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        items = msg.get("codex_reasoning_items")
        if not isinstance(items, list):
            continue
        kept = [item for item in items if _opaque_key(item) not in keys]
        if kept:
            msg["codex_reasoning_items"] = kept
        else:
            msg.pop("codex_reasoning_items", None)


def recover_classified_replay(agent: Any, error: Exception, retry: Any, messages: Any, api_messages: Any) -> bool:
    """One repair per route visit, never a global replay opt-out or a history-wide sweep.

    Summary defaults, ids and provenance change during preflight. Attribute by opaque identity,
    requiring uniqueness independently in the wire, request copy and durable source history.
    """
    if agent.api_mode != "codex_responses" or retry.invalid_encrypted_content_retry_visit == route_visit(agent):
        return False
    source = _history_items(messages)
    counts = [Counter(_opaque_key(item) for item in items)
              for items in (_sent_items(error), _history_items(api_messages), source)]
    eligible = {key for key, count in counts[0].items()
                if key is not None and count == 1 and counts[1][key] == 1 and counts[2][key] == 1}
    if not eligible:
        return False

    from agent.codex_responses_adapter import _canonical_issuer_kind, _model_can_read_blob

    transport = agent._get_transport()
    issuer = getattr(transport, "_last_issuer_kind", None)
    model = getattr(transport, "_last_issuer_model", None)
    if issuer is None or model is None:
        return False
    # An override can send an item from a foreign source history. Do not erase that history.
    eligible = {key for item in source if (key := _opaque_key(item)) in eligible
                and _canonical_issuer_kind(item.get("_issuer_kind")) in (None, issuer)
                and _model_can_read_blob(issuer, item.get("_issuer_model"), model)}
    if not eligible:
        return False
    # Retain the existing unverified-first policy, but restrict its scope to attributable sent items.
    probe = [{"role": "assistant", "codex_reasoning_items":
              deepcopy([item for item in source if _opaque_key(item) in eligible])}]
    if transport.drop_unverified_replay(probe):
        eligible -= {_opaque_key(item) for item in _history_items(probe)}
    retry.invalid_encrypted_content_retry_visit = route_visit(agent)
    _drop_keys(messages, eligible)
    _drop_keys(api_messages, eligible)
    agent._db_flush_scan_prefix = None
    agent._buffer_vprint(f"⚠️  Encrypted replay rejected — dropped {len(eligible)} sent item(s), retrying...")
    return True
