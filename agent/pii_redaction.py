"""Opt-in local PII redaction for outbound hosted-LLM payloads.

This module is the last Python-side boundary before Hermes sends structured
request data to model providers.  When enabled, it redacts text fields in API
payloads with a local backend before hosted-provider dispatch, while preserving
local inference by default through ``hosted_only``.  The feature reduces
accidental exposure of common PII, but it is not a hard security boundary:
opaque binary/media fields, identifiers, and provider metadata are deliberately
left untouched so requests remain valid.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.model_metadata import is_local_endpoint


logger = logging.getLogger(__name__)

SKIP_DISABLED = "disabled"
SKIP_LOCAL_ENDPOINT = "local_endpoint"

_SKIP_STRING_KEYS = {
    "audio_url",
    "base_url",
    "b64_json",
    "cache_control",
    "cache_key",
    "call_id",
    "encrypted_content",
    "encoding",
    "endpoint",
    "extra_headers",
    "data",
    "file_data",
    "file_id",
    "filename",
    "function_call_id",
    "id",
    "image_url",
    "include",
    "item_id",
    "metadata",
    "mime_type",
    "model",
    "name",
    "output_index",
    "previous_response_id",
    "provider",
    "prompt_cache_key",
    "reasoning",
    "reasoning_effort",
    "response_id",
    "role",
    "service_tier",
    "status",
    "store",
    "temperature",
    "tool_call_id",
    "type",
    "url",
}


def load_pii_redaction_config(config: Optional[dict] = None) -> dict:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    raw = (config or {}).get("security", {}).get("pii_redaction", {}) or {}
    rampart = raw.get("rampart", {}) or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "provider": str(raw.get("provider", "rampart") or "rampart"),
        "hosted_only": bool(raw.get("hosted_only", True)),
        "fail_closed": bool(raw.get("fail_closed", True)),
        "timeout_seconds": float(raw.get("timeout_seconds", 10) or 10),
        "rampart": {
            "command": str(rampart.get("command", "") or ""),
            "model": str(rampart.get("model", "nationaldesignstudio/rampart") or "nationaldesignstudio/rampart"),
            "heuristics_only": bool(rampart.get("heuristics_only", False)),
        },
    }


def should_redact_for_llm(
    base_url: Optional[str],
    *,
    config: Optional[dict] = None,
    pii_config: Optional[dict] = None,
) -> bool:
    cfg = pii_config or load_pii_redaction_config(config)
    if not cfg.get("enabled"):
        return False
    if cfg.get("hosted_only", True) and base_url and is_local_endpoint(base_url):
        return False
    return True


def _skip_reason(base_url: Optional[str], cfg: dict) -> Optional[str]:
    if not cfg.get("enabled"):
        return SKIP_DISABLED
    if cfg.get("hosted_only", True) and base_url and is_local_endpoint(base_url):
        return SKIP_LOCAL_ENDPOINT
    return None


def redact_messages_for_llm(
    messages: Any,
    *,
    base_url: Optional[str] = None,
    config: Optional[dict] = None,
    pii_config: Optional[dict] = None,
) -> Tuple[Any, Dict[str, Any]]:
    cfg = pii_config or load_pii_redaction_config(config)
    reason = _skip_reason(base_url, cfg)
    if reason:
        stats = _stats(redacted=False, skipped=True, skipped_reason=reason)
        _log_stats(stats)
        return messages, stats
    redacted, stats = _redact_payload(messages, cfg)
    _log_stats(stats)
    return redacted, stats


def redact_text_for_llm(
    text: str,
    *,
    base_url: Optional[str] = None,
    config: Optional[dict] = None,
    pii_config: Optional[dict] = None,
) -> Tuple[str, Dict[str, Any]]:
    cfg = pii_config or load_pii_redaction_config(config)
    reason = _skip_reason(base_url, cfg)
    if reason:
        stats = _stats(redacted=False, skipped=True, skipped_reason=reason)
        _log_stats(stats)
        return text, stats
    redacted, stats = _redact_payload(str(text), cfg)
    _log_stats(stats)
    return redacted, stats


def maybe_redact_api_kwargs(
    api_kwargs: dict,
    *,
    base_url: Optional[str] = None,
    config: Optional[dict] = None,
    pii_config: Optional[dict] = None,
) -> Tuple[dict, Dict[str, Any]]:
    cfg = pii_config or load_pii_redaction_config(config)
    reason = _skip_reason(base_url, cfg)
    if reason:
        stats = _stats(redacted=False, skipped=True, skipped_reason=reason)
        _log_stats(stats)
        return api_kwargs, stats
    redacted, stats = _redact_payload(api_kwargs, cfg)
    _log_stats(stats)
    return redacted, stats


def _stats(**overrides: Any) -> Dict[str, Any]:
    stats = {
        "enabled": False,
        "redacted": False,
        "skipped": False,
        "skipped_reason": None,
        "backend": None,
        "replacement_count": 0,
        "texts_scanned": 0,
        "texts_changed": 0,
        "chars_in": 0,
        "chars_out": 0,
        "failure": None,
    }
    stats.update(overrides)
    return stats


def _log_stats(stats: Dict[str, Any]) -> None:
    if stats.get("skipped"):
        logger.info(
            "PII redaction skipped: reason=%s backend=%s",
            stats.get("skipped_reason"),
            stats.get("backend"),
        )
        return
    logger.info(
        "PII redaction completed: backend=%s scanned=%s changed=%s replacements=%s failure=%s",
        stats.get("backend"),
        stats.get("texts_scanned"),
        stats.get("texts_changed"),
        stats.get("replacement_count"),
        stats.get("failure"),
    )


def _redact_payload(payload: Any, cfg: dict) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(payload, str):
        return _redact_plain_text(payload, cfg)
    copied = copy.deepcopy(payload)
    slots: List[Tuple[Any, Any]] = []
    texts: List[str] = []
    _collect_text_slots(copied, slots, texts)
    stats = _stats(
        enabled=True,
        redacted=True,
        backend=cfg.get("provider", "rampart"),
        texts_scanned=len(texts),
        chars_in=sum(len(t) for t in texts),
    )
    if not texts:
        stats["chars_out"] = 0
        return copied, stats

    try:
        redacted_texts = _redact_texts(texts, cfg)
    except Exception as exc:
        stats["failure"] = type(exc).__name__
        if cfg.get("fail_closed", True):
            raise RuntimeError(
                "PII redaction failed before provider dispatch; refusing to send unredacted payload"
            ) from exc
        return payload, stats

    if len(redacted_texts) != len(texts):
        stats["failure"] = "InvalidRedactionCount"
        if cfg.get("fail_closed", True):
            raise RuntimeError(
                "PII redaction failed before provider dispatch; backend returned an invalid response"
            )
        return payload, stats

    changed = 0
    for (container, key), value in zip(slots, redacted_texts):
        if value != container[key]:
            changed += 1
        container[key] = value
    stats["texts_changed"] = changed
    stats["replacement_count"] = changed
    stats["chars_out"] = sum(len(t) for t in redacted_texts)
    return copied, stats


def _redact_plain_text(text: str, cfg: dict) -> Tuple[str, Dict[str, Any]]:
    stats = _stats(
        enabled=True,
        redacted=True,
        backend=cfg.get("provider", "rampart"),
        texts_scanned=1 if text else 0,
        chars_in=len(text),
    )
    if not text:
        return text, stats
    try:
        redacted_texts = _redact_texts([text], cfg)
    except Exception as exc:
        stats["failure"] = type(exc).__name__
        if cfg.get("fail_closed", True):
            raise RuntimeError(
                "PII redaction failed before provider dispatch; refusing to send unredacted payload"
            ) from exc
        return text, stats
    if len(redacted_texts) != 1:
        stats["failure"] = "InvalidRedactionCount"
        if cfg.get("fail_closed", True):
            raise RuntimeError(
                "PII redaction failed before provider dispatch; backend returned an invalid response"
            )
        return text, stats
    redacted = redacted_texts[0]
    stats["texts_changed"] = 1 if redacted != text else 0
    stats["replacement_count"] = stats["texts_changed"]
    stats["chars_out"] = len(redacted)
    return redacted, stats


def _collect_text_slots(node: Any, slots: List[Tuple[Any, Any]], texts: List[str], parent_key: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_s = str(key)
            if key_s == "tools" or key_s.lower() in _SKIP_STRING_KEYS:
                continue
            if isinstance(value, str):
                if _is_text_string(key_s, value):
                    slots.append((node, key))
                    texts.append(value)
            else:
                _collect_text_slots(value, slots, texts, key_s)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, str):
                if _is_text_string(parent_key, value):
                    slots.append((node, index))
                    texts.append(value)
            else:
                _collect_text_slots(value, slots, texts, parent_key)


def _is_text_string(key: str, value: str) -> bool:
    if not value:
        return False
    key_l = key.lower()
    if key_l in _SKIP_STRING_KEYS:
        return False
    value_l = value[:128].lower()
    if value_l.startswith(("data:", "http://", "https://", "file://")):
        return False
    return True


def _worker_env() -> dict:
    allowed = {}
    for key in (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SYSTEMROOT",
        "WINDIR",
    ):
        value = os.environ.get(key)
        if value:
            allowed[key] = value
    return allowed


def _redact_texts(texts: List[str], cfg: dict) -> List[str]:
    provider = str(cfg.get("provider", "rampart") or "rampart").lower()
    if provider != "rampart":
        raise RuntimeError("unsupported PII redaction provider")
    return _redact_with_rampart(texts, cfg)


def _redact_with_rampart(texts: List[str], cfg: dict) -> List[str]:
    rampart = cfg.get("rampart", {}) or {}
    command = str(rampart.get("command", "") or "").strip()
    if command:
        cmd = shlex.split(command)
    else:
        worker = Path(__file__).with_name("rampart_pii_worker.mjs")
        cmd = ["node", str(worker)]
    payload = {
        "texts": texts,
        "model": rampart.get("model") or "nationaldesignstudio/rampart",
        "heuristicsOnly": bool(rampart.get("heuristics_only", False)),
    }
    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=float(cfg.get("timeout_seconds", 10) or 10),
            check=False,
            env=_worker_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("PII redaction worker timed out") from exc
    except OSError as exc:
        raise RuntimeError("PII redaction worker unavailable") from exc
    if completed.returncode != 0:
        raise RuntimeError("PII redaction worker failed")
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("PII redaction worker returned invalid JSON") from exc
    redacted = (
        data.get("texts")
        or data.get("redactedTexts")
        or data.get("redacted_texts")
        or data.get("results")
    )
    if not isinstance(redacted, list) or not all(isinstance(item, str) for item in redacted):
        raise RuntimeError("PII redaction worker returned invalid payload")
    return redacted
