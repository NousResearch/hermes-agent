"""Hindsight plugin constants and pure config normalizers (no I/O, no origin imports)."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
from typing import Any, List

# Log under the plugin package's own logger name (loader-path independent).
logger = logging.getLogger(__name__.rpartition(".")[0])

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
# Keep in sync with tools/lazy_deps.py ("memory.hindsight") and plugin.yaml.
# Floor is the live-proven openai-codex runtime (hindsight-all 0.9.2 ships
# hindsight-client 0.9.2). Pre-1.0 ranges use <0.(minor+2).
_MIN_CLIENT_VERSION = "0.9.2"
_CLIENT_VERSION_UPPER = "0.11"
_CLIENT_PIP_SPEC = f"hindsight-client>={_MIN_CLIENT_VERSION},<{_CLIENT_VERSION_UPPER}"
_EMBEDDED_RUNTIME_SPEC = f"hindsight-all>={_MIN_CLIENT_VERSION},<{_CLIENT_VERSION_UPPER}"
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
# ``metadata.source`` on retained memories is OPT-IN (AGENTS.md forbids
# on-by-default attribution tags): ``retain_source`` / HINDSIGHT_RETAIN_SOURCE.
_DEFAULT_RETAIN_SOURCE = ""
# Hindsight brand mark (eye ringed by graph nodes) for the recall/retain indicators.
_HINDSIGHT_GLYPH = "👁️"
# Hindsight 0.5.0 added ``update_mode='append'``; older APIs would silently
# overwrite prior turns under a stable document_id, so they keep the per-process id.
# Mirrors hindsight-integrations/openclaw — Hindsight 0.5.0 added `update_mode='append'` semantics on retain
# (vectorize-io/hindsight#932).
_MIN_VERSION_FOR_UPDATE_MODE_APPEND = "0.5.0"
_VALID_BUDGETS = {"low", "mid", "high"}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openai-codex": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-3.6-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M2.7",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}
# The embedded daemon speaks OpenAI wire format for these providers.
_OPENAI_WIRE_PROVIDERS = {"openai_compatible", "openrouter"}
_OBSERVATION_SCOPE_KEYWORDS = {"per_tag", "combined", "all_combinations"}
_MIN_SCORE_KEYS = ("semantic", "keyword", "reranker", "final")


def _uses_codex_oauth(provider: str) -> bool:
    """True when Hindsight authenticates via existing Codex/ChatGPT OAuth, not an LLM API key."""
    return str(provider or "") == "openai-codex"


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


def _daemon_llm_provider(provider: str) -> str:
    return "openai" if provider in _OPENAI_WIRE_PROVIDERS else provider


def _normalize_retain_tags(value: Any) -> List[str]:
    """Normalize tag config/tool values to a deduplicated list of strings."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            with contextlib.suppress(Exception):
                parsed = json.loads(text)
        raw_items = parsed if isinstance(parsed, list) else text.split(",")
    normalized: list[str] = []
    for item in raw_items:
        tag = str(item).strip()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _normalize_observation_scopes(value: Any) -> Any:
    """Normalize observation_scopes to a keyword string, ``list[list[str]]`` (one inner
    list per consolidation pass), or ``None`` (Hindsight's ``combined`` default).
    Accepts a keyword, a JSON-encoded list, a flat tag list (one scope) or a list of
    tag-lists; anything unrecognized -> ``None`` so we never send an invalid payload."""
    if isinstance(value, str):
        text = value.strip()
        if text in _OBSERVATION_SCOPE_KEYWORDS:
            return text
        if text.startswith("["):
            try:
                return _normalize_observation_scopes(json.loads(text))
            except Exception:
                return None
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if all(isinstance(entry, str) for entry in value):  # flat tag list -> one scope
        value = [value]
    scopes = [
        [str(tag).strip() for tag in entry if str(tag).strip()] if isinstance(entry, (list, tuple))
        else [entry.strip()] if isinstance(entry, str) and entry.strip() else []
        for entry in value
    ]
    return [s for s in scopes if s] or None


def _normalize_min_scores(value: Any) -> dict[str, float] | None:
    """Normalize Hindsight ``min_scores`` floors.

    Absent/empty/invalid input -> ``None`` (existing no-floor behavior). A JSON
    object or dict may set any of semantic/keyword/reranker/final. Unknown keys
    and non-finite values are dropped so initialize never crashes on bad config.
    """
    if value is None or value == "" or value == {}:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except Exception:
            logger.warning("Invalid Hindsight recall_min_scores JSON; ignoring floors")
            return None
    if not isinstance(value, dict):
        logger.warning("Invalid Hindsight recall_min_scores type %s; ignoring floors",
                       type(value).__name__)
        return None
    unknown = [key for key in value if key not in _MIN_SCORE_KEYS]
    if unknown:
        logger.warning("Ignoring unknown Hindsight recall_min_scores keys: %s",
                       sorted(str(k) for k in unknown))
    out: dict[str, float] = {}
    for key in _MIN_SCORE_KEYS:
        if key not in value or value[key] is None or value[key] == "":
            continue
        try:
            score = float(value[key])
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric Hindsight recall_min_scores %s", key)
            continue
        if not math.isfinite(score):
            logger.warning("Ignoring non-finite Hindsight recall_min_scores %s", key)
            continue
        out[key] = score
    return out or None


def _sanitize_bank_segment(value: str) -> str:
    """URL/filesystem-safe bank_id placeholder: runs outside ``[A-Za-z0-9_-]`` (per
    ``str.isalnum``) become one dash; leading/trailing ``-``/``_`` are stripped."""
    # \w == str.isalnum() + "_" for str patterns, so this matches the per-char rule.
    return re.sub(r"[^\w-]+", "-", str(value)).strip("-_") if value else ""


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Render a bank_id template ({profile}, {workspace}, {platform}, {user}, {session}),
    sanitizing each placeholder; the ``-``/``_`` runs empty placeholders leave are
    collapsed (``hermes-{user}`` -> ``hermes``). Empty/invalid template -> *fallback*."""
    if not template:
        return fallback
    try:
        rendered = template.format(**{k: _sanitize_bank_segment(v) for k, v in placeholders.items()})
    except (KeyError, IndexError) as exc:
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r",
                       template, exc, fallback)
        return fallback
    return re.sub(r"([-_])\1+", r"\1", rendered).strip("-_") or fallback
