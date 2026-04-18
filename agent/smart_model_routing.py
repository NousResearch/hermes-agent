"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from utils import is_truthy_value

_COMPLEX_KEYWORDS = {
    "debug",
    "debugging",
    "implement",
    "implementation",
    "refactor",
    "patch",
    "traceback",
    "stacktrace",
    "exception",
    "error",
    "analyze",
    "analysis",
    "investigate",
    "architecture",
    "design",
    "compare",
    "benchmark",
    "optimize",
    "optimise",
    "review",
    "terminal",
    "shell",
    "tool",
    "tools",
    "pytest",
    "test",
    "tests",
    "plan",
    "planning",
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Conservative by design: if the message has signs of code/tool/debugging/
    long-form work, keep the primary model.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 1:
        return None
    if "```" in text or "`" in text:
        return None
    if _URL_RE.search(text):
        return None

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    """
    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"smart route → {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }


# Recognised source kinds for ``model.by_source``. Callers pass one of these
# strings; unknown kinds are ignored (treated as no override).
SOURCE_KIND_OWNER = "owner"
SOURCE_KIND_HUB_PEER = "hub_peer"
SOURCE_KIND_STRANGER = "stranger"
SOURCE_KIND_CRON = "cron"
KNOWN_SOURCE_KINDS = frozenset({
    SOURCE_KIND_OWNER,
    SOURCE_KIND_HUB_PEER,
    SOURCE_KIND_STRANGER,
    SOURCE_KIND_CRON,
})

_OVERRIDE_RUNTIME_KEYS = ("api_key", "base_url", "provider", "api_mode", "command", "args")


def apply_source_override(
    model: str,
    runtime_kwargs: Dict[str, Any],
    model_config: Optional[Dict[str, Any]],
    source_kind: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """Apply per-source model-bundle overrides from ``model.by_source.<kind>``.

    ``config.yaml`` may optionally declare::

        model:
          default: my-fast-model
          provider: custom
          api_key: sk-default
          base_url: https://example.com/v1
          by_source:
            owner:    { model: my-strong-model, api_key: sk-owner }
            hub_peer: { model: my-fast-model }
            stranger: { }             # empty → use base
            cron:     { model: my-fast-model }

    When a ``by_source`` entry exists for ``source_kind``, its fields override
    the passed-in ``(model, runtime_kwargs)`` pair. Missing fields inherit the
    base values (partial overrides are supported). If ``source_kind`` is empty
    or no entry exists, the inputs are returned unchanged.

    Callers are responsible for classifying the source (``"owner"``,
    ``"hub_peer"``, ``"stranger"``, ``"cron"``) using whatever signals they
    have: the gateway uses home-channel identity; cron hardcodes ``"cron"``;
    the CLI hardcodes ``"owner"``.
    """
    if not source_kind:
        return model, runtime_kwargs
    if not isinstance(model_config, dict):
        return model, runtime_kwargs
    by_source = model_config.get("by_source")
    if not isinstance(by_source, dict):
        return model, runtime_kwargs
    entry = by_source.get(source_kind)
    if not isinstance(entry, dict) or not entry:
        return model, runtime_kwargs

    new_model = model
    new_runtime = dict(runtime_kwargs or {})
    applied = []

    if entry.get("model"):
        new_model = str(entry["model"])
        applied.append("model")
    for key in _OVERRIDE_RUNTIME_KEYS:
        val = entry.get(key)
        if val in (None, "", []):
            continue
        if key == "args":
            new_runtime[key] = list(val)
        else:
            new_runtime[key] = val
        applied.append(key)

    if applied:
        import logging
        logging.getLogger(__name__).info(
            "by_source override applied: source_kind=%s fields=%s model=%s provider=%s",
            source_kind, applied, new_model, new_runtime.get("provider"),
        )
    return new_model, new_runtime
