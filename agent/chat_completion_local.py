"""Chat completions against llama-server (sibling of ``chat_completion_helpers``).

Which requests go to llama-server, the turn-start health check that lets a recovered server try
again past the stale breaker, and the live load / prompt-reading notice shown before the first
token.
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.request
from typing import Optional
from urllib.parse import urlparse, urlsplit

from utils import env_int

logger = logging.getLogger(__name__)


def _streams_llamacpp(agent) -> bool:
    """True when this agent's chat requests go to llama-server: the managed server, or a
    ``llamacpp`` provider endpoint. These requests always stream and ask for prompt progress
    (unless the user turned streaming off in config). A non-streamed request can't be cancelled
    through the router, so it would keep the server busy after Hermes gives up on it."""
    if getattr(agent, "api_mode", "") not in ("", "chat_completions"):
        return False
    from hermes_cli.local_runtime.capabilities import is_managed_provider

    return is_managed_provider(str(getattr(agent, "provider", "") or ""), str(getattr(agent, "base_url", "") or ""))


def stale_giveup() -> int:
    """Consecutive stale attempts after which calls fail at once; 0 turns the breaker off."""
    return env_int("HERMES_STREAM_STALE_GIVEUP", 5)


def merge_native_local_settings(agent, request_overrides: dict | None) -> dict:
    """Merge the staged artifact's saved native settings into request overrides.

    Applies only when this agent's chat request targets the managed llama.cpp
    server with a staged model AND the profile already has a settings store.
    Only values the user chose contribute (explicit request + saved settings);
    publisher/catalog defaults remain llama-server's own launch defaults, so a
    hand-edited ``presets.ini`` keeps working.

    Standard OpenAI fields fill in WITHOUT overriding explicit per-request
    values; llama.cpp engine extensions merge into ``extra_body`` PER KEY
    (existing entries win, unrelated extensions survive — never a blind
    update). The caller's dict is never mutated; a merged copy comes back.

    Settings-domain trouble on OUR side (unknown model, corrupt store, torn
    artifact) must never break inference: the request returns unchanged and
    the reason is logged. The only errors that surface are the native schema
    refusing the caller's own explicit per-request payload.
    """
    overrides = dict(request_overrides or {})
    model_id = str(getattr(agent, "model", "") or "")
    if not model_id or getattr(agent, "api_mode", "") not in ("", "chat_completions"):
        return overrides
    from hermes_cli.local_runtime.capabilities import is_managed_provider

    if not is_managed_provider(str(getattr(agent, "provider", "") or ""),
                               str(getattr(agent, "base_url", "") or "")):
        return overrides

    from hermes_cli.local_runtime import native_model_settings as nms
    from hermes_cli.local_runtime.model_settings import (
        InvalidSettingValueError,
        ModelSettingsStore,
        UnknownSettingError,
    )
    from hermes_constants import get_hermes_home

    # Nothing the user saved can exist before the profile has a settings
    # store: skip, and never hash a multi-GB GGUF on a hot path for users who
    # have not touched model settings. (Explicit payloads are validated by the
    # transport's own path in that case.)
    if not ModelSettingsStore(get_hermes_home()).path.exists():
        return overrides

    # The settings-domain slice of the request: schema-named keys riding
    # top-level or inside extra_body. Everything else (speed, routing,
    # reasoning payloads, engine keys the schema never owned) is not ours.
    keys = set(nms.NATIVE_CHAT_SCHEMA.keys())
    explicit = {k: v for k, v in overrides.items() if k in keys}
    existing_extra = overrides.get("extra_body")
    if isinstance(existing_extra, dict):
        for key, value in existing_extra.items():
            if key in keys and key not in explicit:
                explicit[key] = value

    try:
        standard, extra = nms.resolve_native_request_settings(
            model_id, explicit or None, inherited=False)
    except (UnknownSettingError, InvalidSettingValueError):
        # The resolver validates BOTH the caller's payload and the saved row, with the same exception types, so the type says nothing about
        # whose fault it is. Resolve the saved side alone: if THAT is refused, the saved row is the broken one (an older build, tightened
        # bounds, a hand edit) and breaking every chat request for the model over our own data would be wrong. If it resolves fine, the
        # caller's explicit payload is what was refused: surface it.
        try:
            nms.resolve_native_request_settings(model_id, None, inherited=False)
        except Exception as saved_exc:
            logger.warning("saved native model settings for %s are invalid and were ignored: %s", model_id, saved_exc)
            return overrides
        raise
    except Exception as exc:
        logger.debug("native model settings for %s unavailable: %s", model_id, exc)
        return overrides

    for key, value in standard.items():
        overrides.setdefault(key, value)
    if extra:
        merged = dict(existing_extra) if isinstance(existing_extra, dict) else {}
        for key, value in extra.items():
            merged.setdefault(key, value)
        overrides["extra_body"] = merged
    return overrides


def _local_server_answers(agent) -> bool:
    """True when the llama-server behind *agent* answers ``/health`` within a few seconds."""
    parts = urlsplit(str(getattr(agent, "base_url", "") or ""))
    if not parts.scheme or not parts.hostname:
        return False
    # "localhost" can resolve to ::1 first and wait out a timeout on a server bound to IPv4.
    netloc = parts.netloc.replace("localhost", "127.0.0.1", 1) if parts.hostname == "localhost" else parts.netloc
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{parts.scheme}://{netloc}/health", timeout=3) as response:
            return response.status == 200
    except (OSError, ValueError, http.client.HTTPException):
        return False


def retry_recovered_local_server(agent) -> bool:
    """Turn start: when the stale breaker is open, the agent talks to llama-server, and the
    server answers its health check, let one attempt through. Within a turn the breaker stops
    endless retries; across turns, a local server that crashed or fell behind and has since
    recovered gets the user's next message instead of an instant error. The router answers
    ``/health`` even while its model is stuck, so one more stale attempt shuts the breaker again."""
    from agent.chat_completion_helpers import _stale_streak

    giveup, streak = stale_giveup(), _stale_streak(agent)
    if giveup <= 0 or streak < giveup or not _streams_llamacpp(agent) or not _local_server_answers(agent):
        return False
    logger.info("Local model server answers its health check after %d stale attempts; trying once more.", streak)
    agent._consecutive_stale_streams = giveup - 1
    return True


def _managed_local_load_notice(agent, api_kwargs: dict) -> Optional[str]:
    """Live phase notice ("⏳ loading <model> into memory — N%" / "⚙ processing
    prompt — P%") while the managed local server works before the first token;
    None when neither applies. Otherwise a cold load reads as a generic stall."""
    try:
        base = str(getattr(agent, "base_url", "") or "")
        if not base:
            return None
        from agent.chat_completion_helpers import estimate_request_context_tokens
        from hermes_cli.local_runtime.load_progress import get_loading_progress, get_prefill_progress
        from hermes_cli.local_runtime.supervisor import state_path
        state = json.loads(state_path().read_text(encoding="utf-8-sig"))
        managed = urlparse(str(state.get("base_url", ""))).netloc.lower()
        if not managed or urlparse(base).netloc.lower() != managed:
            return None
        model = str(api_kwargs.get("model", ""))
        progress = get_loading_progress().get(model)
        if progress is not None:
            return (f"⏳ loading {model} into memory — {progress['percent']}% "
                "(responses start once the model is loaded)")
        prefill = get_prefill_progress(model)
        if prefill is None:
            return None
        processed = int(prefill["processed"])
        total = estimate_request_context_tokens(api_kwargs)
        if total and total >= processed:
            return f"⚙ processing prompt — {max(0, min(100, round(processed / total * 100)))}%"
        # Counter past the estimate (estimator undercounted): no honest denominator, label-only.
        return "⚙ processing prompt"
    except Exception:
        # A status line must never break the stream it describes.
        logger.debug("local load notice unavailable", exc_info=True)
        return None
