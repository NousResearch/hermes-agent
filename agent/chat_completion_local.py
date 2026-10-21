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


def _targets_managed_server(agent) -> bool:
    """True when *agent* talks to the managed router (the state file's endpoint), not an external one."""
    from hermes_cli.local_runtime.recovery import read_state

    managed = urlparse(str(read_state().get("base_url", ""))).netloc.lower()
    return bool(managed) and urlparse(str(getattr(agent, "base_url", "") or "")).netloc.lower() == managed


def refuse_unfitting_local_load(agent, api_kwargs: dict) -> None:
    """Raise before the request when it would make the managed router cold-load a model that does
    not fit in the memory free right now (#142082). The router would load it anyway and, on
    Windows, page the whole machine to a halt; an external llama-server is not ours to judge."""
    if not _targets_managed_server(agent):
        return
    from hermes_cli.local_runtime.admission import check_load_admission

    refusal = check_load_admission(str(api_kwargs.get("model", "")))
    if refusal:
        from agent.errors import LocalModelAdmissionError

        raise LocalModelAdmissionError(refusal)


def local_stream_kwargs(agent, api_kwargs: dict) -> dict:
    """The streamed llama-server request: refused when its cold load won't fit, else with prompt progress."""
    from agent.local_prompt_progress import with_prompt_progress

    refuse_unfitting_local_load(agent, api_kwargs)
    return with_prompt_progress(api_kwargs)


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
