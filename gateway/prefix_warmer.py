"""Prompt prefix warmer for local backends.

Local llama.cpp-style servers only reuse cached state when a new request's
rendered prompt shares a byte-identical prefix with a state they still hold —
and on hybrid-SSM models (Qwen3-Next/AgentWorld-class), only when a recurrent
state checkpoint exists at or before the divergence point. In practice this
means the first session after an idle stretch (or after other traffic rotated
the cache) pays a full prefill of the shared prompt prefix — tool schemas plus
the stable system prompt, often 10-20k tokens — even though every session on
that profile sends the identical bytes.

This watcher keeps that shared prefix hot. ``build_api_kwargs`` records the
most recent request the agent sent to each local endpoint (see
``agent/prefix_warm_registry.py``); every ``interval_seconds`` the watcher
replays a minimal request per endpoint:

    messages   = [the captured system message, one fixed "." user turn]
    tools      = the captured tool schemas (rendered into the prompt by the
                 server's chat template, so they must match byte-for-byte)
    extra_body = the captured extra_body (may carry template-affecting
                 options such as chat_template_kwargs)
    max_tokens = 1, temperature = 0

The server therefore keeps a cached state whose prompt is exactly the shared
prefix plus a tiny tail. A fresh session reuses that state up to the point
where its prompt diverges — in practice somewhere inside the system block,
because Hermes appends a volatile tail (memory block, user profile, session
timestamp) after the stable head. The stable head (tool schemas + fixed
system prose) is the bulk of the prefix, so realized reuse is typically most
of the entry tax, not all of it; a memory write or date rollover between
warm and reuse shortens it further. Re-warming a state that is still cached
costs only a few prefill tokens plus one generated token, so the
steady-state overhead is negligible.

To avoid competing with live sessions for cache slots, a warm cycle skips
any endpoint that has seen real traffic within ``min_idle_seconds``: recent
traffic means the prefix is already hot, and on few-slot servers a warm
request landing mid-conversation could otherwise evict an active session's
state (llama.cpp assigns requests to slots by longest-common-prefix, so a
warm request can select — and truncate — a slot holding a live session's
KV). The guard keys off request-build times, so a single generation that
runs longer than ``min_idle_seconds`` can still collide; on servers with a
single parallel slot, prefer leaving the warmer off.

Opt-in via config.yaml (the warmer is pointless for cloud providers, which
manage their own prompt caches):

    prefix_warmer:
      enabled: true
      interval_seconds: 240
      min_idle_seconds: 180
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Fixed one-character user turn. Content is irrelevant — it just has to be
# constant so consecutive warm requests hit their own cached state.
_WARM_USER_CONTENT = "."


def warm_once(config: Any) -> int:
    """Send one warm request per recorded local prefix. Returns count warmed.

    Synchronous (called via ``asyncio.to_thread``): the OpenAI client is
    blocking, and a cache-hitting warm request completes in well under a
    second. Errors are logged at debug and never propagate — a local server
    that is restarting or busy simply misses one warm cycle.

    Endpoints with real traffic within ``config.min_idle_seconds`` are
    skipped: their prefix is already hot, and warming a busy few-slot server
    risks evicting a live session's cached state (see module docstring).
    """
    from agent.prefix_warm_registry import get_snapshots, last_traffic_at

    min_idle = float(getattr(config, "min_idle_seconds", 180.0))
    warmed = 0
    for snap in get_snapshots():
        base_url = snap["base_url"]
        idle = time.time() - last_traffic_at(base_url)
        if idle < min_idle:
            logger.debug(
                "prefix_warmer: skipping %s — active %.0fs ago (< %.0fs idle gate)",
                base_url, idle, min_idle,
            )
            continue
        try:
            kwargs: Dict[str, Any] = {
                "model": snap["model"],
                "messages": [
                    {"role": "system", "content": snap["system_content"]},
                    {"role": "user", "content": _WARM_USER_CONTENT},
                ],
                "max_tokens": 1,
                "temperature": 0,
            }
            if snap.get("tools"):
                kwargs["tools"] = snap["tools"]
            if snap.get("extra_body"):
                kwargs["extra_body"] = snap["extra_body"]
            t0 = time.time()
            _send_warm_request(snap["base_url"], snap["api_key"], config, kwargs)
            warmed += 1
            logger.debug(
                "prefix_warmer: warmed %s @ %s in %.2fs",
                snap["model"], snap["base_url"], time.time() - t0,
            )
        except Exception as exc:
            logger.debug(
                "prefix_warmer: warm failed for %s @ %s: %s",
                snap.get("model"), snap.get("base_url"), exc,
            )
    return warmed


def warm_compacted_prefix(
    old_system: str,
    new_system: str,
    compacted_messages: list,
    config: Any,
) -> int:
    """Re-warm a session's local endpoints right after context compaction.

    Compaction rewrites the transcript (summary + surviving tail) and appends
    a compaction note to the system prompt, so the next turn's rendered
    prompt shares no usable prefix with any cached state — the session pays
    the full re-prefill at TTFT (seconds on a 20k+ prefix). This replays the
    *post-compaction* prefix once, so the re-prefill happens now, in the
    background, instead of in front of the user's next message.

    Endpoint selection reuses the warm registry: a snapshot whose recorded
    system content equals the *pre-compaction* system prompt belongs to this
    session's request path (the direct provider, or the MoA acting/aggregator
    call — both record). Only local endpoints are ever recorded, so cloud
    providers are excluded by construction. Returns count warmed.
    """
    from agent.prefix_warm_registry import get_snapshots

    warmed = 0
    messages = [{"role": "system", "content": new_system}]
    messages.extend(compacted_messages)
    for snap in get_snapshots():
        if snap.get("system_content") != old_system:
            continue
        try:
            kwargs: Dict[str, Any] = {
                "model": snap["model"],
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0,
            }
            if snap.get("tools"):
                kwargs["tools"] = snap["tools"]
            if snap.get("extra_body"):
                kwargs["extra_body"] = snap["extra_body"]
            t0 = time.time()
            _send_warm_request(snap["base_url"], snap["api_key"], config, kwargs)
            warmed += 1
            logger.debug(
                "prefix_warmer: post-compaction warm %s @ %s in %.2fs",
                snap["model"], snap["base_url"], time.time() - t0,
            )
        except Exception as exc:
            logger.debug(
                "prefix_warmer: post-compaction warm failed for %s @ %s: %s",
                snap.get("model"), snap.get("base_url"), exc,
            )
    if not warmed:
        # Either no local endpoint is recorded yet, or the recorded system
        # content didn't byte-match the pre-compaction prompt (a divergence
        # in the request path worth knowing about when debugging TTFT).
        logger.debug(
            "prefix_warmer: post-compaction warm matched no snapshots (%d recorded)",
            len(get_snapshots()),
        )
    return warmed


def _send_warm_request(base_url: str, api_key: str, config: Any, kwargs: Dict[str, Any]) -> None:
    """One blocking chat-completions call to the local server."""
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "local",
        timeout=float(getattr(config, "timeout_seconds", 120.0)),
        max_retries=0,
    )
    try:
        client.chat.completions.create(**kwargs)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def prefix_warmer_watcher(runner: Any, config: Any) -> None:
    """Background loop: keep recorded local prompt prefixes warm.

    Follows the gateway watcher conventions (initial settle delay,
    ``runner._running`` loop condition, exceptions confined to one tick).
    The loop also fires one immediate warm after the settle delay so a
    gateway restart re-establishes the warm state without waiting a full
    interval.
    """
    interval = max(30, int(getattr(config, "interval_seconds", 240)))
    await asyncio.sleep(60)  # initial delay — let the gateway fully start
    logger.info("prefix_warmer: watching (interval %ds)", interval)
    while getattr(runner, "_running", False):
        try:
            warmed = await asyncio.to_thread(warm_once, config)
            if warmed:
                logger.debug("prefix_warmer: %d prefix(es) warm", warmed)
        except Exception as exc:
            logger.debug("prefix_warmer: tick error: %s", exc)
        await asyncio.sleep(interval)
