# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/memory/thread-classifier.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""LLM arbiter for "is this new thread actually the same as this
existing candidate thread?" — the fallback for rule 4/5 of thread
attribution when the v0 keyword overlap is ambiguous.

Fire-and-forget path: hard 800ms timeout, failure returns ``None``
and the caller keeps the v0 verdict. Never mutates state.

Ported from BaiLongma's ``thread-classifier.js`` (MIT).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Mapping, Optional

from .threads import Thread


logger = logging.getLogger(__name__)


CLASSIFIER_TIMEOUT_MS = 800
CLASSIFIER_MAX_TOKENS = 120
CLASSIFIER_TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "Thread classifier. A new conversation thread was just created for "
    "the user's message. Decide whether it is actually the SAME ongoing "
    "matter as the candidate existing thread.\n"
    "same: the message continues/resumes the candidate thread's matter "
    "(same task, same object, explicit back-reference).\n"
    "different: a genuinely new matter, even if it shares a domain word "
    "with the candidate.\n"
    "Be conservative: when unsure, answer \"different\" (a duplicate "
    "thread is cheap; a wrong merge pollutes history).\n"
    "Also produce a short human-readable label (<=12 chars, Chinese) "
    "and 2-3 semantic topic words for the NEW message's matter (not "
    "n-grams).\n"
    "Output JSON only."
)


CallLLM = Callable[..., Awaitable[Any]]
"""Async LLM adapter. Must accept keyword args:

    system_prompt, message, temperature, max_tokens, tools,
    thinking, must_reply

and return either a string or an object with a ``.content`` /
``["content"]`` attribute.
"""


def _describe_thread_brief(thread: Optional[Thread]) -> str:
    if thread is None:
        return "(none)"
    topic = ", ".join(thread.topic) if thread.topic else ""
    conclusion = ""
    if thread.conclusions:
        conclusion = f" (conclusion: {thread.conclusions[-1]})"
    summary = f" (summary: {str(thread.summary)[:120]})" if thread.summary else ""
    label = thread.label or topic
    return f'"{label}"{summary or conclusion}'


def _build_user_prompt(
    *,
    new_message: str,
    candidate_thread: Optional[Thread],
    created_topic: list[str],
) -> str:
    msg = str(new_message or "")[:400]
    return "\n".join(
        [
            f"Candidate existing thread = {_describe_thread_brief(candidate_thread)}",
            f'New message = "{msg}"',
            f"v0 topic for new thread = [{', '.join(created_topic or [])}]",
            "",
            'Output JSON: {"verdict": "same|different", "label": "...", "topic": ["w1","w2","w3"]}',
        ]
    )


_THINK_TAG_RE = re.compile(
    r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", re.IGNORECASE
)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _parse_json(text: Optional[str]) -> Optional[Mapping[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    body = _THINK_TAG_RE.sub("", text).strip()
    fence = _FENCE_RE.search(body)
    if fence:
        body = fence.group(1).strip()
    first = body.find("{")
    last = body.rfind("}")
    if first < 0 or last <= first:
        return None
    try:
        parsed = json.loads(body[first : last + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _normalize(raw: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if not raw or not isinstance(raw, Mapping):
        return None
    verdict = str(raw.get("verdict") or "").lower().strip()
    if verdict not in ("same", "different"):
        return None
    label = str(raw.get("label") or "").strip()[:24]
    topic_raw = raw.get("topic")
    topic: list[str] = []
    if isinstance(topic_raw, list):
        for t in topic_raw:
            tw = str(t or "").strip()
            if tw and len(tw) <= 32:
                topic.append(tw)
            if len(topic) == 3:
                break
    return {"verdict": verdict, "label": label, "topic": topic}


async def classify_thread_attribution(
    *,
    call_llm: CallLLM,
    new_message: str,
    candidate_thread: Optional[Thread],
    created_topic: list[str],
    timeout_ms: int = CLASSIFIER_TIMEOUT_MS,
) -> Optional[dict]:
    """Arbitrate "created new thread vs weak-signal candidate = same
    matter?". Returns ``None`` on any failure (import error, timeout,
    parse failure) so the caller keeps the v0 verdict.

    ``call_llm`` is injected — Hermes wires this to the auxiliary-model
    adapter at the seam so the arbiter can run against a small
    router-picked model without perturbing the main conversation cache.
    """
    if not new_message or not isinstance(new_message, str):
        return None

    t0 = time.time()

    system_prompt = SYSTEM_PROMPT
    user_prompt = _build_user_prompt(
        new_message=new_message,
        candidate_thread=candidate_thread,
        created_topic=created_topic,
    )

    async def _work() -> Any:
        return await call_llm(
            system_prompt=system_prompt,
            message=user_prompt,
            temperature=CLASSIFIER_TEMPERATURE,
            thinking=False,
            tools=[],
            max_tokens=CLASSIFIER_MAX_TOKENS,
            must_reply=False,
        )

    try:
        result = await asyncio.wait_for(_work(), timeout=timeout_ms / 1000.0)
    except asyncio.TimeoutError:
        logger.info(
            "[thread-classifier] LLM timeout (%.0fms) → falling back to v0",
            (time.time() - t0) * 1000,
        )
        return None
    except Exception as err:  # noqa: BLE001 — fire-and-forget path
        logger.info(
            "[thread-classifier] LLM raised (%.0fms, %s) → falling back to v0",
            (time.time() - t0) * 1000,
            err,
        )
        return None

    if result is None:
        logger.info("[thread-classifier] LLM returned None → falling back to v0")
        return None

    if isinstance(result, str):
        content = result
    else:
        content = ""
        try:
            content = getattr(result, "content", None) or result["content"]  # type: ignore[index]
        except (KeyError, TypeError):
            pass

    normalized = _normalize(_parse_json(content))
    if not normalized:
        logger.info(
            '[thread-classifier] JSON parse/normalize failed raw="%s" '
            "→ falling back to v0",
            re.sub(r"\s+", " ", str(content))[:160],
        )
        return None

    logger.info(
        '[thread-classifier] verdict=%s label="%s" (%.0fms)',
        normalized["verdict"],
        normalized["label"],
        (time.time() - t0) * 1000,
    )
    return normalized


__all__ = [
    "CLASSIFIER_MAX_TOKENS",
    "CLASSIFIER_TEMPERATURE",
    "CLASSIFIER_TIMEOUT_MS",
    "SYSTEM_PROMPT",
    "CallLLM",
    "classify_thread_attribution",
]
