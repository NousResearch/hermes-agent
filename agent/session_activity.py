"""Shared session activity observation contract: timestamp + bounded description/provenance, observation
only (notification, timeout, kill and retry policy live elsewhere). Provenance is a small closed enum of
*noun* sources; the default agent clock stamps ``unknown`` unless a writer passes ``provenance=``."""

from __future__ import annotations

import sys
import time
from contextlib import suppress
from enum import Enum
from typing import Any, Mapping, Optional

from agent.i18n import get_language, t

ACTIVITY_DESCRIPTION_MAX = 120

# Durable SessionDB heartbeat cadence. Contract: MUST stay >= 30s — the SessionDB write path is contended and
# this observation-only projection never justifies extra write pressure. A code constant on purpose (no config
# can make it a high-frequency writer); matches the kanban auto-heartbeat. force_persist and terminal
# compression stamps (see TERMINAL_COMPRESSION_PROVENANCES) are the only bypasses.
SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0


class ActivityProvenance(str, Enum):
    """Where a durable/in-memory activity stamp came from."""

    UNKNOWN = "unknown"
    # Compression writers: heartbeat, host timeout, cooldown, turn hold.
    # See #72424.
    AGENT_COMPRESSION = "agent.compression"
    AGENT_COMPRESSION_TIMEOUT = "agent.compression_timeout"
    AGENT_COMPRESSION_COOLDOWN = "agent.compression_cooldown"
    AGENT_COMPRESSION_TURNHOLD = "agent.compression_turnhold"


# Provenances that END the user-visible "compressing" phase: the host's progress timeout, a cooldown/
# backoff block, and the gateway's hygiene turn-hold. Each is written once, at a terminal edge, and must
# reach the durable projection IMMEDIATELY — the heartbeat above wrote "context compression in progress"
# moments earlier, so a rate-limited terminal stamp leaves `sessions.last_activity_description` advertising
# a compression that has already stopped, with no later writer to correct it (the agent may be idle, the
# process gone). That is the permanently "stuck / compressing" chat users see.
#
# Distinct from ``conversation_compression._TERMINAL_COMPRESSION_PROVENANCES``, which answers a different
# question — "may a detached heartbeat still overwrite this stamp?" — and deliberately excludes TURNHOLD,
# because after a turn-hold the worker may still be alive and adoptable.
TERMINAL_COMPRESSION_PROVENANCES = frozenset(
    {
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
        ActivityProvenance.AGENT_COMPRESSION_TURNHOLD,
    }
)


def is_terminal_compression_provenance(provenance: Optional[ActivityProvenance | str]) -> bool:
    """True when this stamp ends a compression phase and must bypass the persist rate limit."""
    return normalize_activity_provenance(provenance) in TERMINAL_COMPRESSION_PROVENANCES


def bound_activity_description(description: Optional[str]) -> str:
    """Clamp free-form activity text to the shared description budget."""
    text = (description or "").strip()
    return text if len(text) <= ACTIVITY_DESCRIPTION_MAX else text[: ACTIVITY_DESCRIPTION_MAX - 1] + "…"


# ── User-facing activity labels ────────────────────────────────────────────────────────────────
# ``_touch_activity`` stamps the engine's OWN English bookkeeping ("receiving stream response",
# "starting API call #3") into ``last_activity_desc``; the gateway heartbeat renders that field
# verbatim into the chat. A user then reads engine jargon instead of a status line. The durable
# stamp stays canonical English (diagnostics and SessionDB readers depend on it) — only the
# rendering is localized, here, at the single read-side helper.
#
# Unmatched text passes through unchanged: new writers degrade to the raw label, never to nothing.
_ACTIVITY_LABELS: tuple = ()


def _activity_label_table() -> tuple:
    """Built lazily so ``re`` stays out of this module's import-time path."""
    global _ACTIVITY_LABELS
    if _ACTIVITY_LABELS:
        return _ACTIVITY_LABELS
    import re

    zh: tuple[tuple[str, str], ...] = (
        (r"^initializing$", "正在初始化"),
        (r"^starting new turn$", "正在开始新一轮"),
        (r"^starting API call #(\d+)$", r"正在发起第 \1 次模型调用"),
        (r"^API call #(\d+) completed$", r"第 \1 次模型调用已完成"),
        (r"^receiving stream response$", "正在接收模型回复"),
        (r"^waiting for provider response \(streaming\)$", "正在等待模型回复"),
        (r"^waiting for non-streaming API response$", "正在等待模型回复"),
        (r"^local model loading$", "正在加载本地模型"),
        (r"^waiting for stream response \(([\d.]+)s, ([^)]*)\)$", r"已等待模型回复 \1 秒（\2）"),
        (r"^waiting for provider response \(([\d.]+)s, ([^)]*)\)$", r"已等待模型回复 \1 秒（\2）"),
        (r"^stale stream detected after (\d+)s, reconnecting$", r"模型连接中断 \1 秒，正在重连"),
        (r"^stream retry (\d+)/(\d+) after (\S+)$", r"模型连接重试 \1/\2（\3）"),
        (r"^API error recovery \(attempt (\d+)/(\d+)\)$", r"模型报错，正在重试 \1/\2"),
        (r"^executing tool: (.+)$", r"正在运行：\1"),
        (r"^executing (\d+) tools concurrently: (.+)$", r"正在并行运行 \1 个工具：\2"),
        (r"^sequential tool running \(([\d.]+)s\): (.+)$", r"正在运行 \2（已 \1 秒）"),
        (r"^tool completed: (.+?) \(([\d.]+)s\)(.*)$", r"工具 \1 已完成（\2 秒）\3"),
        (r"^tool results posted, continuing iteration #(\d+)$", r"工具结果已回填，继续第 \1 轮"),
        (r"^context compression started$", "正在压缩上下文"),
        (r"^compressing context$", "正在压缩上下文"),
        (r"^turn end logged$", "本轮已结束"),
        (r"^codex stream killed after .*$", "模型连接已中断，正在恢复"),
        (r"^stale non-streaming call killed after (\d+)s$", r"模型调用超时 \1 秒，正在重试"),
    )
    _ACTIVITY_LABELS = tuple((re.compile(pattern), repl) for pattern, repl in zh)
    return _ACTIVITY_LABELS


def localized_activity_label(description: Optional[str]) -> str:
    """Render one engine activity stamp as a user-facing line.

    Canonical English is kept for every non-Chinese locale (and for unmatched stamps), so this
    never hides an unknown state — it only replaces jargon with plain language.
    """
    text = (description or "").strip()
    if not text:
        return ""
    try:
        if not str(get_language()).lower().startswith("zh"):
            return text
        import re

        # A raw tool id renders through the curated verb catalog ("terminal" -> "运行命令"),
        # so a heartbeat names what the agent is doing, not the internal tool name.
        _tool_match = re.match(r"^executing tool: (.+)$", text)
        if _tool_match:
            _tool = _tool_match.group(1).strip()
            _verb = t(f"display.verb.{_tool}")
            return f"正在{_verb}" if _verb and not _verb.startswith("display.verb.") else f"正在运行：{_tool}"

        for pattern, replacement in _activity_label_table():
            match = pattern.match(text)
            if match:
                return match.expand(replacement)
    except Exception:  # a display helper must never break a turn
        return text
    return text


def normalize_activity_provenance(provenance: Optional[ActivityProvenance | str]) -> ActivityProvenance:
    """Return a known provenance, or ``UNKNOWN`` when unset/unrecognized."""
    if isinstance(provenance, ActivityProvenance):
        return provenance
    try:
        return ActivityProvenance((provenance or "").strip())
    except ValueError:
        return ActivityProvenance.UNKNOWN


def format_iteration_progress(api_call_count: Any, max_iterations: Any) -> str:
    """``iteration N/M`` for user-facing status lines, or ``iteration N`` when the cap is unbounded.

    ``AIAgent.max_iterations`` defaults to ``sys.maxsize`` (unlimited), so printing the pair verbatim
    shows ``iteration 3/9223372036854775807`` in busy acks, heartbeats and timeout diagnostics (#102806).
    """
    try:
        cap = int(max_iterations)
    except (TypeError, ValueError):
        cap = sys.maxsize
    if cap >= sys.maxsize:
        return t("display.iteration_progress.unbounded", n=api_call_count)
    return t("display.iteration_progress.bounded", n=api_call_count, max=cap)


def reset_session_activity_persist_window(agent: Any) -> None:
    """Clear the persist rate-limit so the next stamp writes through (terminal compression labels must not stick on mid-compress text)."""
    with suppress(Exception):
        agent._session_activity_last_persist_mono = 0.0


def build_activity_snapshot(
    *,
    last_activity_at: Optional[float],
    last_activity_description: Optional[str],
    last_activity_provenance: Optional[ActivityProvenance | str] = None,
    now: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the shared activity snapshot (plus optional caller extras)."""
    when = float(last_activity_at) if last_activity_at is not None else None
    clock = float(now if now is not None else time.time())
    desc = bound_activity_description(last_activity_description)
    prov = normalize_activity_provenance(last_activity_provenance).value
    return {
        "last_activity_at": when,
        "last_activity_description": desc,
        "last_activity_provenance": prov,
        "seconds_since_activity": round(clock - when, 1) if when is not None else None,
        # Short aliases used by existing gateway/delegate readers.
        "last_activity_ts": when, "last_activity_desc": desc, "description": desc, "provenance": prov,
        **(extra or {}),
    }


class AwakeIdleMeter:
    """Removes host sleep from a polling watchdog's idle readings.

    Idle time is measured on the wall clock, which keeps running while the host sleeps, so a laptop
    that sleeps 15 minutes mid-turn wakes up "idle" for 15 minutes and the first poll after resume
    kills a turn that never had a chance to run. ``time.monotonic()`` pauses during sleep on macOS
    (``mach_absolute_time``) and Linux (``CLOCK_MONOTONIC``), so the two clocks drift apart by the
    time spent asleep. Call :meth:`measure` once per poll with the wall-clock idle seconds.
    """

    def __init__(self) -> None:
        self._wall, self._mono = time.time(), time.monotonic()
        self._asleep_s = 0.0

    def measure(self, idle_s: float) -> float:
        """Return *idle_s* minus the sleep observed since the agent's last activity."""
        wall, mono = time.time(), time.monotonic()
        slept_s = max(0.0, (wall - self._wall) - (mono - self._mono))
        # Activity since the previous sample (it is younger than the sampling interval) makes all
        # earlier credit predate it. A rarely sampled meter (stale-agent eviction samples only
        # when a message arrives) would otherwise keep charging old sleep against new idle time.
        if idle_s < wall - self._wall:
            self._asleep_s = 0.0
        self._asleep_s += slept_s
        self._wall, self._mono = wall, mono
        # The idle sample may predate sleep observed by these clock reads. Keep that new
        # credit until the next poll; only older credit can be trimmed by renewed activity.
        self._asleep_s = max(0.0, min(self._asleep_s, idle_s + slept_s))
        return max(0.0, idle_s - self._asleep_s)
