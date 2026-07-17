# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/memory/threads.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Thread model — the successor to focus-stack in a dynamic memory pool.

Threads replace the classic focus-stack with three read-time
temperature bands (foreground / warm / cool / cold) driven by an
elapsed-wall-clock function. Key invariants — worth internalising
before touching this module:

* **Epistemology.** Foreground is *written by the actor*
  (``touch_commitment_thread`` / ``open_commitment``); no observer
  ever infers focus.
* **Ontology.** Multiple concurrent threads plus one foreground
  pointer — no stack, no pop. A foreground switch simply moves the
  old thread to background.
* **Decision theory.** Forgetting is a *read-time pure function*
  (:func:`thread_temperature`); no write-time state mutation ever
  drops information. Thread data is append-only.

This module does *not* own persistence, LLM calls, or prompt
rendering — it emits events and exposes queries that the surrounding
system persists / classifies / renders.

Ported from BaiLongma's ``threads.js`` (MIT).
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Optional

from .keywords import extract_keywords


# ── Tuning constants (wall-clock, not ticks — tick intervals differ
# 40x between task and idle modes and are not a valid time unit). ───

WARM_WINDOW_MS = 6 * 60 * 60 * 1000  # 6h active → warm
COOL_WINDOW_MS = 48 * 60 * 60 * 1000  # 48h → cool; older → cold

# Injection budget: how many warm-thread one-line summaries prompt.py
# is allowed to inject. Bounds the injection view, not thread survival.
MAX_WARM_INJECTED = 3

# In-memory cap. Overflowing threads that are cold AND have no open
# commitment are evicted from memory (still in the DB).
MAX_THREADS_IN_MEMORY = 12

# Per-thread rolling conclusion cap.
THREAD_CONCLUSIONS_LIMIT = 5

TOPIC_KEYWORDS_LIMIT = 3
KEYWORD_EXTRACT_BUDGET = 12
MIN_KEYWORDS_FOR_THREAD = 3
MIN_MESSAGE_LENGTH = 4

# "Signature": the wider keyword set used for overlap matching. A
# thread's display topic is a shorter prefix of its signature.
SIGNATURE_LIMIT = 8


# The n-gram extractor happily proposes junk fillers like "这个" and
# "帮我" — high-frequency across all topics, pure noise for overlap
# matching. Filter them out at the attribution boundary; the display
# topic ends up cleaner as a bonus.
_NOISE_TOKEN_RE = re.compile(
    r"^(这个|那个|什么|怎么|为什|可以|我们|你们|他们|帮我|给我|一下|一个|"
    r"继续|部分|现在|今天|明天|昨天|晚上|早上|然后|还是|就是|但是|因为|"
    r"所以|如果|这样|那样|的话|时候|问题|事情|东西|网页|网站|网址|页面|"
    r"链接|地址|文件|文档|内容)"
)


def _filter_noise_tokens(kws: Iterable[str]) -> list[str]:
    out: list[str] = []
    for k in kws or []:
        t = str(k or "").strip()
        if not t:
            continue
        # English/digit tokens: length >= 3 threshold (short acronyms
        # are noise more often than not).
        if re.fullmatch(r"[a-zA-Z0-9_-]+", t):
            if len(t) >= 3:
                out.append(t)
            continue
        if _NOISE_TOKEN_RE.match(t):
            continue
        if len(t) >= 2:
            out.append(t)
    return out


def _extract_attribution_keywords(text: str) -> list[str]:
    """Single entry point used by both message-side attribution and
    thread-signature extraction — matching stays symmetric.
    """
    return _filter_noise_tokens(extract_keywords(str(text or ""), KEYWORD_EXTRACT_BUDGET))


# Asymmetric switching thresholds (DynamicMemoryPool.md 8.5):
#   Foreground continuation is cheap — overlap ≥ 1 counts.
#   Background resume is expensive — overlap ≥ 2 counts.
# The old focus-stack era learned the hard way that one-keyword
# coincidences flip "returned" to the wrong stack frame.
FOREGROUND_OVERLAP_MIN = 1
BACKGROUND_RESUME_OVERLAP_MIN = 2


# One-off leaves that should not open a thread.
_ONE_OFF_LEAF_RE = re.compile(
    r"天气|气温|温度|下雨|下雪|空气质量|AQI|几点|几号|星期几|汇率|股价|"
    r"热搜|新闻|在吗|早上好|晚上好|谢谢|收到",
    re.IGNORECASE,
)

_SUSTAINED_RE = re.compile(
    r"分析|优化|修复|实现|修改|设计|写|做|排查|调试|构建|部署|项目|"
    r"代码|文件|机制|方案|测试|review|debug|fix|implement|build",
    re.IGNORECASE,
)

# Indexical progress queries — carry no topic word, but the sentence
# form itself pins the referent to "the open commitment I owe you".
_INDEXICAL_PROGRESS_RE = re.compile(
    r"(怎么样|咋样|如何了|进度|进展|搞定|好了吗|好了么|好了没|"
    r"完成了吗|完成了没|弄完|做完|干完|干得|干的|还在弄|还在做|"
    r"顺利|卡住|到哪|哪一步)"
)


def is_likely_one_off_leaf(body: str) -> bool:
    text = str(body or "").strip()
    if not text:
        return False
    if _SUSTAINED_RE.search(text):
        return False
    if re.fullmatch(
        r"(hello|hi|hey|在吗|早上好|晚上好|谢谢|收到)", text, re.IGNORECASE
    ):
        return True
    return len(text) <= 40 and bool(_ONE_OFF_LEAF_RE.search(text))


def is_indexical_progress_query(body: str) -> bool:
    """Indexical progress queries are naturally short. A longer
    sentence that happens to mention "进展" is almost certainly a
    substantive request, not a bare progress check.
    """
    text = str(body or "").strip()
    if not text or len(text) > 25:
        return False
    return bool(_INDEXICAL_PROGRESS_RE.search(text))


# ── Anaphora + precise callback (治"打开那个网页"型话题漂移) ─────────
#
# Rule ① Bare anaphora / generic object: user says "this / that" or
#   only operates on a generic placeholder ("open that page"). The
#   referent lives in *the previous turn*, not this sentence. Do not
#   name a new thread; continue the foreground.
# Rule ② "That + concrete noun" — user is naming a distant concept
#   ("that open-source leaderboard"). The noun is itself the precise
#   referent, so relax the resume threshold from ≥ 2 to ≥ 1 on that
#   thread's signature.

_GENERIC_OBJECT_RE = re.compile(
    r"(网页|网站|网址|页面|那页|这页|链接|地址|玩意儿?|东西|文件|文档|内容)"
)
_OPERATION_VERB_RE = re.compile(
    r"(打开|关闭|关掉|启动|运行|播放|暂停|下载|搜索|显示|发送|点开|"
    r"访问|跳转|打开一?下|放一?下|查一?下|搜一?下|看一?下|念一?下|"
    r"读一?下|发一?下)"
)
_DEMONSTRATIVE_RE = re.compile(
    r"(这|那)(个|种|些|位|款|家|件|批|类|段|张|篇|份|首|部|台|项|套|"
    r"条|者|回|次)|它|他|她|刚才|刚刚|刚说|刚提|上面|前面|"
    r"之前(说|讲|提|的|那)|你?刚(说|讲|发|放|提)"
)


@dataclass
class ReferenceClassification:
    kind: Literal["none", "anaphora-recent", "precise-callback"]
    substantive: list[str] = field(default_factory=list)
    referent_kws: list[str] = field(default_factory=list)


def classify_reference(body: str) -> ReferenceClassification:
    """Classify a user message as none / anaphora-recent / precise-callback.

    * ``anaphora-recent`` — Rule ①: continue foreground, never
      new / switch.
    * ``precise-callback`` — Rule ②: try ``referent_kws`` against
      every thread; hits on background at ≥ 1 count as resume.
    * ``none`` — normal keyword attribution applies.

    ``substantive`` is what the sentence carries once operation verbs
    and generic-object placeholders are stripped — used by the
    "explicit-beats-implicit" guard so an anaphoric-looking sentence
    that *does* name a concrete other topic still lands on that topic.
    """
    text = str(body or "").strip()
    if not text:
        return ReferenceClassification(kind="none")

    has_demon = bool(_DEMONSTRATIVE_RE.search(text))
    acts_on_generic = bool(_GENERIC_OBJECT_RE.search(text)) and (
        bool(_OPERATION_VERB_RE.search(text)) or has_demon
    )
    substantive = [
        k
        for k in _extract_attribution_keywords(text)
        if not _OPERATION_VERB_RE.search(k) and not _GENERIC_OBJECT_RE.search(k)
    ]
    bare_operation = bool(_OPERATION_VERB_RE.search(text)) and not substantive
    if acts_on_generic or bare_operation:
        return ReferenceClassification(kind="anaphora-recent", substantive=substantive)
    if has_demon and not substantive:
        return ReferenceClassification(kind="anaphora-recent", substantive=substantive)
    if has_demon and substantive:
        return ReferenceClassification(
            kind="precise-callback",
            substantive=substantive,
            referent_kws=list(substantive),
        )
    return ReferenceClassification(kind="none", substantive=substantive)


# ── Message envelope + id + thread construction ────────────────────


_ENVELOPE_RE = re.compile(
    r"^\[[^\]]+\]\s*[\dTZ:+\-.]+\s*\[[^\]]*\]\s*(.*)$", re.DOTALL
)
_TICK_MESSAGE_RE = re.compile(r"^TICK\s", re.IGNORECASE)


def _is_tick_message(message: Any) -> bool:
    return isinstance(message, str) and bool(_TICK_MESSAGE_RE.match(message.strip()))


def strip_message_envelope(message: Any) -> str:
    """Peel ``[ID:xxx] timestamp [channel] body`` envelopes.

    TICK messages return an empty string — they are heartbeat markers,
    never user content.
    """
    if not message:
        return ""
    if _is_tick_message(message):
        return ""
    text = str(message)
    match = _ENVELOPE_RE.match(text)
    return match.group(1).strip() if match else text.strip()


_id_counter = 0


def _new_id(prefix: str) -> str:
    global _id_counter
    _id_counter = (_id_counter + 1) % 10000
    return (
        f"{prefix}_{int(time.time() * 1000):x}_"
        f"{_id_counter:x}"
        f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[float]:
    """Return ms since epoch for an ISO string, else None."""
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp() * 1000.0
    except (ValueError, TypeError):
        return None


@dataclass
class Thread:
    id: str
    topic: list[str]
    signature: list[str]
    label: str
    summary: str
    conclusions: list[str]
    status: str
    created_at: str
    last_event_at: str
    last_event_tick: int
    hit_count: int
    last_summary_at: str


def make_thread(
    topic: Iterable[str],
    *,
    tick: int = 0,
    label: str = "",
    signature: Optional[Iterable[str]] = None,
) -> Thread:
    now = _now_iso()
    topic_list = list(topic)[:TOPIC_KEYWORDS_LIMIT] if topic else []
    sig_list = (
        list(signature)[:SIGNATURE_LIMIT]
        if signature and any(True for _ in signature)
        else list(topic_list)
    )
    return Thread(
        id=_new_id("th"),
        topic=topic_list,
        signature=sig_list,
        label=label or "",
        summary="",
        conclusions=[],
        status="open",
        created_at=now,
        last_event_at=now,
        last_event_tick=tick,
        hit_count=1,
        last_summary_at=now,
    )


@dataclass
class Commitment:
    id: str
    thread_id: str
    text: str
    status: str
    channel: str
    created_at: str
    closed_at: Optional[str]


# ── ThreadState shape + accessors ──────────────────────────────────


State = MutableMapping[str, Any]


def ensure_thread_state(state: State) -> dict:
    ts = state.get("thread_state")
    if not isinstance(ts, dict):
        ts = {"threads": [], "foreground_id": None, "commitments": []}
        state["thread_state"] = ts
    ts.setdefault("threads", [])
    ts.setdefault("commitments", [])
    ts.setdefault("foreground_id", None)
    return ts


def get_foreground_thread(state: State) -> Optional[Thread]:
    ts = ensure_thread_state(state)
    fg = ts.get("foreground_id")
    if not fg:
        return None
    for t in ts["threads"]:
        if t.id == fg:
            return t
    return None


def get_thread_by_id(state: State, thread_id: Optional[str]) -> Optional[Thread]:
    if not thread_id:
        return None
    for t in ensure_thread_state(state)["threads"]:
        if t.id == thread_id:
            return t
    return None


def get_open_commitments(state: State) -> list[Commitment]:
    return [c for c in ensure_thread_state(state)["commitments"] if c.status == "open"]


def latest_open_commitment(
    state: State, *, channel: str = ""
) -> Optional[Commitment]:
    """Anchor for indexical progress queries. Prefers same-channel;
    within a set, ties break to the most recent (insertion-order-wins,
    matching push order in the JS original).
    """
    open_ = get_open_commitments(state)
    if not open_:
        return None

    def newest_of(candidates: list[Commitment]) -> Commitment:
        best = candidates[0]
        best_ts = _parse_iso(best.created_at) or 0.0
        for c in candidates:
            ts = _parse_iso(c.created_at) or 0.0
            if ts >= best_ts:
                best = c
                best_ts = ts
        return best

    if channel:
        same_channel = [c for c in open_ if c.channel == channel]
        if same_channel:
            return newest_of(same_channel)
    return newest_of(open_)


# ── Commitment lifecycle ───────────────────────────────────────────


def open_commitment(
    state: State,
    *,
    text: str,
    thread_id: Optional[str] = None,
    channel: str = "",
    tick: int = 0,
) -> Commitment:
    """'OK I'll do it' — the actor-writes-focus path.

    Attaches to the specified thread, else the current foreground, else
    creates a fresh thread. If an open commitment already exists on
    the same thread, updates its text in place (task is a singleton).
    """
    ts = ensure_thread_state(state)
    thread = (
        get_thread_by_id(state, thread_id)
        if thread_id
        else get_foreground_thread(state)
    )
    if not thread:
        kws = _extract_attribution_keywords(str(text or ""))
        thread = make_thread(
            kws[:TOPIC_KEYWORDS_LIMIT] if kws else ["任务"],
            tick=tick,
            signature=kws,
        )
        ts["threads"].append(thread)
        ts["foreground_id"] = thread.id

    for existing in ts["commitments"]:
        if existing.status == "open" and existing.thread_id == thread.id:
            existing.text = str(text or existing.text)
            return existing

    commitment = Commitment(
        id=_new_id("cm"),
        thread_id=thread.id,
        text=str(text or ""),
        status="open",
        channel=channel or "",
        created_at=_now_iso(),
        closed_at=None,
    )
    ts["commitments"].append(commitment)
    touch_thread(state, thread.id, tick=tick)
    return commitment


def close_commitment(
    state: State,
    *,
    thread_id: Optional[str] = None,
    commitment_id: Optional[str] = None,
    status: str = "done",
) -> Optional[Commitment]:
    """After close, the thread is no longer pinned — it cools naturally
    by ``last_event_at``. No mutation on any thread state; forgetting
    is a read-time function.
    """
    ts = ensure_thread_state(state)
    target: Optional[Commitment] = None
    if commitment_id:
        for c in ts["commitments"]:
            if c.id == commitment_id and c.status == "open":
                target = c
                break
    else:
        for c in ts["commitments"]:
            if c.status == "open" and (not thread_id or c.thread_id == thread_id):
                target = c
                break
    if not target:
        return None
    target.status = "cancelled" if status == "cancelled" else "done"
    target.closed_at = _now_iso()
    return target


def touch_thread(state: State, thread_id: str, *, tick: int = 0) -> bool:
    """Actor-writes-focus path #2: the agent doing tool calls counts as
    a foreground event. Silences the classic "starved while working"
    failure without any staleness heuristic.
    """
    thread = get_thread_by_id(state, thread_id)
    if not thread:
        return False
    thread.last_event_at = _now_iso()
    thread.last_event_tick = tick
    thread.hit_count += 1
    return True


def touch_commitment_thread(state: State, *, tick: int = 0) -> bool:
    ts = ensure_thread_state(state)
    open_ = latest_open_commitment(state)
    target_id = open_.thread_id if open_ else ts["foreground_id"]
    if not target_id:
        return False
    return touch_thread(state, target_id, tick=tick)


# ── Read-time temperature — never writes ───────────────────────────


Temperature = Literal["foreground", "warm", "cool", "cold"]


def thread_temperature(
    state: State,
    thread: Optional[Thread],
    *,
    now_ms: Optional[float] = None,
) -> Temperature:
    if thread is None:
        return "cold"
    if now_ms is None:
        now_ms = time.time() * 1000.0
    ts = ensure_thread_state(state)
    if thread.id == ts["foreground_id"]:
        return "foreground"
    # An open commitment pins temperature regardless of age.
    if any(c.status == "open" and c.thread_id == thread.id for c in ts["commitments"]):
        return "warm"
    last_ms = _parse_iso(thread.last_event_at) or _parse_iso(thread.created_at) or 0.0
    age = now_ms - last_ms
    if age < WARM_WINDOW_MS:
        return "warm"
    if age < COOL_WINDOW_MS:
        return "cool"
    return "cold"


def _is_thread_cold_by_age(
    state: State,
    thread: Optional[Thread],
    now_ms: Optional[float] = None,
) -> bool:
    """Cold-by-age guard used inside attribution rules. Independent of
    foreground shortcut but honours open-commitment pinning.
    """
    if not thread:
        return True
    ts = ensure_thread_state(state)
    if any(c.status == "open" and c.thread_id == thread.id for c in ts["commitments"]):
        return False
    if now_ms is None:
        now_ms = time.time() * 1000.0
    last_ms = _parse_iso(thread.last_event_at) or _parse_iso(thread.created_at) or 0.0
    age = now_ms - last_ms if last_ms else float("inf")
    return age >= COOL_WINDOW_MS


# ── Injection view — prompt.py's only entry point ──────────────────


def build_thread_view(state: State, *, now_ms: Optional[float] = None) -> dict:
    ts = ensure_thread_state(state)
    if now_ms is None:
        now_ms = time.time() * 1000.0
    foreground = get_foreground_thread(state)
    open_commitments = get_open_commitments(state)
    background_pool = [
        (t, thread_temperature(state, t, now_ms=now_ms))
        for t in ts["threads"]
        if t.id != ts["foreground_id"]
    ]
    warm_only = [x for x in background_pool if x[1] == "warm"]
    warm_only.sort(
        key=lambda x: -(_parse_iso(x[0].last_event_at) or 0.0),
    )
    background = [
        {"thread": thread, "temperature": temp}
        for thread, temp in warm_only[:MAX_WARM_INJECTED]
    ]
    return {
        "foreground": foreground,
        "foreground_commitment": (
            next(
                (c for c in open_commitments if c.thread_id == foreground.id),
                None,
            )
            if foreground
            else None
        ),
        "background": background,
        "open_commitments": open_commitments,
    }


# ── Overlap: intersection of message keywords with thread signature ─


def _overlap_count(thread: Optional[Thread], kws: Iterable[str]) -> int:
    if not thread:
        return 0
    signature_set = set(thread.signature or []) | set(thread.topic or [])
    if not signature_set:
        return 0
    return sum(1 for k in kws if k in signature_set)


# ── User-message attribution (the one place that *judges*) ─────────
#
# Return keys:
#   event: 'created' | 'continued' | 'resumed' | 'ambiguous' | 'noop'
#   thread: the attributed Thread or None
#   switched_from: previous foreground when 'resumed'/'created' flips it
#   via: reason tag ('commitment' / 'anaphora-recent' / 'callback' ...)
#   ambiguous_with: only when overlap=1 with a background thread


@dataclass
class AttributionResult:
    event: Literal["created", "continued", "resumed", "ambiguous", "noop"]
    thread: Optional[Thread]
    switched_from: Optional[Thread] = None
    via: Optional[str] = None
    ambiguous_with: Optional[Thread] = None


def attribute_user_message(
    state: State,
    message: Any,
    *,
    tick: int = 0,
    channel: str = "",
) -> AttributionResult:
    ts = ensure_thread_state(state)
    body = strip_message_envelope(message)
    if not body or len(body) < MIN_MESSAGE_LENGTH:
        return AttributionResult(event="noop", thread=None)
    if is_likely_one_off_leaf(body):
        return AttributionResult(event="noop", thread=None)

    kws = _extract_attribution_keywords(body)
    foreground = get_foreground_thread(state)

    # 1) Indexical progress query → thread of the latest open commitment.
    if is_indexical_progress_query(body):
        commitment = latest_open_commitment(state, channel=channel)
        if commitment:
            target = get_thread_by_id(state, commitment.thread_id)
            names_other = target is not None and any(
                t.id != target.id
                and _overlap_count(t, kws) >= BACKGROUND_RESUME_OVERLAP_MIN
                for t in ts["threads"]
            )
            if target and not names_other:
                switched_from = (
                    foreground if foreground and foreground.id != target.id else None
                )
                ts["foreground_id"] = target.id
                touch_thread(state, target.id, tick=tick)
                return AttributionResult(
                    event="resumed" if switched_from else "continued",
                    thread=target,
                    switched_from=switched_from,
                    via="commitment",
                )
        else:
            if foreground and not _is_thread_cold_by_age(state, foreground):
                touch_thread(state, foreground.id, tick=tick)
                return AttributionResult(
                    event="continued", thread=foreground
                )
            return AttributionResult(event="noop", thread=None)

    # 1.5) Reference classification (治话题漂移).
    ref = classify_reference(body)
    if ref.kind == "anaphora-recent":
        names_other = bool(ref.substantive) and any(
            (not foreground or t.id != foreground.id)
            and _overlap_count(t, ref.substantive) >= BACKGROUND_RESUME_OVERLAP_MIN
            for t in ts["threads"]
        )
        if not names_other and foreground and not _is_thread_cold_by_age(state, foreground):
            touch_thread(state, foreground.id, tick=tick)
            return AttributionResult(
                event="continued", thread=foreground, via="anaphora-recent"
            )
    elif ref.kind == "precise-callback" and ref.referent_kws:
        rkws = ref.referent_kws
        if foreground and _overlap_count(foreground, rkws) >= 1:
            touch_thread(state, foreground.id, tick=tick)
            return AttributionResult(
                event="continued", thread=foreground, via="callback"
            )
        cb_best: Optional[Thread] = None
        cb_overlap = 0
        for t in ts["threads"]:
            if foreground and t.id == foreground.id:
                continue
            n = _overlap_count(t, rkws)
            if n > cb_overlap:
                cb_best = t
                cb_overlap = n
        if cb_best and cb_overlap >= BACKGROUND_RESUME_OVERLAP_MIN:
            switched_from = (
                foreground if foreground and foreground.id != cb_best.id else None
            )
            ts["foreground_id"] = cb_best.id
            touch_thread(state, cb_best.id, tick=tick)
            return AttributionResult(
                event="resumed" if switched_from else "continued",
                thread=cb_best,
                switched_from=switched_from,
                via="callback",
            )
        # Fallback: a demonstrative sentence with no match anywhere is
        # almost never a truly new topic — continue foreground rather
        # than spawn a pseudo-thread.
        if foreground and not _is_thread_cold_by_age(state, foreground):
            touch_thread(state, foreground.id, tick=tick)
            return AttributionResult(
                event="continued", thread=foreground, via="callback-fallback"
            )

    # 2) Keyword-sparse short message: continue foreground (cheap,
    #    self-healing); guard against reviving a cold foreground.
    if len(kws) < MIN_KEYWORDS_FOR_THREAD:
        if foreground and not _is_thread_cold_by_age(state, foreground):
            touch_thread(state, foreground.id, tick=tick)
            return AttributionResult(event="continued", thread=foreground)
        return AttributionResult(event="noop", thread=None)

    # 3) Foreground overlap ≥ 1 → continued.
    if foreground and _overlap_count(foreground, kws) >= FOREGROUND_OVERLAP_MIN:
        touch_thread(state, foreground.id, tick=tick)
        return AttributionResult(event="continued", thread=foreground)

    # 4) Background: overlap ≥ 2 → resumed; = 1 → ambiguous candidate.
    best: Optional[Thread] = None
    best_overlap = 0
    for t in ts["threads"]:
        if foreground and t.id == foreground.id:
            continue
        n = _overlap_count(t, kws)
        if n > best_overlap:
            best = t
            best_overlap = n
    if best and best_overlap >= BACKGROUND_RESUME_OVERLAP_MIN:
        ts["foreground_id"] = best.id
        touch_thread(state, best.id, tick=tick)
        return AttributionResult(
            event="resumed", thread=best, switched_from=foreground
        )

    # 5) Genuinely new topic.
    created = make_thread(kws[:TOPIC_KEYWORDS_LIMIT], tick=tick, signature=kws)
    ts["threads"].append(created)
    switched_from = foreground
    ts["foreground_id"] = created.id
    evict_cold_threads(state)
    if best and best_overlap == 1:
        return AttributionResult(
            event="created",
            thread=created,
            switched_from=switched_from,
            ambiguous_with=best,
        )
    return AttributionResult(
        event="created", thread=created, switched_from=switched_from
    )


# ── Merge (arbitration-follow-up; always safe by construction) ────


def merge_threads(
    state: State, source_id: str, target_id: str
) -> Optional[Thread]:
    """Fold ``source`` into ``target``. Topic + signature unions,
    conclusions concat (bounded), summary preserved if target lacked
    one, commitments retargeted, hit counts summed, foreground
    pointer patched.

    Merge is always safe: threads have no stack invariant to preserve;
    the worst case is over- or under-merging by one topic word.
    """
    ts = ensure_thread_state(state)
    source = get_thread_by_id(state, source_id)
    target = get_thread_by_id(state, target_id)
    if not source or not target or source.id == target.id:
        return None

    topic_set = list(dict.fromkeys([*target.topic, *source.topic]))
    target.topic = topic_set[:TOPIC_KEYWORDS_LIMIT]
    sig_set = list(dict.fromkeys([*target.signature, *source.signature]))
    target.signature = sig_set[:SIGNATURE_LIMIT]

    for c in source.conclusions or []:
        if c not in target.conclusions:
            target.conclusions.append(c)
    while len(target.conclusions) > THREAD_CONCLUSIONS_LIMIT:
        target.conclusions.pop(0)

    if source.summary and not target.summary:
        target.summary = source.summary

    target.hit_count += source.hit_count or 0
    if (_parse_iso(source.last_event_at) or 0.0) > (
        _parse_iso(target.last_event_at) or 0.0
    ):
        target.last_event_at = source.last_event_at
        target.last_event_tick = source.last_event_tick

    for c in ts["commitments"]:
        if c.thread_id == source.id:
            c.thread_id = target.id

    ts["threads"] = [t for t in ts["threads"] if t.id != source.id]
    if ts["foreground_id"] == source.id:
        ts["foreground_id"] = target.id
    return target


def append_conclusion(thread: Thread, conclusion: str) -> None:
    """Add an incremental summary conclusion. Rolling cap; never
    replaces prior text.
    """
    if not thread or not conclusion:
        return
    text = str(conclusion).strip()
    if not text or text in thread.conclusions:
        return
    thread.conclusions.append(text)
    while len(thread.conclusions) > THREAD_CONCLUSIONS_LIMIT:
        thread.conclusions.pop(0)


def evict_cold_threads(
    state: State, *, now_ms: Optional[float] = None
) -> list[Thread]:
    """In-memory slim-down (not forgetting): overflow → drop threads
    that are (a) cold, (b) not foreground, (c) have no open commitment.
    Persistence layer still has them.
    """
    ts = ensure_thread_state(state)
    if len(ts["threads"]) <= MAX_THREADS_IN_MEMORY:
        return []
    if now_ms is None:
        now_ms = time.time() * 1000.0
    evictable = [
        t
        for t in ts["threads"]
        if t.id != ts["foreground_id"]
        and thread_temperature(state, t, now_ms=now_ms) == "cold"
    ]
    evictable.sort(key=lambda t: _parse_iso(t.last_event_at) or 0.0)
    excess = len(ts["threads"]) - MAX_THREADS_IN_MEMORY
    evicted = evictable[:excess]
    if not evicted:
        return []
    evicted_ids = {t.id for t in evicted}
    ts["threads"] = [t for t in ts["threads"] if t.id not in evicted_ids]
    return evicted


def describe_thread(thread: Optional[Thread]) -> str:
    if not thread:
        return ""
    label = thread.label or (",".join(thread.topic) if thread.topic else "")
    last_conclusion = thread.conclusions[-1] if thread.conclusions else ""
    return f"{label} — {last_conclusion}" if last_conclusion else label


def migrate_focus_stack_to_threads(
    focus_stack: list[Mapping[str, Any]], *, tick: int = 0
) -> dict:
    """One-time boot migration from the legacy focus-stack model.

    Stack top becomes foreground; the rest become background threads.
    Commitments cannot be recovered (the old model had no such
    concept), so the result's ``commitments`` list is empty.
    """
    threads: list[Thread] = []
    for frame in focus_stack or []:
        topic = frame.get("topic") if isinstance(frame, Mapping) else None
        if not isinstance(topic, list) or not topic:
            continue
        t = make_thread(topic, tick=tick, signature=list(topic))
        started_at = frame.get("startedAt") or frame.get("started_at")
        if started_at:
            t.created_at = started_at
            t.last_event_at = started_at
        t.last_event_tick = frame.get("lastSeenTick") or frame.get("last_seen_tick") or tick
        t.hit_count = frame.get("hitCount") or frame.get("hit_count") or 1
        conclusions = frame.get("conclusions") or []
        if isinstance(conclusions, list):
            t.conclusions = [
                str(c) for c in conclusions[-THREAD_CONCLUSIONS_LIMIT:]
            ]
        threads.append(t)
    return {
        "threads": threads,
        "foreground_id": threads[-1].id if threads else None,
        "commitments": [],
    }


__all__ = [
    "AttributionResult",
    "BACKGROUND_RESUME_OVERLAP_MIN",
    "COOL_WINDOW_MS",
    "Commitment",
    "FOREGROUND_OVERLAP_MIN",
    "MAX_THREADS_IN_MEMORY",
    "MAX_WARM_INJECTED",
    "MIN_KEYWORDS_FOR_THREAD",
    "MIN_MESSAGE_LENGTH",
    "ReferenceClassification",
    "SIGNATURE_LIMIT",
    "THREAD_CONCLUSIONS_LIMIT",
    "TOPIC_KEYWORDS_LIMIT",
    "Thread",
    "WARM_WINDOW_MS",
    "append_conclusion",
    "attribute_user_message",
    "build_thread_view",
    "classify_reference",
    "close_commitment",
    "describe_thread",
    "ensure_thread_state",
    "evict_cold_threads",
    "get_foreground_thread",
    "get_open_commitments",
    "get_thread_by_id",
    "is_indexical_progress_query",
    "is_likely_one_off_leaf",
    "latest_open_commitment",
    "make_thread",
    "merge_threads",
    "migrate_focus_stack_to_threads",
    "open_commitment",
    "strip_message_envelope",
    "thread_temperature",
    "touch_commitment_thread",
    "touch_thread",
]
