# SPDX-License-Identifier: Apache-2.0
"""Tests for the thread model — attribution, temperature, merge, eviction.

Exercises the read-time-temperature + actor-writes-focus invariants
that separate the thread model from its focus-stack predecessor:

* Foreground never gets revived by a keyword-sparse follow-up if it's
  already cold by age.
* Anaphora + generic-object messages never spawn a pseudo-thread.
* Open commitments pin thread temperature regardless of age.
* Cold, uncommitted background threads evict when over capacity;
  foreground never does.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from agent import threads
from agent.keywords import extract_keywords, extract_keywords_debug
from agent.threads import (
    AttributionResult,
    Commitment,
    Thread,
    attribute_user_message,
    build_thread_view,
    classify_reference,
    close_commitment,
    describe_thread,
    ensure_thread_state,
    evict_cold_threads,
    get_foreground_thread,
    is_indexical_progress_query,
    is_likely_one_off_leaf,
    make_thread,
    merge_threads,
    migrate_focus_stack_to_threads,
    open_commitment,
    strip_message_envelope,
    thread_temperature,
    touch_commitment_thread,
    touch_thread,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _fresh_state() -> dict[str, Any]:
    return {}


def _install_thread(
    state: dict, *, topic: list[str], hours_ago: float = 0.0
) -> Thread:
    """Insert a thread whose ``last_event_at`` is ``hours_ago`` in the past.

    Uses the module's own timestamp shape (ISO w/ ``Z`` suffix); the
    read-time temperature function parses back through ``_parse_iso``.
    """
    ts = ensure_thread_state(state)
    thread = make_thread(topic, tick=0, signature=topic)
    if hours_ago > 0:
        past_ms = time.time() * 1000 - hours_ago * 3600 * 1000
        # ISO Z-form to match the module's default output.
        from datetime import datetime, timezone

        stamp = (
            datetime.fromtimestamp(past_ms / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        thread.last_event_at = stamp
        thread.created_at = stamp
    ts["threads"].append(thread)
    return thread


# ── keywords.py smoke ───────────────────────────────────────────────


def test_keywords_extracts_chinese_and_english() -> None:
    kws = extract_keywords("帮我修复 heartbeat 的 tick 循环 bug", 6)
    assert "heartbeat" in kws
    assert any(k in {"修复", "循环"} for k in kws)


def test_keywords_debug_reveals_filter_stage() -> None:
    dbg = extract_keywords_debug("这个东西怎么样", 4)
    # Filter stage drops noise-only ngrams; final list can be empty.
    assert isinstance(dbg["raw"], list) and dbg["raw"]
    assert isinstance(dbg["filtered"], list)
    assert isinstance(dbg["final"], list)


def test_keywords_ignores_relative_time_words() -> None:
    # "昨天" is a temporal-parser concern, not a topic keyword.
    kws = extract_keywords("昨天 今天 明天", 6)
    assert "昨天" not in kws
    assert "今天" not in kws


# ── Envelope + one-off / indexical detectors ────────────────────────


def test_strip_envelope_removes_channel_prefix() -> None:
    body = strip_message_envelope(
        "[ID:abc] 2026-07-17T10:00:00Z [telegram] 帮我看看那个 PR"
    )
    assert body == "帮我看看那个 PR"


def test_strip_envelope_drops_tick_messages() -> None:
    assert strip_message_envelope("TICK 12345") == ""


def test_is_likely_one_off_leaf_short_weather_query() -> None:
    assert is_likely_one_off_leaf("今天天气怎么样") is True
    assert is_likely_one_off_leaf("请你分析这个火山 API 的响应") is False


def test_is_indexical_progress_query_only_for_short_form() -> None:
    assert is_indexical_progress_query("那个搞定了吗") is True
    assert (
        is_indexical_progress_query(
            "这个进展要详细的报告请给我完整的分析和结论以及可能的问题"
        )
        is False
    )


# ── Reference classifier ────────────────────────────────────────────


def test_classify_reference_anaphora_recent() -> None:
    r = classify_reference("打开那个网页")
    assert r.kind == "anaphora-recent"


def test_classify_reference_precise_callback() -> None:
    r = classify_reference("那个开源天梯榜怎么排的")
    assert r.kind == "precise-callback"
    assert r.referent_kws  # non-empty referent


def test_classify_reference_none_for_substantive_topic() -> None:
    r = classify_reference("我们把心跳循环的重入锁改成 asyncio.Lock")
    assert r.kind == "none"


# ── Attribution ─────────────────────────────────────────────────────


def test_attribution_creates_thread_from_substantive_message() -> None:
    state = _fresh_state()
    res = attribute_user_message(state, "帮我实现心跳循环的重入锁")
    assert res.event == "created"
    assert res.thread is not None
    # Foreground is now the new thread.
    assert get_foreground_thread(state) is res.thread


def test_attribution_continues_foreground_on_keyword_overlap() -> None:
    state = _fresh_state()
    r1 = attribute_user_message(state, "帮我实现心跳循环的重入锁")
    r2 = attribute_user_message(state, "心跳循环的锁改用 asyncio.Lock")
    assert r2.event == "continued"
    assert r2.thread is r1.thread


def test_attribution_noop_for_one_off_leaf() -> None:
    state = _fresh_state()
    r = attribute_user_message(state, "今天天气怎么样")
    assert r.event == "noop"


def test_attribution_anaphora_recent_never_creates_new_thread() -> None:
    """规律①: 'open that page' must not spawn a pseudo browser-thread."""
    state = _fresh_state()
    attribute_user_message(state, "帮我把开源天梯榜的实现细节讲清楚")
    r = attribute_user_message(state, "打开那个网页看看")
    # Should continue foreground (the leaderboard thread), NOT create a
    # new browser/webpage thread.
    assert r.event == "continued"
    assert r.via == "anaphora-recent"


def test_attribution_ignores_cold_foreground_for_sparse_short_message() -> None:
    state = _fresh_state()
    # Foreground that hasn't been touched in 72h → cold.
    _install_thread(state, topic=["旧话题", "冷线索"], hours_ago=72.0)
    ts = ensure_thread_state(state)
    ts["foreground_id"] = ts["threads"][0].id
    r = attribute_user_message(state, "好啊")  # sparse, only 2 chars
    # Sparse short msg + cold foreground → noop, not revival.
    assert r.event == "noop"


def test_attribution_resumes_background_on_strong_overlap() -> None:
    state = _fresh_state()
    r1 = attribute_user_message(state, "帮我看看心跳循环的重入锁")
    r2 = attribute_user_message(state, "另外的开源天梯榜排名怎么算")
    assert r2.event == "created"  # topic shift
    # Now come back to heartbeat with two clear keywords.
    r3 = attribute_user_message(state, "回到心跳循环重入锁那件事")
    assert r3.event == "resumed"
    assert r3.thread is r1.thread


def test_attribution_indexical_progress_routes_to_open_commitment() -> None:
    state = _fresh_state()
    r1 = attribute_user_message(state, "帮我实现心跳循环的重入锁")
    assert r1.thread is not None
    open_commitment(state, text="做心跳循环重入锁的事", thread_id=r1.thread.id)
    # New topic starts…
    attribute_user_message(state, "另外查一下开源天梯榜的分数是怎么算的")
    # …and a bare "how's it going" should point back to the commitment.
    r3 = attribute_user_message(state, "那个搞定了吗")
    assert r3.event == "resumed"
    assert r3.via == "commitment"
    assert r3.thread is r1.thread


# ── Temperature ─────────────────────────────────────────────────────


def test_temperature_foreground_beats_age() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["a", "b"], hours_ago=200.0)
    ensure_thread_state(state)["foreground_id"] = t.id
    assert thread_temperature(state, t) == "foreground"


def test_temperature_warm_within_window() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["a", "b"], hours_ago=3.0)
    assert thread_temperature(state, t) == "warm"


def test_temperature_cool_within_48h() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["a", "b"], hours_ago=12.0)
    assert thread_temperature(state, t) == "cool"


def test_temperature_cold_after_48h() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["a", "b"], hours_ago=72.0)
    assert thread_temperature(state, t) == "cold"


def test_temperature_open_commitment_pins_warm_forever() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["长期任务"], hours_ago=200.0)
    ensure_thread_state(state)["foreground_id"] = t.id
    open_commitment(state, text="长期任务", thread_id=t.id)
    ensure_thread_state(state)["foreground_id"] = None  # drop foreground
    assert thread_temperature(state, t) == "warm"


# ── Commitments ─────────────────────────────────────────────────────


def test_open_commitment_creates_thread_when_no_foreground() -> None:
    state = _fresh_state()
    c = open_commitment(state, text="做心跳循环那个 PR")
    assert isinstance(c, Commitment)
    # A thread got created and placed as foreground.
    fg = get_foreground_thread(state)
    assert fg is not None
    assert c.thread_id == fg.id


def test_open_commitment_singleton_per_thread() -> None:
    state = _fresh_state()
    c1 = open_commitment(state, text="做 PR")
    c2 = open_commitment(state, text="补测试到 PR", thread_id=c1.thread_id)
    # Same commitment id, text updated in place.
    assert c1.id == c2.id
    assert "补测试" in c2.text


def test_close_commitment_marks_done_and_stops_pinning_temperature() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["长期任务"], hours_ago=200.0)
    ensure_thread_state(state)["foreground_id"] = t.id
    open_commitment(state, text="长期任务", thread_id=t.id)
    # open_commitment touches the thread; roll last_event_at back so
    # the age-based check has something to bite on after closing.
    _install_thread  # keep for readability
    from datetime import datetime, timezone

    past_ms = time.time() * 1000 - 200 * 3600 * 1000
    t.last_event_at = (
        datetime.fromtimestamp(past_ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    ensure_thread_state(state)["foreground_id"] = None
    assert thread_temperature(state, t) == "warm"  # commitment pins it
    close_commitment(state, thread_id=t.id, status="done")
    # No longer pinned — falls back to age-based cold.
    assert thread_temperature(state, t) == "cold"


def test_touch_commitment_thread_falls_back_to_foreground() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["a", "b"], hours_ago=0.0)
    ensure_thread_state(state)["foreground_id"] = t.id
    old_ts = t.last_event_at
    time.sleep(0.005)  # ensure timestamp advances
    touch_commitment_thread(state, tick=42)
    assert t.last_event_at != old_ts
    assert t.last_event_tick == 42


def test_touch_thread_bumps_hit_count() -> None:
    state = _fresh_state()
    t = _install_thread(state, topic=["a"], hours_ago=0.0)
    initial = t.hit_count
    touch_thread(state, t.id, tick=5)
    assert t.hit_count == initial + 1


# ── Merge + eviction ────────────────────────────────────────────────


def test_merge_threads_folds_source_into_target() -> None:
    state = _fresh_state()
    a = _install_thread(state, topic=["a", "b"], hours_ago=0.0)
    b = _install_thread(state, topic=["a", "c"], hours_ago=0.0)
    a.conclusions = ["搞定 A 的第一部分"]
    b.conclusions = ["搞定 B 的第一部分"]
    ensure_thread_state(state)["foreground_id"] = b.id

    merged = merge_threads(state, source_id=a.id, target_id=b.id)
    assert merged is b
    assert set(b.topic) >= {"a"}
    assert "搞定 A 的第一部分" in b.conclusions
    assert "搞定 B 的第一部分" in b.conclusions
    # Foreground survives.
    assert ensure_thread_state(state)["foreground_id"] == b.id
    # Source gone.
    assert a not in ensure_thread_state(state)["threads"]


def test_merge_repoints_commitments_when_foreground_was_source() -> None:
    state = _fresh_state()
    a = _install_thread(state, topic=["a"], hours_ago=0.0)
    b = _install_thread(state, topic=["b"], hours_ago=0.0)
    ensure_thread_state(state)["foreground_id"] = a.id
    c = open_commitment(state, text="task on A", thread_id=a.id)

    merge_threads(state, source_id=a.id, target_id=b.id)
    # Commitment now points to b.
    assert c.thread_id == b.id
    # Foreground pointer moved to b.
    assert ensure_thread_state(state)["foreground_id"] == b.id


def test_evict_cold_threads_only_removes_cold_uncommitted() -> None:
    state = _fresh_state()
    # Over capacity with a mix of temperatures.
    ts = ensure_thread_state(state)
    warm = _install_thread(state, topic=["warm"], hours_ago=1.0)
    cold_but_committed = _install_thread(
        state, topic=["committed-old"], hours_ago=200.0
    )
    open_commitment(
        state, text="do committed-old", thread_id=cold_but_committed.id
    )
    cold_evictable = [
        _install_thread(state, topic=[f"old-{i}"], hours_ago=200.0)
        for i in range(threads.MAX_THREADS_IN_MEMORY)
    ]
    ts["foreground_id"] = warm.id

    evicted = evict_cold_threads(state)
    # Some cold, uncommitted threads got dropped.
    assert evicted, "expected at least one eviction over the cap"
    evicted_ids = {t.id for t in evicted}
    assert warm.id not in evicted_ids
    assert cold_but_committed.id not in evicted_ids
    # All evicted came from the plain-cold pool.
    plain_cold_ids = {t.id for t in cold_evictable}
    assert evicted_ids.issubset(plain_cold_ids)


# ── Injection view + describe ───────────────────────────────────────


def test_build_thread_view_caps_warm_background_at_configured_limit() -> None:
    state = _fresh_state()
    fg = _install_thread(state, topic=["fg"], hours_ago=0.0)
    ensure_thread_state(state)["foreground_id"] = fg.id
    for i in range(threads.MAX_WARM_INJECTED + 3):
        _install_thread(state, topic=[f"warm-{i}"], hours_ago=0.5)
    view = build_thread_view(state)
    assert view["foreground"] is fg
    assert len(view["background"]) == threads.MAX_WARM_INJECTED


def test_describe_thread_prefers_last_conclusion() -> None:
    t = make_thread(["a", "b"], label="心跳")
    assert describe_thread(t) == "心跳"
    t.conclusions = ["搞定 A"]
    assert describe_thread(t) == "心跳 — 搞定 A"


# ── Migration ───────────────────────────────────────────────────────


def test_migrate_focus_stack_produces_ordered_threads() -> None:
    result = migrate_focus_stack_to_threads(
        [
            {"topic": ["旧一"], "conclusions": ["a"], "hit_count": 5},
            {"topic": ["旧二"], "startedAt": "2026-06-01T00:00:00Z"},
        ],
        tick=10,
    )
    assert len(result["threads"]) == 2
    # Stack top becomes foreground.
    assert result["foreground_id"] == result["threads"][-1].id
    assert result["commitments"] == []


# ── thread_classifier ───────────────────────────────────────────────


class _FakeLLM:
    def __init__(self, payload: Any):
        self._payload = payload

    async def __call__(self, **kwargs) -> Any:
        return self._payload


@pytest.mark.asyncio
async def test_classifier_parses_verdict() -> None:
    from agent.thread_classifier import classify_thread_attribution

    call_llm = _FakeLLM(
        '```json\n{"verdict":"same","label":"心跳循环","topic":["心跳","循环","重入锁"]}\n```'
    )
    out = await classify_thread_attribution(
        call_llm=call_llm,
        new_message="接着做心跳循环重入锁",
        candidate_thread=make_thread(["心跳", "循环"]),
        created_topic=["心跳", "循环", "重入锁"],
    )
    assert out is not None
    assert out["verdict"] == "same"
    assert out["label"] == "心跳循环"
    assert out["topic"] == ["心跳", "循环", "重入锁"]


@pytest.mark.asyncio
async def test_classifier_returns_none_on_garbage_output() -> None:
    from agent.thread_classifier import classify_thread_attribution

    call_llm = _FakeLLM("no json at all here")
    out = await classify_thread_attribution(
        call_llm=call_llm,
        new_message="接着做",
        candidate_thread=None,
        created_topic=["x"],
    )
    assert out is None


@pytest.mark.asyncio
async def test_classifier_returns_none_on_timeout() -> None:
    from agent.thread_classifier import classify_thread_attribution

    async def slow(**kwargs):  # noqa: ARG001
        import asyncio

        await asyncio.sleep(1.5)
        return '{"verdict":"same"}'

    out = await classify_thread_attribution(
        call_llm=slow,
        new_message="接着做",
        candidate_thread=None,
        created_topic=["x"],
        timeout_ms=50,
    )
    assert out is None


@pytest.mark.asyncio
async def test_classifier_rejects_verdict_outside_enum() -> None:
    from agent.thread_classifier import classify_thread_attribution

    call_llm = _FakeLLM('{"verdict":"maybe","label":"x","topic":["a"]}')
    out = await classify_thread_attribution(
        call_llm=call_llm,
        new_message="msg",
        candidate_thread=None,
        created_topic=[],
    )
    assert out is None
