"""Engagement invariants: the daily rollup reports each closed day once per profile, a compressed
conversation is one session row with its whole volume, and turns-before-switch names the model left."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent.portal_tags import reset_conversation_context, set_conversation_context
from hermes_cli import lifecycle
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_engagement as engagement
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from tests.hermes_cli.test_relay_shared_metrics_runtime import (  # noqa: F401 - fixture
    _stored_values,
    direct_runtime,
)

PUBLIC = "anthropic/claude-sonnet-4"
OTHER = "openai/gpt-5"
DAY1 = datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc).timestamp()


def _flush() -> None:
    for runtime in list(relay_shared_metrics._RUNTIMES.values()):
        runtime.relay.subscribers.flush()


def _turn(session_id, task_id, *, platform="cli", model=PUBLIC, tools=0, clock=None, at=None, took=0.0):
    base = {"session_id": session_id, "task_id": task_id, "provider": "openrouter", "model": model}
    if clock is not None:
        clock["now"] = at
    lifecycle.invoke_hook("pre_llm_call", **base, platform=platform)
    lifecycle.invoke_hook("pre_api_request", **base, api_request_id=f"{task_id}-r")
    lifecycle.invoke_hook("post_api_request", **base, api_request_id=f"{task_id}-r",
                          usage={"prompt_tokens": 1_000, "output_tokens": 5}, context_length=200_000)
    for n in range(tools):
        call = {**base, "tool_call_id": f"{task_id}-c{n}", "tool_name": "todo", "turn_id": ""}
        lifecycle.invoke_hook("pre_tool_call", **call)
        lifecycle.invoke_hook("post_tool_call", **call, status="success")
    if clock is not None:
        clock["now"] = at + took
    relay_shared_metrics.finish_task_run(session_id=session_id, task_id=task_id, platform=platform,
                                         result={"completed": True})


@pytest.fixture
def clock(monkeypatch):
    state = {"now": DAY1}
    monkeypatch.setattr(engagement, "_now", lambda: state["now"])
    return state


def _rows(tmp_path, metric):
    return sorted((json.dumps(d, sort_keys=True), v) for d, v in _stored_values(tmp_path, metric))


def test_a_closed_day_reports_active_time_surfaces_and_primary_model_once(direct_runtime, tmp_path, clock):
    """Two surfaces on day 1 (gaps capped at 5 minutes); nothing is reported until the day closes,
    then exactly once, dated to that day, however many processes of the profile see the next day."""
    _turn("cli-1", "t1", clock=clock, at=DAY1, took=60)            # cli: 1 min
    _turn("cli-1", "t2", clock=clock, at=DAY1 + 180, took=3600)    # +2 min gap, +5 min capped run
    _turn("gw-1", "g1", platform="telegram", model=OTHER, clock=clock, at=DAY1 + 7200, took=30)
    lifecycle.finalize_session(session_id="cli-1")
    lifecycle.finalize_session(session_id="gw-1")
    _flush()
    assert not _stored_values(tmp_path, "hermes.engagement.day.count"), "emitted before the day closed"

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    other_process = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
    resource = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}
    clock["now"] = DAY1 + 86_400
    engagement.record(other_process, resource, surface="cli", route=None)
    _turn("cli-2", "t3", clock=clock, at=DAY1 + 86_400 + 5)
    _flush()

    assert _rows(tmp_path, "hermes.engagement.surface_day.count") == sorted([
        (json.dumps({"active_minutes_bucket": "5m_to_30m", "surface": "cli"}, sort_keys=True), 1),
        (json.dumps({"active_minutes_bucket": "lt_5m", "surface": "gateway"}, sort_keys=True), 1),
    ])
    assert _stored_values(tmp_path, "hermes.engagement.day.count") == [(
        {"active_minutes_bucket": "5m_to_30m", "active_profile_count_bucket": "1", "primary_model": PUBLIC,
         "primary_provider": "openrouter", "surfaces_used_count": "2"}, 1)]
    periods = {c["period_start"] for c in other_process.counter_snapshot() if c["metric_name"].startswith("hermes.engagement")}
    assert periods == {"2026-09-27"}


def test_internal_and_unattended_turns_are_not_engagement(direct_runtime, tmp_path, clock):
    """API-server and cron turns add no active time, surface, profile or primary model: a day of them
    alone reports nothing, and they never outvote the person's model."""
    _turn("api-1", "a1", platform="api_server", model=OTHER, clock=clock, at=DAY1)
    _turn("cron-1", "c1", platform="cron", model=OTHER, clock=clock, at=DAY1 + 300, took=7200)
    _turn("cli-1", "t0", clock=clock, at=DAY1 + 86_400)       # day 2 closes day 1
    _flush()
    assert not _stored_values(tmp_path, "hermes.engagement.day.count")
    for n in range(5):
        _turn("api-1", f"a{n + 2}", platform="api_server", model=OTHER, clock=clock, at=DAY1 + 86_400 + 60 * n)
    _turn("cli-1", "t1", clock=clock, at=DAY1 + 86_400 + 400)
    clock["now"] = DAY1 + 2 * 86_400
    _turn("cli-1", "t2", clock=clock, at=DAY1 + 2 * 86_400)
    _flush()
    [(day, _)] = _stored_values(tmp_path, "hermes.engagement.day.count")
    assert (day["primary_model"], day["surfaces_used_count"]) == (PUBLIC, "1")
    assert [d["surface"] for d, _ in _stored_values(tmp_path, "hermes.engagement.surface_day.count")] == ["cli"]


def test_a_late_interaction_never_reopens_a_closed_day():
    """An interaction sampled before midnight that lands after the day rolled folds into the newer day:
    the closed day is reported once, and the only close path is a later clock."""
    resource = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}
    midnight = int(datetime(2026, 9, 28, tzinfo=timezone.utc).timestamp() * 1000)
    state, _ = engagement.apply(None, now_ms=midnight - 3_600_000, resource=resource, surface="cli")
    state, closed = engagement.apply(state, now_ms=midnight + 500, resource=resource, surface="cli")
    assert [r[3] for r in closed] == ["2026-09-27", "2026-09-27"]
    state, late = engagement.apply(state, now_ms=midnight - 500, resource=resource, surface="cli")
    assert late == [] and state["day"] == "2026-09-28"
    _, next_day = engagement.apply(state, now_ms=midnight + 86_400_000, resource=resource, surface="cli")
    assert {r[3] for r in next_day} == {"2026-09-28"}


def test_collection_off_writes_no_engagement_state(direct_runtime, tmp_path, clock, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly", lambda: {})
    _turn("s1", "t1", clock=clock, at=DAY1)
    clock["now"] = DAY1 + 86_400
    _turn("s1", "t2", clock=clock, at=DAY1 + 86_400)
    lifecycle.finalize_session(session_id="s1")
    assert not (tmp_path / "hermes-home" / "telemetry").exists()


def test_a_compressed_conversation_is_one_session_row_with_its_whole_volume(direct_runtime, tmp_path):
    """Compression hands s1 off to s1c (the one seam that continues a conversation under a new id): one
    session row, turns / API calls / tool calls / messages summed over both segments, whatever order
    the surface closes the two ids in."""
    token = set_conversation_context("s1")
    try:
        _turn("s1", "t1", tools=2)
        _turn("s1", "t2", tools=1)
        relay_shared_metrics.rotate_segment("s1", "s1c")
        _turn("s1c", "t3", tools=0)
    finally:
        reset_conversation_context(token)
    lifecycle.finalize_session(session_id="s1")
    _flush()
    assert not _stored_values(tmp_path, "hermes.session.count"), "a segment reported on its own"
    lifecycle.finalize_session(session_id="s1c")
    _flush()

    [(row, value)] = _stored_values(tmp_path, "hermes.session.count")
    assert value == 1
    assert {k: row[k] for k in ("turn_count_bucket", "model_call_count_bucket", "tool_call_count_bucket",
                                "message_count_bucket", "user_turn_count_bucket")} == {
        "turn_count_bucket": "3_to_5", "model_call_count_bucket": "3_to_5",
        "tool_call_count_bucket": "3_to_5", "message_count_bucket": "6_to_10",
         "user_turn_count_bucket": "3_to_5"}


def test_only_compression_joins_segments_and_the_retired_segment_closes_at_hand_off(direct_runtime, tmp_path):
    """A reset / /new / /branch child shares the conversation root (the Portal attribution id) but is a
    new conversation: compressed g1 -> g1c, then two resets give 3 rows by the time the last one closes,
    with nothing left open. A compression mid-turn closes the old segment when that turn ends."""
    token = set_conversation_context("g1")
    try:
        _turn("g1", "t1", platform="telegram")
        relay_shared_metrics.rotate_segment("g1", "g1c")
        _turn("g1c", "t2", platform="telegram")
        relay_shared_metrics.close_session_run("g1c")
        _turn("g2", "t3", platform="telegram")
        relay_shared_metrics.close_session_run("g2")
        _turn("g3", "t4", platform="telegram")
        relay_shared_metrics.close_session_run("g3")
        lifecycle.invoke_hook("pre_llm_call", session_id="m1", task_id="t5", platform="cli")
        relay_shared_metrics.rotate_segment("m1", "m1c")
        lifecycle.finalize_session(session_id="m1c")  # tip closed while the compression turn still runs
        relay_shared_metrics.finish_task_run(session_id="m1", task_id="t5", platform="cli", result={"completed": True})
    finally:
        reset_conversation_context(token)
    _flush()
    rows = sorted((d["platform"], d["turn_count_bucket"], v) for d, v in _stored_values(tmp_path, "hermes.session.count"))
    assert rows == [("none", "1", 1), ("telegram", "1", 2), ("telegram", "2", 1)]
    runtime = next(iter(relay_shared_metrics._RUNTIMES.values()))
    assert not runtime._lineages and not runtime._lineage_of and not runtime._sessions


def test_compression_commit_hands_the_segment_off(monkeypatch):
    """The committed rotation (and a stale agent adopting the live tip) is the hand-off seam."""
    from types import SimpleNamespace

    from agent import conversation_compression

    calls = []
    monkeypatch.setattr(relay_shared_metrics, "rotate_segment", lambda old, new: calls.append((old, new)))
    agent = SimpleNamespace(context_compressor=SimpleNamespace(), platform="cli", _gateway_session_key=None)
    conversation_compression._notify_context_engine_compression_complete(agent, new_session_id="s1c", old_session_id="s1")
    assert calls == [("s1", "s1c")]


def test_switch_and_undo_right_after_a_rotation_reach_the_conversation(direct_runtime, tmp_path):
    """/model or /undo on the rotated-to id before it served a turn still sees the conversation."""
    from hermes_cli.observability.shared_metrics_events import record_model_switch

    _turn("s1", "t1")
    _turn("s1", "t2")
    relay_shared_metrics.rotate_segment("s1", "s1c")
    relay_shared_metrics.record_session_friction("undo", "s1c", {"provider": "openrouter", "model": PUBLIC})
    record_model_switch(from_provider="openrouter", to_provider="openrouter", surface="cli",
                        from_model=PUBLIC, session_id="s1c")
    _flush()
    assert _stored_values(tmp_path, "hermes.model_switch_after.count") == [
        ({"model": PUBLIC, "provider": "openrouter", "turns_before_switch_bucket": "2_to_3"}, 1)]
    [(wasted, _)] = _stored_values(tmp_path, "hermes.wasted_tokens.count")
    assert wasted["tokens_bucket"] != "unknown"


def test_a_background_review_fork_on_the_session_id_adds_no_session_volume(direct_runtime, tmp_path):
    """Review forks reuse the parent's session id and never fire pre_llm_call: not the user's turns."""
    _turn("s1", "t1")
    relay_shared_metrics.start_task_run(session_id="s1", task_id="review", platform="cli")
    base = {"session_id": "s1", "task_id": "review", "provider": "openrouter", "model": PUBLIC}
    lifecycle.invoke_hook("pre_api_request", **base, api_request_id="rv-r")
    lifecycle.invoke_hook("post_api_request", **base, api_request_id="rv-r", usage={"prompt_tokens": 9})
    relay_shared_metrics.finish_task_run(session_id="s1", task_id="review", platform="cli", result={"completed": True})
    lifecycle.finalize_session(session_id="s1")
    _flush()
    [(row, _)] = _stored_values(tmp_path, "hermes.session.count")
    assert (row["turn_count_bucket"], row["model_call_count_bucket"], row["message_count_bucket"],
            row["user_turn_count_bucket"]) == ("1", "1", "2", "1")


def test_turns_before_switch_ignore_failover_and_review_forks(direct_runtime, tmp_path):
    """The run counts the route each user turn was sent on: a turn that failed over to a fallback
    still served the selected model, and a background review fork on another model is not a turn."""
    from hermes_cli.observability.shared_metrics_events import record_model_switch

    _turn("s1", "t1")
    base = {"session_id": "s1", "task_id": "t2", "provider": "openrouter"}
    lifecycle.invoke_hook("pre_llm_call", **base, model=PUBLIC, platform="cli")
    lifecycle.invoke_hook("pre_api_request", **base, model=PUBLIC, api_request_id="t2-r")
    lifecycle.invoke_hook("api_request_error", **base, model=PUBLIC, api_request_id="t2-r")
    lifecycle.invoke_hook("pre_api_request", **base, model=OTHER, api_request_id="t2-fallback")
    lifecycle.invoke_hook("post_api_request", **base, model=OTHER, api_request_id="t2-fallback", usage={})
    relay_shared_metrics.finish_task_run(session_id="s1", task_id="t2", platform="cli", result={"completed": True})
    relay_shared_metrics.start_task_run(session_id="s1", task_id="review", platform="cli")
    lifecycle.invoke_hook("pre_api_request", session_id="s1", task_id="review", provider="openrouter",
                          model=OTHER, api_request_id="rv-r")
    relay_shared_metrics.finish_task_run(session_id="s1", task_id="review", platform="cli", result={"completed": True})
    record_model_switch(from_provider="openrouter", to_provider="openrouter", surface="cli",
                        from_model=PUBLIC, session_id="s1")
    _flush()
    assert _stored_values(tmp_path, "hermes.model_switch_after.count") == [
        ({"model": PUBLIC, "provider": "openrouter", "turns_before_switch_bucket": "2_to_3"}, 1)]


def test_turns_before_switch_count_the_model_left_across_rotation(direct_runtime, tmp_path):
    """/model counts how many turns the model being left served in the conversation (compression
    segments included); a second switch before any turn on the new model reports nothing."""
    from hermes_cli.observability.shared_metrics_events import record_model_switch

    token = set_conversation_context("s1")
    try:
        _turn("s1", "t1")
        _turn("s1c", "t2")
        _turn("s1c", "t3")
    finally:
        reset_conversation_context(token)
    record_model_switch(from_provider="openrouter", to_provider="openrouter", surface="cli",
                        from_model=PUBLIC, session_id="s1c")
    record_model_switch(from_provider="openrouter", to_provider="openrouter", surface="cli",
                        from_model=OTHER, session_id="s1c")
    _flush()
    assert _stored_values(tmp_path, "hermes.model_switch_after.count") == [
        ({"model": PUBLIC, "provider": "openrouter", "turns_before_switch_bucket": "2_to_3"}, 1)]


def _profile_store(home):
    root = home / "telemetry" / "shared_metrics"
    return SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")


def _day_rows(home):
    db = home / "telemetry" / "shared_metrics" / "metrics.sqlite3"
    if not db.exists():
        return []
    return [(c["period_start"], c["dimensions"]) for c in _profile_store(home).counter_snapshot()
            if c["metric_name"] == "hermes.engagement.day.count"]


@pytest.mark.parametrize("root_collects", [True, False])
def test_active_profiles_are_counted_once_per_host_day_on_the_root_profiles_row(tmp_path, clock, root_collects):
    """Profiles A -> B -> A on one day (any process, any runtime): the root (default) profile's day
    row reports 2 distinct active profiles, even though the root itself was idle; the profiles' own
    rows report 0 so no row claims the host total twice. A root with collection off is untouched."""
    root = tmp_path / "root"
    (root / "profiles").mkdir(parents=True)
    (root / "config.yaml").write_text(f"telemetry:\n  shared_metrics:\n    enabled: {str(root_collects).lower()}\n")
    a, b = root / "profiles" / "a", root / "profiles" / "b"
    resource = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}
    for home, at in ((a, 0), (b, 60), (a, 120), (a, 86_400)):
        clock["now"] = DAY1 + at
        engagement.record(_profile_store(home), resource, surface="cli", route=None)

    zero = {"active_minutes_bucket": "lt_5m", "active_profile_count_bucket": "0", "primary_model": "none",
            "primary_provider": "none", "surfaces_used_count": "1"}
    assert _day_rows(a) == [("2026-09-27", zero)]
    assert _day_rows(b) == []  # B has not seen a later day yet; its row will carry 0 too
    if root_collects:
        assert _day_rows(root) == [("2026-09-27", {
            "active_minutes_bucket": "0", "active_profile_count_bucket": "2", "primary_model": "none",
            "primary_provider": "none", "surfaces_used_count": "0"})]
    else:
        assert not (root / "telemetry").exists()


@pytest.mark.parametrize(("count", "bucket"), [(250, "101_to_250"), (251, "251_to_1000"), (1001, "gte_1001")])
def test_long_conversations_bucket_past_251(count, bucket):
    from hermes_cli.observability import shared_metrics_contract as contract

    assert contract.long_size_bucket(count) == bucket
    assert bucket in contract.LONG_SIZE_BUCKETS
