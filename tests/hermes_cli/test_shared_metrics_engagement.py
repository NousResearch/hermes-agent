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


def test_collection_off_writes_no_engagement_state(direct_runtime, tmp_path, clock, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly", lambda: {})
    _turn("s1", "t1", clock=clock, at=DAY1)
    clock["now"] = DAY1 + 86_400
    _turn("s1", "t2", clock=clock, at=DAY1 + 86_400)
    lifecycle.finalize_session(session_id="s1")
    assert not (tmp_path / "hermes-home" / "telemetry").exists()


def test_a_compressed_conversation_is_one_session_row_with_its_whole_volume(direct_runtime, tmp_path):
    """Legacy rotating compaction continues s1 as s1c under one conversation root: one session row,
    turns / API calls / tool calls / messages summed over both segments, whatever the close order."""
    token = set_conversation_context("s1")
    try:
        _turn("s1", "t1", tools=2)
        _turn("s1", "t2", tools=1)
        _turn("s1c", "t3", tools=0)
    finally:
        reset_conversation_context(token)
    lifecycle.finalize_session(session_id="s1c")
    _flush()
    assert not _stored_values(tmp_path, "hermes.session.count"), "a segment reported on its own"
    lifecycle.finalize_session(session_id="s1")
    _flush()

    [(row, value)] = _stored_values(tmp_path, "hermes.session.count")
    assert value == 1
    assert {k: row[k] for k in ("turn_count_bucket", "model_call_count_bucket", "tool_call_count_bucket",
                                "message_count_bucket")} == {
        "turn_count_bucket": "3_to_5", "model_call_count_bucket": "3_to_5",
        "tool_call_count_bucket": "3_to_5", "message_count_bucket": "6_to_10"}


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
