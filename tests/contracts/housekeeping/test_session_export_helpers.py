"""``hermes_cli.session_export``'s store-side helpers over the reference backend: ``export_timings``, ``to_legacy_export_dict`` and ``assert_exports_safe``.

Every expectation is M1's (``hermes_state_portability._export_timings``, the ``export_session`` / ``export_session_lineage`` dicts, ``SessionDB.assert_exports_safe``) in the same scenario.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli.session_export import assert_exports_safe, export_timings, to_legacy_export_dict
from persistence import ExportTooLarge, SessionId

NOW = time.time()
DAY = 86400.0

_ROW_KEYS = [
    "id", "runtime_revision", "runtime_generation", "source", "created_source", "user_id", "session_key", "chat_id", "chat_type", "thread_id",
    "display_name", "origin_json", "expiry_finalized", "model", "model_config", "system_prompt", "system_prompt_hash", "parent_session_id",
    "started_at", "ended_at", "end_reason", "message_count", "tool_call_count", "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_write_tokens", "reasoning_tokens", "cwd", "git_branch", "git_repo_root", "git_metadata_generation", "billing_provider",
    "billing_base_url", "billing_mode", "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source", "pricing_version", "title",
    "title_source", "last_activity_at", "last_activity_description", "last_activity_provenance", "api_call_count", "handoff_state",
    "handoff_platform", "handoff_error", "compression_failure_cooldown_until", "compression_failure_error", "compression_fallback_streak",
    "compression_ineffective_count", "compression_recovery_deadline", "compression_overload_streak", "profile_name", "transport_profile",
    "rewind_count", "archived", "auto_archived", "pinned", "hidden", "last_read_at", "tool_names", "transcript_revision",
]
_MESSAGE_KEYS = [
    "id", "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name", "effect_disposition", "timestamp", "token_count",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items", "platform_message_id",
    "observed", "active", "compacted", "api_content", "display_kind", "display_metadata", "message_uid", "absorbed_message_uids",
    "tool_call_uids", "tool_call_uid",
]


def _payload(sid, started, ended=None, end_reason=None, parent=None):
    return {"id": sid, "source": "cli", "started_at": started, "ended_at": ended, "end_reason": end_reason, "model": "model-a",
            "parent_session_id": parent, "messages": [{"role": "user", "content": "hello", "timestamp": started},
                                                      {"role": "assistant", "content": "hi", "timestamp": started + 1}]}


# ── export_timings ────────────────────────────────────────────────────────────


def test_export_timings_matches_the_message_timestamps():
    sample = [{"id": 1, "role": "user", "timestamp": 1000.0}, {"id": 2, "role": "assistant", "timestamp": 1002.5, "tool_calls": [{"id": "a"}, {"id": "b"}]},
              {"id": 3, "role": "tool", "timestamp": "bad"}, {"id": 4, "role": "assistant", "timestamp": 1010.0, "tool_calls": {"x": 1}},
              {"id": 5, "role": None, "timestamp": 1009.0}]
    assert export_timings(sample, "s-x") == {
        "source": "message_timestamps", "available": True, "complete": False, "unavailable_reason": None,
        "message_timestamps": {"available": 4, "missing": 1},
        "first_message_at": "1970-01-01T00:16:40+00:00", "last_message_at": "1970-01-01T00:16:49+00:00",
        "wall_clock_ms": 9000, "largest_gap_ms": 7500,
        "role_counts": {"user": 1, "assistant": 2, "tool": 1, "unknown": 1}, "tool_result_count": 1, "tool_calls_emitted": 3,
        "intervals": [
            {"from_message_id": 1, "to_message_id": 2, "from_role": "user", "to_role": "assistant", "gap_ms": 2500},
            {"from_message_id": 2, "to_message_id": 4, "from_role": "assistant", "to_role": "assistant", "gap_ms": 7500},
            {"from_message_id": 4, "to_message_id": 5, "from_role": "assistant", "to_role": None, "gap_ms": 0},
        ],
    }


def test_export_timings_of_no_messages():
    assert export_timings([], None) == {
        "source": "message_timestamps", "available": False, "complete": False, "unavailable_reason": "no_timestamped_messages",
        "message_timestamps": {"available": 0, "missing": 0}, "first_message_at": None, "last_message_at": None, "wall_clock_ms": None,
        "largest_gap_ms": None, "role_counts": {}, "tool_result_count": 0, "tool_calls_emitted": 0, "intervals": [],
    }


# ── to_legacy_export_dict ─────────────────────────────────────────────────────


def test_session_export_dict_shape(backend, scope_a):
    store = backend.open(scope_a)
    assert store.maintenance.import_sessions([_payload("s-e3", NOW, NOW, "done")]).ok
    one = to_legacy_export_dict(store.maintenance.export_session(SessionId("s-e3"), include_inactive=True))
    # CONTRACT: the row's columns in table order, then ``messages`` and ``timings``; each message is the stored row dict.
    assert list(one) == [*_ROW_KEYS, "messages", "timings"]
    assert list(one["messages"][0]) == _MESSAGE_KEYS
    assert [{k: m[k] for k in ("role", "content", "active", "compacted")} for m in one["messages"]] == [
        {"role": "user", "content": "hello", "active": 1, "compacted": 0}, {"role": "assistant", "content": "hi", "active": 1, "compacted": 0}]
    assert one["timings"] == export_timings(one["messages"], "s-e3")


def test_lineage_export_dict_shape(backend, scope_a):
    store = backend.open(scope_a)
    assert store.maintenance.import_sessions([_payload("s-root", NOW - 2 * DAY, NOW - 2 * DAY + 10, "compression"),
                                              _payload("s-tip", NOW - 2 * DAY + 20, parent="s-root")]).ok
    lineage = to_legacy_export_dict(store.maintenance.export_session_lineage(SessionId("s-tip")), session_id="s-tip")
    # CONTRACT: the tip's dict with the lineage keys replaced in place and ``segments`` and ``lineage_session_ids`` appended.
    assert list(lineage) == [*_ROW_KEYS, "messages", "timings", "segments", "lineage_session_ids"]
    assert (lineage["id"], lineage["lineage_session_ids"], lineage["message_count"]) == ("s-tip", ["s-root", "s-tip"], 4)
    assert [segment["id"] for segment in lineage["segments"]] == ["s-root", "s-tip"]
    assert list(lineage["segments"][0]) == [*_ROW_KEYS, "messages", "timings"]
    assert lineage["messages"] == [m for segment in lineage["segments"] for m in segment["messages"]]
    assert lineage["timings"] == export_timings(lineage["messages"], "s-tip")


# ── assert_exports_safe ───────────────────────────────────────────────────────


@pytest.fixture
def store(backend, scope_a):
    store = backend.open(scope_a)
    assert store.maintenance.import_sessions([_payload("s-e3", NOW, NOW, "done")]).ok
    return store


def test_a_session_over_the_limit_refuses_the_export(store):
    with pytest.raises(ExportTooLarge) as refused:
        assert_exports_safe(store, ["s-e3"], max_messages=1)
    assert str(refused.value) == (
        "Session 's-e3' has more than 1 exportable messages; the JSON/JSONL backup is built in memory and capped per session. "
        "Use the dashboard Sessions page's streaming Export action, or set sessions.max_export_messages: 0 in config.yaml to disable the guard.")


@pytest.mark.parametrize(("ids", "limit"), [(["s-e3"], 2), (["nope"], 1), ([], -1)], ids=["at_limit", "unknown", "negative_without_ids"])
def test_exports_within_the_limit_pass(store, ids, limit):
    assert assert_exports_safe(store, ids, max_messages=limit) is None


def test_a_zero_limit_disables_the_guard_without_a_read(store, monkeypatch):
    monkeypatch.setattr(store.maintenance, "assert_export_safe", lambda *a, **k: pytest.fail("read with the guard disabled"))
    assert assert_exports_safe(store, ["s-e3"], max_messages=0) is None


def test_a_negative_limit_is_refused(store):
    with pytest.raises(ValueError, match="^max_messages must be non-negative$"):
        assert_exports_safe(store, ["s-e3"], max_messages=-1)


def test_no_limit_reads_the_configured_one(store, monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", dict)
    assert assert_exports_safe(store, ["s-e3"]) is None
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {"sessions": {"max_export_messages": 1}})
    with pytest.raises(ExportTooLarge):
        assert_exports_safe(store, ["s-e3"])
