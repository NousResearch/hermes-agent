"""Unit tests for ``cron.subagent_topic_reaper``.

Phase B2 coverage:

* Empty / missing state → zeroed summary.
* TTL boundary semantics (strict ``>``, not ``>=``).
* Dry-run never calls Telegram and never mutates state.
* Happy-path deletion + state pruning.
* Error classification (``topic_not_found`` / ``chat_not_found`` → ``skipped_gone``;
  other codes → ``failed`` with entry retained).
* Malformed entries — missing ids short-circuit without API call; bad timestamp
  is treated as expired.
* Re-load-before-save race: router writes a new entry while reaper is working
  → that entry survives the prune.
* Summary schema (all keys and per-detail fields).
* Never-raise contract — ``_load_state`` / ``_save_state`` explosions become
  ``summary["error"]``, not uncaught exceptions.

All tests are fully offline: ``tools.telegram_topic_tool`` symbols are
monkey-patched so no network traffic ever occurs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.subagent_topic_reaper import reap_expired_topics
from gateway.subagent_topic_router import _load_state, _save_state


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    """A nested path for the state file — exercises mkdir."""
    return tmp_path / "state" / "subagent_topics.json"


def _entry(
    *,
    chat_id: str = "-100111",
    thread_id: str = "1",
    topic_name: str = "SA test",
    age_ago_seconds: Optional[float] = None,
    last_message_ts: Optional[float] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a minimal state-file topic entry."""
    now = now if now is not None else time.time()
    if age_ago_seconds is not None:
        last_message_ts = now - age_ago_seconds
    if last_message_ts is None:
        last_message_ts = now
    return {
        "chat_id": chat_id,
        "thread_id": thread_id,
        "topic_name": topic_name,
        "created_ts": last_message_ts,
        "last_message_ts": last_message_ts,
        "parent_chat_id": chat_id,
        "parent_thread_id": "9",
    }


def _seed(path: Path, topics: Dict[str, Dict[str, Any]]) -> None:
    """Write a state file with the given topics mapping."""
    _save_state(path, {"version": 1, "topics": topics})


@pytest.fixture
def patched_tg():
    """Patch the telegram topic tool symbols the reaper imports lazily.

    Important: the reaper does ``from tools.telegram_topic_tool import ...``
    *inside* ``reap_expired_topics``, so we must patch on the real module
    (``tools.telegram_topic_tool``), not on ``cron.subagent_topic_reaper``.
    """
    with patch(
        "tools.telegram_topic_tool._load_telegram_token",
        return_value=("fake-token", None),
    ) as p_token, patch(
        "tools.telegram_topic_tool._run_topic_op",
        new_callable=AsyncMock,
    ) as p_run, patch(
        "tools.telegram_topic_tool._classify_error",
        return_value="unknown",
    ) as p_classify:
        p_run.return_value = {"success": True, "action": "delete"}
        yield SimpleNamespace(token=p_token, run=p_run, classify=p_classify)


# ---------------------------------------------------------------------------
# A. No state / empty state
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_no_state_file_returns_zero_summary(self, state_path, patched_tg):
        assert not state_path.exists()
        summary = reap_expired_topics(state_path=state_path)
        assert summary["scanned"] == 0
        assert summary["expired"] == 0
        assert summary["deleted"] == 0
        assert summary["skipped_gone"] == 0
        assert summary["failed"] == 0
        assert summary["details"] == []
        assert "error" not in summary
        patched_tg.run.assert_not_called()

    def test_empty_topics_returns_zero(self, state_path, patched_tg):
        _seed(state_path, {})
        summary = reap_expired_topics(state_path=state_path)
        assert summary["scanned"] == 0
        assert summary["expired"] == 0
        assert summary["deleted"] == 0
        assert summary["details"] == []
        assert "error" not in summary
        patched_tg.run.assert_not_called()

    def test_no_expired_returns_zero_deletions(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "sid-a": _entry(age_ago_seconds=10, now=now),
                "sid-b": _entry(age_ago_seconds=100, now=now),
                "sid-c": _entry(age_ago_seconds=500, now=now),
            },
        )
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["scanned"] == 3
        assert summary["expired"] == 0
        assert summary["deleted"] == 0
        assert summary["details"] == []
        patched_tg.run.assert_not_called()


# ---------------------------------------------------------------------------
# B. TTL boundary
# ---------------------------------------------------------------------------


class TestTTLBoundary:
    def test_topic_exactly_at_ttl_not_expired(self, state_path, patched_tg):
        """Age == ttl should NOT be considered expired (code uses ``>``)."""
        now = 1_000_000.0
        ttl = 3600
        _seed(state_path, {"sid": _entry(age_ago_seconds=ttl, now=now)})
        summary = reap_expired_topics(
            state_path=state_path, now=now, ttl_seconds=ttl
        )
        assert summary["scanned"] == 1
        assert summary["expired"] == 0
        patched_tg.run.assert_not_called()

    def test_topic_over_ttl_expired(self, state_path, patched_tg):
        now = 1_000_000.0
        ttl = 3600
        _seed(state_path, {"sid": _entry(age_ago_seconds=ttl + 1, now=now)})
        summary = reap_expired_topics(
            state_path=state_path, now=now, ttl_seconds=ttl
        )
        assert summary["expired"] == 1
        assert summary["deleted"] == 1
        patched_tg.run.assert_awaited_once()

    def test_custom_ttl_respected(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "sid-young": _entry(age_ago_seconds=3599, now=now),
                "sid-old": _entry(age_ago_seconds=3601, now=now),
            },
        )
        summary = reap_expired_topics(
            state_path=state_path, now=now, ttl_seconds=3600
        )
        assert summary["scanned"] == 2
        assert summary["expired"] == 1
        assert summary["deleted"] == 1
        # Only the old one got deleted
        assert patched_tg.run.await_count == 1


# ---------------------------------------------------------------------------
# C. Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_does_not_call_telegram(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid-expired": _entry(age_ago_seconds=999_999, now=now)},
        )
        summary = reap_expired_topics(
            state_path=state_path, now=now, dry_run=True
        )
        assert summary["dry_run"] is True
        assert summary["expired"] == 1
        patched_tg.run.assert_not_called()
        patched_tg.token.assert_not_called()

    def test_dry_run_does_not_modify_state(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid-expired": _entry(age_ago_seconds=999_999, now=now)},
        )
        before = state_path.read_bytes()
        reap_expired_topics(state_path=state_path, now=now, dry_run=True)
        after = state_path.read_bytes()
        assert before == after

    def test_dry_run_returns_status_dry_run(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "sid-1": _entry(age_ago_seconds=999_999, now=now),
                "sid-2": _entry(age_ago_seconds=999_999, now=now),
            },
        )
        summary = reap_expired_topics(
            state_path=state_path, now=now, dry_run=True
        )
        assert summary["deleted"] == 0
        assert summary["failed"] == 0
        assert summary["skipped_gone"] == 0
        assert len(summary["details"]) == 2
        for d in summary["details"]:
            assert d["status"] == "dry_run"
            assert d["error"] is None


# ---------------------------------------------------------------------------
# D. Real deletion
# ---------------------------------------------------------------------------


class TestDeletion:
    def test_deletion_happy_path(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid-A": _entry(age_ago_seconds=999_999, now=now, thread_id="5")},
        )
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["deleted"] == 1
        assert summary["details"][0]["status"] == "deleted"
        patched_tg.run.assert_awaited_once()
        # Check positional + kwargs on the call
        call = patched_tg.run.await_args
        assert call.args[0] == "fake-token"
        assert call.args[1] == "delete"
        assert call.kwargs["thread_id"] == "5"
        assert call.kwargs["name"] is None
        # State pruned
        new_state = _load_state(state_path)
        assert "sid-A" not in new_state["topics"]

    def test_multiple_expired_all_deleted(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                f"sid-{i}": _entry(
                    age_ago_seconds=999_999, now=now, thread_id=str(i)
                )
                for i in range(3)
            },
        )
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["scanned"] == 3
        assert summary["deleted"] == 3
        assert patched_tg.run.await_count == 3
        new_state = _load_state(state_path)
        assert new_state["topics"] == {}

    def test_mixed_expired_and_fresh(self, state_path, patched_tg):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "fresh-1": _entry(age_ago_seconds=10, now=now, thread_id="1"),
                "fresh-2": _entry(age_ago_seconds=20, now=now, thread_id="2"),
                "old-1": _entry(age_ago_seconds=999_999, now=now, thread_id="11"),
                "old-2": _entry(age_ago_seconds=999_999, now=now, thread_id="22"),
                "old-3": _entry(age_ago_seconds=999_999, now=now, thread_id="33"),
            },
        )
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["scanned"] == 5
        assert summary["expired"] == 3
        assert summary["deleted"] == 3
        assert patched_tg.run.await_count == 3
        new_state = _load_state(state_path)
        assert set(new_state["topics"].keys()) == {"fresh-1", "fresh-2"}

    def test_deletion_updates_state_file_atomically(
        self, state_path, patched_tg
    ):
        """Ensure the state file is *overwritten* (not appended) with
        only the surviving entries."""
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "fresh": _entry(age_ago_seconds=10, now=now, thread_id="1"),
                "old": _entry(age_ago_seconds=999_999, now=now, thread_id="2"),
            },
        )
        reap_expired_topics(state_path=state_path, now=now)
        # File should parse cleanly as a single JSON doc (not appended).
        raw = state_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["topics"].keys() == {"fresh"}
        # Exactly one top-level JSON object.
        assert raw.count('"version"') == 1


# ---------------------------------------------------------------------------
# E. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_no_token_returns_no_token_error(self, state_path):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        with patch(
            "tools.telegram_topic_tool._load_telegram_token",
            return_value=(None, "missing env var"),
        ), patch(
            "tools.telegram_topic_tool._run_topic_op",
            new_callable=AsyncMock,
        ) as p_run:
            summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary.get("error") == "no_token"
        assert summary["deleted"] == 0
        p_run.assert_not_called()
        # State untouched
        new_state = _load_state(state_path)
        assert "sid" in new_state["topics"]

    def test_topic_not_found_classified_as_skipped_gone(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        patched_tg.run.side_effect = RuntimeError("boom: topic")
        patched_tg.classify.return_value = "topic_not_found"
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["deleted"] == 0
        assert summary["skipped_gone"] == 1
        assert summary["failed"] == 0
        assert summary["details"][0]["status"] == "skipped_gone"
        assert summary["details"][0]["error"] == "topic_not_found"
        # Pruned from state
        new_state = _load_state(state_path)
        assert "sid" not in new_state["topics"]

    def test_chat_not_found_classified_as_skipped_gone(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        patched_tg.run.side_effect = RuntimeError("boom: chat")
        patched_tg.classify.return_value = "chat_not_found"
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["skipped_gone"] == 1
        assert summary["details"][0]["status"] == "skipped_gone"
        new_state = _load_state(state_path)
        assert "sid" not in new_state["topics"]

    def test_no_rights_classified_as_failed_and_retained(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        patched_tg.run.side_effect = RuntimeError("no rights")
        patched_tg.classify.return_value = "no_rights"
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["deleted"] == 0
        assert summary["failed"] == 1
        assert summary["skipped_gone"] == 0
        assert summary["details"][0]["status"] == "failed"
        assert "no_rights" in summary["details"][0]["error"]
        # Entry retained in state for the next sweep
        new_state = _load_state(state_path)
        assert "sid" in new_state["topics"]

    def test_unknown_error_classified_as_failed(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        patched_tg.run.side_effect = RuntimeError("weird")
        patched_tg.classify.return_value = "unknown"
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["failed"] == 1
        assert summary["details"][0]["status"] == "failed"
        new_state = _load_state(state_path)
        assert "sid" in new_state["topics"]

    def test_classifier_itself_raises_fallback_to_unknown(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        patched_tg.run.side_effect = RuntimeError("boom")
        patched_tg.classify.side_effect = RuntimeError("classifier broken")
        summary = reap_expired_topics(state_path=state_path, now=now)
        # Falls back to "unknown" → failed, retained in state.
        assert summary["failed"] == 1
        assert summary["details"][0]["status"] == "failed"
        assert "unknown" in summary["details"][0]["error"]
        new_state = _load_state(state_path)
        assert "sid" in new_state["topics"]

    def test_missing_chat_id_or_thread_id_failed_without_api_call(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "sid-bad": _entry(
                    age_ago_seconds=999_999, now=now, chat_id="", thread_id="5"
                ),
                "sid-bad2": _entry(
                    age_ago_seconds=999_999, now=now, chat_id="-100", thread_id=""
                ),
            },
        )
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["failed"] == 2
        assert summary["deleted"] == 0
        for d in summary["details"]:
            assert d["status"] == "failed"
            assert "missing chat_id or thread_id" in d["error"]
        patched_tg.run.assert_not_called()
        # Retained in state (failed)
        new_state = _load_state(state_path)
        assert "sid-bad" in new_state["topics"]
        assert "sid-bad2" in new_state["topics"]

    def test_malformed_last_ts_treated_as_expired(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        entry = _entry(age_ago_seconds=10, now=now)
        entry["last_message_ts"] = "not a number"
        _seed(state_path, {"sid": entry})
        summary = reap_expired_topics(state_path=state_path, now=now)
        # last_ts parses as 0 → age ~= now → expired under default TTL
        assert summary["expired"] == 1
        assert summary["deleted"] == 1
        patched_tg.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# F. Re-load-before-save anti-race
# ---------------------------------------------------------------------------


class TestReloadRace:
    def test_concurrent_router_write_not_clobbered(
        self, state_path, patched_tg
    ):
        """While the reaper is iterating deletions, a brand-new topic (D)
        is written into the state file.  When the reaper prunes, D must
        survive."""
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "A": _entry(age_ago_seconds=999_999, now=now, thread_id="1"),
                "B": _entry(age_ago_seconds=999_999, now=now, thread_id="2"),
                "C": _entry(age_ago_seconds=10, now=now, thread_id="3"),
            },
        )

        call_counter = {"n": 0}

        async def _run_side_effect(*args, **kwargs):
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                # Simulate the router writing a new topic D concurrently.
                cur = _load_state(state_path)
                cur["topics"]["D"] = _entry(
                    age_ago_seconds=1, now=now, thread_id="99"
                )
                _save_state(state_path, cur)
            return {"success": True}

        patched_tg.run.side_effect = _run_side_effect

        summary = reap_expired_topics(state_path=state_path, now=now)

        assert summary["deleted"] == 2  # A + B
        assert summary["expired"] == 2

        new_state = _load_state(state_path)
        # A and B should be pruned; C (fresh) and D (newly written) survive.
        assert set(new_state["topics"].keys()) == {"C", "D"}


# ---------------------------------------------------------------------------
# G. Summary structure
# ---------------------------------------------------------------------------


class TestSummaryStructure:
    def test_summary_has_all_expected_keys(self, state_path, patched_tg):
        summary = reap_expired_topics(state_path=state_path)
        for key in (
            "scanned",
            "expired",
            "deleted",
            "skipped_gone",
            "failed",
            "dry_run",
            "details",
        ):
            assert key in summary, f"missing key {key}"

    def test_details_per_item_has_required_fields(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {
                "sid-x": _entry(
                    age_ago_seconds=999_999,
                    now=now,
                    thread_id="7",
                    topic_name="SA xyz",
                ),
            },
        )
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert len(summary["details"]) == 1
        detail = summary["details"][0]
        for field in (
            "session_id",
            "chat_id",
            "thread_id",
            "topic_name",
            "last_message_ts",
            "age_seconds",
            "status",
            "error",
        ):
            assert field in detail, f"missing detail field {field}"
        assert detail["session_id"] == "sid-x"
        assert detail["thread_id"] == "7"
        assert detail["topic_name"] == "SA xyz"
        assert detail["status"] == "deleted"
        assert detail["error"] is None
        assert detail["age_seconds"] > 999_000


# ---------------------------------------------------------------------------
# H. State save error
# ---------------------------------------------------------------------------


class TestStateSaveError:
    def test_state_save_failure_reports_error_but_does_not_raise(
        self, state_path, patched_tg
    ):
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )

        original_load = _load_state
        call_count = {"n": 0}

        def flaky_save(_path, _state):
            raise RuntimeError("disk exploded")

        # The reaper does ``from gateway.subagent_topic_router import
        # _load_state, _save_state`` inside the function, so patching on
        # that module takes effect at call time.
        with patch(
            "gateway.subagent_topic_router._save_state",
            side_effect=flaky_save,
        ):
            summary = reap_expired_topics(state_path=state_path, now=now)

        # Deletion counted even though state save blew up.
        assert summary["deleted"] == 1
        assert summary.get("error") == "state_save_failed"


# ---------------------------------------------------------------------------
# I. Never-raise contract
# ---------------------------------------------------------------------------


class TestNeverRaises:
    def test_never_raises_on_load_state_failure(self, state_path, patched_tg):
        def boom(_path):
            raise RuntimeError("load crashed hard")

        # Patch at import site (the gateway module) — the reaper's lazy
        # ``from gateway.subagent_topic_router import _load_state`` picks
        # this up.
        with patch(
            "gateway.subagent_topic_router._load_state",
            side_effect=boom,
        ):
            # Must not raise.
            summary = reap_expired_topics(state_path=state_path)
        assert summary.get("error") == "state_load_failed"
        assert summary["scanned"] == 0
        assert summary["deleted"] == 0
        patched_tg.run.assert_not_called()

    def test_never_raises_on_run_topic_op_catastrophic(
        self, state_path, patched_tg
    ):
        """Even a bare ``BaseException``-ish behaviour shouldn't escape;
        classifier path still catches plain ``Exception`` and demotes to
        failed."""
        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )
        patched_tg.run.side_effect = Exception("anything")
        patched_tg.classify.return_value = "unknown"
        summary = reap_expired_topics(state_path=state_path, now=now)
        assert summary["failed"] == 1


# ---------------------------------------------------------------------------
# J. CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_dry_run_json_output(
        self, state_path, patched_tg, capsys, monkeypatch
    ):
        """``_main --dry-run --json`` emits valid JSON with dry_run=true."""
        from cron import subagent_topic_reaper as reaper_mod

        now = 1_000_000.0
        _seed(
            state_path,
            {"sid": _entry(age_ago_seconds=999_999, now=now)},
        )

        # Steer _main's call to reap_expired_topics into our tmp state path.
        real_reap = reaper_mod.reap_expired_topics

        def _reap_with_path(*, ttl_seconds, dry_run):
            return real_reap(
                ttl_seconds=ttl_seconds,
                dry_run=dry_run,
                state_path=state_path,
                now=now,
            )

        monkeypatch.setattr(
            reaper_mod, "reap_expired_topics", _reap_with_path
        )
        monkeypatch.setattr(
            "sys.argv",
            ["subagent_topic_reaper", "--dry-run", "--json"],
        )

        rc = reaper_mod._main()
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["dry_run"] is True
        assert data["scanned"] == 1
        assert data["expired"] == 1

    def test_main_no_args_prints_summary(
        self, state_path, patched_tg, capsys, monkeypatch
    ):
        from cron import subagent_topic_reaper as reaper_mod

        real_reap = reaper_mod.reap_expired_topics

        def _reap_with_path(*, ttl_seconds, dry_run):
            return real_reap(
                ttl_seconds=ttl_seconds,
                dry_run=dry_run,
                state_path=state_path,
            )

        monkeypatch.setattr(
            reaper_mod, "reap_expired_topics", _reap_with_path
        )
        monkeypatch.setattr(
            "sys.argv",
            ["subagent_topic_reaper"],
        )

        rc = reaper_mod._main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Scanned:" in captured.out
