"""Tests for the admission-verification fix landed in #24826.

Pre-fix, ``hermes meet join`` returned exit 0 as soon as the bot
subprocess spawned, even when the host denied admission.  The Pamela
auto-join cron trusted that exit code and silently recorded denied
meetings as joined, suppressing future retries.

These tests pin the new contract on three layers so a future
refactor can't regress the flow:

* ``_classify_admission`` truth table — the canonical mapping from
  raw bot status fields (``inCall``, ``joinedAt``, ``error``,
  ``leaveReason``, ``exited``) and process liveness to a single
  ``admissionState`` verdict.
* ``status()`` — must surface the derived ``joined`` /
  ``admissionState`` / normalised ``error`` fields without
  breaking the existing schema.
* ``wait_for_join()`` — terminal-state handling, timeout, and
  no-active-meeting paths, all driven through injected ``sleep``
  / ``now`` so the tests run in milliseconds.
* ``hermes meet join`` exit codes — the user-facing contract that
  cron / auto-join scripts branch on.
* ``meet_join`` agent tool — synchronous-admission opt-in folds
  the verdict into the tool response without breaking the v1
  async default.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _seed_active_meeting(hermes_home: Path, *, bot_status: dict | None) -> Path:
    """Create the on-disk state ``status()`` and friends read from."""
    from plugins.google_meet import process_manager as pm

    out_dir = hermes_home / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True, exist_ok=True)
    pm._write_active({
        "pid": 99999,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })
    if bot_status is not None:
        (out_dir / "status.json").write_text(json.dumps(bot_status))
    return out_dir


# ---------------------------------------------------------------------------
# _classify_admission — the truth table
# ---------------------------------------------------------------------------

class TestClassifyAdmission:
    def test_in_call_no_error_is_joined(self):
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": True, "error": None}, alive=True,
        )
        assert verdict == {"joined": True, "admissionState": "joined", "error": None}

    def test_joined_at_alone_counts_as_joined(self):
        """``joinedAt`` sticks even after ``inCall`` flips back on a clean leave."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": False, "joinedAt": 1714159200.0}, alive=False,
        )
        assert verdict["joined"] is True
        assert verdict["admissionState"] == "joined"

    def test_error_set_blocks_joined_even_with_in_call_true(self):
        """If the bot raised an error mid-call, surface failure not joined."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": True, "error": "transient page crash"}, alive=False,
        )
        assert verdict["joined"] is False
        assert verdict["admissionState"] == "failed"
        assert "transient page crash" in verdict["error"]

    def test_denied_leave_reason_classifies_as_denied(self):
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": False, "leaveReason": "denied",
             "error": "host denied admission"},
            alive=False,
        )
        assert verdict["joined"] is False
        assert verdict["admissionState"] == "denied"
        assert verdict["error"] == "host denied admission"

    def test_denied_inferred_from_error_text_when_leave_reason_missing(self):
        """Bots that only flush ``error`` (no leaveReason yet) still read as denied."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": False, "error": "host denied admission"},
            alive=True,
        )
        assert verdict["admissionState"] == "denied"

    def test_lobby_timeout_classified_distinctly_from_denied(self):
        """Cron schedulers want to retry lobby-timeouts but not outright denials."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": False, "leaveReason": "lobby_timeout",
             "error": "lobby timeout — host never admitted"},
            alive=False,
        )
        assert verdict["admissionState"] == "lobby_timeout"

    def test_exited_without_admission_is_failed(self):
        """Bot crashed before getting in — neither joined nor explicitly denied."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": False, "exited": True}, alive=False,
        )
        assert verdict["admissionState"] == "failed"
        assert verdict["joined"] is False

    def test_dead_pid_with_no_status_signals_failed(self):
        """``alive=False`` is sufficient even when status.json is empty."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission({}, alive=False)
        assert verdict["admissionState"] == "failed"

    def test_alive_bot_no_signals_yet_is_waiting(self):
        """Initial flush — bot just spawned, status.json fields default to false/null."""
        from plugins.google_meet.process_manager import _classify_admission

        verdict = _classify_admission(
            {"inCall": False, "error": None, "leaveReason": None,
             "exited": False, "joinedAt": None},
            alive=True,
        )
        assert verdict == {"joined": False, "admissionState": "waiting", "error": None}


# ---------------------------------------------------------------------------
# status() — derived fields are appended without breaking the v1 schema
# ---------------------------------------------------------------------------

class TestStatusDerivedFields:
    def test_joined_meeting_surfaces_derived_joined_field(self, _isolate_home):
        from plugins.google_meet import process_manager as pm

        _seed_active_meeting(_isolate_home, bot_status={
            "inCall": True, "joinedAt": 1714159200.0, "error": None,
        })

        with patch.object(pm, "_pid_alive", return_value=True):
            res = pm.status()

        assert res["ok"] is True
        # Existing v1 fields preserved.
        assert res["meetingId"] == "abc-defg-hij"
        assert res["url"] == "https://meet.google.com/abc-defg-hij"
        assert res["alive"] is True
        # Derived fields the fix adds.
        assert res["joined"] is True
        assert res["admissionState"] == "joined"
        assert res["error"] is None

    def test_denied_meeting_surfaces_admission_state(self, _isolate_home):
        """Reproduces the exact bug from issue #24826."""
        from plugins.google_meet import process_manager as pm

        _seed_active_meeting(_isolate_home, bot_status={
            "inCall": False,
            "joinedAt": None,
            "error": "host denied admission",
            "leaveReason": "denied",
            "exited": True,
        })

        with patch.object(pm, "_pid_alive", return_value=False):
            res = pm.status()

        assert res["joined"] is False
        assert res["admissionState"] == "denied"
        assert res["error"] == "host denied admission"

    def test_derived_field_wins_over_raw_bot_field_with_same_name(
        self, _isolate_home,
    ):
        """If meet_bot ever ships a same-named ``joined`` key the verdict wins."""
        from plugins.google_meet import process_manager as pm

        _seed_active_meeting(_isolate_home, bot_status={
            # Hypothetical future raw key that disagrees with the verdict.
            "joined": True,
            "inCall": False,
            "leaveReason": "denied",
            "error": "host denied admission",
        })

        with patch.object(pm, "_pid_alive", return_value=False):
            res = pm.status()

        # Truth-table verdict takes precedence.
        assert res["joined"] is False
        assert res["admissionState"] == "denied"

    def test_status_with_no_active_meeting_unchanged(self):
        """Pre-existing 'no active meeting' shape is preserved verbatim."""
        from plugins.google_meet import process_manager as pm

        assert pm.status() == {"ok": False, "reason": "no active meeting"}


# ---------------------------------------------------------------------------
# wait_for_join() — poll loop terminal states + timeout
# ---------------------------------------------------------------------------

class _FakeClock:
    """Inject deterministic time + sleep into wait_for_join."""

    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s

    def now(self) -> float:
        return self.t


class TestWaitForJoin:
    def test_returns_no_active_immediately_when_nothing_running(self):
        from plugins.google_meet import process_manager as pm

        clk = _FakeClock()
        verdict = pm.wait_for_join(timeout=10.0, sleep=clk.sleep, now=clk.now)
        assert verdict == {
            "ok": False, "reason": "no active meeting",
            "waitOutcome": "no_active",
        }
        # Did not sleep — terminal on first probe.
        assert clk.sleeps == []

    def test_returns_joined_on_first_admitted_status(self, _isolate_home):
        from plugins.google_meet import process_manager as pm

        out_dir = _seed_active_meeting(_isolate_home, bot_status={
            "inCall": True, "joinedAt": 1714159200.0,
        })
        del out_dir  # unused

        clk = _FakeClock()
        with patch.object(pm, "_pid_alive", return_value=True):
            verdict = pm.wait_for_join(
                timeout=10.0, sleep=clk.sleep, now=clk.now,
            )

        assert verdict["waitOutcome"] == "joined"
        assert verdict["joined"] is True
        # First poll succeeded — no sleep.
        assert clk.sleeps == []

    def test_polls_until_admission_flips_to_in_call(self, _isolate_home):
        """Bot goes lobby → admitted on the third poll."""
        from plugins.google_meet import process_manager as pm

        out_dir = _seed_active_meeting(_isolate_home, bot_status={
            "inCall": False, "joinedAt": None, "lobbyWaiting": True,
        })

        statuses = [
            {"inCall": False, "lobbyWaiting": True},
            {"inCall": False, "lobbyWaiting": True},
            {"inCall": True, "joinedAt": 1714159210.0},
        ]
        flush_idx = {"i": 0}

        def _flush_next():
            i = flush_idx["i"]
            if i < len(statuses):
                (out_dir / "status.json").write_text(json.dumps(statuses[i]))
                flush_idx["i"] += 1

        clk = _FakeClock()

        def _sleep_and_advance(s: float):
            clk.sleep(s)
            _flush_next()

        _flush_next()  # seed first status
        with patch.object(pm, "_pid_alive", return_value=True):
            verdict = pm.wait_for_join(
                timeout=60.0, sleep=_sleep_and_advance, now=clk.now,
            )

        assert verdict["waitOutcome"] == "joined"
        # Two sleeps to consume statuses[1] and statuses[2].
        assert len(clk.sleeps) == 2

    def test_short_circuits_on_denied(self, _isolate_home):
        from plugins.google_meet import process_manager as pm

        _seed_active_meeting(_isolate_home, bot_status={
            "inCall": False, "leaveReason": "denied",
            "error": "host denied admission", "exited": True,
        })

        clk = _FakeClock()
        with patch.object(pm, "_pid_alive", return_value=False):
            verdict = pm.wait_for_join(
                timeout=60.0, sleep=clk.sleep, now=clk.now,
            )

        assert verdict["waitOutcome"] == "denied"
        assert verdict["joined"] is False
        assert verdict["error"] == "host denied admission"
        assert clk.sleeps == []  # no waiting needed

    def test_short_circuits_on_lobby_timeout(self, _isolate_home):
        from plugins.google_meet import process_manager as pm

        _seed_active_meeting(_isolate_home, bot_status={
            "inCall": False, "leaveReason": "lobby_timeout",
            "error": "lobby timeout — host never admitted",
            "exited": True,
        })

        clk = _FakeClock()
        with patch.object(pm, "_pid_alive", return_value=False):
            verdict = pm.wait_for_join(
                timeout=60.0, sleep=clk.sleep, now=clk.now,
            )

        assert verdict["waitOutcome"] == "lobby_timeout"

    def test_returns_timeout_when_bot_stays_in_lobby(self, _isolate_home):
        """Bot is alive but lobby never resolves — wait budget gates the loop."""
        from plugins.google_meet import process_manager as pm

        _seed_active_meeting(_isolate_home, bot_status={
            "inCall": False, "lobbyWaiting": True,
        })

        clk = _FakeClock()
        with patch.object(pm, "_pid_alive", return_value=True):
            verdict = pm.wait_for_join(
                timeout=3.0, poll_interval=1.0,
                sleep=clk.sleep, now=clk.now,
            )

        assert verdict["waitOutcome"] == "timeout"
        assert verdict["joined"] is False
        # We slept 3 times before the deadline check fired.
        assert len(clk.sleeps) >= 3
        assert "timed out" in verdict["error"].lower()


# ---------------------------------------------------------------------------
# CLI — exit codes drive the auto-join script's safety
# ---------------------------------------------------------------------------

class TestJoinExitCodeMapping:
    @pytest.mark.parametrize(
        "outcome,expected",
        [
            ("joined", 0),
            ("denied", 3),
            ("lobby_timeout", 4),
            ("failed", 5),
            ("no_active", 5),
            ("timeout", 6),
            ("unknown_future_value", 1),
        ],
    )
    def test_outcome_maps_to_documented_exit_code(self, outcome, expected):
        from plugins.google_meet.cli import _join_exit_code

        assert _join_exit_code({"waitOutcome": outcome}) == expected

    def test_no_outcome_field_falls_back_to_generic_failure(self):
        from plugins.google_meet.cli import _join_exit_code

        assert _join_exit_code({}) == 1


class TestCmdJoinSyncBehavior:
    def test_no_wait_preserves_legacy_spawn_and_return(self, _isolate_home):
        """Default exit-0 path when caller opted out of synchronous verification."""
        from plugins.google_meet import cli as meet_cli
        from plugins.google_meet import process_manager as pm

        with patch.object(pm, "start", return_value={
            "ok": True, "pid": 1, "meeting_id": "abc-defg-hij",
        }) as start_mock, patch.object(pm, "wait_for_join") as wait_mock:
            rc = meet_cli._cmd_join(
                url="https://meet.google.com/abc-defg-hij",
                guest_name="Hermes Agent", duration=None, headed=False,
                no_wait=True,
            )

        assert rc == 0
        start_mock.assert_called_once()
        # --no-wait must NOT touch the verification poll loop.
        wait_mock.assert_not_called()

    def test_default_waits_and_exits_zero_on_admission(self, _isolate_home):
        from plugins.google_meet import cli as meet_cli
        from plugins.google_meet import process_manager as pm

        with patch.object(pm, "start", return_value={
            "ok": True, "pid": 1, "meeting_id": "abc-defg-hij",
        }), patch.object(pm, "wait_for_join", return_value={
            "ok": True, "waitOutcome": "joined", "joined": True,
            "admissionState": "joined",
        }):
            rc = meet_cli._cmd_join(
                url="https://meet.google.com/abc-defg-hij",
                guest_name="Hermes Agent", duration=None, headed=False,
            )

        assert rc == 0

    def test_default_waits_and_exits_three_on_denied(self, _isolate_home, capsys):
        """The exact regression from issue #24826: denied admission ⇒ nonzero."""
        from plugins.google_meet import cli as meet_cli
        from plugins.google_meet import process_manager as pm

        with patch.object(pm, "start", return_value={
            "ok": True, "pid": 1, "meeting_id": "abc-defg-hij",
        }), patch.object(pm, "wait_for_join", return_value={
            "ok": True, "waitOutcome": "denied", "joined": False,
            "admissionState": "denied", "error": "host denied admission",
        }):
            rc = meet_cli._cmd_join(
                url="https://meet.google.com/abc-defg-hij",
                guest_name="Hermes Agent", duration=None, headed=False,
            )

        assert rc == 3
        captured = capsys.readouterr().out
        assert '"waitOutcome": "denied"' in captured
        assert '"joined": false' in captured

    def test_spawn_failure_returns_one_without_polling(self, _isolate_home):
        """If pm.start() fails we surface the v1 generic-failure exit code."""
        from plugins.google_meet import cli as meet_cli
        from plugins.google_meet import process_manager as pm

        with patch.object(pm, "start", return_value={
            "ok": False, "error": "subprocess failed",
        }), patch.object(pm, "wait_for_join") as wait_mock:
            rc = meet_cli._cmd_join(
                url="https://meet.google.com/abc-defg-hij",
                guest_name="Hermes Agent", duration=None, headed=False,
            )

        assert rc == 1
        wait_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Agent tool — wait_for_admission opt-in
# ---------------------------------------------------------------------------

class TestMeetJoinToolWaitForAdmission:
    def test_default_off_preserves_v1_async_behaviour(self):
        """Without wait_for_admission the response shape is unchanged."""
        from plugins.google_meet import process_manager as pm
        from plugins.google_meet.tools import handle_meet_join

        with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
             patch.object(pm, "start", return_value={
                 "ok": True, "pid": 1, "meeting_id": "abc-defg-hij",
             }), patch.object(pm, "wait_for_join") as wait_mock:
            out = json.loads(handle_meet_join({
                "url": "https://meet.google.com/abc-defg-hij",
            }))

        assert out["success"] is True
        wait_mock.assert_not_called()

    def test_wait_true_returns_success_only_when_admitted(self):
        from plugins.google_meet import process_manager as pm
        from plugins.google_meet.tools import handle_meet_join

        with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
             patch.object(pm, "start", return_value={
                 "ok": True, "pid": 1, "meeting_id": "abc-defg-hij",
             }), patch.object(pm, "wait_for_join", return_value={
                 "ok": True, "waitOutcome": "joined", "joined": True,
                 "admissionState": "joined",
             }):
            out = json.loads(handle_meet_join({
                "url": "https://meet.google.com/abc-defg-hij",
                "wait_for_admission": True,
            }))

        assert out["success"] is True
        assert out["waitOutcome"] == "joined"

    def test_wait_true_returns_failure_on_denial(self):
        """The exact agent-side regression mirror of issue #24826."""
        from plugins.google_meet import process_manager as pm
        from plugins.google_meet.tools import handle_meet_join

        with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
             patch.object(pm, "start", return_value={
                 "ok": True, "pid": 1, "meeting_id": "abc-defg-hij",
             }), patch.object(pm, "wait_for_join", return_value={
                 "ok": True, "waitOutcome": "denied", "joined": False,
                 "admissionState": "denied", "error": "host denied admission",
             }):
            out = json.loads(handle_meet_join({
                "url": "https://meet.google.com/abc-defg-hij",
                "wait_for_admission": True,
            }))

        assert out["success"] is False
        assert out["admissionState"] == "denied"
        assert out["error"] == "host denied admission"

    def test_wait_seconds_threaded_through_to_pm(self):
        from plugins.google_meet import process_manager as pm
        from plugins.google_meet.tools import handle_meet_join

        with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
             patch.object(pm, "start", return_value={
                 "ok": True, "pid": 1,
             }), patch.object(pm, "wait_for_join", return_value={
                 "ok": True, "waitOutcome": "joined", "joined": True,
             }) as wait_mock:
            handle_meet_join({
                "url": "https://meet.google.com/abc-defg-hij",
                "wait_for_admission": True,
                "wait_seconds": 30,
            })

        wait_mock.assert_called_once()
        kwargs = wait_mock.call_args.kwargs
        assert kwargs["timeout"] == 30.0

    def test_malformed_wait_seconds_falls_back_to_default(self):
        """Bad numeric input from the model must not crash the tool."""
        from plugins.google_meet import process_manager as pm
        from plugins.google_meet.tools import handle_meet_join

        with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
             patch.object(pm, "start", return_value={
                 "ok": True, "pid": 1,
             }), patch.object(pm, "wait_for_join", return_value={
                 "ok": True, "waitOutcome": "joined", "joined": True,
             }) as wait_mock:
            handle_meet_join({
                "url": "https://meet.google.com/abc-defg-hij",
                "wait_for_admission": True,
                "wait_seconds": "not a number",
            })

        kwargs = wait_mock.call_args.kwargs
        assert kwargs["timeout"] == pm.DEFAULT_JOIN_WAIT_S

    def test_wait_skipped_when_pm_start_failed(self):
        """Spawn failure short-circuits before touching wait_for_join."""
        from plugins.google_meet import process_manager as pm
        from plugins.google_meet.tools import handle_meet_join

        with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
             patch.object(pm, "start", return_value={
                 "ok": False, "error": "subprocess failed",
             }), patch.object(pm, "wait_for_join") as wait_mock:
            out = json.loads(handle_meet_join({
                "url": "https://meet.google.com/abc-defg-hij",
                "wait_for_admission": True,
            }))

        assert out["success"] is False
        wait_mock.assert_not_called()


# ---------------------------------------------------------------------------
# DENIAL_LEAVE_REASONS — guard against silent meet_bot renames
# ---------------------------------------------------------------------------

def test_denial_leave_reasons_match_meet_bot_emissions():
    """Any leave_reason emitted by meet_bot's pre-admission failure paths
    must be present in DENIAL_LEAVE_REASONS so wait_for_join classifies
    it as a terminal failure rather than spinning indefinitely.

    Pin the canonical strings here so a future rename in meet_bot trips
    a unit test before it silently converts a denial into a 'failed'
    bucket (or worse, a 'waiting' state that hangs the wait loop).
    """
    from plugins.google_meet.process_manager import DENIAL_LEAVE_REASONS

    # Strings literally emitted by ``state.set(leave_reason=...)`` in
    # meet_bot.py — see the pre-admission failure paths around the
    # admission-detection loop.
    expected = {"denied", "lobby_timeout", "page_closed", "duration_expired"}
    assert expected.issubset(DENIAL_LEAVE_REASONS)
