"""Unbacked follow-up promise guard.

The agent is inert between turns. A final reply that promises a later follow-up ("I'll check again in ~10
minutes") with nothing armed to wake it — no notify-on-complete background process, no background delegation, no
cron job, /loop or /heartbeat — is never kept; the user finds out only by asking. The guard re-prompts once on the
shared continuation budget so the model does the work now, arms a wake source, or drops the promise.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_followup_promise import has_armed_wake_source, unbacked_promise_phrase


@pytest.fixture(autouse=True)
def empty_schedule_store():
    """No /loop or /heartbeat rows by default; a missing store would read as unknown (armed)."""
    with patch("hermes_cli.goals._get_session_db", return_value=SimpleNamespace(get_meta=lambda _key: None)):
        yield


class TestPromiseDetector:
    @pytest.mark.parametrize("text", [
        # Real reply tails that were never followed up.
        "Not stalled, slow. It is writing files locally first. I'll check again in ~10 minutes; if the "
        "transcript hasn't advanced by then I'll steer it.",
        "It should finish in time: the 17:47 sweep is well inside the limit. I'll check on it after the sweep.",
        "Nothing merged; the live host is untouched. I'll ping you when W1 completes or blocks.",
        "A background watcher is following CI and the review bot; I'll report back when it finishes.",
        "Both reviews are running. I will report back with the merged, verified findings once both reviews return.",
        "Kicked off the build. I'll let you know when it's done.",
        "I will check again tomorrow.",
        "I will report back in ten minutes.",
        # A closing courtesy after the promise does not resolve it.
        "I'll report back when it finishes. Thanks!",
        "I'll report back when it finishes. Let me know if you need anything else.",
        "I'll report back when it finishes.\nThanks!",
        "I'll report back when it finishes. No action needed.",
        # A semicolon that only continues the plan.
        "Not stalled. I'll check again in ~10 minutes; if it hasn't advanced by then I'll steer it.",
        # A colon or quotes around part of the model's own promise.
        "Status: I'll report back when it finishes.",
        "The build is running: I'll report back when it finishes.",
        "I'll report back `when it finishes`.",
        "I'll report back when `make test` finishes.",
        "I'll check again in `ten minutes`.",
    ])
    def test_flags_deferred_followups(self, text):
        assert unbacked_promise_phrase(text)

    @pytest.mark.parametrize("text", [
        # Waits on the user, who will be writing anyway.
        "Send the Discord DM now. Tell me when it's sent and I'll poll the control plane for the wake signature.",
        "Tap that, then I'll re-check.",
        "Tell me when you've finished the authorization and I'll re-check LIST_TEAMS.",
        "If you can get a capture, I'll finish the updater properly.",
        # Offers and present-tense work, not deferred promises.
        "Want me to watch the checks and report back when they settle?",
        "The tests pass and the PR is merged.",
        "I'll report back the numbers below: 3 failures, all flaky.",
        # A promise mid-reply, followed by the real result.
        "I'll check back when it finishes. It finished: exit 0, 412 tests passed. Here is the summary of what "
        "changed in the build, the cache behaviour and the lint baseline, with no further action needed from you. "
        "The release notes are attached, the changelog is updated, and the tag is pushed to the remote. Nothing "
        "else is pending on this branch and the worktree has been removed. Everything is green across all lanes.",
        # ...and the same shape when the result is short.
        "I will check again when it finishes. It finished: exit 0, all tests passed.",
        "I will report back when it finishes. Result: the build passed; nothing remains.",
        # Quoted, fenced or introduced text is not the model speaking.
        'The log contains the text: "I will report back when it finishes."',
        "```text\nI will report back when it finishes.\n```",
        "The example reply is: I will report back when it finishes.",
        "I'll report back when it finishes: it finished successfully.",
        # Any later statement might report the outcome: silent, even when it does not.
        "I'll report back when it finishes. It crashed.",
        "I'll report back when it finishes. It timed out.",
        "I'll report back when it finishes. It was cancelled.",
        "I'll report back when it finishes. The build is complete; all checks are green.",
        "I'll report back when it finishes. I am watching it.",
        "I'll report back when it finishes; thanks for the earlier results.",
        # Anything that might make it wait on, or be offered to, the user: silent.
        "I'll report back when it finishes. Send me the output first.",
        "If you want me to, I'll report back when it finishes.",
        "If you like, I'll report back when it finishes.",
        "I will report back when it finishes, if you are busy.",
        "",
    ])
    def test_ignores_non_promises(self, text):
        assert not unbacked_promise_phrase(text)


def _agent(**kw):
    return SimpleNamespace(session_id="sess-1", platform="desktop", _delegate_depth=0,
                           _process_owner_task_ids={"task-1"}, **kw)


class TestWakeSources:
    @pytest.fixture()
    def registry(self):
        from tools.process_registry import process_registry
        saved = dict(process_registry._running)
        process_registry._running.clear()
        yield process_registry
        process_registry._running.clear()
        process_registry._running.update(saved)

    def _spawn(self, registry, owner="task-1", **kw):
        from tools.process_registry import ProcessSession
        proc = ProcessSession(id=f"proc_{len(registry._running)}", command="ci-watch", owner_task_id=owner,
                              started_at=time.time(), **kw)
        registry._running[proc.id] = proc

    def test_nothing_armed(self, registry):
        assert not has_armed_wake_source(_agent(), [])

    @pytest.mark.parametrize("notify", [{"notify_on_complete": True}, {"watch_patterns": ["VERDICT"]}])
    def test_notifying_background_process_counts(self, registry, notify):
        self._spawn(registry, **notify)
        assert has_armed_wake_source(_agent(), [])

    def test_silent_or_foreign_process_does_not_count(self, registry):
        self._spawn(registry)  # no notify: it will never wake the agent
        self._spawn(registry, owner="another-task", notify_on_complete=True)
        assert not has_armed_wake_source(_agent(), [])

    def test_live_background_delegation_counts(self, registry):
        with patch("tools.async_delegation.has_live_for_session", return_value=True) as live:
            assert has_armed_wake_source(_agent(), [])
        live.assert_any_call(parent_session_id="sess-1")

    def test_delegation_dispatched_before_compression_rotation_counts(self, registry):
        """The record keeps the pre-rotation id; the agent's id has moved on to the continuation."""
        from tools import async_delegation as d
        db = SimpleNamespace(get_compression_lineage=lambda sid: ["sess-0", "sess-1"])
        saved = dict(d._records)
        d._records["deleg_probe"] = {"status": "running", "parent_session_id": "sess-0"}
        try:
            assert has_armed_wake_source(_agent(_session_db=db), [])
            # Without the lineage the same record is invisible: the guard would nag a correct promise.
            assert not has_armed_wake_source(_agent(), [])
        finally:
            d._records.clear()
            d._records.update(saved)

    @staticmethod
    def _cron_turn(args, result, *, bridged=False):
        if bridged:
            fn = {"name": "tool_call", "arguments": json.dumps({"calls": [{"name": "cronjob", "arguments": args}]})}
        else:
            fn = {"name": "cronjob", "arguments": json.dumps(args)}
        return [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"id": "c1", "function": fn}]},
                {"role": "tool", "tool_call_id": "c1", "content": result}]

    @pytest.fixture()
    def cron(self, tmp_path, monkeypatch):
        """The real cronjob handler against a temp job store, so results have their real shapes."""
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
        from tools.cronjob_tools import cronjob

        def call(**args):
            return args, cronjob(**args)
        return call

    @pytest.mark.parametrize("bridged", [False, True])
    def test_scheduled_cron_create_this_turn_counts(self, registry, cron, bridged):
        args, result = cron(action="create", prompt="Check the build", schedule="every 10m")
        assert has_armed_wake_source(_agent(), self._cron_turn(args, result, bridged=bridged))

    @pytest.mark.parametrize("bridged", [False, True])
    def test_cron_call_that_arms_nothing_does_not_count(self, registry, cron, bridged):
        _args, created = cron(action="create", prompt="Check", schedule="every 10m")
        job_id = json.loads(created)["job_id"]
        for args, result in (cron(action="list"), cron(action="pause", job_id=job_id),
                             cron(action="remove", job_id=job_id), cron(action="create", prompt="Check"),
                             cron(action="create", prompt="Check", schedule="every 10m", paused=True)):
            assert not has_armed_wake_source(_agent(), self._cron_turn(args, result, bridged=bridged)), (args, result)

    def test_cron_resume_counts(self, registry, cron):
        _args, created = cron(action="create", prompt="Check", schedule="every 10m", paused=True)
        args, result = cron(action="resume", job_id=json.loads(created)["job_id"])
        assert has_armed_wake_source(_agent(), self._cron_turn(args, result))

    @pytest.mark.parametrize("job,armed", [
        ({"executed": True, "execution_mode": "background"}, True),    # outcome re-enters the conversation
        # An inline run of a one-shot: its outcome is already in this turn and nothing is left to run.
        ({"executed": True, "execution_success": False, "state": "completed", "next_run_at": None}, False),
        # An inline run of a recurring job: this occurrence is done, the next one is still armed.
        ({"executed": True, "execution_mode": None, "enabled": True, "state": "scheduled",
          "next_run_at": "2026-10-20T16:22:24+11:00"}, True),
    ])
    def test_cron_run_counts_only_when_its_outcome_is_still_coming(self, registry, job, armed):
        result = json.dumps({"success": True, "job": job})
        assert has_armed_wake_source(_agent(), self._cron_turn({"action": "run", "job_id": "j1"}, result)) is armed

    def test_cron_from_an_earlier_turn_does_not_count(self, registry, cron):
        args, result = cron(action="create", prompt="Check", schedule="every 10m")
        messages = [*self._cron_turn(args, result)[1:], {"role": "user", "content": "now what"}]
        assert not has_armed_wake_source(_agent(), messages)

    def test_unreadable_source_fails_open(self, registry):
        with patch("tools.async_delegation.has_live_for_session", side_effect=RuntimeError("registry down")):
            assert has_armed_wake_source(_agent(), [])

    def test_active_loop_counts_and_an_unreadable_schedule_store_fails_open(self, registry):
        rows = {"loop:sess-1": json.dumps({"status": "active", "prompt": "check"})}
        db = SimpleNamespace(get_meta=rows.get)
        with patch("hermes_cli.goals._get_session_db", return_value=db):
            assert has_armed_wake_source(_agent(), [])
            rows["loop:sess-1"] = json.dumps({"status": "cleared"})
            assert not has_armed_wake_source(_agent(), [])

        def unavailable(_key):
            raise RuntimeError("DB unavailable")

        with patch("hermes_cli.goals._get_session_db", return_value=SimpleNamespace(get_meta=unavailable)):
            assert has_armed_wake_source(_agent(), [])
        with patch("hermes_cli.goals._get_session_db", return_value=None):
            assert has_armed_wake_source(_agent(), [])


# ── through the real turn loop ─────────────────────────────────────────────

PROMISE = "The review is running in the background. I'll report back when it finishes."


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, platform="desktop")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}
    return agent


def _run(agent, stages):
    from tests.agent.test_run_agent import _mock_response
    agent.client.chat.completions.create.side_effect = [_mock_response(content=t, finish_reason="stop")
                                                        for t in stages]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("start the review")


def _last_user_row_sent(agent, call_index):
    kwargs = agent.client.chat.completions.create.call_args_list[call_index].kwargs
    return [m["content"] for m in kwargs["messages"] if m.get("role") == "user"][-1]


def test_unbacked_promise_is_reprompted_once(loop_agent):
    result = _run(loop_agent, [PROMISE, "Review started; ask me for the result when you want it."])

    assert loop_agent.client.chat.completions.create.call_count == 2
    nudge = _last_user_row_sent(loop_agent, 1)
    assert "nothing will run you again" in nudge and "I'll report back when it finishes" in nudge
    assert result["final_response"] == "Review started; ask me for the result when you want it."


def test_persistent_promise_is_bounded(loop_agent):
    """The shared continuation budget caps it: a model that keeps promising still ends the turn."""
    result = _run(loop_agent, [PROMISE, PROMISE, PROMISE])

    assert loop_agent.client.chat.completions.create.call_count == 3
    assert result["final_response"] == PROMISE


def test_armed_wake_source_lets_the_promise_stand(loop_agent):
    with patch("agent.turn_followup_promise.has_armed_wake_source", return_value=True):
        result = _run(loop_agent, [PROMISE])

    assert loop_agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == PROMISE


def test_stall_guards_off_disables_it(loop_agent):
    loop_agent._stall_guards = False
    _run(loop_agent, [PROMISE])

    assert loop_agent.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("attrs", [{"platform": "cron"}, {"_delegate_depth": 1}])
def test_cron_runs_and_delegated_children_are_exempt(loop_agent, attrs):
    for name, value in attrs.items():
        setattr(loop_agent, name, value)
    _run(loop_agent, [PROMISE])

    assert loop_agent.client.chat.completions.create.call_count == 1
