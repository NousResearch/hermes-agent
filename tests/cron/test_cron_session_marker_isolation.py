"""Regression tests for the cron-session approval marker.

The default deployment runs the cron ticker in-process inside the gateway, in
the same process that serves interactive users. The "this is a cron context"
marker must therefore be per-job context state, not a process-global env var.
A leaked marker makes the approval gate treat every later interactive user as a
cron session: under cron_mode=deny their dangerous commands are hard-blocked
with a misleading message, and under cron_mode=approve they are auto-approved
with no human prompt.

The pre-fix code set os.environ["HERMES_CRON_SESSION"] in run_job and never
cleared it. The fix carries the marker on a per-job ContextVar instead.
"""

import contextvars
import os
import sys
import types


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import cron.scheduler as cron_scheduler  # noqa: E402
import run_agent  # noqa: E402


_JOB = {"id": "j1", "name": "Marker Test", "prompt": "ping", "model": "test-cron-default-model"}


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        return None


# Per-test ContextVar hygiene (resetting every session var, including the cron
# marker, to its _UNSET default) lives in the shared autouse fixture in
# tests/cron/conftest.py, so it also covers the other cron tests that drive
# the real run_job directly in the pytest context.


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)
    # Accept the full current resolve_runtime_provider keyword surface
    # (requested, explicit_api_key, explicit_base_url, target_model, …).
    # run_job now always passes target_model=…; a narrow lambda raises
    # TypeError before any cron-marker assertion is reached.
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kwargs: {
            "provider": "openai",
            "api_mode": "chat_completions",
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )


def _make_stub_agent(record=None):
    """An AIAgent whose turn returns a canned dict without any network call.

    When *record* is given, it captures whether the approval gate sees a cron
    session while the job's turn is executing (proving cron_mode still applies).
    """

    class _StubAgent(run_agent.AIAgent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("skip_context_files", True)
            kwargs.setdefault("skip_memory", True)
            kwargs.setdefault("max_iterations", 2)
            super().__init__(*args, **kwargs)
            self._cleanup_task_resources = lambda task_id: None
            self._persist_session = lambda messages, history=None: None
            self._save_trajectory = lambda messages, user_message, completed: None

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            if record is not None:
                from tools.approval import _is_cron_session

                record["cron_in_job"] = _is_cron_session()
            return {"final_response": "done", "turn_exit_reason": ""}

    return _StubAgent


def test_run_job_does_not_leak_cron_marker_into_process_env(monkeypatch):
    """run_job must not leave HERMES_CRON_SESSION set in os.environ.

    Fails on the pre-fix code, which set the process-global env var and never
    cleared it, so the in-process gateway ticker leaked it to every later user.
    """
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _make_stub_agent())
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    success, _out, final_response, error = cron_scheduler.run_job(dict(_JOB))

    assert success is True and error is None
    assert final_response == "done"
    assert os.environ.get("HERMES_CRON_SESSION") is None, "cron marker leaked into process env"


def test_bound_gateway_session_not_shadowed_by_in_process_cron(monkeypatch):
    """After a cron job runs in-process, a bound interactive gateway session is
    still recognized as a gateway approval context, not routed to cron_mode.

    Fails on the pre-fix code: the leaked process-global marker made
    _is_gateway_approval_context return False for the real user.
    """
    from gateway.session_context import set_session_vars, clear_session_vars
    from tools.approval import _is_gateway_approval_context

    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _make_stub_agent())
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    cron_scheduler.run_job(dict(_JOB))

    tokens = set_session_vars(platform="telegram", chat_id="c1", chat_name="Chat")
    try:
        assert _is_gateway_approval_context() is True
    finally:
        clear_session_vars(tokens)


def test_marker_is_active_inside_the_cron_job(monkeypatch):
    """The approval gate sees the cron marker while the job's turn executes, so
    cron_mode is still applied to a cron agent's commands."""
    record = {}
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _make_stub_agent(record))
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    cron_scheduler.run_job(dict(_JOB))

    assert record.get("cron_in_job") is True


def test_is_cron_session_prefers_contextvar_then_env(monkeypatch):
    from gateway.session_context import _VAR_MAP, _UNSET
    from tools.approval import _is_cron_session

    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)
    assert _is_cron_session() is False

    _VAR_MAP["HERMES_CRON_SESSION"].set("1")
    assert _is_cron_session() is True

    # An explicitly-set empty value is authoritative (no env fallback) and
    # reads as non-cron. This is exactly why run_job must reset() the marker
    # to its pre-job state instead of set(""), see the test below.
    _VAR_MAP["HERMES_CRON_SESSION"].set("")
    assert _is_cron_session() is False

    # os.environ fallback keeps the standalone `hermes cron` process and tests working
    _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert _is_cron_session() is True


def test_env_fallback_survives_a_completed_run_job(monkeypatch):
    """After a real run_job completes in this context, the os.environ fallback
    still works: an env-marked read (the standalone `hermes cron` process, or
    a test that sets the flag directly) is still classified as cron.

    Fails when run_job resets the marker with set("") instead of
    reset(token): get_session_env treats the explicit "" as authoritative and
    never falls back to os.environ, so every env-marked cron read after the
    first job in this context is misclassified as non-cron, cron_mode is
    skipped, and a dangerous command is auto-approved instead of blocked
    under cron_mode deny.
    """
    from tools.approval import _is_cron_session

    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _make_stub_agent())
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    ok = cron_scheduler.run_job(dict(_JOB))[0]
    assert ok is True
    # The marker is back to its pre-job state, not pinned to an explicit "".
    assert _is_cron_session() is False

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert _is_cron_session() is True, (
        "env-marked cron read misclassified as non-cron after a completed run_job"
    )


def test_cron_marker_isolated_between_contexts(monkeypatch):
    """A marker set inside one job's context is invisible to a sibling context
    (a concurrent gateway request). This per-context isolation is the fix."""
    from gateway.session_context import _VAR_MAP
    from tools.approval import _is_cron_session

    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    def _job_ctx():
        _VAR_MAP["HERMES_CRON_SESSION"].set("1")
        return _is_cron_session()

    ctx = contextvars.copy_context()
    assert ctx.run(_job_ctx) is True
    # Outside that copied context the marker was never set.
    assert _is_cron_session() is False


def test_two_real_run_jobs_isolate_marker_across_contexts(monkeypatch):
    """Two real run_job calls, each dispatched in its own context (as the
    in-process ticker does), keep the marker to their own job: each turn sees
    cron_mode, and neither leaves it set in a sibling context or the base
    thread. Drives the raw ContextVar isolation through the real run_job path,
    not a synthetic set()."""
    from gateway.session_context import set_session_vars, clear_session_vars
    from tools.approval import _is_cron_session, _is_gateway_approval_context

    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    # Job A in its own context sees cron while its turn runs.
    rec_a = {}
    monkeypatch.setattr(run_agent, "AIAgent", _make_stub_agent(rec_a))
    ctx_a = contextvars.copy_context()
    ok_a = ctx_a.run(lambda: cron_scheduler.run_job(dict(_JOB))[0])
    assert ok_a is True and rec_a.get("cron_in_job") is True
    # A's marker never escaped into the base context.
    assert _is_cron_session() is False

    # A concurrent interactive gateway session (base context) stays gateway.
    tokens = set_session_vars(platform="telegram", chat_id="c1", chat_name="Chat")
    try:
        assert _is_gateway_approval_context() is True
    finally:
        clear_session_vars(tokens)

    # Job B in a second context: same story, no bleed from A's run.
    rec_b = {}
    monkeypatch.setattr(run_agent, "AIAgent", _make_stub_agent(rec_b))
    ctx_b = contextvars.copy_context()
    ok_b = ctx_b.run(lambda: cron_scheduler.run_job({**_JOB, "id": "j2"})[0])
    assert ok_b is True and rec_b.get("cron_in_job") is True
    assert _is_cron_session() is False


def test_marker_cleared_even_when_agent_raises(monkeypatch):
    """An exception inside AIAgent.run_conversation still unwinds the marker
    via run_job's finally. A raise mid-tick must not leave the contextvar set
    on a reused loop-thread context (or in os.environ).
    """
    from tools.approval import _is_cron_session

    class _RaisingAgent(run_agent.AIAgent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("skip_context_files", True)
            kwargs.setdefault("skip_memory", True)
            kwargs.setdefault("max_iterations", 2)
            super().__init__(*args, **kwargs)
            self._cleanup_task_resources = lambda task_id: None
            self._persist_session = lambda messages, history=None: None
            self._save_trajectory = lambda messages, user_message, completed: None

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            raise RuntimeError("simulated agent crash mid-tick")

    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _RaisingAgent)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    success, _out, _final, error = cron_scheduler.run_job(dict(_JOB))

    assert success is False
    assert error and "simulated agent crash" in error
    assert _is_cron_session() is False
    assert os.environ.get("HERMES_CRON_SESSION") is None


def test_stale_process_env_does_not_reclassify_bound_gateway_session(monkeypatch):
    """A leftover HERMES_CRON_SESSION=1 in os.environ must not push a bound
    interactive gateway session into the cron approval branch.

    The ContextVar is never bound for that interactive turn, so a naive env
    fallback would fire. Live gateway identity wins: the user stays on the
    interactive approval path instead of the cron hard-block.
    """
    from gateway.session_context import set_session_vars, clear_session_vars
    from tools.approval import (
        _is_cron_session,
        _is_gateway_approval_context,
        check_dangerous_command,
    )

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr("tools.approval._get_cron_approval_mode", lambda: "deny")

    tokens = set_session_vars(platform="telegram", chat_id="c1", chat_name="Chat")
    try:
        assert _is_cron_session() is False
        assert _is_gateway_approval_context() is True
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        msg = (result.get("message") or "").lower()
        # Interactive gateway path asks the user; cron deny would hard-block
        # with "without a user present" and never set approval_required.
        assert "without a user present" not in msg
        assert result.get("status") == "approval_required"
    finally:
        clear_session_vars(tokens)


def test_cron_contextvar_drives_deny_for_dangerous_and_execute_code(monkeypatch):
    """When the per-job ContextVar is bound, cron_mode=deny hard-blocks both
    the terminal dangerous-command path and execute_code. The marker is the
    real policy input, not just a helper used by unit tests.
    """
    from gateway.session_context import _VAR_MAP, _UNSET
    from tools import approval as approval_mod

    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(approval_mod, "_get_cron_approval_mode", lambda: "deny")
    _VAR_MAP["HERMES_CRON_SESSION"].set("1")
    try:
        term = approval_mod.check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert term.get("approved") is False
        assert "without a user present" in (term.get("message") or "").lower()

        code = approval_mod.check_execute_code_guard(
            "import os; os.system('id')", "local"
        )
        assert code.get("approved") is False
        assert "without a user present" in (code.get("message") or "").lower()
    finally:
        _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)


def test_cron_deny_not_bypassed_by_exec_ask_flag(monkeypatch):
    """HERMES_EXEC_ASK=1 must not skip cron-deny in check_all_command_guards.

    Cron often shares a process with the gateway, which can leave interactive
    approval flags set. Cron has no user to answer an ask prompt, so
    cron_mode=deny has to win over the ask gate.
    """
    from gateway.session_context import _VAR_MAP, _UNSET
    from tools import approval as approval_mod

    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval_mod, "_get_cron_approval_mode", lambda: "deny")
    _VAR_MAP["HERMES_CRON_SESSION"].set("1")
    try:
        result = approval_mod.check_all_command_guards("rm -rf /tmp/stuff", "local")
        assert result.get("approved") is False
        assert "without a user present" in (result.get("message") or "").lower()
        assert result.get("status") != "approval_required"
    finally:
        _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)


def test_stale_env_does_not_block_execute_code_in_gateway_session(monkeypatch):
    """A stale HERMES_CRON_SESSION=1 in os.environ must not push a bound
    interactive gateway session's execute_code into the cron deny path.

    Companion to test_stale_process_env_does_not_reclassify_bound_gateway_session
    which covers the same scenario for check_dangerous_command. The
    execute_code guard has its own cron branch, so it needs its own
    assertion that the stale-env fallback is suppressed when a gateway
    session is live.

    Production trigger (#73195): a Feishu user replies to a cron-delivered
    message card. The gateway process still has HERMES_CRON_SESSION=1 in
    os.environ from the prior cron tick, but the interactive turn should
    reach the normal approval path, not the cron hard-block.
    """
    from gateway.session_context import set_session_vars, clear_session_vars
    from tools.approval import check_execute_code_guard

    # Simulate a leaked process env from a prior cron tick
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr("tools.approval._get_cron_approval_mode", lambda: "deny")

    # Gateway binds an interactive session (e.g. Feishu reply)
    tokens = set_session_vars(platform="feishu", chat_id="c1", chat_name="Chat")
    try:
        result = check_execute_code_guard(
            "print('hello world')", "local", has_host_access=False
        )
        # Must NOT be hard-blocked by cron deny. The gateway path returns
        # approval_pending (interactive prompt queued) which is correct.
        msg = (result.get("message") or "").lower()
        assert "without a user present" not in msg, (
            f"Cron deny message reached an interactive gateway session: {result}"
        )
        # approval_pending or approved are both fine — cron hard-block is not.
        assert result.get("approval_pending") or result.get("approved") is True, (
            f"execute_code was cron-blocked despite a live gateway session: {result}"
        )
    finally:
        clear_session_vars(tokens)
