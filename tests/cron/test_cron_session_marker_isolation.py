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
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
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
