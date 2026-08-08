"""Cron workdir isolation regressions across terminal environment reuse.

Workdir cron jobs pin their cwd via ContextVars for the whole agent run.  The
terminal dispatch path must treat that cron cwd as authoritative over a cached
environment's mutable ``env.cwd`` while leaving ordinary interactive cwd
persistence intact.
"""

import concurrent.futures
import json
import sys
import threading


def test_concurrent_cron_workdirs_override_reused_terminal_environment(
    tmp_path, monkeypatch
):
    """Two cron agents sharing ``default`` execute terminal commands in their own cwd."""
    from unittest.mock import MagicMock

    import cron.scheduler as sched
    import tools.terminal_tool as terminal_tool
    from tools.environments.local import LocalEnvironment

    workdirs = {name: tmp_path / name for name in ("A", "B")}
    for workdir in workdirs.values():
        workdir.mkdir()
    stale_cwd = tmp_path / "stale"
    stale_cwd.mkdir()

    # Reproduce the reviewer's routing path: both top-level cron agents collapse
    # to the same pre-existing ``default`` environment.  The barrier is inside
    # execute(), after terminal_tool has resolved each command cwd, so overlap is
    # deterministic without sleeps.
    env = LocalEnvironment(cwd=str(stale_cwd), timeout=10)
    real_execute = env.execute
    dispatch_barrier = threading.Barrier(2)

    def execute_together(command, **kwargs):
        dispatch_barrier.wait(timeout=5)
        return real_execute(command, **kwargs)

    env.execute = execute_together
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0.0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(stale_cwd),
            "timeout": 10,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    from model_tools import handle_function_call

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *_args, **_kwargs):
            result = json.loads(
                handle_function_call(
                    "terminal",
                    {"command": "pwd"},
                    task_id=self.session_id,
                    session_id=self.session_id,
                )
            )
            assert result["exit_code"] == 0, result
            return {"final_response": result["output"].strip(), "messages": []}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake_run_agent = type(sys)("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(sched, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        sched, "_build_job_prompt", lambda job, prerun_script=None: "run pwd"
    )
    monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
    monkeypatch.setattr(sched, "_teardown_cron_agent", lambda agent, job_id: None)
    monkeypatch.setattr("hermes_state.SessionDB", lambda: MagicMock())
    monkeypatch.setattr("hermes_cli.env_loader.load_hermes_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache", lambda: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "test",
            "api_key": "key",
            "base_url": "http://test.local",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    jobs = [
        {
            "id": name.lower(),
            "name": name,
            "prompt": "run pwd",
            "workdir": str(workdir),
            "schedule_display": "manual",
        }
        for name, workdir in workdirs.items()
    ]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(sched.run_job, jobs))
    finally:
        env.cleanup()

    assert all(success for success, *_rest in results), results
    observed = {job["id"]: result[2] for job, result in zip(jobs, results)}
    assert observed == {
        "a": str(workdirs["A"].resolve()),
        "b": str(workdirs["B"].resolve()),
    }
    assert set(terminal_tool._active_environments) == {"default"}


def test_run_job_resets_authoritative_session_cwd_when_body_raises(tmp_path):
    """A raising cron run must reset its cwd and command-authority marker."""
    from unittest.mock import MagicMock, patch
    import cron.scheduler as sched
    from agent.runtime_cwd import _SESSION_CWD, _SESSION_CWD_AUTHORITATIVE

    workdir = tmp_path / "proj"
    workdir.mkdir()
    job = {"id": "boom-job", "name": "boom", "prompt": "hi", "workdir": str(workdir)}

    # Force a raise immediately after the authoritative cwd is installed —
    # the exact window where leaked state would corrupt a later command.
    real_info = sched.logger.info

    def _raise_on_workdir_log(msg, *args, **kwargs):
        if isinstance(msg, str) and "using workdir" in msg:
            raise RuntimeError("boom")
        return real_info(msg, *args, **kwargs)

    _SESSION_CWD.set("")
    _SESSION_CWD_AUTHORITATIVE.set(False)

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch.object(sched.logger, "info", side_effect=_raise_on_workdir_log), \
         patch("hermes_state.SessionDB", return_value=MagicMock()):
        # run_job catches its own body exceptions and returns a failure tuple;
        # both ContextVars must be cleared on that path.
        success, _out, _final, _err = sched.run_job(job)

    assert success is False

    assert _SESSION_CWD.get() == "", "_SESSION_CWD leaked after cron failure"
    assert _SESSION_CWD_AUTHORITATIVE.get() is False, (
        "cron command-authority marker leaked after failure"
    )
