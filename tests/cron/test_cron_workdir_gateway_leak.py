"""Regression for #69396: cron workdir must not leak into gateway sessions.

A workdir cron job used to mutate process-global ``os.environ["TERMINAL_CWD"]``
for the duration of its agent run. Interactive gateway sessions created in that
window inherited the job's directory via ``resolve_context_cwd()`` and loaded
its ``AGENTS.md`` as ``# Project Context`` — then followed the cron task
instead of the user's message.

The fix pins cron workdir on the task-local ``_SESSION_CWD`` ContextVar (and a
per-session cwd record) and leaves ``TERMINAL_CWD`` alone.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def gateway_and_cron_dirs(tmp_path):
    gateway = tmp_path / "gateway-home"
    cron = tmp_path / "cron-brief"
    gateway.mkdir()
    cron.mkdir()
    (cron / "AGENTS.md").write_text(
        "# Daily Brief\nProduce a SHORT home + server brief...\n",
        encoding="utf-8",
    )
    (gateway / "AGENTS.md").write_text(
        "# Gateway workspace\nAnswer the user's chat.\n",
        encoding="utf-8",
    )
    return gateway, cron


def test_run_job_workdir_does_not_mutate_terminal_cwd(gateway_and_cron_dirs, monkeypatch):
    """While a workdir job runs, process-global TERMINAL_CWD stays at the
    gateway/scheduler value — concurrent resolve_context_cwd() callers without
    a session pin must not see the cron directory.
    """
    import cron.scheduler as sched
    from agent.prompt_builder import build_context_files_prompt
    from agent.runtime_cwd import resolve_context_cwd

    gateway, cron = gateway_and_cron_dirs
    monkeypatch.setenv("TERMINAL_CWD", str(gateway))

    observed: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            observed["skip_context_files"] = kwargs.get("skip_context_files")
            observed["terminal_cwd_during_init"] = os.environ.get("TERMINAL_CWD")
            observed["context_cwd_during_init"] = resolve_context_cwd()
            # Simulate a concurrent gateway session that has no session cwd pin
            # (the pre-fix path): it must still resolve to the gateway cwd.
            observed["peer_context_cwd"] = None
            observed["peer_prompt_has_cron_agents"] = None

            def _peer_observe():
                observed["peer_context_cwd"] = resolve_context_cwd()
                prompt = build_context_files_prompt(
                    cwd=str(observed["peer_context_cwd"])
                    if observed["peer_context_cwd"]
                    else None
                )
                observed["peer_prompt_has_cron_agents"] = (
                    "Produce a SHORT home + server brief" in prompt
                )
                observed["peer_prompt_has_gateway_agents"] = (
                    "Answer the user's chat" in prompt
                )

            # Run the peer observation on a fresh thread with a CLEARED
            # ContextVar (no inherited cron pin) — matches a gateway handler
            # that never called set_session_cwd(cron_workdir).
            t = threading.Thread(target=_peer_observe)
            t.start()
            t.join(timeout=5)

        def run_conversation(self, *_a, **_kw):
            observed["terminal_cwd_during_run"] = os.environ.get("TERMINAL_CWD")
            observed["context_cwd_during_run"] = resolve_context_cwd()
            return {"final_response": "done", "messages": []}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    import sys

    fake_mod = type(sys)("run_agent")
    fake_mod.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

    from hermes_cli import runtime_provider as _rtp

    monkeypatch.setattr(
        _rtp,
        "resolve_runtime_provider",
        lambda **_kw: {
            "provider": "test",
            "api_key": "k",
            "base_url": "http://test.local",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
    monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv", lambda **_kw: None
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache", lambda: None
    )

    job = {
        "id": "leak-job",
        "name": "leak-job",
        "workdir": str(cron),
        "schedule_display": "manual",
    }

    success, *_ = sched.run_job(job)
    assert success is True

    # Cron agent sees the workdir via ContextVar.
    assert observed["skip_context_files"] is False
    assert observed["context_cwd_during_init"] == cron.resolve()
    assert observed["context_cwd_during_run"] == cron.resolve()

    # Process-global env never flipped to the cron workdir.
    assert observed["terminal_cwd_during_init"] == str(gateway)
    assert observed["terminal_cwd_during_run"] == str(gateway)
    assert os.environ["TERMINAL_CWD"] == str(gateway)

    # Concurrent unpinned peer kept the gateway cwd + AGENTS.md.
    assert observed["peer_context_cwd"] == gateway.resolve()
    assert observed["peer_prompt_has_cron_agents"] is False
    assert observed["peer_prompt_has_gateway_agents"] is True


def test_run_job_workdir_leaves_terminal_cwd_untouched_on_error(
    gateway_and_cron_dirs, monkeypatch
):
    """Even when the workdir log raises, TERMINAL_CWD must be unchanged and
    the writer lock must still be released (lock-leak regression from #56265).
    """
    import cron.scheduler as sched

    gateway, cron = gateway_and_cron_dirs
    monkeypatch.setenv("TERMINAL_CWD", str(gateway))

    job = {
        "id": "boom-job",
        "name": "boom",
        "prompt": "hi",
        "workdir": str(cron),
    }

    real_info = sched.logger.info

    def _raise_on_workdir_log(msg, *args, **kwargs):
        if isinstance(msg, str) and "using workdir" in msg:
            raise RuntimeError("boom")
        return real_info(msg, *args, **kwargs)

    with patch("cron.scheduler._hermes_home", gateway.parent), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch.object(sched.logger, "info", side_effect=_raise_on_workdir_log), \
         patch("hermes_state.SessionDB", return_value=MagicMock()):
        success, *_ = sched.run_job(job)

    assert success is False
    assert os.environ["TERMINAL_CWD"] == str(gateway)

    acquired = threading.Event()

    def try_acquire():
        sched._terminal_cwd_lock.acquire_write()
        try:
            acquired.set()
        finally:
            sched._terminal_cwd_lock.release_write()

    t = threading.Thread(target=try_acquire, daemon=True)
    t.start()
    assert acquired.wait(timeout=5), "writer lock was leaked by run_job on exception"
    t.join(timeout=5)


def test_no_agent_workdir_does_not_chdir_process(tmp_path, monkeypatch):
    """no_agent jobs must pass workdir as subprocess cwd, not os.chdir()."""
    import cron.scheduler as sched

    workdir = tmp_path / "brief"
    workdir.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "probe.py"
    script.write_text(
        "import os\nprint(os.getcwd())\n",
        encoding="utf-8",
    )

    prior = os.getcwd()
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)

    captured: dict = {}

    def _capture_run(*args, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        captured["process_cwd"] = os.getcwd()

        class _R:
            returncode = 0
            stdout = kwargs["cwd"] or ""
            stderr = ""

        return _R()

    monkeypatch.setattr(sched.subprocess, "run", _capture_run)

    job = {
        "id": "na",
        "name": "na",
        "no_agent": True,
        "script": "probe.py",
        "workdir": str(workdir),
        "schedule": {"kind": "cron", "expr": "0 0 * * *"},
    }
    ok, doc, final, err = sched.run_job(job)
    assert ok is True, err
    assert Path(captured["cwd"]).resolve() == workdir.resolve()
    assert Path(captured["process_cwd"]).resolve() == Path(prior).resolve()
    assert Path(os.getcwd()).resolve() == Path(prior).resolve()
