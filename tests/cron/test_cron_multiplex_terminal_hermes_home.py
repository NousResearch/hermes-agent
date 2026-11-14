"""#77464 repro: under a multiplexed host, a cron job for a secondary profile whose agent runs a
terminal command must see THAT profile's HERMES_HOME — through the real owner bridge
(``gateway.session_cron.execute`` -> ``_profile_runtime_scope`` -> ``asyncio.to_thread(run_job)``)
and the real terminal tool / local backend."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    # The owner bridge scopes to ``state.db``'s directory, so each profile's db lives in its home;
    # the Hermes root sits beside (not at) ``Path.home()/.hermes`` so the live-db guard allows it.
    root = tmp_path / "hermes-root"
    beta = root / "profiles" / "beta"
    gamma = root / "profiles" / "gamma"
    (tmp_path / "user").mkdir()
    for home in (root, beta, gamma):
        (home / "cron").mkdir(parents=True)
        (home / "config.yaml").write_text("model:\n  default: test-model\n  provider: test\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    from agent.secret_scope import set_multiplex_active
    set_multiplex_active(True)
    from tools.terminal_tool import cleanup_all_environments
    try:
        yield root, beta, gamma
    finally:
        cleanup_all_environments()  # module-global env cache: a kept shell would outlive the test
        set_multiplex_active(False)


def _install_agent(monkeypatch, seen):
    import cron.scheduler as sched
    from cron import scheduler_delivery as sched_delivery
    from hermes_cli import runtime_provider as rtp

    class TerminalAgent:
        def __init__(self, **_kw):
            pass

        def run_conversation(self, *_a, task_id=None, **_kw):
            from tools.terminal_tool import terminal_tool
            out = json.loads(terminal_tool("echo HH=$HERMES_HOME", task_id=task_id))
            seen.append(out["output"].strip().rpartition("HH=")[2])
            return {"final_response": "done", "messages": [{"role": "assistant", "content": "done"}]}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake = type(sys)("run_agent")
    fake.AIAgent = TerminalAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake)
    monkeypatch.setattr(rtp, "resolve_runtime_provider", lambda **_kw: {
        "provider": "test", "api_key": "k", "base_url": "http://test.local", "api_mode": "chat_completions"})
    monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None, **kw: "hi")
    monkeypatch.setattr(sched_delivery, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)


def _fire(home: Path, job_id: str):
    """Run one job through the owner bridge for *home*'s authority."""
    from gateway.session_contract import SessionRef
    from gateway.session_cron import execute
    from hermes_state_registry import acquire, release

    job = {"id": job_id, "name": job_id, "schedule_display": "manual", "model": "test-model"}
    ref = SessionRef(home.name, f"cron-{job_id}")
    db = acquire(home / "state.db")
    db.create_session(ref.session_id, source="cron")
    authority = SimpleNamespace(
        db=db, profile_id=home.name, pending_results={},
        sessions={ref.session_id: SimpleNamespace(source=SimpleNamespace(user_id="cron-owner"))})
    row = {"admission_id": f"adm-{job_id}", "request_id": f"req-{job_id}", "principal_id": "cron-owner",
           "owner_epoch": 1, "generation": 1, "payload": {"text": ""}}
    policy = SimpleNamespace(request_json=json.dumps(
        {"cron_job": job, "extra_prompt": None, "request_id": f"req-{job_id}"}))
    try:
        asyncio.run(execute(authority, ref, row, policy))
    finally:
        release(db)


def test_secondary_profile_cron_terminal_sees_its_own_hermes_home(two_profiles, monkeypatch):
    root, beta, gamma = two_profiles
    seen: list[str] = []
    _install_agent(monkeypatch, seen)
    # Launch profile first, then two secondaries, then back. A lost scope at any hop, or two homes
    # sharing one terminal environment (its bash snapshot pins the FIRST creator's HERMES_HOME and
    # is sourced over the per-command env), shows up as a repeated, wrong home.
    for home, job_id in ((root, "j-root"), (beta, "j-beta"), (gamma, "j-gamma"), (beta, "j-beta2")):
        _fire(home, job_id)
    assert [Path(p).resolve() for p in seen] == [
        root.resolve(), beta.resolve(), gamma.resolve(), beta.resolve()]
