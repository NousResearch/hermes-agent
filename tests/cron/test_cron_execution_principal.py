"""Authenticated cron execution principal (#130722).

Covers the lifecycle, isolation, and non-leakage contract for
``cron/execution_authority.py`` plus the scheduler integration in
``cron/scheduler.py::_run_one_job_body``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cron.executions as executions
import cron.jobs as cron_jobs
import cron.scheduler as sched
from cron.execution_authority import (
    _GRANT_VAR,
    bump_runtime_generation_for_tests,
    get_runtime_generation,
    scoped_execution_grant,
    try_mint_grant,
    verify_cron_execution,
)


def _aware_instant() -> str:
    return datetime.now(timezone.utc).isoformat()


def _home():
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()).resolve()


@pytest.fixture(autouse=True)
def _clean_grant():
    # ContextVar is process/thread-local: never leak a grant between tests.
    token = _GRANT_VAR.set(None)
    try:
        yield
    finally:
        try:
            _GRANT_VAR.reset(token)
        except Exception:
            _GRANT_VAR.set(None)


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    """Point both the jobs store and the executions ledger at one temp home."""
    home = tmp_path / "cronhome"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # cron.jobs honours the override; executions follows get_hermes_home().
    with cron_jobs.use_cron_store(home):
        yield home


def _claimed(job_id: str, *, source: str = "builtin", instant=None):
    if instant is None and source == "builtin":
        instant = _aware_instant()
    return executions.create_execution(job_id, source=source, scheduled_instant=instant)


def _running(job_id: str, *, source: str = "builtin", instant=None):
    row = _claimed(job_id, source=source, instant=instant)
    running = executions.mark_execution_running(row["id"])
    assert running is not None and running["status"] == "running"
    return running


# --- genuine scheduled occurrence -------------------------------------------


def test_genuine_builtin_scheduled_mints_and_verifies(cron_home):
    row = _running("job-genuine")
    token = try_mint_grant(row["id"])
    assert token is not None
    try:
        proj = verify_cron_execution(expected_job_id="job-genuine")
        assert proj is not None
        assert proj.job_id == "job-genuine"
        assert proj.execution_id == row["id"]
        assert proj.source == "builtin"
        assert proj.scheduled_instant == row["scheduled_instant"]
        assert proj.profile_home == str(_home())
        assert proj.ownership_generation == get_runtime_generation()
        assert proj.purpose_digest is None
        # No nonce on the projection.
        assert "nonce" not in proj.to_dict()
        assert "nonce" not in json.dumps(proj.to_dict())
    finally:
        _GRANT_VAR.reset(token)


def test_projection_is_readonly_and_purpose_bound(cron_home):
    row = _running("job-purpose")
    token = try_mint_grant(row["id"])
    assert token is not None
    try:
        digest = hashlib.sha256(b"exact low-risk plan").hexdigest()
        proj = verify_cron_execution(expected_job_id="job-purpose", purpose_digest=digest)
        assert proj is not None and proj.purpose_digest == digest
        with pytest.raises(Exception):
            proj.job_id = "other"  # frozen
        # Wrong job binding fails.
        assert verify_cron_execution(expected_job_id="other-job") is None
        # Empty purpose fails closed.
        assert verify_cron_execution(expected_job_id="job-purpose", purpose_digest="  ") is None
        assert verify_cron_execution(expected_job_id="job-purpose", purpose_digest=123) is None
        # A different purpose yields a different projection, not a replay.
        other = verify_cron_execution(expected_job_id="job-purpose", purpose_digest=hashlib.sha256(b"other op").hexdigest())
        assert other is not None and other.purpose_digest != digest
    finally:
        _GRANT_VAR.reset(token)


def test_source_builtin_alone_is_insufficient(cron_home):
    # Off-schedule builtin run: scheduled_instant NULL carries no identity.
    row = executions.create_execution("job-offschedule", source="builtin", scheduled_instant=None)
    assert executions.mark_execution_running(row["id"]) is not None
    assert try_mint_grant(row["id"]) is None
    assert verify_cron_execution(expected_job_id="job-offschedule") is None


def test_manual_direct_and_external_sources_never_mint(cron_home):
    for source, job in (("direct", "job-direct"), ("chronos", "job-ext"), ("builtin-manual", "job-man")):
        src = "direct" if source == "direct" else ("chronos" if source == "chronos" else "builtin")
        instant = None if source != "chronos" else _aware_instant()
        # Manual builtin fires carry no instant; direct/external carry their own source.
        row = executions.create_execution(job, source=src, scheduled_instant=instant)
        assert executions.mark_execution_running(row["id"]) is not None
        # External provider with an instant is still ineligible (source != builtin).
        # Direct/manual with no instant are ineligible.
        eligible = src == "builtin" and instant is not None
        token = try_mint_grant(row["id"])
        if eligible:
            assert token is not None
            _GRANT_VAR.reset(token)
        else:
            assert token is None
        assert verify_cron_execution(expected_job_id=job) is None


def test_claimed_row_cannot_mint_before_running(cron_home):
    row = _claimed("job-claimed")
    assert row["status"] == "claimed"
    assert try_mint_grant(row["id"]) is None
    assert verify_cron_execution() is None


def test_ordinary_chat_has_no_authority(cron_home):
    assert verify_cron_execution() is None
    assert verify_cron_execution(expected_job_id="anything") is None


# --- negative: forgery -------------------------------------------------------


def test_copied_task_id_and_forged_env_cannot_validate(cron_home, monkeypatch):
    row = _running("job-real")
    token = try_mint_grant(row["id"])
    assert token is not None
    task_id = f"cron:{row['job_id']}:{row['id']}"
    _GRANT_VAR.reset(token)
    # Outside the grant, even with the exact task_id and forged env, no authority.
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_ID", "forged")
    assert verify_cron_execution(expected_job_id="job-real") is None
    # Forged hook kwargs are consistency inputs, never the credential.
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    seen = []

    def _recorder(tool_name, args, **kw):
        seen.append(verify_cron_execution(expected_job_id="job-real"))
        return None

    from hermes_cli import plugins as plugins_mod

    mgr = plugins_mod.get_plugin_manager()
    mgr._hooks.setdefault("pre_tool_call", []).append(_recorder)
    try:
        _dispatch_pre_tool_call_hooks(
            "some_tool", {}, task_id=task_id, session_id="forged",
            tool_call_id="x", turn_id="y", api_request_id="z",
        )
    finally:
        try:
            mgr._hooks["pre_tool_call"].remove(_recorder)
        except ValueError:
            pass
    assert seen == [None]


def test_synthetic_hook_payload_cannot_mint_or_validate(cron_home):
    from hermes_cli import plugins as plugins_mod

    mgr = plugins_mod.get_plugin_manager()
    observed = []

    def _hook(tool_name, args, **kw):
        observed.append(verify_cron_execution())
        return None

    mgr._hooks.setdefault("pre_tool_call", []).append(_hook)
    try:
        # Synthetic payload outside any grant.
        plugins_mod.invoke_hook(
            "pre_tool_call", tool_name="t", args={},
            task_id="cron:job-real:deadbeef", session_id="s",
            tool_call_id="c", turn_id="t", api_request_id="a",
        )
        assert observed == [None]
        # Inside a real grant the same hook observes the projection.
        row = _running("job-hook-real")
        with scoped_execution_grant(row["id"]):
            plugins_mod.invoke_hook(
                "pre_tool_call", tool_name="t", args={},
                task_id=f"cron:{row['job_id']}:{row['id']}", session_id="s",
                tool_call_id="c", turn_id="t", api_request_id="a",
            )
            assert observed[-1] is not None
            assert observed[-1].job_id == "job-hook-real"
    finally:
        try:
            mgr._hooks["pre_tool_call"].remove(_hook)
        except ValueError:
            pass


# --- isolation: delegation, subprocess, nested process ------------------------


def test_delegated_child_cannot_observe_grant(cron_home):
    from agent.delegation_context import delegated_child_context

    row = _running("job-deleg")
    with scoped_execution_grant(row["id"]):
        assert verify_cron_execution(expected_job_id="job-deleg") is not None
        with delegated_child_context("child-session"):
            assert verify_cron_execution(expected_job_id="job-deleg") is None
            assert verify_cron_execution() is None
        # Parent authority restored after the child scope.
        assert verify_cron_execution(expected_job_id="job-deleg") is not None


def test_terminal_subprocess_env_carries_no_grant(cron_home):
    row = _running("job-terminal")
    with scoped_execution_grant(row["id"]):
        grant = _GRANT_VAR.get()
        assert grant is not None
        nonce = grant.nonce
        assert nonce
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env(scrub_secrets=True, inherit_profile_home=True)
        blob = json.dumps({k: v for k, v in env.items()}, ensure_ascii=False)
        assert nonce not in blob
        for key in env:
            assert "CRON_GRANT" not in key.upper()
            assert "CRON_NONCE" not in key.upper()
            assert "CRON_EXECUTION_GRANT" not in key.upper()
        # The nonce never appears in hook payload serialization either.
        from hermes_cli import plugins as plugins_mod

        payload = {"task_id": f"cron:{row['job_id']}:{row['id']}", "session_id": "s"}
        assert nonce not in json.dumps(payload)


def test_nested_hermes_process_cannot_validate(cron_home):
    row = _running("job-nested")
    with scoped_execution_grant(row["id"]):
        grant = _GRANT_VAR.get()
        assert grant is not None
        nonce = grant.nonce
        home = str(_home())
        repo = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["HERMES_HOME"] = home
        env["PYTHONPATH"] = repo
        code = (
            "from cron.execution_authority import verify_cron_execution; "
            "print(verify_cron_execution())"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=repo, env=env,
            text=True, capture_output=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "None"
        assert nonce not in (proc.stdout + proc.stderr)


def test_registered_cli_command_context_has_no_grant(cron_home):
    # CLI commands run in a separate process / without the grant ContextVar.
    # Simulating the child side: no grant is visible even with the same home.
    home = str(_home())
    repo = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["PYTHONPATH"] = repo
    code = (
        "from hermes_cli.plugins import PluginContext, PluginManager; "
        "from hermes_cli.plugins_manifest import PluginManifest; "
        "mgr=PluginManager(); "
        "ctx=PluginContext(PluginManifest(name='probe', source='user'), mgr); "
        "print(ctx.runtime.verify_cron_execution())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env,
        text=True, capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "None"


# --- restart-safe worker -----------------------------------------------------


def test_worker_mints_only_after_adoption_cas(cron_home):
    row = _claimed("job-handoff")
    # Claimed but not running: no grant.
    assert try_mint_grant(row["id"]) is None
    assert executions.mark_execution_handoff_pending(row["id"]) is not None
    # Handoff pending (still claimed): adoption CAS not yet won.
    assert try_mint_grant(row["id"]) is None
    adopted = executions.adopt_claimed_execution(row["id"])
    assert adopted is not None and adopted["status"] == "running"
    token = try_mint_grant(row["id"])
    assert token is not None
    try:
        assert verify_cron_execution(expected_job_id="job-handoff") is not None
    finally:
        _GRANT_VAR.reset(token)


def test_external_worker_payload_carries_no_bearer(cron_home):
    job = {"id": "job-payload", "execution_id": "exec-123", "prompt": "x"}
    payload = {
        "job": job,
        "profile_home": str(_home()),
        "multiplex_active": False,
    }
    blob = json.dumps(payload)
    assert "nonce" not in blob.lower()
    assert "grant" not in blob.lower()
    # The dispatcher never serializes authority: the job dict itself has none.
    assert "nonce" not in json.dumps(job).lower()


# --- multi-profile isolation -------------------------------------------------


def test_two_profiles_cannot_observe_each_other(cron_home, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_a = cron_home
    home_b = tmp_path / "homeb"
    (home_b / "cron").mkdir(parents=True)
    tok_b = set_hermes_home_override(str(home_b))
    try:
        row_b = _running("shared-name")
    finally:
        reset_hermes_home_override(tok_b)
    row_a = _running("shared-name")
    assert row_a["id"] != row_b["id"]
    with scoped_execution_grant(row_a["id"]):
        assert verify_cron_execution(expected_job_id="shared-name") is not None
        proj_a = verify_cron_execution(expected_job_id="shared-name")
        assert proj_a.execution_id == row_a["id"]
        # Same job name in the other profile is a different execution: bound.
        assert proj_a.execution_id != row_b["id"]
        tok = set_hermes_home_override(str(home_b))
        try:
            # In profile B the A-grant is invisible (different ledger + key).
            assert verify_cron_execution(expected_job_id="shared-name") is None
        finally:
            reset_hermes_home_override(tok)
        # Back in A, authority is still valid (still owned, still running).
        assert verify_cron_execution(expected_job_id="shared-name") is not None


def test_profile_scope_change_invalidates(cron_home, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    row = _running("job-profile")
    with scoped_execution_grant(row["id"]):
        assert verify_cron_execution() is not None
        other = tmp_path / "otherhome"
        (other / "cron").mkdir(parents=True)
        tok = set_hermes_home_override(str(other))
        try:
            assert verify_cron_execution() is None
        finally:
            reset_hermes_home_override(tok)


# --- invalidation ------------------------------------------------------------


def test_terminal_row_invalidates_immediately(cron_home):
    row = _running("job-terminal-state")
    with scoped_execution_grant(row["id"]):
        assert verify_cron_execution() is not None
        assert executions.finish_execution(row["id"], success=True) is not None
        assert verify_cron_execution() is None
    # A new grant cannot be minted for the terminal row.
    assert try_mint_grant(row["id"]) is None


def test_ownership_loss_invalidates(cron_home, monkeypatch):
    row = _running("job-ownership")
    with scoped_execution_grant(row["id"]):
        assert verify_cron_execution() is not None
        # Simulate another process stealing the row: rewrite owner.
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET process_id=?, pid=? WHERE id=?",
                ("foreign-process", 999999, row["id"]),
            )
        assert verify_cron_execution() is None


def test_recycled_pid_with_stale_start_fingerprint_invalidates(cron_home):
    # A recycled PID keeps the process_id/pid numbers but is a different
    # incarnation; ownership is ``(pid, started_at)`` (cron/AGENTS.md), so a
    # running row with a stale start-time fingerprint must not verify.
    current = executions._process_start_time(os.getpid())
    if current is None:
        pytest.skip("start-time fingerprint unavailable on this host")
    row = _running("job-recycled")
    with scoped_execution_grant(row["id"]):
        assert verify_cron_execution() is not None
        # Same process_id/pid, but the row names an older incarnation.
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET process_started_at=? WHERE id=?",
                (current - 10000, row["id"]),
            )
        assert executions._owner_is_live(os.getpid(), current - 10000) is False
        assert verify_cron_execution() is None


def test_generation_change_invalidates(cron_home):
    row = _running("job-generation")
    with scoped_execution_grant(row["id"]):
        assert verify_cron_execution() is not None
        bump_runtime_generation_for_tests()
        assert verify_cron_execution() is None


# --- trusted plugin context --------------------------------------------------


def test_plugin_runtime_verify_and_fail_closed_veto(cron_home):
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest
    from tools.registry import registry

    mgr = PluginManager()
    ctx = PluginContext(PluginManifest(name="probe-plugin", source="user"), mgr)
    assert hasattr(ctx, "runtime")
    assert verify_cron_execution() is None
    assert ctx.runtime.verify_cron_execution() is None

    row = _running("job-plugin-ctx")
    with scoped_execution_grant(row["id"]):
        proj = ctx.runtime.verify_cron_execution(expected_job_id="job-plugin-ctx")
        assert proj is not None and proj.job_id == "job-plugin-ctx"
        assert "nonce" not in proj.to_dict()

    # Fail-closed pre_tool_call veto is unchanged by the new surface.
    from hermes_cli import plugins as plugins_mod

    def _veto(tool_name, args, **kw):
        if tool_name == "blocked_tool":
            return {"action": "block", "message": "denied by fixture"}
        return None

    mgr2 = plugins_mod.get_plugin_manager()
    mgr2._hooks.setdefault("pre_tool_call", []).append(_veto)
    try:
        block, _ = plugins_mod._dispatch_pre_tool_call_hooks("blocked_tool", {})
        assert block == "denied by fixture"
        block2, _ = plugins_mod._dispatch_pre_tool_call_hooks("allowed_tool", {})
        assert block2 is None
    finally:
        try:
            mgr2._hooks["pre_tool_call"].remove(_veto)
        except ValueError:
            pass


def test_handler_context_is_out_of_band_from_model_args(cron_home):
    from tools.registry import registry

    row = _running("job-handler-ctx")
    seen = {}

    def _handler(args, **kw):
        # Model args carry no authority; verification is out-of-band.
        seen["args_has_nonce"] = "nonce" in json.dumps(args or {})
        seen["proj"] = verify_cron_execution(expected_job_id="job-handler-ctx")
        assert "nonce" not in json.dumps(kw, default=str)
        return json.dumps({"ok": True})

    registry.register(
        name="_cron_principal_probe", toolset="debugging",
        schema={"name": "_cron_principal_probe", "description": "probe",
                "parameters": {"type": "object", "properties": {}}},
        handler=_handler,
    )
    try:
        with scoped_execution_grant(row["id"]):
            # Even when the model forges cron-looking args, only the grant decides.
            result = registry.dispatch(
                "_cron_principal_probe",
                {"task_id": f"cron:{row['job_id']}:{row['id']}", "nonce": "forged"},
            )
            assert json.loads(result).get("ok") is True
            assert seen["args_has_nonce"] is True  # model CAN send the word; it buys nothing
            assert seen["proj"] is not None
            assert seen["proj"].execution_id == row["id"]
        # Outside the grant the same forged args validate nothing.
        seen.clear()
        registry.dispatch("_cron_principal_probe", {"task_id": "cron:job-handler-ctx:" + row["id"]})
        assert seen["proj"] is None
    finally:
        registry.deregister("_cron_principal_probe")


# --- E2E via the real scheduler ----------------------------------------------


def _e2e_job(job_id: str, execution_id: str) -> dict:
    return {
        "id": job_id,
        "name": job_id,
        "prompt": "e2e prompt",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "deliver": "local",
        "execution_id": execution_id,
    }


def test_e2e_genuine_scheduled_run_verifies_in_agent_turn(cron_home):
    """Real ``run_one_job`` -> ``_run_one_job_body`` -> ``run_job`` path.

    Only the agent run is stubbed; durable ownership, grant mint, ContextVar
    propagation, and verification are all real.
    """
    instant = _aware_instant()
    claimed = executions.create_execution("job-e2e-real", source="builtin", scheduled_instant=instant)
    assert claimed["status"] == "claimed"
    job = _e2e_job("job-e2e-real", claimed["id"])
    captured = {}
    purpose = hashlib.sha256(b"exact low-risk plan").hexdigest()

    def fake_run_job(job_arg, **kw):
        captured["execution_id"] = kw.get("execution_id")
        proj = verify_cron_execution(expected_job_id="job-e2e-real", purpose_digest=purpose)
        captured["proj"] = proj
        assert kw.get("execution_id") == claimed["id"]
        return True, "# doc\n\n## Response\n\ndone\n", "done", None

    with cron_jobs.use_cron_store(cron_home), \
         patch.object(sched, "run_job", side_effect=fake_run_job), \
         patch.object(sched, "_deliver_result", return_value=None):
        assert sched.run_one_job(dict(job)) is True
    assert captured["proj"] is not None
    assert captured["proj"].job_id == "job-e2e-real"
    assert captured["proj"].execution_id == claimed["id"]
    assert captured["proj"].scheduled_instant == claimed["scheduled_instant"]
    assert captured["proj"].purpose_digest == purpose
    assert "nonce" not in captured["proj"].to_dict()
    # Terminal write invalidates afterwards.
    assert verify_cron_execution(expected_job_id="job-e2e-real") is None
    row = executions.get_execution(claimed["id"])
    assert row is not None and row["status"] in ("completed", "failed")


def test_e2e_manual_run_has_no_authority(cron_home):
    """Real ``run_one_job`` for a manual (``source=direct``) fire never mints."""
    job = _e2e_job("job-e2e-manual", "")
    job.pop("execution_id")
    captured = {}

    def fake_run_job(job_arg, **kw):
        captured["proj"] = verify_cron_execution(expected_job_id="job-e2e-manual")
        captured["execution_id"] = kw.get("execution_id")
        return True, "# doc\n\n## Response\n\ndone\n", "done", None

    with cron_jobs.use_cron_store(cron_home), \
         patch.object(sched, "run_job", side_effect=fake_run_job), \
         patch.object(sched, "_deliver_result", return_value=None):
        assert sched.run_one_job(dict(job)) is True
    assert captured["proj"] is None
    # The manual execution was still recorded (source=direct).
    row = executions.get_execution(captured["execution_id"])
    assert row is not None and row["source"] == "direct"


def test_e2e_fixture_plugin_tool_verifies_only_in_authorized_turn(cron_home):
    """Fixture plugin tool + pre_tool_call veto through the real dispatch path."""
    from hermes_cli import plugins as plugins_mod
    from tools.registry import registry

    instant = _aware_instant()
    claimed = executions.create_execution("job-e2e-plugin", source="builtin", scheduled_instant=instant)
    job = _e2e_job("job-e2e-plugin", claimed["id"])
    tool_seen = {}
    hook_seen = []

    def _probe_handler(args, **kw):
        tool_seen["proj"] = verify_cron_execution(expected_job_id="job-e2e-plugin")
        return json.dumps({"ok": True})

    def _pre_hook(tool_name, args, **kw):
        # Observe every tool call; never block the probe.
        hook_seen.append(verify_cron_execution(expected_job_id="job-e2e-plugin"))
        if tool_name == "_cron_blocked_fixture":
            return {"action": "block", "message": "denied by fixture"}
        return None

    registry.register(
        name="_cron_probe_fixture", toolset="debugging",
        schema={"name": "_cron_probe_fixture", "description": "probe",
                "parameters": {"type": "object", "properties": {}}},
        handler=_probe_handler,
    )
    mgr = plugins_mod.get_plugin_manager()
    mgr._hooks.setdefault("pre_tool_call", []).append(_pre_hook)
    try:
        def fake_run_job(job_arg, **kw):
            import model_tools

            # Real tool dispatch (pre_tool_call hook + registry handler) inside
            # the authorized grant scope.
            task_id = f"cron:{job_arg['id']}:{kw.get('execution_id')}"
            res = model_tools.handle_function_call(
                "_cron_probe_fixture", {}, task_id=task_id, session_id="s",
            )
            tool_seen["result"] = res
            blocked = model_tools.handle_function_call("_cron_blocked_fixture", {}, task_id=task_id)
            tool_seen["blocked"] = blocked
            return True, "# doc\n\n## Response\n\ndone\n", "done", None

        with cron_jobs.use_cron_store(cron_home), \
             patch.object(sched, "run_job", side_effect=fake_run_job), \
             patch.object(sched, "_deliver_result", return_value=None):
            assert sched.run_one_job(dict(job)) is True
        assert tool_seen["proj"] is not None
        assert tool_seen["proj"].execution_id == claimed["id"]
        assert json.loads(tool_seen["result"]).get("ok") is True
        # The veto still fails closed inside the authorized turn.
        assert "denied by fixture" in tool_seen["blocked"]
        assert hook_seen and all(h is not None for h in hook_seen)
    finally:
        try:
            mgr._hooks["pre_tool_call"].remove(_pre_hook)
        except ValueError:
            pass
        registry.deregister("_cron_probe_fixture")


# --- non-leakage -------------------------------------------------------------


def test_authority_never_in_prompts_history_logs_hook_env_or_handoff(cron_home, caplog):
    import logging

    row = _running("job-leak")
    with scoped_execution_grant(row["id"]):
        grant = _GRANT_VAR.get()
        assert grant is not None
        nonce = grant.nonce
        assert len(nonce) >= 32
        # Prompt building never embeds the grant.
        from cron.scheduler_prompt import _build_job_prompt

        job = {"id": "job-leak", "prompt": "do the thing", "schedule": {"kind": "interval", "minutes": 5}}
        prompt = _build_job_prompt(job)
        assert nonce not in prompt
        assert nonce not in json.dumps(job)
        # Hook payload serialization.
        payload = {"task_id": f"cron:job-leak:{row['id']}", "args": {"x": 1}}
        assert nonce not in json.dumps(payload)
        # Subprocess env.
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env(scrub_secrets=True, inherit_profile_home=True)
        assert nonce not in json.dumps(env)
        # External handoff data.
        handoff = {"job": {"id": "job-leak", "execution_id": row["id"]},
                   "profile_home": str(_home()), "multiplex_active": False}
        assert nonce not in json.dumps(handoff)
        # Logs: mint/verify emit no nonce even at DEBUG.
        with caplog.at_level(logging.DEBUG, logger="cron.execution_authority"):
            verify_cron_execution(expected_job_id="job-leak")
            assert nonce not in caplog.text
        # Session-history shape: grant is not a message field.
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}]
        assert nonce not in json.dumps(messages)


def test_prompt_caching_and_alternation_untouched(cron_home):
    """Grant mint/verify must not mutate prompts or message roles."""
    from hermes_cli.plugins import render_system_prompt_sections

    before = render_system_prompt_sections({})
    row = _running("job-cache")
    with scoped_execution_grant(row["id"]):
        during = render_system_prompt_sections({})
        assert verify_cron_execution(expected_job_id="job-cache") is not None
    after = render_system_prompt_sections({})
    assert [(s.id, s.content) for s in before] == [(s.id, s.content) for s in during]
    assert [(s.id, s.content) for s in before] == [(s.id, s.content) for s in after]
    # Strict alternation helper: a tool result still pairs one-to-one.
    messages = [{"role": "user", "content": "hi"}]
    with scoped_execution_grant(row["id"]):
        messages.append({"role": "assistant", "content": "working",
                         "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]})
        messages.append({"role": "tool", "content": "{}", "tool_call_id": "c1"})
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool"]
    assert all(a != b for a, b in zip(roles, roles[1:]))


def test_grant_repr_redacts_nonce(cron_home):
    row = _running("job-repr")
    with scoped_execution_grant(row["id"]):
        grant = _GRANT_VAR.get()
        assert grant is not None
        assert grant.nonce not in repr(grant)
        assert "<redacted>" in repr(grant)


def test_grant_propagates_to_agent_worker_thread(cron_home):
    """The agent runs on a pool worker via ``copy_context().run``; the grant follows."""
    import concurrent.futures
    import contextvars

    row = _running("job-propagation")
    with scoped_execution_grant(row["id"]):
        ctx = contextvars.copy_context()

        def _verify_in_worker():
            return verify_cron_execution(expected_job_id="job-propagation")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(ctx.run, _verify_in_worker)
            proj = fut.result(timeout=10)
        assert proj is not None and proj.execution_id == row["id"]


def test_concurrent_jobs_in_threads_stay_isolated(cron_home):
    """Two jobs on different threads never observe each other's authority."""
    import concurrent.futures
    import contextvars

    row_a = _running("job-concurrent-a")
    row_b = _running("job-concurrent-b")
    results = {}

    def _run_in_scope(row, key):
        with scoped_execution_grant(row["id"]):
            results[key] = verify_cron_execution()
            # Cross-check inside the scope must fail.
            other = row_b if key == "a" else row_a
            results[key + "-cross"] = verify_cron_execution(expected_job_id=other["job_id"])

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(ctx_a.run, _run_in_scope, row_a, "a")
        fb = pool.submit(ctx_b.run, _run_in_scope, row_b, "b")
        fa.result(timeout=10)
        fb.result(timeout=10)
    assert results["a"] is not None and results["a"].execution_id == row_a["id"]
    assert results["b"] is not None and results["b"].execution_id == row_b["id"]
    assert results["a-cross"] is None
    assert results["b-cross"] is None
