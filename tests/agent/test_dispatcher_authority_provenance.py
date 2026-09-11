"""P1-B regressions: dispatcher authority provenance vs lineage.

Matrix (ACCEPTANCE_MATRIX P1-B):
- dispatcher worker ALLOW
- dispatcher -> delegate ALLOW
- dispatcher -> nested delegate ALLOW
- cron / non-dispatcher DENY (dominating veto at every depth)
- cron -> delegate DENY
- subprocess without ownership proof DENY (generic Kanban env inherited,
  marker consumed/absent, no ContextVar)
- authority probe/import failure DENY
"""
from __future__ import annotations

import pytest

from agent import delegation_context as dc


@pytest.fixture(autouse=True)
def _isolated_authority():
    token = dc._DISPATCHER_AUTHORITY.set(False)
    delegated = dc._DELEGATED_CHILD_CONTEXT.set(False)
    non_disp = dc._NON_DISPATCHER_OWNED_CONTEXT.set(False)
    veto = dc._NON_DISPATCHER_VETO.set(False)
    yield
    dc._DISPATCHER_AUTHORITY.reset(token)
    dc._DELEGATED_CHILD_CONTEXT.reset(delegated)
    dc._NON_DISPATCHER_OWNED_CONTEXT.reset(non_disp)
    dc._NON_DISPATCHER_VETO.reset(veto)


def _worker_env(task_id: str) -> dict[str, str]:
    """Env exactly as _default_spawn builds it for an authorized worker."""
    proof, nonce = dc._dispatcher_ownership_proof(task_id)
    return {
        "HERMES_SESSION_SOURCE": "kanban",
        "HERMES_KANBAN_TASK": task_id,
        "HERMES_KANBAN_WORKSPACE": "/tmp/ws",
        "HERMES_KANBAN_TERMINAL_RUNTIME": "{}",
        dc.DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV: f"{proof}.{nonce}",
    }


def test_dispatcher_worker_allowed():
    env = _worker_env("t_allow")
    tok = dc.bootstrap_dispatcher_authority(task_id="t_allow", environ=env)
    assert dc.has_dispatcher_owned_authority()
    # one-shot: the marker is consumed from the process environment
    assert dc.DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV not in env
    dc.exit_dispatcher_authority(tok)


def test_delegate_of_worker_allowed():
    tok = dc.bootstrap_dispatcher_authority(
        task_id="t_del", environ=_worker_env("t_del")
    )
    with dc.delegated_child_context():
        # delegated_child_context inherits positive parent authority
        assert dc.has_dispatcher_owned_authority()
    dc.exit_dispatcher_authority(tok)


def test_nested_delegate_allowed():
    tok = dc.bootstrap_dispatcher_authority(
        task_id="t_nest", environ=_worker_env("t_nest")
    )
    with dc.delegated_child_context(), dc.delegated_child_context():
        assert dc.has_dispatcher_owned_authority()
    dc.exit_dispatcher_authority(tok)


def test_cron_denied_and_veto_dominates_delegation():
    tok = dc.bootstrap_dispatcher_authority(
        task_id="t_cron", environ=_worker_env("t_cron")
    )
    # cron scheduler enters the dominating veto for job execution
    with dc.non_dispatcher_owned_context(), dc.non_dispatcher_authority_veto():
        assert not dc.has_dispatcher_owned_authority()
        # cron -> delegate stays denied: veto dominates every depth
        with dc.delegated_child_context(), dc.delegated_child_context():
            assert not dc.has_dispatcher_owned_authority()
            with pytest.raises(dc.DispatcherAuthorityError):
                dc.delegated_child_inherits_authority()
    # veto is scoped: outside it the worker's own authority still holds
    assert dc.has_dispatcher_owned_authority()
    dc.exit_dispatcher_authority(tok)


def test_subprocess_without_proof_denied():
    """A child process inherits generic Kanban env but no ContextVar and no
    unconsumed ownership marker — it must not regain authority."""
    fresh_process_env = {
        "HERMES_SESSION_SOURCE": "kanban",
        "HERMES_KANBAN_TASK": "t_child",
        "HERMES_KANBAN_WORKSPACE": "/tmp/ws",
        "HERMES_KANBAN_TERMINAL_RUNTIME": "{}",
        # marker deliberately ABSENT: consumed by the parent bootstrap
    }
    with pytest.raises(dc.DispatcherAuthorityError):
        dc.bootstrap_dispatcher_authority(
            task_id="t_child", environ=fresh_process_env
        )
    assert not dc.has_dispatcher_owned_authority()


def test_forged_or_stale_marker_denied():
    bad = _worker_env("t_bad")
    bad[dc.DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV] = (
        "0" * 32 + "." + "f" * 32  # valid shape, wrong value
    )
    with pytest.raises(dc.DispatcherAuthorityError):
        dc.bootstrap_dispatcher_authority(task_id="t_bad", environ=bad)
    assert not dc.has_dispatcher_owned_authority()


def test_probe_failure_denies():
    original = dc._dispatcher_ownership_proof
    env = _worker_env("t_probe")

    def _explode(_task_id):
        raise RuntimeError("secret scope unavailable")

    dc._dispatcher_ownership_proof = _explode
    try:
        with pytest.raises(dc.DispatcherAuthorityError):
            dc.bootstrap_dispatcher_authority(task_id="t_probe", environ=env)
    finally:
        dc._dispatcher_ownership_proof = original
    assert not dc.has_dispatcher_owned_authority()


def test_bootstrap_task_mismatch_denied_and_consumes_marker():
    env = _worker_env("t_real")
    with pytest.raises(dc.DispatcherAuthorityError):
        dc.bootstrap_dispatcher_authority(task_id="t_other", environ=env)
    assert dc.DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV not in env


def test_terminal_tool_gate_defaults_closed(monkeypatch):
    """terminal_tool's runtime gate denies without positive proof even when
    the full generic dispatcher identity is present in the environment."""
    from tools import terminal_tool

    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_gate")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/ws")
    monkeypatch.setenv("HERMES_KANBAN_TERMINAL_RUNTIME", "{}")
    # No bootstrap ever ran in this (fresh) context.
    assert terminal_tool._kanban_runtime_context_allowed() is False


def test_present_runtime_without_authority_refuses_profile_docker_fallback(
    monkeypatch,
):
    """Inherited runtime env cannot degrade into lower-authority Docker config."""
    from tools import terminal_tool

    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_inherited")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/ws")
    monkeypatch.setenv("HERMES_KANBAN_TERMINAL_RUNTIME", "{}")

    with pytest.raises(RuntimeError, match="without dispatcher authority"):
        terminal_tool._load_kanban_runtime([])
    with pytest.raises(RuntimeError, match="without dispatcher authority"):
        terminal_tool._resolve_container_task_id(None)
