"""A host's cron fires keep the LAUNCH profile's env-only secrets and ``TERMINAL_*``.

Regression for #191 on the cron path. Each fire binds its own secret + terminal scope
(``_run_one_job_body``), and the restart-safe worker rebuilt its scope in a child process; both
built every profile from its files only. Once the host multiplexes a scoped miss no longer reaches
``os.environ``, so the launch profile's cron jobs lost whatever systemd ``Environment=`` / ``op run``
injected (provider keys read as unset, ``TERMINAL_ENV=docker`` ran on the host). A bound terminal
scope is the whole policy on a single-profile host too, so there the same fires ran an env-only
``TERMINAL_ENV=docker`` on the host while the profile's own unscoped turns were sandboxed. The launch
home must bind its files over the launch env (live before activation, frozen at it); a secondary,
including its worker child whose env still carries launch residue, resolves from its own files only.
"""
import contextlib
import json
import os
from pathlib import Path

import pytest

import cron.scheduler as scheduler
import hermes_constants
from agent import secret_scope
from agent.secret_scope import get_secret
from cron.scheduler_provider import _profile_cron_scope
from tools.terminal_scope import terminal_env
from tui_gateway import launch_profile_policy

LAUNCH_ENV = {
    "OPENROUTER_API_KEY": "sk-from-systemd",
    "TERMINAL_ENV": "docker",
    "TERMINAL_LOCAL_MEMORY_MAX_MB": "64",
}
LAUNCH_VIEW = ("sk-from-systemd", "docker", "64")
FILES_ONLY_VIEW = (None, "local", "")


@pytest.fixture
def launch_host(tmp_path, monkeypatch):
    """A host that never multiplexes but still ticks a secondary's store (cron ownership is not
    gated on ``gateway.multiplex_profiles``)."""
    root = tmp_path / ".hermes"
    coder = root / "profiles" / "coder"
    for home in (root, coder):
        (home / "cron").mkdir(parents=True)
    (coder / ".env").write_text("CODER_FILE_KEY=coder\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name, value in LAUNCH_ENV.items():
        monkeypatch.setenv(name, value)

    seen = []

    def fake_run_job(job, **_kwargs):  # the agent run: record what the fire's scope resolves
        seen.append((get_secret("OPENROUTER_API_KEY"), terminal_env("TERMINAL_ENV"),
                     terminal_env("TERMINAL_LOCAL_MEMORY_MAX_MB")))
        return True, "output", "final response", None

    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a, **_k: str(tmp_path / "out.txt"))
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_a, **_k: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_k: None)
    return root, coder, seen


@pytest.fixture
def host(launch_host, monkeypatch):
    launch_profile_policy.activate_multi_profile_hosting()  # what the host gateway does at boot
    # A secondary's context rewrites the process env after activation; the frozen snapshot must win.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-poisoned-after-activation")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return launch_host


def _fire_in_process(homes):
    for home in homes:
        with _profile_cron_scope(home):
            assert scheduler.run_one_job({"id": f"job-{home.name}", "name": "probe"})


@contextlib.contextmanager
def _as_fresh_worker_process(env, monkeypatch):
    """Run the worker half as the new process would: its env is exactly what the host built, and no
    process-level latch (multiplex flag, launch-home pin, frozen launch env) is inherited."""
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(secret_scope, "_AUTO_PINNED_HOME", None)
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", None)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _fire_through_workers(homes, tmp_path, monkeypatch):
    """Host side builds each worker's env + payload; each worker then runs as the separate process
    it would be. Returns ``(payload, env)`` per fire; the host's own state is untouched after."""
    from tools.process_registry import GatewayChildDispatch

    spawned = []

    class _Spawned(Exception):
        pass

    def popen(command, *, env, **_kwargs):
        payload = Path(command[command.index("--external-worker-file") + 1])
        ack = Path(command[command.index("--ack-file") + 1])
        spawned.append((payload.read_text(encoding="utf-8"), ack, dict(env)))
        raise _Spawned  # the child half runs below, as the separate process it would be

    with monkeypatch.context() as mp:
        mp.setattr("tools.process_registry.restart_safe_gateway_child_argv",
                   lambda command, **_: GatewayChildDispatch("degraded", command))
        mp.setattr(scheduler, "mark_execution_handoff_pending", lambda eid: {"id": eid})
        mp.setattr("cron.executions.adopt_claimed_execution",
                   lambda eid: {"id": eid, "status": "running"})
        mp.setattr(scheduler.subprocess, "Popen", popen)
        for n, home in enumerate(homes):
            with _profile_cron_scope(home), pytest.raises(_Spawned):
                scheduler._launch_external_cron_worker({"id": "job", "execution_id": f"exec-{n}"})
        for n, (payload_text, ack, env) in enumerate(spawned):
            payload = tmp_path / f"payload-{n}.json"
            payload.write_text(payload_text, encoding="utf-8")
            with _as_fresh_worker_process(env, mp):
                assert scheduler._run_external_worker_payload(payload, ack)
    return [(json.loads(payload_text), env) for payload_text, _ack, env in spawned]


def test_in_process_fire_binds_the_launch_env_only_for_the_launch_profile(host):
    root, coder, seen = host
    _fire_in_process((root, coder, root))
    assert seen == [LAUNCH_VIEW, FILES_ONLY_VIEW, LAUNCH_VIEW]


def test_worker_handoff_grants_the_launch_env_only_to_the_launch_profiles_worker(
        host, tmp_path, monkeypatch):
    root, coder, seen = host
    fired = _fire_through_workers((root, coder, root), tmp_path, monkeypatch)  # A -> B -> A
    assert all(payload["multiplex_active"] is True for payload, _env in fired)
    # The launch profile's own credential reaches only its worker, never a secondary's.
    assert [env.get("OPENROUTER_API_KEY") for _payload, env in fired] == [
        "sk-from-systemd", None, "sk-from-systemd"]
    assert seen == [LAUNCH_VIEW, FILES_ONLY_VIEW, LAUNCH_VIEW]


def test_single_profile_host_fires_overlay_the_live_launch_terminal_policy(
        launch_host, tmp_path, monkeypatch):
    root, coder, seen = launch_host
    assert terminal_env("TERMINAL_ENV") == "docker"  # an unscoped launch-profile turn on this host

    _fire_in_process((root, coder, root))
    fired = _fire_through_workers((root, coder, root), tmp_path, monkeypatch)
    assert seen == [LAUNCH_VIEW, FILES_ONLY_VIEW, LAUNCH_VIEW] * 2
    assert [env.get("OPENROUTER_API_KEY") for _payload, env in fired] == [
        "sk-from-systemd", None, "sk-from-systemd"]

    # Nothing froze the launch env before activation: activation captures the env as it is then.
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    launch_profile_policy.activate_multi_profile_hosting()
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "1")
    _fire_in_process((root,))
    assert seen[-1] == ("sk-from-systemd", "docker", "128")
