"""Abandoned-request lifecycle: work sent to the managed local server must
die when its caller goes away, and teardown must never orphan VRAM.

The incident this guards: auxiliary calls (title generation + retries)
queued at the router behind a cold model load, their clients timed out
and hung up, and the router then dispatched them anyway. Non-streamed
responses write the socket only after the FULL generation, so nothing
noticed the dead clients — two uncapped decodes ran at full GPU for the
better part of an hour with nobody listening.

Three contracts, one per failure link:
1. Auxiliary requests to the managed local endpoint are always streamed
   (a dead client then cancels decode at the first chunk write).
2. Explicit caller max_tokens caps reach the managed local endpoint
   (title generation's 64-token cap must not be silently dropped).
3. Supervisor teardown terminates the whole process tree, and a router
   respawn reaps orphaned model children first (each holds GiB of VRAM).
"""

from __future__ import annotations

import json
import subprocess
import types

import pytest

import agent.auxiliary_client as aux
from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor


MANAGED_URL = "http://127.0.0.1:18434/v1"


@pytest.fixture
def managed_state(tmp_path, monkeypatch):
    """A supervisor state file declaring the managed endpoint, cache reset."""
    state = tmp_path / "server.json"
    state.write_text(json.dumps({"base_url": MANAGED_URL, "api_key": "k",
                                 "pid": 4242}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.state_path",
                        lambda: state)
    monkeypatch.setattr(aux, "_managed_local_cache", (0.0, ""))
    return state


# ── 1. managed endpoint is always streamed ───────────────────


def test_managed_endpoint_requires_stream(managed_state):
    assert aux._provider_requires_stream("custom", MANAGED_URL) is True


def test_managed_detection_matches_netloc_not_substring(managed_state):
    # Same host, different port: a user's own external server — untouched.
    assert aux._provider_requires_stream("custom", "http://127.0.0.1:8080/v1") is False


def test_no_state_file_means_no_managed_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.state_path",
                        lambda: tmp_path / "absent.json")
    monkeypatch.setattr(aux, "_managed_local_cache", (0.0, ""))
    assert aux._is_managed_local_endpoint(MANAGED_URL) is False


def test_remote_providers_unaffected(managed_state):
    assert aux._provider_requires_stream("nous",
                                         "https://inference-api.nousresearch.com/v1/") is False


# ── 2. explicit caps reach the managed endpoint ──────────────


def test_explicit_max_tokens_forwarded_to_managed_local(managed_state, monkeypatch):
    monkeypatch.setattr(aux, "_current_custom_base_url", lambda: MANAGED_URL)
    kwargs = aux._build_call_kwargs(
        "custom", "Qwen-Local", [{"role": "user", "content": "hi"}],
        max_tokens=64, timeout=30.0, task="title_generation")
    assert kwargs.get("max_tokens") == 64 or kwargs.get("max_completion_tokens") == 64, (
        "explicit caller cap dropped on the managed local endpoint — an "
        "EOS-less generation then runs to the full context window")


def test_no_default_cap_policy_unchanged_for_remote(monkeypatch):
    # A generic remote provider still drops the cap (the forwarding gate
    # is an allow-list). openrouter no longer qualifies as the example
    # here: main forwards its caps deliberately (#41035, 402 affordability).
    monkeypatch.setattr(aux, "_managed_local_cache", (0.0, ""))
    kwargs = aux._build_call_kwargs(
        "openai", "some/model", [{"role": "user", "content": "hi"}],
        max_tokens=64, timeout=30.0)
    assert "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs


# ── 3. teardown kills the tree; respawn reaps orphans ────────


class _FakeChild:
    def __init__(self, pid):
        self.pid = pid
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def is_running(self):
        return not self.terminated and not self.killed

    def kill(self):
        self.killed = True


def test_terminate_tree_terminates_children_too(monkeypatch):
    children = [_FakeChild(101), _FakeChild(102)]

    class _FakeParentProc:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            assert recursive is True
            return children

    fake_psutil = types.SimpleNamespace(Process=_FakeParentProc)
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    class _FakeRouter:
        pid = 4242
        terminated = False

        def terminate(self):
            _FakeRouter.terminated = True

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None

    LlamaServerSupervisor._terminate_tree(_FakeRouter())
    assert _FakeRouter.terminated
    assert all(c.terminated for c in children), (
        "router children orphaned on stop — each holds GiB of VRAM")


def test_terminate_tree_survives_missing_psutil(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)

    class _FakeRouter:
        pid = 4242
        terminated = False

        def terminate(self):
            _FakeRouter.terminated = True

        def wait(self, timeout=None):
            return 0

    LlamaServerSupervisor._terminate_tree(_FakeRouter())
    assert _FakeRouter.terminated  # router still stopped without psutil


def test_reap_orphans_kills_only_our_parentless_binaries(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server.exe"
    exe.write_text("")

    orphan = _FakeChild(300)
    adopted = _FakeChild(301)     # parent alive -> not an orphan
    foreign = _FakeChild(302)     # different binary -> never touched

    def _info(pid, exe_path, ppid):
        p = _FakeChild(pid)
        p.info = {"exe": exe_path, "ppid": ppid}
        return p

    procs = [
        _info(300, str(exe), 9999),          # dead parent -> reap
        _info(301, str(exe), 1),             # live parent -> keep
        _info(302, str(tmp_path / "other.exe"), 9999),  # foreign -> keep
    ]
    reaped = []
    for p in procs:
        p.kill = lambda p=p: reaped.append(p.info and p.pid)

    class _NoSuch(Exception):
        pass

    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: procs,
        pid_exists=lambda pid: pid == 1,
        NoSuchProcess=_NoSuch,
        AccessDenied=_NoSuch,
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.server_binary",
                        lambda install_dir: exe)

    sup = LlamaServerSupervisor.__new__(LlamaServerSupervisor)
    sup.install_dir = tmp_path
    sup.proc = None
    sup._reap_orphaned_children()

    assert reaped == [300], f"reaped {reaped}; wanted only the orphan (300)"
