"""Regression: with the configured port already held by an incumbent llama-server, a replacement
router failed to bind, but ``_wait_health`` accepted the incumbent's /health as its own and published
state for a dead pid -- leaving an unsupervised server the Local Models UI reported as "Turn on".
"""
from __future__ import annotations

import socket

import pytest


def test_start_refuses_a_busy_explicit_port_instead_of_adopting_its_health(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        busy = holder.getsockname()[1]
        sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=busy)
        spawned = []
        monkeypatch.setattr(sup, "_spawn", lambda: spawned.append(True))
        with pytest.raises(RuntimeError, match="already in use"):
            sup.start(timeout_s=1)
        assert not spawned, "spawned a router onto a port another server already holds"


def test_port_probe_sees_reuseaddr_listener_but_not_a_freed_port():
    from hermes_cli.local_runtime import supervisor

    with socket.socket() as holder:  # llama-server's HTTP listener also sets SO_REUSEADDR
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        assert supervisor._port_in_use(port) is True
    assert supervisor._port_in_use(port) is False


def test_wait_health_rejects_health_answered_after_own_router_exited(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59996)
    polls = iter([None, 1, 1, 1])  # alive at the pre-check, dead by the time health answers
    sup.proc = SimpleNamespace(pid=4321, returncode=1, poll=lambda: next(polls))

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(supervisor.urllib.request, "urlopen", lambda *a, **k: _Ok())
    with pytest.raises(RuntimeError, match="exited"):
        sup._wait_health(2)
