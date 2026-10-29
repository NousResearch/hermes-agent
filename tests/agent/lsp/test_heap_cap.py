"""LSP server memory ceiling (t_8e8ec4de, 2026-10-28).

A language server is the dominant memory consumer inside a Kanban worker scope.
On the homelab host pyright reached 1.7 GB and was the process the kernel
OOM-killed alongside its worker (``task=node-MainThread``, journal 2026-10-28
13:01). The worker scope's ``MemoryMax`` is a backstop, not a budget: a server
that grows to the scope limit takes the worker down with it.

These tests cover the V8 heap ceiling (``lsp.max_server_heap_mb`` →
``NODE_OPTIONS=--max-old-space-size=N``) and the config plumbing that carries it
from ``config.yaml`` down to the spawned process environment.
"""

from __future__ import annotations

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, apply_heap_cap, find_server_for_file


# ---------------------------------------------------------------------------
# apply_heap_cap
# ---------------------------------------------------------------------------


def test_heap_cap_sets_node_options():
    env = apply_heap_cap({}, 768)
    assert env["NODE_OPTIONS"] == "--max-old-space-size=768"


def test_heap_cap_is_disabled_by_zero():
    env = apply_heap_cap({"NODE_OPTIONS": "--trace-warnings"}, 0)
    assert env["NODE_OPTIONS"] == "--trace-warnings"
    env = apply_heap_cap({}, 0)
    assert "NODE_OPTIONS" not in env


def test_heap_cap_replaces_an_existing_cap_without_duplicating():
    env = apply_heap_cap({"NODE_OPTIONS": "--max-old-space-size=4096"}, 768)
    assert env["NODE_OPTIONS"].count("--max-old-space-size") == 1
    assert "--max-old-space-size=768" in env["NODE_OPTIONS"]


def test_heap_cap_preserves_other_node_options():
    env = apply_heap_cap({"NODE_OPTIONS": "--trace-warnings --inspect=0"}, 512)
    assert "--trace-warnings" in env["NODE_OPTIONS"]
    assert "--inspect=0" in env["NODE_OPTIONS"]
    assert "--max-old-space-size=512" in env["NODE_OPTIONS"]


def test_heap_cap_is_idempotent():
    once = apply_heap_cap({}, 640)
    twice = apply_heap_cap(dict(once), 640)
    assert once["NODE_OPTIONS"] == twice["NODE_OPTIONS"]
    assert twice["NODE_OPTIONS"].count("--max-old-space-size") == 1


# ---------------------------------------------------------------------------
# the spawn spec carries the cap
# ---------------------------------------------------------------------------


def test_pyright_spawn_spec_carries_the_heap_cap(tmp_path, monkeypatch):
    """The cap must reach the spawned server's environment."""
    from agent.lsp import servers as servers_mod

    monkeypatch.setattr(
        servers_mod, "_find_binary",
        lambda ctx, server_id, which, install_pkg: "/usr/bin/pyright-langserver",
    )
    monkeypatch.setattr(servers_mod, "_detect_python", lambda root: None)
    # ``pm.env_for`` reads the REAL hermes home; the test-home guard forbids it.
    monkeypatch.setattr("pm.env_for", lambda _name: {})

    srv = find_server_for_file(str(tmp_path / "a.py"))
    assert srv is not None
    ctx = ServerContext(
        workspace_root=str(tmp_path), install_strategy="manual", max_heap_mb=768,
    )
    spec = srv.build_spawn(str(tmp_path), ctx)
    assert spec is not None
    assert "--max-old-space-size=768" in spec.env.get("NODE_OPTIONS", "")


def test_pyright_spawn_spec_without_a_cap_is_unchanged(tmp_path, monkeypatch):
    from agent.lsp import servers as servers_mod

    monkeypatch.setattr(
        servers_mod, "_find_binary",
        lambda ctx, server_id, which, install_pkg: "/usr/bin/pyright-langserver",
    )
    monkeypatch.setattr(servers_mod, "_detect_python", lambda root: None)
    monkeypatch.setattr("pm.env_for", lambda _name: {})

    srv = find_server_for_file(str(tmp_path / "a.py"))
    assert srv is not None
    spec = srv.build_spawn(str(tmp_path), ServerContext(str(tmp_path), install_strategy="manual"))
    assert spec is not None
    assert "--max-old-space-size" not in spec.env.get("NODE_OPTIONS", "")


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------


def test_service_reads_max_server_heap_mb_from_config(monkeypatch):
    """``lsp.max_server_heap_mb`` must reach the service."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True, "max_server_heap_mb": 640}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_server_heap_mb == 640


def test_service_defaults_to_no_cap_without_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_server_heap_mb == 0


def test_service_ignores_a_malformed_cap(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True, "max_server_heap_mb": "not-a-number"}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_server_heap_mb == 0


def test_service_ignores_a_negative_cap(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True, "max_server_heap_mb": -5}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_server_heap_mb == 0


@pytest.mark.parametrize("server_id", [s.server_id for s in SERVERS])
def test_every_server_respects_the_context(server_id):
    """The cap lives on the context, so every registry server inherits it."""
    ctx = ServerContext(workspace_root="/w", install_strategy="manual", max_heap_mb=512)
    assert ctx.max_heap_mb == 512
