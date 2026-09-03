"""Tests for laravel-lsp server registration and .blade.php matching.

Verifies that:
- ``.blade.php`` files route to ``laravel-lsp``, not ``intelephense``.
- ``.php`` files still route to ``intelephense`` (no regression).
- ``_file_ext_or_basename`` returns the full ``.blade.php`` double extension.
- ``language_id_for`` returns ``"blade"`` for ``.blade.php`` files.
- ``_spawn_laravel_lsp`` builds the correct command when the binary is on PATH.
"""
from __future__ import annotations

import pytest

from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    _file_ext_or_basename,
    _spawn_laravel_lsp,
    find_server_for_file,
    language_id_for,
)


def test_blade_php_routes_to_laravel_lsp():
    """find_server_for_file must return laravel-lsp for .blade.php files."""
    srv = find_server_for_file("/project/resources/views/welcome.blade.php")
    assert srv is not None
    assert srv.server_id == "laravel-lsp"


def test_php_still_routes_to_intelephense():
    """Plain .php files must still match intelephense, not laravel-lsp."""
    srv = find_server_for_file("/project/app/Models/User.php")
    assert srv is not None
    assert srv.server_id == "intelephense"


def test_file_ext_or_basename_blade_php():
    """Double extension .blade.php must be returned in full."""
    assert _file_ext_or_basename("/x/welcome.blade.php") == ".blade.php"
    assert _file_ext_or_basename("/x/UPPER.blade.php") == ".blade.php"


def test_file_ext_or_basename_regular_php():
    """Regular .php extension must still work."""
    assert _file_ext_or_basename("/x/User.php") == ".php"


def test_language_id_for_blade():
    """language_id_for must return 'blade' for .blade.php files."""
    assert language_id_for("/x/welcome.blade.php") == "blade"
    assert language_id_for("/x/User.php") == "php"


def test_laravel_lsp_spawn_command(monkeypatch):
    """_spawn_laravel_lsp must produce [bin, 'lsp'] when binary is on PATH."""
    monkeypatch.setattr(
        "agent.lsp.servers._which", lambda *names: "/fake/bin/laravel-lsp"
    )
    ctx = ServerContext(workspace_root="/project", install_strategy="manual")
    spec = _spawn_laravel_lsp("/project", ctx)
    assert spec is not None
    assert spec.command == ["/fake/bin/laravel-lsp", "lsp"]
    assert spec.cwd == "/project"


def test_laravel_lsp_spawn_returns_none_when_missing(monkeypatch):
    """When binary is not on PATH and install is manual, must return None."""
    monkeypatch.setattr("agent.lsp.servers._which", lambda *names: None)
    # _spawn_laravel_lsp falls back to try_install (manual → _existing_binary),
    # which probes the real PATH; patch it so the test is hermetic even on a
    # machine where laravel-lsp happens to be installed.
    monkeypatch.setattr("agent.lsp.install.try_install", lambda pkg, strategy: None)
    ctx = ServerContext(workspace_root="/project", install_strategy="manual")
    spec = _spawn_laravel_lsp("/project", ctx)
    assert spec is None


def test_laravel_lsp_in_servers_registry():
    """laravel-lsp must be registered in SERVERS before intelephense."""
    ids = [s.server_id for s in SERVERS]
    assert "laravel-lsp" in ids
    assert ids.index("laravel-lsp") < ids.index("intelephense")
