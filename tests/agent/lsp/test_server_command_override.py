"""Regression: a configured ``lsp.servers.<id>.command`` is a full argv, not just ``argv[0]``.

``_find_binary()`` returned only the first element, so a command list such as
``["/path/to/node", "/path/to/pyright/langserver.index.js", "--stdio"]`` collapsed to
``node --stdio`` and the spawn died with ``bad option: --stdio`` (Node exit 9) — while the
documented one-element form (a pinned binary path) has to keep behaving exactly as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.lsp import servers


def _ctx(root: str, **overrides) -> servers.ServerContext:
    return servers.ServerContext(workspace_root=root, binary_overrides=overrides)


# The powershell entry is excluded on purpose: its ``command`` is documented as the
# PowerShellEditorServices *bundle directory*, not an argv, so it is a different key shape.
_SERVERS = [s for s in servers.SERVERS if s.server_id != "powershell"]


@pytest.mark.parametrize("srv", _SERVERS, ids=lambda s: s.server_id)
def test_multi_element_command_override_is_honored_verbatim(srv, tmp_path):
    """Every registered server spawns the whole configured argv, args included."""
    override = [sys.executable, "langserver.index.js", "--stdio"]
    spec = srv.build_spawn(str(tmp_path), _ctx(str(tmp_path), **{srv.server_id: override}))
    assert spec is not None, f"{srv.server_id}: a valid argv[0] must resolve"
    assert spec.command == override


def test_single_element_override_keeps_the_servers_default_args(tmp_path):
    """A one-element override is a pinned binary path — default args still get appended."""
    root = str(tmp_path)
    spec = servers.find_server_for_file("x.ts").build_spawn(root, _ctx(root, typescript=[sys.executable]))
    assert spec is not None
    assert spec.command == [sys.executable, "--stdio"]


def test_pyright_sibling_swap_applies_only_to_a_pinned_binary(tmp_path):
    """The pyright→pyright-langserver sibling swap survives for a pinned binary and is
    skipped when the user named the whole command themselves."""
    cli = tmp_path / "pyright"
    cli.write_text("")
    sibling = tmp_path / "pyright-langserver"
    sibling.write_text("")
    root = str(tmp_path)

    pinned = servers._spawn_pyright(root, _ctx(root, pyright=[str(cli)]))
    assert pinned is not None
    assert pinned.command == [str(sibling), "--stdio"]

    explicit = servers._spawn_pyright(root, _ctx(root, pyright=[str(cli), "--stdio", "--verbose"]))
    assert explicit is not None
    assert explicit.command == [str(cli), "--stdio", "--verbose"]
