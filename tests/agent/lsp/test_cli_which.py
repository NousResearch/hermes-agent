"""``hermes lsp which`` resolves a server id through its install recipe."""
from __future__ import annotations

from agent.lsp import cli, install


def test_which_resolves_a_server_whose_id_differs_from_its_recipe_key(monkeypatch, capsys):
    binaries = {"typescript-language-server": "/opt/lsp/bin/typescript-language-server"}
    monkeypatch.setattr(install, "_existing_binary", binaries.get)

    assert cli._cmd_which("typescript") == 0
    assert capsys.readouterr().out == "/opt/lsp/bin/typescript-language-server\n"
