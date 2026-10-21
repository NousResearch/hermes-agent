"""`hermes import-agent` left core for the agent-importers catalog plugin.

Real parser and real config files under a temp HERMES_HOME; the catalog lookup and the install are
stand-ins (the live install is exercised end to end outside the unit suite).
"""

from __future__ import annotations

import sys

import pytest

import hermes_cli.left_core_migration as lcm


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_undelivered", {})
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


def test_old_spelling_without_the_plugin_names_the_install(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes", "import-agent", "claude-code", "--dry-run"])
    from hermes_cli.main import _build_cli_parser

    parser, _subparsers = _build_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(sys.argv[1:])
    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "moved out of core into the 'agent-importers' plugin, which is not loaded" in err
    assert "hermes plugins install agent-importers" in err


def test_only_a_home_with_registered_imports_gets_the_plugin(tmp_path):
    """A registered sync source (scheduled `--sync` runs read it) is use; a home that never imported
    is left alone: the old spelling names the install instead."""
    synced, untouched = tmp_path / "synced", tmp_path / "untouched"
    for home in (synced, untouched):
        home.mkdir()
        (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (synced / "import-sync.json").write_text('{"version": 1, "agents": {}}', encoding="utf-8")
    calls, said = [], []

    def install(name):
        calls.append(name)
        (synced / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(untouched, install=install, say=said.append) == []
    assert calls == [] and said == []
    assert lcm.migrate_home(synced, install=install, say=said.append) == ["agent-importers"]
    assert calls == ["agent-importers"]
    assert "✓ `hermes import-agent` moved out of core" in said[0]
