"""`hermes import-agent` and `hermes claw` (OpenClaw migration) left core for the agent-importers
catalog plugin.

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


@pytest.mark.parametrize("argv", [["import-agent", "claude-code", "--dry-run"], ["claw", "migrate", "--dry-run"]])
def test_old_spelling_without_the_plugin_names_the_install(argv, tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    from hermes_cli.main import _build_cli_parser

    parser, _subparsers = _build_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(sys.argv[1:])
    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert f"`hermes {argv[0]}` moved out of core into the 'agent-importers' plugin, which is not loaded" in err
    assert "hermes plugins install agent-importers" in err
    assert lcm.moved_command_hint("hermes auth", argv[0]) == ""  # never a `hermes auth` login spelling


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


def test_a_scheduled_sync_on_a_home_nothing_migrated_installs_the_plugin_and_runs(tmp_path, monkeypatch, capsys):
    """An upgrade that ran no `hermes update` (a Docker image swap) leaves a home with sync sources and
    no plugin; when the scheduled `hermes import-agent --sync` is its only Hermes process, that run
    installs the plugin and parses as the plugin's own command."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (home / "import-sync.json").write_text('{"version": 1, "agents": {}}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "import-agent", "--sync"])
    installs = []

    def install_into(target, consent=None):
        def install(name):  # what the catalog install leaves behind: the plugin, enabled
            installs.append((target, name))
            plugin = target / "plugins" / name
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\n", encoding="utf-8")
            (plugin / "__init__.py").write_text(
                "def register(ctx):\n"
                "    ctx.register_cli_command('import-agent', 'import', "
                "lambda p: p.add_argument('--sync', action='store_true'), lambda args: 'synced')\n",
                encoding="utf-8")
            (target / "config.yaml").write_text(f"model: test\nplugins:\n  enabled: [{name}]\n", encoding="utf-8")
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    from hermes_cli.main import _build_cli_parser

    parser, _subparsers = _build_cli_parser()
    args = parser.parse_args(sys.argv[1:])

    assert installs == [(home, "agent-importers")]
    assert args.sync is True and args.func(args) == "synced"
    assert "installed the 'agent-importers' plugin" in capsys.readouterr().err
