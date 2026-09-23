"""Update-time recovery messages identify the home they describe."""

import pytest

from hermes_cli import memory_provider_migration as mig
from hermes_cli import plugin_python_deps as deps


@pytest.mark.parametrize("outcome", ["success", "failure", "exception", "missing_catalog"])
def test_update_outcomes_identify_each_home(tmp_path, monkeypatch, outcome):
    homes = [tmp_path / "alpha", tmp_path / "beta"]
    provider = "scout_missing_provider"
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text(
            f"memory:\n  provider: {provider}\n", encoding="utf-8"
        )
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr(deps, "dependency_homes", lambda: homes)
    monkeypatch.setattr(mig, "catalog_source", lambda name: None if outcome == "missing_catalog" else name)
    calls = []

    def installer(home):
        def install(name):
            calls.append((home, name))
            if outcome == "exception":
                raise OSError("offline")
            return {"ok": outcome == "success", "error": "offline"}
        return install

    monkeypatch.setattr(mig, "_install_into", installer)
    said = []
    assert mig.migrate_all_homes(say=said.append) == ([provider, provider] if outcome == "success" else [])
    assert len(said) == len(homes)
    for home, message in zip(homes, said):
        assert str(home) in message
        assert provider in message
        assert str(homes[1] if home == homes[0] else homes[0]) not in message
    assert calls == ([] if outcome == "missing_catalog" else [(h, provider) for h in homes])


def test_update_noop_remains_silent(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.yaml").write_text("memory:\n  provider: none\n", encoding="utf-8")
    monkeypatch.setattr(deps, "dependency_homes", lambda: [tmp_path])
    assert mig.migrate_all_homes() == []
    assert capsys.readouterr().out == ""


def test_update_callbacks_bind_home_and_preserve_error_isolation(tmp_path, monkeypatch, capsys):
    homes = [tmp_path / "bad", tmp_path / "good"]
    monkeypatch.setattr(deps, "dependency_homes", lambda: homes)
    callbacks = []

    def migrate(home, *, install, say):
        callbacks.append(say)
        if home == homes[0]:
            raise OSError("unreadable config")
        say("installed")
        return "provider"

    monkeypatch.setattr(mig, "migrate_home", migrate)
    assert mig.migrate_all_homes() == ["provider"]
    assert str(homes[1]) in capsys.readouterr().out
    # Even a retained callback must refer to its own iteration, not the final home.
    callbacks[0]("late diagnostic")
    output = capsys.readouterr().out
    assert str(homes[0]) in output
    assert str(homes[1]) not in output
