"""``hermes config add`` / ``remove``: one item of a list setting, through the real CLI parser."""

import argparse

import pytest

import hermes_yaml as yaml


@pytest.fixture
def home(tmp_path, monkeypatch):
    import hermes_cli.config as cfg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").touch()
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return tmp_path


def _hermes(*argv):
    from hermes_cli.config import config_command
    from hermes_cli.subcommands.config import build_config_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_config_parser(parser.add_subparsers(dest="command"), cmd_config=config_command)
    args = parser.parse_args(["config", *argv])
    args.func(args)


def _saved(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8-sig"))


def test_add_and_remove_edit_one_item_and_keep_comments(home, capsys):
    (home / "config.yaml").write_text(
        "# operator note\nterminal:\n  backend: local  # pinned\n", encoding="utf-8")

    _hermes("add", "toolsets", "web")  # absent key: starts from the default list
    _hermes("add", "toolsets", "web")  # already there: no duplicate
    _hermes("add", "--json", "hooks.pre_llm_call", '{"command": "~/check.py", "timeout": 9}')
    _hermes("add", "agent.disabled_toolsets", "off")  # stays the string, not False
    assert _saved(home)["toolsets"] == ["hermes-cli", "web"]
    assert _saved(home)["hooks"]["pre_llm_call"] == [{"command": "~/check.py", "timeout": 9}]
    assert _saved(home)["agent"]["disabled_toolsets"] == ["off"]
    assert "already contains 'web'" in capsys.readouterr().out

    _hermes("remove", "toolsets", "hermes-cli")
    _hermes("remove", "toolsets", "hermes-cli")  # absent: nothing to do, not an error

    text = (home / "config.yaml").read_text(encoding="utf-8-sig")
    assert "# operator note" in text and "backend: local  # pinned" in text
    assert _saved(home)["toolsets"] == ["web"]
    assert "is not in toolsets" in capsys.readouterr().out


@pytest.mark.parametrize(("argv", "saved", "refusal"), [
    pytest.param(("model", "gpt"), "", "is not a list setting", id="scalar-default"),
    pytest.param(("providers", "x"), "", "is not a list setting (it holds a mapping)", id="known-mapping"),
    pytest.param(("toolsets", "web"), "toolsets: hermes-cli\n", "not a list", id="saved-non-list"),
    pytest.param(("--json", "toolsets", "{web"), "", "not valid JSON", id="bad-json"),
    pytest.param(("OPENROUTER_API_KEY", "x"), "", "environment setting", id="env-key"),
    pytest.param(("custom_providers.0.models", "x"), "", "is a list with no item '0'", id="index-into-missing-list"),
    pytest.param(("custom_providers.0.models", "x"), "custom_providers: null\n", "is a list with no item '0'",
                 id="index-into-null-list"),
    pytest.param(("custom_providers.0.models", "x"), "custom_providers: []\n", "is a list with no item '0'",
                 id="index-past-end-of-list"),
])
def test_non_list_targets_are_refused_and_nothing_is_written(home, capsys, argv, saved, refusal):
    if saved:
        (home / "config.yaml").write_text(saved, encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _hermes("add", *argv)

    assert exc.value.code == 1
    assert refusal in capsys.readouterr().err
    assert (home / "config.yaml").read_text(encoding="utf-8-sig") == saved if saved else not (home / "config.yaml").exists()


def test_a_number_never_matches_a_boolean_inside_an_object_item(home, capsys):
    (home / "config.yaml").write_text(
        "custom_providers:\n  - name: x\n    models_discovered: true\n", encoding="utf-8")

    _hermes("remove", "--json", "custom_providers", '{"name": "x", "models_discovered": 1}')
    _hermes("add", "--json", "custom_providers", '{"name": "x", "models_discovered": 1}')
    _hermes("add", "--json", "custom_providers", '{"models_discovered": true, "name": "x"}')  # same object
    # ``True == 1`` in Python, so compare the stored types, not just the values.
    assert [type(p["models_discovered"]) for p in _saved(home)["custom_providers"]] == [bool, int]
    out = capsys.readouterr().out
    assert "is not in custom_providers" in out and "already contains" in out


def test_output_masks_saved_credentials_in_the_list(home, capsys):
    secret = "opaque-saved-token-0123456789abcdef"
    (home / "config.yaml").write_text(
        f"custom_providers:\n  - name: a\n    api_key: {secret}\n", encoding="utf-8")

    _hermes("add", "--json", "custom_providers", '{"name": "b", "base_url": "https://b.example"}')
    _hermes("add", "--json", "custom_providers", f'{{"name": "a", "api_key": "{secret}"}}')  # no change

    assert secret not in capsys.readouterr().out
    assert _saved(home)["custom_providers"][0]["api_key"] == secret


def test_administrator_managed_list_is_refused(home, tmp_path_factory, monkeypatch, capsys):
    from hermes_cli import managed_scope

    managed = tmp_path_factory.mktemp("managed")
    (managed / "config.yaml").write_text("toolsets:\n  - hermes-cli\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    try:
        with pytest.raises(SystemExit):
            _hermes("add", "toolsets", "web")
    finally:
        managed_scope.invalidate_managed_cache()

    assert "managed by your administrator" in capsys.readouterr().err
    assert not (home / "config.yaml").exists()
