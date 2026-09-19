import yaml

import hermes_cli.config as config


def test_explicit_null_survives_include_stripping():
    assert config._strip_included_config_values(
        {"model": {"base_url": None}}, {"model": {"base_url": "https://included.example"}},
        preserve_keys={("model", "base_url")},
    ) == {"model": {"base_url": None}}
    assert config._strip_included_config_values({"a": 1, "b": 2}, {"a": 1}) == {"b": 2}


def test_include_save_reload_and_dependency_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "config.yaml"
    included = tmp_path / "included.yaml"
    root.write_text("include: included.yaml\nmodel:\n  base_url: null\n")
    included.write_text("model:\n  default: example/first\n  base_url: https://included.example\napprovals:\n  deny: ['protected-command']\n")
    loaded = config.load_config()
    assert loaded["model"]["base_url"] is None
    assert loaded["approvals"]["deny"] == ["protected-command"]
    config.save_config(loaded)
    saved = yaml.safe_load(root.read_text())
    assert saved["model"]["base_url"] is None
    assert "default" not in saved["model"]
    assert "approvals" not in saved
    assert config.load_config()["model"]["default"] == "example/first"

    # Cache hits stat include dependencies, but do not parse the YAML again.
    original_loader = config.fast_safe_load
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return original_loader(*args, **kwargs)
    monkeypatch.setattr(config, "fast_safe_load", counted)
    readonly = config.load_config_readonly()
    assert config.load_config_readonly() is readonly
    assert not calls

    included.write_text("approvals: [broken\n")
    assert config.load_config()["approvals"]["deny"] == ["protected-command"]
    included.write_text("model:\n  default: example/repaired\n")
    assert config.load_config()["model"]["default"] == "example/repaired"
