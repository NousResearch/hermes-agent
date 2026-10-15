from pathlib import Path

import pytest

from hermes_cli import config, config_includes
from hermes_cli.config_effective import load_user_config_effective


@pytest.mark.parametrize("loader", [config.load_config, load_user_config_effective])
def test_cold_broken_include_refuses_defaults_and_recovers(tmp_path, monkeypatch, loader):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "config.yaml"
    root.write_text("include: policy.yaml\napprovals:\n  deny: [root-guard]\n")
    inc = tmp_path / "policy.yaml"
    inc.write_text("approvals: [broken\n")
    original = root.read_bytes()
    # No known complete policy exists: returning defaults/{} loses deny rules.
    with pytest.raises(ValueError, match="Invalid config include"):
        loader()
    inc.write_text("display:\n  language: fr\n")
    result = loader()
    assert result["approvals"]["deny"] == ["root-guard"]
    assert result["display"]["language"] == "fr"
    assert root.read_bytes() == original


@pytest.mark.parametrize("loader", [config.load_config, load_user_config_effective])
def test_warm_broken_include_keeps_policy_then_recovers(tmp_path, monkeypatch, loader):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("include: policy.yaml\n")
    inc = tmp_path / "policy.yaml"
    inc.write_text("approvals:\n  deny: [old-guard]\n")
    assert loader()["approvals"]["deny"] == ["old-guard"]
    inc.write_text("approvals: [broken\n")
    assert loader()["approvals"]["deny"] == ["old-guard"]
    inc.write_text("approvals:\n  deny: [repaired-guard]\n")
    assert loader()["approvals"]["deny"] == ["repaired-guard"]


@pytest.mark.parametrize("loader", [config.load_config, load_user_config_effective])
def test_new_broken_dependency_repair_invalidates_fallback(tmp_path, monkeypatch, loader):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "config.yaml"
    root.write_text("approvals:\n  deny: [original-guard]\n")
    assert loader()["approvals"]["deny"] == ["original-guard"]
    root.write_text("include: new.yaml\n")
    inc = tmp_path / "new.yaml"
    inc.write_text("approvals: [broken\n")
    assert loader()["approvals"]["deny"] == ["original-guard"]
    inc.write_text("approvals:\n  deny: [new-guard]\n")
    assert loader()["approvals"]["deny"] == ["new-guard"]


def _deny_stat(monkeypatch, target):
    real_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path == target:
            raise PermissionError("include stat denied")
        return real_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)


@pytest.mark.parametrize("loader", [config.load_config, load_user_config_effective])
def test_include_that_cannot_be_stated_is_not_a_missing_include(tmp_path, monkeypatch, loader):
    """Only absence is skipped: a denied stat may hide policy, cold or warm."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "config.yaml"
    root.write_text("include: policy.yaml\n")
    inc = tmp_path / "policy.yaml"
    inc.write_text("approvals:\n  deny: [included-guard]\n")
    original = root.read_bytes()
    with monkeypatch.context() as fault:
        _deny_stat(fault, inc)
        with pytest.raises(config_includes.ConfigIncludeError, match="Unreadable config include"):
            loader()
    assert loader()["approvals"]["deny"] == ["included-guard"]
    inc.write_text("approvals:\n  deny: [edited-guard]\n")
    with monkeypatch.context() as fault:
        _deny_stat(fault, inc)
        assert loader()["approvals"]["deny"] == ["included-guard"]
    assert loader()["approvals"]["deny"] == ["edited-guard"]
    assert root.read_bytes() == original


@pytest.mark.parametrize("loader", [config.load_config, load_user_config_effective])
@pytest.mark.parametrize("payload", ["- approvals\n", "7\n", "false\n"])
def test_non_mapping_include_is_refused_not_read_as_empty(tmp_path, monkeypatch, loader, payload):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("include: policy.yaml\napprovals:\n  deny: [root-guard]\n")
    inc = tmp_path / "policy.yaml"
    inc.write_text(payload)
    with pytest.raises(config_includes.ConfigIncludeError, match="Invalid config include"):
        loader()
    inc.write_text("approvals:\n  deny: [repaired-guard]\n")
    assert loader()["approvals"]["deny"] == ["root-guard"]
    inc.write_text("display:\n  language: fr\n")
    assert loader()["display"]["language"] == "fr"


@pytest.mark.parametrize("loader", [config.load_config, load_user_config_effective])
@pytest.mark.parametrize("payload", ["", "null\n", "# placeholder\n"])
def test_empty_include_declares_nothing(tmp_path, monkeypatch, loader, payload):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("include: policy.yaml\napprovals:\n  deny: [root-guard]\n")
    (tmp_path / "policy.yaml").write_text(payload)
    assert loader()["approvals"]["deny"] == ["root-guard"]
