import pytest

from hermes_cli import config
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
