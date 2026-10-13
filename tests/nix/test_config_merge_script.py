"""Behavioral tests for the Python script embedded by nix/configMergeScript.nix."""

import importlib.util
import json
import stat
from pathlib import Path

import pytest
from ruamel.yaml import YAML


SCRIPT_PATH = Path(__file__).parents[2] / "nix" / "configMergeScript.py"
SPEC = importlib.util.spec_from_file_location("config_merge_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
config_merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(config_merge)


def _yaml():
    yaml = YAML(typ="safe", pure=True)
    yaml.version = (1, 1)
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False
    return yaml


def _inputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(b"model: user-model\ncustom: keep-me\n")
    nix_path = tmp_path / "nix.json"
    nix_path.write_text(json.dumps({"model": "nix-model", "nested": {"owned": True}}))
    return nix_path, config_path


def test_dump_failure_preserves_existing_config(tmp_path, monkeypatch):
    nix_path, config_path = _inputs(tmp_path)
    original = config_path.read_bytes()
    yaml = _yaml()

    def fail_dump(data, stream):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(yaml, "dump", fail_dump)

    with pytest.raises(OSError, match="simulated disk write failure"):
        config_merge.merge_config(nix_path, config_path, yaml)

    assert config_path.read_bytes() == original
    assert list(tmp_path.glob(".config.yaml.*.tmp")) == []


def test_success_keeps_user_keys_and_applies_nix_precedence(tmp_path):
    nix_path, config_path = _inputs(tmp_path)

    config_merge.merge_config(nix_path, config_path, _yaml())

    result = _yaml().load(config_path.read_text())
    assert result == {
        "model": "nix-model",
        "custom": "keep-me",
        "nested": {"owned": True},
    }


def test_merge_can_run_again_after_dump_failure(tmp_path, monkeypatch):
    nix_path, config_path = _inputs(tmp_path)
    yaml = _yaml()

    def fail_dump(data, stream):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(yaml, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated disk write failure"):
        config_merge.merge_config(nix_path, config_path, yaml)

    config_merge.merge_config(nix_path, config_path, _yaml())
    result = _yaml().load(config_path.read_text())
    assert result["model"] == "nix-model"
    assert result["custom"] == "keep-me"


@pytest.mark.platforms("posix")
def test_atomic_merge_preserves_config_symlink_and_mode(tmp_path):
    nix_path, config_path = _inputs(tmp_path)
    alias_path = tmp_path / "config-link.yaml"
    alias_path.symlink_to(config_path)
    original_mode = stat.S_IMODE(config_path.stat().st_mode)

    config_merge.merge_config(nix_path, alias_path, _yaml())

    assert alias_path.is_symlink()
    assert stat.S_IMODE(config_path.stat().st_mode) == original_mode
    result = _yaml().load(config_path.read_text())
    assert result["model"] == "nix-model"
    assert result["custom"] == "keep-me"
