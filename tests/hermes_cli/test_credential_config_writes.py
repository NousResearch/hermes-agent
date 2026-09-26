"""Credential config writes keep comments without recreating named homes."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from agent import secret_scope
from hermes_cli import auth, config, credential_lifecycle, profile_lifecycle
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    result = [root / "profiles" / name for name in ("alpha", "beta")]
    for home in result:
        home.mkdir(parents=True)
    return result


def _seed(home):
    old = "fixture-" + home.name
    text = (
        "# Keep the operator's provider notes\n"
        "model:\n"
        "  provider: custom  # selected intentionally\n"
        "  default: vendor/model\n"
        "  base_url: https://old.example.invalid/v1\n"
        f"  api_key: {old}\n"
        "providers:\n"
        "  local:\n"
        f"    api: {old}  # endpoint alias, not a credential\n"
        f"    api_key: {old}\n"
        "approvals:\n"
        '  mode: "off"  # keep quoted\n'
    )
    (home / "config.yaml").write_text(text, encoding="utf-8")
    return old


def _write(operation, old):
    if operation == "write":
        raw = config.read_raw_config()
        raw["model"]["default"] = "chosen-model"
        return config.atomic_config_write(config.get_config_path(), raw)
    return {
        "switch": lambda: auth._update_config_for_provider(
            "openrouter", "https://new.example.invalid/v1/", default_model="chosen-model",
        ),
        "reset": auth._reset_config_provider,
        "rotate": lambda: credential_lifecycle._scrub_config_yaml_mirrors(old, old + "-new"),
        "remove": lambda: credential_lifecycle._scrub_config_yaml_mirrors(old, None),
    }[operation]()


@pytest.mark.parametrize("operation", ["switch", "reset", "rotate", "remove", "write"])
def test_credential_writes_keep_comments_and_profile_scope(homes, monkeypatch, operation):
    # No mkdir of a named root is permitted, even when it already exists: a
    # DELETE between its liveness check and mkdir would resurrect the path.
    real_mkdir = Path.mkdir

    def no_named_root_mkdir(path, *args, **kwargs):
        assert path not in homes, "config persistence must not create a named home"
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", no_named_root_mkdir)
    for home in (homes[0], homes[1], homes[0]):
        other = homes[1] if home == homes[0] else homes[0]
        other_path = other / "config.yaml"
        before = other_path.read_bytes() if other_path.exists() else None
        old = _seed(home)
        original = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        home_token = set_hermes_home_override(home)
        secret_token = secret_scope.set_secret_scope({})
        try:
            _write(operation, old)
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

        text = (home / "config.yaml").read_text(encoding="utf-8")
        assert "# Keep the operator's provider notes" in text
        assert "# selected intentionally" in text
        assert "# endpoint alias, not a credential" in text
        assert '"off"  # keep quoted' in text
        data = yaml.safe_load(text)
        assert data["providers"]["local"]["api"] == old
        assert (other_path.read_bytes() if other_path.exists() else None) == before
        expected_model = {
            "switch": {
                "provider": "openrouter", "default": "chosen-model",
                "base_url": "https://new.example.invalid/v1",
            },
            "reset": {**original["model"], "provider": "auto", "base_url": auth.OPENROUTER_BASE_URL},
            "rotate": {**original["model"], "api_key": old + "-new"},
            "remove": {key: value for key, value in original["model"].items() if key != "api_key"},
            "write": {**original["model"], "default": "chosen-model"},
        }[operation]
        assert data["model"] == expected_model
        expected_key = {"rotate": old + "-new", "remove": None}.get(operation, old)
        assert data["providers"]["local"].get("api_key") == expected_key
        if operation == "switch":
            assert json.loads((home / "auth.json").read_text())["active_provider"] == "openrouter"
        elif operation == "remove":
            assert "api_key" not in data["providers"]["local"]


@pytest.mark.parametrize("operation", ["switch", "reset", "rotate", "remove", "write"])
def test_credential_write_cannot_recreate_home_deleted_after_read(homes, monkeypatch, operation):
    import utils

    home = homes[0]
    old = _seed(home)
    real_dump = utils._roundtrip_dump
    deleted = False

    def delete_before_dump(path, *args, **kwargs):
        nonlocal deleted
        assert path == home / "config.yaml"
        profile_lifecycle.mark_profile_deleting(home)
        shutil.rmtree(home)
        deleted = True
        return real_dump(path, *args, **kwargs)

    monkeypatch.setattr(utils, "_roundtrip_dump", delete_before_dump)
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope({})
    try:
        with pytest.raises(FileNotFoundError):
            _write(operation, old)
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
    assert deleted, "the write must reach the post-read deletion barrier"
    assert not home.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(home)