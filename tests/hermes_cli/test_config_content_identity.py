"""Content-identity and canonical-path regressions for config caching."""

import os
import time
from unittest.mock import patch

import pytest

from hermes_cli.config import load_config, save_config


def test_saved_config_is_last_known_good_through_symlinked_home(
    tmp_path, capsys
):
    """Writer and loader must address last-known-good state by one path."""
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    with patch.dict(os.environ, {"HERMES_HOME": str(linked_home)}):
        save_config(
            {
                "model": {"default": "test/saved-model"},
                "approvals": {"deny": ["curl*evil.com*"]},
            }
        )
        time.sleep(0.05)
        (real_home / "config.yaml").write_text(
            "approvals:\n  deny: [unclosed\n", encoding="utf-8"
        )

        after = load_config()

    assert after["model"]["default"] == "test/saved-model"
    assert after["approvals"]["deny"] == ["curl*evil.com*"]
    assert "settings it loaded before the edit" in capsys.readouterr().err


def test_unreadable_user_config_cannot_reuse_missing_file_cache(tmp_path, monkeypatch):
    from hermes_cli import config as config_mod
    from hermes_cli.config_read_errors import FailedConfigRead
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("model: managed-model\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    config_mod._LOAD_CONFIG_CACHE.clear()
    assert not isinstance(config_mod.load_config(), FailedConfigRead)
    def unavailable(*args, **kwargs):
        raise PermissionError("temporary read failure")
    monkeypatch.setattr(config_mod, "_load_config_cache_sig", unavailable)
    monkeypatch.setattr(config_mod, "read_file_with_signature", unavailable)
    result = config_mod.load_config()
    assert isinstance(result, FailedConfigRead)
    assert isinstance(result.read_error, PermissionError)
