"""Tests for ``_save_auth_store`` ownership alignment and ``_load_auth_store``
PermissionError handling — guards the docker-exec login flow described in #15718.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from hermes_cli import auth as auth_mod


@pytest.fixture()
def isolated_auth_file(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    monkeypatch.setattr(auth_mod, "_auth_file_path", lambda: auth_file)
    return auth_file


# =========================================================================
# _align_owner_to_parent
# =========================================================================

class TestAlignOwnerToParent:
    def test_skips_when_not_root(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}")
        with (
            patch("os.geteuid", return_value=1234, create=True),
            patch("os.chown") as mock_chown,
        ):
            auth_mod._align_owner_to_parent(target)
        mock_chown.assert_not_called()

    def test_skips_when_parent_owned_by_root(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}")
        fake_parent_stat = type(
            "FakeStat", (), {"st_uid": 0, "st_gid": 0, "st_mode": 0o755}
        )()
        with (
            patch("os.geteuid", return_value=0, create=True),
            patch("pathlib.Path.stat", return_value=fake_parent_stat),
            patch("os.chown") as mock_chown,
        ):
            auth_mod._align_owner_to_parent(target)
        mock_chown.assert_not_called()

    def test_chowns_when_root_and_parent_owned_by_user(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}")
        fake_parent_stat = type(
            "FakeStat", (), {"st_uid": 10000, "st_gid": 10000, "st_mode": 0o755}
        )()
        with (
            patch("os.geteuid", return_value=0, create=True),
            patch.object(type(target.parent), "stat", return_value=fake_parent_stat),
            patch("os.chown") as mock_chown,
        ):
            auth_mod._align_owner_to_parent(target)
        mock_chown.assert_called_once_with(target, 10000, 10000)

    def test_chown_failure_is_swallowed(self, tmp_path, caplog):
        target = tmp_path / "auth.json"
        target.write_text("{}")
        fake_parent_stat = type(
            "FakeStat", (), {"st_uid": 10000, "st_gid": 10000, "st_mode": 0o755}
        )()
        with (
            patch("os.geteuid", return_value=0, create=True),
            patch.object(type(target.parent), "stat", return_value=fake_parent_stat),
            patch("os.chown", side_effect=PermissionError("nope")),
            caplog.at_level(logging.DEBUG, logger="hermes_cli.auth"),
        ):
            # Must not raise.
            auth_mod._align_owner_to_parent(target)


# =========================================================================
# _save_auth_store wires _align_owner_to_parent in
# =========================================================================

class TestSaveAuthStoreOwnership:
    def test_save_calls_align_owner_to_parent(self, isolated_auth_file):
        with patch("hermes_cli.auth._align_owner_to_parent") as mock_align:
            saved = auth_mod._save_auth_store({"providers": {"openai-codex": {"token": "x"}}})
        assert saved == isolated_auth_file
        mock_align.assert_called_once_with(isolated_auth_file)
        # And the file actually got written.
        on_disk = json.loads(saved.read_text())
        assert on_disk["providers"]["openai-codex"]["token"] == "x"


# =========================================================================
# _load_auth_store distinguishes PermissionError from "corrupt"
# =========================================================================

class TestLoadAuthStorePermissionError:
    def test_permission_error_does_not_create_corrupt_copy(
        self, isolated_auth_file, caplog,
    ):
        isolated_auth_file.write_text(json.dumps({"providers": {}}))
        with (
            patch.object(type(isolated_auth_file), "read_text",
                         side_effect=PermissionError("EACCES")),
            caplog.at_level(logging.WARNING, logger="hermes_cli.auth"),
        ):
            store = auth_mod._load_auth_store()

        assert store == {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
        corrupt = isolated_auth_file.with_suffix(".json.corrupt")
        assert not corrupt.exists(), "PermissionError shouldn't create a .corrupt copy"
        assert "not readable" in caplog.text
        assert "chown" in caplog.text  # Hint surfaced for the docker-exec case.

    def test_other_exceptions_still_create_corrupt_copy(self, isolated_auth_file):
        # Sanity-check: the existing path still treats a real parse error as corrupt.
        isolated_auth_file.write_text("{not valid json")
        store = auth_mod._load_auth_store()
        assert store["providers"] == {}
        corrupt = isolated_auth_file.with_suffix(".json.corrupt")
        assert corrupt.exists()
