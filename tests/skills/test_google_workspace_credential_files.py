"""Regression test: google-workspace SKILL.md must declare required_credential_files.

PR #9931 accidentally removed the required_credential_files header, which broke
credential file mounting in Docker/Modal remote backends (#16452). This test
prevents the regression from silently reappearing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/SKILL.md"
)

_EXPECTED_PATHS = {"google_token.json", "google_client_secret.json", "google_workspace_auth_contexts.json"}


def _parse_frontmatter(content: str) -> dict:
    from agent.skill_utils import parse_frontmatter

    fm, _ = parse_frontmatter(content)
    return fm


class TestGoogleWorkspaceCredentialFiles:
    def test_required_credential_files_present_in_skill_md(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        fm = _parse_frontmatter(content)
        entries = fm.get("required_credential_files")
        assert entries, "required_credential_files missing from google-workspace SKILL.md"
        assert isinstance(entries, list), "required_credential_files must be a list"
        paths = {
            (e["path"] if isinstance(e, dict) else e)
            for e in entries
        }
        assert _EXPECTED_PATHS <= paths, (
            f"Missing entries in required_credential_files: {_EXPECTED_PATHS - paths}"
        )
        groups = {
            entry["path"]: entry.get("alternative_group")
            for entry in entries
            if isinstance(entry, dict)
        }
        assert groups == {
            "google_workspace_auth_contexts.json": "named-context",
            "google_token.json": "legacy",
            "google_client_secret.json": "legacy",
        }
        entries_by_path = {
            entry["path"]: entry for entry in entries if isinstance(entry, dict)
        }
        assert (
            entries_by_path["google_workspace_auth_contexts.json"][
                "readiness_json_path"
            ]
            == "contexts.*.token"
        )
        required_token_keys = ["refresh_token", "client_id", "client_secret"]
        assert (
            entries_by_path["google_workspace_auth_contexts.json"][
                "readiness_json_required_keys"
            ]
            == required_token_keys
        )
        assert (
            entries_by_path["google_token.json"]["readiness_json_required_keys"]
            == required_token_keys
        )

    def test_entries_are_registered_when_files_exist(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_token.json").write_text(
            json.dumps(
                {
                    "refresh_token": "refresh",
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                }
            )
        )
        (hermes_home / "google_client_secret.json").write_text("{}")
        (hermes_home / "google_workspace_auth_contexts.json").write_text("{}")

        from tools.credential_files import (
            clear_credential_files,
            get_credential_file_mounts,
            register_credential_files,
        )

        clear_credential_files()
        try:
            content = SKILL_MD.read_text(encoding="utf-8")
            fm = _parse_frontmatter(content)
            entries = fm.get("required_credential_files", [])

            with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
                missing = register_credential_files(entries)

            assert missing == [], f"Unexpected missing files: {missing}"
            mounts = get_credential_file_mounts()
            container_paths = {m["container_path"] for m in mounts}
            assert "/root/.hermes/google_token.json" in container_paths
            assert "/root/.hermes/google_client_secret.json" in container_paths
        finally:
            clear_credential_files()

    def test_named_context_only_is_available_and_mounted(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_workspace_auth_contexts.json").write_text(
            json.dumps(
                {
                    "contexts": {
                        "named": {
                            "token": {
                                "refresh_token": "refresh",
                                "client_id": "client-id",
                                "client_secret": "client-secret",
                            }
                        }
                    }
                }
            )
        )

        from tools.credential_files import (
            clear_credential_files,
            get_credential_file_mounts,
        )
        from tools.skills_tool import skill_view

        clear_credential_files()
        try:
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
                patch("tools.skills_tool.SKILLS_DIR", SKILL_MD.parents[2]),
            ):
                result = json.loads(skill_view("google-workspace"))

            assert result["setup_needed"] is False
            assert result["missing_credential_files"] == []
            container_paths = {
                mount["container_path"] for mount in get_credential_file_mounts()
            }
            assert container_paths == {
                "/root/.hermes/google_workspace_auth_contexts.json"
            }
        finally:
            clear_credential_files()

    def test_legacy_only_is_available_and_mounted(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_token.json").write_text(
            json.dumps(
                {
                    "refresh_token": "refresh",
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                }
            )
        )
        (hermes_home / "google_client_secret.json").write_text("{}")

        from tools.credential_files import (
            clear_credential_files,
            get_credential_file_mounts,
        )
        from tools.skills_tool import skill_view

        clear_credential_files()
        try:
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
                patch("tools.skills_tool.SKILLS_DIR", SKILL_MD.parents[2]),
            ):
                result = json.loads(skill_view("google-workspace"))

            assert result["setup_needed"] is False
            assert result["missing_credential_files"] == []
            container_paths = {
                mount["container_path"] for mount in get_credential_file_mounts()
            }
            assert container_paths == {
                "/root/.hermes/google_token.json",
                "/root/.hermes/google_client_secret.json",
            }
        finally:
            clear_credential_files()

    def test_no_credentials_requires_setup(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        from tools.credential_files import clear_credential_files
        from tools.skills_tool import skill_view

        clear_credential_files()
        try:
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
                patch("tools.skills_tool.SKILLS_DIR", SKILL_MD.parents[2]),
            ):
                result = json.loads(skill_view("google-workspace"))

            assert result["setup_needed"] is True
            assert result["readiness_status"] == "setup_needed"
            assert result["missing_credential_files"]
        finally:
            clear_credential_files()

    def test_named_context_without_token_requires_setup(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_workspace_auth_contexts.json").write_text(
            json.dumps(
                {
                    "contexts": {
                        "named": {
                            "client_secret": {"installed": {"client_id": "client"}}
                        }
                    }
                }
            )
        )

        from tools.credential_files import clear_credential_files
        from tools.skills_tool import skill_view

        clear_credential_files()
        try:
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
                patch("tools.skills_tool.SKILLS_DIR", SKILL_MD.parents[2]),
            ):
                result = json.loads(skill_view("google-workspace"))

            assert result["setup_needed"] is True
            assert result["readiness_status"] == "setup_needed"
            assert result["missing_credential_files"]
        finally:
            clear_credential_files()

    def test_named_refresh_token_without_client_fields_requires_setup(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_workspace_auth_contexts.json").write_text(
            json.dumps(
                {
                    "contexts": {
                        "named": {"token": {"refresh_token": "refresh"}}
                    }
                }
            )
        )

        from tools.credential_files import clear_credential_files
        from tools.skills_tool import skill_view

        clear_credential_files()
        try:
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
                patch("tools.skills_tool.SKILLS_DIR", SKILL_MD.parents[2]),
            ):
                result = json.loads(skill_view("google-workspace"))

            assert result["setup_needed"] is True
            assert result["readiness_status"] == "setup_needed"
        finally:
            clear_credential_files()

    def test_incomplete_legacy_layout_requires_setup(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_client_secret.json").write_text("{}")

        from tools.credential_files import clear_credential_files
        from tools.skills_tool import skill_view

        clear_credential_files()
        try:
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
                patch("tools.skills_tool.SKILLS_DIR", SKILL_MD.parents[2]),
            ):
                result = json.loads(skill_view("google-workspace"))

            assert result["setup_needed"] is True
            assert result["readiness_status"] == "setup_needed"
            assert result["missing_credential_files"]
        finally:
            clear_credential_files()
