"""Tests for hermes_cli/web_git.py — commit attribution trailers."""

import pytest
from unittest.mock import MagicMock


# ── _build_commit_trailer ──────────────────────────────────────────────


class TestBuildCommitTrailer:
    """Unit tests for the _build_commit_trailer helper."""

    def test_default_trailer(self, monkeypatch):
        """Default config produces Co-authored-by: Hermes Agent <hermes@nousresearch.com>."""
        from hermes_cli.web_git import _build_commit_trailer

        # Simulate a config with git section missing (falls through to defaults)
        def mock_load_config():
            return {"git": {}}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == "\nCo-authored-by: Hermes Agent <hermes@nousresearch.com>\n"

    def test_default_trailer_missing_git_section(self, monkeypatch):
        """When the git section is entirely absent, defaults are used."""
        from hermes_cli.web_git import _build_commit_trailer

        def mock_load_config():
            return {}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == "\nCo-authored-by: Hermes Agent <hermes@nousresearch.com>\n"

    def test_trailer_disabled(self, monkeypatch):
        """commit_trailer: False returns empty string."""
        from hermes_cli.web_git import _build_commit_trailer

        def mock_load_config():
            return {"git": {"commit_trailer": False}}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == ""

    def test_generated_by_trailer(self, monkeypatch):
        """commit_trailer_type: 'generated-by' produces Generated-by: Hermes Agent."""
        from hermes_cli.web_git import _build_commit_trailer

        def mock_load_config():
            return {"git": {"commit_trailer_type": "generated-by"}}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == "\nGenerated-by: Hermes Agent\n"

    def test_custom_name_and_email(self, monkeypatch):
        """Custom name/email are reflected in the trailer."""
        from hermes_cli.web_git import _build_commit_trailer

        def mock_load_config():
            return {
                "git": {
                    "commit_trailer_name": "MyBot",
                    "commit_trailer_email": "bot@example.com",
                }
            }

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == "\nCo-authored-by: MyBot <bot@example.com>\n"

    def test_custom_name_generated_by(self, monkeypatch):
        """Custom name with generated-by type."""
        from hermes_cli.web_git import _build_commit_trailer

        def mock_load_config():
            return {
                "git": {
                    "commit_trailer_type": "generated-by",
                    "commit_trailer_name": "CodingAgent",
                }
            }

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == "\nGenerated-by: CodingAgent\n"

    def test_load_config_exception_is_safe(self, monkeypatch):
        """When load_config raises, _build_commit_trailer returns empty string."""
        from hermes_cli.web_git import _build_commit_trailer

        def mock_load_config():
            raise RuntimeError("config broken")

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        result = _build_commit_trailer()
        assert result == ""


# ── review_commit ──────────────────────────────────────────────────────


class TestReviewCommitTrailer:
    """Integration-style: verify the trailer is actually passed to git commit."""

    def test_trailer_appended_to_message(self, monkeypatch):
        """review_commit passes message + trailer to git."""
        from hermes_cli.web_git import review_commit

        # Mock config to return a known trailer
        def mock_load_config():
            return {"git": {"commit_trailer_type": "generated-by"}}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        # Track the actual git commit command
        actual_commit_args = []

        def fake_git(cwd, args):
            actual_commit_args.append(args)
            return (0, "", "")

        def fake_git_ok(cwd, args):
            actual_commit_args.append(args)
            return None

        monkeypatch.setattr("hermes_cli.web_git._git", fake_git)
        monkeypatch.setattr("hermes_cli.web_git._git_ok", fake_git_ok)
        monkeypatch.setattr("hermes_cli.web_git._git_out", lambda *a: "## main...origin/main\n")
        monkeypatch.setattr("hermes_cli.web_git._entry_staged", lambda *a: True)

        review_commit(".", "feat: add widget", push=False)

        # Find the commit command
        commit_cmd = None
        for args in actual_commit_args:
            if args and args[0] == "commit":
                commit_cmd = args
                break

        assert commit_cmd is not None, "git commit was never called"
        assert commit_cmd == [
            "commit", "-m",
            "feat: add widget\nGenerated-by: Hermes Agent\n",
        ], f"Unexpected commit args: {commit_cmd}"

    def test_trailer_disabled_no_trailer_in_message(self, monkeypatch):
        """When commit_trailer is false, the message is passed as-is."""
        from hermes_cli.web_git import review_commit

        def mock_load_config():
            return {"git": {"commit_trailer": False}}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        actual_commit_args = []

        def fake_git_ok(cwd, args):
            actual_commit_args.append(args)
            return None

        monkeypatch.setattr("hermes_cli.web_git._git", lambda *a: (0, "", ""))
        monkeypatch.setattr("hermes_cli.web_git._git_ok", fake_git_ok)
        monkeypatch.setattr("hermes_cli.web_git._git_out", lambda *a: "## main...origin/main\n")
        monkeypatch.setattr("hermes_cli.web_git._entry_staged", lambda *a: True)

        review_commit(".", "feat: plain commit", push=False)

        commit_cmd = None
        for args in actual_commit_args:
            if args and args[0] == "commit":
                commit_cmd = args
                break

        assert commit_cmd is not None, "git commit was never called"
        assert commit_cmd == ["commit", "-m", "feat: plain commit"], (
            f"Unexpected commit args: {commit_cmd}"
        )

    def test_auto_stage_then_commit_with_trailer(self, monkeypatch):
        """When nothing is staged, review_commit stages everything first."""
        from hermes_cli.web_git import review_commit

        def mock_load_config():
            return {"git": {"commit_trailer_type": "generated-by"}}

        monkeypatch.setattr("hermes_cli.config.load_config", mock_load_config)

        actual_commands = []

        def fake_has_staged(tag, xy):
            # Simulate that nothing is staged
            return False

        def fake_git(cwd, args):
            if args[0] == "status":
                return (0, "1 . N... 0\tREADME.md\0", "")
            return (0, "", "")

        def fake_git_ok(cwd, args):
            actual_commands.append(args)
            return None

        monkeypatch.setattr("hermes_cli.web_git._git", fake_git)
        monkeypatch.setattr("hermes_cli.web_git._git_ok", fake_git_ok)
        monkeypatch.setattr("hermes_cli.web_git._entry_staged", fake_has_staged)

        review_commit(".", "fix: resolve timeout", push=False)

        # Should have: add -A then commit with trailer
        assert len(actual_commands) >= 2, f"Expected at least 2 commands, got: {actual_commands}"

        add_cmd = actual_commands[0]
        assert add_cmd == ["add", "-A"], f"Expected add -A, got: {add_cmd}"

        commit_cmd = actual_commands[1]
        assert commit_cmd == [
            "commit", "-m",
            "fix: resolve timeout\nGenerated-by: Hermes Agent\n",
        ], f"Unexpected commit: {commit_cmd}"
