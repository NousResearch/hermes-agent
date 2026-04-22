"""Consent-flow tests for the shell-hook allowlist.

Covers the prompt/non-prompt decision tree: TTY vs non-TTY, and the
three accept-hooks channels (--accept-hooks, HERMES_ACCEPT_HOOKS env,
hooks_auto_accept: config key).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import shell_hooks


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


def _write_hook_script(tmp_path: Path) -> Path:
    script = tmp_path / "hook.sh"
    script.write_text("#!/usr/bin/env bash\nprintf '{}\\n'\n")
    script.chmod(0o755)
    return script


# ── TTY prompt flow ───────────────────────────────────────────────────────


class TestTTYPromptFlow:
    def test_first_use_prompts_and_approves(self, tmp_path):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()

        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="y"):
            mock_stdin.isatty.return_value = True
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=False,
            )
        assert len(registered) == 1

        entry = shell_hooks.allowlist_entry_for("on_session_start", str(script))
        assert entry is not None
        assert entry["event"] == "on_session_start"
        assert entry["command"] == str(script)

    def test_first_use_prompts_and_rejects(self, tmp_path):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()

        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="n"):
            mock_stdin.isatty.return_value = True
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=False,
            )
        assert registered == []
        assert shell_hooks.allowlist_entry_for(
            "on_session_start", str(script),
        ) is None

    def test_subsequent_use_does_not_prompt(self, tmp_path):
        """After the first approval, re-registration must be silent."""
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()

        # First call: TTY, approved.
        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="y"):
            mock_stdin.isatty.return_value = True
            shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=False,
            )

        # Reset registration set but keep the allowlist on disk.
        shell_hooks.reset_for_tests()

        # Second call: TTY, input() must NOT be called.
        with patch("sys.stdin") as mock_stdin, patch(
            "builtins.input", side_effect=AssertionError("should not prompt"),
        ):
            mock_stdin.isatty.return_value = True
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=False,
            )
        assert len(registered) == 1


# ── non-TTY flow ──────────────────────────────────────────────────────────


class TestNonTTYFlow:
    def test_no_tty_no_flag_skips_registration(self, tmp_path):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=False,
            )
        assert registered == []

    def test_no_tty_with_argument_flag_accepts(self, tmp_path):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=True,
            )
        assert len(registered) == 1

    def test_no_tty_with_env_accepts(self, tmp_path, monkeypatch):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}},
                accept_hooks=False,
            )
        assert len(registered) == 1

    def test_no_tty_with_config_accepts(self, tmp_path):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            registered = shell_hooks.register_from_config(
                {
                    "hooks_auto_accept": True,
                    "hooks": {"on_session_start": [{"command": str(script)}]},
                },
                accept_hooks=False,
            )
        assert len(registered) == 1


# ── Allowlist + revoke + mtime ────────────────────────────────────────────


class TestAllowlistOps:
    def test_mtime_recorded_on_approval(self, tmp_path):
        script = _write_hook_script(tmp_path)
        shell_hooks._record_approval("on_session_start", str(script))

        entry = shell_hooks.allowlist_entry_for(
            "on_session_start", str(script),
        )
        assert entry is not None
        assert entry["script_mtime_at_approval"] is not None
        # ISO-8601 Z-suffix
        assert entry["script_mtime_at_approval"].endswith("Z")

    def test_revoke_removes_entry(self, tmp_path):
        script = _write_hook_script(tmp_path)
        shell_hooks._record_approval("on_session_start", str(script))
        assert shell_hooks.allowlist_entry_for(
            "on_session_start", str(script),
        ) is not None

        removed = shell_hooks.revoke(str(script))
        assert removed == 1
        assert shell_hooks.allowlist_entry_for(
            "on_session_start", str(script),
        ) is None

    def test_revoke_unknown_returns_zero(self, tmp_path):
        assert shell_hooks.revoke(str(tmp_path / "never-approved.sh")) == 0

    def test_tilde_path_approval_records_resolvable_mtime(self, tmp_path, monkeypatch):
        """If the command uses ~ the approval must still find the file."""
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "hook.sh"
        target.write_text("#!/usr/bin/env bash\n")
        target.chmod(0o755)

        shell_hooks._record_approval("on_session_start", "~/hook.sh")
        entry = shell_hooks.allowlist_entry_for(
            "on_session_start", "~/hook.sh",
        )
        assert entry is not None
        # Must not be None — the tilde was expanded before stat().
        assert entry["script_mtime_at_approval"] is not None

    def test_duplicate_approval_replaces_mtime(self, tmp_path):
        """Re-approving the same pair refreshes the approval timestamp."""
        script = _write_hook_script(tmp_path)
        shell_hooks._record_approval("on_session_start", str(script))
        original_entry = shell_hooks.allowlist_entry_for(
            "on_session_start", str(script),
        )
        assert original_entry is not None

        # Touch the script to bump its mtime then re-approve.
        import os
        import time
        new_mtime = original_entry.get("script_mtime_at_approval")
        time.sleep(0.01)
        os.utime(script, None)  # current time

        shell_hooks._record_approval("on_session_start", str(script))

        # Exactly one entry per (event, command).
        approvals = shell_hooks.load_allowlist().get("approvals", [])
        matching = [
            e for e in approvals
            if e.get("event") == "on_session_start"
            and e.get("command") == str(script)
        ]
        assert len(matching) == 1
