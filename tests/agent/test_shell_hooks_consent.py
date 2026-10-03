"""Consent-flow tests for the shell-hook allowlist.

Covers the prompt/non-prompt decision tree: TTY vs non-TTY, and the
three accept-hooks channels (--accept-hooks, HERMES_ACCEPT_HOOKS env,
hooks_auto_accept: config key).
"""

from __future__ import annotations

import asyncio
import threading
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



# ── Allowlist + revoke + mtime ────────────────────────────────────────────


class TestAllowlistOps:



    @pytest.mark.platforms("linux")
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


# ── hooks_auto_accept config parsing ──────────────────────────────────────


class TestHooksAutoAcceptParsing:
    """Regression guard: YAML-string values must not silently auto-accept.

    ``bool("false")`` is ``True`` in Python, so the old ``return bool(cfg_val)``
    path treated ``hooks_auto_accept: "false"`` (quoted YAML string) as a
    truthy opt-in, silently bypassing user consent for every shell hook.
    """

    def test_bool_true_accepts(self):
        assert shell_hooks._resolve_effective_accept(
            {"hooks_auto_accept": True}, accept_hooks_arg=False,
        ) is True








    def test_none_rejects(self):
        assert shell_hooks._resolve_effective_accept(
            {"hooks_auto_accept": None}, accept_hooks_arg=False,
        ) is False

    def test_integer_ignored(self):
        # Only bool and str are honored; anything else (including 1) is False.
        assert shell_hooks._resolve_effective_accept(
            {"hooks_auto_accept": 1}, accept_hooks_arg=False,
        ) is False



# ── Consent prompt on the event loop thread (#120356) ──────────────────────

class TestEventLoopConsentDoesNotBlockForever:
    """Regression for #120356.

    A launcher that hides a console still leaves stdin a TTY, so ``register_from_config()``
    reached ``input()`` on the asyncio event loop thread during gateway startup. The loop
    stopped ticking, the liveness watchdog missed its probes and hard-exited the process.
    The contract: a prompt raised on the event loop is bounded and fails closed.
    """

    def test_unanswered_prompt_on_loop_thread_declines_instead_of_hanging(self, tmp_path, monkeypatch):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        monkeypatch.setattr(shell_hooks, "_CONSENT_TIMEOUT_S", 0.05)

        async def scenario() -> list:
            with patch("sys.stdin") as mock_stdin, patch(
                "builtins.input", lambda *_a, **_k: "y",  # would approve if it were ever reached
            ):
                mock_stdin.isatty.return_value = True  # hidden-but-present console
                # A read that never returns models a console nobody is watching.
                with patch(
                    "agent.shell_hooks._read_answer_off_loop", return_value=None,
                ):
                    return shell_hooks.register_from_config(
                        {"hooks": {"on_session_start": [{"command": str(script)}]}},
                        accept_hooks=False,
                    )

        registered = asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
        assert registered == [], "an unanswered consent prompt must fail closed"
        assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is None

    def test_answered_prompt_on_loop_thread_still_approves(self, tmp_path, monkeypatch):
        """The bound must not cost us the prompt: a real answer is still honoured."""
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        monkeypatch.setattr(shell_hooks, "_CONSENT_TIMEOUT_S", 5.0)

        async def scenario() -> list:
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                return shell_hooks.register_from_config(
                    {"hooks": {"on_session_start": [{"command": str(script)}]}},
                    accept_hooks=False,
                )

        # input() is patched globally so the worker thread reads the same stubbed stream.
        with patch("builtins.input", return_value="y"):
            registered = asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))
        assert len(registered) == 1
        assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is not None

    def test_loop_thread_prompt_reads_off_the_loop_thread(self, tmp_path, monkeypatch):
        """The invariant behind the fix: the blocking read never runs on the loop thread.

        ``register_from_config`` is synchronous, so the loop cannot tick while it is on the
        stack either way. What matters is that the *read* is dispatched to a worker, leaving
        the loop free to run its own callbacks (the liveness probe) during the wait.
        """
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        monkeypatch.setattr(shell_hooks, "_CONSENT_TIMEOUT_S", 5.0)

        loop_thread = threading.current_thread()
        read_threads: list = []

        def record_thread(_prompt: str = "") -> str:
            read_threads.append(threading.current_thread())
            return "y"

        async def scenario() -> list:
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                return shell_hooks.register_from_config(
                    {"hooks": {"on_session_start": [{"command": str(script)}]}},
                    accept_hooks=False,
                )

        with patch("builtins.input", record_thread):
            asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))

        assert read_threads, "the consent prompt never read an answer"
        assert loop_thread not in read_threads, (
            "the consent read ran on the event loop thread — the gateway would freeze there"
        )
