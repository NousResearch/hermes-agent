"""Consent-flow tests for the shell-hook allowlist.

Covers the prompt/non-prompt decision tree: TTY vs non-TTY, and the
three accept-hooks channels (--accept-hooks, HERMES_ACCEPT_HOOKS env,
hooks_auto_accept: config key).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
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
    script.write_text("#!/usr/bin/env bash\nprintf '{}\\n'\n", encoding="utf-8")
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
                accept_hooks=False, interactive=True,
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
                accept_hooks=False, interactive=True,
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
                accept_hooks=False, interactive=True,
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
                accept_hooks=False, interactive=True,
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



# ── Never prompt outside a foreground CLI command (#120356, #127822) ──────


def _tty_stdin_that_nobody_answers(monkeypatch) -> list:
    """A hidden console (Windows ``.vbs`` launcher) or ``NUL`` makes ``isatty()`` True with no
    person behind it; record every ``input()`` instead of blocking on it."""
    asked: list = []
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda *a: asked.append(a) or "y")
    return asked


def _write_config_with_hook(tmp_path: Path) -> Path:
    script = _write_hook_script(tmp_path)
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"hooks:\n  on_session_start:\n    - command: {json.dumps(str(script))}\n", encoding="utf-8")
    return script


class TestNeverPromptOutsideForegroundCLI:
    def test_default_registration_declines_without_reading_a_tty_stdin(self, tmp_path, monkeypatch, caplog):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        asked = _tty_stdin_that_nobody_answers(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="agent.shell_hooks"):
            registered = shell_hooks.register_from_config(
                {"hooks": {"on_session_start": [{"command": str(script)}]}})

        assert asked == []
        assert registered == []
        assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is None
        assert str(shell_hooks.allowlist_path()) in caplog.text  # names the profile home to approve in

    def test_opt_in_still_approves_without_a_prompt(self, tmp_path, monkeypatch):
        from hermes_cli import plugins

        script = _write_hook_script(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        asked = _tty_stdin_that_nobody_answers(monkeypatch)
        monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")

        registered = shell_hooks.register_from_config(
            {"hooks": {"on_session_start": [{"command": str(script)}]}})

        assert asked == []
        assert len(registered) == 1
        assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is not None

    def test_gateway_registration_on_the_event_loop_never_prompts(self, tmp_path, monkeypatch, caplog):
        """The P0 shape: gateway startup and every secondary-profile load register from the loop
        thread; a prompt there froze the loop until the liveness watchdog exited 75."""
        from gateway.run_startup import GatewayStartupMixin
        from hermes_cli import plugins

        script = _write_config_with_hook(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        asked = _tty_stdin_that_nobody_answers(monkeypatch)

        async def _boot() -> None:
            asyncio.get_running_loop()  # registered from the loop thread, as startup does
            GatewayStartupMixin._register_config_hooks("registration failed", level=logging.ERROR)

        with caplog.at_level(logging.WARNING):
            asyncio.run(_boot())

        assert asked == []
        assert "registration failed" not in caplog.text
        assert "not allowlisted" in caplog.text and str(script) in caplog.text
        assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is None
        assert plugins._plugin_manager._hooks.get("on_session_start", []) == []

    def test_force_reload_reregistration_never_prompts(self, tmp_path, monkeypatch, caplog):
        from hermes_cli import plugins

        script = _write_config_with_hook(tmp_path)
        plugins._plugin_manager = plugins.PluginManager()
        asked = _tty_stdin_that_nobody_answers(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="agent.shell_hooks"):
            shell_hooks.re_register_config_hooks()

        assert asked == []
        assert "not allowlisted" in caplog.text
        assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is None


# ── Allowlist + revoke + mtime ────────────────────────────────────────────


class TestAllowlistOps:



    @pytest.mark.platforms("linux")
    def test_tilde_path_approval_records_resolvable_mtime(self, tmp_path, monkeypatch):
        """If the command uses ~ the approval must still find the file."""
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "hook.sh"
        target.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
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




# ── approval covers the script bytes it saw ──────────────────────────────


def _fire(script: Path, *, fail_closed: bool = False):
    spec = shell_hooks.ShellHookSpec(event="pre_tool_call", command=str(script), fail_closed=fail_closed)
    return shell_hooks._make_callback(spec)(tool_name="terminal", args={}, session_id="s", task_id="t")


def _marker_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "guard.sh"
    script.write_text(f"#!/usr/bin/env bash\necho {body} >> {tmp_path / 'ran.log'}\nprintf '{{}}\\n'\n", encoding="utf-8")
    script.chmod(0o755)
    return script


@pytest.mark.platforms("posix")
def test_changed_script_does_not_run_until_approved_again(tmp_path):
    """An approved hook whose script changed afterwards never runs the new bytes: skipped, or a block when
    fail_closed; approving it again (hash refreshed) lets it run."""
    script = _marker_script(tmp_path, "approved")
    shell_hooks._record_approval("pre_tool_call", str(script))
    assert _fire(script) is None
    _marker_script(tmp_path, "tampered")  # same path, new bytes (an update, a pull, an edit)

    assert _fire(script) is None
    blocked = _fire(script, fail_closed=True)
    assert blocked["action"] == "block" and "changed since approval" in blocked["message"]
    assert (tmp_path / "ran.log").read_text().split() == ["approved"]

    from hermes_cli import plugins
    plugins._plugin_manager = plugins.PluginManager()
    cfg = {"hooks": {"pre_tool_call": [{"command": str(script), "fail_closed": True}]}}
    with patch("sys.stdin") as stdin, patch("builtins.input", return_value="n"):
        stdin.isatty.return_value = True  # declining the re-ask keeps the guard wired, still refusing
        assert len(shell_hooks.register_from_config(cfg, interactive=True)) == 1
    verdicts = plugins._plugin_manager.invoke_hook("pre_tool_call", tool_name="terminal", args={}, session_id="s", task_id="t")
    assert any(isinstance(v, dict) and v.get("action") == "block" for v in verdicts)

    shell_hooks.reset_for_tests()  # a fresh worker process registering with --accept-hooks re-pins
    shell_hooks.register_from_config(cfg, accept_hooks=True)
    _fire(script)
    assert (tmp_path / "ran.log").read_text().split() == ["approved", "tampered"]


@pytest.mark.platforms("posix")
def test_approval_from_before_pinning_is_pinned_while_unchanged(tmp_path):
    """An allowlist entry written before content pinning keeps working while its recorded mtime still
    matches, gets its hash pinned at that run, and a later edit is then refused; one whose mtime moved
    since approval is refused at once."""
    script = _marker_script(tmp_path, "v1")
    legacy = {"event": "pre_tool_call", "command": str(script), "approved_at": "2026-01-01T00:00:00Z",
              "script_mtime_at_approval": shell_hooks.script_mtime_iso(str(script))}
    shell_hooks.save_allowlist({"approvals": [legacy]})

    _fire(script)
    pinned = shell_hooks.allowlist_entry_for("pre_tool_call", str(script))
    assert pinned["script_hash_at_approval"] == shell_hooks.script_content_hash(str(script))
    _marker_script(tmp_path, "v2")
    _fire(script)
    assert (tmp_path / "ran.log").read_text().split() == ["v1"]

    shell_hooks.save_allowlist({"approvals": [{**legacy, "script_mtime_at_approval": "2000-01-01T00:00:00Z"}]})
    assert _fire(script, fail_closed=True)["action"] == "block"
