"""The dispatcher wrapper must not preempt a shell hook's own gate (#132096).

A shell hook registered for ``pre_tool_call`` carries its own matcher, subprocess timeout and
``fail_closed`` policy. These tests pin the interaction contract: the matcher decides which tools
the hook gates, the shell layer's timeout decision is the one that counts, and every
skip/suppression outcome honors the hook's ``fail_closed`` — never the dispatcher's blanket
fail-closed policy for ``pre_tool_call``.
"""

from __future__ import annotations

import time

from agent import shell_hooks
from hermes_cli.plugins import PluginManager, _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE


def _spawn_result(**overrides):
    result = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False,
              "elapsed_seconds": 0.0, "error": None}
    result.update(overrides)
    return result


def _shell_hook_callback(*, fail_closed: bool, matcher: str = "terminal", timeout: float = 0.1):
    spec = shell_hooks.ShellHookSpec(
        event="pre_tool_call", command="python hook.py", matcher=matcher,
        timeout=timeout, fail_closed=fail_closed)
    return shell_hooks._make_callback(spec)


class TestShellHookWrapperTimeout:
    def test_wrapper_timeout_fails_open_for_fail_open_hook(self, monkeypatch):
        """Dispatcher wrapper expires first with fail_closed: false — no block directive (#132096)."""
        import hermes_cli.plugins as plugins_mod
        import hermes_cli.plugins_dispatch as dispatch_mod

        monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.15)
        monkeypatch.setattr(dispatch_mod, "_SHELL_HOOK_WRAPPER_MARGIN_SECS", 0.2, raising=False)
        monkeypatch.setattr(
            shell_hooks, "_spawn", lambda spec, stdin_json: (time.sleep(5), _spawn_result())[1])

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [_shell_hook_callback(fail_closed=False)]
        t0 = time.monotonic()
        results = mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={}, tool_call_id="t1")
        assert time.monotonic() - t0 < 3.0  # the wrapper cap (0.1 + 0.2) expired, not the sleep
        assert results == []  # fail open: fail_closed: false must not block however it times out

    def test_wrapper_timeout_fails_closed_when_hook_declares_it(self, monkeypatch):
        """A shell hook that declares fail_closed: true keeps the block on a wrapper timeout."""
        import hermes_cli.plugins as plugins_mod
        import hermes_cli.plugins_dispatch as dispatch_mod

        monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.15)
        monkeypatch.setattr(dispatch_mod, "_SHELL_HOOK_WRAPPER_MARGIN_SECS", 0.2, raising=False)
        monkeypatch.setattr(
            shell_hooks, "_spawn", lambda spec, stdin_json: (time.sleep(5), _spawn_result())[1])

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [_shell_hook_callback(fail_closed=True)]
        results = mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={}, tool_call_id="t1")
        assert results == [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]

    def test_suppression_window_never_blocks_unmatched_or_fail_open_tools(self, monkeypatch):
        """After one wrapper timeout, unmatched tools bypass the hook and matched ones fail open.

        Pre-fix, the 60 s suppression blocked every tool of every kind for a fail-open hook.
        """
        import hermes_cli.plugins as plugins_mod
        import hermes_cli.plugins_dispatch as dispatch_mod

        monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.15)
        monkeypatch.setattr(dispatch_mod, "_SHELL_HOOK_WRAPPER_MARGIN_SECS", 0.2, raising=False)
        monkeypatch.setattr(
            shell_hooks, "_spawn", lambda spec, stdin_json: (time.sleep(5), _spawn_result())[1])

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [_shell_hook_callback(fail_closed=False)]
        assert mgr.invoke_hook(
            "pre_tool_call", tool_name="terminal", args={}, tool_call_id="t1") == []
        # Inside the suppression window now: the matcher keeps non-matching tools out entirely…
        assert mgr.invoke_hook(
            "pre_tool_call", tool_name="read_file", args={}, tool_call_id="t2") == []
        # …and a matching tool is skipped fail-open, not fail-closed.
        assert mgr.invoke_hook(
            "pre_tool_call", tool_name="terminal", args={}, tool_call_id="t3") == []


class TestShellLayerDecisionWins:
    def test_shell_timeout_fails_open_before_wrapper_expires(self, monkeypatch):
        """The shell layer's own timeout returns first and its fail-open decision is final."""
        import hermes_cli.plugins as plugins_mod
        import hermes_cli.plugins_dispatch as dispatch_mod

        monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 30.0)
        monkeypatch.setattr(dispatch_mod, "_SHELL_HOOK_WRAPPER_MARGIN_SECS", 10.0, raising=False)
        monkeypatch.setattr(shell_hooks, "_spawn", lambda spec, stdin_json: _spawn_result(timed_out=True))

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [_shell_hook_callback(fail_closed=False)]
        results = mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={}, tool_call_id="t1")
        assert results == []
        # No wrapper timeout fired, so no suppression window either.
        assert mgr._hook_timeout_suppressed_until == {}

    def test_shell_timeout_block_carries_the_hook_message_not_the_wrapper_one(self, monkeypatch):
        """A fail-closed shell hook times out in the shell layer — the block names the hook, not the dispatcher."""
        import hermes_cli.plugins as plugins_mod
        import hermes_cli.plugins_dispatch as dispatch_mod

        monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 30.0)
        monkeypatch.setattr(dispatch_mod, "_SHELL_HOOK_WRAPPER_MARGIN_SECS", 10.0, raising=False)
        monkeypatch.setattr(shell_hooks, "_spawn", lambda spec, stdin_json: _spawn_result(timed_out=True))

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [_shell_hook_callback(fail_closed=True)]
        results = mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={}, tool_call_id="t1")
        assert results == [{
            "action": "block",
            "message": "hook python hook.py failed closed: timed out after 0.1s",
        }]
