"""Tests for the shell-hooks subprocess bridge (agent.shell_hooks).

These tests focus on the pure translation layer — JSON serialisation,
JSON parsing, matcher behaviour, block-schema correctness, and the
subprocess runner's graceful error handling.  Consent prompts are
covered in ``test_shell_hooks_consent.py``.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict

import pytest

from agent import shell_hooks


# ── helpers ───────────────────────────────────────────────────────────────


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return path


def _allowlist_pair(monkeypatch, tmp_path, event: str, command: str) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    shell_hooks._record_approval(event, command)


@pytest.fixture(autouse=True)
def _reset_registration_state():
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


# ── _parse_response ───────────────────────────────────────────────────────


class TestParseResponse:
    def test_block_claude_code_style(self):
        r = shell_hooks._parse_response(
            "pre_tool_call",
            '{"decision": "block", "reason": "nope"}',
        )
        assert r == {"action": "block", "message": "nope"}

    def test_block_canonical_style(self):
        r = shell_hooks._parse_response(
            "pre_tool_call",
            '{"action": "block", "message": "nope"}',
        )
        assert r == {"action": "block", "message": "nope"}

    def test_block_canonical_wins_over_claude_style(self):
        r = shell_hooks._parse_response(
            "pre_tool_call",
            '{"action": "block", "message": "canonical", '
            '"decision": "block", "reason": "claude"}',
        )
        assert r == {"action": "block", "message": "canonical"}

    def test_empty_stdout_returns_none(self):
        assert shell_hooks._parse_response("pre_tool_call", "") is None
        assert shell_hooks._parse_response("pre_tool_call", "   ") is None

    def test_invalid_json_returns_none(self):
        assert shell_hooks._parse_response("pre_tool_call", "not json") is None

    def test_non_dict_json_returns_none(self):
        assert shell_hooks._parse_response("pre_tool_call", "[1, 2]") is None

    def test_non_block_pre_tool_call_returns_none(self):
        r = shell_hooks._parse_response("pre_tool_call", '{"decision": "allow"}')
        assert r is None

    def test_pre_llm_call_context_passthrough(self):
        r = shell_hooks._parse_response(
            "pre_llm_call", '{"context": "today is Friday"}',
        )
        assert r == {"context": "today is Friday"}

    def test_subagent_stop_context_passthrough(self):
        r = shell_hooks._parse_response(
            "subagent_stop", '{"context": "child role=leaf"}',
        )
        assert r == {"context": "child role=leaf"}

    def test_pre_llm_call_block_ignored(self):
        """Only pre_tool_call honors block directives."""
        r = shell_hooks._parse_response(
            "pre_llm_call", '{"decision": "block", "reason": "no"}',
        )
        assert r is None


# ── _serialize_payload ────────────────────────────────────────────────────


class TestSerializePayload:
    def test_basic_pre_tool_call_schema(self):
        raw = shell_hooks._serialize_payload(
            "pre_tool_call",
            {
                "tool_name": "terminal",
                "args": {"command": "ls"},
                "session_id": "sess-1",
                "task_id": "t-1",
                "tool_call_id": "c-1",
            },
        )
        payload = json.loads(raw)
        assert payload["hook_event_name"] == "pre_tool_call"
        assert payload["tool_name"] == "terminal"
        assert payload["tool_input"] == {"command": "ls"}
        assert payload["session_id"] == "sess-1"
        assert "cwd" in payload
        # task_id / tool_call_id end up under extra
        assert payload["extra"]["task_id"] == "t-1"
        assert payload["extra"]["tool_call_id"] == "c-1"

    def test_args_not_dict_becomes_null(self):
        raw = shell_hooks._serialize_payload(
            "pre_tool_call", {"args": ["not", "a", "dict"]},
        )
        payload = json.loads(raw)
        assert payload["tool_input"] is None

    def test_parent_session_id_used_when_no_session_id(self):
        raw = shell_hooks._serialize_payload(
            "subagent_stop", {"parent_session_id": "p-1"},
        )
        payload = json.loads(raw)
        assert payload["session_id"] == "p-1"

    def test_unserialisable_extras_stringified(self):
        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        raw = shell_hooks._serialize_payload(
            "on_session_start", {"obj": Weird()},
        )
        payload = json.loads(raw)
        assert payload["extra"]["obj"] == "<weird>"


# ── Matcher behaviour ─────────────────────────────────────────────────────


class TestMatcher:
    def test_no_matcher_fires_for_any_tool(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher=None,
        )
        assert spec.matches_tool("terminal")
        assert spec.matches_tool("write_file")

    def test_single_name_matcher(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="terminal",
        )
        assert spec.matches_tool("terminal")
        assert not spec.matches_tool("web_search")

    def test_alternation_matcher(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="terminal|file",
        )
        assert spec.matches_tool("terminal")
        assert spec.matches_tool("file")
        assert not spec.matches_tool("web")

    def test_invalid_regex_falls_back_to_literal(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="foo[bar",
        )
        assert spec.matches_tool("foo[bar")
        assert not spec.matches_tool("foo")

    def test_matcher_ignored_when_no_tool_name(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="terminal",
        )
        assert not spec.matches_tool(None)

    def test_matcher_leading_whitespace_stripped(self):
        """YAML quirks can introduce leading/trailing whitespace — must
        not silently break the matcher."""
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher=" terminal ",
        )
        assert spec.matcher == "terminal"
        assert spec.matches_tool("terminal")

    def test_matcher_trailing_newline_stripped(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="terminal\n",
        )
        assert spec.matches_tool("terminal")

    def test_whitespace_only_matcher_becomes_none(self):
        """A matcher that's pure whitespace is treated as 'no matcher'."""
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="   ",
        )
        assert spec.matcher is None
        assert spec.matches_tool("anything")


# ── End-to-end subprocess behaviour ───────────────────────────────────────


class TestCallbackSubprocess:
    def test_timeout_returns_none(self, tmp_path):
        # Script that sleeps forever; we set a 1s timeout.
        script = _write_script(
            tmp_path, "slow.sh",
            "#!/usr/bin/env bash\nsleep 60\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call", command=str(script), timeout=1,
        )
        cb = shell_hooks._make_callback(spec)
        assert cb(tool_name="terminal") is None

    def test_malformed_json_stdout_returns_none(self, tmp_path):
        script = _write_script(
            tmp_path, "bad_json.sh",
            "#!/usr/bin/env bash\necho 'not json at all'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
        )
        cb = shell_hooks._make_callback(spec)
        # Matcher is None so the callback fires for any tool.
        assert cb(tool_name="terminal") is None

    def test_non_zero_exit_with_block_stdout_still_blocks(self, tmp_path):
        """A script that signals failure via exit code AND prints a block
        directive must still block — scripts should be free to mix exit
        codes with parseable output."""
        script = _write_script(
            tmp_path, "exit1_block.sh",
            "#!/usr/bin/env bash\n"
            'printf \'{"decision": "block", "reason": "via exit 1"}\\n\'\n'
            "exit 1\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
        )
        cb = shell_hooks._make_callback(spec)
        assert cb(tool_name="terminal") == {"action": "block", "message": "via exit 1"}

    def test_block_translation_end_to_end(self, tmp_path):
        """v1 schema-bug regression gate.

        Shell hook returns the Claude-Code-style payload and the bridge
        must translate it to the canonical Hermes block shape so that
        get_pre_tool_call_block_message() surfaces the block.
        """
        script = _write_script(
            tmp_path, "blocker.sh",
            "#!/usr/bin/env bash\n"
            'printf \'{"decision": "block", "reason": "no terminal"}\\n\'\n',
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call",
            command=str(script),
            matcher="terminal",
        )
        cb = shell_hooks._make_callback(spec)
        result = cb(tool_name="terminal", args={"command": "rm -rf /"})
        assert result == {"action": "block", "message": "no terminal"}

    def test_block_aggregation_through_plugin_manager(self, tmp_path, monkeypatch):
        """Registering via register_from_config makes
        get_pre_tool_call_block_message surface the block — the real
        end-to-end control flow used by run_agent._invoke_tool."""
        from hermes_cli import plugins

        script = _write_script(
            tmp_path, "block.sh",
            "#!/usr/bin/env bash\n"
            'printf \'{"decision": "block", "reason": "blocked-by-shell"}\\n\'\n',
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")

        # Fresh manager
        plugins._plugin_manager = plugins.PluginManager()

        cfg = {
            "hooks": {
                "pre_tool_call": [
                    {"matcher": "terminal", "command": str(script)},
                ],
            },
        }
        registered = shell_hooks.register_from_config(cfg, accept_hooks=True)
        assert len(registered) == 1

        msg = plugins.get_pre_tool_call_block_message(
            tool_name="terminal",
            args={"command": "rm"},
        )
        assert msg == "blocked-by-shell"

    def test_matcher_regex_filters_callback(self, tmp_path, monkeypatch):
        """A matcher set to 'terminal' must not fire for 'web_search'."""
        calls = tmp_path / "calls.log"
        script = _write_script(
            tmp_path, "log.sh",
            f"#!/usr/bin/env bash\n"
            f"echo \"$(cat -)\" >> {calls}\n"
            f"printf '{{}}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call",
            command=str(script),
            matcher="terminal",
        )
        cb = shell_hooks._make_callback(spec)
        cb(tool_name="terminal", args={"command": "ls"})
        cb(tool_name="web_search", args={"q": "x"})
        cb(tool_name="file_read", args={"path": "x"})
        assert calls.exists()
        # Only the terminal call wrote to the log
        assert calls.read_text().count("pre_tool_call") == 1

    def test_payload_schema_delivered(self, tmp_path):
        capture = tmp_path / "payload.json"
        script = _write_script(
            tmp_path, "capture.sh",
            f"#!/usr/bin/env bash\ncat - > {capture}\nprintf '{{}}\\n'\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
        )
        cb = shell_hooks._make_callback(spec)
        cb(
            tool_name="terminal",
            args={"command": "echo hi"},
            session_id="sess-77",
            task_id="task-77",
        )
        payload = json.loads(capture.read_text())
        assert payload["hook_event_name"] == "pre_tool_call"
        assert payload["tool_name"] == "terminal"
        assert payload["tool_input"] == {"command": "echo hi"}
        assert payload["session_id"] == "sess-77"
        assert "cwd" in payload
        assert payload["extra"]["task_id"] == "task-77"

    def test_pre_llm_call_context_flows_through(self, tmp_path):
        script = _write_script(
            tmp_path, "ctx.sh",
            "#!/usr/bin/env bash\n"
            'printf \'{"context": "env-note"}\\n\'\n',
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_llm_call", command=str(script),
        )
        cb = shell_hooks._make_callback(spec)
        result = cb(
            session_id="s1", user_message="hello",
            conversation_history=[], is_first_turn=True,
            model="gpt-4", platform="cli",
        )
        assert result == {"context": "env-note"}

    def test_shlex_handles_paths_with_spaces(self, tmp_path):
        dir_with_space = tmp_path / "path with space"
        dir_with_space.mkdir()
        script = _write_script(
            dir_with_space, "ok.sh",
            "#!/usr/bin/env bash\nprintf '{}\\n'\n",
        )
        # Quote the path so shlex keeps it as a single token.
        spec = shell_hooks.ShellHookSpec(
            event="post_tool_call",
            command=f'"{script}"',
        )
        cb = shell_hooks._make_callback(spec)
        # No crash = shlex parsed it correctly.
        assert cb(tool_name="terminal") is None  # empty object parses to None

    def test_missing_binary_logged_not_raised(self, tmp_path):
        spec = shell_hooks.ShellHookSpec(
            event="on_session_start",
            command=str(tmp_path / "does-not-exist"),
        )
        cb = shell_hooks._make_callback(spec)
        # Must not raise — agent loop should continue.
        assert cb(session_id="s") is None

    def test_non_executable_binary_logged_not_raised(self, tmp_path):
        path = tmp_path / "no-exec"
        path.write_text("#!/usr/bin/env bash\necho hi\n")
        # Intentionally do NOT chmod +x.
        spec = shell_hooks.ShellHookSpec(
            event="on_session_start", command=str(path),
        )
        cb = shell_hooks._make_callback(spec)
        assert cb(session_id="s") is None


# ── config parsing ────────────────────────────────────────────────────────


class TestParseHooksBlock:
    def test_valid_entry(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {"matcher": "terminal", "command": "/tmp/hook.sh", "timeout": 30},
            ],
        })
        assert len(specs) == 1
        assert specs[0].event == "pre_tool_call"
        assert specs[0].matcher == "terminal"
        assert specs[0].command == "/tmp/hook.sh"
        assert specs[0].timeout == 30

    def test_unknown_event_skipped(self, caplog):
        specs = shell_hooks._parse_hooks_block({
            "pre_tools_call": [  # typo
                {"command": "/tmp/hook.sh"},
            ],
        })
        assert specs == []

    def test_missing_command_skipped(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [{"matcher": "terminal"}],
        })
        assert specs == []

    def test_timeout_clamped_to_max(self):
        specs = shell_hooks._parse_hooks_block({
            "post_tool_call": [
                {"command": "/tmp/slow.sh", "timeout": 9999},
            ],
        })
        assert specs[0].timeout == shell_hooks.MAX_TIMEOUT_SECONDS

    def test_non_int_timeout_defaulted(self):
        specs = shell_hooks._parse_hooks_block({
            "post_tool_call": [
                {"command": "/tmp/x.sh", "timeout": "thirty"},
            ],
        })
        assert specs[0].timeout == shell_hooks.DEFAULT_TIMEOUT_SECONDS

    def test_non_list_event_skipped(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": "not a list",
        })
        assert specs == []

    def test_none_hooks_block(self):
        assert shell_hooks._parse_hooks_block(None) == []
        assert shell_hooks._parse_hooks_block("string") == []
        assert shell_hooks._parse_hooks_block([]) == []

    def test_non_tool_event_matcher_warns_and_drops(self, caplog):
        """matcher: is only honored for pre/post_tool_call; must warn
        and drop on other events so the spec reflects runtime."""
        import logging
        cfg = {"pre_llm_call": [{"matcher": "terminal", "command": "/bin/echo"}]}
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            specs = shell_hooks._parse_hooks_block(cfg)
        assert len(specs) == 1 and specs[0].matcher is None
        assert any(
            "only honored for pre_tool_call" in r.getMessage()
            and "pre_llm_call" in r.getMessage()
            for r in caplog.records
        )


# ── Idempotent registration ───────────────────────────────────────────────


class TestIdempotentRegistration:
    def test_double_call_registers_once(self, tmp_path, monkeypatch):
        from hermes_cli import plugins

        script = _write_script(tmp_path, "h.sh",
                               "#!/usr/bin/env bash\nprintf '{}\\n'\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")

        plugins._plugin_manager = plugins.PluginManager()

        cfg = {"hooks": {"on_session_start": [{"command": str(script)}]}}

        first = shell_hooks.register_from_config(cfg, accept_hooks=True)
        second = shell_hooks.register_from_config(cfg, accept_hooks=True)
        assert len(first) == 1
        assert second == []
        # Only one callback on the manager
        mgr = plugins.get_plugin_manager()
        assert len(mgr._hooks.get("on_session_start", [])) == 1

    def test_same_command_different_matcher_registers_both(
        self, tmp_path, monkeypatch,
    ):
        """Same script used for different matchers under one event must
        register both callbacks — dedupe keys on (event, matcher, command)."""
        from hermes_cli import plugins

        script = _write_script(tmp_path, "h.sh",
                               "#!/usr/bin/env bash\nprintf '{}\\n'\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")

        plugins._plugin_manager = plugins.PluginManager()

        cfg = {
            "hooks": {
                "pre_tool_call": [
                    {"matcher": "terminal", "command": str(script)},
                    {"matcher": "web_search", "command": str(script)},
                ],
            },
        }

        registered = shell_hooks.register_from_config(cfg, accept_hooks=True)
        assert len(registered) == 2
        mgr = plugins.get_plugin_manager()
        assert len(mgr._hooks.get("pre_tool_call", [])) == 2


# ── Allowlist concurrency ─────────────────────────────────────────────────


class TestAllowlistConcurrency:
    """Regression tests for the Codex#1 finding: simultaneous
    _record_approval() calls used to collide on a fixed tmp path and
    silently lose entries under read-modify-write races."""

    def test_parallel_record_approval_does_not_lose_entries(
        self, tmp_path, monkeypatch,
    ):
        import threading

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        N = 32
        barrier = threading.Barrier(N)
        errors: list = []

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=5)
                shell_hooks._record_approval(
                    "on_session_start", f"/bin/hook-{i}.sh",
                )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"worker errors: {errors}"

        data = shell_hooks.load_allowlist()
        commands = {e["command"] for e in data["approvals"]}
        assert commands == {f"/bin/hook-{i}.sh" for i in range(N)}, (
            f"expected all {N} entries, got {len(commands)}"
        )

    def test_non_posix_fallback_does_not_self_deadlock(
        self, tmp_path, monkeypatch,
    ):
        """Regression: on platforms without fcntl, the fallback lock must
        be separate from _registered_lock.  register_from_config holds
        _registered_lock while calling _record_approval (via the consent
        prompt path), so a shared non-reentrant lock would self-deadlock."""
        import threading

        monkeypatch.setattr(shell_hooks, "fcntl", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        completed = threading.Event()
        errors: list = []

        def target() -> None:
            try:
                with shell_hooks._registered_lock:
                    shell_hooks._record_approval(
                        "on_session_start", "/bin/x.sh",
                    )
                completed.set()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
                completed.set()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        if not completed.wait(timeout=3.0):
            pytest.fail(
                "non-POSIX fallback self-deadlocked — "
                "_locked_update_approvals must not reuse _registered_lock",
            )
        t.join(timeout=1.0)
        assert not errors, f"errors: {errors}"
        assert shell_hooks._is_allowlisted(
            "on_session_start", "/bin/x.sh",
        )

    def test_save_allowlist_failure_logs_actionable_warning(
        self, tmp_path, monkeypatch, caplog,
    ):
        """Persistence failures must log the path, errno, and
        re-prompt consequence so "hermes keeps asking" is debuggable."""
        import logging
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            shell_hooks.tempfile, "mkstemp",
            lambda *a, **kw: (_ for _ in ()).throw(OSError(28, "No space")),
        )
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            shell_hooks.save_allowlist({"approvals": []})
        msg = next(
            (r.getMessage() for r in caplog.records
             if "Failed to persist" in r.getMessage()), "",
        )
        assert "shell-hooks-allowlist.json" in msg
        assert "No space" in msg
        assert "re-prompt" in msg

    def test_script_is_executable_handles_interpreter_prefix(self, tmp_path):
        """For ``python3 hook.py`` and similar the interpreter reads
        the script, so X_OK on the script itself is not required —
        only R_OK.  Bare invocations still require X_OK."""
        script = tmp_path / "hook.py"
        script.write_text("print()\n")  # readable, NOT executable

        # Interpreter prefix: R_OK is enough.
        assert shell_hooks.script_is_executable(f"python3 {script}")
        assert shell_hooks.script_is_executable(f"/usr/bin/env python3 {script}")

        # Bare invocation on the same non-X_OK file: not runnable.
        assert not shell_hooks.script_is_executable(str(script))

        # Flip +x; bare invocation is now runnable too.
        script.chmod(0o755)
        assert shell_hooks.script_is_executable(str(script))

    def test_command_script_path_resolution(self):
        """Regression: ``_command_script_path`` used to return the first
        shlex token, which picked the interpreter (``python3``, ``bash``,
        ``/usr/bin/env``) instead of the actual script for any
        interpreter-prefixed command.  That broke
        ``hermes hooks doctor``'s executability check and silently
        disabled mtime drift detection for such hooks."""
        cases = [
            # bare path
            ("/path/hook.sh", "/path/hook.sh"),
            ("/bin/echo hi", "/bin/echo"),
            ("~/hook.sh", "~/hook.sh"),
            ("hook.sh", "hook.sh"),
            # interpreter prefix
            ("python3 /path/hook.py", "/path/hook.py"),
            ("bash /path/hook.sh", "/path/hook.sh"),
            ("bash ~/hook.sh", "~/hook.sh"),
            ("python3 -u /path/hook.py", "/path/hook.py"),
            ("nice -n 10 /path/hook.sh", "/path/hook.sh"),
            # /usr/bin/env shebang form — must find the *script*, not env
            ("/usr/bin/env python3 /path/hook.py", "/path/hook.py"),
            ("/usr/bin/env bash /path/hook.sh", "/path/hook.sh"),
            # no path-like tokens → fallback to first token
            ("my-binary --verbose", "my-binary"),
            ("python3 -c 'print(1)'", "python3"),
            # unparseable (unbalanced quotes) → return command as-is
            ("python3 'unterminated", "python3 'unterminated"),
            # empty
            ("", ""),
        ]
        for command, expected in cases:
            got = shell_hooks._command_script_path(command)
            assert got == expected, f"{command!r} -> {got!r}, expected {expected!r}"

    def test_save_allowlist_uses_unique_tmp_paths(self, tmp_path, monkeypatch):
        """Two save_allowlist calls in flight must use distinct tmp files
        so the loser's os.replace does not ENOENT on the winner's sweep."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        p = shell_hooks.allowlist_path()
        p.parent.mkdir(parents=True, exist_ok=True)

        tmp_paths_seen: list = []
        real_mkstemp = shell_hooks.tempfile.mkstemp

        def spying_mkstemp(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            tmp_paths_seen.append(path)
            return fd, path

        monkeypatch.setattr(shell_hooks.tempfile, "mkstemp", spying_mkstemp)

        shell_hooks.save_allowlist({"approvals": [{"event": "a", "command": "x"}]})
        shell_hooks.save_allowlist({"approvals": [{"event": "b", "command": "y"}]})

        assert len(tmp_paths_seen) == 2
        assert tmp_paths_seen[0] != tmp_paths_seen[1]
