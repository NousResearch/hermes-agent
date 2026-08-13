"""Tests for the shell-hooks subprocess bridge (agent.shell_hooks).

These tests focus on the pure translation layer — JSON serialisation,
JSON parsing, matcher behaviour, block-schema correctness, and the
subprocess runner's graceful error handling.  Consent prompts are
covered in ``test_shell_hooks_consent.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

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



    def test_empty_stdout_returns_none(self):
        assert shell_hooks._parse_response("pre_tool_call", "") is None
        assert shell_hooks._parse_response("pre_tool_call", "   ") is None















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




# ── Matcher behaviour ─────────────────────────────────────────────────────


class TestMatcher:


    def test_alternation_matcher(self):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher="terminal|file",
        )
        assert spec.matches_tool("terminal")
        assert spec.matches_tool("file")
        assert not spec.matches_tool("web")



    def test_matcher_leading_whitespace_stripped(self):
        """YAML quirks can introduce leading/trailing whitespace — must
        not silently break the matcher."""
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command="echo", matcher=" terminal ",
        )
        assert spec.matcher == "terminal"
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



    def test_python_only_event_refused(self, caplog):
        # transform_api_error_classification returns a classification directive that
        # _parse_response has no channel for — a shell registration would
        # be silently ignored, so it must be refused with a warning.
        specs = shell_hooks._parse_hooks_block({
            "transform_api_error_classification": [
                {"command": "/tmp/hook.sh"},
            ],
        })
        assert specs == []
        assert any("Python-plugin-only" in r.message for r in caplog.records)

    def test_timeout_clamped_to_max(self):
        specs = shell_hooks._parse_hooks_block({
            "post_tool_call": [
                {"command": "/tmp/slow.sh", "timeout": 9999},
            ],
        })
        assert specs[0].timeout == shell_hooks.MAX_TIMEOUT_SECONDS



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


# ── fail_closed parsing ───────────────────────────────────────────────────


class TestFailClosedParsing:
    def test_fail_closed_parsed(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {"command": "/tmp/h.sh", "fail_closed": True},
            ],
        })
        assert len(specs) == 1
        assert specs[0].fail_closed is True

    def test_fail_closed_defaults_false(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [{"command": "/tmp/h.sh"}],
        })
        assert specs[0].fail_closed is False

    def test_failclosed_camel_alias(self):
        """Cursor/Claude Code configs spell it failClosed."""
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {"command": "/tmp/h.sh", "failClosed": True},
            ],
        })
        assert specs[0].fail_closed is True

    def test_canonical_wins_over_alias(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {"command": "/tmp/h.sh", "fail_closed": False,
                 "failClosed": True},
            ],
        })
        assert specs[0].fail_closed is False

    def test_non_bool_warns_and_defaults_false(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            specs = shell_hooks._parse_hooks_block({
                "pre_tool_call": [
                    {"command": "/tmp/h.sh", "fail_closed": "yes"},
                ],
            })
        assert specs[0].fail_closed is False
        assert any(
            "fail_closed must be a boolean" in r.getMessage()
            for r in caplog.records
        )

    def test_fail_closed_on_non_blocking_event_warns_and_ignores(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            specs = shell_hooks._parse_hooks_block({
                "on_session_start": [
                    {"command": "/tmp/h.sh", "fail_closed": True},
                ],
            })
        assert specs[0].fail_closed is False
        assert any(
            "fail_closed" in r.getMessage() and "ignored" in r.getMessage()
            for r in caplog.records
        )


# ── _evaluate_result semantics ────────────────────────────────────────────


def _spawn_result(**overrides):
    base = {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "elapsed_seconds": 0.1,
        "error": None,
    }
    base.update(overrides)
    return base


class TestEvaluateResult:
    def _spec(self, event="pre_tool_call", fail_closed=False):
        return shell_hooks.ShellHookSpec(
            event=event, command="/tmp/h.sh", fail_closed=fail_closed,
        )

    # -- exit code 2 = block --------------------------------------------

    def test_exit_2_blocks_with_stderr_message(self):
        r = shell_hooks._evaluate_result(
            self._spec(),
            _spawn_result(returncode=2, stderr="policy violation\n"),
        )
        assert r == {"action": "block", "message": "policy violation"}

    def test_exit_2_blocks_with_default_message(self):
        r = shell_hooks._evaluate_result(
            self._spec(), _spawn_result(returncode=2),
        )
        assert r == {
            "action": "block",
            "message": shell_hooks._DEFAULT_BLOCK_MESSAGE,
        }

    def test_exit_2_stdout_block_json_wins(self):
        r = shell_hooks._evaluate_result(
            self._spec(),
            _spawn_result(
                returncode=2,
                stdout='{"decision": "block", "reason": "from stdout"}',
                stderr="from stderr",
            ),
        )
        assert r == {"action": "block", "message": "from stdout"}

    def test_exit_2_on_non_blocking_event_does_not_block(self):
        r = shell_hooks._evaluate_result(
            self._spec(event="on_session_start"),
            _spawn_result(returncode=2, stderr="boom"),
        )
        assert r is None

    def test_other_nonzero_exit_still_parses_stdout(self):
        r = shell_hooks._evaluate_result(
            self._spec(),
            _spawn_result(
                returncode=1,
                stdout='{"decision": "block", "reason": "nope"}',
            ),
        )
        assert r == {"action": "block", "message": "nope"}

    def test_nonzero_exit_without_directive_is_none(self):
        r = shell_hooks._evaluate_result(
            self._spec(), _spawn_result(returncode=1, stderr="oops"),
        )
        assert r is None

    # -- fail_closed ------------------------------------------------------

    def test_spawn_error_fails_open_by_default(self):
        r = shell_hooks._evaluate_result(
            self._spec(), _spawn_result(error="No such file"),
        )
        assert r is None

    def test_spawn_error_fail_closed_blocks(self):
        r = shell_hooks._evaluate_result(
            self._spec(fail_closed=True), _spawn_result(error="No such file"),
        )
        assert r["action"] == "block"
        assert "failed closed" in r["message"]
        assert "No such file" in r["message"]

    def test_timeout_fails_open_by_default(self):
        r = shell_hooks._evaluate_result(
            self._spec(), _spawn_result(timed_out=True),
        )
        assert r is None

    def test_timeout_fail_closed_blocks(self):
        spec = self._spec(fail_closed=True)
        r = shell_hooks._evaluate_result(spec, _spawn_result(timed_out=True))
        assert r["action"] == "block"
        assert f"timed out after {spec.timeout}s" in r["message"]

    def test_unparseable_stdout_fail_closed_blocks(self):
        r = shell_hooks._evaluate_result(
            self._spec(fail_closed=True),
            _spawn_result(stdout="Traceback (most recent call last): ..."),
        )
        assert r["action"] == "block"
        assert "unparseable stdout" in r["message"]

    def test_unparseable_stdout_fails_open_by_default(self):
        r = shell_hooks._evaluate_result(
            self._spec(),
            _spawn_result(stdout="Traceback (most recent call last): ..."),
        )
        assert r is None

    def test_valid_noop_json_passes_fail_closed(self):
        """A clean {} no-op must NOT be blocked by fail_closed."""
        r = shell_hooks._evaluate_result(
            self._spec(fail_closed=True), _spawn_result(stdout="{}"),
        )
        assert r is None

    def test_empty_stdout_passes_fail_closed(self):
        r = shell_hooks._evaluate_result(
            self._spec(fail_closed=True), _spawn_result(stdout=""),
        )
        assert r is None

    def test_fail_closed_on_non_blocking_event_still_fails_open(self):
        """Defense in depth: even if a spec sneaks past parsing with
        fail_closed on a non-blocking event, runtime fails open."""
        r = shell_hooks._evaluate_result(
            self._spec(event="on_session_start", fail_closed=True),
            _spawn_result(error="boom"),
        )
        assert r is None


# ── exit-2 / fail_closed end-to-end ──────────────────────────────────────


class TestFailSemanticsEndToEnd:
    def test_exit_2_script_blocks(self, tmp_path):
        script = _write_script(
            tmp_path, "exit2.sh",
            "#!/usr/bin/env bash\n"
            'echo "rm -rf is not permitted" >&2\n'
            "exit 2\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
        )
        cb = shell_hooks._make_callback(spec)
        result = cb(tool_name="terminal", args={"command": "rm -rf /"})
        assert result == {
            "action": "block", "message": "rm -rf is not permitted",
        }

    def test_fail_closed_missing_command_blocks(self, tmp_path):
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call",
            command=str(tmp_path / "does-not-exist.sh"),
            fail_closed=True,
        )
        cb = shell_hooks._make_callback(spec)
        result = cb(tool_name="terminal", args={"command": "ls"})
        assert result is not None and result["action"] == "block"
        assert "failed closed" in result["message"]

    def test_run_once_reflects_exit_2_block(self, tmp_path):
        """hermes hooks test must mirror production semantics."""
        script = _write_script(
            tmp_path, "exit2.sh",
            "#!/usr/bin/env bash\n"
            'echo "denied" >&2\n'
            "exit 2\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
        )
        result = shell_hooks.run_once(
            spec, {"tool_name": "terminal", "args": {"command": "ls"}},
        )
        assert result["returncode"] == 2
        assert result["parsed"] == {"action": "block", "message": "denied"}

    def test_run_once_reflects_fail_closed_timeout(self, tmp_path):
        script = _write_script(
            tmp_path, "sleepy.sh",
            "#!/usr/bin/env bash\nsleep 5\n",
        )
        spec = shell_hooks.ShellHookSpec(
            event="pre_tool_call", command=str(script),
            timeout=1, fail_closed=True,
        )
        result = shell_hooks.run_once(
            spec, {"tool_name": "terminal", "args": {"command": "ls"}},
        )
        assert result["timed_out"] is True
        assert result["parsed"]["action"] == "block"
        assert "failed closed" in result["parsed"]["message"]
