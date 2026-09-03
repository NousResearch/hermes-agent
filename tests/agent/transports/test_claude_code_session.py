"""Tests for ClaudeCodeSession — drive turns through a fake ``claude`` binary.

The fake (``fake_claude_cli.py``) is a real subprocess speaking stream-json
over real pipes, so these tests cover the parts that a mocked client cannot:
the warm-up-before-init handshake, reader-thread EOF/identity handling,
post-result draining, interrupt via the control channel, and retirement on
an unexpected exit.
"""

from __future__ import annotations

import json
import os
import queue
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.transports import claude_code_session as session_mod
from agent.transports.claude_code_session import (
    DEFAULT_DENY_RULES,
    SETTINGS_MARKER_KEY,
    ClaudeCodeSession,
    build_child_env,
    ensure_settings_file,
    hermes_tool_name,
    resolve_oauth_token,
    resolve_permission,
    resume_transcript_exists,
    write_mcp_config,
)

_FAKE = Path(__file__).with_name("fake_claude_cli.py")


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path: Path, monkeypatch):
    """Every session in these tests gets a scratch HERMES_HOME (so the
    Hermes-owned CLAUDE_CONFIG_DIR lands under tmp) and a fake setup-token."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-leak")
    return home


@pytest.fixture
def fake_claude(tmp_path: Path):
    """Return (claude_bin, record_path): a wrapper script + where the fake
    dumps its argv/env."""
    record = tmp_path / "record.json"
    wrapper = tmp_path / "claude"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"export FAKE_CLAUDE_RECORD={json.dumps(str(record))}\n"
        f"exec {json.dumps(sys.executable)} {json.dumps(str(_FAKE))} \"$@\"\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return str(wrapper), record


def _session(fake_claude, **kwargs) -> ClaudeCodeSession:
    claude_bin, _ = fake_claude
    events: list[dict] = []
    kwargs.setdefault("on_event", events.append)
    kwargs.setdefault("expose_hermes_tools", False)
    kwargs.setdefault("startup_timeout", 20.0)
    session = ClaudeCodeSession(claude_bin=claude_bin, **kwargs)
    session._test_events = events  # type: ignore[attr-defined]
    return session


class TestStaticHelpers:
    def test_child_env_strips_api_keys_and_guarantees_identity(self):
        env = build_child_env(
            {"ANTHROPIC_API_KEY": "sk-ant-should-not-leak", "ANTHROPIC_AUTH_TOKEN": "x", "PATH": "/usr/bin"},
            config_dir="/cfg",
        )
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "fake-setup-token"
        assert env["CLAUDE_CONFIG_DIR"] == "/cfg"
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
        assert env["USER"] and env["LOGNAME"] == env["USER"]
        assert env["HOME"]

    @pytest.mark.parametrize(
        "mode, expected_mode, must_allow",
        [
            ("auto", "acceptEdits", "Bash"),
            ("approval-required", "default", "Read"),
            ("unrestricted", "bypassPermissions", None),
            ("yolo", "bypassPermissions", None),
            (None, "acceptEdits", "Bash"),
        ],
    )
    def test_permission_mapping(self, mode, expected_mode, must_allow):
        got_mode, allowed = resolve_permission(mode)
        assert got_mode == expected_mode
        if must_allow:
            assert must_allow in allowed

    def test_unknown_security_mode_fails_closed(self):
        got_mode, allowed = resolve_permission("garbage")
        assert got_mode == "default"  # approval-required mapping, never auto/bypass
        assert "Bash" not in allowed and "Write" not in allowed

    def test_oauth_token_is_required(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="claude setup-token"):
            resolve_oauth_token()
        monkeypatch.setenv("MY_TOKEN", "abc")
        assert resolve_oauth_token("MY_TOKEN") == "abc"

    def test_settings_file_written_once_with_default_deny_list(self, tmp_path):
        path = str(tmp_path / "cc" / "settings.json")
        ensure_settings_file(path)
        data = json.loads(Path(path).read_text())
        assert data["permissions"]["deny"] == list(DEFAULT_DENY_RULES)
        assert "Bash(git push *)" in data["permissions"]["deny"]
        assert SETTINGS_MARKER_KEY in data
        # User edits survive: a second call with different rules is a no-op.
        Path(path).write_text('{"permissions": {"deny": ["Bash(custom *)"]}}')
        ensure_settings_file(path, ["Bash(other *)"])
        assert json.loads(Path(path).read_text())["permissions"]["deny"] == ["Bash(custom *)"]

    def test_resume_transcript_lookup(self, tmp_path):
        cfg = tmp_path / "cfg"
        sid = "11111111-2222-4333-8444-555555555555"
        assert resume_transcript_exists(str(cfg), sid) is False
        (cfg / "projects" / "-some-cwd").mkdir(parents=True)
        (cfg / "projects" / "-some-cwd" / f"{sid}.jsonl").write_text("{}\n")
        assert resume_transcript_exists(str(cfg), sid) is True

    def test_approval_required_never_preapproves_bash(self):
        _, allowed = resolve_permission("approval-required")
        assert "Bash" not in allowed and "Write" not in allowed

    def test_hermes_tool_name_strips_mcp_prefix(self):
        assert hermes_tool_name("mcp__hermes-tools__web_search") == "web_search"
        assert hermes_tool_name("Bash") == "Bash"

    def test_mcp_config_launches_hermes_tools_server(self, tmp_path):
        path = write_mcp_config(directory=str(tmp_path))
        payload = json.loads(Path(path).read_text())
        server = payload["mcpServers"]["hermes-tools"]
        assert server["command"] == sys.executable
        assert server["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
        # Secrets are inherited from the process env, never written to disk.
        assert not any(k.endswith("_API_KEY") for k in server["env"])


class TestLifecycle:
    def test_warmup_is_sent_before_init_and_surfaces_session_id(self, fake_claude):
        session = _session(fake_claude, system_prompt="SYS-PROMPT", model="sonnet")
        try:
            sid = session.ensure_started()
            assert sid == "sess-fake-0001"
            assert session.init_info["mcp_servers"][0]["status"] == "connected"
            # The fake emits init only after it has read a user line, so the
            # only way init_info is populated is warm-up -> init -> result.
            _, record = fake_claude
            rec = json.loads(record.read_text())
            argv = rec["argv"]
            assert argv[:2] == ["-p", "--verbose"]
            assert "--input-format" in argv and "stream-json" in argv
            assert "--include-partial-messages" in argv
            prompt_file = argv[argv.index("--append-system-prompt-file") + 1]
            assert Path(prompt_file).read_text() == "SYS-PROMPT"
            assert argv[argv.index("--model") + 1] == "sonnet"
            assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
            assert "ANTHROPIC_API_KEY" not in rec["env"]
            assert rec["env"]["USER"]
            # Warm-up is silent: nothing reached the UI hook except init.
            kinds = {e["kind"] for e in session._test_events}
            assert kinds == {"init"}
            assert session.ensure_started() == sid  # idempotent
        finally:
            session.close()

    def test_child_is_hermes_owned_claude_code(self, fake_claude, _isolated_hermes_home):
        """HIGH-2: exact argv + env of the isolation contract."""
        home = _isolated_hermes_home
        session = _session(fake_claude, system_prompt="SYS", expose_hermes_tools=True)
        try:
            session.ensure_started()
            _, record = fake_claude
            rec = json.loads(record.read_text())
            argv, env = rec["argv"], rec["env"]
            cfg_dir = str(home / "claude-code")
            assert env["CLAUDE_CONFIG_DIR"] == cfg_dir
            assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
            assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "fake-setup-token"
            assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
            assert env["HOME"] and env["USER"] and env["LOGNAME"]
            assert rec["cwd"] == str(home / "claude-code" / "workspace")
            assert argv[argv.index("--setting-sources") + 1] == ""
            assert argv[argv.index("--settings") + 1] == str(home / "claude-code" / "settings.json")
            assert "--strict-mcp-config" in argv
            assert argv[argv.index("--session-id") + 1] == session.requested_session_id
            assert "--resume" not in argv
            # Prompt travels through a file inside the Hermes-owned dir, not argv.
            assert "--append-system-prompt" not in argv
            prompt_file = argv[argv.index("--append-system-prompt-file") + 1]
            assert prompt_file.startswith(cfg_dir)
            assert Path(prompt_file).read_text() == "SYS"
            # The deny list was materialised.
            settings = json.loads((home / "claude-code" / "settings.json").read_text())
            assert "Bash(git push *)" in settings["permissions"]["deny"]
        finally:
            session.close()
        assert not Path(prompt_file).exists()

    def test_resumes_existing_transcript(self, fake_claude, _isolated_hermes_home):
        sid = "11111111-2222-4333-8444-555555555555"
        cfg = _isolated_hermes_home / "claude-code" / "projects" / "-ws"
        cfg.mkdir(parents=True)
        (cfg / f"{sid}.jsonl").write_text("{}\n")
        session = _session(fake_claude, session_id=sid)
        try:
            session.ensure_started()
            _, record = fake_claude
            argv = json.loads(record.read_text())["argv"]
            assert argv[argv.index("--resume") + 1] == sid
            assert "--session-id" not in argv
            assert session.resumed is True
        finally:
            session.close()

    def test_approval_required_without_approver_warns_once(self, fake_claude):
        session = _session(fake_claude, security_mode="approval-required")
        try:
            session.ensure_started()
            statuses = [e["text"] for e in session._test_events if e["kind"] == "status"]
            assert statuses and "no interactive approver" in statuses[0]
            _, record = fake_claude
            argv = json.loads(record.read_text())["argv"]
            assert argv[argv.index("--permission-mode") + 1] == "default"
            assert argv[argv.index("--permission-prompt-tool") + 1] == "stdio"
            assert "Bash" not in argv[argv.index("--allowedTools") + 1].split(",")
        finally:
            session.close()

    def test_resume_rejected_rotates_to_fresh_id_and_persists(self, fake_claude, _isolated_hermes_home, monkeypatch):
        sid = "11111111-2222-4333-8444-555555555555"
        cfg = _isolated_hermes_home / "claude-code"
        (cfg / "projects" / "-ws").mkdir(parents=True)
        (cfg / "projects" / "-ws" / f"{sid}.jsonl").write_text("{}\n")
        monkeypatch.setenv("FAKE_CLAUDE_REJECT_RESUME", "1")
        # The real CLI refuses --session-id for an id whose transcript exists.
        monkeypatch.setenv("FAKE_CLAUDE_USED_IDS", sid)
        session = _session(fake_claude, session_id=sid, session_key="hermes-sess-1")
        try:
            session.ensure_started()
            new_id = session.requested_session_id
            assert new_id != sid and session.resumed is False
            result = session.run_turn("after fallback")
            assert result.error is None and result.final_text == "echo: after fallback"
            mapping = json.loads((cfg / "hermes-sessions.json").read_text())
            assert mapping == {"hermes-sess-1": new_id}
        finally:
            session.close()
        # A later session for the same Hermes session picks up the rotated id.
        monkeypatch.delenv("FAKE_CLAUDE_REJECT_RESUME")
        again = _session(fake_claude, session_id=sid, session_key="hermes-sess-1")
        try:
            assert again.requested_session_id == new_id
            again.ensure_started()
            assert again.run_turn("hello").final_text == "echo: hello"
        finally:
            again.close()

    def test_mcp_config_flag_and_allowlist(self, fake_claude, tmp_path):
        session = _session(fake_claude, expose_hermes_tools=True)
        try:
            session.ensure_started()
            _, record = fake_claude
            argv = json.loads(record.read_text())["argv"]
            mcp_path = argv[argv.index("--mcp-config") + 1]
            assert Path(mcp_path).exists()
            allowed = argv[argv.index("--allowedTools") + 1].split(",")
            assert "mcp__hermes-tools" in allowed and "Bash" in allowed
        finally:
            session.close()
        assert not Path(mcp_path).exists()  # temp file cleaned on close

    def test_bypass_permissions_only_when_unrestricted(self, fake_claude):
        session = _session(fake_claude, security_mode="unrestricted")
        try:
            session.ensure_started()
            _, record = fake_claude
            argv = json.loads(record.read_text())["argv"]
            assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
            assert "--allowedTools" not in argv
        finally:
            session.close()

    def test_close_idempotent_and_kills_process(self, fake_claude):
        session = _session(fake_claude)
        session.ensure_started()
        assert session.is_alive()
        session.close()
        session.close()
        assert not session.is_alive()


class TestRunTurn:
    def test_text_turn_projects_message_and_streams_deltas(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            result = session.run_turn("hello there")
            assert result.error is None
            assert result.final_text == "echo: hello there"
            assert result.projected_messages == [
                {"role": "assistant", "content": "echo: hello there"}
            ]
            assert result.tool_iterations == 0
            assert result.session_id == "sess-fake-0001"
            assert result.token_usage_last["cache_read_input_tokens"] == 100
            assert result.model_context_window == 200000
            deltas = "".join(
                e["text"] for e in session._test_events if e["kind"] == "text_delta"
            )
            assert deltas == "echo: hello there"
        finally:
            session.close()

    def test_tool_use_is_projected_as_tool_calls_and_tool_result(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            result = session.run_turn("please TOOL")
            assert result.error is None
            assert result.tool_iterations == 1
            assert result.final_text == "The version is 6.2"
            roles = [m["role"] for m in result.projected_messages]
            assert roles == ["assistant", "tool", "assistant"]
            call = result.projected_messages[0]["tool_calls"][0]
            assert call["id"] == "toolu_fake_1"
            assert call["function"]["name"] == "web_search"  # MCP prefix stripped
            assert json.loads(call["function"]["arguments"]) == {"query": "swift version"}
            tool_msg = result.projected_messages[1]
            assert tool_msg["tool_call_id"] == "toolu_fake_1"
            assert tool_msg["content"] == "Swift 6.2"
            kinds = [e["kind"] for e in session._test_events]
            assert "tool_started" in kinds and "tool_completed" in kinds
            started = next(e for e in session._test_events if e["kind"] == "tool_started")
            assert started["wire_name"] == "mcp__hermes-tools__web_search"
            assert started["name"] == "web_search"
            interim = [e for e in session._test_events if e["kind"] == "assistant_message"]
            assert interim and interim[0]["text"] == "Let me search."
        finally:
            session.close()

    def test_stdout_is_drained_before_turn_is_declared_finished(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            result = session.run_turn("say TRAILER")
            assert result.error is None
            # The fake flushed a rate_limit_event right behind `result`; it
            # must be consumed now, not surface at the start of the next turn.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not session._lines.empty():
                time.sleep(0.01)
            assert session._lines.empty()
            follow_up = session.run_turn("second")
            assert follow_up.final_text == "echo: second"
        finally:
            session.close()

    def test_same_process_serves_consecutive_turns(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            pid = session.pid
            session.run_turn("one")
            session.run_turn("two")
            assert session.pid == pid and session.is_alive()
        finally:
            session.close()


class TestRetirement:
    def test_nonzero_exit_mid_turn_retires_session(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            result = session.run_turn("please CRASH")
            assert result.should_retire is True
            assert result.error and "exited unexpectedly (code 3)" in result.error
            assert not session.is_alive()
            # A subsequent turn on the dead process fails fast, no hang.
            again = session.run_turn("hello")
            assert again.should_retire is True and again.error
        finally:
            session.close()

    def test_stale_reader_exit_does_not_poison_replacement(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            old_pid = session.pid
            old_reader = session._reader_thread
            assert old_reader is not None
            session.restart(system_prompt="NEW-PROMPT")
            assert session.pid != old_pid
            # Let the retired process' reader hit EOF and run its finally.
            old_reader.join(timeout=5)
            assert not old_reader.is_alive()
            # Identity check: the stale EOF must not have been enqueued for
            # the healthy replacement.
            assert not any(
                item is session_mod._EOF for item in list(session._lines.queue)
            )
            result = session.run_turn("after restart")
            assert result.error is None and result.final_text == "echo: after restart"
            assert session.needs_respawn("NEW-PROMPT") is False
        finally:
            session.close()

    def test_interrupt_unblocks_turn(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            box: queue.Queue = queue.Queue()

            def _run():
                box.put(session.run_turn("HANG forever", turn_timeout=30, idle_timeout=30))

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            time.sleep(0.3)
            started = time.monotonic()
            session.request_interrupt()
            result = box.get(timeout=10)
            assert result.interrupted is True
            assert time.monotonic() - started < 8
            # The CLI answered the interrupt itself, so the process is reusable.
            assert session.is_alive()
            assert session.run_turn("still here").final_text == "echo: still here"
        finally:
            session.close()

    def test_idle_timeout_retires(self, fake_claude):
        session = _session(fake_claude)
        try:
            session.ensure_started()
            result = session.run_turn("HANG", turn_timeout=5, idle_timeout=0.3)
            assert result.should_retire is True
            assert "no output" in (result.error or "")
        finally:
            session.close()


class TestPermissionPrompt:
    """``--permission-prompt-tool stdio``: can_use_tool routed to Hermes approvals."""

    def _run(self, fake_claude, callback):
        session = _session(
            fake_claude, security_mode="approval-required", approval_callback=callback
        )
        try:
            session.ensure_started()
            return session.run_turn("please PERM"), session
        finally:
            session.close()

    def test_approve_runs_the_tool(self, fake_claude):
        seen = []

        def approve(command, description, **kw):
            seen.append((command, description, kw))
            return "once"

        result, session = self._run(fake_claude, approve)
        assert result.error is None
        assert seen and seen[0][0] == "touch marker"
        assert "Claude Code requests Bash" in seen[0][1]
        assert seen[0][2] == {"allow_permanent": False}
        tool_msg = next(m for m in result.projected_messages if m["role"] == "tool")
        assert tool_msg["content"] == "PERM-RAN"
        assert result.final_text == "ran it"
        # No "no approver" notice when a callback is wired.
        assert not [e for e in session._test_events if e["kind"] == "status"]

    def test_deny_yields_error_tool_result(self, fake_claude):
        result, _ = self._run(fake_claude, lambda *a, **k: "deny")
        assert result.error is None  # the turn itself completes
        tool_msg = next(m for m in result.projected_messages if m["role"] == "tool")
        assert tool_msg["content"].startswith("[error]")
        assert "Denied by the Hermes user" in tool_msg["content"]
        assert result.final_text == "it was denied"

    def test_no_approver_denies(self, fake_claude):
        result, session = self._run(fake_claude, None)
        tool_msg = next(m for m in result.projected_messages if m["role"] == "tool")
        assert "no interactive approver" in tool_msg["content"]
        assert result.final_text == "it was denied"
        assert [e for e in session._test_events if e["kind"] == "status"]

    def test_timeout_denies(self, fake_claude):
        result, _ = self._run(fake_claude, lambda *a, **k: "timeout")
        tool_msg = next(m for m in result.projected_messages if m["role"] == "tool")
        assert "timed out" in tool_msg["content"]

    def test_callback_exception_denies(self, fake_claude):
        def boom(*a, **k):
            raise RuntimeError("ui gone")

        result, _ = self._run(fake_claude, boom)
        tool_msg = next(m for m in result.projected_messages if m["role"] == "tool")
        assert "approval prompt failed" in tool_msg["content"]


class TestSessionIdRotation:
    SID = "11111111-2222-4333-8444-555555555555"

    def _transcript(self, home):
        cfg = home / "claude-code"
        (cfg / "projects" / "-ws").mkdir(parents=True, exist_ok=True)
        (cfg / "projects" / "-ws" / f"{self.SID}.jsonl").write_text("{}\n")
        return cfg

    def test_resume_disabled_with_existing_transcript_rotates_before_spawn(
        self, fake_claude, _isolated_hermes_home, monkeypatch
    ):
        cfg = self._transcript(_isolated_hermes_home)
        monkeypatch.setenv("FAKE_CLAUDE_USED_IDS", self.SID)  # CLI: "already in use"
        session = _session(fake_claude, session_id=self.SID, session_key="k1", resume=False)
        try:
            session.ensure_started()
            assert session.requested_session_id != self.SID
            _, record = fake_claude
            argv = json.loads(record.read_text())["argv"]
            assert "--resume" not in argv
            assert argv[argv.index("--session-id") + 1] == session.requested_session_id
            assert session.run_turn("x").final_text == "echo: x"
            assert json.loads((cfg / "hermes-sessions.json").read_text())["k1"] == session.requested_session_id
        finally:
            session.close()

    def test_non_rejection_warmup_error_keeps_id(self, fake_claude, _isolated_hermes_home, monkeypatch):
        cfg = self._transcript(_isolated_hermes_home)
        monkeypatch.setenv("FAKE_CLAUDE_FAIL_WARMUP", "auth")
        session = _session(fake_claude, session_id=self.SID, session_key="k1")
        with pytest.raises(RuntimeError, match="Failed to authenticate"):
            session.ensure_started()
        assert session.requested_session_id == self.SID
        assert not (cfg / "hermes-sessions.json").exists()

    def test_rejection_marker_detection(self):
        from agent.transports.claude_code_session import is_session_id_rejection

        assert is_session_id_rejection("Error: No conversation found with session ID abc")
        assert is_session_id_rejection("Session ID abc is already in use.")
        assert not is_session_id_rejection("Failed to authenticate: OAuth session expired")
        assert not is_session_id_rejection(None)

    def test_session_map_concurrent_writes_keep_all_entries(self, tmp_path):
        from agent.transports.claude_code_session import load_session_map, save_session_mapping

        cfg = str(tmp_path / "cfg")
        workers = [
            threading.Thread(target=save_session_mapping, args=(cfg, f"key-{i}", f"id-{i}"))
            for i in range(16)
        ]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        assert load_session_map(cfg) == {f"key-{i}": f"id-{i}" for i in range(16)}

    def test_resume_is_restored_after_a_rotation(self, fake_claude, _isolated_hermes_home, monkeypatch):
        """rejection -> rotate -> prompt change -> restart resumes the NEW
        transcript instead of rotating a second time."""
        cfg = self._transcript(_isolated_hermes_home)
        monkeypatch.setenv("FAKE_CLAUDE_REJECT_RESUME_ID", self.SID)
        monkeypatch.setenv("FAKE_CLAUDE_USED_IDS", self.SID)
        session = _session(fake_claude, session_id=self.SID, session_key="k1", system_prompt="P1")
        try:
            session.ensure_started()
            new_id = session.requested_session_id
            assert new_id != self.SID
            # The real CLI would have written this transcript during the fresh spawn.
            (cfg / "projects" / "-ws" / f"{new_id}.jsonl").write_text("{}\n")
            session.restart(system_prompt="P2")
            assert session.requested_session_id == new_id  # no second rotation
            assert session.resumed is True
            _, record = fake_claude
            argv = json.loads(record.read_text())["argv"]
            assert argv[argv.index("--resume") + 1] == new_id
            assert session.run_turn("x").final_text == "echo: x"
        finally:
            session.close()
