from __future__ import annotations

import queue
from pathlib import Path

import pytest

from cli import HermesCLI


def make_cli() -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    cli._agent_running = False
    cli._pending_handoff_new = None
    cli._pending_input = queue.Queue()
    cli._last_chat_result = None
    cli._last_turn_failed = False
    cli._last_turn_partial = False
    cli._last_turn_interrupted = False
    cli._spinner_text = ""
    cli._tool_start_time = 0.0
    cli._pending_tool_info = {}
    cli._last_scrollback_tool = ""
    return cli


def drain(q: queue.Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class TestHandoffNewRegistry:
    def test_handoff_registry_has_new_subcommand_and_cli_only(self):
        from hermes_cli.commands import resolve_command, GATEWAY_KNOWN_COMMANDS

        cmd = resolve_command("handoff")
        assert cmd is not None
        assert cmd.cli_only is True
        assert cmd.subcommands == ("new",)
        assert "new" in cmd.args_hint
        assert "handoff" not in GATEWAY_KNOWN_COMMANDS


class TestHandoffDispatcher:
    def test_legacy_platform_uses_gateway_handler(self, monkeypatch):
        cli = make_cli()
        called = []
        monkeypatch.setattr(cli, "_handle_gateway_handoff_command", lambda cmd: called.append(("gateway", cmd)) or True)
        monkeypatch.setattr(cli, "_handle_handoff_new_command", lambda cmd: called.append(("new", cmd)) or True)

        assert cli._handle_handoff_command("/handoff discord") is True

        assert called == [("gateway", "/handoff discord")]

    def test_new_subcommand_uses_new_handler(self, monkeypatch):
        cli = make_cli()
        called = []
        monkeypatch.setattr(cli, "_handle_gateway_handoff_command", lambda cmd: called.append(("gateway", cmd)) or True)
        monkeypatch.setattr(cli, "_handle_handoff_new_command", lambda cmd: called.append(("new", cmd)) or True)

        assert cli._handle_handoff_command("/handoff NeW") is True

        assert called == [("new", "/handoff NeW")]


class TestHandoffNewParser:
    def test_default_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        cli = make_cli()

        parsed = cli._parse_handoff_new_command("/handoff new")

        assert parsed["path"] == tmp_path / ".hermes" / "handoffs" / "latest.md"
        assert parsed["title"] is None
        assert parsed["continuation"] == f"Continue from {parsed['path']}"

    def test_uppercase_command_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        cli = make_cli()

        parsed = cli._parse_handoff_new_command("/HANDOFF new")

        assert parsed["path"] == tmp_path / ".hermes" / "handoffs" / "latest.md"
        assert parsed["title"] is None
        assert parsed["continuation"] == f"Continue from {parsed['path']}"

    def test_path_title_after_and_custom_prompt(self, tmp_path):
        cli = make_cli()
        target = tmp_path / "handoff.md"

        parsed = cli._parse_handoff_new_command(
            f'/handoff new {target} --title "Fresh start" --prompt "Resume here"'
        )

        assert parsed["path"] == target
        assert parsed["title"] == "Fresh start"
        assert parsed["continuation"] == "Resume here"

    def test_title_before_path(self, tmp_path):
        cli = make_cli()
        target = tmp_path / "handoff.md"

        parsed = cli._parse_handoff_new_command(f'/handoff new --title "T" {target}')

        assert parsed["path"] == target
        assert parsed["title"] == "T"

    def test_quoted_path(self, tmp_path):
        cli = make_cli()
        target = tmp_path / "path with spaces.md"

        parsed = cli._parse_handoff_new_command(f'/handoff new "{target}"')

        assert parsed["path"] == target

    @pytest.mark.parametrize(
        "cmd, match",
        [
            ("/handoff new --bogus", "Unknown flag"),
            ("/handoff new one.md two.md", "Only one"),
            ("/handoff new --title", "Missing value"),
            ("/handoff new --prompt ''", "cannot be empty"),
        ],
    )
    def test_rejects_invalid_forms(self, cmd, match):
        cli = make_cli()

        with pytest.raises(ValueError, match=match):
            cli._parse_handoff_new_command(cmd)


class TestHandoffNewHandler:
    def test_busy_rejection(self, monkeypatch):
        cli = make_cli()
        cli._agent_running = True
        monkeypatch.setattr(cli, "chat", lambda prompt: (_ for _ in ()).throw(AssertionError("chat called")))

        assert cli._handle_handoff_new_command("/handoff new") is True
        assert cli._pending_handoff_new is None

    def test_already_pending_rejection(self, monkeypatch):
        cli = make_cli()
        cli._pending_handoff_new = {"path": Path("x")}
        monkeypatch.setattr(cli, "chat", lambda prompt: (_ for _ in ()).throw(AssertionError("chat called")))

        assert cli._handle_handoff_new_command("/handoff new") is True
        assert cli._pending_handoff_new == {"path": Path("x")}


def test_writer_prompt_contains_marker_and_exact_start_prompt(tmp_path):
    cli = make_cli()
    target = tmp_path / "handoff.md"

    prompt = cli._build_handoff_writer_prompt(target, "Continue here")

    assert f"HANDOFF_WRITTEN {target}" in prompt
    assert "section titled exactly `Start prompt`" in prompt
    assert "Continue here" in prompt


class TestHandoffNewCompletion:
    def prepare_pending(self, cli, path, continuation="Continue", title=None, pre_stat=None):
        cli._pending_handoff_new = {
            "path": path,
            "title": title,
            "continuation": continuation,
            "pre_stat": pre_stat,
        }
        cli._last_chat_result = {"final_response": f"HANDOFF_WRITTEN {path}"}

    def test_stale_existing_file_does_not_switch(self, monkeypatch, tmp_path):
        cli = make_cli()
        target = tmp_path / "handoff.md"
        target.write_text("old", encoding="utf-8")
        st = target.stat()
        self.prepare_pending(cli, target, pre_stat={"mtime_ns": st.st_mtime_ns, "size": st.st_size})
        calls = []
        monkeypatch.setattr(cli, "new_session", lambda title=None: calls.append(title))

        cli._maybe_finish_handoff_new()

        assert calls == []
        assert drain(cli._pending_input) == []
        assert cli._pending_handoff_new is None

    @pytest.mark.parametrize("make_file", [False, True])
    def test_missing_or_empty_file_does_not_switch(self, monkeypatch, tmp_path, make_file):
        cli = make_cli()
        target = tmp_path / "handoff.md"
        if make_file:
            target.write_text("", encoding="utf-8")
        self.prepare_pending(cli, target)
        calls = []
        monkeypatch.setattr(cli, "new_session", lambda title=None: calls.append(title))

        cli._maybe_finish_handoff_new()

        assert calls == []
        assert drain(cli._pending_input) == []
        assert cli._pending_handoff_new is None

    @pytest.mark.parametrize("flag", ["_last_turn_interrupted", "_last_turn_failed", "_last_turn_partial"])
    def test_interrupted_failed_or_partial_does_not_switch(self, monkeypatch, tmp_path, flag):
        cli = make_cli()
        target = tmp_path / "handoff.md"
        target.write_text("handoff", encoding="utf-8")
        self.prepare_pending(cli, target)
        setattr(cli, flag, True)
        calls = []
        monkeypatch.setattr(cli, "new_session", lambda title=None: calls.append(title))

        cli._maybe_finish_handoff_new()

        assert calls == []
        assert drain(cli._pending_input) == []
        assert cli._pending_handoff_new is None

    def test_success_switches_session_and_queues_continuation_first(self, monkeypatch, tmp_path):
        cli = make_cli()
        target = tmp_path / "handoff.md"
        target.write_text("handoff", encoding="utf-8")
        self.prepare_pending(cli, target, continuation="Continue from file", title="Next")
        cli._pending_input.put("held one")
        cli._pending_input.put("held two")
        calls = []
        monkeypatch.setattr(cli, "new_session", lambda title=None: calls.append(title))

        cli._maybe_finish_handoff_new()

        assert calls == ["Next"]
        assert drain(cli._pending_input) == ["Continue from file", "held one", "held two"]
        assert cli._pending_handoff_new is None

    def test_missing_marker_warns_but_switches(self, monkeypatch, tmp_path):
        cli = make_cli()
        target = tmp_path / "handoff.md"
        target.write_text("handoff", encoding="utf-8")
        self.prepare_pending(cli, target, continuation="Continue", title=None)
        cli._last_chat_result = {"final_response": "done"}
        calls = []
        monkeypatch.setattr(cli, "new_session", lambda title=None: calls.append(title))

        cli._maybe_finish_handoff_new()

        assert calls == [None]
        assert drain(cli._pending_input) == ["Continue"]

    def test_handler_success_creates_parent_and_runs_writer(self, monkeypatch, tmp_path):
        cli = make_cli()
        target = tmp_path / "nested" / "handoff.md"
        calls = []

        def fake_chat(prompt):
            assert str(target) in prompt
            target.write_text("handoff", encoding="utf-8")
            cli._last_chat_result = {"final_response": f"HANDOFF_WRITTEN {target}"}
            cli._last_turn_failed = False
            cli._last_turn_partial = False
            cli._last_turn_interrupted = False
            calls.append("chat")

        monkeypatch.setattr(cli, "chat", fake_chat)
        monkeypatch.setattr(cli, "new_session", lambda title=None: calls.append(("new_session", title)))

        assert cli._handle_handoff_new_command(f'/handoff new {target} --title "T"') is True

        assert target.parent.is_dir()
        assert calls == ["chat", ("new_session", "T")]
        assert drain(cli._pending_input) == [f"Continue from {target}"]
