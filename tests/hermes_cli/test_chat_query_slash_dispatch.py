from __future__ import annotations

from argparse import Namespace
from types import ModuleType

import pytest


def _chat_args(**overrides):
    base = {
        "accept_hooks": False,
        "checkpoints": False,
        "cli": True,
        "compact": False,
        "continue_last": None,
        "ignore_rules": False,
        "ignore_user_config": False,
        "image": None,
        "max_turns": None,
        "model": None,
        "pass_session_id": False,
        "provider": None,
        "query": None,
        "quiet": False,
        "resume": None,
        "safe_mode": False,
        "skills": None,
        "slash": None,
        "source": None,
        "toolsets": None,
        "tui": False,
        "tui_dev": False,
        "verbose": False,
        "worktree": False,
        "yolo": False,
    }
    base.update(overrides)
    return Namespace(**base)


class _Console:
    def print(self, *args, **_kwargs):
        print(*args)


class _BaseFakeCLI:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.agent = None
        self.chats = []
        self.commands = []
        self.config = {}
        self.console = _Console()
        self.conversation_history = []
        self.preloaded_skills = []
        self.session_id = "slash-test-session"
        self.system_prompt = ""
        type(self).instances.append(self)

    def _show_security_advisories(self):
        return None

    def _print_exit_summary(self):
        return None


def _install_fake_cli(monkeypatch, cli_mod, fake_cls):
    fake_cls.instances = []
    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cls)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    return fake_cls


def test_shared_slash_classifier_recognizes_commands_but_not_paths_or_quotes():
    from hermes_cli.slash_dispatch import classify_slash_text, looks_like_slash_command

    assert looks_like_slash_command("/goal resume")
    assert not looks_like_slash_command("/tmp/worktree/README.md fix this")
    assert not looks_like_slash_command("C:/tmp/worktree/README.md fix this")
    assert not looks_like_slash_command('"/goal resume"')

    goal = classify_slash_text("/goal resume")
    assert goal.looks_like_command
    assert goal.known
    assert goal.canonical == "goal"
    assert goal.raw_args == "resume"

    closure = classify_slash_text("/closure latest --session-id abc")
    assert closure.looks_like_command
    assert closure.known
    assert closure.canonical == "closure"
    assert closure.raw_args == "latest --session-id abc"

    unknown = classify_slash_text("/not-a-real-command")
    assert unknown.looks_like_command
    assert not unknown.known


def test_closure_slash_is_noninteractive_safe():
    from hermes_cli.slash_dispatch import classify_slash_text, is_noninteractive_safe_slash

    closure = classify_slash_text("/closure latest --session-id abc")

    assert is_noninteractive_safe_slash(closure)


def test_chat_query_goal_resume_dispatches_locally_without_agent_turn(monkeypatch, capsys):
    import cli as cli_mod

    class DispatchOnlyCLI(_BaseFakeCLI):
        def _claim_active_session(self, *_args, **_kwargs):
            raise AssertionError("slash dispatch should not claim an active session")

        def chat(self, *_args, **_kwargs):
            raise AssertionError("slash dispatch should not reach chat()")

        def process_command(self, command):
            self.commands.append(command)
            print(f"handled {command}")
            return True

    fake_cls = _install_fake_cli(monkeypatch, cli_mod, DispatchOnlyCLI)

    cli_mod.main(query="/goal resume", toolsets="web")

    assert fake_cls.instances[0].commands == ["/goal resume"]
    assert "handled /goal resume" in capsys.readouterr().out


def test_chat_query_unknown_slash_exits_before_agent_turn(monkeypatch, capsys):
    import cli as cli_mod

    class NoAgentCLI(_BaseFakeCLI):
        def _claim_active_session(self, *_args, **_kwargs):
            raise AssertionError("unknown slash command should not claim an active session")

        def chat(self, *_args, **_kwargs):
            raise AssertionError("unknown slash command should not reach chat()")

    _install_fake_cli(monkeypatch, cli_mod, NoAgentCLI)

    with pytest.raises(SystemExit) as exc:
        cli_mod.main(query="/not-a-real-command", toolsets="web")

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown command `/not-a-real-command`" in captured.err
    assert "without the leading slash" in captured.err


def test_chat_query_goal_text_is_rejected_instead_of_half_queued(monkeypatch, capsys):
    import cli as cli_mod

    class NoDispatchCLI(_BaseFakeCLI):
        def _claim_active_session(self, *_args, **_kwargs):
            raise AssertionError("unsupported goal slash should not claim active session")

        def chat(self, *_args, **_kwargs):
            raise AssertionError("unsupported goal slash should not reach chat()")

        def process_command(self, *_args, **_kwargs):
            raise AssertionError("goal text should not be half-queued in noninteractive mode")

    _install_fake_cli(monkeypatch, cli_mod, NoDispatchCLI)

    with pytest.raises(SystemExit) as exc:
        cli_mod.main(query="/goal build the whole kanban", toolsets="web")

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not available through noninteractive" in err
    assert "hermes goal" in err


@pytest.mark.parametrize(
    "query",
    [
        "/tmp/worktree/README.md fix this",
        "C:/tmp/worktree/README.md fix this",
        '"/goal resume"',
    ],
)
def test_chat_query_path_and_quoted_slash_text_reaches_agent_turn(monkeypatch, query):
    import cli as cli_mod

    class AgentCLI(_BaseFakeCLI):
        def _claim_active_session(self, *_args, **_kwargs):
            return True

        def chat(self, message, images=None):
            self.chats.append((message, images))
            return "ok"

    fake_cls = _install_fake_cli(monkeypatch, cli_mod, AgentCLI)

    cli_mod.main(query=query, toolsets="web")

    assert fake_cls.instances[0].chats == [(query, None)]


def test_chat_slash_parser_argument_is_available():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(["chat", "--slash", "/goal resume"])

    assert args.slash == "/goal resume"


def test_cmd_chat_forwards_slash_to_cli_main(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}
    fake_cli = ModuleType("cli")
    fake_cli.main = lambda **kwargs: captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "cli", fake_cli)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)

    main_mod.cmd_chat(_chat_args(slash="/goal resume"))

    assert captured["slash"] == "/goal resume"
