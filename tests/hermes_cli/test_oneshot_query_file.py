"""Regression tests for top-level one-shot prompt file / stdin support."""

import argparse
import io
import json
import sys

import pytest

import hermes_bootstrap
import hermes_cli._early_recovery


@pytest.fixture(autouse=True)
def _stub_early_recovery(monkeypatch):
    _ = hermes_bootstrap
    monkeypatch.setattr(hermes_cli._early_recovery, "restore_interrupted_pull", lambda *_a, **_k: False)


@pytest.mark.parametrize("fast_launch", ["0", "1"])
@pytest.mark.parametrize("flag", ["-Q", "--query-file", "--prompt-file"])
def test_top_level_query_file_routes_into_oneshot_without_argv_prompt(monkeypatch, tmp_path, fast_launch, flag):
    prompt_path = tmp_path / "task-brief.txt"
    prompt_body = 'Run vite preview --port 4179 && echo "$(uname)"'
    prompt_path.write_text(prompt_body, encoding="utf-8")

    seen = []
    monkeypatch.setenv("HERMES_DISABLE_FAST_CHAT_LAUNCH", fast_launch)
    monkeypatch.setattr(sys, "argv", ["hermes", flag, str(prompt_path), "-m", "glm-5.3-flash"])
    import hermes_cli.main as main_mod
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr(
        "hermes_cli.gateway_chat_startup.launch_gateway_chat",
        lambda args: seen.append((args.oneshot, args.model)) or 0,
    )

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 0
    assert seen == [(prompt_body, "glm-5.3-flash")]


@pytest.mark.parametrize("argv", [
    ["hermes", "-z", "-"],
    ["hermes", "--oneshot", "-"],
    ["hermes", "-Q", "-"],
    ["hermes", "--query-file", "-"],
])
def test_top_level_oneshot_reads_stdin_when_dash_given(monkeypatch, argv):
    seen = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdin", io.StringIO("Prompt via stdin with 'quotes' and $(subshell)\n"))
    import hermes_cli.main as main_mod
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr(
        "hermes_cli.gateway_chat_startup.launch_gateway_chat",
        lambda args: seen.append(args.oneshot) or 0,
    )

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 0
    assert seen == ["Prompt via stdin with 'quotes' and $(subshell)\n"]


def test_top_level_oneshot_and_query_file_are_mutually_exclusive():
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["-z", "inline", "-Q", "/tmp/prompt.txt"])
    assert excinfo.value.code == 2


def test_chat_subcommand_quiet_and_query_file_remain_unmodified():
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "-Q", "--query-file", "/tmp/dm.txt"])
    assert args.command == "chat"
    assert args.quiet is True
    assert args.query_file == "/tmp/dm.txt"
    assert getattr(args, "oneshot_query_file", None) is None


@pytest.mark.parametrize("case, expected_err", [
    ("missing", "cannot read --query-file"),
    ("empty_file", "is empty"),
    ("empty_stdin", "stdin prompt for -z/--oneshot is empty"),
])
def test_invalid_oneshot_prompt_inputs_exit_2_before_startup(monkeypatch, capsys, tmp_path, case, expected_err):
    if case == "missing":
        monkeypatch.setattr(sys, "argv", ["hermes", "-Q", str(tmp_path / "missing.txt")])
    elif case == "empty_file":
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["hermes", "--query-file", str(empty)])
    else:
        monkeypatch.setattr(sys, "argv", ["hermes", "-z", "-"])
        monkeypatch.setattr(sys, "stdin", io.StringIO("  \n"))

    import hermes_cli.main as main_mod
    startup_called = []
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: startup_called.append(True))

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 2
    assert startup_called == []
    assert expected_err in capsys.readouterr().err


def test_launch_from_args_resolves_oneshot_query_file_and_writes_usage_file_on_error(tmp_path, capsys):
    from hermes_cli import gateway_chat

    usage = tmp_path / "usage.json"
    args = argparse.Namespace(
        oneshot_query_file=str(tmp_path / "nope.txt"),
        oneshot=None,
        usage_file=str(usage),
        output_format="text",
        model=None,
    )
    code = gateway_chat.launch_from_args(args)
    assert code == 2
    assert "cannot read --query-file" in capsys.readouterr().err
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["failed"] is True
    assert report["completed"] is False
