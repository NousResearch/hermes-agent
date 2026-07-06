from __future__ import annotations

from pathlib import Path

import pytest

WINDOWS_PROMPT_PATH = r"C:\Users\Admin\Documents\Hermes monitoring\runs\prompt.txt"


def test_resume_query_file_argv_keeps_windows_path_with_spaces_atomic():
    from hermes_cli.windows_launch import build_query_file_argv

    argv = build_query_file_argv(
        session_id="20260704_211738_60f6ef",
        prompt_path=WINDOWS_PROMPT_PATH,
        model="gpt-5.5",
    )

    assert argv == [
        "-m",
        "hermes_cli.main",
        "chat",
        "--resume",
        "20260704_211738_60f6ef",
        "--model",
        "gpt-5.5",
        "--query-file",
        WINDOWS_PROMPT_PATH,
    ]
    assert argv[argv.index("--query-file") + 1] == WINDOWS_PROMPT_PATH
    assert any("Hermes monitoring" in arg for arg in argv)
    assert not any("\n" in arg or "\r" in arg for arg in argv)


def test_query_file_launch_keeps_executable_cwd_and_prompt_path_with_spaces_atomic():
    from hermes_cli.windows_launch import build_query_file_launch

    python_exe = r"C:\Program Files\Python311\python.exe"
    cwd = r"C:\Users\Admin\AppData\Local\hermes\hermes-agent"
    prompt_path = r"C:\Users\Admin\Documents\Hermes monitoring\runs\prompt with spaces.md"

    launch = build_query_file_launch(
        python_exe=python_exe,
        session_id="resume-session",
        prompt_path=prompt_path,
        model="gpt-5.5",
        cwd=cwd,
    )

    assert launch.shell is False
    assert launch.cwd == cwd
    assert launch.executable == python_exe
    assert launch.popen_argv[0] == python_exe
    assert launch.args[launch.args.index("--query-file") + 1] == prompt_path
    assert prompt_path in launch.popen_argv
    assert not any(arg == "Hermes" or arg == "monitoring" for arg in launch.popen_argv)


def test_prompt_file_launch_writes_multiline_prompt_instead_of_shell_joining(tmp_path):
    from hermes_cli.windows_launch import build_prompt_file_launch

    prompt = "first recovery line\nsecond recovery line"
    prompt_path = tmp_path / "Hermes monitoring" / "runs" / "prompt.txt"
    launch = build_prompt_file_launch(
        python_exe=r"C:\Users\Admin\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe",
        session_id="resume-session",
        prompt_text=prompt,
        prompt_path=prompt_path,
        model="gpt-5.5",
    )

    assert prompt_path.read_text(encoding="utf-8") == prompt
    assert launch.popen_argv == [launch.executable, *launch.args]
    assert launch.args[launch.args.index("--query-file") + 1] == str(prompt_path)
    assert prompt not in launch.popen_argv
    assert not any("\n" in arg or "\r" in arg for arg in launch.popen_argv)
    assert launch.shell is False


def test_long_prompt_file_launch_keeps_prompt_text_out_of_argv(tmp_path):
    from hermes_cli.windows_launch import build_prompt_file_launch

    prompt = "Long Phase 6 recovery prompt. " * 500
    prompt_path = tmp_path / "Hermes monitoring" / "runs" / "long prompt.md"

    launch = build_prompt_file_launch(
        python_exe=r"C:\Program Files\Python311\python.exe",
        session_id="resume-session",
        prompt_text=prompt,
        prompt_path=prompt_path,
        model="gpt-5.5",
    )

    assert prompt_path.read_text(encoding="utf-8") == prompt
    assert "--query-file" in launch.args
    assert "--query" not in launch.args
    assert prompt not in launch.popen_argv
    assert launch.args[launch.args.index("--query-file") + 1] == str(prompt_path)


def test_launch_smoke_classifies_parser_usage_without_db_message_as_launch_failure():
    from hermes_cli.windows_launch import classify_launch_smoke

    result = classify_launch_smoke(
        stdout="",
        stderr=(
            "usage: hermes [-h] ...\n"
            "hermes: error: unrecognized arguments: monitoring\\runs\\prompt.txt\n"
        ),
        db_message_created=False,
    )

    assert result.status == "launch_failure"
    assert result.reason == "parser_output_without_db_message"
    assert result.parser_output_detected is True


def test_launch_smoke_does_not_override_real_agent_progress():
    from hermes_cli.windows_launch import classify_launch_smoke

    result = classify_launch_smoke(
        stdout="usage: hermes appeared in quoted user text",
        stderr="",
        db_message_created=True,
    )

    assert result.status == "ok"
    assert result.reason == "db_message_created"
    assert result.parser_output_detected is True


def test_oneshot_resume_is_rejected_with_supported_query_file_alternative(capsys):
    from hermes_cli._parser import build_top_level_parser
    import hermes_cli.main as hermes_main

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args([
        "--oneshot",
        "summarize this",
        "--resume",
        "20260704_211738_60f6ef",
    ])

    with pytest.raises(SystemExit) as exc:
        hermes_main._reject_unsupported_oneshot_resume(parser, args)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--oneshot cannot be combined with --resume" in captured.err
    assert "chat --resume" in captured.err
    assert "--query-file" in captured.err
