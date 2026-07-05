from __future__ import annotations

from io import StringIO

import pytest

class _Console:
    def print(self, *args, **_kwargs):
        print(*args)


class _FakeCLI:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.agent = None
        self.chats = []
        self.config = {}
        self.console = _Console()
        self.conversation_history = []
        self.preloaded_skills = []
        self.session_id = "query-file-session"
        self.system_prompt = ""
        type(self).instances.append(self)

    def _claim_active_session(self, *_args, **_kwargs):
        return True

    def _show_security_advisories(self):
        return None

    def _print_exit_summary(self):
        return None

    def chat(self, message, images=None):
        self.chats.append((message, images))
        return "ok"


def _install_fake_cli(monkeypatch, cli_mod, fake_cls=_FakeCLI):
    fake_cls.instances = []
    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cls)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    return fake_cls


def test_chat_parser_accepts_query_file_and_stdin_query():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()

    file_args = parser.parse_args(["chat", "--query-file", "prompt.txt"])
    stdin_args = parser.parse_args(["chat", "--stdin-query"])

    assert file_args.query_file == "prompt.txt"
    assert stdin_args.stdin_query is True


def test_legacy_cli_exports_atomic_windows_resume_query_file_argv_builder():
    import cli as cli_mod

    prompt_path = r"C:\Users\Admin\Documents\Hermes monitoring\runs\prompt.txt"

    argv = cli_mod.build_windows_query_file_resume_argv(
        session_id="resume-session",
        prompt_path=prompt_path,
        model="gpt-5.5",
    )

    assert argv == [
        "-m",
        "hermes_cli.main",
        "chat",
        "--resume",
        "resume-session",
        "--model",
        "gpt-5.5",
        "--query-file",
        prompt_path,
    ]
    assert argv[argv.index("--query-file") + 1] == prompt_path
    assert not any("\n" in arg or "\r" in arg for arg in argv)


def test_query_file_loads_utf8_and_normalizes_crlf(tmp_path, monkeypatch):
    import cli as cli_mod

    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"first line\r\nsecond line\r\n")
    fake_cls = _install_fake_cli(monkeypatch, cli_mod)

    cli_mod.main(query_file=str(prompt), toolsets="web")

    assert fake_cls.instances[0].chats == [("first line\nsecond line\n", None)]


def test_stdin_query_loads_piped_text_and_normalizes_cr(monkeypatch):
    import cli as cli_mod

    fake_cls = _install_fake_cli(monkeypatch, cli_mod)
    monkeypatch.setattr(cli_mod.sys, "stdin", StringIO("first\rsecond\r\n"))

    cli_mod.main(stdin_query=True, toolsets="web")

    assert fake_cls.instances[0].chats == [("first\nsecond\n", None)]


def test_query_sources_are_mutually_exclusive_before_cli_init(tmp_path, monkeypatch):
    import cli as cli_mod

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("file prompt", encoding="utf-8")

    class ExplodingCLI(_FakeCLI):
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("CLI should not be initialized for invalid query flags")

    monkeypatch.setattr(cli_mod, "HermesCLI", ExplodingCLI)

    with pytest.raises(ValueError) as exc:
        cli_mod.main(query="inline prompt", query_file=str(prompt), toolsets="web")

    assert "mutually exclusive" in str(exc.value)


def test_empty_query_file_fails_before_active_session_claim(tmp_path, monkeypatch):
    import cli as cli_mod

    prompt = tmp_path / "empty.txt"
    prompt.write_text("", encoding="utf-8")

    class NoClaimCLI(_FakeCLI):
        def _claim_active_session(self, *_args, **_kwargs):
            raise AssertionError("empty query file should not claim active session")

    _install_fake_cli(monkeypatch, cli_mod, NoClaimCLI)

    with pytest.raises(ValueError) as exc:
        cli_mod.main(query_file=str(prompt), toolsets="web")

    assert "empty" in str(exc.value).lower()
