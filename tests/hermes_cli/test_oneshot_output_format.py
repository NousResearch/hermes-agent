import json

from hermes_cli._parser import build_top_level_parser
from hermes_cli import oneshot


def test_run_oneshot_json_output_emits_single_json_object(monkeypatch, capsys):
    monkeypatch.setattr(oneshot, "_run_agent", lambda *args, **kwargs: "hello from hermes")

    rc = oneshot.run_oneshot("say hi", output_format="json")

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {"response": "hello from hermes"}


def test_run_oneshot_text_output_remains_plain_text(monkeypatch, capsys):
    monkeypatch.setattr(oneshot, "_run_agent", lambda *args, **kwargs: "plain text")

    rc = oneshot.run_oneshot("say hi", output_format="text")

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "plain text\n"


def test_run_oneshot_rejects_invalid_programmatic_output_format(capsys):
    rc = oneshot.run_oneshot("say hi", output_format="xml")

    captured = capsys.readouterr()
    assert rc == 2
    assert "--output-format must be one of: text, json" in captured.err
    assert captured.out == ""


def test_top_level_parser_accepts_oneshot_output_format():
    parser, _subparsers, _chat_parser = build_top_level_parser()

    args = parser.parse_args(["--oneshot", "say hi", "--output-format", "json"])

    assert args.oneshot == "say hi"
    assert args.output_format == "json"
