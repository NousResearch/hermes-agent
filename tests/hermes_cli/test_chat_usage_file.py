"""Regression tests for chat -q usage-report output."""

from hermes_cli._parser import build_top_level_parser


def test_chat_query_accepts_usage_file_after_subcommand():
    parser, _subparsers, _chat_parser = build_top_level_parser()

    args = parser.parse_args(
        ["chat", "--usage-file", "/tmp/chat-usage.json", "-Q", "-q", "inspect"]
    )

    assert args.command == "chat"
    assert args.query == "inspect"
    assert args.usage_file == "/tmp/chat-usage.json"
