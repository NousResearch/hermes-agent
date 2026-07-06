from hermes_cli._parser import build_top_level_parser


def test_chat_reasoning_flag_is_ephemeral_invocation_arg():
    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=lambda _args: None)

    args = parser.parse_args([
        "chat",
        "--provider",
        "openai-codex",
        "--model",
        "gpt-5.5",
        "--reasoning",
        "xhigh",
        "-q",
        "hi",
    ])

    assert args.reasoning_effort == "xhigh"
