from __future__ import annotations

import os
import sys
import types

import pytest


def _parser():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=lambda _args: None)
    return parser


@pytest.mark.parametrize(
    "argv",
    [
        ["--disable-fallback-model", "chat"],
        ["chat", "--disable-fallback-model"],
    ],
)
def test_disable_fallback_model_survives_before_or_after_chat(argv) -> None:
    args = _parser().parse_args(argv)

    assert args.command == "chat"
    assert args.disable_fallback_model is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--session-id", "paperclip-123", "chat"],
        ["chat", "--session-id", "paperclip-123"],
    ],
)
def test_session_id_survives_before_or_after_chat(argv) -> None:
    args = _parser().parse_args(argv)

    assert args.command == "chat"
    assert args.session_id == "paperclip-123"


def test_cmd_chat_exports_hard_stop_and_forwards_fresh_session(
    monkeypatch,
) -> None:
    import hermes_cli.main as main_mod

    args = _parser().parse_args(
        [
            "chat",
            "--session-id",
            "paperclip-123",
            "--disable-fallback-model",
        ]
    )
    captured: dict[str, object] = {}
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kwargs: captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)

    main_mod.cmd_chat(args)

    assert os.environ["HERMES_DISABLE_FALLBACK_MODEL"] == "1"
    assert captured["disable_fallback_model"] is True
    assert captured["session_id"] == "paperclip-123"


@pytest.mark.parametrize("resume_args", [["--resume", "existing"], ["--continue"]])
def test_session_id_rejects_resume_and_continue(monkeypatch, capsys, resume_args) -> None:
    import hermes_cli.main as main_mod

    args = _parser().parse_args(
        ["chat", "--session-id", "paperclip-123", *resume_args]
    )
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 2
    assert "cannot be combined with --resume or --continue" in capsys.readouterr().err


def test_session_id_rejects_tui(monkeypatch, capsys) -> None:
    import hermes_cli.main as main_mod

    args = _parser().parse_args(["chat", "--session-id", "paperclip-123"])
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: True)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 2
    assert "supported only by classic/headless chat" in capsys.readouterr().err
