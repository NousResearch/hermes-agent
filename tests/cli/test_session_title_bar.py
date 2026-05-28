from unittest.mock import patch

import cli


BASE_CONFIG = {
    "display": {
        "session_title_bar": {
            "enabled": True,
            "format": "{title}",
            "foreground": "#87CEEB",
            "background": "#1a1a2e",
        }
    }
}


def _make_cli():
    cli_obj = object.__new__(cli.HermesCLI)
    cli_obj.config = BASE_CONFIG
    cli_obj.session_id = "test-session"
    cli_obj._session_db = None
    cli_obj._app = None
    cli_obj._session_title = ""
    return cli_obj


def test_session_title_bar_config_defaults_enabled():
    cfg = cli._session_title_bar_config({"display": {}})

    assert cfg["enabled"] is True
    assert cfg["format"] == "{title}"
    assert cfg["foreground"] == "#87CEEB"
    assert cfg["background"] == "#1a1a2e"


def test_session_title_bar_sanitizes_control_sequences():
    assert cli._clean_session_title("Bad\033]0;Injected\007\nTitle") == "Bad Title"


def test_session_title_bar_renders_current_title_fragments():
    cli_obj = _make_cli()
    cli_obj._session_title = "Builder Ledger"

    fragments = cli_obj._get_session_title_bar_fragments(width=80)
    text = "".join(part for _style, part in fragments)

    assert text == " Builder Ledger ".ljust(80)
    assert len(text) == 80
    assert fragments[0][0] == "class:session-title-bar"


def test_session_title_bar_trims_to_terminal_width():
    cli_obj = _make_cli()
    cli_obj._session_title = "A very long topic name that should not wrap the terminal chrome"

    text = "".join(part for _style, part in cli_obj._get_session_title_bar_fragments(width=32))

    assert len(text) == 32
    assert text.rstrip().endswith("...")


def test_session_title_bar_style_uses_configured_colors():
    cli_obj = _make_cli()
    cli_obj.config = {
        "display": {
            "session_title_bar": {
                "enabled": True,
                "foreground": "#FFD700",
                "background": "#111827",
            }
        }
    }
    cli_obj._tui_style_base = {"session-title-bar": "bg:#1a1a2e #87CEEB"}

    styles = cli_obj._build_tui_style_dict()

    assert styles["session-title-bar"] == "bg:#111827 #FFD700"


def test_session_title_bar_can_be_disabled():
    cli_obj = _make_cli()
    cli_obj.config = {"display": {"session_title_bar": {"enabled": False}}}
    cli_obj._session_title = "Hidden"

    assert cli_obj._get_session_title_bar_fragments(width=80) == []


def test_title_command_updates_in_terminal_session_title(monkeypatch):
    clean_config = {
        "model": {"default": "test-model", "provider": "auto", "base_url": ""},
        "display": {
            "compact": False,
            "tool_progress": "off",
            "resume_display": "full",
            "session_title_bar": {
                "enabled": True,
                "format": "{title}",
            },
        },
        "agent": {},
        "terminal": {"env_type": "local"},
    }

    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict(cli.__dict__, {"CLI_CONFIG": clean_config}),
    ):
        cli_obj = cli.HermesCLI()
        cli_obj._session_db.create_session(session_id=cli_obj.session_id, source="cli")
        cli_obj.process_command("/title Builder Ledger")

    assert cli_obj._session_title == "Builder Ledger"


def test_auto_title_callback_updates_in_terminal_session_title():
    cli_obj = _make_cli()

    cli.HermesCLI._on_auto_title(cli_obj, "Server Cleanup")

    assert cli_obj._session_title == "Server Cleanup"


def test_status_bar_pads_to_terminal_width_for_uniform_chrome(monkeypatch):
    cli_obj = _make_cli()
    cli_obj._status_bar_visible = True
    monkeypatch.setattr(cli_obj, "_get_tui_terminal_width", lambda: 80)
    monkeypatch.setattr(
        cli_obj,
        "_get_status_bar_snapshot",
        lambda: {
            "model_short": "gpt-test",
            "duration": "1m",
            "context_percent": 42,
            "context_length": 100000,
            "context_tokens": 42000,
            "compressions": 0,
            "active_background_tasks": 0,
            "prompt_elapsed": "⏲ 0s",
        },
    )

    fragments = cli_obj._get_status_bar_fragments()
    text = "".join(part for _style, part in fragments)

    assert len(text) == 80
    assert fragments[-1][0] == "class:status-bar"
    assert fragments[-1][1].isspace()
