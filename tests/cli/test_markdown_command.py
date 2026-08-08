"""Classic CLI /markdown command tests."""

from unittest.mock import patch


def _stub(mode: str):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_resume_sessions = None
    cli.final_response_markdown = mode
    return cli


def test_markdown_command_enables_render_mode():
    import cli as climod

    cli = _stub("strip")
    with (
        patch.object(climod, "_cprint") as printed,
        patch.object(climod, "save_config_value") as saved,
    ):
        assert cli.process_command("/markdown") is True

    assert cli.final_response_markdown == "render"
    saved.assert_called_once_with("display.final_response_markdown", "render")
    assert "ON" in str(printed.call_args)


def test_md_alias_disables_render_mode_to_strip():
    import cli as climod

    cli = _stub("render")
    with (
        patch.object(climod, "_cprint") as printed,
        patch.object(climod, "save_config_value") as saved,
    ):
        assert cli.process_command("/md") is True

    assert cli.final_response_markdown == "strip"
    saved.assert_called_once_with("display.final_response_markdown", "strip")
    assert "OFF" in str(printed.call_args)
