from hermes_cli import cli_output, colors
from hermes_cli.colors import Colors


def _force_color(monkeypatch, enabled=True):
    monkeypatch.setattr(colors, "should_use_color", lambda: enabled)


def test_print_helpers_use_semantic_palette(monkeypatch, capsys):
    """Each helper renders its semantic role's color + prefix."""
    _force_color(monkeypatch, True)

    cli_output.print_success("saved")
    cli_output.print_warning("careful")
    cli_output.print_error("boom")
    out = capsys.readouterr().out

    assert Colors.GREEN in out and "✓ saved" in out
    assert Colors.YELLOW in out and "⚠ careful" in out
    assert Colors.RED in out and "✗ boom" in out


def test_header_uses_design_system_heading_style(monkeypatch, capsys):
    """Headers render in the cyan+bold design-system heading style.

    Regression guard for issue #34566: section headers previously used a
    one-off yellow here while config.py/cron.py used cyan, producing the
    inconsistent visual hierarchy the issue calls out.
    """
    _force_color(monkeypatch, True)

    cli_output.print_header("Section")
    out = capsys.readouterr().out

    assert Colors.CYAN in out
    assert Colors.BOLD in out
    assert "Section" in out


def test_password_prompt_uses_masked_secret_prompt(monkeypatch):
    seen = {}

    def fake_masked_secret_prompt(display):
        seen["display"] = display
        return " secret "

    monkeypatch.setattr(cli_output, "masked_secret_prompt", fake_masked_secret_prompt)

    assert cli_output.prompt("API key", default="old", password=True) == "secret"
    assert "API key [old]" in seen["display"]


def test_empty_password_prompt_returns_default(monkeypatch):
    monkeypatch.setattr(cli_output, "masked_secret_prompt", lambda _display: "")

    assert cli_output.prompt("API key", default="old", password=True) == "old"
