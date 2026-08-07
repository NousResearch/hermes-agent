from wcwidth import wcswidth

from hermes_cli import cli_output


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


def test_format_box_uses_terminal_cell_width_for_every_line():
    lines = cli_output.format_box(
        [
            ("⚕ Single-cell symbol", 4),
            ("🦋 Double-cell symbol", 4),
        ],
        divider_after={0},
    )

    assert all(wcswidth(line) == cli_output.STANDARD_BOX_WIDTH for line in lines)
    assert lines[1].startswith("│    ⚕ Single-cell symbol")
    assert lines[2].startswith("├")
    assert lines[3].startswith("│    🦋 Double-cell symbol")
