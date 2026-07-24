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


def test_visible_prompt_strips_bracketed_paste_markers(monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        lambda _display: "\x1b[200~ pasted value \x1b[201~",
    )

    assert cli_output.prompt("Name") == "pasted value"


def test_secret_prompt_strips_bracketed_paste_markers(monkeypatch):
    monkeypatch.setattr(
        cli_output,
        "masked_secret_prompt",
        lambda _display: "\x1b[200~secret-token\x1b[201~",
    )

    assert cli_output.prompt("Bot token", password=True) == "secret-token"
