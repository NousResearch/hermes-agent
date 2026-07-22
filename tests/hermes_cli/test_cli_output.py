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


def test_prompt_strips_trailing_cr_from_piped_stdin(monkeypatch):
    """PowerShell ``"y" | hermes …`` carries CR on the first line (see #60244).

    ``prompt()`` must still treat the trailing ``\r`` as whitespace so the
    answer compares cleanly against ``y`` / ``n``.
    """
    monkeypatch.setattr("builtins.input", lambda _prompt: "y\r")

    assert cli_output.prompt("Continue?", default="n", password=False) == "y"


def test_prompt_returns_default_for_cr_only_stdin(monkeypatch):
    """A piped bare ``\r`` should still be treated as an empty answer.

    This mirrors the PowerShell pipe quirk where the first line of stdin
    may arrive as ``"\\r"`` and the install prompts would otherwise fall
    through to the input comparison instead of the default.
    """
    monkeypatch.setattr("builtins.input", lambda _prompt: "\r")

    assert cli_output.prompt("Continue?", default="y", password=False) == "y"
