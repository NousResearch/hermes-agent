"""Non-TTY / non-interactive guards for hermes_cli.setup.prompt_yes_no."""

from types import SimpleNamespace

import hermes_cli.setup as setup_mod


def test_prompt_yes_no_non_tty_returns_default_without_input(monkeypatch):
    """Piped stdin without HERMES_NONINTERACTIVE must not block on input()."""
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(setup_mod.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("input() must not be called for non-TTY stdin")
        ),
    )

    assert setup_mod.prompt_yes_no("q?", default=True) is True
    assert setup_mod.prompt_yes_no("q?", default=False) is False


def test_prompt_yes_no_none_stdin_returns_default_without_input(monkeypatch):
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(setup_mod.sys, "stdin", None)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("input() must not be called when stdin is None")
        ),
    )

    assert setup_mod.prompt_yes_no("q?", default=True) is True


def test_prompt_yes_no_tty_still_calls_input(monkeypatch):
    """True TTY + interactive remains on the existing input() path."""
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(setup_mod.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    calls = []

    def fake_input(_prompt=""):
        calls.append(_prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)

    assert setup_mod.prompt_yes_no("q?", default=True) is True
    assert len(calls) == 1


def test_prompt_yes_no_noninteractive_env_still_returns_default(monkeypatch):
    monkeypatch.setenv("HERMES_NONINTERACTIVE", "1")
    monkeypatch.setattr(setup_mod.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HERMES_NONINTERACTIVE must skip input()")
        ),
    )

    assert setup_mod.prompt_yes_no("q?", default=False) is False
