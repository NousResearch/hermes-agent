"""The CLI publishes its turn state through the pane title.

Terminal multiplexers (herdr) derive an agent's working/idle state from the title, so the title
must flip on every turn and must stay untouched outside those environments.
"""

from hermes_cli import cli_chat_turn_mixin


class _FakeApp:
    def __init__(self):
        self.titles = []

    def set_title(self, text):
        self.titles.append(text)


def _install_fake_app(monkeypatch):
    app = _FakeApp()
    monkeypatch.setattr("prompt_toolkit.application.get_app", lambda: app)
    return app


def test_turn_state_is_published_in_the_title(monkeypatch):
    """A herdr pane sees ⏳ while the turn runs and ✓ once it finishes."""
    monkeypatch.setenv("HERDR_ENV", "1")
    app = _install_fake_app(monkeypatch)

    cli_chat_turn_mixin._set_pane_title("⏳ hermes")
    cli_chat_turn_mixin._set_pane_title("✓ hermes")

    assert app.titles == ["⏳ hermes", "✓ hermes"]


def test_title_is_untouched_outside_herdr(monkeypatch):
    """A plain terminal never gets repainted by this hook."""
    monkeypatch.delenv("HERDR_ENV", raising=False)
    app = _install_fake_app(monkeypatch)

    cli_chat_turn_mixin._set_pane_title("⏳ hermes")

    assert app.titles == []


def test_missing_application_is_not_an_error(monkeypatch):
    """One-shot runs have no prompt_toolkit application; the hook stays silent."""
    monkeypatch.setenv("HERDR_ENV", "1")

    def _no_app():
        raise RuntimeError("no running application")

    monkeypatch.setattr("prompt_toolkit.application.get_app", _no_app)

    cli_chat_turn_mixin._set_pane_title("✓ hermes")
