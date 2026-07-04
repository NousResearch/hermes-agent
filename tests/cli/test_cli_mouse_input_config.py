"""Mouse cursor placement toggle for the Hermes CLI TUI (#58170).

Default behavior is unchanged (mouse off). The opt-in path is driven by
``display.mouse_input`` in config.yaml so users can enable SGR mouse mode
without a code change. The Hermes cleanup block already strips SGR mouse
escape sequences on exit — opt-in does not risk garbage in scrollback.
"""

import pytest

import cli as cli_module


@pytest.fixture
def cli_config(monkeypatch):
    """Yield a mutable dict exposed as CLI_CONFIG for the duration of the test."""
    cfg: dict = {"display": {}}
    monkeypatch.setattr(cli_module, "CLI_CONFIG", cfg, raising=False)
    yield cfg
    cfg.clear()


class TestMouseSupportToggle:
    def test_off_by_default(self, cli_config):
        assert cli_module._resolve_mouse_support() is False

    def test_missing_display_section(self, cli_config):
        cli_config.clear()
        assert cli_module._resolve_mouse_support() is False

    def test_explicit_true(self, cli_config):
        cli_config["display"]["mouse_input"] = True
        assert cli_module._resolve_mouse_support() is True

    def test_explicit_false(self, cli_config):
        cli_config["display"]["mouse_input"] = False
        assert cli_module._resolve_mouse_support() is False

    def test_truthy_non_bool_string_coerces(self, cli_config):
        """bool("yes") → True; documented as opt-in flip via bool coercion."""
        cli_config["display"]["mouse_input"] = "yes"
        # Pin the conversion behavior; a future refactor must not silently
        # change whether string-true enables mouse mode.
        assert cli_module._resolve_mouse_support() is True

    def test_malformed_display_section_falls_back_to_false(self, cli_config):
        """A ``display`` value that isn't a dict must NOT crash TUI."""
        class Boom:
            def get(self, *a, **kw):
                raise RuntimeError("bad config")

        cli_config["display"] = Boom()
        assert cli_module._resolve_mouse_support() is False


class TestApplicationMouseKwargWiring:
    """Pins the wiring at the Application() construction site.

    We don't import the full Application factory path — its logging/
    config init has side effects that aren't worth exercising in unit
    tests. Instead we patch the helper to a sentinel and grep the
    surrounding source for ``mouse_support=_resolve_mouse_support()``
    so a future refactor cannot silently revert the wiring.
    """

    def test_helper_used_in_application_construction(self):
        import inspect
        from pathlib import Path

        repo_root = Path(cli_module.__file__).resolve().parent
        cli_text = (repo_root / "cli.py").read_text(encoding="utf-8")
        # Pin: the build site must use the helper.
        # Specific enough to avoid clashing with other Application() calls,
        # since the prompt_toolkit Application constructor is the only place
        # that takes ``mouse_support=``.
        assert "mouse_support=_resolve_mouse_support()" in cli_text, (
            "cli.py must wire Application(mouse_support=...) through the "
            "_resolve_mouse_support helper so config toggle stays live (#58170)"
        )
        # Anti-regression: the historical hardcoded off-flag must be gone.
        assert (
            "mouse_support=False,\n" not in cli_text
            and "mouse_support = False" not in cli_text
        ), (
            "Hardcoded mouse_support=False in cli.py contradicts #58170 — "
            "config toggle should be the source of truth."
        )
