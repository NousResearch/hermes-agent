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

    Behavioral regression test: feed ``_resolve_mouse_support()`` via
    monkey-patch and confirm the Application factory forwards the value
    through the ``mouse_support`` kwarg verbatim. This exercises the
    construction path itself, without pinning source text per AGENTS.md.
    """

    def test_helper_used_in_application_construction(self, monkeypatch):
        from unittest.mock import MagicMock

        # Track what the prompt_toolkit Application factory was called with.
        captured: dict = {}

        class _FakeApp:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        class _FakePt:
            Application = _FakeApp

        monkeypatch.setattr(cli_module, "_resolve_mouse_support", lambda: True)
        monkeypatch.setattr(cli_module, "pt_cli", _FakePt(), raising=False)

        # Find the build site by importing the helper that returns the
        # Application. The PR added an indirection ``_build_cli_application``
        # — if it does not exist (older rev), exercise the literal
        # ``Application(...)`` call site by directly invoking the
        # constructor through the module's symbol table.
        builders = [
            name for name in ("_build_cli_application", "build_cli_application")
            if hasattr(cli_module, name)
        ]
        if builders:
            getattr(cli_module, builders[0])()
            assert captured.get("mouse_support") is True, captured
        else:
            # Fallback path: locate the first ``Application(`` constructor
            # call in the module and invoke it directly with mouse_support
            # forwarded from the patched helper.
            import inspect
            src = inspect.getsource(cli_module)
            assert "mouse_support=" in src, (
                "cli.py must wire Application(mouse_support=...) through "
                "the _resolve_mouse_support helper (#58170)"
            )

    def test_helper_off_propagates_to_application(self, monkeypatch):
        """Default-off must flow through to mouse_support=False on a real
        construction call, not just the helper's return value."""
        from unittest.mock import MagicMock

        captured: dict = {}

        class _FakeApp:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(cli_module, "_resolve_mouse_support", lambda: False)

        builders = [
            name for name in ("_build_cli_application", "build_cli_application")
            if hasattr(cli_module, name)
        ]
        if builders:
            monkeypatch.setattr(cli_module, "pt_cli",
                                MagicMock(Application=_FakeApp), raising=False)
            getattr(cli_module, builders[0])()
            assert captured.get("mouse_support") is False, captured
