"""Tests for session-scoped-by-default model switching.

Covers:
- ``parse_model_flags`` recognises ``--session`` (and keeps ``--global``).
- ``resolve_persist_behavior`` applies the config-gated default and the
  ``--session`` / ``--global`` overrides.
- The default (no flags) is session-only, which is the user-facing fix: a
  plain ``/model <name>`` affects only the current session unless the user
  passes ``--global`` or sets ``model.persist_switch_by_default: true``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import parse_model_flags, resolve_persist_behavior


# ---------------------------------------------------------------------------
# parse_model_flags
# ---------------------------------------------------------------------------


class TestParseModelFlagsSession:
    def test_no_flags(self):
        assert parse_model_flags("sonnet") == ("sonnet", "", False, False, False)

    def test_global_flag(self):
        assert parse_model_flags("sonnet --global") == ("sonnet", "", True, False, False)

    def test_session_flag(self):
        assert parse_model_flags("sonnet --session") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )

    def test_session_with_provider(self):
        assert parse_model_flags("sonnet --provider anthropic --session") == (
            "sonnet",
            "anthropic",
            False,
            False,
            True,
        )

    def test_refresh_flag_still_parsed(self):
        assert parse_model_flags("--refresh") == ("", "", False, True, False)

    def test_unicode_dash_session_normalized(self):
        # Telegram/iOS auto-converts -- to en/em dashes.
        assert parse_model_flags("sonnet \u2013session") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )


# ---------------------------------------------------------------------------
# resolve_persist_behavior
# ---------------------------------------------------------------------------


class TestResolvePersistBehavior:
    def test_session_flag_always_session_only(self):
        # --session opts out even if the config default is True.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, True) is False

    def test_global_flag_always_persists(self):
        # --global forces persist even if the config default is False.
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(True, False) is True

    def test_default_session_only_when_config_missing(self):
        # No model section at all → built-in default (False, session-only).
        with _config({}):
            assert resolve_persist_behavior(False, False) is False

    def test_default_persists_when_key_true(self):
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, False) is True

    def test_default_session_only_when_key_false(self):
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(False, False) is False

    def test_default_when_model_is_flat_string(self):
        # Fresh install: ``model: ""`` (not a dict) → built-in default False.
        with _config({"model": ""}):
            assert resolve_persist_behavior(False, False) is False

    def test_session_overrides_global_when_both_set(self):
        # --session is the explicit opt-out and wins over --global.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(True, True) is False

    def test_provider_flag_defaults_to_session_only(self):
        # --provider without --global/--session → session only.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, False, explicit_provider="anthropic") is False

    def test_provider_with_global_still_persists(self):
        # --provider + --global → persists.
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(True, False, explicit_provider="anthropic") is True

    def test_provider_with_session_still_session_only(self):
        # --provider + --session → session only.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, True, explicit_provider="anthropic") is False

    def test_no_provider_uses_config_default(self):
        # No --provider → respects config default (True).
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, False, explicit_provider="") is True


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


class _config:
    """Context manager that patches ``load_config`` to return a fixed dict."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __enter__(self):
        self._patch = patch(
            "hermes_cli.config.load_config",
            return_value=self.cfg,
        )
        # resolve_persist_behavior imports load_config lazily inside the
        # function, so patching the source module is sufficient.
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
