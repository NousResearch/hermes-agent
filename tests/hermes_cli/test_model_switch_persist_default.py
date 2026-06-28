"""Tests for persist-by-default model switching.

Covers:
- ``parse_model_flags`` recognises ``--session`` (and keeps ``--global``).
- ``resolve_persist_behavior`` applies the config-gated default and the
  ``--session`` / ``--global`` overrides.
- The default (no flags) persists, which is the user-facing fix: a plain
  ``/model <name>`` survives across sessions.
"""

from unittest.mock import patch

from hermes_cli.model_switch import parse_model_flags, resolve_persist_behavior


# ---------------------------------------------------------------------------
# parse_model_flags
# ---------------------------------------------------------------------------


class TestParseModelFlagsSession:
    def test_no_flags(self):
        assert parse_model_flags("sonnet") == ("sonnet", "", False, False, False, None)

    def test_global_flag(self):
        assert parse_model_flags("sonnet --global") == ("sonnet", "", True, False, False, None)

    def test_session_flag(self):
        assert parse_model_flags("sonnet --session") == (
            "sonnet",
            "",
            False,
            False,
            True,
            None,
        )

    def test_session_with_provider(self):
        assert parse_model_flags("sonnet --provider anthropic --session") == (
            "sonnet",
            "anthropic",
            False,
            False,
            True,
            None,
        )

    def test_refresh_flag_still_parsed(self):
        assert parse_model_flags("--refresh") == ("", "", False, True, False, None)

    def test_max_context_flag(self):
        assert parse_model_flags("glm-5.2 --provider zai --max-context 262144") == (
            "glm-5.2",
            "zai",
            False,
            False,
            False,
            "262144",
        )

    def test_context_length_alias_flag(self):
        assert parse_model_flags("glm-5.2 --context-length auto --session") == (
            "glm-5.2",
            "",
            False,
            False,
            True,
            "auto",
        )

    def test_missing_max_context_value_marks_invalid_without_model_leak(self):
        assert parse_model_flags("--max-context") == ("", "", False, False, False, "")

    def test_missing_context_length_value_marks_invalid_without_model_leak(self):
        assert parse_model_flags("glm-5.2 --context-length") == (
            "glm-5.2",
            "",
            False,
            False,
            False,
            "",
        )

    def test_unicode_dash_session_normalized(self):
        # Telegram/iOS auto-converts -- to en/em dashes.
        assert parse_model_flags("sonnet \u2013session") == (
            "sonnet",
            "",
            False,
            False,
            True,
            None,
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

    def test_default_persists_when_config_missing(self):
        # No model section at all → built-in default (True, None).
        with _config({}):
            assert resolve_persist_behavior(False, False) is True

    def test_default_persists_when_key_true(self):
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, False) is True

    def test_default_session_only_when_key_false(self):
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(False, False) is False

    def test_default_when_model_is_flat_string(self):
        # Fresh install: ``model: ""`` (not a dict) → built-in default True.
        with _config({"model": ""}):
            assert resolve_persist_behavior(False, False) is True

    def test_session_overrides_global_when_both_set(self):
        # --session is the explicit opt-out and wins over --global.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(True, True) is False


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
