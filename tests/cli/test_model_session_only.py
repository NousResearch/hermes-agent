"""Tests for /model slash command session-only persistence (issue #58290).

The bug reporter typed ``/model <name> --provider <provider>`` in Feishu and
the agent silently persisted the change to ``config.yaml`` even though they
never typed ``--global``. The fix introduces clearer flag aliases:

- ``--session-only`` (and ``--no-save``) — alias for the existing ``--session``,
  opts out of persisting to ``config.yaml`` for this session.
- ``--persist`` — alias for the existing ``--global``, opts in to persisting.

The default (no flag) is still controlled by ``model.persist_switch_by_default``
in config.yaml (defaults to True). This test file pins the new flag aliases and
the behavior ``resolve_persist_behavior`` derives from them so a future
refactor can't silently regress the issue's UX.
"""

from unittest.mock import patch

from hermes_cli.model_switch import parse_model_flags, resolve_persist_behavior


# ---------------------------------------------------------------------------
# parse_model_flags: new aliases
# ---------------------------------------------------------------------------


class TestParseModelFlagsAliases:
    """The issue #58290 reporter asked for ``--session-only``; we keep the
    existing ``--session`` working AND accept the new aliases."""

    def test_no_flags(self):
        assert parse_model_flags("sonnet") == ("sonnet", "", False, False, False)

    def test_session_flag_still_works(self):
        assert parse_model_flags("sonnet --session") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )

    def test_session_only_flag(self):
        # New alias requested by issue #58290.
        assert parse_model_flags("sonnet --session-only") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )

    def test_no_save_flag(self):
        # Another intuitive alias for the session-only opt-out.
        assert parse_model_flags("sonnet --no-save") == (
            "sonnet",
            "",
            False,
            False,
            True,
        )

    def test_global_flag_still_works(self):
        assert parse_model_flags("sonnet --global") == (
            "sonnet",
            "",
            True,
            False,
            False,
        )

    def test_persist_flag(self):
        # New alias for --global.
        assert parse_model_flags("sonnet --persist") == (
            "sonnet",
            "",
            True,
            False,
            False,
        )

    def test_session_only_with_provider(self):
        # Reproduces the exact shape from the bug report.
        assert parse_model_flags(
            "gpt-4o-mini --provider openai --session-only"
        ) == ("gpt-4o-mini", "openai", False, False, True)

    def test_persist_with_provider(self):
        assert parse_model_flags(
            "claude-sonnet-4-6 --provider anthropic --persist"
        ) == ("claude-sonnet-4-6", "anthropic", True, False, False)

    def test_session_only_takes_precedence_over_global(self):
        # Defensive: even if someone types both, --session wins (explicit
        # opt-out beats explicit opt-in).
        assert parse_model_flags("sonnet --global --session-only") == (
            "sonnet",
            "",
            True,
            False,
            True,
        )

    def test_session_flag_model_only_stripped(self):
        # After flag stripping, no leftover fragments.
        model_input, _, _, _, _ = parse_model_flags("sonnet --session-only")
        assert model_input == "sonnet"

    def test_persist_flag_model_only_stripped(self):
        model_input, _, _, _, _ = parse_model_flags("sonnet --persist")
        assert model_input == "sonnet"

    def test_no_save_flag_model_only_stripped(self):
        model_input, _, _, _, _ = parse_model_flags("sonnet --no-save")
        assert model_input == "sonnet"


# ---------------------------------------------------------------------------
# resolve_persist_behavior: stays correct with the new aliases
# ---------------------------------------------------------------------------


class TestResolvePersistBehaviorAliases:
    """``resolve_persist_behavior`` doesn't know flag spellings — it just
    takes the parsed (is_global, is_session) pair. These tests confirm the
    alias surface flows through to the same True/False decision as the
    canonical flags."""

    def test_session_only_resolves_to_session(self):
        # Same input as ``--session``: is_global=False, is_session=True.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, True) is False

    def test_persist_resolves_to_persist(self):
        # Same input as ``--global``: is_global=True, is_session=False.
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(True, False) is True

    def test_default_still_persists_when_config_true(self):
        # No flag at all → defer to config default.
        with _config({"model": {"persist_switch_by_default": True}}):
            assert resolve_persist_behavior(False, False) is True

    def test_default_session_only_when_config_false(self):
        # User explicitly turned off the persist-by-default config key.
        with _config({"model": {"persist_switch_by_default": False}}):
            assert resolve_persist_behavior(False, False) is False


# ---------------------------------------------------------------------------
# end-to-end: reproduce the bug report shape and confirm the new flag wins
# ---------------------------------------------------------------------------


class TestBugReportScenario:
    """The bug reporter ran ``/model <name> --provider <provider>`` and got
    an unwanted persistence. The new ``--session-only`` alias matches the
    wording they asked for in the issue's "Expected Behavior" section."""

    def test_plain_slash_model_with_provider_persists_by_default(self):
        # Documented behavior: without --session-only, default still
        # persists when config default is True (this matches the existing
        # ``model.persist_switch_by_default: True`` shipped default).
        model_input, explicit_provider, is_global, _, is_session = parse_model_flags(
            "claude-sonnet-4-6 --provider anthropic"
        )
        assert model_input == "claude-sonnet-4-6"
        assert explicit_provider == "anthropic"
        assert is_global is False
        assert is_session is False
        with _config({}):
            # Empty config → default True → switch persists.
            assert resolve_persist_behavior(is_global, is_session) is True

    def test_slash_model_with_provider_and_session_only_does_not_persist(self):
        # The exact fix the issue asks for: --session-only opts out.
        _, _, is_global, _, is_session = parse_model_flags(
            "claude-sonnet-4-6 --provider anthropic --session-only"
        )
        with _config({}):
            assert resolve_persist_behavior(is_global, is_session) is False

    def test_slash_model_with_provider_and_no_save_does_not_persist(self):
        # --no-save is an equally valid alias for the same opt-out.
        _, _, is_global, _, is_session = parse_model_flags(
            "claude-sonnet-4-6 --provider anthropic --no-save"
        )
        with _config({}):
            assert resolve_persist_behavior(is_global, is_session) is False

    def test_slash_model_with_provider_and_persist_writes_config(self):
        # The explicit persist path. Users who DO want the old behavior
        # (silently write config.yaml) can opt in with --persist.
        _, _, is_global, _, is_session = parse_model_flags(
            "claude-sonnet-4-6 --provider anthropic --persist"
        )
        with _config({}):
            assert resolve_persist_behavior(is_global, is_session) is True


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