"""Tests for resolve_provider_client fall-through log dedup (salvage #56283).

Both fall-through branches (unknown provider, unhandled auth_type) were demoted
from ``logger.warning`` to ``logger.debug`` with per-process dedup: the first
occurrence surfaces for diagnostics; identical repeats are suppressed for the
lifetime of the process so a retry loop can't spam the logs.
"""

import logging

import agent.auxiliary_client as ac
from agent.auxiliary_client import resolve_provider_client


class TestUnknownProviderDedup:
    def setup_method(self):
        ac._LOGGED_UNKNOWN_PROVIDER_KEYS.clear()

    def test_unknown_provider_logs_debug_once_not_warning(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            client, model = resolve_provider_client("no_such_provider_xyz", "")
        assert (client, model) == (None, None)
        recs = [
            r for r in caplog.records
            if "unknown provider" in r.getMessage()
        ]
        # Exactly one record, and it is DEBUG (never WARNING).
        assert len(recs) == 1
        assert recs[0].levelno == logging.DEBUG
        assert not any(r.levelno >= logging.WARNING for r in recs)

    def test_unknown_provider_repeat_is_suppressed(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            resolve_provider_client("no_such_provider_xyz", "")
            resolve_provider_client("no_such_provider_xyz", "")
            resolve_provider_client("no_such_provider_xyz", "")
        recs = [
            r for r in caplog.records
            if "unknown provider" in r.getMessage()
        ]
        # Three calls, one log line — dedup suppressed the repeats.
        assert len(recs) == 1

    def test_distinct_unknown_providers_each_log_once(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            resolve_provider_client("bogus_a", "")
            resolve_provider_client("bogus_b", "")
        recs = [
            r for r in caplog.records
            if "unknown provider" in r.getMessage()
        ]
        assert len(recs) == 2


class TestUnhandledAuthTypeDedup:
    def setup_method(self):
        ac._LOGGED_UNHANDLED_AUTHTYPE_KEYS.clear()

    def test_unhandled_auth_type_logs_debug_once_not_warning(self, caplog, monkeypatch):
        import hermes_cli.auth as auth
        from hermes_cli.auth import ProviderConfig

        # A registered provider whose auth_type matches no handled branch →
        # the terminal "unhandled auth_type" fall-through.
        bogus = ProviderConfig(
            id="bogus_authtype",
            name="Bogus",
            auth_type="totally_unhandled_scheme",
        )
        patched = dict(auth.PROVIDER_REGISTRY)
        patched["bogus_authtype"] = bogus
        monkeypatch.setattr(auth, "PROVIDER_REGISTRY", patched)

        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            client, model = resolve_provider_client("bogus_authtype", "")
            resolve_provider_client("bogus_authtype", "")  # repeat → suppressed

        assert (client, model) == (None, None)
        recs = [
            r for r in caplog.records
            if "unhandled auth_type" in r.getMessage()
        ]
        # Two calls, one DEBUG record, never WARNING.
        assert len(recs) == 1
        assert recs[0].levelno == logging.DEBUG
        assert not any(r.levelno >= logging.WARNING for r in recs)


class TestUnsupportedOAuthDedup:
    def setup_method(self):
        ac._LOGGED_UNSUPPORTED_OAUTH_KEYS.clear()

    def test_unsupported_oauth_provider_logs_debug_once(self, caplog, monkeypatch):
        import hermes_cli.auth as auth
        from hermes_cli.auth import ProviderConfig

        # A registered oauth_* provider that is not one of the directly-handled
        # names (nous / openai-codex / xai-oauth) → the OAuth dead-end branch.
        bogus = ProviderConfig(
            id="bogus_oauth",
            name="BogusOAuth",
            auth_type="oauth_device_code",
        )
        patched = dict(auth.PROVIDER_REGISTRY)
        patched["bogus_oauth"] = bogus
        monkeypatch.setattr(auth, "PROVIDER_REGISTRY", patched)

        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            resolve_provider_client("bogus_oauth", "")
            resolve_provider_client("bogus_oauth", "")

        recs = [
            r for r in caplog.records
            if "OAuth provider" in r.getMessage() and "not " in r.getMessage()
        ]
        assert len(recs) == 1
        assert recs[0].levelno == logging.DEBUG
        assert not any(r.levelno >= logging.WARNING for r in recs)
