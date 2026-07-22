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


class TestNamedCustomProviderNoKeyDedup:
    """A named custom provider (config.yaml providers/custom_providers entry)
    with no resolvable api_key repeats the identical warning on every call
    until the user edits config — same dead-end shape as the OAuth/extproc
    branches above."""

    def setup_method(self):
        ac._LOGGED_NAMED_CUSTOM_NOKEY_KEYS.clear()

    def test_no_key_logs_debug_once_not_warning(self, caplog, monkeypatch):
        import hermes_cli.runtime_provider as rp

        entry = {"name": "my-custom", "base_url": "https://custom.example/v1"}
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda _p: entry)

        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            resolve_provider_client("my-custom", "some-model")
            resolve_provider_client("my-custom", "some-model")  # repeat → suppressed

        recs = [
            r for r in caplog.records
            if "has no resolvable" in r.getMessage()
        ]
        assert len(recs) == 1
        assert recs[0].levelno == logging.DEBUG
        assert not any(r.levelno >= logging.WARNING for r in recs)


class TestNamedCustomProviderNoBaseUrlDedup:
    def setup_method(self):
        ac._LOGGED_NAMED_CUSTOM_NOBASEURL_KEYS.clear()

    def test_no_base_url_logs_debug_once_not_warning(self, caplog, monkeypatch):
        import hermes_cli.runtime_provider as rp

        entry = {"name": "my-custom-2", "api_key": "sk-test"}
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda _p: entry)

        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            client, model = resolve_provider_client("my-custom-2", "some-model")
            resolve_provider_client("my-custom-2", "some-model")  # repeat → suppressed

        assert (client, model) == (None, None)
        recs = [
            r for r in caplog.records
            if "has no base_url" in r.getMessage()
        ]
        assert len(recs) == 1
        assert recs[0].levelno == logging.DEBUG
        assert not any(r.levelno >= logging.WARNING for r in recs)


class TestCopilotAcpNoModelDedup:
    def setup_method(self):
        ac._LOGGED_COPILOT_ACP_NOMODEL_KEYS.clear()

    def test_no_model_logs_debug_once_not_warning(self, caplog, monkeypatch):
        import hermes_cli.auth as auth

        monkeypatch.setattr(ac, "_read_main_model", lambda: "")
        monkeypatch.setattr(ac, "_get_aux_model_for_provider", lambda _p: "")
        # The CLI binary isn't installed on the test machine; resolving
        # external-process credentials would otherwise raise before this
        # branch's own model check runs.
        monkeypatch.setattr(
            auth, "resolve_external_process_provider_credentials",
            lambda _p: {"api_key": "k", "base_url": "http://local", "command": "copilot", "args": []},
        )

        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            client, model = resolve_provider_client("copilot-acp", "")
            resolve_provider_client("copilot-acp", "")  # repeat → suppressed

        assert (client, model) == (None, None)
        recs = [
            r for r in caplog.records
            if "no model was provided or configured" in r.getMessage()
        ]
        assert len(recs) == 1
        assert recs[0].levelno == logging.DEBUG
        assert not any(r.levelno >= logging.WARNING for r in recs)
