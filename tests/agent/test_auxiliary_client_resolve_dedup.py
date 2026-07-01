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
