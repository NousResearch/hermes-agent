"""Issue #115439: the missing-allowlist startup warning must be deferred until
after enabled platforms are known, and skipped when none are enabled.

Running with zero messaging platforms (e.g. dashboard/API-only) should not warn
about missing messaging allowlists.
"""

from __future__ import annotations

import inspect
import logging
import types

from gateway.run_startup import GatewayStartupMixin


def _host():
    host = object.__new__(GatewayStartupMixin)
    host.config = types.SimpleNamespace(platforms={})
    return host


def _clear_allowlist_env(monkeypatch):
    host = _host()
    for var in (
        *host._BUILTIN_ALLOWED_USERS_VARS,
        "GATEWAY_ALLOW_ALL_USERS",
        *host._BUILTIN_ALLOW_ALL_VARS,
    ):
        monkeypatch.delenv(var, raising=False)


def _warn_records(caplog):
    return [r for r in caplog.records if "No env user allowlists" in r.getMessage()]


class TestDeferredAllowlistWarning:
    def test_warn_helper_fires_without_allowlists(self, monkeypatch, caplog):
        _clear_allowlist_env(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            _host()._start_warn_missing_allowlist()
        assert _warn_records(caplog), "expected missing-allowlist warning with empty env"

    def test_warn_helper_quiet_when_allowlist_set(self, monkeypatch, caplog):
        _clear_allowlist_env(monkeypatch)
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            _host()._start_warn_missing_allowlist()
        assert not _warn_records(caplog)

    def test_policy_check_can_skip_warning_but_keeps_refusal(
        self, monkeypatch, caplog
    ):
        # Early startup call must be able to run the open-policy refusal
        # without emitting the (not yet gated) allowlist warning.
        _clear_allowlist_env(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            refused = _host()._start_check_access_policy(warn_missing_allowlist=False)
        assert refused is False
        assert not _warn_records(caplog)

    def test_start_defers_warning_past_prefilter_and_gates_on_count(self):
        src = inspect.getsource(GatewayStartupMixin.start)
        assert "self._start_check_access_policy(warn_missing_allowlist=False)" in src, (
            "early access-policy check must not warn before enabled platforms are known"
        )
        assert "self._start_check_access_policy()" not in src
        warn_call = "self._start_warn_missing_allowlist()"
        assert warn_call in src, "warning must be emitted from start() after prefilter"
        prefilter_pos = src.index("self._start_prefilter_platforms()")
        warn_pos = src.index(warn_call)
        assert prefilter_pos < warn_pos, "warning must come after enabled_platform_count is computed"
        assert "enabled_platform_count" in src[prefilter_pos:warn_pos], (
            "warning must be gated on enabled_platform_count"
        )
