# Copyright 2026 Nous Research (Licensed under the Apache License, Version 2.0)
"""Per-attempt quota observation + ctx.quota_fetch contract tests.

Covers: outcome classification (200/401/403/429/5xx/timeout), header window
capture and Retry-After parsing, fail-closed capability gating of the
``quota_observation`` hook field, and the ``ctx.quota_fetch`` host boundary
(OpenRouter official key limits only, scoped to the requesting pool entry).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from agent.credential_scope import capture_attempt_credential_scope
from agent.quota_fetch import OFFICIAL_KEY_LIMITS
from agent.quota_observation import (
    OUTCOME_AUTH_FAILURE,
    OUTCOME_OK,
    OUTCOME_PROVIDER_ERROR,
    OUTCOME_RATE_LIMITED,
    OUTCOME_TIMEOUT,
    OUTCOME_UNKNOWN,
    build_quota_observation,
    classify_outcome,
)

_PROVIDER = "openrouter"
_TOKEN_A = "synthetic-token-AAAA-not-a-real-key"
_TOKEN_B = "synthetic-token-BBBB-not-a-real-key"
_SCOPE_A = "a" * 32
_SCOPE_B = "b" * 32


def _entry(entry_id, token, *, scope_id, priority=0):
    return PooledCredential.from_dict(_PROVIDER, {
        "id": entry_id, "label": entry_id, "auth_type": "api_key",
        "priority": priority, "access_token": token, "source": "manual", "scope_id": scope_id,
    })


def _pool():
    return CredentialPool(provider=_PROVIDER, entries=[
        _entry("entry-a", _TOKEN_A, scope_id=_SCOPE_A, priority=0),
        _entry("entry-b", _TOKEN_B, scope_id=_SCOPE_B, priority=1),
    ])


class _Agent:
    def __init__(self, pool, entry_id, api_key):
        self.provider = _PROVIDER
        self._credential_pool = pool
        self._credential_pool_entry_id = entry_id
        self.api_key = api_key


class _Headers(dict):
    def items(self):
        return super().items()


def _response(status=200, headers=None):
    return SimpleNamespace(status_code=status, headers=_Headers(headers or {}))


# ── outcome classification ────────────────────────────────────────────────────

class TestClassifyOutcome:
    def test_ok(self):
        assert classify_outcome(200) == OUTCOME_OK

    def test_auth_failures(self):
        assert classify_outcome(401) == OUTCOME_AUTH_FAILURE
        assert classify_outcome(403) == OUTCOME_AUTH_FAILURE

    def test_rate_limited_is_not_quota_exhausted(self):
        # 429 maps to rate_limited; the taxonomy deliberately has no exhausted class.
        assert classify_outcome(429) == OUTCOME_RATE_LIMITED
        assert "exhausted" not in OUTCOME_RATE_LIMITED

    def test_provider_error(self):
        assert classify_outcome(500) == OUTCOME_PROVIDER_ERROR
        assert classify_outcome(503) == OUTCOME_PROVIDER_ERROR

    def test_timeout_and_unknown(self):
        assert classify_outcome(None, "timeout") == OUTCOME_TIMEOUT
        assert classify_outcome(None) == OUTCOME_UNKNOWN
        assert classify_outcome(418) == OUTCOME_UNKNOWN


# ── observation building ─────────────────────────────────────────────────────

class TestObservation:
    def test_200_with_rate_limit_headers_captures_windows(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        observation = build_quota_observation(agent, http_response=_response(200, {
            "x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "87",
            "x-ratelimit-reset-requests": "42",
            "x-ratelimit-remaining-tokens": "999",
        }), api_request_id="req-1")
        assert observation["outcome"] == OUTCOME_OK
        assert observation["status_code"] == 200
        assert observation["identity_status"] == "known"
        assert observation["credential_scope_id"] == _SCOPE_A
        windows = observation["rate_limit"]
        assert windows["requests_min"] == {"limit": 100, "remaining": 87, "reset_seconds": 42.0}
        # Partial window: provider sent only the remaining header — kept, never zero-filled.
        assert windows["tokens_min"] == {"limit": None, "remaining": 999, "reset_seconds": None}
        # Window the provider did not send at all is absent.
        assert "requests_hour" not in windows
        assert "tokens_hour" not in windows

    def test_200_without_headers_claims_nothing(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        observation = build_quota_observation(agent, http_response=_response(200), api_request_id="req-1")
        assert observation["outcome"] == OUTCOME_OK
        assert observation["rate_limit"] is None
        assert observation["retry_after_seconds"] is None

    def test_zero_remaining_headers_do_not_imply_exhaustion(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        observation = build_quota_observation(agent, http_response=_response(200, {
            "x-ratelimit-limit-requests-1h": "1000", "x-ratelimit-remaining-requests-1h": "0",
        }), api_request_id="req-1")
        assert observation["outcome"] == OUTCOME_OK
        assert observation["rate_limit"]["requests_hour"]["remaining"] == 0

    def test_429_carries_retry_after_seconds(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        observation = build_quota_observation(agent, http_response=_response(429, {"Retry-After": "30"}),
                                              status_code=429, api_request_id="req-1")
        assert observation["outcome"] == OUTCOME_RATE_LIMITED
        assert observation["retry_after_seconds"] == 30.0

    def test_429_without_retry_after_reports_unknown(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        observation = build_quota_observation(agent, http_response=_response(429), status_code=429,
                                              api_request_id="req-1")
        assert observation["outcome"] == OUTCOME_RATE_LIMITED
        assert observation["retry_after_seconds"] is None

    def test_retry_after_http_date_is_parsed(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        reset = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() + 90))
        observation = build_quota_observation(agent, http_response=_response(429, {"retry-after": reset}),
                                              status_code=429, api_request_id="req-1")
        assert 0.0 <= observation["retry_after_seconds"] <= 90.0 + 2.0

    def test_retry_after_garbage_and_far_future(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        garbage = build_quota_observation(agent, http_response=_response(429, {"retry-after": "soon"}),
                                          status_code=429, api_request_id="req-1")
        assert garbage["retry_after_seconds"] is None
        far = build_quota_observation(agent, http_response=_response(429, {"retry-after": "99999999"}),
                                      status_code=429, api_request_id="req-1")
        assert far["retry_after_seconds"] <= 7 * 24 * 3600.0

    def test_unsupported_header_meaning_is_passed_through_not_interpreted(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        observation = build_quota_observation(agent, http_response=_response(200, {
            "x-ratelimit-limit-tokens": "not-a-number", "x-ratelimit-remaining-tokens": "12",
        }), api_request_id="req-1")
        assert observation["outcome"] == OUTCOME_OK
        assert observation["rate_limit"]["tokens_min"]["limit"] is None
        assert observation["rate_limit"]["tokens_min"]["remaining"] == 12

    @pytest.mark.parametrize("invalid", ("nan", "inf", "-inf", "1e309", "-2", "1.5"))
    def test_untrusted_rate_limit_counts_are_unknown_without_dropping_observation(self, invalid):
        """Malformed provider counters never raise, truncate, or erase sound fields."""
        agent = _Agent(_pool(), "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-bad-count")
        observed = build_quota_observation(agent, http_response=_response(429, {
            "x-ratelimit-limit-requests": invalid,
            "x-ratelimit-remaining-requests": invalid,
            "x-ratelimit-reset-requests": "22",
            "x-ratelimit-remaining-tokens": "10",
        }), api_request_id="req-bad-count")
        assert observed["outcome"] == OUTCOME_RATE_LIMITED
        assert observed["credential_scope_id"] == _SCOPE_A
        assert observed["rate_limit"]["requests_min"] == {
            "limit": None, "remaining": None, "reset_seconds": 22.0,
        }
        assert observed["rate_limit"]["tokens_min"]["remaining"] == 10

    @pytest.mark.parametrize("invalid", ("nan", "inf", "-inf", "1e309", "-5"))
    def test_untrusted_reset_and_retry_after_are_unknown(self, invalid):
        """A malformed wait never becomes a fabricated timeout or skips the hook."""
        agent = _Agent(_pool(), "entry-a", _TOKEN_A)
        observed = build_quota_observation(agent, http_response=_response(429, {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "20",
            "x-ratelimit-reset-requests": invalid,
            "Retry-After": invalid,
        }))
        assert observed["outcome"] == OUTCOME_RATE_LIMITED
        assert observed["rate_limit"]["requests_min"] == {
            "limit": 100, "remaining": 20, "reset_seconds": None,
        }
        assert observed["retry_after_seconds"] is None

    def test_provider_errors_and_timeout(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        assert build_quota_observation(agent, status_code=503, api_request_id="req-1")["outcome"] == OUTCOME_PROVIDER_ERROR
        assert build_quota_observation(agent, status_code=401, api_request_id="req-1")["outcome"] == OUTCOME_AUTH_FAILURE
        assert build_quota_observation(agent, reason="timeout", api_request_id="req-1")["outcome"] == OUTCOME_TIMEOUT

    def test_observation_binds_to_the_attempt_scope_across_rotation(self):
        """Retry rotating to B must not re-attribute attempt 1's observation."""
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        agent._credential_pool_entry_id, agent.api_key = "entry-b", _TOKEN_B
        obs = build_quota_observation(agent, status_code=429, api_request_id="req-1")
        assert obs["credential_scope_id"] == _SCOPE_A
        capture_attempt_credential_scope(agent, "req-2")
        obs2 = build_quota_observation(agent, status_code=429, api_request_id="req-2")
        assert obs2["credential_scope_id"] == _SCOPE_B

    def test_observation_carries_no_secret(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        capture_attempt_credential_scope(agent, "req-1")
        obs = build_quota_observation(agent, http_response=_response(200, {"x": "y"}), api_request_id="req-1")
        encoded = json.dumps(obs)
        assert _TOKEN_A not in encoded and _TOKEN_B not in encoded

    def test_missing_pool_means_unknown_scope(self):
        agent = _Agent(None, None, "synthetic-key")
        capture_attempt_credential_scope(agent, "req-1")
        obs = build_quota_observation(agent, status_code=401, api_request_id="req-1")
        assert obs["identity_status"] == "unknown"
        assert obs["credential_scope_id"] is None


# ── capability-gated hook field ──────────────────────────────────────────────

@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _register(manager, plugin_id, sink):
    from hermes_cli.plugins import PluginContext, PluginManifest

    manifest = PluginManifest(name=plugin_id, source="user", key=plugin_id)
    ctx = PluginContext(manifest, manager)

    def _cb(**kwargs):
        sink.append(kwargs)

    ctx.register_hook("post_api_request", _cb)
    return _cb


def _observation_payload():
    return {"schema_version": 1, "provider_id": _PROVIDER, "credential_scope_id": _SCOPE_A,
            "identity_status": "known", "observed_at": 0.0, "outcome": "ok",
            "status_code": 200, "retry_after_seconds": None, "rate_limit": None}


class TestObservationGate:
    def test_quota_observation_capability_is_consent_only(self):
        from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY

        spec = CAPABILITY_REGISTRY["llm.quota_observation"]
        assert spec.legacy_path == ()
        assert spec.description

    def test_granted_plugin_receives_observation_ungranted_does_not(self, hermes_home):
        from hermes_cli.plugin_capabilities import record_consent
        from hermes_cli.plugins import PluginManager

        record_consent("obs-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        manager = PluginManager()
        granted, denied = [], []
        _register(manager, "obs-plugin", granted)
        _register(manager, "silent-plugin", denied)
        manager.invoke_hook("post_api_request", provider=_PROVIDER, api_request_id="req-1",
                            quota_observation=_observation_payload())
        assert granted and granted[0]["quota_observation"]["outcome"] == "ok"
        assert denied and "quota_observation" not in denied[0]
        assert denied[0]["provider"] == _PROVIDER

    def test_scope_and_observation_gate_independently(self, hermes_home):
        from hermes_cli.plugin_capabilities import record_consent
        from hermes_cli.plugins import PluginManager

        record_consent("scope-only", ["llm.credential_scope"], ["llm.credential_scope"])
        manager = PluginManager()
        seen = []
        _register(manager, "scope-only", seen)
        manager.invoke_hook("post_api_request", provider=_PROVIDER, api_request_id="req-1",
                            credential_scope={"provider_id": _PROVIDER, "credential_scope_id": _SCOPE_A,
                                              "identity_status": "known"},
                            quota_observation=_observation_payload())
        assert seen and "credential_scope" in seen[0]
        assert "quota_observation" not in seen[0]

    def test_ownerless_callback_is_fail_closed(self, hermes_home):
        from hermes_cli.plugins import PluginManager

        manager = PluginManager()
        seen = []
        manager._hooks.setdefault("post_api_request", []).append(lambda **kw: seen.append(kw))
        manager.invoke_hook("post_api_request", provider=_PROVIDER, api_request_id="req-1",
                            quota_observation=_observation_payload())
        assert seen and "quota_observation" not in seen[0]

    def test_malformed_capstone_payload_still_delivers_or_strips_cleanly(self, hermes_home):
        """The gate never crashes on hostile payload shapes."""
        from hermes_cli.plugin_capabilities import record_consent
        from hermes_cli.plugins import PluginManager

        record_consent("hostile-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        manager = PluginManager()
        seen = []
        _register(manager, "hostile-plugin", seen)
        manager.invoke_hook("post_api_request", provider=_PROVIDER, api_request_id="req-1",
                            quota_observation={"not": "an observation"})
        assert seen and seen[0]["quota_observation"] == {"not": "an observation"}  # pass-through, never crash


# ── ctx.quota_fetch ──────────────────────────────────────────────────────────

def _write_pool(home, rows):
    (home / "auth.json").write_text(json.dumps({"credential_pool": {_PROVIDER: rows}}), encoding="utf-8")


def _row(token, scope_id):
    return {"id": scope_id[:8], "label": "row", "auth_type": "api_key", "priority": 0,
            "access_token": token, "source": "manual", "scope_id": scope_id}


def _ctx(plugin_id="quota-plugin"):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manifest = PluginManifest(name=plugin_id, source="user", key=plugin_id)
    return PluginContext(manifest, PluginManager())


class TestQuotaFetch:
    def test_capability_not_granted_refuses_before_network(self, hermes_home, monkeypatch):
        calls = []
        monkeypatch.setattr("agent.quota_fetch._http_get_json",
                            lambda *a, **kw: calls.append((a, kw)) or (500, None))
        result = _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                    quota_source=OFFICIAL_KEY_LIMITS)
        assert result["status"] == "unavailable"
        assert result["reason"] == "capability_not_granted"
        assert calls == []

    def test_unsupported_provider_and_source_refuse_before_network(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        calls = []
        monkeypatch.setattr("agent.quota_fetch._http_get_json",
                            lambda *a, **kw: calls.append((a, kw)) or (500, None))
        ctx = _ctx()
        assert ctx.quota_fetch(provider_id="nvidia", credential_scope_id=_SCOPE_A,
                               quota_source=OFFICIAL_KEY_LIMITS)["reason"] == "unsupported_provider"
        assert ctx.quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                               quota_source="monthly")["reason"] == "unsupported_quota_source"
        assert calls == []

    def test_malformed_scope_refuses_before_network(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        calls = []
        monkeypatch.setattr("agent.quota_fetch._http_get_json",
                            lambda *a, **kw: calls.append((a, kw)) or (500, None))
        ctx = _ctx()
        for bad in ("zz" * 16, "A" * 32, "g" * 32, "", None):
            assert ctx.quota_fetch(provider_id="openrouter", credential_scope_id=bad,
                                   quota_source=OFFICIAL_KEY_LIMITS)["reason"] == "invalid_scope"
        assert calls == []

    def test_unknown_scope_refuses_before_network(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        _write_pool(hermes_home, [_row(_TOKEN_A, _SCOPE_A)])
        calls = []
        monkeypatch.setattr("agent.quota_fetch._http_get_json",
                            lambda *a, **kw: calls.append((a, kw)) or (500, None))
        result = _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_B,
                                    quota_source=OFFICIAL_KEY_LIMITS)
        assert result["reason"] == "scope_not_found"
        assert calls == []

    def test_scope_token_missing_refuses(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        row = _row(_TOKEN_A, _SCOPE_A)
        del row["access_token"]
        _write_pool(hermes_home, [row])
        monkeypatch.setattr("agent.quota_fetch._http_get_json", lambda *a, **kw: (500, None))
        assert _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                  quota_source=OFFICIAL_KEY_LIMITS)["reason"] == "scope_token_unavailable"

    def test_uses_only_the_scoped_credential_and_never_leaks_it(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        _write_pool(hermes_home, [_row(_TOKEN_A, _SCOPE_A), _row(_TOKEN_B, _SCOPE_B)])
        seen = {}

        def fake_http(url, token, timeout):
            seen["url"], seen["token"], seen["timeout"] = url, token, timeout
            return 200, {"data": {"free_model_daily_requests": {"limit": 100, "remaining": 40, "used": 60}}}

        monkeypatch.setattr("agent.quota_fetch._http_get_json", fake_http)
        now = time.time()
        result = _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_B,
                                    quota_source=OFFICIAL_KEY_LIMITS)
        assert seen["token"] == _TOKEN_B
        assert seen["url"] == "https://openrouter.ai/api/v1/key"
        assert seen["timeout"] <= 2.0
        assert result["status"] == "ok"
        assert (result["limit"], result["remaining"], result["used"]) == (100, 40, 60)
        assert result["remaining"] + result["used"] == result["limit"]
        assert result["provider_id"] == "openrouter"
        assert result["credential_scope_id"] == _SCOPE_B
        assert now <= result["observed_at"] < result["expires_at"] <= result["observed_at"] + 120.0 + 1e-6
        encoded = json.dumps(result)
        assert _TOKEN_A not in encoded and _TOKEN_B not in encoded

    def test_credential_rotation_scope_unavailable_after_pool_rewrite(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        _write_pool(hermes_home, [_row(_TOKEN_A, _SCOPE_A)])
        monkeypatch.setattr("agent.quota_fetch._http_get_json", lambda *a, **kw: (200, {"data": {"free_model_daily_requests": {"limit": 5, "remaining": 5, "used": 0}}}))
        assert _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                  quota_source=OFFICIAL_KEY_LIMITS)["status"] == "ok"
        # Pool rewritten without the entry (re-add mints a new scope): old scope dies with it.
        _write_pool(hermes_home, [_row(_TOKEN_A, "c" * 32)])
        assert _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                  quota_source=OFFICIAL_KEY_LIMITS)["reason"] == "scope_not_found"

    def test_provider_401_is_unavailable_not_fabrication(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        _write_pool(hermes_home, [_row(_TOKEN_A, _SCOPE_A)])
        monkeypatch.setattr("agent.quota_fetch._http_get_json", lambda *a, **kw: (401, None))
        assert _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                  quota_source=OFFICIAL_KEY_LIMITS)["status"] == "unavailable"

    def test_garbage_shape_is_unavailable(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        _write_pool(hermes_home, [_row(_TOKEN_A, _SCOPE_A)])
        for body in ({"data": {}}, {"data": {"free_model_daily_requests": {"limit": 3, "remaining": 9, "used": 0}}},
                     {"data": {"free_model_daily_requests": {"limit": "3", "remaining": 0, "used": 3}}},
                     "not-a-dict", None):
            monkeypatch.setattr("agent.quota_fetch._http_get_json", lambda *a, b=body, **kw: (200, b))
            assert _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                      quota_source=OFFICIAL_KEY_LIMITS)["status"] == "unavailable"

    def test_network_failure_is_unavailable(self, hermes_home, monkeypatch):
        from hermes_cli.plugin_capabilities import record_consent

        record_consent("quota-plugin", ["llm.quota_observation"], ["llm.quota_observation"])
        _write_pool(hermes_home, [_row(_TOKEN_A, _SCOPE_A)])

        def boom(*a, **kw):
            raise OSError("connection refused")

        monkeypatch.setattr("agent.quota_fetch._http_get_json", boom)
        result = _ctx().quota_fetch(provider_id="openrouter", credential_scope_id=_SCOPE_A,
                                    quota_source=OFFICIAL_KEY_LIMITS)
        assert result["status"] == "unavailable"
        assert "synthetic-token" not in json.dumps(result)
