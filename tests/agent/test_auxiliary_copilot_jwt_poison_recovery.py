"""A Copilot JWT GitHub revoked before its stored ``expires_at`` must self-heal, not poison
every call for the process lifetime (#135645).

The aux client saw a bare ``403 forbidden`` from ``*.githubcopilot.com`` (Copilot Business,
``api.business.githubcopilot.com``) whenever GitHub invalidated the cached exchanged JWT early.
``_is_auth_error`` only knows 401 (+xAI's ``bad-credentials`` 403), so no refresh fired; and even
the 401 refresh path was defeated: ``_refresh_copilot_credentials`` popped only the in-process
``_jwt_cache``, while ``_exchange_copilot_token_locked``'s disk fast-path happily reloaded the
still-"fresh" poisoned entry from ``~/.hermes/.copilot_jwt.json``.

These tests pin the three recovery pieces: the 403 classification (stale JWT, not billing), the
host table row that maps Business subdomains to the copilot refresher, and the refresher's
evict-before-exchange ordering that defeats the disk fast-path.
"""

import time
from types import SimpleNamespace

import agent.auxiliary_client as aux
from agent.conversation_loop import _is_stale_copilot_credential_error
from hermes_cli import copilot_auth


class _ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _client_with_key(value):
    stub = SimpleNamespace()
    setattr(stub, "api_key", value)
    return stub


def test_bare_forbidden_403_classifies_as_stale_copilot_jwt():
    assert (
        aux._is_stale_copilot_jwt_error(
            _ApiError("Error code: 403 - forbidden", status_code=403), "copilot"
        )
        is True
    )


def test_billing_worded_403_is_quota_not_credentials():
    assert (
        aux._is_stale_copilot_jwt_error(
            _ApiError("Error code: 403 - insufficient credits", status_code=403),
            "copilot",
        )
        is False
    )


def test_403_on_other_providers_stays_a_permission_denial():
    assert (
        aux._is_stale_copilot_jwt_error(
            _ApiError("Error code: 403 - forbidden", status_code=403), "openai"
        )
        is False
    )
    assert (
        aux._is_stale_copilot_jwt_error(
            _ApiError("Error code: 401 - unauthorized", status_code=401), "copilot"
        )
        is False
    )


def test_business_copilot_host_resolves_to_the_auth_refresher():
    # The refresh table previously pinned ``api.githubcopilot.com``, so the Business endpoint
    # (``api.business.githubcopilot.com``) never mapped to the copilot refresher on auto routes.
    for base in (
        "https://api.githubcopilot.com",
        "https://api.business.githubcopilot.com",
    ):
        assert (
            aux._provider_for_host(base, aux._AUTH_REFRESH_PROVIDER_BY_HOST)
            == "copilot"
        )
        assert aux._auth_refresh_provider_for_route("auto", base) == "copilot"


def test_aux_ladder_refreshes_copilot_after_forbidden_403(monkeypatch):
    events = []
    monkeypatch.setattr(
        copilot_auth, "resolve_copilot_token", lambda: ("ghu_raw", "env")
    )
    monkeypatch.setattr(
        copilot_auth,
        "evict_cached_exchanged_token",
        lambda raw: events.append(("evict", raw)),
    )

    def _exchange(raw, **kwargs):
        events.append(("exchange", raw))
        return ("tok_fresh_jwt", time.time() + 1800, None)

    monkeypatch.setattr(copilot_auth, "exchange_copilot_token", _exchange)

    route = SimpleNamespace(
        client=_client_with_key("tok_poisoned_jwt"),
        task="compression",
        tag="",
        resolved_provider="copilot",
        base_info="https://api.business.githubcopilot.com",
        resolved_model="gpt-5",
        final_model="gpt-5",
        main_runtime=None,
    )
    retry = aux._ladder_credential_rungs(
        _ApiError("Error code: 403 - forbidden", status_code=403), route, {}, False
    )
    # The credential rung fires (not the pool-rotation or fallback paths) and retries the same
    # provider with a freshly exchanged JWT.
    assert next(retry).kind == "retry_same_provider"
    retry.close()
    # Eviction (in-process + disk) must precede the exchange, or the disk fast-path in
    # ``_exchange_copilot_token_locked`` would reload the poisoned entry and defeat the refresh.
    assert events == [("evict", "ghu_raw"), ("exchange", "ghu_raw")]


def test_refresh_skips_eviction_when_no_raw_token(monkeypatch):
    monkeypatch.setattr(copilot_auth, "resolve_copilot_token", lambda: ("", "env"))

    def _forbidden(*args, **kwargs):
        raise AssertionError("evict/exchange must not run without a raw token")

    monkeypatch.setattr(copilot_auth, "evict_cached_exchanged_token", _forbidden)
    monkeypatch.setattr(copilot_auth, "exchange_copilot_token", _forbidden)
    assert aux._refresh_copilot_credentials() is False


def test_main_client_classifier_accepts_revoked_jwt_403():
    # The main-loop single-shot re-exchange trigger (``turn_api_error``) shares this classifier.
    assert _is_stale_copilot_credential_error(403, "forbidden") is True
    assert (
        _is_stale_copilot_credential_error(None, "Error code: 403 - forbidden") is True
    )


def test_main_client_classifier_keeps_rejecting_non_stale_403s():
    assert _is_stale_copilot_credential_error(403, "insufficient credits") is False
    assert (
        _is_stale_copilot_credential_error(403, "access to this model is denied")
        is False
    )


def test_main_client_classifier_keeps_the_400_integrator_markers():
    assert (
        _is_stale_copilot_credential_error(
            400, "model_not_available_for_integrator: this model is not available"
        )
        is True
    )
    assert _is_stale_copilot_credential_error(400, "unknown parameter") is False
