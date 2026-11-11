"""Failed runtime bearers renew their dispatched pool row, never an ambient login."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import providers
from providers.base import ProviderProfile
from agent.credential_auth_recovery import recover_auth_failure


class ScopedProfile(ProviderProfile):
    def refresh_pool_runtime_credentials(self, entry, *, reason):
        return None

    def refresh_runtime_credentials(self, *, reason):
        raise AssertionError("ambient login must not refresh a pooled request")

    def is_stale_credential_error(self, *, status_code, message):
        return status_code == 400


@pytest.fixture
def scoped(monkeypatch):
    profile = ScopedProfile(name="scoped", base_url="https://api.scoped.example/v1")
    monkeypatch.setattr(
        providers,
        "get_provider_profile",
        lambda name: profile if name == "scoped" else None,
    )
    monkeypatch.setattr(providers, "profile_for_host", lambda *a, **k: None)
    renewed = SimpleNamespace(
        id="failed-row", runtime_api_key="renewed", runtime_base_url=profile.base_url
    )
    pool = SimpleNamespace(
        provider="scoped",
        current=Mock(
            return_value=SimpleNamespace(id="healthy-row", runtime_api_key="healthy")
        ),
        try_refresh_matching=Mock(return_value=None),
        try_refresh_runtime_matching=Mock(return_value=renewed),
        has_credentials=Mock(return_value=True),
        entries=Mock(return_value=[]),
    )
    agent = SimpleNamespace(
        provider="scoped",
        api_mode="chat_completions",
        api_key="rejected",
        base_url=profile.base_url,
        _credential_pool_entry_id="failed-row",
        _credential_pool=pool,
        _swap_credential=Mock(return_value=True),
        _buffer_vprint=Mock(),
        _is_entitlement_failure=Mock(return_value=False),
        _try_refresh_provider_runtime_credentials=Mock(return_value=True),
    )
    return agent, pool, renewed


def auth_recover(agent, pool, rotate):
    return recover_auth_failure(
        agent,
        pool,
        status_code=401,
        has_retried_429=True,
        error_context={},
        api_key_hint="rejected",
        credential_id="failed-row",
        rotate_and_swap=rotate,
    )


def test_401_renews_dispatched_row_not_current_or_oauth_peer(scoped):
    agent, pool, renewed = scoped
    rotate = Mock(return_value=False)
    assert auth_recover(agent, pool, rotate) == (True, True)
    pool.try_refresh_runtime_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected", reason="unauthorized"
    )
    pool.try_refresh_matching.assert_not_called()
    pool.current.assert_not_called()
    rotate.assert_not_called()
    agent._swap_credential.assert_called_once_with(renewed)


@pytest.mark.parametrize("rotated", [False, True])
def test_failed_401_renewal_rotates_then_never_uses_ambient(scoped, rotated):
    from agent.turn_recovery import _refresh_credentials_after_401
    from agent.turn_retry_state import TurnRetryState

    agent, pool, _ = scoped
    pool.try_refresh_runtime_matching.return_value = None
    rotate = Mock(return_value=rotated)
    assert auth_recover(agent, pool, rotate) == (rotated, False if rotated else True)
    rotate.assert_called_once_with(401, "auth refresh failed")
    if not rotated:
        assert not _refresh_credentials_after_401(
            agent, Exception("401"), TurnRetryState(), 401
        )
    agent._try_refresh_provider_runtime_credentials.assert_not_called()


class StaleError(Exception):
    status_code = 400
    body = {"error": {"message": "Unsupported parameter: stale bearer"}}


def settle_stale(agent):
    from agent.error_classifier import classify_api_error
    from agent.turn_api_error import settle_unrecovered_error
    from agent.turn_retry_state import TurnRetryState

    err = StaleError("Unsupported parameter: stale bearer")
    return settle_unrecovered_error(
        agent,
        api_error=err,
        classified=classify_api_error(err, provider=agent.provider),
        _retry=TurnRetryState(),
        status_code=400,
        error_msg=str(err),
        is_context_length_error=False,
        is_rate_limited=False,
        _is_zai_coding_overload=False,
        _provider=agent.provider,
        _base=agent.base_url,
        _model="m",
        messages=[],
        api_messages=[],
        api_kwargs={},
        active_system_prompt="",
        conversation_history=None,
        approx_tokens=10,
        retry_count=1,
        max_retries=3,
        compression_attempts=0,
        api_call_count=1,
    )


def test_stale_400_renews_exact_row_without_benching_rotating_or_ambient(scoped):
    agent, pool, renewed = scoped
    pool.mark_exhausted_and_rotate = Mock()
    result = settle_stale(agent)
    assert result.action == "continue"
    assert result.retry_count == 0
    pool.try_refresh_runtime_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected", reason="stale"
    )
    pool.mark_exhausted_and_rotate.assert_not_called()
    pool.try_refresh_matching.assert_not_called()
    agent._try_refresh_provider_runtime_credentials.assert_not_called()
    agent._swap_credential.assert_called_once_with(renewed)


class AuthError(Exception):
    status_code = 401


def test_auxiliary_pool_renews_sender_row_not_shared_current(scoped, monkeypatch):
    from agent import auxiliary_client as aux

    agent, pool, _ = scoped
    pool.try_refresh_current = Mock(return_value=None)
    pool.mark_exhausted_and_rotate = Mock(return_value=None)
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    monkeypatch.setattr(aux, "_evict_cached_clients", Mock())
    sender = SimpleNamespace(
        api_key="rejected",
        base_url=agent.base_url,
        _hermes_credential_pool_entry_id="failed-row",
    )
    route = SimpleNamespace(
        resolved_provider="scoped", client=sender, main_runtime=None
    )
    failure = aux.aux_errors.Failure(
        AuthError("refused"),
        aux.aux_errors.classify_attempt(AuthError("refused"), "scoped"),
        client=sender,
    )
    assert aux._recover_failure_pool(route, failure)
    pool.try_refresh_runtime_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected", reason="unauthorized"
    )
    pool.try_refresh_current.assert_not_called()
    pool.mark_exhausted_and_rotate.assert_not_called()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_plugin_oauth_client_carries_dispatched_row_identity(
    scoped, monkeypatch, explicit, async_mode
):
    from agent import auxiliary_client as aux
    from agent.auxiliary_plugin_oauth import resolve_plugin_oauth_client

    agent, pool, _ = scoped
    row = SimpleNamespace(
        id="failed-row", runtime_api_key="rejected", runtime_base_url=agent.base_url
    )
    pool.entries.return_value = [row]
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *a: (True, row))
    monkeypatch.setattr(
        aux,
        "_api_key_profile_supplied_client",
        lambda *a, **k: SimpleNamespace(api_key=k["api_key"]),
    )
    monkeypatch.setattr(
        aux,
        "_route_client",
        lambda req, c, m: (
            (SimpleNamespace(api_key=c.api_key), m) if req.async_mode else (c, m)
        ),
    )
    req = SimpleNamespace(
        provider="scoped",
        model="m",
        explicit_api_key="rejected" if explicit else None,
        explicit_base_url=agent.base_url,
        async_mode=async_mode,
    )
    client, model = resolve_plugin_oauth_client(req)
    assert model == "m"
    assert getattr(client, "_hermes_credential_pool_entry_id", None) == "failed-row"


def test_auxiliary_auth_ladder_skips_ambient_refresh_before_pool(scoped, monkeypatch):
    from agent import auxiliary_client as aux

    agent, pool, _ = scoped
    ambient = Mock(return_value=None)
    monkeypatch.setattr(
        providers.get_provider_profile("scoped"), "refresh_runtime_credentials", ambient
    )
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    monkeypatch.setattr(aux, "_evict_cached_clients", Mock())
    sender = SimpleNamespace(
        api_key="rejected",
        base_url=agent.base_url,
        _hermes_credential_pool_entry_id="failed-row",
    )
    route = SimpleNamespace(
        resolved_provider="scoped",
        client=sender,
        main_runtime=None,
        task="compression",
        tag="",
        base_info=agent.base_url,
        resolved_model="m",
        final_model="m",
    )
    err = AuthError("refused")
    failure = aux.aux_errors.Failure(
        err, aux.aux_errors.classify_attempt(err, "scoped"), client=sender
    )
    ladder = aux._ladder_credential_rungs(failure, route, {}, False)
    step = next(ladder)
    assert step.kind == "retry_same_provider"
    ambient.assert_not_called()
    pool.try_refresh_runtime_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected", reason="unauthorized"
    )
    ladder.close()


@pytest.mark.parametrize("rotated", [False, True])
def test_auxiliary_declined_renewal_benches_only_sender_row(
    scoped, monkeypatch, rotated
):
    from agent import auxiliary_client as aux

    agent, pool, renewed = scoped
    pool.try_refresh_runtime_matching.return_value = None
    pool.mark_exhausted_and_rotate = Mock(return_value=renewed if rotated else None)
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    monkeypatch.setattr(aux, "_evict_cached_clients", Mock())
    sender = SimpleNamespace(
        api_key="rejected",
        base_url=agent.base_url,
        _hermes_credential_pool_entry_id="failed-row",
    )
    route = SimpleNamespace(
        resolved_provider="scoped", client=sender, main_runtime=None
    )
    err = AuthError("refused")
    failure = aux.aux_errors.Failure(
        err, aux.aux_errors.classify_attempt(err, "scoped"), client=sender
    )
    assert aux._recover_failure_pool(route, failure) is rotated
    assert (
        pool.mark_exhausted_and_rotate.call_args.kwargs["credential_id"] == "failed-row"
    )
    assert pool.mark_exhausted_and_rotate.call_args.kwargs["api_key_hint"] == "rejected"


@pytest.mark.parametrize("async_mode", [False, True])
def test_auxiliary_fallback_candidate_renews_its_dispatched_row(
    scoped, monkeypatch, async_mode
):
    from agent import auxiliary_client as aux

    agent, pool, _ = scoped
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    monkeypatch.setattr(aux, "_evict_cached_clients", Mock())
    monkeypatch.setattr(aux, "_quarantine_fallback_candidate", Mock())
    sender = SimpleNamespace(
        api_key="rejected",
        base_url=agent.base_url,
        _hermes_credential_pool_entry_id="failed-row",
    )
    replacement = SimpleNamespace(api_key="renewed", base_url=agent.base_url)
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **k: (replacement, "m"))
    destination = SimpleNamespace(
        provider="scoped",
        base_url=agent.base_url,
        model="m",
        api_mode="chat_completions",
    )
    ladder = aux._fallback_candidate_ladder(
        sender,
        {},
        destination,
        lambda *a: (destination, {}),
        task="compression",
        fb_label="scoped",
        tag="",
        async_mode=async_mode,
    )
    assert next(ladder).args[0] is sender
    err = AuthError("refused")
    failure = aux.aux_errors.Failure(
        err, aux.aux_errors.classify_attempt(err, "scoped"), client=sender
    )
    try:
        retry = ladder.send((None, failure, ()))
    except StopIteration:
        pytest.fail("scoped fallback credential was discarded instead of renewed")
    assert retry.args[0] is replacement
    pool.try_refresh_runtime_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected", reason="unauthorized"
    )
    ladder.close()


def test_runtime_renewal_does_not_loop_forever_on_repeated_401(scoped):
    agent, pool, _ = scoped
    rotate = Mock(return_value=False)
    results = [auth_recover(agent, pool, rotate)[0] for _ in range(3)]
    assert results == [True, True, False]


def test_genuine_oauth_keeps_matching_peer_refresh_without_runtime_hook(scoped):
    agent, pool, renewed = scoped
    pool.provider = agent.provider = "openai-codex"
    pool.try_refresh_matching.return_value = renewed
    assert auth_recover(agent, pool, Mock(return_value=False)) == (True, True)
    pool.try_refresh_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected"
    )
    pool.try_refresh_runtime_matching.assert_not_called()


def test_declined_stale_repair_never_resolves_ambient_or_benches(scoped, monkeypatch):
    from agent import turn_api_error as errors

    agent, pool, _ = scoped
    pool.try_refresh_runtime_matching.return_value = None
    pool.mark_exhausted_and_rotate = Mock()
    agent._has_pending_fallback = Mock(return_value=False)
    agent._try_activate_fallback = Mock(return_value=False)
    monkeypatch.setattr(errors, "settle_delivered_partial", lambda *a: None)
    monkeypatch.setattr(
        errors,
        "nonretryable_client_error_result",
        lambda *a, **k: {"error": "original"},
    )
    result = settle_stale(agent)
    assert result.action == "return"
    pool.try_refresh_runtime_matching.assert_called_once_with(
        credential_id="failed-row", api_key_hint="rejected", reason="stale"
    )
    pool.mark_exhausted_and_rotate.assert_not_called()
    pool.try_refresh_matching.assert_not_called()
    agent._try_refresh_provider_runtime_credentials.assert_not_called()
    agent._swap_credential.assert_not_called()


def test_other_provider_pool_does_not_block_legacy_401_refresh(scoped):
    from agent.turn_recovery import _refresh_credentials_after_401
    from agent.turn_retry_state import TurnRetryState

    agent, pool, _ = scoped
    pool.provider = "other"
    assert _refresh_credentials_after_401(
        agent, AuthError("refused"), TurnRetryState(), 401
    )
    agent._try_refresh_provider_runtime_credentials.assert_called_once_with(
        reason="unauthorized"
    )
    pool.try_refresh_runtime_matching.assert_not_called()
