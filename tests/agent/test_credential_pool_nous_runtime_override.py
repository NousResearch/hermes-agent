"""``PooledCredential.runtime_base_url`` must re-read NOUS_INFERENCE_BASE_URL on every use.

With the documented dev/staging override set, a Nous 401 recovery refreshed the token and
retried against the stored production host instead of the override: the rotated bearer was
sent to a host the operator never chose, and any non-production gateway's turn failed with a
connection error. The stored ``inference_base_url`` cannot carry the override anyway — the
network-provenance allowlist heals that field back to production — so the only trustworthy
source is the profile-scoped env resolver, consulted at read time, exactly like the
openai-codex branch below it treats HERMES_CODEX_BASE_URL.
"""

from agent.credential_pool import PooledCredential

PRODUCTION = "https://inference-api.nousresearch.com/v1"


def _nous_entry(**kwargs) -> PooledCredential:
    defaults = dict(
        provider="nous",
        id="nous-1",
        label="nous",
        auth_type="oauth",
        priority=0,
        source="manual",
        access_token="token",
    )
    defaults.update(kwargs)
    return PooledCredential(**defaults)


def test_override_wins_over_healed_production_inference_url(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "http://127.0.0.1:8443/v1")
    entry = _nous_entry(inference_base_url=PRODUCTION, base_url=PRODUCTION)
    assert entry.runtime_base_url == "http://127.0.0.1:8443/v1"


def test_without_override_inference_base_url_wins(monkeypatch):
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    entry = _nous_entry(inference_base_url=PRODUCTION, base_url="https://fallback.example/v1")
    assert entry.runtime_base_url == PRODUCTION


def test_without_override_or_inference_falls_back_to_base_url(monkeypatch):
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    entry = _nous_entry(base_url="https://fallback.example/v1")
    assert entry.runtime_base_url == "https://fallback.example/v1"


def test_override_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "http://127.0.0.1:8443/v1/")
    entry = _nous_entry(inference_base_url=PRODUCTION)
    assert entry.runtime_base_url == "http://127.0.0.1:8443/v1"


def test_blank_override_falls_back_to_row_urls(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "   ")
    entry = _nous_entry(inference_base_url=PRODUCTION)
    assert entry.runtime_base_url == PRODUCTION


def test_other_providers_still_return_base_url(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "http://127.0.0.1:8443/v1")
    entry = PooledCredential(
        provider="openai",
        id="oa-1",
        label="openai",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-x",
        base_url="https://api.openai.com/v1",
    )
    assert entry.runtime_base_url == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# The 401 recovery boundary (#121323 / review on #121339): the proof is not the
# getter's string precedence but what _swap_credential binds as the retry
# destination — the freshly rotated bearer must never be re-attached to the
# persisted production host while the operator's override is set.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

# Dummy fixture value (never a real credential): what a 401 refresh mints.
_ROTATED = "dummy-rotated-bearer"

STAGING_OVERRIDE = "https://nous-staging.example/v1"


def _swap_agent(*, api_key: str = "token", base_url: str = PRODUCTION):
    """Minimal agent with the REAL ``AIAgent._swap_credential`` bound and the
    rebuild hooks recorded, so the test observes the exact client-rebuild inputs."""
    from run_agent import AIAgent

    agent = SimpleNamespace(
        provider="nous", model="Hermes-4-405B", api_mode="chat_completions",
        api_key=api_key, base_url=base_url,
        _client_kwargs={"api_key": api_key, "base_url": base_url},
        _credential_pool_entry_id="nous-1",
        _is_entitlement_failure=lambda *a, **k: False,
        _reapply_route_client_config=MagicMock(),
        _replace_primary_openai_client=MagicMock(),
    )
    agent._swap_credential = AIAgent._swap_credential.__get__(agent)
    return agent


def _pool_returning(entry) -> MagicMock:
    pool = MagicMock()
    pool.provider = "nous"
    pool.try_refresh_matching.return_value = entry
    return pool


def test_401_recovery_rebinds_rotated_bearer_to_override_not_production(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", STAGING_OVERRIDE)
    from agent.agent_runtime_helpers import recover_with_credential_pool

    rotated = _nous_entry(inference_base_url=PRODUCTION, base_url=PRODUCTION,
                          access_token=_ROTATED)
    agent = _swap_agent()
    agent._credential_pool = _pool_returning(rotated)

    recovered, _ = recover_with_credential_pool(agent, status_code=401, has_retried_429=False)

    assert recovered is True
    agent._credential_pool.try_refresh_matching.assert_called_once()
    # Retry destination and bearer: the client rebuild consumes exactly these.
    assert agent.api_key == _ROTATED
    assert agent.base_url == STAGING_OVERRIDE
    assert agent._client_kwargs["api_key"] == _ROTATED
    assert agent._client_kwargs["base_url"] == STAGING_OVERRIDE
    agent._replace_primary_openai_client.assert_called_once_with(reason="credential_rotation")
    assert PRODUCTION not in (agent.base_url, agent._client_kwargs["base_url"])


def test_401_recovery_without_override_still_uses_stored_url(monkeypatch):
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    from agent.agent_runtime_helpers import recover_with_credential_pool

    rotated = _nous_entry(inference_base_url=PRODUCTION, base_url=PRODUCTION,
                          access_token=_ROTATED)
    agent = _swap_agent()
    agent._credential_pool = _pool_returning(rotated)

    recovered, _ = recover_with_credential_pool(agent, status_code=401, has_retried_429=False)

    assert recovered is True
    assert agent.api_key == _ROTATED
    assert agent.base_url == PRODUCTION


def test_401_recovery_lost_scope_does_not_inherit_launch_profile_override(monkeypatch):
    """Multiplexed call that lost its profile scope: the ambient (launch profile's)
    process-wide override must not reach the retry destination — the resolver is
    simply absent, so the rotation rebinds to the credential's stored route."""
    from agent import secret_scope

    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://launch-profile.example/v1")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(None)
    try:
        from agent.agent_runtime_helpers import recover_with_credential_pool

        rotated = _nous_entry(inference_base_url=PRODUCTION, base_url=PRODUCTION,
                              access_token=_ROTATED)
        agent = _swap_agent()
        agent._credential_pool = _pool_returning(rotated)

        recovered, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False)

        assert recovered is True
        assert agent.base_url == PRODUCTION
        assert "launch-profile.example" not in agent.base_url
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_runtime_base_url_follows_the_scoped_profile_a_b_a(monkeypatch):
    """A → B → A profile switches on one live entry: each read takes the ACTIVE
    profile's override, never the launch profile's process-wide value."""
    from agent import secret_scope

    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://launch-profile.example/v1")
    entry = _nous_entry(inference_base_url=PRODUCTION, base_url=PRODUCTION)
    secret_scope.set_multiplex_active(True)
    try:
        for scoped_url in (
            "https://profile-a.example/v1",
            "https://profile-b.example/v1",
            "https://profile-a.example/v1",
        ):
            token = secret_scope.set_secret_scope({"NOUS_INFERENCE_BASE_URL": scoped_url})
            try:
                assert entry.runtime_base_url == scoped_url
            finally:
                secret_scope.reset_secret_scope(token)
    finally:
        secret_scope.set_multiplex_active(False)
