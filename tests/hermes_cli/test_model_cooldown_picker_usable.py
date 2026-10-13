"""A model-scoped cooldown must not demote the whole provider (#127682).

A ``model_entitlement`` bench parks one (credential, model) pair for up to a
year, while the credential keeps serving every other model. Provider-level
gates that answer "can this credential serve anything" — the picker's pool
probe and the startup cooldown notice — read the unscoped ``has_available()``,
whose conservative "any cooldown blocks" answer collapsed the provider row to
its saved model (desktop picker) or dropped it entirely (bots picker) and
printed a year-long cooldown notice.

Real ``agent.credential_pool`` against a real temp auth store: one API-key
credential, one entitlement bench on ``MODEL_A``.
"""
import json
import time

import pytest

KEY = "sk-ant-api03-synthetic-test-key-0000"
MODEL_A = "claude-sonnet-4-5"
MODEL_B = "claude-haiku-4-5"


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "seat", "label": "seat", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": KEY,
    }]}}))
    from agent.credential_pool import load_pool
    fresh = load_pool("anthropic")
    assert fresh.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=KEY, failure_reason="model_entitlement", model=MODEL_A,
    ) is None  # a sole credential has nothing to rotate to
    return load_pool("anthropic")


def test_provider_level_probe_ignores_model_scoped_cooldown(pool):
    assert pool.has_credentials()
    # The routing default stays conservative: a benched model blocks every
    # unscoped route, so nothing hands this credential to an unnamed model.
    assert pool.has_available() is False
    assert pool.select() is None
    # The provider-level question — "can this credential serve anything" — is
    # not answered by a cooldown that concerns one model.
    assert pool.has_available(any_model=True) is True
    # Naming the benched model still fails; naming any other model still works.
    assert pool.has_available(model=MODEL_A) is False
    assert pool.has_available(model=MODEL_B) is True


def test_credential_wide_exhaustion_still_benches_the_provider(pool):
    assert pool.mark_exhausted_and_rotate(
        status_code=429, api_key_hint=KEY, failure_reason="rate_limit",
    ) is None
    assert pool.has_available(any_model=True) is False


def test_picker_pool_probe_keeps_the_provider_selectable(pool):
    from hermes_cli.model_switch_providers import _credential_pool_is_usable

    assert _credential_pool_is_usable("anthropic") is True


def test_startup_notice_silent_under_model_only_cooldown(monkeypatch):
    """A model-only cooldown must not print a "cooling down" startup notice.

    The notice derives its duration from ``next_available_at``, which counts
    the 365-day entitlement bench — on main the notice claims the provider is
    cooling down for a year while every unbenched model keeps working."""
    import agent.credential_pool as cp
    from hermes_cli.cli_agent_setup_mixin import _credential_pool_notice

    class _ModelOnlyCooldownPool:
        def has_credentials(self) -> bool:
            return True

        def has_available(self, *, model=None, any_model=False) -> bool:
            return any_model

        def next_available_at(self, **_kwargs):
            return time.time() + 365 * 24 * 3600

        def entries(self):
            return []

    monkeypatch.setattr(cp, "load_pool", lambda _provider: _ModelOnlyCooldownPool())
    cooling, lines = _credential_pool_notice("anthropic")
    assert cooling is False and lines == []
