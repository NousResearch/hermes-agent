"""Multiplexed-gateway invariants: per-turn config, credentials and hooks follow the ROUTED profile
(HERMES_HOME override), not the launch home the module constants were frozen from."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home A (HERMES_HOME) and a routed profile B with different config everywhere."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    for home, turns, model in ((a, 7, "A/fallback"), (b, 99, "B/fallback")):
        (home / "config.yaml").write_text(yaml.safe_dump({
            "agent": {"max_turns": turns},
            "fallback_providers": [{"provider": "openrouter", "model": model}],
        }), encoding="utf-8")
    return a, b


def _under(home: Path, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_max_turns_and_fallback_chain_follow_routed_profile(two_homes, monkeypatch):
    a, b = two_homes
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    monkeypatch.setattr(gateway_run, "_hermes_home", a)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    runner = SimpleNamespace(_fallback_model=None)
    refresh = GatewayRunner._refresh_fallback_model.__get__(runner)

    # Default profile turn warms the process-wide bridge/slot with A's values...
    assert gateway_run._current_max_iterations() == 7
    assert refresh() == [{"provider": "openrouter", "model": "A/fallback"}]
    # ...and a routed turn for B must still see B's config, not the launch home's.
    assert _under(b, gateway_run._current_max_iterations) == 99
    assert _under(b, refresh) == [{"provider": "openrouter", "model": "B/fallback"}]
    # Back on the default profile the chain is A's again (per-home last-known-good, not last writer).
    assert refresh() == [{"provider": "openrouter", "model": "A/fallback"}]


def test_aux_nous_auth_reads_routed_profile_auth_json(two_homes, monkeypatch):
    a, b = two_homes
    import agent.auxiliary_client as aux

    for home, token in ((a, "TOKEN_A"), (b, "TOKEN_B")):
        (home / "auth.json").write_text(json.dumps({
            "version": 1, "active_provider": "nous",
            "providers": {"nous": {"agent_key": token, "access_token": token}},
        }), encoding="utf-8")
    monkeypatch.setattr(aux, "_select_pool_entry", lambda _provider: (False, None))

    assert (aux._read_nous_auth() or {}).get("access_token") == "TOKEN_A"
    assert (_under(b, aux._read_nous_auth) or {}).get("access_token") == "TOKEN_B"


def _write_hook(home: Path, name: str) -> None:
    hook_dir = home / "hooks" / name
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text(f"name: {name}\nevents: ['agent:start']\n", encoding="utf-8")
    (hook_dir / "handler.py").write_text(
        "def handle(event_type, context):\n    context.setdefault('seen', []).append(__name__)\n",
        encoding="utf-8")


@pytest.mark.asyncio
async def test_gateway_hooks_fire_per_routed_profile(two_homes):
    """Profile B's hooks/ runs for B's turns and A's handlers never see B's context (and vice versa)."""
    a, b = two_homes
    _write_hook(a, "hook-a")
    _write_hook(b, "hook-b")
    from gateway.hooks import ProfileHookRegistries

    hooks = ProfileHookRegistries()
    ctx_a: dict = {}
    await hooks.emit("agent:start", ctx_a)
    token = set_hermes_home_override(str(b))
    try:
        ctx_b: dict = {}
        await hooks.emit("agent:start", ctx_b)
        assert [h["name"] for h in hooks.loaded_hooks] == ["hook-b"]
    finally:
        reset_hermes_home_override(token)
    assert ctx_a.get("seen") == ["hermes_hook_hook-a"]
    assert ctx_b.get("seen") == ["hermes_hook_hook-b"]


def test_media_policy_reads_routed_profile_config_not_env(two_homes, monkeypatch):
    a, b = two_homes
    from gateway import media_policy

    (a / "config.yaml").write_text(yaml.safe_dump(
        {"gateway": {"strict": True, "media_delivery_allow_dirs": ["/srv/a"]}}), encoding="utf-8")
    (b / "config.yaml").write_text(yaml.safe_dump(
        {"gateway": {"strict": False, "media_delivery_allow_dirs": ["/srv/b"]}}), encoding="utf-8")
    # Gateway startup bridges the LAUNCH profile's policy into the process env.
    for var in ("HERMES_MEDIA_DELIVERY_STRICT", "HERMES_MEDIA_ALLOW_DIRS"):
        monkeypatch.delenv(var, raising=False)
    from hermes_cli.config import load_config
    media_policy.apply_media_policy_env(load_config())
    assert media_policy.media_delivery_strict() is True
    assert media_policy.media_delivery_allow_dirs() == "/srv/a"

    assert _under(b, media_policy.media_delivery_strict) is False
    assert _under(b, media_policy.media_delivery_allow_dirs) == "/srv/b"
