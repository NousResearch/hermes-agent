"""Solstice invariants: a pre-release provider stays off every discovery surface until signed in, and a
brokered OAuth row with an opaque bearer rotates on its stored expiry."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.auth
import providers
from hermes_cli import auth_oauth_pkce_plugin as pkce
from providers import register_provider
from providers.base import ProviderProfile


def _pool_row(provider: str, *, access: str, expires_at_ms: int) -> dict:
    return {"id": "abc123", "label": provider, "auth_type": "oauth", "priority": 0, "source": pkce.POOL_SOURCE,
            "access_token": access, "refresh_token": "1//rt-0", "expires_at_ms": expires_at_ms}


def _write_pool(home, provider: str, row: dict) -> None:
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {provider: [row]}}))


def test_solstice_is_off_every_discovery_surface_until_signed_in_but_always_resolves(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.models import list_available_providers
    from hermes_cli.models_catalog_static import listed_canonical_providers
    from hermes_cli.provider_catalog import provider_catalog
    from hermes_cli.web_routers.oauth import _build_oauth_catalog

    def surfaces() -> dict[str, bool]:
        return {
            "canonical": any(p.slug == "solstice" for p in listed_canonical_providers()),
            "provider_list": any(r["id"] == "solstice" for r in list_available_providers()),
            "setup_catalog": any(d.slug == "solstice" for d in provider_catalog()),
            "accounts_tab": any(r["id"] == "solstice" for r in _build_oauth_catalog()),
        }

    assert providers.get_provider_profile("solstice").hidden is True
    assert surfaces() == dict.fromkeys(surfaces(), False)
    # Typed paths never consult the gate: a user who names it can sign in and run it.
    assert hermes_cli.auth.resolve_provider("solstice") == "solstice"
    assert hermes_cli.auth.resolve_provider("solstice-oauth") == "solstice"

    _write_pool(home, "solstice", _pool_row("solstice", access="ya29.x", expires_at_ms=int(time.time() * 1000) + 3_600_000))
    assert surfaces() == dict.fromkeys(surfaces(), True)


def test_brokered_opaque_bearer_rotates_on_stored_expiry_and_keeps_the_refresh_token(monkeypatch, tmp_path, request):
    """Google access tokens are opaque (no JWT ``exp``): the pool must rotate on ``expires_at_ms``, through the
    plugin's broker instead of a token URL, and keep the refresh token when the response omits it."""
    from agent.credential_pool import load_pool

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    grants: list[dict] = []

    def broker(grant: dict) -> dict:
        grants.append(grant)
        return {"access_token": f"ya29.fresh-{len(grants)}", "expires_in": 3600}

    name = "example-brokered"
    cfg = pkce.OAuthPKCEConfig(client_id="", authorize_url="", token_url="", token_request=broker)
    register_provider(ProviderProfile(name=name, auth_type="oauth_external", base_url="https://example.invalid/v1",
                                      auth_handler=pkce.pkce_auth_handler(cfg), refresh_credential=pkce.pkce_refresh_credential(cfg)))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(name, None), hermes_cli.auth.PROVIDER_REGISTRY.pop(name, None)))

    now_ms = int(time.time() * 1000)
    _write_pool(home, name, _pool_row(name, access="ya29.live", expires_at_ms=now_ms + 3_600_000))
    assert load_pool(name).select().access_token == "ya29.live" and grants == []  # fresh: no broker call

    _write_pool(home, name, _pool_row(name, access="ya29.stale", expires_at_ms=now_ms - 1000))
    leased = load_pool(name).select()
    assert leased.access_token == "ya29.fresh-1"
    assert grants == [{"grant_type": "refresh_token", "refresh_token": "1//rt-0"}]
    disk = json.loads((home / "auth.json").read_text())["credential_pool"][name][0]
    assert (disk["access_token"], disk["refresh_token"]) == ("ya29.fresh-1", "1//rt-0")
    assert disk["expires_at_ms"] > now_ms + 3_000_000

    # A grant the broker rejects is DEAD and stays DEAD through the inference 401 that follows (never re-benched
    # as EXHAUSTED, which would replay the dead token every cooldown).
    from agent.credential_pool import STATUS_DEAD
    from hermes_cli.auth_constants import AuthError

    def dead_broker(grant: dict) -> dict:
        raise AuthError("rejected", provider=name, code="invalid_grant", relogin_required=True)

    cfg_dead = pkce.OAuthPKCEConfig(client_id="", authorize_url="", token_url="", token_request=dead_broker)
    providers._REGISTRY[name] = ProviderProfile(name=name, auth_type="oauth_external", base_url="https://example.invalid/v1",
                                                refresh_credential=pkce.pkce_refresh_credential(cfg_dead))
    _write_pool(home, name, _pool_row(name, access="ya29.revoked", expires_at_ms=now_ms + 3_600_000))
    pool = load_pool(name)
    pool.select()
    assert pool.try_refresh_current() is None
    pool.mark_exhausted_and_rotate(status_code=401, api_key_hint="ya29.revoked")
    assert json.loads((home / "auth.json").read_text())["credential_pool"][name][0]["last_status"] == STATUS_DEAD


def test_signed_in_solstice_picker_row_is_the_plugin_list_in_order(monkeypatch, tmp_path):
    """The per-user-quota endpoint has no model listing, so the picker row is the plugin's fixed list,
    in its order: the first entry is the default model and the last is the per-model quota reserve."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_pool(home, "solstice", _pool_row("solstice", access="ya29.x", expires_at_ms=int(time.time() * 1000) + 3_600_000))
    from hermes_cli.model_switch_providers import list_authenticated_providers

    profile = providers.get_provider_profile("solstice")
    assert profile is not None
    row = next(r for r in list_authenticated_providers() if r["slug"] == "solstice")
    assert row["models"] == list(profile.fallback_models)


def test_solstice_targets_v1beta_and_lists_exactly_the_served_models():
    """The Interactions endpoint lives on v1beta; the list is the served IDs, Hydrogen (the
    default) first, Flash-Lite (aux model and quota reserve) last."""
    profile = providers.get_provider_profile("solstice")

    assert profile.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert profile.fallback_models == (
        "gemini-4-hydrogen", "gemini-4-neon", "gemini-4-carbon", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
    )
    assert profile.default_aux_model == "gemini-flash-lite-latest"


@pytest.mark.parametrize(
    ("model", "efforts"),
    [("gemini-4-hydrogen", ["low", "medium", "high", "xhigh"]), ("gemini-4-neon", ["low", "medium", "high", "xhigh"]),
     ("gemini-4-carbon", ["low", "medium", "high"]), ("gemini-3.5-flash-lite", ["minimal", "low", "medium", "high"]),
     ("gemini-flash-lite-latest", ["minimal", "low", "medium", "high"])],
)
def test_setup_effort_picker_offers_exactly_each_models_levels(model, efforts):
    from hermes_cli.main_provider_setup import _main_model_reasoning_efforts

    assert _main_model_reasoning_efforts(model, "solstice") == efforts


def test_setup_effort_picker_keeps_the_generic_ladder_for_an_unlisted_model(monkeypatch):
    from hermes_cli.main_provider_setup import _main_model_reasoning_efforts
    from hermes_constants import VALID_REASONING_EFFORTS

    monkeypatch.setattr("agent.models_dev.get_model_capabilities", lambda *a, **k: None)
    assert _main_model_reasoning_efforts("gemini-4-xenon", "solstice") == list(VALID_REASONING_EFFORTS)


def test_profile_loads_and_shapes_thinking_without_httpx():
    """Discovery imports the profile from interpreters without the app's HTTP stack."""
    code = (
        "import sys; sys.modules['httpx'] = None\n"
        "import providers\n"
        "p = providers.get_provider_profile('solstice')\n"
        "assert p.build_extra_body(model='gemini-4-carbon', reasoning_config=None) == "
        "{'thinking_config': {'includeThoughts': True, 'thinkingLevel': 'medium'}}\n"
        "assert 'httpx' not in {m for m in sys.modules if sys.modules[m] is not None}\n"
    )
    root = Path(__file__).resolve().parents[3]
    done = subprocess.run([sys.executable, "-c", code], check=False, cwd=root, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
