"""Solstice invariants: a pre-release provider stays off every discovery surface until signed in, and a
brokered OAuth row with an opaque bearer rotates on its stored expiry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

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


def test_bundled_provider_discovery_survives_an_interpreter_without_an_http_client(tmp_path):
    """Every bundled provider plugin must import in an interpreter that has no HTTP client.

    Provider discovery is not exclusive to the full runtime: ``hermes update`` ends in a ``pm ensure``
    whose worker imports ``hermes_cli.config``, which runs ``_inject_profile_env_vars()`` ->
    ``list_providers()`` at import time, on a four-package PM interpreter with no ``httpx``. A plugin
    that pulls an HTTP client at import time drops out of that registry — the observed
    ``hermes update`` line ``Failed to load bundled provider plugin solstice: No module named 'httpx'``.
    """
    repo = Path(hermes_cli.__file__).resolve().parents[1]
    probe = "\n".join((
        "import sys",
        "sys.path.insert(0, sys.argv[1])",
        "sys.path.insert(0, sys.argv[2])  # the running test's own dependency tree (`-I` drops PYTHONPATH)",
        "sys.modules['httpx'] = None  # stand in for an interpreter with no HTTP client",
        "import providers",
        "profile = providers.get_provider_profile('solstice')",
        "assert profile is not None and profile.name == 'solstice', profile",
    ))
    done = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(repo), sysconfig.get_path("purelib")],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_HOME": str(tmp_path / "hermes")},
    )
    assert done.returncode == 0, f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
