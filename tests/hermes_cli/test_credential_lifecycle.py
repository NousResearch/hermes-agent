"""E2E tests for the unified provider-credential lifecycle (#51071 #59761 #62269).

A provider API key can live in .env, auth.json's credential_pool, and
config.yaml mirrors at once. These tests drive the REAL dashboard endpoint
handlers (PUT/DELETE /api/env) against real on-disk fixtures in a temp
HERMES_HOME (tests/conftest.py isolation) and assert every store agrees
afterwards.

All fake secrets are constructed at runtime so no key-shaped literal ever
lands in the repo.
"""

import json

import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

# Runtime-constructed fake credentials (never literal key-shaped strings).
FAKE_ZAI_KEY = "zk-" + "a" * 24
FAKE_OAUTH_TOKEN = "oa-" + "b" * 24
NEW_KEY = "zk-" + "c" * 24


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    """Fresh HERMES_HOME with .env + auth.json + config.yaml fixtures."""
    home = tmp_path / "cred_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()
    return home


def _write_env(home, **pairs):
    home.joinpath(".env").write_text(
        "".join(f"{k}={v}\n" for k, v in pairs.items()), encoding="utf-8"
    )
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()


def _write_auth(home, pool):
    home.joinpath("auth.json").write_text(
        json.dumps({"credential_pool": pool}), encoding="utf-8"
    )


def _read_auth(home):
    return json.loads(home.joinpath("auth.json").read_text(encoding="utf-8"))


def _zai_pool_fixture():
    """One env-seeded API-key entry plus one OAuth entry for the same provider."""
    return {
        "zai": [
            {
                "id": "e1",
                "label": "env",
                "auth_type": "api_key",
                "priority": 0,
                "source": "env:ZAI_API_KEY",
                "access_token": FAKE_ZAI_KEY,
            },
            {
                "id": "o1",
                "label": "oauth",
                "auth_type": "oauth",
                "priority": 0,
                "source": "device_code",
                "access_token": FAKE_OAUTH_TOKEN,
                "refresh_token": "rt-" + "d" * 16,
            },
        ]
    }


# ---------------------------------------------------------------------------
# DELETE — #51071 / #59761: stale credential_pool entries must be pruned
# ---------------------------------------------------------------------------




def test_delete_clears_provider_models_cache(hermes_home):
    _write_env(hermes_home, ZAI_API_KEY=FAKE_ZAI_KEY)
    _write_auth(hermes_home, {"zai": [_zai_pool_fixture()["zai"][0]]})
    cache_path = hermes_home / "provider_models_cache.json"
    cache_path.write_text(
        json.dumps({"zai": {"models": ["glm-5"], "ts": 0}}), encoding="utf-8"
    )

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "zai" not in cache


# ---------------------------------------------------------------------------
# UPDATE — #62269: config.yaml mirrors of the old key must rotate with .env
# ---------------------------------------------------------------------------


def _write_config(home, text):
    home.joinpath("config.yaml").write_text(text, encoding="utf-8")


def test_update_rotates_config_yaml_model_mirror(hermes_home):
    old = "sk-oe-" + "f" * 24
    new = "sk-oe-" + "g" * 24
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(
        hermes_home,
        "model:\n"
        "  provider: custom\n"
        "  default: my-model\n"
        "  base_url: https://llm.example.test/v1\n"
        f"  api_key: {old}\n",
    )

    resp = client.put(
        "/api/env", json={"key": "OPENAI_API_KEY", "value": new}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert "model.api_key" in resp.json().get("config_updates", [])

    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert old not in cfg_text, "stale old key left in config.yaml (#62269)"
    assert new in cfg_text, "config.yaml mirror not rotated to the new key"

    from hermes_cli.config import load_env

    assert load_env()["OPENAI_API_KEY"] == new




# ---------------------------------------------------------------------------
# Keyed `providers` schema (v12+) — where the dashboard writes custom
# endpoints. Its inline api_key is a real credential and higher-precedence
# than the env var, so a stale copy left here shadows a rotation (#62269) and
# survives a "remove from EVERY store" delete.
# ---------------------------------------------------------------------------


def test_update_rotates_keyed_providers_mirror(hermes_home):
    old = "sk-kp-" + "n" * 24
    new = "sk-kp-" + "o" * 24
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(
        hermes_home,
        "providers:\n"
        "  myendpoint:\n"
        "    base_url: https://llm.example.test/v1\n"
        f"    api_key: {old}\n",
    )

    resp = client.put(
        "/api/env", json={"key": "OPENAI_API_KEY", "value": new}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert "providers.myendpoint.api_key" in resp.json().get("config_updates", [])

    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert old not in cfg_text, "stale key in providers.<id> shadows the rotation (#62269)"
    assert new in cfg_text, "keyed providers mirror not rotated to the new key"


def test_delete_scrubs_keyed_providers_mirror(hermes_home):
    old = "sk-kp-" + "p" * 24
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(
        hermes_home,
        "providers:\n"
        "  myendpoint:\n"
        "    base_url: https://llm.example.test/v1\n"
        f"    api_key: {old}\n",
    )

    resp = client.request(
        "DELETE", "/api/env", json={"key": "OPENAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert "providers.myendpoint.api_key" in resp.json()["config_scrubbed"]
    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert old not in cfg_text, "delete must clear the credential from EVERY store"


def test_scrub_never_touches_providers_base_url_alias(hermes_home):
    """In the keyed ``providers`` schema ``api`` is the base_url alias, NOT a
    credential. Even if the .env value coincided with a base_url, the scrub
    must not rewrite a provider's endpoint URL."""
    old = "https://llm.example.test/v1"  # a URL that also happens to be the key value
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(
        hermes_home,
        "providers:\n"
        "  myendpoint:\n"
        f"    api: {old}\n"          # base_url alias — must be preserved
        "    model: my-model\n",
    )

    resp = client.request(
        "DELETE", "/api/env", json={"key": "OPENAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert old in cfg_text, "providers.<id>.api is a base_url and must survive"
    assert "providers.myendpoint.api" not in resp.json().get("config_scrubbed", [])


# ---------------------------------------------------------------------------
# Suppression round-trip: delete sticks, re-add lifts it
# ---------------------------------------------------------------------------


