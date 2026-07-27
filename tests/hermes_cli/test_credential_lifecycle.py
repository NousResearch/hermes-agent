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


def test_delete_env_key_prunes_env_seeded_pool_entry(hermes_home):
    _write_env(hermes_home, ZAI_API_KEY=FAKE_ZAI_KEY)
    _write_auth(hermes_home, _zai_pool_fixture())

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "zai" in body["pool_pruned"]

    # .env cleared
    from hermes_cli.config import load_env

    assert "ZAI_API_KEY" not in load_env()

    # auth.json: env-seeded entry gone, OAuth entry preserved
    store = _read_auth(hermes_home)
    sources = [e["source"] for e in store["credential_pool"]["zai"]]
    assert "env:ZAI_API_KEY" not in sources
    assert "device_code" in sources, "OAuth grant must survive an API-key delete"


def test_delete_env_key_removes_provider_pool_key_when_emptied(hermes_home):
    """A provider whose ONLY pool entry was env-seeded disappears entirely."""
    _write_env(hermes_home, ZAI_API_KEY=FAKE_ZAI_KEY)
    _write_auth(
        hermes_home,
        {"zai": [_zai_pool_fixture()["zai"][0]]},  # env entry only
    )

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    store = _read_auth(hermes_home)
    assert "zai" not in store.get("credential_pool", {}), (
        "provider must vanish from credential_pool so the model picker "
        "stops listing it (#51071)"
    )


def test_delete_survives_pool_reload(hermes_home):
    """#59761: the pool loader must not resurrect the entry after 'restart'."""
    _write_env(hermes_home, ZAI_API_KEY=FAKE_ZAI_KEY)
    _write_auth(hermes_home, {"zai": [_zai_pool_fixture()["zai"][0]]})

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200

    # Simulate restart: reload the pool from disk the way startup does.
    from agent.credential_pool import load_pool

    entries = load_pool("zai").entries()
    assert entries == [], f"stale entries resurrected: {[e.source for e in entries]}"


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


def test_delete_pool_only_credential_still_cleans_up(hermes_home):
    """Stale pool entry with NO .env line (the #59761 restart state) is
    removable through the same delete button instead of 404ing."""
    _write_env(hermes_home)  # empty .env
    _write_auth(hermes_home, {"zai": [_zai_pool_fixture()["zai"][0]]})

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    store = _read_auth(hermes_home)
    assert "zai" not in store.get("credential_pool", {})


def test_delete_unknown_key_404s(hermes_home):
    _write_env(hermes_home)
    resp = client.request(
        "DELETE", "/api/env", json={"key": "NEVER_SET_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 404


def test_delete_does_not_touch_other_providers(hermes_home):
    _write_env(hermes_home, ZAI_API_KEY=FAKE_ZAI_KEY)
    other_key = "dk-" + "e" * 24
    pool = _zai_pool_fixture()
    pool["deepseek"] = [
        {
            "id": "d1",
            "label": "env",
            "auth_type": "api_key",
            "priority": 0,
            "source": "env:DEEPSEEK_API_KEY",
            "access_token": other_key,
        }
    ]
    _write_auth(hermes_home, pool)

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    store = _read_auth(hermes_home)
    assert [e["source"] for e in store["credential_pool"]["deepseek"]] == [
        "env:DEEPSEEK_API_KEY"
    ]


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


def test_update_rotates_custom_provider_mirror(hermes_home):
    old = "sk-cp-" + "h" * 24
    new = "sk-cp-" + "i" * 24
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(
        hermes_home,
        "custom_providers:\n"
        "  - name: myendpoint\n"
        "    base_url: https://llm.example.test/v1\n"
        f"    api_key: {old}\n",
    )

    resp = client.put(
        "/api/env", json={"key": "OPENAI_API_KEY", "value": new}, headers=HEADERS
    )
    assert resp.status_code == 200
    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert old not in cfg_text
    assert new in cfg_text


def test_update_leaves_unrelated_config_keys_alone(hermes_home):
    """A DIFFERENT key configured inline must not be rewritten by value-match."""
    old = "sk-un-" + "j" * 24
    unrelated = "sk-un-" + "k" * 24
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(hermes_home, f"model:\n  provider: custom\n  api_key: {unrelated}\n")

    resp = client.put(
        "/api/env",
        json={"key": "OPENAI_API_KEY", "value": "sk-un-" + "l" * 24},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert unrelated in cfg_text, "unrelated inline key must be preserved"


def test_delete_scrubs_config_yaml_mirror(hermes_home):
    old = "sk-dl-" + "m" * 24
    _write_env(hermes_home, OPENAI_API_KEY=old)
    _write_config(hermes_home, f"model:\n  provider: custom\n  api_key: {old}\n")

    resp = client.request(
        "DELETE", "/api/env", json={"key": "OPENAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert "model.api_key" in resp.json()["config_scrubbed"]
    cfg_text = hermes_home.joinpath("config.yaml").read_text(encoding="utf-8")
    assert old not in cfg_text


# ---------------------------------------------------------------------------
# Suppression round-trip: delete sticks, re-add lifts it
# ---------------------------------------------------------------------------


def test_delete_then_resave_round_trip(hermes_home):
    _write_env(hermes_home, ZAI_API_KEY=FAKE_ZAI_KEY)
    _write_auth(hermes_home, {"zai": [_zai_pool_fixture()["zai"][0]]})

    resp = client.request(
        "DELETE", "/api/env", json={"key": "ZAI_API_KEY"}, headers=HEADERS
    )
    assert resp.status_code == 200

    from hermes_cli.auth import is_source_suppressed

    assert is_source_suppressed("zai", "env:ZAI_API_KEY"), (
        "delete must suppress the env source so a lingering shell export "
        "can't re-seed the pool"
    )

    resp = client.put(
        "/api/env", json={"key": "ZAI_API_KEY", "value": NEW_KEY}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert not is_source_suppressed("zai", "env:ZAI_API_KEY"), (
        "an explicit re-save must lift the suppression (like `hermes auth add`)"
    )

    from hermes_cli.config import load_env

    assert load_env()["ZAI_API_KEY"] == NEW_KEY
