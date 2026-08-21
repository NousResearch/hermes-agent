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
# Suppression round-trip: delete sticks, re-add lifts it
# ---------------------------------------------------------------------------


def test_save_env_value_refuses_onepassword_managed_key(hermes_home):
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        '      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"\n',
    )
    from hermes_cli.config import save_env_value

    with pytest.raises(ValueError, match="1Password"):
        save_env_value("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 24)

    assert not hermes_home.joinpath(".env").exists()


def test_save_env_value_unaffected_for_key_not_mapped_in_onepassword(hermes_home):
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        '      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"\n',
    )
    from hermes_cli.config import load_env, save_env_value

    new_value = "sk-or-" + "y" * 24
    save_env_value("OPENROUTER_API_KEY", new_value)
    assert load_env()["OPENROUTER_API_KEY"] == new_value


def test_save_env_value_unaffected_when_onepassword_disabled(hermes_home):
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: false\n"
        "    env:\n"
        '      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"\n',
    )
    from hermes_cli.config import load_env, save_env_value

    new_value = "sk-ant-" + "w" * 24
    save_env_value("ANTHROPIC_API_KEY", new_value)
    assert load_env()["ANTHROPIC_API_KEY"] == new_value


def test_get_env_reports_onepassword_managed_key(hermes_home, monkeypatch):
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        '      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"\n',
    )
    resolved = "sk-ant-" + "z" * 24
    monkeypatch.setenv("ANTHROPIC_API_KEY", resolved)

    resp = client.get("/api/env", headers=HEADERS)
    assert resp.status_code == 200
    row = resp.json()["ANTHROPIC_API_KEY"]
    assert row["is_set"] is True
    assert row["managed_by"] == "onepassword"
    assert resolved not in resp.text


def test_get_env_unmapped_key_has_no_managed_by(hermes_home):
    resp = client.get("/api/env", headers=HEADERS)
    assert resp.status_code == 200
    row = resp.json()["ANTHROPIC_API_KEY"]
    assert row["is_set"] is False
    assert row.get("managed_by") is None


# ---------------------------------------------------------------------------
# Final-review fix 1: onepassword_managed_env_keys() must be evaluated inside
# the REQUESTED profile's scope, not the dashboard/default profile's.
# ---------------------------------------------------------------------------


def test_get_env_onepassword_mapping_is_profile_scoped(hermes_home, monkeypatch):
    """A key mapped via 1Password in profile A must not appear managed when
    profile B (which never mapped it) is queried, and vice versa.

    Regression for the final-review finding that ``onepassword_managed_env_keys()``
    was called OUTSIDE ``with _profile_scope(profile):`` in
    ``_get_env_vars_sync()``, so it always read the dashboard/default
    profile's config.yaml regardless of the ``?profile=`` query param.
    """
    from hermes_cli import profiles as profiles_mod

    # Default profile (the dashboard's own home, via the hermes_home fixture)
    # maps ANTHROPIC_API_KEY via 1Password. It must NOT leak into the
    # "worker" profile's view.
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        '      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"\n',
    )
    resolved = "sk-ant-" + "z" * 24
    monkeypatch.setenv("ANTHROPIC_API_KEY", resolved)

    worker_home = profiles_mod.get_profile_dir("worker")
    worker_home.mkdir(parents=True)
    # "worker" maps a DIFFERENT key (OPENROUTER_API_KEY) via 1Password, and
    # does NOT map ANTHROPIC_API_KEY at all.
    (worker_home / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        '      OPENROUTER_API_KEY: "op://Private/OpenRouter/credential"\n',
        encoding="utf-8",
    )

    # Default profile: ANTHROPIC_API_KEY is 1Password-managed, OPENROUTER is not.
    default_resp = client.get("/api/env", headers=HEADERS)
    assert default_resp.status_code == 200
    default_rows = default_resp.json()
    assert default_rows["ANTHROPIC_API_KEY"]["managed_by"] == "onepassword"
    assert default_rows["OPENROUTER_API_KEY"].get("managed_by") is None

    # "worker" profile: OPENROUTER_API_KEY is 1Password-managed there, and
    # ANTHROPIC_API_KEY must NOT be reported as managed (it's only mapped in
    # the default profile's config.yaml, which "worker" must not see).
    worker_resp = client.get("/api/env?profile=worker", headers=HEADERS)
    assert worker_resp.status_code == 200
    worker_rows = worker_resp.json()
    assert worker_rows["OPENROUTER_API_KEY"]["managed_by"] == "onepassword"
    assert worker_rows["ANTHROPIC_API_KEY"].get("managed_by") is None
    assert worker_rows["ANTHROPIC_API_KEY"]["is_set"] is False


# ---------------------------------------------------------------------------
# Final-review fix 2: a stale plaintext .env copy must win over the
# 1Password mapping — the row must behave like a normal, editable key.
# ---------------------------------------------------------------------------


def test_get_env_onepassword_mapped_key_with_stale_env_value_is_editable(
    hermes_home,
):
    """If .env still has a real value for a 1Password-mapped key (e.g. the
    user mapped it without cleaning up the old plaintext copy), the row must
    NOT be reported as locked/managed — it must look like a normal key so
    the user can remove the stale plaintext value from the Keys page.
    """
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        '      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"\n',
    )
    stale_value = "sk-ant-" + "s" * 24
    _write_env(hermes_home, ANTHROPIC_API_KEY=stale_value)

    resp = client.get("/api/env", headers=HEADERS)
    assert resp.status_code == 200
    row = resp.json()["ANTHROPIC_API_KEY"]
    assert row["is_set"] is True
    assert row.get("managed_by") is None


# ---------------------------------------------------------------------------
# Final-review fix 3: a key ONLY mapped via 1Password (never in .env, never
# in OPTIONAL_ENV_VARS/the provider catalog) must still appear as a row.
# ---------------------------------------------------------------------------


def test_get_env_onepassword_only_custom_key_appears_as_a_row(hermes_home, monkeypatch):
    """A custom-endpoint key that's ONLY mapped via 1Password (e.g. a
    HERMES_CUSTOM_<slug>_API_KEY that's never written to .env, never hand
    declared in OPTIONAL_ENV_VARS, and never in the provider catalog) must
    still show up in GET /api/env — otherwise it's invisible on the Keys page.
    """
    custom_key = "HERMES_CUSTOM_ACMELLM_API_KEY"
    _write_config(
        hermes_home,
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        f'      {custom_key}: "op://Private/AcmeLLM/credential"\n',
    )
    resolved = "sk-acme-" + "q" * 24
    monkeypatch.setenv(custom_key, resolved)

    resp = client.get("/api/env", headers=HEADERS)
    assert resp.status_code == 200
    rows = resp.json()
    assert custom_key in rows
    row = rows[custom_key]
    assert row["is_set"] is True
    assert row["managed_by"] == "onepassword"
    assert resolved not in resp.text

