"""E2E tests for the unified provider-credential lifecycle (#51071 #59761 #62269).

A provider API key can live in .env, auth.json's credential_pool, and
config.yaml mirrors at once. These tests drive the REAL dashboard endpoint
handlers (PUT/DELETE /api/env) against real on-disk fixtures in a temp
HERMES_HOME (tests/conftest.py isolation) and assert every store agrees
afterwards.

All fake secrets are constructed at runtime so no key-shaped literal ever
lands in the repo.
"""

import errno
import json
import os

import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

# Runtime-constructed fake credentials (never literal key-shaped strings).
FAKE_ZAI_KEY = "zk-" + "a" * 24
FAKE_OAUTH_TOKEN = "oa-" + "b" * 24
NEW_KEY = "zk-" + "c" * 24

FAULT_ENV_VAR = "AUDIT_PROVIDER_API_KEY"
FAULT_OLD_KEY = "old-" + "d" * 24
FAULT_NEW_KEY = "new-" + "e" * 24


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


def _seed_fault_transaction(home, monkeypatch):
    """One env credential mirrored by both runtime precedence-bearing inline schemas."""
    monkeypatch.delenv(FAULT_ENV_VAR, raising=False)
    env_path = home / ".env"
    config_path = home / "config.yaml"
    env_path.write_text(
        f"# keep env comment\n{FAULT_ENV_VAR}={FAULT_OLD_KEY}\nUNCHANGED=value\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "# keep config comment\n"
        "model:\n"
        "  provider: custom\n"
        "  default: audit/model\n"
        "  base_url: https://audit.invalid/v1\n"
        f"  key_env: {FAULT_ENV_VAR}  # keep inline comment\n"
        f"  api_key: {FAULT_OLD_KEY}\n"
        "providers:\n"
        "  audit-endpoint:\n"
        "    base_url: https://audit.invalid/v1\n"
        f"    api_key: {FAULT_OLD_KEY}\n"
        "unrelated: \"off\"\n",
        encoding="utf-8",
    )
    if os.name == "posix":
        env_path.chmod(0o640)
        config_path.chmod(0o640)

    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config_mod.invalidate_env_cache()
    return env_path, config_path


def _fail_first_write(monkeypatch, target, name):
    real = getattr(target, name)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EIO, f"injected {name} failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, fail_once)
    return lambda: calls


def _assert_transaction_artifacts_preserved(env_path, config_path):
    env_text = env_path.read_text(encoding="utf-8")
    config_text = config_path.read_text(encoding="utf-8")
    assert "# keep env comment" in env_text
    assert "UNCHANGED=value" in env_text
    assert "# keep config comment" in config_text
    assert "# keep inline comment" in config_text
    assert 'unrelated: "off"' in config_text
    if os.name == "posix":
        assert env_path.stat().st_mode & 0o777 == 0o640
        assert config_path.stat().st_mode & 0o777 == 0o640


def _resolved_audit_runtime_key():
    from hermes_cli import config as config_mod
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod.invalidate_env_cache()
    return resolve_runtime_provider(requested="custom")["api_key"]


@pytest.mark.parametrize("failed_write", ["config", "env"])
def test_rotation_retries_each_failed_write_and_converges(
    hermes_home, monkeypatch, failed_write
):
    """A failed mirror-first rotation keeps .env provenance for a convergent retry."""
    from hermes_cli import config as config_mod
    from hermes_cli.credential_lifecycle import save_provider_env_credential

    env_path, config_path = _seed_fault_transaction(hermes_home, monkeypatch)
    target_name = "atomic_config_replace" if failed_write == "config" else "save_env_value"
    call_count = _fail_first_write(monkeypatch, config_mod, target_name)

    with pytest.raises(OSError, match=f"injected {target_name} failure"):
        save_provider_env_credential(FAULT_ENV_VAR, FAULT_NEW_KEY)

    # If the mirror write failed, neither store moved. If .env failed, every higher-precedence
    # mirror already points at the requested value while .env retains the old retry provenance.
    assert config_mod.load_env()[FAULT_ENV_VAR] == FAULT_OLD_KEY
    config_after_fault = config_mod.read_user_config_raw(config_path)
    expected_inline = FAULT_OLD_KEY if failed_write == "config" else FAULT_NEW_KEY
    assert config_after_fault["model"]["api_key"] == expected_inline
    assert config_after_fault["providers"]["audit-endpoint"]["api_key"] == expected_inline

    result = save_provider_env_credential(FAULT_ENV_VAR, FAULT_NEW_KEY)

    assert result["ok"] is True
    assert call_count() == 2
    assert config_mod.load_env()[FAULT_ENV_VAR] == FAULT_NEW_KEY
    final_config = config_mod.read_user_config_raw(config_path)
    assert final_config["model"]["api_key"] == FAULT_NEW_KEY
    assert final_config["providers"]["audit-endpoint"]["api_key"] == FAULT_NEW_KEY
    assert FAULT_OLD_KEY not in env_path.read_text(encoding="utf-8")
    assert FAULT_OLD_KEY not in config_path.read_text(encoding="utf-8")
    assert _resolved_audit_runtime_key() == FAULT_NEW_KEY
    _assert_transaction_artifacts_preserved(env_path, config_path)


@pytest.mark.parametrize("failed_write", ["config", "env"])
def test_revocation_retries_each_failed_write_and_converges(
    hermes_home, monkeypatch, failed_write
):
    """A failed mirror-first revocation keeps .env provenance for a convergent retry."""
    from hermes_cli import config as config_mod
    from hermes_cli.credential_lifecycle import remove_provider_env_credential

    env_path, config_path = _seed_fault_transaction(hermes_home, monkeypatch)
    target_name = "atomic_config_replace" if failed_write == "config" else "remove_env_value"
    call_count = _fail_first_write(monkeypatch, config_mod, target_name)

    with pytest.raises(OSError, match=f"injected {target_name} failure"):
        remove_provider_env_credential(FAULT_ENV_VAR)

    assert config_mod.load_env()[FAULT_ENV_VAR] == FAULT_OLD_KEY
    config_after_fault = config_mod.read_user_config_raw(config_path)
    if failed_write == "config":
        assert config_after_fault["model"]["api_key"] == FAULT_OLD_KEY
        assert config_after_fault["providers"]["audit-endpoint"]["api_key"] == FAULT_OLD_KEY
    else:
        assert "api_key" not in config_after_fault["model"]
        assert "api_key" not in config_after_fault["providers"]["audit-endpoint"]

    result = remove_provider_env_credential(FAULT_ENV_VAR)

    assert result["ok"] is True
    assert result["found"] is True
    assert call_count() == 2
    assert FAULT_ENV_VAR not in config_mod.load_env()
    final_config = config_mod.read_user_config_raw(config_path)
    assert "api_key" not in final_config["model"]
    assert "api_key" not in final_config["providers"]["audit-endpoint"]
    assert FAULT_OLD_KEY not in env_path.read_text(encoding="utf-8")
    assert FAULT_OLD_KEY not in config_path.read_text(encoding="utf-8")
    assert _resolved_audit_runtime_key() != FAULT_OLD_KEY
    _assert_transaction_artifacts_preserved(env_path, config_path)


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
# Desktop PUT /api/env — #96058: credential_pool must be materialized so the
# live runtime picks up the new key without waiting for its next background
# load_pool() or a separate `hermes auth add`.
# ---------------------------------------------------------------------------


# OpenCode Go is a registered api_key provider with api_key_env_vars containing
# the single env var OPENCODE_GO_API_KEY — exercising the exact reproducer
# from issue #96058 (Ubuntu 24.04, openai_sdk 2.24.0, provider=opencode-go).
OPENCODE_KEY_NEW = "ocg-" + "e" * 28


def test_put_api_env_materializes_credential_pool_entry(hermes_home):
    """Desktop Providers → API keys → Save must write a credential_pool entry.

    Pre-fix: save_provider_env_credential only mutated .env. The live pool
    kept authenticating with a stale higher-precedence config.yaml mirror or
    the old cached credential until a separate ``hermes auth add opencode-go``
    ran. auth.json mtime was unchanged before/after Save (#96058).

    Post-fix: the same PUT /api/env call must also materialize an entry under
    ``credential_pool.<provider>`` in auth.json so the next request
    authenticates immediately, matching ``hermes auth add <provider> --type
    api-key`` behavior.
    """
    # Start clean: empty auth.json so the only way a pool entry shows up is
    # via the PUT /api/env handler we're testing.
    _write_auth(hermes_home, {})

    resp = client.put(
        "/api/env",
        json={"key": "OPENCODE_GO_API_KEY", "value": OPENCODE_KEY_NEW},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("ok") is True
    assert body.get("key") == "OPENCODE_GO_API_KEY"

    # auth.json must now have a credential_pool entry for opencode-go. The
    # exact source string lives in source="env:OPENCODE_GO_API_KEY" — env
    # sources are sanitized on disk (the raw token is replaced with a
    # fingerprint; the canonical secret lives in .env and gets re-hydrated by
    # load_pool() on each read). The critical observable is: load_pool() on
    # the next call returns an in-memory entry carrying the just-saved token.
    auth = _read_auth(hermes_home)
    pool = auth.get("credential_pool", {})
    assert "opencode-go" in pool, (
        "PUT /api/env did not materialize a credential_pool entry for "
        "opencode-go (#96058)"
    )
    entries = pool["opencode-go"]
    assert isinstance(entries, list) and entries, pool
    matched_disk = [
        e for e in entries
        if isinstance(e.get("source"), str)
        and e["source"] == "env:OPENCODE_GO_API_KEY"
    ]
    assert matched_disk, (
        f"credential_pool.opencode-go env-seeded reference missing on disk; "
        f"got {entries!r}"
    )
    # And: a fresh load_pool() must surface the just-saved token to the
    # runtime. This is the actual end-to-end contract — anything weaker
    # means the OpenAI client will 401 because it never receives the new key.
    from agent.credential_pool import load_pool
    pool_obj = load_pool("opencode-go")
    runtime_entries = pool_obj.entries()
    matched_runtime = [
        e for e in runtime_entries
        if e.access_token == OPENCODE_KEY_NEW
        and isinstance(e.source, str)
        and e.source == "env:OPENCODE_GO_API_KEY"
    ]
    assert matched_runtime, (
        "load_pool('opencode-go') did not surface the just-saved token; "
        "the live runtime will keep 401'ing (#96058). "
        f"Got sources: {[e.source for e in runtime_entries]!r}"
    )
    assert matched_runtime[0].auth_type == "api_key"




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




# ---------------------------------------------------------------------------
# GET /api/env — provider_primary pass-through for the Desktop Keys tab
# ---------------------------------------------------------------------------
# The Desktop groups a provider card's rows by provider_label and picks the
# card's main "Paste key" field from `provider_primary` first. That flag is
# computed per catalog entry in _catalog_provider_env_metadata (index == 0 of
# the provider's own api_key_env_vars) but _row used to drop it, so a card's
# own first credential arrived with provider_primary=None and the grouping
# fell back to the first non-advanced key var — which, for a profile-shared
# credential contributed by peer providers (DASHSCOPE_API_KEY is index >= 1
# of alibaba-coding-plan-cn), could be a FOREIGN tier's key.


def test_get_api_env_passes_provider_primary_through(hermes_home):
    """Every provider card's own index-0 credential must stay its main field."""
    resp = client.get("/api/env", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    env = resp.json()

    # The CN Coding Plan card: its own key is its primary; the shared
    # DASHSCOPE_API_KEY alias joins the card marked primary=False.
    assert env["ALIBABA_CODING_PLAN_CN_API_KEY"]["provider_primary"] is True
    dashscope = env["DASHSCOPE_API_KEY"]["provider_profiles"]
    cn_profile = next(
        p for p in dashscope if p["provider"] == "alibaba-coding-plan-cn"
    )
    assert cn_profile["primary"] is False, (
        "DASHSCOPE_API_KEY is a fallback alias for alibaba-coding-plan-cn; "
        "marking it primary would re-point the card's main field away from "
        "ALIBABA_CODING_PLAN_CN_API_KEY"
    )
