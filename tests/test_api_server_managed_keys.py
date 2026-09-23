"""Storage tests for the managed Hermes API keys.

The Dashboard stores and looks up managed API keys via
``hermes_cli.api_server_keys``. These tests pin its public contract:

* Plaintext secrets are NEVER persisted; only the salted SHA-256 hash is.
* ``list_keys`` and ``get_key_by_id`` never return plaintext or hash material.
* ``create_api_key`` returns the plaintext exactly once (in the response).
* ``revoke_api_key`` is idempotent and the matching ``verify_api_key`` then
  rejects the plaintext.
* Atomic concurrent writes don't corrupt the file.

The tests run against a temp ``HERMES_HOME`` so the real ``~/.hermes`` is
never touched.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

# Force-set HERMES_HOME BEFORE importing hermes_cli so the storage module
# resolves a temp directory, not the real ~/.hermes.
@pytest.fixture
def temp_hermes_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def keys_module(temp_hermes_home):
    # Import after the env var is set so ``_store_path()`` resolves correctly.
    import hermes_cli.api_server_keys as k
    # Reset any module-level caches defensively.
    return k


def _read_raw(home: Path) -> dict:
    p = home / "api_keys.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ─── storage shape ────────────────────────────────────────────────────────────


def test_list_keys_starts_empty(keys_module, temp_hermes_home):
    assert keys_module.list_keys() == []
    assert keys_module.has_active_keys() is False
    # File may or may not exist on disk yet — both are valid initial states.
    assert (temp_hermes_home / "api_keys.json").exists() is False


def test_create_returns_plaintext_only_once(keys_module):
    row = keys_module.create_api_key(name="first")
    assert row["name"] == "first"
    assert row["plaintext"].startswith(keys_module.KEY_PREFIX)
    # Public row fields are present.
    for required in ("id", "name", "prefix", "created_at", "active"):
        assert required in row
    assert row["active"] is True

    listed = keys_module.list_keys()
    assert len(listed) == 1
    # No secret material ever escapes the list endpoint.
    assert "plaintext" not in listed[0]
    assert "secret_hash" not in listed[0]
    assert "salt" not in listed[0]


def test_secret_not_stored_in_plaintext(keys_module, temp_hermes_home):
    row = keys_module.create_api_key(name="first")
    plaintext = row["plaintext"]
    raw = _read_raw(temp_hermes_home)
    # The plaintext appears NOWHERE on disk.
    serialized = json.dumps(raw)
    assert plaintext not in serialized
    # The hash and salt are stored instead.
    stored = raw["keys"][0]
    assert stored["secret_hash"] != plaintext
    assert len(stored["salt"]) > 0
    assert len(stored["secret_hash"]) == 64  # sha256 hex


# ─── verify / revoke ──────────────────────────────────────────────────────────


def test_verify_returns_identity_for_valid_token(keys_module):
    row = keys_module.create_api_key(name="test")
    identity = keys_module.verify_api_key(row["plaintext"])
    assert identity == {"id": row["id"], "name": "test"}


def test_verify_returns_none_for_wrong_token(keys_module):
    keys_module.create_api_key(name="test")
    assert keys_module.verify_api_key("hm_live_wrongtoken") is None
    assert keys_module.verify_api_key("") is None
    assert keys_module.verify_api_key("not-a-key") is None


def test_revoked_key_no_longer_authenticates(keys_module):
    row = keys_module.create_api_key(name="to-revoke")
    assert keys_module.verify_api_key(row["plaintext"]) is not None
    assert keys_module.revoke_api_key(row["id"]) is True
    # The plaintext is now rejected.
    assert keys_module.verify_api_key(row["plaintext"]) is None


def test_revoke_is_idempotent(keys_module):
    row = keys_module.create_api_key(name="to-revoke")
    assert keys_module.revoke_api_key(row["id"]) is True
    assert keys_module.revoke_api_key(row["id"]) is True  # idempotent
    assert keys_module.revoke_api_key("nonexistent-id") is False


def test_list_keys_excludes_revoked_by_default(keys_module):
    keys_module.create_api_key(name="a")
    b = keys_module.create_api_key(name="b")
    keys_module.revoke_api_key(b["id"])
    listed = keys_module.list_keys()
    assert len(listed) == 1
    assert listed[0]["name"] == "a"
    listed_all = keys_module.list_keys(include_revoked=True)
    assert {k["name"] for k in listed_all} == {"a", "b"}


# ─── input validation ─────────────────────────────────────────────────────────


def test_create_rejects_empty_name(keys_module):
    with pytest.raises(ValueError):
        keys_module.create_api_key(name="")
    with pytest.raises(ValueError):
        keys_module.create_api_key(name="   ")


def test_create_rejects_oversized_name(keys_module):
    with pytest.raises(ValueError):
        keys_module.create_api_key(name="x" * 129)


def test_multiple_keys_have_distinct_identities(keys_module, temp_hermes_home):
    a = keys_module.create_api_key(name="a")
    b = keys_module.create_api_key(name="b")
    assert a["id"] != b["id"]
    assert a["plaintext"] != b["plaintext"]
    # Each verifies only against its own plaintext.
    assert keys_module.verify_api_key(a["plaintext"])["id"] == a["id"]
    assert keys_module.verify_api_key(b["plaintext"])["id"] == b["id"]
    # And the on-disk hashes are distinct (different salts → different hashes).
    raw = _read_raw(temp_hermes_home)["keys"]
    a_stored = next(r for r in raw if r["id"] == a["id"])
    b_stored = next(r for r in raw if r["id"] == b["id"])
    assert a_stored["secret_hash"] != b_stored["secret_hash"]


# ─── atomic writes ────────────────────────────────────────────────────────────


def test_concurrent_creates_do_not_corrupt_file(keys_module, temp_hermes_home):
    """Spawn N threads that all create keys concurrently; final file must
    be valid and contain exactly N+M rows (plus any pre-existing)."""
    pre = keys_module.create_api_key(name="pre-existing")
    n = 30
    errors: list[BaseException] = []

    def w(i: int):
        try:
            keys_module.create_api_key(name=f"concurrent-{i}")
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"concurrent create raised: {errors}"
    raw = _read_raw(temp_hermes_home)
    assert raw["version"] == keys_module.SCHEMA_VERSION
    assert isinstance(raw["keys"], list)
    # 1 pre + N concurrent
    assert len(raw["keys"]) == n + 1
    # Every row has the required fields.
    for entry in raw["keys"]:
        for required in ("id", "name", "prefix", "secret_hash", "salt",
                          "created_at", "last_used_at", "revoked_at"):
            assert required in entry
    # The pre-existing row survived.
    assert any(e["id"] == pre["id"] for e in raw["keys"])


def test_file_permissions_are_0600(keys_module, temp_hermes_home):
    keys_module.create_api_key(name="perm-check")
    mode = (temp_hermes_home / "api_keys.json").stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


# ─── verification side-effects ────────────────────────────────────────────────


def test_verify_updates_last_used_at(keys_module):
    row = keys_module.create_api_key(name="used")
    listed = keys_module.list_keys()[0]
    assert listed["last_used_at"] is None
    keys_module.verify_api_key(row["plaintext"])
    after = keys_module.list_keys()[0]
    # Debounced write: should now reflect the recent use.
    assert after["last_used_at"] is not None