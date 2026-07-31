"""Runtime credential projection under a shared HERMES_AUTH_HOME.

Rows derived from one runtime's ``.env`` or ``config.yaml`` are filtered before
pool construction and never persisted into a residence shared by several
runtime homes. A no-override home keeps the historical persistence behavior.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest


def _write_env(home: Path, content: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(content, encoding="utf-8")


def _write_config(home: Path, config: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


def _activate_home(monkeypatch, home: Path) -> None:
    from hermes_cli.config import invalidate_env_cache

    monkeypatch.setenv("HERMES_HOME", str(home))
    invalidate_env_cache()


def _persisted_pool(auth_file: Path) -> dict:
    if not auth_file.exists():
        return {}
    store = json.loads(auth_file.read_text(encoding="utf-8"))
    pool = store.get("credential_pool")
    return pool if isinstance(pool, dict) else {}


def test_env_credentials_stay_session_private_in_a_shared_residence(
    monkeypatch, tmp_path
):
    from agent.credential_pool import load_pool

    residence = tmp_path / "residence"
    home_a = tmp_path / "runtime-a"
    home_b = tmp_path / "runtime-b"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    auth_file = residence / "auth.json"

    # Session A owns the key: it must see and keep its env credential…
    _write_env(home_a, "OPENROUTER_API_KEY=key-from-a\n")
    _activate_home(monkeypatch, home_a)
    pool_a = load_pool("openrouter")
    env_entries = [
        entry for entry in pool_a.entries() if entry.source == "env:OPENROUTER_API_KEY"
    ]
    assert len(env_entries) == 1
    assert env_entries[0].access_token == "key-from-a"
    # …but the shared store must never learn it.
    assert "env:" not in json.dumps(_persisted_pool(auth_file))

    # Session B has no such variable: nothing to inherit, nothing to select.
    _write_env(home_b, "")
    _activate_home(monkeypatch, home_b)
    pool_b = load_pool("openrouter")
    assert pool_b.entries() == []
    assert pool_b.current() is None


def test_env_suppression_is_sticky_only_in_the_owning_runtime(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from agent.credential_pool import load_pool
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_remove_command

    residence = tmp_path / "residence"
    home_a = tmp_path / "runtime-a"
    home_b = tmp_path / "runtime-b"
    source = "env:OPENROUTER_API_KEY"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    _write_env(home_a, "OPENROUTER_API_KEY=key-from-a\n")
    _write_env(home_b, "OPENROUTER_API_KEY=key-from-b\n")

    _activate_home(monkeypatch, home_a)
    assert [entry.access_token for entry in load_pool("openrouter").entries()] == [
        "key-from-a"
    ]
    auth_remove_command(SimpleNamespace(provider="openrouter", target="1"))
    _write_env(home_a, "OPENROUTER_API_KEY=key-from-a\n")
    _activate_home(monkeypatch, home_a)
    assert is_source_suppressed("openrouter", source)
    assert load_pool("openrouter").entries() == []

    local_path = home_a / ".credential_suppressions.json"
    local_state = json.loads(local_path.read_text(encoding="utf-8"))
    assert local_state["suppressed_sources"] == {"openrouter": [source]}
    if os.name != "nt":
        assert stat.S_IMODE(local_path.stat().st_mode) == 0o600
    assert (home_a / ".credential_suppressions.lock").is_file()
    assert not [path for path in home_a.iterdir() if path.name.endswith(".tmp")]
    shared_auth = residence / "auth.json"
    shared_state = (
        json.loads(shared_auth.read_text(encoding="utf-8"))
        if shared_auth.exists()
        else {}
    )
    assert source not in json.dumps(shared_state.get("suppressed_sources", {}))

    _activate_home(monkeypatch, home_b)
    assert not is_source_suppressed("openrouter", source)
    assert [entry.access_token for entry in load_pool("openrouter").entries()] == [
        "key-from-b"
    ]

    _activate_home(monkeypatch, home_a)
    assert is_source_suppressed("openrouter", source)
    assert load_pool("openrouter").entries() == []


def test_legacy_runtime_suppression_migrates_to_the_first_runtime(
    monkeypatch,
    tmp_path,
):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        is_source_suppressed,
    )

    residence = tmp_path / "residence"
    home_a = tmp_path / "runtime-a"
    home_b = tmp_path / "runtime-b"
    source = "env:OPENROUTER_API_KEY"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    _write_env(home_a, "OPENROUTER_API_KEY=key-from-a\n")
    _write_env(home_b, "OPENROUTER_API_KEY=key-from-b\n")
    _activate_home(monkeypatch, home_a)

    with _auth_store_lock():
        shared = _load_auth_store()
        shared["suppressed_sources"] = {
            "openrouter": [source],
            "openai-codex": ["device_code"],
        }
        _save_auth_store(shared)

    assert is_source_suppressed("openrouter", source)
    assert load_pool("openrouter").entries() == []
    local_path = home_a / ".credential_suppressions.json"
    local = json.loads(local_path.read_text(encoding="utf-8"))
    assert local["suppressed_sources"] == {"openrouter": [source]}
    if os.name != "nt":
        assert stat.S_IMODE(local_path.stat().st_mode) == 0o600

    shared = json.loads((residence / "auth.json").read_text(encoding="utf-8"))
    assert shared["suppressed_sources"] == {
        "openai-codex": ["device_code"]
    }

    _activate_home(monkeypatch, home_b)
    assert not is_source_suppressed("openrouter", source)
    assert [entry.access_token for entry in load_pool("openrouter").entries()] == [
        "key-from-b"
    ]
    assert not (home_b / ".credential_suppressions.json").exists()


def test_legacy_suppression_migration_cache_is_scoped_to_residence(
    monkeypatch,
    tmp_path,
):
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        is_source_suppressed,
    )

    runtime_home = tmp_path / "runtime"
    residence_a = tmp_path / "residence-a"
    residence_b = tmp_path / "residence-b"
    _activate_home(monkeypatch, runtime_home)

    for residence, source in (
        (residence_a, "env:OPENROUTER_API_KEY"),
        (residence_b, "config:Foo"),
    ):
        monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
        with _auth_store_lock():
            shared = _load_auth_store()
            shared["suppressed_sources"] = {"custom:foo": [source]}
            _save_auth_store(shared)

        assert is_source_suppressed("custom:foo", source)
        migrated = json.loads(
            (runtime_home / ".credential_suppressions.json").read_text(
                encoding="utf-8"
            )
        )
        assert source in migrated["suppressed_sources"]["custom:foo"]
        shared = json.loads(
            (residence / "auth.json").read_text(encoding="utf-8")
        )
        assert "suppressed_sources" not in shared


def test_durable_source_suppression_remains_in_the_auth_residence(
    monkeypatch, tmp_path
):
    from hermes_cli.auth import is_source_suppressed, suppress_credential_source

    residence = tmp_path / "residence"
    home_a = tmp_path / "runtime-a"
    home_b = tmp_path / "runtime-b"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    _activate_home(monkeypatch, home_a)
    suppress_credential_source("openai-codex", "device_code")
    shared_path = residence / "auth.json"
    shared = (
        json.loads(shared_path.read_text(encoding="utf-8"))
        if shared_path.exists()
        else {}
    )
    assert shared["suppressed_sources"] == {
        "openai-codex": ["device_code"]
    }
    assert not (home_a / ".credential_suppressions.json").exists()

    _activate_home(monkeypatch, home_b)
    assert is_source_suppressed("openai-codex", "device_code")


@pytest.mark.parametrize(
    ("source", "config_factory"),
    [
        (
            "config:Foo",
            lambda key: {
                "custom_providers": [
                    {
                        "name": "Foo",
                        "base_url": "https://example.test/v1",
                        "api_key": key,
                    }
                ]
            },
        ),
        (
            "model_config",
            lambda key: {
                "custom_providers": [
                    {
                        "name": "Foo",
                        "base_url": "https://example.test/v1",
                    }
                ],
                "model": {
                    "provider": "custom",
                    "base_url": "https://example.test/v1",
                    "api_key": key,
                },
            },
        ),
    ],
)
def test_config_suppression_does_not_cross_runtime_homes(
    monkeypatch, tmp_path, source, config_factory
):
    from types import SimpleNamespace

    from agent.credential_pool import load_pool
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_remove_command

    residence = tmp_path / "residence"
    home_a = tmp_path / "runtime-a"
    home_b = tmp_path / "runtime-b"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    _write_config(home_a, config_factory("key-from-a"))
    _write_config(home_b, config_factory("key-from-b"))

    _activate_home(monkeypatch, home_a)
    pool_a = load_pool("custom:foo")
    assert [(entry.source, entry.access_token) for entry in pool_a.entries()] == [
        (source, "key-from-a")
    ]
    auth_remove_command(SimpleNamespace(provider="custom:foo", target="1"))
    _write_config(home_a, config_factory("key-from-a"))
    assert is_source_suppressed("custom:foo", source)
    assert load_pool("custom:foo").entries() == []

    _activate_home(monkeypatch, home_b)
    assert not is_source_suppressed("custom:foo", source)
    pool_b = load_pool("custom:foo")
    assert [(entry.source, entry.access_token) for entry in pool_b.entries()] == [
        (source, "key-from-b")
    ]
    shared_path = residence / "auth.json"
    shared = (
        json.loads(shared_path.read_text(encoding="utf-8"))
        if shared_path.exists()
        else {}
    )
    assert source not in json.dumps(shared.get("credential_pool", {}))
    assert source not in json.dumps(shared.get("suppressed_sources", {}))


@pytest.mark.parametrize(
    ("source", "config"),
    [
        (
            "config:Foo",
            {
                "custom_providers": [
                    {
                        "name": "Foo",
                        "base_url": "https://example.test/v1",
                        "api_key": "key-from-b",
                    }
                ]
            },
        ),
        (
            "model_config",
            {
                "custom_providers": [
                    {
                        "name": "Foo",
                        "base_url": "https://example.test/v1",
                    }
                ],
                "model": {
                    "provider": "custom",
                    "base_url": "https://example.test/v1",
                    "api_key": "key-from-b",
                },
            },
        ),
    ],
)
def test_legacy_runtime_rows_are_filtered_before_pool_construction(
    monkeypatch, tmp_path, source, config
):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        read_credential_pool,
    )

    residence = tmp_path / "residence"
    home_b = tmp_path / "runtime-b"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    _write_config(home_b, config)
    _activate_home(monkeypatch, home_b)

    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("credential_pool", {})["custom:foo"] = [
            {
                "id": "runtime-a",
                "source": source,
                "auth_type": "api_key",
                "label": source,
                "priority": 0,
                "secret_fingerprint": "sha256:runtime-a",
                "last_status": "dead",
                "last_status_at": 1,
                "last_error_code": 401,
                "last_error_reason": "invalid_token",
            }
        ]
        _save_auth_store(store)

    assert read_credential_pool("custom:foo") == []
    pool_b = load_pool("custom:foo")
    assert len(pool_b.entries()) == 1
    assert pool_b.entries()[0].access_token == "key-from-b"
    assert pool_b.entries()[0].last_status is None
    assert _persisted_pool(residence / "auth.json")["custom:foo"] == []


@pytest.mark.parametrize(
    ("status_code", "error_context", "expected_status"),
    [
        (429, {"reason": "rate_limit_exceeded"}, "exhausted"),
        (401, {"reason": "invalid_token"}, "dead"),
    ],
)
@pytest.mark.parametrize("source", ["config:Foo", "model_config"])
def test_config_credential_failures_do_not_cross_runtime_homes(
    monkeypatch, tmp_path, source, status_code, error_context, expected_status
):
    from agent.credential_pool import load_pool

    residence = tmp_path / "residence"
    home_a = tmp_path / "runtime-a"
    home_b = tmp_path / "runtime-b"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    def config(key):
        custom_provider = {
            "name": "Foo",
            "base_url": "https://example.test/v1",
        }
        result = {"custom_providers": [custom_provider]}
        if source == "config:Foo":
            custom_provider["api_key"] = key
        else:
            result["model"] = {
                "provider": "custom",
                "base_url": "https://example.test/v1",
                "api_key": key,
            }
        return result

    _write_config(home_a, config("key-from-a"))
    _write_config(home_b, config("key-from-b"))

    _activate_home(monkeypatch, home_a)
    pool_a = load_pool("custom:foo")
    assert [entry.source for entry in pool_a.entries()] == [source]
    pool_a.mark_exhausted_and_rotate(
        status_code=status_code,
        error_context=error_context,
        api_key_hint="key-from-a",
    )
    assert pool_a.entries()[0].last_status == expected_status
    assert _persisted_pool(residence / "auth.json").get("custom:foo", []) == []

    _activate_home(monkeypatch, home_b)
    pool_b = load_pool("custom:foo")
    assert len(pool_b.entries()) == 1
    assert pool_b.entries()[0].access_token == "key-from-b"
    assert pool_b.entries()[0].last_status is None
    assert pool_b.has_available()
    assert pool_b.peek() is not None


def test_foreign_persisted_env_rows_are_scrubbed_once_then_stable(
    monkeypatch, tmp_path
):
    """A legacy env row on disk is filtered before construction and removed.

    The removal happens exactly once — after the store is clean, repeated
    loads must not rewrite bytes, mtime, or ``updated_at``.
    """
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

    residence = tmp_path / "residence"
    home_b = tmp_path / "runtime-b"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    _write_env(home_b, "")
    _activate_home(monkeypatch, home_b)
    auth_file = residence / "auth.json"

    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("credential_pool", {})["openrouter"] = [
            {
                "id": "foreign-env",
                "source": "env:OPENROUTER_API_KEY",
                "auth_type": "api_key",
                "label": "OPENROUTER_API_KEY",
                "access_token": "someone-elses-secret",
                "priority": 0,
            },
            {
                "id": "manual-1",
                "source": "manual",
                "auth_type": "api_key",
                "label": "manual key",
                "access_token": "sk-or-manual",
                "priority": 1,
            },
        ]
        _save_auth_store(store)

    pool = load_pool("openrouter")
    assert [entry.id for entry in pool.entries()] == ["manual-1"]
    persisted = _persisted_pool(auth_file)["openrouter"]
    assert [entry["id"] for entry in persisted] == ["manual-1"]

    before_bytes = auth_file.read_bytes()
    before_mtime = auth_file.stat().st_mtime_ns
    reloaded = load_pool("openrouter")
    assert [entry.id for entry in reloaded.entries()] == ["manual-1"]
    assert auth_file.read_bytes() == before_bytes
    assert auth_file.stat().st_mtime_ns == before_mtime


def test_cooldown_persistence_never_republishes_env_rows(monkeypatch, tmp_path):
    """Real pool mutations keep env rows out of the shared store.

    Exhaustion marking goes through ``_persist`` — not through a direct
    ``write_credential_pool`` call — so this proves the disk boundary holds
    on the live rotation path. The owning session keeps the cooldown on its
    in-memory env entry.
    """
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool
    from hermes_cli.auth import write_credential_pool

    residence = tmp_path / "residence"
    home = tmp_path / "runtime-a"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    _write_env(home, "OPENROUTER_API_KEY=key-from-a\n")
    _activate_home(monkeypatch, home)
    auth_file = residence / "auth.json"

    write_credential_pool(
        "openrouter",
        [
            {
                "id": "manual-1",
                "source": "manual",
                "auth_type": "api_key",
                "label": "manual key",
                "access_token": "sk-or-manual",
                "priority": 0,
            }
        ],
    )

    pool = load_pool("openrouter")
    assert {entry.source for entry in pool.entries()} == {
        "manual",
        "env:OPENROUTER_API_KEY",
    }

    # A dirty persist: the manual entry's cooldown must land on disk while
    # the env row stays memory-only.
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "rate_limit_exceeded"},
        api_key_hint="sk-or-manual",
    )
    persisted = _persisted_pool(auth_file)["openrouter"]
    assert [entry["id"] for entry in persisted] == ["manual-1"]
    assert persisted[0]["last_status"] == STATUS_EXHAUSTED
    assert "env:" not in json.dumps(persisted)

    # Exhausting the env entry itself cools it down in memory only.
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "rate_limit_exceeded"},
        api_key_hint="key-from-a",
    )
    env_entry = next(
        entry for entry in pool.entries()
        if entry.source == "env:OPENROUTER_API_KEY"
    )
    assert env_entry.last_status == STATUS_EXHAUSTED
    persisted = _persisted_pool(auth_file)["openrouter"]
    assert "env:" not in json.dumps(persisted)


def test_write_credential_pool_filters_runtime_rows_at_the_disk_boundary(
    monkeypatch, tmp_path
):
    """Both caller entries and disk-merge entries are filtered."""
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        write_credential_pool,
    )

    residence = tmp_path / "residence"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    auth_file = residence / "auth.json"

    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("credential_pool", {})["openrouter"] = [
            {
                "id": "disk-env",
                "source": "env:OPENROUTER_API_KEY",
                "auth_type": "api_key",
                "label": "OPENROUTER_API_KEY",
                "access_token": "disk-secret",
                "priority": 0,
            },
        ]
        _save_auth_store(store)

    entries = [
        {
            "id": "manual-1",
            "source": "manual",
            "auth_type": "api_key",
            "label": "manual key",
            "access_token": "sk-or-manual",
            "priority": 0,
        },
        {
            "id": "session-env",
            "source": "env:OPENROUTER_API_KEY",
            "auth_type": "api_key",
            "label": "OPENROUTER_API_KEY",
            "access_token": "session-secret",
            "priority": 1,
        },
        {
            "id": "session-config",
            "source": "config:Foo",
            "auth_type": "api_key",
            "label": "Foo",
            "access_token": "config-secret",
            "priority": 2,
        },
        {
            "id": "session-model",
            "source": "model_config",
            "auth_type": "api_key",
            "label": "model_config",
            "access_token": "model-secret",
            "priority": 3,
        },
    ]
    path = write_credential_pool("openrouter", [dict(e) for e in entries])
    assert path == residence.resolve() / "auth.json"
    persisted = _persisted_pool(auth_file)["openrouter"]
    assert [entry["id"] for entry in persisted] == ["manual-1"]
    serialized = json.dumps(persisted)
    assert "env:" not in serialized
    assert "config:" not in serialized
    assert "model_config" not in serialized

    # An unchanged persistable projection is a byte/mtime/updated_at no-op.
    before_bytes = auth_file.read_bytes()
    before_mtime = auth_file.stat().st_mtime_ns
    before_updated_at = json.loads(before_bytes)["updated_at"]
    write_credential_pool("openrouter", [dict(e) for e in entries])
    assert auth_file.read_bytes() == before_bytes
    assert auth_file.stat().st_mtime_ns == before_mtime
    assert json.loads(auth_file.read_bytes())["updated_at"] == before_updated_at


def test_no_override_home_keeps_persisting_env_rows(monkeypatch, tmp_path):
    """Without a distinct residence the store and .env share one session."""
    from agent.credential_pool import load_pool

    home = tmp_path / "runtime"
    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    _write_env(home, "OPENROUTER_API_KEY=key-local\n")
    _activate_home(monkeypatch, home)

    pool = load_pool("openrouter")
    sources = [entry.source for entry in pool.entries()]
    assert "env:OPENROUTER_API_KEY" in sources
    persisted = _persisted_pool(home / "auth.json")["openrouter"]
    assert any(
        entry.get("source") == "env:OPENROUTER_API_KEY" for entry in persisted
    )


def test_path_equal_override_keeps_persisting_env_rows(monkeypatch, tmp_path):
    """A path-equal override is a total no-op, not a session split."""
    from agent.credential_pool import load_pool

    home = tmp_path / "runtime"
    _write_env(home, "OPENROUTER_API_KEY=key-local\n")
    _activate_home(monkeypatch, home)
    monkeypatch.setenv("HERMES_AUTH_HOME", str(home))

    pool = load_pool("openrouter")
    assert any(
        entry.source == "env:OPENROUTER_API_KEY" for entry in pool.entries()
    )
    persisted = _persisted_pool(home / "auth.json")["openrouter"]
    assert any(
        entry.get("source") == "env:OPENROUTER_API_KEY" for entry in persisted
    )
