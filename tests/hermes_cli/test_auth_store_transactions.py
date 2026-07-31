"""Transaction identity for the three credential stores.

Each primary-auth, Anthropic-OAuth, and shared-Nous transaction must resolve
its store path once, then use that pinned path for both the lock and the data
operations — a symlinked residence retargeted mid-transaction must not split
the lock from the data file. Later, independent transactions are free to
follow the new target; symlinked/mounted layouts stay supported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_constants import HermesAuthHomeError


@pytest.fixture
def residence_link(monkeypatch, tmp_path):
    """A credential residence reached through a launcher-owned symlink."""
    store_a = tmp_path / "store-a"
    store_b = tmp_path / "store-b"
    store_a.mkdir()
    store_b.mkdir()
    link = tmp_path / "residence"
    link.symlink_to(store_a, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(link))
    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)
    return link, store_a.resolve(), store_b.resolve()


def _retarget(link: Path, target: Path) -> None:
    link.unlink()
    link.symlink_to(target, target_is_directory=True)


def test_primary_transaction_pins_data_and_lock_to_one_resolution(residence_link):
    from hermes_cli.auth import (
        _auth_file_path,
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        _store_provider_state,
    )

    link, store_a, store_b = residence_link
    with _auth_store_lock() as pinned:
        assert pinned == store_a / "auth.json"
        assert (store_a / "auth.lock").exists()
        _retarget(link, store_b)
        # The transaction keeps using the path its lock protects.
        assert _auth_file_path() == pinned
        store = _load_auth_store()
        _store_provider_state(store, "nous", {"value": "pinned"}, set_active=False)
        _save_auth_store(store)

    persisted = json.loads((store_a / "auth.json").read_text(encoding="utf-8"))
    assert persisted["providers"]["nous"] == {"value": "pinned"}
    assert not (store_b / "auth.json").exists()

    # A later, independent transaction may follow the retargeted link.
    with _auth_store_lock() as repinned:
        assert repinned == store_b / "auth.json"
        assert (store_b / "auth.lock").exists()
        store = _load_auth_store()
        _store_provider_state(store, "nous", {"value": "moved"}, set_active=False)
        _save_auth_store(store)
    moved = json.loads((store_b / "auth.json").read_text(encoding="utf-8"))
    assert moved["providers"]["nous"] == {"value": "moved"}
    # The first store was left exactly as the first transaction wrote it.
    assert json.loads(
        (store_a / "auth.json").read_text(encoding="utf-8")
    )["providers"]["nous"] == {"value": "pinned"}


def test_anthropic_transaction_pins_path_across_retarget(residence_link):
    from hermes_cli.auth import anthropic_oauth_store_lock

    link, store_a, store_b = residence_link
    with anthropic_oauth_store_lock() as pinned:
        assert pinned == store_a / ".anthropic_oauth.json"
        assert (store_a / ".anthropic_oauth.lock").exists()
        _retarget(link, store_b)
        pinned.write_text(json.dumps({"accessToken": "pinned"}), encoding="utf-8")

    saved = json.loads(
        (store_a / ".anthropic_oauth.json").read_text(encoding="utf-8")
    )
    assert saved["accessToken"] == "pinned"
    assert not (store_b / ".anthropic_oauth.json").exists()

    with anthropic_oauth_store_lock() as repinned:
        assert repinned == store_b / ".anthropic_oauth.json"


def test_shared_nous_transaction_pins_path_across_retarget(residence_link):
    from hermes_cli.auth import (
        _nous_shared_store_lock,
        _read_shared_nous_state,
        _write_shared_nous_state,
    )

    link, store_a, store_b = residence_link
    with _nous_shared_store_lock() as pinned:
        assert pinned == store_a / "shared" / "nous_auth.json"
        _retarget(link, store_b)
        _write_shared_nous_state(
            {"access_token": "pinned-access", "refresh_token": "pinned-refresh"}
        )
        shared = _read_shared_nous_state()
        assert (shared or {}).get("access_token") == "pinned-access"

    assert (store_a / "shared" / "nous_auth.json").is_file()
    assert not (store_b / "shared").exists()

    with _nous_shared_store_lock() as repinned:
        assert repinned == store_b / "shared" / "nous_auth.json"
        assert _read_shared_nous_state() is None


def test_nested_explicit_global_lock_keeps_outer_default_target(monkeypatch, tmp_path):
    """An explicitly targeted inner transaction must not retarget the outer one.

    Profile-to-global write-through opens ``_auth_store_lock(target_path=...)``
    inside a default transaction; if the nested lock replaced the pinned
    default path, the outer save would land in the global store.
    """
    from hermes_cli.auth import (
        _auth_file_path,
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        _store_provider_state,
    )

    residence = tmp_path / "residence"
    global_store = tmp_path / "global" / "auth.json"
    global_store.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    expected = residence.resolve() / "auth.json"
    with _auth_store_lock() as outer:
        assert outer == expected
        with _auth_store_lock(target_path=global_store) as inner:
            assert inner == global_store.resolve()
            assert _auth_file_path() == outer
        assert _auth_file_path() == outer
        store = _load_auth_store()
        _store_provider_state(store, "nous", {"value": "outer"}, set_active=False)
        assert _save_auth_store(store) == outer

    persisted = json.loads(expected.read_text(encoding="utf-8"))
    assert persisted["providers"]["nous"] == {"value": "outer"}
    assert not global_store.exists()


def test_shared_nous_stays_at_default_root_for_path_equal_override(
    monkeypatch, tmp_path
):
    """Only a genuinely distinct residence relocates the shared store."""
    from hermes_cli.auth import _nous_shared_auth_dir

    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)
    operator = tmp_path / "operator"
    root = operator / ".hermes"
    profile_home = root / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(operator))
    monkeypatch.setattr(Path, "home", lambda: operator)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    assert _nous_shared_auth_dir() == root / "shared"

    # Spelling out the directory already in use — profile home or root — is a
    # total no-op for the shared store.
    monkeypatch.setenv("HERMES_AUTH_HOME", str(profile_home))
    assert _nous_shared_auth_dir() == root / "shared"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(root))
    assert _nous_shared_auth_dir() == root / "shared"

    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert _nous_shared_auth_dir() == residence.resolve() / "shared"


def test_shared_nous_dir_resolves_against_an_explicit_runtime_home(
    monkeypatch, tmp_path
):
    """Lifecycle/backup callers scope resolution to a home they name.

    The ambient active profile must not leak into the mapping: a named
    profile home maps to its own root's ``shared/``, a default/custom root
    maps to itself, a path-equal override changes nothing, and only a
    genuinely distinct residence relocates the store.
    """
    from hermes_cli.auth import _nous_shared_auth_dir

    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)
    root = tmp_path / "root"
    profile_home = root / "profiles" / "work"
    profile_home.mkdir(parents=True)
    custom_home = tmp_path / "opt-data"
    custom_home.mkdir()
    # Ambient runtime state deliberately points elsewhere.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-runtime"))

    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    assert _nous_shared_auth_dir(profile_home) == root / "shared"
    assert _nous_shared_auth_dir(custom_home) == custom_home / "shared"

    # Path-equal overrides — profile home or its root — are a no-op.
    monkeypatch.setenv("HERMES_AUTH_HOME", str(profile_home))
    assert _nous_shared_auth_dir(profile_home) == root / "shared"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(root))
    assert _nous_shared_auth_dir(profile_home) == root / "shared"

    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    assert _nous_shared_auth_dir(profile_home) == residence.resolve() / "shared"
    assert _nous_shared_auth_dir(custom_home) == residence.resolve() / "shared"

    # HERMES_SHARED_AUTH_DIR outranks everything, explicit home included.
    shared_dir = tmp_path / "explicit-shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    assert _nous_shared_auth_dir(profile_home) == shared_dir


@pytest.mark.parametrize("value", ("", "   ", "relative/auth", "~/auth"))
def test_strict_credential_consumers_reject_invalid_residence(
    monkeypatch, tmp_path, value
):
    """Actual-I/O consumers raise instead of falling back to HERMES_HOME."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    monkeypatch.setenv("HERMES_AUTH_HOME", value)

    from agent.anthropic_adapter import _get_hermes_oauth_file
    from agent.auxiliary_client import _auth_json_path
    from hermes_cli.auth import _auth_file_path, anthropic_oauth_store_lock
    from tools.managed_tool_gateway import auth_json_path

    for consumer in (
        _auth_file_path,
        _get_hermes_oauth_file,
        _auth_json_path,
        auth_json_path,
    ):
        with pytest.raises(HermesAuthHomeError):
            consumer()
    with pytest.raises(HermesAuthHomeError):
        with anthropic_oauth_store_lock():
            pass

    # Availability probes fail closed — they must not read the runtime-home
    # store the launcher asked to stop using.
    (runtime / "auth.json").write_text(
        json.dumps(
            {"providers": {"xai-oauth": {"tokens": {"access_token": "tok"}}}}
        ),
        encoding="utf-8",
    )
    from tools.xai_http import has_xai_credentials

    assert not has_xai_credentials()

    # Total classifiers stay usable so import-time guards never crash.
    from agent.file_safety import build_write_denied_paths, get_read_block_error

    assert get_read_block_error(str(tmp_path / "notes.txt")) is None
    assert build_write_denied_paths(str(tmp_path))
