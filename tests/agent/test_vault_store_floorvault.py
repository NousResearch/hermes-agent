"""FloorVault store (``vault.store: floorvault``): same contract as the Fernet ``VaultStore``,
contextual binding, fail-closed key custody, and the Fernet -> FloorVault migration."""

import secrets
import sqlite3
from unittest.mock import patch

import pytest

pytest.importorskip("floorvault")

from floorvault import FloorVault

from agent.vault_store import VaultError, VaultItemMeta, VaultStore
from agent.vault_store_floorvault import (
    APP_INSTANCE_ID,
    FloorVaultStore,
    migrate_from_fernet,
)

LOGIN = {"identifier_type": "email", "identifier": "u@example.com", "password": "hunter2-pw",
         "otp_secret": "JBSWY3DPEHPK3PXP"}
CARD = {"card_number": "4111111111111111", "exp_month": "12", "exp_year": "2030", "cvc": "123"}


def _store(path, key=None) -> FloorVaultStore:
    crypto = FloorVault(key or secrets.token_bytes(32), app_instance_id=APP_INSTANCE_ID)
    return FloorVaultStore(path, crypto=crypto)


def test_crud_matches_fernet_contract(tmp_path):
    store = _store(tmp_path / "vault")
    login = store.add_item("login", "Example", dict(LOGIN), origin="https://example.com:443/login")
    card = store.add_item("payment", "Visa", dict(CARD))

    assert isinstance(login, VaultItemMeta) and login.id.startswith("vault_")
    assert login.origin == "https://example.com" and login.identifier == "u@example.com" and login.has_otp
    assert {m.id for m in store.list_items()} == {login.id, card.id}
    assert store.resolve_secret(login.id) == {"password": "hunter2-pw", "otp_secret": "JBSWY3DPEHPK3PXP"}
    assert store.resolve_secret(card.id) == CARD
    assert store.get_meta("vault_missing") is None
    assert store.remove_item(card.id) and not store.remove_item(card.id)
    assert store.has_items()


def test_nothing_sensitive_is_stored_in_plaintext(tmp_path):
    store = _store(tmp_path / "vault")
    store.add_item("login", "Example label", dict(LOGIN), origin="https://example.com")
    blob = (tmp_path / "vault" / "vault.db").read_bytes()
    for needle in (b"hunter2-pw", b"JBSWY3DPEHPK3PXP", b"u@example.com", b"example.com", b"Example label"):
        assert needle not in blob


def test_ciphertext_moved_to_another_record_fails_closed(tmp_path):
    store = _store(tmp_path / "vault")
    a = store.add_item("login", "A", dict(LOGIN), origin="https://a.example")
    b = store.add_item("login", "B", {**LOGIN, "password": "other"}, origin="https://b.example")
    with sqlite3.connect(tmp_path / "vault" / "vault.db") as conn:
        conn.execute("UPDATE vault_items SET payload_cipher = (SELECT payload_cipher FROM vault_items WHERE id = ?)"
                     " WHERE id = ?", (a.id, b.id))
    with pytest.raises(VaultError, match="failed authentication"):
        store.resolve_secret(b.id)


def test_wrong_key_cannot_read(tmp_path):
    item = _store(tmp_path / "vault").add_item("login", "A", dict(LOGIN), origin="https://a.example")
    with pytest.raises(VaultError):
        _store(tmp_path / "vault").resolve_secret(item.id)


def test_database_copy_cannot_cross_hermes_profiles(tmp_path, monkeypatch):
    # Headless profiles may intentionally share one bootstrap key; the
    # app-instance binding must still prevent a copied database from opening.
    key = secrets.token_hex(32)
    monkeypatch.setenv("FLOOR_VAULT_KEY", key)
    monkeypatch.setenv("HERMES_VAULT_KEY", key)
    source = FloorVaultStore(tmp_path / "profile-a" / "vault")
    item = source.add_item("login", "Example", dict(LOGIN), origin="https://example.com")
    copied = FloorVaultStore(tmp_path / "profile-b" / "vault")
    (copied._base / "vault.db").write_bytes((source._base / "vault.db").read_bytes())

    with pytest.raises(VaultError, match="failed authentication"):
        copied.resolve_secret(item.id)


def test_namespaced_key_takes_precedence_over_legacy_appstate_key(tmp_path, monkeypatch):
    monkeypatch.setenv("APPSTATE_KEY", secrets.token_hex(32))
    monkeypatch.setenv("FLOOR_VAULT_KEY", secrets.token_hex(32))
    store = FloorVaultStore(tmp_path / "profile" / "vault")
    item = store.add_item("login", "Example", dict(LOGIN), origin="https://example.com")

    monkeypatch.setenv("APPSTATE_KEY", secrets.token_hex(32))
    reopened = FloorVaultStore(tmp_path / "profile" / "vault")
    assert reopened.resolve_secret(item.id)["password"] == LOGIN["password"]


def test_validation_errors_are_vault_errors(tmp_path):
    store = _store(tmp_path / "vault")
    with pytest.raises(VaultError):
        store.add_item("login", "No origin", dict(LOGIN))
    with pytest.raises(VaultError):
        store.add_item("payment", "Partial", {"card_number": "4111"})


def test_headless_without_key_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOOR_VAULT_KEY", raising=False)
    monkeypatch.delenv("VAULT_MASTER_KEY", raising=False)
    with patch("floorvault.providers.adaptive.AdaptiveKeyProvider.resolve_key",
               side_effect=__import__("floorvault").KeyProviderError("no OS key store")):
        with pytest.raises(VaultError, match="master key unavailable"):
            FloorVaultStore(tmp_path / "vault")
    assert not list((tmp_path / "vault").glob("*.key"))


def test_env_key_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOOR_VAULT_KEY", secrets.token_hex(32))
    item = FloorVaultStore(tmp_path / "vault").add_item("login", "A", dict(LOGIN), origin="https://a.example")
    assert FloorVaultStore(tmp_path / "vault").resolve_secret(item.id)["password"] == "hunter2-pw"


def test_malformed_env_key_reports_the_real_cause(tmp_path, monkeypatch):
    # A set-but-malformed key must not read back as "is not set": the mapped
    # VaultError keeps floorvault's own message, which names the fix.
    monkeypatch.delenv("APPSTATE_KEY", raising=False)
    monkeypatch.delenv("VAULT_MASTER_KEY", raising=False)
    monkeypatch.setenv("FLOOR_VAULT_KEY", "not-hex")
    with pytest.raises(VaultError, match="64 hexadecimal"):
        FloorVaultStore(tmp_path / "vault")


def test_generic_kind_is_rejected(tmp_path):
    # floorvault accepts a 4th kind ("generic") the Fernet store never writes;
    # letting it through would strand a ghost item if the user reverts.
    store = _store(tmp_path / "vault")
    with pytest.raises(VaultError, match="unknown vault kind"):
        store.add_item("generic", "G", {"anything": "goes"})


def test_has_items_swallows_failures_like_fernet(tmp_path, monkeypatch):
    store = _store(tmp_path / "vault")
    monkeypatch.setattr(store._store, "has_items",
                        lambda: (_ for _ in ()).throw(Exception("db gone")))
    assert store.has_items() is False


@pytest.mark.platforms("posix")
def test_vault_db_is_owner_only(tmp_path):
    _store(tmp_path / "vault")
    import stat
    assert stat.S_IMODE((tmp_path / "vault" / "vault.db").stat().st_mode) == 0o600


def test_vault_store_key_is_a_known_config_key():
    # Without the registry entry, `hermes config set/get vault.store` warns the
    # key is unrecognized even though the runtime reads it.
    from hermes_cli.config import _validate_config_key
    assert _validate_config_key("vault.store") == (True, None)


def test_config_selects_store(tmp_path, monkeypatch):
    import agent.vault_store as vs
    import agent.vault_store_floorvault as fv

    monkeypatch.setattr(vs, "get_hermes_home", lambda: str(tmp_path))
    monkeypatch.setattr(vs, "configured_store_name", lambda: "floorvault")
    sentinel = object()
    monkeypatch.setattr(fv, "get_floorvault_store", lambda base: sentinel if base == tmp_path / "vault" else None)
    assert vs.get_vault_store() is sentinel
    monkeypatch.setattr(vs, "configured_store_name", lambda: "fernet")
    assert isinstance(vs.get_vault_store(), VaultStore)


def test_unknown_store_name_is_rejected(monkeypatch):
    import agent.vault_store as vs

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"vault": {"store": "plaintext"}})
    with pytest.raises(VaultError, match="unknown vault.store"):
        vs.configured_store_name()


def test_migration_keeps_handles_and_verifies(tmp_path):
    legacy = VaultStore(tmp_path / "vault")
    login = legacy.add_item("login", "Example", dict(LOGIN), origin="https://example.com")
    card = legacy.add_item("payment", "Visa", dict(CARD))
    target = _store(tmp_path / "vault")

    result = migrate_from_fernet(legacy, target)
    assert result == {"total": 2, "copied": 2, "skipped": 0, "verified": True, "mismatched": []}
    assert target.resolve_secret(login.id) == legacy.resolve_secret(login.id)
    assert target.resolve_secret(card.id) == CARD
    assert target.get_meta(login.id).identifier == "u@example.com"
    # Legacy files are untouched; a rerun is a no-op.
    assert legacy.resolve_secret(login.id)["password"] == "hunter2-pw"
    assert migrate_from_fernet(legacy, target) == {"total": 2, "copied": 0, "skipped": 2, "verified": True,
                                                  "mismatched": []}


def test_migration_reports_mismatch(tmp_path):
    legacy = VaultStore(tmp_path / "vault")
    login = legacy.add_item("login", "Example", dict(LOGIN), origin="https://example.com")
    target = _store(tmp_path / "vault")
    target.add_item("login", "Example", {**LOGIN, "password": "different"}, origin="https://example.com",
                    item_id=login.id)
    result = migrate_from_fernet(legacy, target)
    assert result["verified"] is False and result["mismatched"] == [login.id]


def test_migration_detects_label_mismatch(tmp_path):
    legacy = VaultStore(tmp_path / "vault")
    login = legacy.add_item("login", "Example", dict(LOGIN), origin="https://example.com")
    target = _store(tmp_path / "vault")
    target.add_item("login", "Different label", dict(LOGIN), origin="https://example.com",
                    item_id=login.id)
    assert migrate_from_fernet(legacy, target)["mismatched"] == [login.id]


@pytest.mark.parametrize('parameters,canonical', [
    ('digits=8', 'JBSWY3DPEHPK3PXP|8|30|SHA1'),
    ('period=60&algorithm=SHA256', 'JBSWY3DPEHPK3PXP|6|60|SHA256'),
    ('digits=8&period=60&algorithm=SHA512', 'JBSWY3DPEHPK3PXP|8|60|SHA512'),
])
def test_migration_preserves_custom_totp_parameters(tmp_path, parameters, canonical):
    legacy = VaultStore(tmp_path / 'vault')
    item = legacy.add_item('login', 'Custom OTP', {
        **LOGIN, 'otp_secret': f'otpauth://totp/Test?secret=JBSWY3DPEHPK3PXP&{parameters}',
    }, origin='https://example.com')
    target = _store(tmp_path / 'vault')
    assert migrate_from_fernet(legacy, target)['verified']
    assert target.resolve_secret(item.id)['otp_secret'] == canonical


def test_migration_rerun_does_not_resurrect_deleted_items(tmp_path):
    legacy = VaultStore(tmp_path / 'vault')
    item = legacy.add_item('login', 'Example', dict(LOGIN), origin='https://example.com')
    key = secrets.token_bytes(32)
    target = _store(tmp_path / 'vault', key)
    assert migrate_from_fernet(legacy, target)['verified']
    assert target.remove_item(item.id)
    # Reopen, as a later CLI invocation would: retirement must survive process exit.
    reopened = _store(tmp_path / 'vault', key)
    assert migrate_from_fernet(legacy, reopened) == {
        'total': 1, 'copied': 0, 'skipped': 1, 'verified': True, 'mismatched': [],
    }
    assert reopened.get_meta(item.id) is None


def test_cli_cutover_rejects_waiting_fernet_writer(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from agent.vault_store_floorvault import get_floorvault_store
    from hermes_cli import config, vault

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('APPSTATE_KEY', raising=False)
    monkeypatch.setenv('FLOOR_VAULT_KEY', secrets.token_hex(32))
    legacy = VaultStore()
    item = legacy.add_item('login', 'Original', dict(LOGIN), origin='https://example.com')
    stale = VaultStore()
    outcomes = []
    started = threading.Event()

    def writer():
        started.set()
        try:
            outcomes.append(stale.add_item('login', 'Late', dict(LOGIN), origin='https://example.com'))
        except VaultError as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=writer)
    save_config = config.save_config

    def save_and_start(cfg):
        save_config(cfg)
        thread.start()
        assert started.wait(5)

    monkeypatch.setattr(config, 'save_config', save_and_start)
    try:
        vault._cmd_migrate_floorvault(SimpleNamespace(remove_legacy=True))
    finally:
        if thread.ident is not None:
            thread.join(5)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], VaultError)
    assert get_floorvault_store(tmp_path / 'vault').get_meta(item.id) is not None
    assert not legacy._vault_path.exists()
    assert not legacy._key_path.exists()


def test_fernet_write_guard_uses_own_home_and_allows_explicit_revert(tmp_path):
    from hermes_cli.config import atomic_config_write

    owner = tmp_path / 'other-profile'
    legacy = VaultStore(owner / 'vault')
    item = legacy.add_item('login', 'Example', dict(LOGIN), origin='https://example.com')
    atomic_config_write(owner / 'config.yaml', {'vault': {'store': 'floorvault'}})
    with pytest.raises(VaultError):
        legacy.add_item('login', 'Stale', dict(LOGIN), origin='https://example.com')
    with pytest.raises(VaultError):
        legacy.remove_item(item.id)
    atomic_config_write(owner / 'config.yaml', {'vault': {'store': 'fernet'}})
    assert legacy.remove_item(item.id)


def test_fernet_write_guard_ignores_frozen_worker_config(tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_write

    legacy = VaultStore(tmp_path / 'vault')
    legacy.add_item('login', 'Original', dict(LOGIN), origin='https://example.com')
    atomic_config_write(tmp_path / 'config.yaml', {'vault': {'store': 'floorvault'}})
    monkeypatch.setattr('agent.safe_worker_policy.worker_config_snapshot',
                        lambda: {'vault': {'store': 'fernet'}})
    with pytest.raises(VaultError):
        legacy.add_item('login', 'Frozen worker', dict(LOGIN), origin='https://example.com')


def test_cli_preserves_legacy_if_config_save_does_not_activate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import vault

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('APPSTATE_KEY', raising=False)
    monkeypatch.setenv('FLOOR_VAULT_KEY', secrets.token_hex(32))
    legacy = VaultStore()
    item = legacy.add_item('login', 'Original', dict(LOGIN), origin='https://example.com')
    # save_config also returns normally without writing on managed installations.
    monkeypatch.setattr('hermes_cli.config.save_config', lambda cfg: None)
    with pytest.raises(VaultError):
        vault._cmd_migrate_floorvault(SimpleNamespace(remove_legacy=True))
    assert legacy._key_path.exists()
    assert legacy.resolve_secret(item.id)['password'] == 'hunter2-pw'


@pytest.mark.platforms('posix')
def test_verified_callback_holds_cross_process_legacy_lock(tmp_path):
    import subprocess
    import sys

    legacy = VaultStore(tmp_path / 'vault')
    legacy.add_item('login', 'Original', dict(LOGIN), origin='https://example.com')
    target = _store(tmp_path / 'vault')
    blocked = []

    def check_lock():
        probe = subprocess.run([sys.executable, '-c', '''
import fcntl, sys
with open(sys.argv[1], 'a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(0)
sys.exit(1)
''', str(tmp_path / 'vault' / '.vault.lock')], capture_output=True, timeout=10, check=False)
        blocked.append(probe.returncode == 0)

    assert migrate_from_fernet(legacy, target, on_verified=check_lock)['verified']
    assert blocked == [True]


@pytest.mark.parametrize("exc", [
    ValueError("master_key must be exactly 32 bytes (got 64)"),
    OSError("permission denied"),
    RuntimeError("Attempted to access wiped HardenedMemoryKey"),
])
def test_foreign_store_failures_map_to_vault_error(tmp_path, exc):
    # floorvault internals surface stdlib errors — a malformed OS-custody blob
    # raises ValueError inside the key container, fs failures raise OSError — and
    # callers only handle VaultError, so none of these may pass through raw.
    with patch("floorvault.providers.adaptive.AdaptiveKeyProvider.resolve_key", side_effect=exc):
        with pytest.raises(VaultError):
            FloorVaultStore(tmp_path / "vault")


def test_cli_legacy_removal_failure_surfaces_as_vault_error(tmp_path, monkeypatch):
    # `--remove-legacy` unlink failures must not produce a raw traceback; the
    # verified config flip still stands (legacy files remaining is the safe side).
    from pathlib import Path
    from types import SimpleNamespace
    from agent.vault_store import configured_store_name
    from hermes_cli import vault

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("APPSTATE_KEY", raising=False)
    monkeypatch.setenv("FLOOR_VAULT_KEY", secrets.token_hex(32))
    legacy = VaultStore()
    legacy.add_item("login", "Original", dict(LOGIN), origin="https://example.com")

    real_unlink = Path.unlink

    def selective_unlink(self, missing_ok=False):
        if self.name in {"vault.json.enc", "vault.key"}:
            raise OSError("read-only filesystem")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", selective_unlink)
    with pytest.raises(VaultError, match="post-verification step failed"):
        vault._cmd_migrate_floorvault(SimpleNamespace(remove_legacy=True))
    assert configured_store_name(tmp_path / "config.yaml") == "floorvault"
    assert (tmp_path / "vault" / "vault.db").exists()


def test_cli_verification_failure_names_the_mismatched_handle(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    from rich.console import Console
    from agent.vault_store import configured_store_name
    from agent.vault_store_floorvault import get_floorvault_store
    from hermes_cli import vault

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("APPSTATE_KEY", raising=False)
    monkeypatch.setenv("FLOOR_VAULT_KEY", secrets.token_hex(32))
    legacy = VaultStore()
    item = legacy.add_item("login", "Original", dict(LOGIN), origin="https://example.com")
    get_floorvault_store(tmp_path / "vault").add_item(
        "login", "Original", {**LOGIN, "password": "different"}, origin="https://example.com",
        item_id=item.id)
    recorder = Console(record=True, file=io.StringIO(), force_terminal=False)
    monkeypatch.setattr(vault, "_console", lambda: recorder)
    vault._cmd_migrate_floorvault(SimpleNamespace(remove_legacy=True))
    out = recorder.export_text()
    assert "Verification failed" in out and item.id in out
    assert configured_store_name(tmp_path / "config.yaml") == "fernet"


def test_cli_error_output_preserves_bracketed_names(monkeypatch):
    # FloorVault messages carry "[macos]"/"[linux]" extra names; printed raw, Rich
    # markup swallows them — escape so the user keeps the actionable part.
    import io
    from types import SimpleNamespace
    from rich.console import Console
    from hermes_cli import vault

    recorder = Console(record=True, file=io.StringIO(), force_terminal=False)
    monkeypatch.setattr(vault, "_console", lambda: recorder)

    def boom(_args):
        raise VaultError("install floorvault[macos] to enable the macOS Keychain tier")

    vault.vault_command(SimpleNamespace(_vault_handler=boom))
    assert "[macos]" in recorder.export_text()
