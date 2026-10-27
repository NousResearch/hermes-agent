"""Where the local vault key lives: the 0600 ``vault.key`` file (default) or the OS keychain.

Contracts (temp HERMES_HOME + an in-memory keyring backend; nothing touches a real keychain):
- keychain mode stores a NEW vault's key in the keychain, one entry per profile, and leaves no key
  file beside the ciphertext;
- ``migrate_key`` writes the keychain first, proves it decrypts the vault, and only then removes the
  file; a store that cannot return the key aborts with the file intact (and back again for ``file``);
- a vault whose key is missing says how to recover instead of minting a new key over it;
- no reachable OS keychain (headless Linux, no keyring extra) falls back to the key file.
"""

from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path

import keyring
from cryptography.fernet import Fernet
import pytest
from keyring.backend import KeyringBackend
from keyring.backends import fail
from keyring.errors import PasswordDeleteError

from agent.vault_store import VaultError, VaultStore

_LOGIN = {"identifier_type": "email", "identifier": "jane@example.com", "password": "dummy-pw-not-real-42"}


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.entries.get((service, username))

    def set_password(self, service, username, password):
        self.entries[(service, username)] = password

    def delete_password(self, service, username):
        if self.entries.pop((service, username), None) is None:
            raise PasswordDeleteError("no such entry")


class ForgetfulKeyring(MemoryKeyring):
    """Accepts the write but hands back something else (a broken or mis-scoped store)."""

    def get_password(self, service, username):
        return "bm90LXRoZS1rZXktbm90LXRoZS1rZXktbm90LXRoZS1rZXk="


@pytest.fixture
def home(tmp_path, monkeypatch):
    def make(name: str = "default", key_storage: str | None = None):
        path = tmp_path / name
        path.mkdir()
        if key_storage:
            (path / "config.yaml").write_text(f"vault:\n  key_storage: {key_storage}\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(path))
        return path
    yield make
    keyring.set_keyring(fail.Keyring())


@pytest.fixture
def keychain():
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    return backend


def _add(store: VaultStore) -> str:
    return store.add_item("login", "Example", dict(_LOGIN), origin="https://example.com").id


def test_keychain_mode_puts_a_new_vault_key_in_the_keychain_per_profile(home, keychain):
    a = home("a", key_storage="keychain")
    item_a = _add(VaultStore())
    assert not (a / "vault" / "vault.key").exists()
    assert (a / "vault" / "vault.json.enc").stat().st_size > 0
    assert len(keychain.entries) == 1
    assert VaultStore().resolve_secret(item_a)["password"] == _LOGIN["password"]

    b = home("b", key_storage="keychain")
    item_b = _add(VaultStore())
    assert not (b / "vault" / "vault.key").exists()
    assert len(keychain.entries) == 2, "each profile gets its own keychain entry"
    assert VaultStore().resolve_secret(item_b)["password"] == _LOGIN["password"]
    assert VaultStore(a / "vault").resolve_secret(item_a)["password"] == _LOGIN["password"]


def test_migration_writes_keychain_verifies_then_removes_the_file_and_back(home, keychain):
    h = home()
    store = VaultStore()
    item = _add(store)
    key_file = h / "vault" / "vault.key"
    old_key = key_file.read_bytes().strip()

    assert store.migrate_key("keychain") == "keychain"
    assert not key_file.exists()
    assert list(keychain.entries.values()) == [old_key.decode()]
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]

    assert store.migrate_key("file") == "file"
    assert key_file.read_bytes().strip() == old_key
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert keychain.entries == {}
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]


def test_migration_keeps_the_key_file_when_the_keychain_cannot_return_the_key(home):
    h = home()
    keyring.set_keyring(ForgetfulKeyring())
    store = VaultStore()
    item = _add(store)
    with pytest.raises(VaultError, match="keychain"):
        store.migrate_key("keychain")
    assert (h / "vault" / "vault.key").exists()
    assert not (h / "vault" / "vault.keyref").exists()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]


@pytest.mark.parametrize("lose", ["keychain entry", "key file"])
def test_missing_key_with_an_existing_vault_explains_recovery_and_mints_nothing(home, keychain, lose):
    h = home(key_storage="keychain" if lose == "keychain entry" else None)
    _add(VaultStore())
    blob = (h / "vault" / "vault.json.enc").read_bytes()
    if lose == "keychain entry":
        keychain.entries.clear()
    else:
        (h / "vault" / "vault.key").unlink()

    with pytest.raises(VaultError) as excinfo:
        VaultStore().list_items()
    message = str(excinfo.value)
    assert "hermes vault key" in message and "vault.json.enc" in message
    assert (h / "vault" / "vault.json.enc").read_bytes() == blob, "the ciphertext must be left alone"
    assert not (h / "vault" / "vault.key").exists(), "no fresh key may be minted over an existing vault"
    assert len(keychain.entries) == 0
    if lose == "keychain entry":  # and a rerun of the move must not report the lost entry as fine
        with pytest.raises(VaultError, match="hermes vault key"):
            VaultStore().migrate_key("keychain")


@pytest.mark.parametrize("headless", ["no secret service", "no keyring extra"])
def test_keychain_mode_without_a_reachable_keychain_falls_back_to_the_key_file(home, monkeypatch, headless):
    h = home(key_storage="keychain")
    if headless == "no secret service":
        keyring.set_keyring(fail.Keyring())  # what keyring resolves to on a headless Linux box
    else:
        monkeypatch.setitem(sys.modules, "keyring", None)  # extra not installed: import fails
    item = _add(VaultStore())
    key_file = h / "vault" / "vault.key"
    assert key_file.exists() and stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert not (h / "vault" / "vault.keyref").exists()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]

    with pytest.raises(VaultError, match="keychain"):
        VaultStore().migrate_key("keychain")
    assert key_file.exists()


def test_a_stray_key_file_that_cannot_open_the_vault_never_costs_the_keychain_copy(home, keychain):
    h = home(key_storage="keychain")
    item = _add(VaultStore())
    (h / "vault" / "vault.key").write_bytes(Fernet.generate_key())  # e.g. restored from an old backup
    good = dict(keychain.entries)
    for target in ("file", "keychain"):
        with pytest.raises(VaultError, match="does not decrypt"):
            VaultStore().migrate_key(target)
    assert keychain.entries == good
    (h / "vault" / "vault.key").unlink()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]


def test_cloned_and_renamed_keychain_vaults_never_share_or_overwrite_an_entry(home, keychain):
    """A profile clone copies vault.keyref; a rename moves it. Neither copy may later overwrite or delete
    the entry another copy still depends on, and a new vault at a reused name gets its own entry."""
    work = home("work", key_storage="keychain")
    item = _add(VaultStore())
    a, b = home("clone-a", key_storage="keychain"), home("clone-b", key_storage="keychain")
    for clone in (a, b):
        shutil.copytree(work / "vault", clone / "vault")
    assert VaultStore(a / "vault").migrate_key("file") == "file"
    assert VaultStore(b / "vault").migrate_key("keychain") == "keychain"
    assert VaultStore(work / "vault").migrate_key("file") == "file"  # deletes only the original's own entry
    for vault in (a, b, work):
        assert VaultStore(vault / "vault").resolve_secret(item)["password"] == _LOGIN["password"]

    assert VaultStore(work / "vault").migrate_key("keychain") == "keychain"
    renamed = home("renamed", key_storage="keychain")
    (work / "vault").rename(renamed / "vault")
    _add(VaultStore(work / "vault"))  # a new vault where the renamed one used to live
    assert VaultStore(renamed / "vault").resolve_secret(item)["password"] == _LOGIN["password"]


def test_clone_all_owns_its_key_before_the_source_ever_migrates(home, keychain, tmp_path, monkeypatch):
    """A --clone-all copy gets its own keychain entry when it is created, so the source moving its key
    to a file (which deletes the source's entry) cannot strand a clone that was never opened."""
    from hermes_cli.profiles import create_profile

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = home(".hermes", key_storage="keychain")
    item = _add(VaultStore())
    clone = create_profile("vault-clone", clone_from="default", clone_all=True, no_alias=True)

    assert VaultStore().migrate_key("file") == "file"  # the clone has not opened its vault yet
    assert VaultStore(source / "vault").resolve_secret(item)["password"] == _LOGIN["password"]
    assert VaultStore(clone / "vault").resolve_secret(item)["password"] == _LOGIN["password"]
    assert not (clone / "vault" / "vault.key").exists(), "the clone keeps keychain storage"
    assert len(keychain.entries) == 1, "only the clone's own entry is left"


def test_clone_all_without_a_reachable_keychain_is_refused_not_published_sharing_the_key(
        home, keychain, tmp_path, monkeypatch):
    from hermes_cli.profiles import create_profile, get_profile_dir

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home(".hermes", key_storage="keychain")
    _add(VaultStore())
    keyring.set_keyring(fail.Keyring())  # e.g. an SSH session that cannot reach the login keychain
    with pytest.raises(ValueError, match="OS keychain"):
        create_profile("vault-clone", clone_from="default", clone_all=True, no_alias=True)
    assert not get_profile_dir("vault-clone").exists()


def test_a_clone_that_fails_to_publish_leaves_no_keychain_entry_behind(home, keychain, tmp_path, monkeypatch):
    from hermes_cli import profiles_initialization
    from hermes_cli.profiles import create_profile, get_profile_dir

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = home(".hermes", key_storage="keychain")
    item = _add(VaultStore())

    def fail_layout(*args, **kwargs):
        raise RuntimeError("dummy failure after the clone got its own entry")

    monkeypatch.setattr(profiles_initialization, "finish_profile_layout", fail_layout)
    with pytest.raises(RuntimeError):
        create_profile("vault-clone", clone_from="default", clone_all=True, no_alias=True)
    assert not get_profile_dir("vault-clone").exists()
    assert len(keychain.entries) == 1, "the unpublished clone's entry is removed; the source's is kept"
    assert VaultStore(source / "vault").resolve_secret(item)["password"] == _LOGIN["password"]


def test_a_clone_failing_before_custody_never_deletes_the_source_entry(home, keychain, tmp_path, monkeypatch):
    """Rollback deletes only the entry made for this clone, never one inferred from the source's files
    (here the source vault moves away mid-clone, and the copy still names the source's entry)."""
    from hermes_cli import auth
    from hermes_cli.profiles import create_profile

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = home(".hermes", key_storage="keychain")
    item = _add(VaultStore())
    moved = tmp_path / "moved-vault"

    def fail_mid_copy(profile_dir):
        (source / "vault").rename(moved)
        raise RuntimeError("dummy failure while the copy still names the source's entry")

    monkeypatch.setattr(auth, "strip_cloned_single_use_oauth_grants", fail_mid_copy)
    with pytest.raises(RuntimeError):
        create_profile("vault-clone", clone_from="default", clone_all=True, no_alias=True)
    assert len(keychain.entries) == 1
    assert VaultStore(moved).resolve_secret(item)["password"] == _LOGIN["password"]


def test_migration_whose_pointer_cannot_be_saved_leaves_no_untracked_entry(home, keychain, monkeypatch):
    from agent import vault_keychain

    h = home()
    item = _add(VaultStore())

    def failing_write_ref(path, ref):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vault_keychain, "write_ref", failing_write_ref)
    with pytest.raises(VaultError, match="vault.keyref"):
        VaultStore().migrate_key("keychain")
    assert keychain.entries == {}, "a retry would mint another random account, so nothing may be left"
    assert (h / "vault" / "vault.key").exists()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]


@pytest.mark.parametrize("pointer_landed", [False, True])
def test_copied_vault_keeps_opening_when_saving_its_own_pointer_fails(home, keychain, monkeypatch, pointer_landed):
    from agent import vault_keychain

    work = home("work", key_storage="keychain")
    item = _add(VaultStore())
    clone = home("clone", key_storage="keychain")
    shutil.copytree(work / "vault", clone / "vault")
    real_write_ref = vault_keychain.write_ref

    def failing_write_ref(path, ref):
        if pointer_landed:  # the replace happened, a later step (temp cleanup) failed
            real_write_ref(path, ref)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vault_keychain, "write_ref", failing_write_ref)
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]
    # no half-made entry left behind; an entry the saved pointer names is never deleted
    assert len(keychain.entries) == (2 if pointer_landed else 1)
    monkeypatch.setattr(vault_keychain, "write_ref", real_write_ref)
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"], "and on every later open"


def test_cleanup_of_an_already_deleted_keychain_entry_completes(home, keychain):
    h = home(key_storage="keychain")
    item = _add(VaultStore())
    keyref = (h / "vault" / "vault.keyref").read_bytes()
    assert VaultStore().migrate_key("file") == "file"
    (h / "vault" / "vault.keyref").write_bytes(keyref)  # stopped after deleting the entry, before the pointer
    assert VaultStore().migrate_key("file") == "file"
    assert "keychain_entry_left" not in VaultStore().key_location()
    assert not (h / "vault" / "vault.keyref").exists()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]
