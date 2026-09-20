"""A transient read failure on auth.json must not degrade to an empty store.

``_load_auth_store`` treated every exception as corruption and returned
``{"version": ..., "providers": {}}``. This module does read-modify-write in
roughly fifteen places, so an ``OSError`` (EMFILE under fd exhaustion, EACCES,
EIO, a stalled mount) followed by any ``_save_auth_store`` rewrote auth.json
with an empty provider set and destroyed every stored credential.

Genuine corruption still degrades, still preserves a copy, and now only claims
to have preserved one when the copy actually landed.
"""

import errno
import json
import logging
from pathlib import Path

import pytest

import hermes_cli.auth as auth


def test_shared_profile_auth_symlink_resolves_to_one_atomic_store(tmp_path, monkeypatch):
    """A deliberate shared auth alias must lock and replace the root store, not unlink itself.

    Named profiles normally own independent credentials. Operators may explicitly make a
    profile's ``auth.json`` a symlink to the root store when every role uses one subscription.
    The active auth path must resolve that alias before lock/write selection so OAuth refresh
    rotation remains one atomic transaction across the fleet.
    """
    root = tmp_path / "root"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / ".share-profile-auth").write_text("enabled\n", encoding="utf-8")
    root_auth = root / "auth.json"
    root_auth.write_text(
        json.dumps({"version": 1, "providers": {"openai-codex": {"tokens": {"access_token": "a"}}}}),
        encoding="utf-8",
    )
    profile_auth = profile / "auth.json"
    profile_auth.symlink_to(root_auth)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(auth, "get_hermes_home", lambda: profile)

    assert auth._auth_file_path() == root_auth.resolve()
    with auth._auth_store_lock():
        store = auth._load_auth_store()
        store["providers"]["openai-codex"]["last_refresh"] = "rotated"
        auth._save_auth_store(store)

    assert profile_auth.is_symlink(), "atomic refresh must not replace the shared-store alias"
    assert json.loads(root_auth.read_text(encoding="utf-8"))["providers"]["openai-codex"]["last_refresh"] == "rotated"


@pytest.mark.parametrize("via_default_store", [False, True], ids=["direct", "chained"])
def test_profile_auth_symlink_to_unrelated_json_is_rejected(
    tmp_path, monkeypatch, via_default_store
):
    root = tmp_path / "root"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / ".share-profile-auth").write_text("enabled\n", encoding="utf-8")
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"keep":"me"}\n', encoding="utf-8")
    target = unrelated
    if via_default_store:
        target = root / "auth.json"
        target.symlink_to(unrelated)
    (profile / "auth.json").symlink_to(target)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(auth, "get_hermes_home", lambda: profile)

    with pytest.raises(RuntimeError, match="shared auth symlink"):
        auth._save_auth_store({"version": auth.AUTH_STORE_VERSION, "providers": {}})

    assert unrelated.read_text(encoding="utf-8") == '{"keep":"me"}\n'


@pytest.fixture
def store_file(tmp_path):
    f = tmp_path / "auth.json"
    f.write_text(
        json.dumps({"version": 1, "providers": {"nous": {"api_key": "secret"}}}),
        encoding="utf-8",
    )
    return f


def _fail_read(exc):
    def _read(self, *args, **kwargs):
        raise exc
    return _read


@pytest.mark.parametrize(
    "exc",
    [
        OSError(errno.EMFILE, "Too many open files"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EIO, "Input/output error"),
    ],
    ids=["emfile", "eacces", "eio"],
)
def test_read_failure_raises_and_leaves_the_store_alone(store_file, monkeypatch, exc):
    from pathlib import Path

    before = store_file.read_bytes()
    monkeypatch.setattr(Path, "read_text", _fail_read(exc))

    with pytest.raises(OSError):
        auth._load_auth_store(store_file)

    assert store_file.read_bytes() == before, "the store on disk was modified"
    assert not store_file.with_suffix(".json.corrupt").exists(), (
        "a read failure is not corruption and must not write a .corrupt sidecar"
    )


def test_unparseable_json_still_degrades_and_preserves_a_copy(store_file):
    store_file.write_text("{ not json", encoding="utf-8")

    result = auth._load_auth_store(store_file)

    assert result == {"version": auth.AUTH_STORE_VERSION, "providers": {}}
    corrupt = store_file.with_suffix(".json.corrupt")
    assert corrupt.exists(), "genuine corruption must still be preserved"
    assert corrupt.read_text(encoding="utf-8") == "{ not json"


def test_healthy_store_is_returned_unchanged(store_file):
    result = auth._load_auth_store(store_file)
    assert result["providers"]["nous"]["api_key"] == "secret"


def test_log_does_not_claim_a_backup_that_was_not_written(
    store_file, monkeypatch, caplog
):
    """The old message advertised the .corrupt path even when copy2 failed."""
    import shutil

    store_file.write_text("{ not json", encoding="utf-8")

    def _no_copy(*args, **kwargs):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(shutil, "copy2", _no_copy)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        result = auth._load_auth_store(store_file)

    assert result == {"version": auth.AUTH_STORE_VERSION, "providers": {}}
    assert not store_file.with_suffix(".json.corrupt").exists()
    text = caplog.text
    assert "could NOT be preserved" in text
    assert "Corrupt file preserved at" not in text
