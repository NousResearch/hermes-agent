"""A transient read failure on auth.json must not degrade to an empty store.

``_load_auth_store`` treated every exception as corruption and returned
``{"version": ..., "providers": {}}``. This module does read-modify-write in
roughly fifteen places, so an ``OSError`` (EMFILE under fd exhaustion, EACCES,
EIO, a stalled mount) followed by any ``_save_auth_store`` rewrote auth.json
with an empty provider set and destroyed every stored credential.

Genuine corruption still degrades, still preserves a copy (now under a
timestamped filename so a second corruption event can't clobber a prior
backup), and still only claims to have preserved one when the copy actually
landed. The resulting in-memory store is also stamped as a load-failure
fallback so ``_save_auth_store`` refuses to ever flush it to disk while it
is still empty.
"""

import errno
import glob
import json
import logging

import pytest

import hermes_cli.auth as auth


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
    assert not glob.glob(str(store_file) + ".corrupt*"), (
        "a read failure is not corruption and must not write a .corrupt sidecar"
    )


def test_unparseable_json_still_degrades_and_preserves_a_copy(store_file):
    store_file.write_text("{ not json", encoding="utf-8")

    result = auth._load_auth_store(store_file)

    assert result["version"] == auth.AUTH_STORE_VERSION
    assert result["providers"] == {}
    matches = glob.glob(str(store_file) + ".corrupt.*")
    assert len(matches) == 1, "genuine corruption must still be preserved"
    corrupt = matches[0]
    assert not corrupt.endswith(".corrupt"), (
        "backup filename must be content-addressed, not the old fixed auth.json.corrupt"
    )
    assert corrupt.endswith(".bak")
    with open(corrupt, encoding="utf-8") as fh:
        assert fh.read() == "{ not json"


def test_repeated_loads_of_unchanged_corruption_reuse_one_backup(store_file):
    """The corrupt file stays on disk, so this branch runs on every load.

    A unique-per-attempt backup name would mint a new copy each time and
    amplify disk usage without bound. Identical bytes must map to one backup.
    """
    store_file.write_text("{ not json", encoding="utf-8")

    for _ in range(25):
        auth._load_auth_store(store_file)

    backups = glob.glob(str(store_file) + ".corrupt.*")
    assert len(backups) == 1, (
        "repeated loads of UNCHANGED corrupt bytes must reuse a single "
        f"content-addressed backup, got {len(backups)}"
    )
    with open(backups[0], encoding="utf-8") as fh:
        assert fh.read() == "{ not json"


def test_changed_corrupt_bytes_get_their_own_backup(store_file):
    """Deduping must not lose genuinely different corrupt content."""
    store_file.write_text("{ not json one", encoding="utf-8")
    auth._load_auth_store(store_file)
    assert len(glob.glob(str(store_file) + ".corrupt.*")) == 1

    store_file.write_text("{ not json two", encoding="utf-8")
    auth._load_auth_store(store_file)

    backups = sorted(glob.glob(str(store_file) + ".corrupt.*"))
    assert len(backups) == 2, "changed corrupt bytes must be preserved separately"
    contents = set()
    for path in backups:
        with open(path, encoding="utf-8") as fh:
            contents.add(fh.read())
    assert contents == {"{ not json one", "{ not json two"}


def test_corrupt_backups_are_capped_by_retention(store_file, monkeypatch):
    """A mutating-corruption loop must not accumulate backups forever."""
    monkeypatch.setattr(auth, "_CORRUPT_AUTH_BACKUP_RETENTION", 5)

    for i in range(20):
        store_file.write_text(f"{{ not json {i}", encoding="utf-8")
        auth._load_auth_store(store_file)

    backups = glob.glob(str(store_file) + ".corrupt.*")
    assert len(backups) <= 5, (
        f"retention cap must bound corrupt backups, got {len(backups)}"
    )
    # The most recent corruption must be among what we kept.
    contents = set()
    for path in backups:
        with open(path, encoding="utf-8") as fh:
            contents.add(fh.read())
    assert "{ not json 19" in contents


def test_healthy_store_is_returned_unchanged(store_file):
    result = auth._load_auth_store(store_file)
    assert result["providers"]["nous"]["api_key"] == "secret"
    assert auth._LOAD_FAILURE_MARKER not in result


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

    assert result["providers"] == {}
    assert not glob.glob(str(store_file) + ".corrupt*")
    text = caplog.text
    assert "could NOT be preserved" in text
    assert "Corrupt file preserved at" not in text


def test_load_failure_store_cannot_be_flushed_to_disk(store_file):
    """A load-failure fallback store must never reach _save_auth_store's write.

    This is the guard that closes the actual data-loss bug: even if some
    calling code did the old-style "load, then unconditionally save" thing
    after a corruption-triggered fallback, the write itself must refuse.
    """
    store_file.write_text("{ not json", encoding="utf-8")
    fallback_store = auth._load_auth_store(store_file)
    assert fallback_store.get(auth._LOAD_FAILURE_MARKER) is True

    with pytest.raises(auth.AuthStoreWriteGuardError):
        auth._save_auth_store(fallback_store, target_path=store_file)

    # The original (corrupt) content must remain completely untouched, and no
    # new payload must have been written in its place.
    with open(store_file, encoding="utf-8") as fh:
        assert fh.read() == "{ not json"


def test_load_failure_store_can_be_saved_once_populated(tmp_path):
    """Once a caller adds a real provider, the guard no longer applies."""
    store_file = tmp_path / "auth.json"
    store_file.write_text("{ not json", encoding="utf-8")
    fallback_store = auth._load_auth_store(store_file)

    fallback_store["providers"]["nous"] = {"api_key": "new-secret"}
    saved_path = auth._save_auth_store(fallback_store, target_path=store_file)

    assert saved_path == store_file
    on_disk = json.loads(store_file.read_text(encoding="utf-8"))
    assert on_disk["providers"]["nous"]["api_key"] == "new-secret"
    assert auth._LOAD_FAILURE_MARKER not in on_disk


def test_save_of_ordinary_empty_store_is_still_allowed(tmp_path):
    """A genuinely empty store from an explicit user action (no marker) still saves."""
    store_file = tmp_path / "auth.json"
    empty_store = {"version": auth.AUTH_STORE_VERSION, "providers": {}}

    saved_path = auth._save_auth_store(empty_store, target_path=store_file)

    assert saved_path == store_file
    on_disk = json.loads(store_file.read_text(encoding="utf-8"))
    assert on_disk["providers"] == {}
