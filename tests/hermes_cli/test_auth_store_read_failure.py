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


def test_unparseable_json_still_degrades_and_preserves_a_timestamped_copy(store_file):
    store_file.write_text("{ not json", encoding="utf-8")

    result = auth._load_auth_store(store_file)

    assert result["version"] == auth.AUTH_STORE_VERSION
    assert result["providers"] == {}
    matches = glob.glob(str(store_file) + ".corrupt.*")
    assert len(matches) == 1, "genuine corruption must still be preserved, timestamped"
    corrupt = matches[0]
    assert not corrupt.endswith(".corrupt"), (
        "backup filename must be timestamped, not the old fixed auth.json.corrupt"
    )
    with open(corrupt, encoding="utf-8") as fh:
        assert fh.read() == "{ not json"


def test_second_corruption_event_does_not_clobber_first_backup(store_file):
    store_file.write_text("{ not json one", encoding="utf-8")
    auth._load_auth_store(store_file)
    first_backups = sorted(glob.glob(str(store_file) + ".corrupt.*"))
    assert len(first_backups) == 1

    store_file.write_text("{ not json two", encoding="utf-8")
    auth._load_auth_store(store_file)
    second_backups = sorted(glob.glob(str(store_file) + ".corrupt.*"))

    assert len(second_backups) == 2, "a second corruption must not overwrite the first backup"
    with open(first_backups[0], encoding="utf-8") as fh:
        assert fh.read() == "{ not json one", "the first backup's content must survive"


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
