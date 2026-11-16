"""The one-time import runs under one exclusive per-profile import lock, from one read of the files.

- Concurrent first boots of one profile (a second provider instance on another thread stands in for
  a second process: the lock is a file lock) never lose an OAuth record: a boot that arrives while
  another is importing waits for it and then starts from a fresh read; when the first crashed after
  its token refresh (``auth.json`` already bootstrap-only), the second imports from its staging.
- ``auth.json`` is read once: the record set sent is the one read under the lock, whatever another
  writer does to the file meanwhile.
- The loader's own sanitiser normalises ``.env`` before the import reads it, so its mode and line
  endings are exactly the local store's (Windows text mode simulated: CRLF), and the cleanup keeps
  the mode it finds.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.test_remote_import_refusal import (
    REFUSED_ENV, _files, _fresh, _nous_auth, _nous_credential, _write,
)
from tests.plugins.secret_store.test_remote_load_at_start import _boot

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

SAFE = "OPENAI_API_KEY=imported-key\n"


def _stub_token(monkeypatch):
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


# --- concurrent first boots -------------------------------------------------------------------------

def _two_boots(plane, monkeypatch, *, a_crashes):
    """Process A boots a never-imported profile whose Nous token is in the refresh window: its real
    refresh saves auth.json with the bootstrap records only. Right after that refresh, process B
    boots the same profile; A goes on only once B is waiting for the import lock (or, without one,
    has read the files), then crashes before its fetch or completes. Returns the event log."""
    from hermes_cli import auth
    from plugins.secret_store.remote import backend, importer
    _fresh(monkeypatch, {})
    _nous_credential(monkeypatch, [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (plane.home / ".env").write_text(SAFE, encoding="utf-8")
    _nous_auth(plane.home, expires_in=90)
    prov_a, prov_b = routing.routed_provider(), backend.RemoteSecretStoreProvider()
    log, errors, b_progress = [], [], threading.Event()
    real_lock, real_capture, real_finish = auth._file_lock, importer.capture, importer.finish
    real_bearer = backend.RemoteSecretStoreProvider._bearer

    def who():
        return "B" if threading.current_thread().name == "procB" else "A"
    # Each "process" has its own selected store.
    monkeypatch.setattr(routing, "routed_provider", lambda: prov_b if who() == "B" else prov_a)

    @contextmanager
    def file_lock(path, *args, **kwargs):
        if who() == "B" and Path(path).name == importer.LOCK:
            b_progress.set()
        with real_lock(path, *args, **kwargs):
            yield

    def capture(home):
        log.append(f"{who()}.capture")
        if who() == "B":
            b_progress.set()
        return real_capture(home)

    def finish(*args):
        log.append(f"{who()}.finish")
        return real_finish(*args)

    def run_b():
        try:
            prov_b.boot(plane.home)
        except BaseException as exc:  # reported through the log
            errors.append(repr(exc))
    b = threading.Thread(target=run_b, name="procB", daemon=True)

    def bearer(home):
        token = real_bearer(home)
        if who() == "A":
            log.append("A.refreshed")
            b.start()
            assert b_progress.wait(20)
            if a_crashes:
                log.append("A.crashed")
                raise RuntimeError("process A crashed after its refresh")
        return token
    monkeypatch.setattr(auth, "_file_lock", file_lock)
    monkeypatch.setattr(importer, "capture", capture)
    monkeypatch.setattr(importer, "finish", finish)
    monkeypatch.setattr(backend.RemoteSecretStoreProvider, "_bearer", staticmethod(bearer))
    if a_crashes:
        with pytest.raises(RuntimeError, match="process A crashed"):
            prov_a.boot(plane.home)
    else:
        prov_a.boot(plane.home)
    b.join(20)
    assert not b.is_alive() and errors == []
    return log, prov_b


def _no_record_lost(plane, prov):
    sent = json.dumps([p["body"] for p in plane.patches()])
    assert "codex-local" in sent and "pool-local" in sent and "imported-key" in sent
    view = prov.auth_view(plane.home)
    assert view["openai-codex"] == {"tokens": {"access_token": "codex-local"}}
    assert "codex-local" not in (plane.home / "auth.json").read_text(encoding="utf-8")
    assert (plane.home / ".env").read_text(encoding="utf-8") == ""
    assert not (plane.home / ".secrets-import.json").exists()
    assert not (plane.home / ".env.unimported").exists() and not (plane.home / "auth.unimported.json").exists()
    assert prov.doctor_warnings(plane.home) == ()


def test_a_boot_waits_for_a_concurrent_import_then_finds_it_done(plane, monkeypatch):
    log, prov_b = _two_boots(plane, monkeypatch, a_crashes=False)
    assert log == ["A.capture", "A.refreshed", "A.finish", "B.capture", "B.finish"]
    assert len(plane.patches()) == 1   # A's import; B read afresh and had nothing left to send
    _no_record_lost(plane, prov_b)


def test_a_crash_after_the_refresh_never_loses_an_oauth_record(plane, monkeypatch):
    """The reviewer's crash-before-fetch: A staged, refreshed (auth.json now bootstrap-only) and
    died before its fetch. B, which waited, imports every record from A's staging."""
    log, prov_b = _two_boots(plane, monkeypatch, a_crashes=True)
    assert log == ["A.capture", "A.refreshed", "A.crashed", "B.capture", "B.finish"]
    assert len(plane.patches()) == 1
    _no_record_lost(plane, prov_b)


# --- auth.json is read once ---------------------------------------------------------------------

def test_an_auth_record_added_and_removed_during_the_import_is_never_sent(plane, monkeypatch):
    """ABA: right after the read, another writer adds a record; it removes it again right after the
    staging is written. The PATCH carries exactly the records read under the lock, and the cleanup
    (the file holds those bytes again) leaves auth.json with its bootstrap records."""
    from plugins.secret_store.remote import importer
    _stub_token(monkeypatch)
    _fresh(monkeypatch, {})
    (plane.home / ".env").write_text(SAFE, encoding="utf-8")
    auth = plane.home / "auth.json"
    auth.write_text(json.dumps({"version": 1, "providers": {"example": {"key": "auth-checked"}}}), encoding="utf-8")
    original = auth.read_bytes()
    real_check, real_write = importer.non_literal_kept_names, importer._write_private_json

    def check_then_edit(*args):
        auth.write_text(json.dumps({"version": 1, "providers": {
            "example": {"key": "auth-checked"}, "sneak": {"key": "auth-unchecked"}}}), encoding="utf-8")
        return real_check(*args)

    def write_then_undo(path, data):
        real_write(path, data)
        if Path(path).name == importer.STAGING:
            auth.write_bytes(original)
    monkeypatch.setattr(importer, "non_literal_kept_names", check_then_edit)
    monkeypatch.setattr(importer, "_write_private_json", write_then_undo)
    _boot(plane.home)
    assert [list(p["body"]["auth"]) for p in plane.patches()] == [["example"]]
    assert "auth-unchecked" not in json.dumps(plane.patches())
    assert "auth-checked" not in auth.read_text(encoding="utf-8")


# --- the loader's sanitiser: permissions and line endings ------------------------------------------

@pytest.mark.parametrize("store", ["local", "remote", "remote-kept-only"])
def test_a_sanitised_env_is_left_0600_as_the_local_store_leaves_it(plane, monkeypatch, store):
    """A 0644 ``.env`` the loader sanitises (a NUL) is replaced 0600 by the loader's own rewrite."""
    _stub_token(monkeypatch)
    _fresh(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = plane.home / ".env"
    env.write_bytes(b"GATEWAY_RELAY_IDP_AUDIENCE=aud\x00x\n" if store == "remote-kept-only"
                    else b"GATEWAY_RELAY_IDP_AUDIENCE=aud\nOPENAI_API_KEY=imported\x00-key\n")
    os.chmod(env, 0o644)
    if store == "local":
        monkeypatch.setenv("HERMES_SECRET_STORE", "local")
        routing.reset_for_tests()
    _boot(plane.home)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_bytes().startswith(b"GATEWAY_RELAY_IDP_AUDIENCE=aud")


def test_the_cleanup_keeps_the_env_mode_and_writes_its_side_files_0600(plane, monkeypatch):
    _stub_token(monkeypatch)
    _fresh(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "plane-value"})
    env = plane.home / ".env"
    env.write_text("GATEWAY_RELAY_IDP_AUDIENCE=aud\n" + SAFE, encoding="utf-8")
    os.chmod(env, 0o640)
    _boot(plane.home)   # the plane already has the profile: the local value is set aside
    assert env.read_text(encoding="utf-8") == "GATEWAY_RELAY_IDP_AUDIENCE=aud\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o640
    assert stat.S_IMODE((plane.home / ".env.unimported").stat().st_mode) == 0o600


@contextmanager
def _windows_text_mode(monkeypatch):
    """Simulate Windows text mode for writes the loader opens with ``os.fdopen(fd, "w")``: CRLF."""
    real = os.fdopen

    def fdopen(fd, mode="r", *args, **kwargs):
        if "w" in mode and "b" not in mode:
            kwargs.setdefault("newline", "\r\n")
        return real(fd, mode, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(os, "fdopen", fdopen)
        yield


def test_crlf_after_the_loaders_rewrite_is_imported_as_the_local_store_reads_it(plane, monkeypatch):
    """The loader's sanitiser (here rewriting for a NUL) writes in text mode, CRLF on Windows; a
    multi-line value then reads ``first\\r\\nsecond`` locally. The import reads the file that rewrite
    left, so it stores exactly that value, and the kept line keeps its CRLF."""
    raw = b'GATEWAY_RELAY_IDP_AUDIENCE=aud\nRV_MULTI="first\nsecond"\nRV_PAD=pad\x00ded\n'
    _stub_token(monkeypatch)
    for name in ("RV_MULTI", "RV_PAD"):
        monkeypatch.delenv(name, raising=False)
    env = plane.home / ".env"
    _fresh(monkeypatch, {})
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    routing.reset_for_tests()
    env.write_bytes(raw)
    with _windows_text_mode(monkeypatch):
        _boot(plane.home)
    local = {name: os.environ.get(name) for name in ("RV_MULTI", "RV_PAD")}
    assert local == {"RV_MULTI": "first\r\nsecond", "RV_PAD": "padded"}   # the simulation took effect
    _fresh(monkeypatch, {})
    monkeypatch.setenv("HERMES_SECRET_STORE", "remote")
    routing.reset_for_tests()
    for name in ("RV_MULTI", "RV_PAD"):
        monkeypatch.delenv(name, raising=False)
    env.write_bytes(raw)
    with _windows_text_mode(monkeypatch):
        _boot(plane.home)
    assert plane.patches()[0]["body"]["set"] == local
    assert {name: os.environ.get(name) for name in local} == local
    assert env.read_bytes() == b"GATEWAY_RELAY_IDP_AUDIENCE=aud\r\n"


def test_a_refusal_with_a_sanitised_env_and_an_earlier_staging_writes_nothing_else(plane, monkeypatch):
    """Refusal on a file the loader sanitises, with an earlier unfinished staging: the files are as
    the loader's own sanitiser leaves them (every load runs it), the staging byte for byte."""
    from plugins.secret_store.remote import importer
    _stub_token(monkeypatch)
    _fresh(monkeypatch, {})
    _write(plane.home, SAFE.replace("imported", "imp\x00orted") + REFUSED_ENV)
    (plane.home / importer.STAGING).write_text(json.dumps({"env": {"RV_EARLIER": "e"}, "auth": {}}), encoding="utf-8")
    os.chmod(plane.home / importer.STAGING, 0o600)
    _nous_auth(plane.home, expires_in=10**6)
    before = _files(plane.home)
    with pytest.raises(routing.SecretStoreUnavailable):
        _boot(plane.home)
    assert _files(plane.home) == before and plane.requests == []
