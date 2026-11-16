"""The one-time import checks and stages the same captured bytes.

The boot reads the profile's ``.env`` and ``auth.json`` once; the kept-line check and the staging
both use those bytes. A file that changes before the staging completes is checked again on its new
bytes (and staged from them), so an edit landing between the check and the staging can never be
imported unchecked (rv27 P3). A file that keeps changing fails closed. Every refusal leaves the
files, and any earlier unfinished staging, byte for byte as they were.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.test_remote_import_refusal import (
    REF, REFUSED_ENV, _named_without_values,
)
from tests.plugins.secret_store.test_remote_load_at_start import _boot, _fresh_start, _readers

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

SAFE = "OPENAI_API_KEY=imported-key\n"


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _after_check(monkeypatch, edit, *, times=1):
    """Run *edit* right after the check returns (before the staging), *times* times."""
    from plugins.secret_store.remote import importer
    real = importer.refusal_names
    calls = []

    def check_then_edit(*args, **kwargs):
        names = real(*args, **kwargs)
        calls.append(names)
        if len(calls) <= times:
            edit()
        return names
    monkeypatch.setattr(importer, "refusal_names", check_then_edit)
    return calls


def _append(path, text):
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)


def test_an_unsafe_edit_between_the_check_and_the_staging_is_checked_and_refused(plane, monkeypatch):
    """The reviewer's barrier: the check passes on a safe file, then another writer appends a
    kept line the import would change. The boot checks the new bytes and stops: nothing sent,
    no staging, the files as the writer left them."""
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(SAFE, encoding="utf-8")
    calls = _after_check(monkeypatch, lambda: _append(env, REFUSED_ENV))
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _named_without_values(err, REF, "default")
    assert calls == [[], [REF]]
    assert plane.requests == [] and not (plane.home / importer.STAGING).exists()
    assert env.read_text(encoding="utf-8") == SAFE + REFUSED_ENV


def test_a_safe_edit_between_the_check_and_the_staging_imports_the_new_bytes(plane, monkeypatch):
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(SAFE, encoding="utf-8")
    calls = _after_check(monkeypatch, lambda: _append(env, "RV_LATE_KEY=late-value\n"))
    _boot(plane.home)
    assert calls == [[], []]
    assert len(plane.patches()) == 1
    assert plane.patches()[0]["body"]["set"] == {"OPENAI_API_KEY": "imported-key", "RV_LATE_KEY": "late-value"}
    assert set(_readers(plane.home, "RV_LATE_KEY").values()) == {"late-value"}


def test_an_auth_json_edit_between_the_check_and_the_staging_is_checked_again(plane, monkeypatch):
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text(SAFE, encoding="utf-8")
    auth = plane.home / "auth.json"
    record = {"version": 1, "providers": {"openai-codex": {"tokens": {"access_token": "codex-late"}}}}
    calls = _after_check(monkeypatch, lambda: auth.write_text(json.dumps(record), encoding="utf-8"))
    _boot(plane.home)
    assert len(calls) == 2 and len(plane.patches()) == 1
    assert "codex-late" in json.dumps(plane.patches()[0]["body"]["auth"])


def test_a_file_that_keeps_changing_fails_closed_untouched(plane, monkeypatch):
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(SAFE, encoding="utf-8")
    edits = []

    def edit():
        edits.append(1)
        _append(env, f"RV_EDIT_{len(edits)}=x\n")
    _after_check(monkeypatch, edit, times=100)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    shown = str(err.value.code)
    assert "'default'" in shown and "kept changing" in shown and "imported-key" not in shown
    assert len(edits) == importer.STAGE_ROUNDS
    assert plane.requests == [] and not (plane.home / importer.STAGING).exists()


def test_a_refusal_after_a_restaging_round_restores_the_earlier_staging(plane, monkeypatch):
    """An unfinished staging from an earlier boot survives a refusal on a later round byte for byte."""
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(SAFE, encoding="utf-8")
    staged = plane.home / importer.STAGING
    staged.write_text(json.dumps({"env": {"RV_EARLIER": "earlier"}, "auth": {}}), encoding="utf-8")
    staged.chmod(0o600)
    _after_check(monkeypatch, lambda: _append(env, REFUSED_ENV))
    earlier = staged.read_bytes()
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _named_without_values(err, REF, "default")
    assert staged.read_bytes() == earlier
    assert env.read_text(encoding="utf-8") == SAFE + REFUSED_ENV
    assert plane.requests == []


def test_the_staging_holds_exactly_the_bytes_the_check_verified(plane, monkeypatch):
    """An edit made after the check and undone right after the staging is written (the recapture then
    sees the checked bytes again) never reaches the staging or the plane: the staging is built
    from the captured bytes, not from a second read of the file."""
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(SAFE, encoding="utf-8")
    _after_check(monkeypatch, lambda: _append(env, "RV_SNEAK=unchecked\n"))
    real_write = importer._write_private_json

    def write_then_undo(path, data):
        real_write(path, data)
        if Path(path).name == importer.STAGING:
            env.write_text(SAFE, encoding="utf-8")
    monkeypatch.setattr(importer, "_write_private_json", write_then_undo)
    _boot(plane.home)
    assert len(plane.patches()) == 1
    assert plane.patches()[0]["body"]["set"] == {"OPENAI_API_KEY": "imported-key"}
