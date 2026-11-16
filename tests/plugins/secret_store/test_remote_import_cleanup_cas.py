"""The one-time import's cleanup rewrites only the bytes it imported.

After the plane takes the import, ``.env`` / ``auth.json`` are rewritten to their kept-local
entries only while each still holds the bytes the check read and the staging was built from. A
line or record another writer adds between the PATCH and the cleanup was never sent; the cleanup
leaves that file whole, and the next start sets it aside in ``*.unimported`` (the plane already
holds the profile, so it wins) with a ``hermes doctor`` warning. Nothing is ever lost.
"""
from __future__ import annotations

import json

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.test_remote_load_at_start import _boot, _fresh_start

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

KEPT = "GATEWAY_RELAY_IDP_AUDIENCE=kept-audience\n"
IMPORTED = "RV_IMPORTED=imported-value\n"
LATE = "RV_LATE=late-value\n"
AUTH = {"providers": {"example": {"key": "auth-imported"}}}


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _before_cleanup(monkeypatch, edit):
    """Run *edit* after the plane answered the import, right before the cleanup."""
    from plugins.secret_store.remote import importer
    real = importer.finish

    def edit_then_finish(*args):
        edit()
        return real(*args)
    edit_then_finish.real = real
    monkeypatch.setattr(importer, "finish", edit_then_finish)


def _append(path, text):
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)


def _restart(plane, monkeypatch):
    """A later start: a new process, the cleanup no longer interrupted."""
    from plugins.secret_store.remote import importer
    monkeypatch.setattr(importer, "finish", importer.finish.real)
    _fresh_start(monkeypatch, {})
    _boot(plane.home)


def test_an_env_line_added_after_the_patch_is_kept_then_set_aside_at_the_next_start(plane, monkeypatch):
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(KEPT + IMPORTED, encoding="utf-8")
    _before_cleanup(monkeypatch, lambda: _append(env, LATE))
    _boot(plane.home)
    assert [p["body"]["set"] for p in plane.patches()] == [{"RV_IMPORTED": "imported-value"}]
    assert env.read_text(encoding="utf-8") == KEPT + IMPORTED + LATE   # left whole: nothing lost
    from plugins.secret_store.remote import importer
    assert not (plane.home / importer.STAGING).exists()

    _restart(plane, monkeypatch)
    assert len(plane.patches()) == 1   # the plane holds the profile: it wins, nothing re-sent
    assert env.read_text(encoding="utf-8") == KEPT
    unimported = (plane.home / importer.UNIMPORTED_ENV).read_text(encoding="utf-8")
    assert "RV_LATE=late-value" in unimported and "RV_IMPORTED" not in unimported
    warnings = routing.routed_provider().doctor_warnings(plane.home)
    assert warnings and importer.UNIMPORTED_ENV in warnings[0][0] and "late-value" not in json.dumps(warnings)


def test_an_auth_record_added_after_the_patch_is_kept_then_set_aside_at_the_next_start(plane, monkeypatch):
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text(KEPT, encoding="utf-8")
    auth = plane.home / "auth.json"
    auth.write_text(json.dumps(AUTH), encoding="utf-8")
    late = {"providers": {**AUTH["providers"], "late": {"key": "auth-late"}}}
    _before_cleanup(monkeypatch, lambda: auth.write_text(json.dumps(late), encoding="utf-8"))
    _boot(plane.home)
    sent = json.dumps(plane.patches()[0]["body"]["auth"])
    assert "auth-imported" in sent and "auth-late" not in sent
    assert json.loads(auth.read_text(encoding="utf-8")) == late   # left whole: nothing lost

    _restart(plane, monkeypatch)
    assert len(plane.patches()) == 1
    assert "auth-late" not in auth.read_text(encoding="utf-8")
    aside = json.loads((plane.home / importer.UNIMPORTED_AUTH).read_text(encoding="utf-8"))
    assert aside["providers"] == {"late": {"key": "auth-late"}}
    assert routing.routed_provider().doctor_warnings(plane.home)


def test_unchanged_files_are_still_rewritten_to_the_kept_entries(plane, monkeypatch):
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_text(KEPT + IMPORTED, encoding="utf-8")
    auth = plane.home / "auth.json"
    auth.write_text(json.dumps(AUTH), encoding="utf-8")
    _boot(plane.home)
    assert len(plane.patches()) == 1
    assert env.read_text(encoding="utf-8") == KEPT
    assert "auth-imported" not in auth.read_text(encoding="utf-8")
    assert not (plane.home / importer.UNIMPORTED_ENV).exists()
    assert not routing.routed_provider().doctor_warnings(plane.home)


def test_a_sanitized_env_is_still_rewritten(plane, monkeypatch):
    """The loader's sanitizing is written before the PATCH, and the cleanup compares against the
    sanitized bytes, so an unchanged file the loader sanitizes is still cleaned."""
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    env.write_bytes((KEPT + "RV_IMPORTED=imported\x00-value\n").encode("utf-8"))
    _boot(plane.home)
    assert plane.patches()[0]["body"]["set"] == {"RV_IMPORTED": "imported-value"}
    assert env.read_text(encoding="utf-8") == KEPT


def test_an_env_created_after_the_check_is_left_alone(plane, monkeypatch):
    _fresh_start(monkeypatch, {})
    env = plane.home / ".env"
    (plane.home / "auth.json").write_text(json.dumps(AUTH), encoding="utf-8")
    _before_cleanup(monkeypatch, lambda: env.write_text(LATE, encoding="utf-8"))
    _boot(plane.home)
    assert env.read_text(encoding="utf-8") == LATE
