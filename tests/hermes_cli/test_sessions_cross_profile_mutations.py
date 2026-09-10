"""Cross-profile session mutations resolve the owning profile (regression).

The Desktop app scopes every REST call with a request-level profile, which
the Electron main process turns into a ``?profile=<name>`` query string in
global-remote mode (``pathWithGlobalRemoteProfile`` in
apps/desktop/electron/connection-config.ts) — the same query channel every
sibling session READ endpoint already declares. The session MUTATION
endpoints read the profile only from the request body, so a ``?profile=``
query was silently dropped: the handler opened the DEFAULT store, failed to
resolve a foreign session id, and 404'd (PATCH) or reported a no-op delete
(DELETE) — and because the store's own owner is knowable server-side, both
are pure routing bugs.

Separately, the Desktop renderer's session sync attaches ``profile`` only
when it can attribute one to the row (see the renderer-side counterpart,
PR #99376); for a row outside its active-profile slice the write goes out
with NO profile at all. The server can resolve that owner itself: a session
id absent from the default store but present in exactly one other served
local store belongs to that profile.

These tests pin both behaviors: mutation endpoints honor ``?profile=``
(body wins when both are sent), and a profile-less PATCH/DELETE of a session
owned by another local profile is routed to its single owner instead of
404-ing / no-op-ing into the default store.
"""

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _seed_profile_session(profile_name, session_id):
    """Create ``session_id`` inside ``profile_name``'s own state.db."""
    from hermes_cli import profiles as profiles_mod
    from hermes_state import SessionDB

    home = profiles_mod.get_profile_dir(profile_name)
    home.mkdir(parents=True, exist_ok=True)

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, source="desktop", profile_name=profile_name)
        db.append_message(session_id, role="user", content="hello")
    finally:
        db.close()
    return home


def _pinned_state(home, session_id):
    """The pinned flag for ``session_id`` in ``home``'s store, or None when
    the row (or the whole store) is absent — never raises."""
    import sqlite3

    conn = sqlite3.connect(str(home / "state.db"))
    try:
        try:
            row = conn.execute(
                "SELECT pinned FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            # Store never opened a sessions table — nothing can be pinned there.
            return None
        return None if row is None else bool(row[0])
    finally:
        conn.close()


def test_patch_session_pinned_honors_query_profile(client):
    """?profile=worker must pin/unpin in the WORKER store, not the default."""
    from hermes_constants import get_hermes_home

    worker_home = _seed_profile_session("worker", "worker-pin-test")

    # Pin via the ?profile= query channel the Desktop API client uses in
    # global-remote mode; the body carries only the flag.
    resp = client.patch("/api/sessions/worker-pin-test?profile=worker", json={"pinned": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["pinned"] is True

    # Landed in the worker store...
    assert _pinned_state(worker_home, "worker-pin-test") is True
    # ...and nowhere near the default store.
    assert _pinned_state(get_hermes_home(), "worker-pin-test") is None

    # Unpin back through the same query-scoped path.
    resp = client.patch("/api/sessions/worker-pin-test?profile=worker", json={"pinned": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["pinned"] is False
    assert _pinned_state(worker_home, "worker-pin-test") is False


def test_patch_session_body_profile_wins_over_query(client):
    """Explicit body.profile beats a conflicting ?profile= query."""
    worker_home = _seed_profile_session("worker", "worker-body-profile")

    resp = client.patch(
        "/api/sessions/worker-body-profile?profile=default",
        json={"pinned": True, "profile": "worker"},
    )
    assert resp.status_code == 200, resp.text
    assert _pinned_state(worker_home, "worker-body-profile") is True


def test_patch_session_unknown_profile_404s(client):
    """A nonexistent profile must 404, not silently write to the default store."""
    resp = client.patch(
        "/api/sessions/whatever?profile=no-such-profile",
        json={"pinned": True},
    )
    assert resp.status_code == 404


def test_patch_session_no_profile_resolves_cross_profile_owner(client):
    """A profile-less PATCH of a session that lives in another served LOCAL
    profile must resolve the owner instead of 404-ing into the default store.

    The Desktop renderer's session sync drops the profile hint when it cannot
    attribute the row to the active profile — the failure behind
    'unpinning a cross-profile session silently resurrects it'."""
    from hermes_constants import get_hermes_home

    worker_home = _seed_profile_session("worker", "worker-no-hint")

    # Pin it first WITH the hint (as the app does when it knows the owner).
    resp = client.patch("/api/sessions/worker-no-hint?profile=worker", json={"pinned": True})
    assert resp.status_code == 200, resp.text
    assert _pinned_state(worker_home, "worker-no-hint") is True

    # Now unpin with NO profile at all — the app's failing shape. The default
    # store cannot resolve the id, so the handler must hunt the single owner.
    resp = client.patch("/api/sessions/worker-no-hint", json={"pinned": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["pinned"] is False
    assert _pinned_state(worker_home, "worker-no-hint") is False

    # The default store was never touched.
    assert _pinned_state(get_hermes_home(), "worker-no-hint") is None


def test_delete_session_no_profile_resolves_cross_profile_owner(client):
    """A profile-less DELETE of a session owned by another served profile
    must delete it — not report an idempotent already_absent no-op that lets
    the row silently resurrect on the next sidebar refetch."""
    from hermes_constants import get_hermes_home

    worker_home = _seed_profile_session("worker", "worker-delete-no-hint")

    resp = client.delete("/api/sessions/worker-delete-no-hint")
    assert resp.status_code == 200, resp.text
    assert "already_absent" not in resp.json()
    assert resp.json()["ok"] is True

    # Row is gone from the worker store, and the default store never held it.
    assert _pinned_state(worker_home, "worker-delete-no-hint") is None
    assert _pinned_state(get_hermes_home(), "worker-delete-no-hint") is None


def test_bulk_delete_honors_query_profile(client):
    """POST /api/sessions/bulk-delete?profile=worker deletes worker rows."""
    worker_home = _seed_profile_session("worker", "worker-bulk-del")

    resp = client.post(
        "/api/sessions/bulk-delete?profile=worker",
        json={"ids": ["worker-bulk-del"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] == 1
    assert _pinned_state(worker_home, "worker-bulk-del") is None  # row gone
