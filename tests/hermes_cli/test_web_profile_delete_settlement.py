"""``DELETE /api/profiles/{name}``: a settle-pending delete reports a partial success, not a 500.

Deleting a profile can end with the directory gone and the durable session/routing identity
settlement still pending (#112727, delete side of #111926); ``delete_profile`` reports that as
``ProfileIdentitySettlementPending`` *after* the filesystem half completed. Flattened by the
router's generic exception mapping into HTTP 500, it read as "delete failed" — a dashboard client
then retried into a 404. The endpoint owns the partial-success shape; a genuine filesystem
failure (rmtree) must keep the 500.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Path.home() and HERMES_HOME both point into the temp dir (profile ops are HOME-anchored)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


@pytest.fixture()
def client(profile_env):
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


def test_settle_pending_delete_reports_partial_success(client, monkeypatch):
    """The profile really is deleted; only the identity settlement is missing — report both."""
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "_cleanup_gateway_service", lambda *a, **k: None)
    # A live multiplexer owns the routing index and none is reachable here, so the settlement
    # stays pending — the exact state delete_profile reports after the directory is gone.
    monkeypatch.setattr(profiles_mod, "_live_default_multiplexer", lambda: True)
    profiles_mod.create_profile("gone", no_alias=True)
    gone_dir = profiles_mod.get_profile_dir("gone")

    resp = client.delete("/api/profiles/gone")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["settlement_pending"] is True
    assert body["identity_settled"] is False
    assert body["retry_command"] == "hermes profile purge-identity gone"
    assert body["path"] == str(gone_dir)
    # Not lying about the filesystem half: the directory is really gone.
    assert not gone_dir.exists()


def test_filesystem_remove_failure_still_reports_500(client, monkeypatch):
    """A plain rmtree failure keeps the generic 500 — not every delete error is a partial success."""
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "_cleanup_gateway_service", lambda *a, **k: None)

    def boom(profile_dir, onexc_handler):
        raise OSError("device busy")

    monkeypatch.setattr(profiles_mod, "_rmtree_with_retry", boom)
    profiles_mod.create_profile("gone", no_alias=True)

    resp = client.delete("/api/profiles/gone")

    assert resp.status_code == 500
