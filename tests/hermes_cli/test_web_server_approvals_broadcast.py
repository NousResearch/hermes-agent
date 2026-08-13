"""Approvals config saves must re-emit session.info to live gateway sessions.

Regression for the desktop "YOLO toggle does nothing / flips back off" bug:
the settings page saves ``approvals.mode`` through REST ``PUT /api/config``
(and the raw editor through ``PUT /api/config/raw``), which wrote config.yaml
and emitted nothing. Enforcement follows the file immediately (the approval
gate re-reads config per command), but every live session's YOLO/approval
indicator repaints only on a ``session.info`` event, so the UI kept showing
stale bypass state, in the dangerous direction. The ``config.set`` RPC path
already re-emits after a mode flip; these tests pin the REST paths to the
same contract.
"""

import types

import pytest


@pytest.fixture
def client(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


@pytest.fixture
def broadcast_calls(monkeypatch):
    """Stub the in-memory gateway module seam and record broadcasts."""
    import sys

    calls = []
    # tests/conftest.py's session-reaper teardown walks tui_gateway.server
    # attributes; the stub must carry an empty _sessions to survive it.
    stub = types.SimpleNamespace(
        broadcast_session_info=lambda: calls.append(True),
        _sessions={},
    )
    monkeypatch.setitem(sys.modules, "tui_gateway.server", stub)
    return calls


class TestApprovalsSaveBroadcast:
    def test_get_shaped_record_roundtrip_does_not_broadcast(self, client, broadcast_calls):
        """The settings page PUTs the defaulted GET record back verbatim on
        every autosave. That must not broadcast: disk holds sparse YAML while
        GET returns defaults, so a block-level compare is always-unequal (the
        review-caught spam bug). Only an effective mode change may emit."""
        record = client.get("/api/config").json()
        assert "approvals" in record

        first = client.put("/api/config", json={"config": record})
        assert first.status_code == 200
        second = client.put("/api/config", json={"config": record})
        assert second.status_code == 200
        assert not broadcast_calls, (
            "autosaving the unmodified GET record broadcast session.info; "
            "every settings autosave would walk all live sessions"
        )

        flipped = {**record, "approvals": {**record["approvals"], "mode": "off"}}
        resp = client.put("/api/config", json={"config": flipped})
        assert resp.status_code == 200
        assert len(broadcast_calls) == 1, (
            "an actual approvals.mode change in the GET-shaped record must "
            "broadcast exactly once"
        )

    def test_approvals_mode_change_broadcasts(self, client, broadcast_calls):
        resp = client.put("/api/config", json={"config": {"approvals": {"mode": "off"}}})
        assert resp.status_code == 200
        assert broadcast_calls, (
            "PUT /api/config changed approvals.mode but no session.info "
            "broadcast reached the gateway, so live sessions keep painting "
            "stale YOLO/approval state"
        )

    def test_non_approvals_change_does_not_broadcast(self, client, broadcast_calls):
        resp = client.put("/api/config", json={"config": {"display": {"skin": "mono"}}})
        assert resp.status_code == 200
        assert not broadcast_calls, (
            "a save that never touched approvals must not spam session.info"
        )

    def test_approvals_noop_save_does_not_broadcast(self, client, broadcast_calls):
        first = client.put("/api/config", json={"config": {"approvals": {"mode": "off"}}})
        assert first.status_code == 200
        broadcast_calls.clear()

        again = client.put("/api/config", json={"config": {"approvals": {"mode": "off"}}})
        assert again.status_code == 200
        assert not broadcast_calls, (
            "saving an identical approvals block is a no-op and must not "
            "re-emit session.info"
        )

    def test_other_profile_save_does_not_broadcast(self, client, broadcast_calls, monkeypatch, tmp_path):
        from hermes_cli import web_server

        profile_dir = tmp_path / "profiles" / "other"
        profile_dir.mkdir(parents=True)
        monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda name: profile_dir)

        resp = client.put(
            "/api/config",
            json={"config": {"approvals": {"mode": "off"}}, "profile": "other"},
        )
        assert resp.status_code == 200
        assert not broadcast_calls, (
            "a profile-scoped save targets a different HERMES_HOME than this "
            "process's gateway sessions; broadcasting our own sessions' "
            "unchanged state is wrong"
        )

    def test_gateway_not_imported_is_a_noop(self, client, monkeypatch):
        import sys

        monkeypatch.delitem(sys.modules, "tui_gateway.server", raising=False)
        resp = client.put("/api/config", json={"config": {"approvals": {"mode": "smart"}}})
        assert resp.status_code == 200
        assert "tui_gateway.server" not in sys.modules, (
            "the broadcast seam must not IMPORT the gateway; a process "
            "without one has no sessions to notify"
        )

    def test_raw_save_deleting_approvals_block_broadcasts(self, client, broadcast_calls):
        seed = client.put(
            "/api/config/raw",
            json={"yaml_text": "approvals:\n  mode: 'off'\n"},
        )
        assert seed.status_code == 200
        broadcast_calls.clear()

        # Full-document replacement that drops the approvals block entirely:
        # effective mode falls back to default (manual), so indicators must
        # repaint.
        resp = client.put(
            "/api/config/raw",
            json={"yaml_text": "display:\n  skin: default\n"},
        )
        assert resp.status_code == 200
        assert broadcast_calls, (
            "deleting the approvals block changes the effective mode and "
            "must broadcast"
        )

    def test_raw_save_approvals_change_broadcasts(self, client, broadcast_calls):
        resp = client.put(
            "/api/config/raw",
            json={"yaml_text": "approvals:\n  mode: 'off'\n"},
        )
        assert resp.status_code == 200
        assert broadcast_calls, (
            "PUT /api/config/raw changed approvals but no session.info "
            "broadcast reached the gateway"
        )

    def test_raw_save_without_approvals_change_does_not_broadcast(self, client, broadcast_calls):
        seed = client.put(
            "/api/config/raw",
            json={"yaml_text": "approvals:\n  mode: manual\ndisplay:\n  skin: default\n"},
        )
        assert seed.status_code == 200
        broadcast_calls.clear()

        resp = client.put(
            "/api/config/raw",
            json={"yaml_text": "approvals:\n  mode: manual\ndisplay:\n  skin: mono\n"},
        )
        assert resp.status_code == 200
        assert not broadcast_calls


class TestGatewayBroadcastHelper:
    def test_broadcast_session_info_emits_for_live_sessions(self, _isolate_hermes_home, monkeypatch):
        """tui_gateway.server.broadcast_session_info walks _sessions and emits."""
        from tui_gateway import server

        emitted = []
        monkeypatch.setattr(
            server, "_emit_session_info_for_session",
            lambda sid, sess: emitted.append(sid),
        )
        monkeypatch.setattr(
            server, "_sessions",
            {"s1": {"agent": object()}, "s2": {"agent": object()}},
        )

        server.broadcast_session_info()

        assert sorted(emitted) == ["s1", "s2"]

    def test_approvals_slash_mirror_broadcasts(self, _isolate_hermes_home, monkeypatch):
        """/approvals <mode> through the slash worker persists config out of
        band; the mirror must repaint live sessions and the bare read-only
        form must not."""
        from tui_gateway import server

        calls = []
        monkeypatch.setattr(server, "broadcast_session_info", lambda: calls.append(True))

        session = {"agent": None}
        server._mirror_slash_side_effects("sid1", session, "/approvals off")
        assert calls, "/approvals <mode> writes approvals.mode and must broadcast"

        calls.clear()
        server._mirror_slash_side_effects("sid1", session, "/approvals")
        assert not calls, "bare /approvals only reads the mode"
