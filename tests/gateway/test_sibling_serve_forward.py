"""Regression & polarity tests for cross-surface sibling-serve prompt forwarding (CF-80 / CF-65).

Covers:
1. Mid-turn cross-surface ``prompt.submit`` forwarding & live event relay over loopback WebSocket
   when a sibling ``serve`` process (e.g. ``mobile`` on ``:9119`` vs ``desktop`` ``serve --isolated``)
   holds an active turn lease (``turn_active=True``) in ``active_sessions.json``.
2. Automatic lease takeover (``SIBLING_LEASE_RECLAIMED``) when the sibling ``serve`` process reports
   ``running=False`` on ``session.resume`` despite ``turn_active=True`` in ``active_sessions.json``.
3. ``_take_over_idle_cross_surface_lease`` polarity when ``turn_active=True`` has no live turn marker
   in ``interrupted_turns.json``: allowed when ``is_mobile_handoff and age_s >= 10.0``, refused when
   ``age_s < 10.0`` or when a live turn marker exists.
4. ``_history_has_dangling_tool_splice`` & ``_adopt_out_of_band_turns`` healing when a session was
   resumed mid-turn (stripping ``tool_calls`` from the trailing ``assistant`` row) before the sibling
   surface appended the ``tool`` result row.
5. Cross-surface ``session.interrupt`` and ``session.redirect`` forwarding to the sibling ``serve`` owner.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import types
from pathlib import Path

from websockets.asyncio.server import serve as ws_serve

from hermes_cli.active_sessions import (
    ActiveSessionLease,
    _lease_entry,
    _lease_paths,
    _write_entries,
)
from tui_gateway.prompt_turn import _history_has_dangling_tool_splice
from tui_gateway.sibling_serve_forward import (
    SIBLING_LEASE_RECLAIMED,
    try_interrupt_sibling_serve_owner,
    try_submit_to_sibling_serve_owner,
)


def _start_mock_sibling_ws_server(*, running: bool = True, submit_status: str = "redirected"):
    """Spin up a real loopback WebSocket server speaking the ``/api/ws`` JSON-RPC protocol."""
    received_rpcs: list[dict] = []
    ready_evt = threading.Event()
    stop_evt = threading.Event()
    bound: dict[str, int] = {}

    async def _handler(ws):
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"type": "gateway.ready"}}) + "\n")
        resumed_count = 0
        async for raw in ws:
            msg = json.loads(raw)
            received_rpcs.append(msg)
            rid = msg.get("id")
            method = msg.get("method")
            if method == "session.resume":
                resumed_count += 1
                is_running = running if resumed_count == 1 else False
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {
                                "session_id": "owner_live_sid",
                                "session_key": msg["params"]["session_id"],
                                "running": is_running,
                                "queued": None,
                            },
                        }
                    )
                    + "\n"
                )
            elif method == "prompt.submit":
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {"status": submit_status},
                        }
                    )
                    + "\n"
                )
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "event",
                            "params": {
                                "type": "tool.start",
                                "session_id": "owner_live_sid",
                                "tool": "terminal",
                            },
                        }
                    )
                    + "\n"
                )
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "event",
                            "params": {
                                "type": "message.delta",
                                "session_id": "owner_live_sid",
                                "text": "Steered reply from mobile owner.",
                            },
                        }
                    )
                    + "\n"
                )
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "event",
                            "params": {
                                "type": "message.complete",
                                "session_id": "owner_live_sid",
                                "text": "Steered reply from mobile owner.",
                            },
                        }
                    )
                    + "\n"
                )
            elif method in ("session.interrupt", "session.redirect", "session.steer"):
                txt_param = str((msg.get("params") or {}).get("text") or "")
                corr_status = "rejected" if txt_param == "REJECT_ME" else ("interrupted" if method == "session.interrupt" else "redirected")
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {"status": corr_status},
                        }
                    )
                    + "\n"
                )

    def _thread_main():
        async def _serve():
            async with ws_serve(_handler, "127.0.0.1", 0) as srv:
                for sock in srv.sockets:
                    bound["port"] = sock.getsockname()[1]
                    break
                ready_evt.set()
                while not stop_evt.is_set():
                    await asyncio.sleep(0.05)

        asyncio.run(_serve())

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()
    assert ready_evt.wait(timeout=5.0)
    return bound["port"], received_rpcs, stop_evt


def _make_fake_server(emitted: list[tuple[str, str, dict | None]], adopted: list[str]):
    return types.SimpleNamespace(
        _TRUNCATION_PARAMS=("truncate_to_ordinal", "truncate_to_row_id"),
        _session_resume_lock=threading.Lock(),
        current_transport=lambda: "desktop_transport",
        _rebind_live_transport=lambda sid, session, t: None,
        _is_mobile_session=lambda session: session.get("source") == "mobile",
        _session_source=lambda session: str(session.get("source") or "desktop"),
        _ensure_active_session_slot=lambda sid, session: "still_owned",
        _adopt_out_of_band_turns=lambda session: adopted.append(str(session.get("session_key") or "")),
        _emit=lambda evt_type, sid, payload=None: emitted.append((evt_type, sid, payload)),
        _emit_session_info=lambda sid, session: emitted.append(("session.info.local", sid, None)),
        _lease_metadata=lambda sid, turn_active=None: {"live_session_id": sid, "turn_active": bool(turn_active)},
        _attach_lease=lambda session, lease: session.__setitem__("active_session_lease", lease),
        _ok=lambda rid, result: {"jsonrpc": "2.0", "id": rid, "result": result},
    )


def test_mid_turn_cross_surface_submit_relays_events_and_adopts_history(tmp_path: Path, monkeypatch):
    """When mobile holds an active turn lease, desktop prompt.submit forwards & streams events instead of 4090."""
    port, received_rpcs, stop_evt = _start_mock_sibling_ws_server(running=True, submit_status="redirected")
    try:
        profile_home = tmp_path / "hermes_home"
        profile_home.mkdir(parents=True, exist_ok=True)
        state_path, _ = _lease_paths(registry_home=profile_home)
        entry = _lease_entry(
            lease_id="mobile_lease_1",
            session_id="20261111_140951_081c55",
            surface="mobile",
            metadata={"live_session_id": "owner_live_sid", "turn_active": True},
            track_liveness=True,
        )
        _write_entries(state_path, [entry])

        monkeypatch.setattr(
            "tui_gateway.sibling_serve_forward.discover_sibling_serve_endpoint",
            lambda pid: ("127.0.0.1", port, "test-token") if pid == os.getpid() else None,
        )

        emitted: list[tuple[str, str, dict | None]] = []
        adopted: list[str] = []
        server = _make_fake_server(emitted, adopted)
        session = {
            "session_key": "20261111_140951_081c55",
            "profile_home": str(profile_home),
            "source": "desktop",
            "running": False,
            "history_lock": threading.Lock(),
        }

        resp = try_submit_to_sibling_serve_owner(
            server, 42, "desktop_sid", session, "Kolla även drift-guard", {}, None
        )
        assert isinstance(resp, dict)
        assert resp["result"]["status"] == "streaming"

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(ev[0] == "message.complete" for ev in emitted):
                break
            time.sleep(0.05)

        event_types = [ev[0] for ev in emitted]
        assert "message.start" in event_types
        assert "tool.start" in event_types
        assert "message.delta" in event_types
        assert "message.complete" in event_types
        assert adopted == ["20261111_140951_081c55"]
        assert session["running"] is False
        assert "_sibling_serve_relay" not in session
        methods_called = [r.get("method") for r in received_rpcs]
        assert "session.resume" in methods_called
        assert "prompt.submit" in methods_called
    finally:
        stop_evt.set()


def test_sibling_serve_already_idle_reclaims_lease_for_local_turn(tmp_path: Path, monkeypatch):
    """When the sibling serve reports running=False on session.resume, reclaim the lease locally."""
    port, _, stop_evt = _start_mock_sibling_ws_server(running=False)
    try:
        profile_home = tmp_path / "hermes_home_idle"
        profile_home.mkdir(parents=True, exist_ok=True)
        state_path, _ = _lease_paths(registry_home=profile_home)
        entry = _lease_entry(
            lease_id="mobile_lease_idle",
            session_id="20261111_140951_081c55",
            surface="mobile",
            metadata={"live_session_id": "owner_live_sid", "turn_active": True},
            track_liveness=True,
        )
        _write_entries(state_path, [entry])

        monkeypatch.setattr(
            "tui_gateway.sibling_serve_forward.discover_sibling_serve_endpoint",
            lambda pid: ("127.0.0.1", port, "test-token"),
        )

        emitted: list[tuple[str, str, dict | None]] = []
        adopted: list[str] = []
        server = _make_fake_server(emitted, adopted)
        session = {
            "session_key": "20261111_140951_081c55",
            "profile_home": str(profile_home),
            "source": "desktop",
            "running": False,
            "history_lock": threading.Lock(),
        }

        resp = try_submit_to_sibling_serve_owner(
            server, 43, "desktop_sid", session, "Ny tur lokalt", {}, None
        )
        assert resp is SIBLING_LEASE_RECLAIMED
        assert isinstance(session.get("active_session_lease"), ActiveSessionLease)
        assert adopted == ["20261111_140951_081c55"]
    finally:
        stop_evt.set()


def test_cross_surface_interrupt_forwards_to_sibling_owner(tmp_path: Path, monkeypatch):
    """session.interrupt on a non-owning surface forwards session.interrupt to the active sibling serve."""
    port, received_rpcs, stop_evt = _start_mock_sibling_ws_server(running=True)
    try:
        entry = {"session_id": "20261111_140951_081c55", "pid": 999991, "metadata": {"turn_active": True}}
        monkeypatch.setattr("tui_gateway.sibling_serve_forward._find_lease_owner_entry", lambda key, home: entry)
        monkeypatch.setattr(
            "tui_gateway.sibling_serve_forward.discover_sibling_serve_endpoint",
            lambda pid: ("127.0.0.1", port, "test-token"),
        )
        session = {
            "session_key": "20261111_140951_081c55",
            "profile_home": str(tmp_path),
            "history_lock": threading.Lock(),
        }
        server = _make_fake_server([], [])
        assert try_interrupt_sibling_serve_owner(server, session) is True
        assert any(r.get("method") == "session.interrupt" for r in received_rpcs)
    finally:
        stop_evt.set()


def test_history_has_dangling_tool_splice_polarity():
    """Positive & negative polarity for mid-turn resume stripped tool_calls followed by a foreign tool row."""
    # Positive control: assistant had tool_calls stripped (empty/None) at mid-turn resume, followed by tool row.
    corrupted = [
        {"role": "user", "content": "Kolla alla jobb", "_row_id": 1},
        {"role": "assistant", "content": "Kör kontroll...", "_row_id": 2},
        {"role": "tool", "content": "exit 0", "tool_call_id": "call_1", "_row_id": 3},
    ]
    assert _history_has_dangling_tool_splice(corrupted) is True

    # Negative control: assistant has tool_calls intact before tool rows.
    clean = [
        {"role": "user", "content": "Kolla alla jobb", "_row_id": 1},
        {
            "role": "assistant",
            "content": "Kör kontroll...",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
            "_row_id": 2,
        },
        {"role": "tool", "content": "exit 0", "tool_call_id": "call_1", "_row_id": 3},
        {"role": "assistant", "content": "Klart!", "_row_id": 4},
    ]
    assert _history_has_dangling_tool_splice(clean) is False


def test_take_over_idle_cross_surface_lease_stale_turn_active_polarity(tmp_path: Path, monkeypatch):
    """Stale turn_active=True with NO live turn marker is taken over after 10s on mobile/desktop handoff,
    but refused within 10s or when a live turn marker exists."""
    from tui_gateway import server as gw_server

    profile_home = tmp_path / "hermes_home_polarity"
    profile_home.mkdir(parents=True, exist_ok=True)
    state_path, _ = _lease_paths(registry_home=profile_home)

    def _seed_lease(age_seconds: float, surface: str = "mobile"):
        now = time.time()
        entry = _lease_entry(
            lease_id="stale_turn_active_lease",
            session_id="20261111_140951_081c55",
            surface=surface,
            metadata={"live_session_id": "mobile_sid", "turn_active": True},
            track_liveness=True,
        )
        entry["started_at"] = now - age_seconds
        entry["updated_at"] = now - age_seconds
        _write_entries(state_path, [entry])

    session = {
        "session_key": "20261111_140951_081c55",
        "profile_home": str(profile_home),
        "source": "desktop",
        "history_lock": threading.Lock(),
    }

    # 1. Negative control A: live turn marker exists -> refused even after 30s
    _seed_lease(30.0, surface="mobile")
    monkeypatch.setattr(gw_server, "_session_has_live_turn_marker", lambda s, k: True)
    assert gw_server._take_over_idle_cross_surface_lease(
        "desktop_sid", session, "20261111_140951_081c55", surface="desktop"
    ) is False

    # 2. Negative control B: no live turn marker, but age < 10s (admission window) -> refused
    _seed_lease(3.0, surface="mobile")
    monkeypatch.setattr(gw_server, "_session_has_live_turn_marker", lambda s, k: False)
    assert gw_server._take_over_idle_cross_surface_lease(
        "desktop_sid", session, "20261111_140951_081c55", surface="desktop"
    ) is False

    # 3. Positive control: no live turn marker and age >= 10s on mobile/desktop handoff -> taken over!
    _seed_lease(15.0, surface="mobile")
    assert gw_server._take_over_idle_cross_surface_lease(
        "desktop_sid", session, "20261111_140951_081c55", surface="desktop"
    ) is True
    assert isinstance(session.get("active_session_lease"), ActiveSessionLease)


def test_publish_host_rendezvous_writes_peer_and_host_record(tmp_path: Path, monkeypatch):
    """_publish_host_rendezvous must write both serve-peer-<pid>.json and host-serve.json cleanly."""
    from gateway import host_rendezvous as hr
    from hermes_cli import web_server
    from tui_gateway.sibling_serve_forward import discover_sibling_serve_endpoint

    lock_dir = tmp_path / "gw-locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    monkeypatch.setattr(hr, "cleanup_on_exit", lambda role: None)
    monkeypatch.setattr("atexit.register", lambda fn, *a, **kw: fn)
    monkeypatch.setattr(web_server, "is_desktop_owned_backend", lambda: False)

    web_server._publish_host_rendezvous("127.0.0.1", 19119, isolated=False)
    try:
        rec = hr.read_record(hr.ROLE_SERVE)
        assert rec is not None
        assert rec.port == 19119
        peer_file = lock_dir / f"serve-peer-{os.getpid()}.json"
        assert peer_file.is_file()
        # Verify discover_sibling_serve_endpoint resolves via peer file for a different simulated pid
        peer_data = json.loads(peer_file.read_text(encoding="utf-8"))
        fake_pid = os.getpid() + 77777
        (lock_dir / f"serve-peer-{fake_pid}.json").write_text(
            json.dumps({**peer_data, "pid": fake_pid}), encoding="utf-8"
        )
        ep = discover_sibling_serve_endpoint(fake_pid)
        assert ep == ("127.0.0.1", 19119, peer_data["token"])
    finally:
        hr.release_host_lock(hr.ROLE_SERVE)


def test_sibling_serve_correction_forwards_and_verifies_rpc_polarity(tmp_path: Path, monkeypatch):
    """try_correct_sibling_serve_owner must return True on accepted redirect/steer and False on 'rejected'."""
    from tui_gateway.sibling_serve_forward import try_correct_sibling_serve_owner

    profile_home = tmp_path / "hermes_home_corr"
    profile_home.mkdir(parents=True, exist_ok=True)
    state_path, _ = _lease_paths(registry_home=profile_home)

    port, received_rpcs, stop_evt = _start_mock_sibling_ws_server(running=True)
    try:
        entry = _lease_entry(
            lease_id="mobile_lease_corr",
            session_id="20261111_140951_081c55",
            surface="mobile",
            metadata={"live_session_id": "owner_live_sid", "turn_active": True},
            track_liveness=True,
        )
        _write_entries(state_path, [entry])
        monkeypatch.setattr(
            "tui_gateway.sibling_serve_forward.discover_sibling_serve_endpoint",
            lambda pid: ("127.0.0.1", port, "test-token"),
        )
        session = {
            "session_key": "20261111_140951_081c55",
            "profile_home": str(profile_home),
            "source": "desktop",
            "history": [],
            "history_lock": threading.Lock(),
            "running": False,
        }
        server = _make_fake_server([], [])
        assert try_correct_sibling_serve_owner(server, session, "session.redirect", "Byt spår nu") is True
        assert try_correct_sibling_serve_owner(server, session, "session.steer", "REJECT_ME") is False
        methods = [r.get("method") for r in received_rpcs]
        assert "session.redirect" in methods
        assert "session.steer" in methods
    finally:
        stop_evt.set()



