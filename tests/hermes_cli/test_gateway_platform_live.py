"""``hermes config set`` reports what the running gateway did, from the gateway's own verdict.

A platform the gateway tried to start and could not bring up (Feishu with its SDK not importable) was
listed as "applied it now: started feishu" and, on the next line, as "could not start": both cannot be true.
"""
import json
import socket
import threading
import time

import pytest

from hermes_cli import gateway_platform_live


def _note(monkeypatch, tmp_path, answer):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_platform_live, "reconcile_running_gateway_platforms", lambda name, home: answer)
    return gateway_platform_live.config_set_live_note("FEISHU_APP_SECRET")


def test_a_platform_that_did_not_come_up_is_never_reported_as_applied(monkeypatch, tmp_path):
    note = _note(monkeypatch, tmp_path, {"applied": True, "changed": True, "started": ["feishu"], "restarted": [],
                                         "stopped": [], "platforms": {"feishu": "unavailable"}, "reasons": {}})

    assert "applied it now" not in note
    assert note.count("feishu") == 1 and "could not start feishu" in note


def test_each_platform_is_reported_once_with_its_own_outcome(monkeypatch, tmp_path):
    note = _note(monkeypatch, tmp_path, {
        "applied": True, "changed": True, "started": ["qqbot", "feishu"], "restarted": [], "stopped": ["slack"],
        "platforms": {"qqbot": "retrying", "feishu": "unavailable"},
        "reasons": {"feishu": "the running gateway cannot import this platform's Python dependencies"}})

    applied, failed = note.splitlines()
    assert applied.strip() == "Running gateway applied it now: started qqbot; stopped slack"
    assert "could not start feishu: the running gateway cannot import" in failed


@pytest.fixture
def live_socket(tmp_path, monkeypatch):
    """A real control socket at the profile home's path: records each connection; answers with
    ``reply`` when set, else holds the connection open without answering (a gateway that blocks)."""
    from gateway.control_socket import resolve_server_socket_path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    bind, pointer = resolve_server_socket_path(tmp_path)
    bind.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(bind))
    server.listen()
    if pointer is not None:
        pointer.write_text(str(bind), encoding="utf-8")
    state = {"connections": 0, "reply": None, "held": []}

    def serve():
        while True:
            try:
                peer, _ = server.accept()
            except OSError:
                return
            state["connections"] += 1
            peer.recv(65536)
            if state["reply"] is None:
                state["held"].append(peer)
                continue
            peer.sendall(json.dumps({"ok": True, "id": 1, "result": state["reply"]}).encode() + b"\n")
            peer.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield state
    server.close()
    for peer in state["held"]:
        peer.close()
    bind.unlink(missing_ok=True)


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("key", ["model", "terminal.backend", "OPENROUTER_API_KEY"])
def test_a_key_that_cannot_change_a_platform_never_waits_on_the_gateway(live_socket, key):
    started = time.monotonic()
    assert gateway_platform_live.config_set_live_note(key) is None
    assert time.monotonic() - started < 5
    assert live_socket["connections"] == 0


@pytest.mark.platforms("linux", "macos")
def test_a_platform_key_asks_the_gateway_and_says_what_is_still_connecting(live_socket):
    live_socket["reply"] = {"applied": True, "pending": True, "starting": ["telegram"], "restarting": [],
                            "stopping": []}
    note = gateway_platform_live.config_set_live_note("TELEGRAM_BOT_TOKEN")
    assert live_socket["connections"] == 1
    assert note == "  Running gateway is starting telegram (still connecting)"
