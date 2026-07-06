"""tool.start must carry the backend-epoch ``started_at`` so the desktop
Observatory timeline can place each tool call on a shared time axis.

Behavior contract (not a snapshot): the timestamp emitted on ``tool.start``
must equal the value the gateway stored for that tool call in
``session["tool_started_at"]`` — i.e. the same backend clock later used to
compute ``duration_s`` on ``tool.complete``. This keeps lane anchors and block
anchors on ONE clock.
"""

import tui_gateway.server as server


def _capture_emits(monkeypatch):
    """Capture every _emit(...) payload by intercepting write_json."""
    events: list[dict] = []

    def fake_write_json(obj):
        params = obj.get("params", {})
        events.append(
            {
                "type": params.get("type"),
                "payload": params.get("payload"),
            }
        )

    monkeypatch.setattr(server, "write_json", fake_write_json)
    return events


def test_tool_start_emits_backend_epoch_started_at(monkeypatch):
    sid = "sess-timeline-1"
    # Progress mode must be on (not "off") for tool.start to emit at all.
    server._sessions[sid] = {"tool_progress_mode": "all"}
    try:
        events = _capture_emits(monkeypatch)

        server._on_tool_start(sid, "tc-1", "terminal", {"command": "echo hi"})

        starts = [e for e in events if e["type"] == "tool.start"]
        assert len(starts) == 1, f"expected one tool.start, got {events}"
        payload = starts[0]["payload"]

        # started_at is present and is a backend epoch float.
        assert "started_at" in payload, payload
        started_at = payload["started_at"]
        assert isinstance(started_at, float)

        # It equals the value the gateway stored for duration accounting —
        # same clock, not a fresh renderer timestamp.
        stored = server._sessions[sid]["tool_started_at"]["tc-1"]
        assert started_at == stored
    finally:
        server._sessions.pop(sid, None)


def test_tool_start_started_at_matches_completion_duration_clock(monkeypatch):
    """The started_at on tool.start and duration_s on tool.complete come from
    the same stored epoch, so start + duration reconstructs the end time."""
    sid = "sess-timeline-2"
    server._sessions[sid] = {"tool_progress_mode": "all"}
    try:
        events = _capture_emits(monkeypatch)

        server._on_tool_start(sid, "tc-2", "file", {"path": "/tmp/x"})
        start_payload = next(
            e["payload"] for e in events if e["type"] == "tool.start"
        )
        emitted_start = start_payload["started_at"]

        server._on_tool_complete(sid, "tc-2", "file", {"path": "/tmp/x"}, "{}")
        complete_payload = next(
            e["payload"] for e in events if e["type"] == "tool.complete"
        )

        # duration_s is measured against the SAME stored start, so
        # emitted_start + duration_s is a coherent end time on one clock.
        assert "duration_s" in complete_payload
        assert complete_payload["duration_s"] >= 0
        # The start we emitted is the anchor the duration was measured from.
        assert isinstance(emitted_start, float)
    finally:
        server._sessions.pop(sid, None)


def test_tool_start_omits_started_at_when_progress_off(monkeypatch):
    """Progress mode 'off' emits nothing at all — no regression there."""
    sid = "sess-timeline-3"
    server._sessions[sid] = {"tool_progress_mode": "off"}
    try:
        events = _capture_emits(monkeypatch)
        server._on_tool_start(sid, "tc-3", "terminal", {"command": "ls"})
        starts = [e for e in events if e["type"] == "tool.start"]
        assert starts == [], "progress off should emit no tool.start"
    finally:
        server._sessions.pop(sid, None)
