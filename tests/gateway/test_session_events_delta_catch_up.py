"""The gateway session authority's reconnect replay survives a long streamed reply.

``SessionEvents.since`` (``session.events.since`` on the canonical socket) and ``catch_up`` read the
same shared replay ring. A reply of more chunks than the ring holds used to answer
``snapshot_required``, so a reconnecting viewer dropped its live turn and re-snapshotted.
"""

from tui_gateway import event_replay


def test_a_reconnect_after_a_long_reply_replays_it_instead_of_requiring_a_snapshot():
    from gateway.session_events import SessionEvents

    event_replay.reset_replay_state()
    event_replay.configure(coalesce_deltas=True)  # dashboard.replay_coalesce_deltas: true
    stream = SessionEvents()
    stream.publish("chat", {"turn_id": "t"}, event_type="message.start")
    reply, offset = "", 0
    for index in range(1500):
        chunk = f"token{index} "
        stream.publish("chat", {"text": chunk, "turn_id": "t", "segment": 0, "offset": offset},
                       event_type="message.delta")
        offset += len(chunk)
        reply += chunk
    epoch, _sequence = stream.watermark()

    replay = stream.since(epoch, 0)
    assert not replay["snapshot_required"]
    assert [frame["type"] for frame in replay["events"]] == ["message.start", "message.delta"]
    delta = replay["events"][-1]
    assert delta["session_id"] == "chat" and delta["payload"]["text"] == reply and delta["payload"]["offset"] == 0

    class Sink:
        frames = []

        def write(self, frame):
            self.frames.append(frame)
            return True

    with stream.lock:
        stream.catch_up(Sink(), epoch, 0)
    assert [frame["params"]["type"] for frame in Sink.frames] == ["message.start", "message.delta"]
    event_replay.configure(coalesce_deltas=False)
    event_replay.reset_replay_state()
