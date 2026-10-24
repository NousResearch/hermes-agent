"""Behaviour contracts for ``hermes wall`` (hermes_cli/subcommands/wall.py).

``wall`` exists so the operator's exit status answers "did the whole fleet get
the notice?" — so these tests pin the three things that make that true: the
roster, the verify-the-settle rule, and the never-truncate body rule.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from hermes_cli.subcommands import wall


def _agents(*names: str) -> list[wall.Agent]:
    return [wall.Agent(name=n, home=Path(f"/tmp/wall-home/{n}"), target=f"fleet/{n}")
            for n in names]


def _envelope(result: str, **extra) -> dict:
    return {"object": "hermes.peer.send_result", "result": result, **extra}


class _Transport:
    """Scripted stand-in for the peer transport, recording every call it gets."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[dict] = []

    def __call__(self, target, chunk, wait_seconds, timeout):
        self.calls.append({"target": target, "chunk": chunk, "wait": wait_seconds})
        queue = self.answers.get(target)
        if not queue:
            return 1, None, "Peer 'fleet' is unreachable: connection refused"
        return queue[0] if len(queue) == 1 else queue.pop(0)

    def chunks_for(self, target: str) -> list[str]:
        return [c["chunk"] for c in self.calls if c["target"] == target]


def _run(message, agents, runner, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    code = wall.run_wall(message, out=out, err=err, runner=runner, sleep=lambda _s: None,
                         registry_fn=lambda: (agents, []), **kwargs)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# body planning: chunk, never truncate
# --------------------------------------------------------------------------- #

def test_a_body_that_fits_is_delivered_verbatim():
    body = "gateway restarting in five minutes"
    assert wall.plan_chunks(body) == [body]


def test_a_body_that_is_already_clipped_is_refused():
    with pytest.raises(wall.WallRefused) as refusal:
        wall.plan_chunks("the notice begins here and then ...[truncated]")
    assert "truncat" in str(refusal.value).lower()


def _text_of(part: str) -> str:
    """A part's text without its ordinal prefix and continuation marker."""
    body = part.split(" ", 1)[1]
    return body.split(" [continues in ")[0]


def test_a_long_body_becomes_ordinal_parts_that_lose_nothing():
    body = " ".join(f"w{i:03d}" for i in range(300))  # ~1.5k chars to split
    parts = wall.plan_chunks(body, max_chunk_chars=1000, max_chunks=5)
    assert len(parts) > 1
    assert all(len(part) <= 1000 for part in parts)
    assert parts[0].startswith("(1/")
    assert parts[-1].startswith(f"({len(parts)}/{len(parts)}) ")
    rebuilt = " ".join(_text_of(part) for part in parts)
    assert rebuilt == body


def test_a_part_never_stops_mid_clause_and_says_when_it_continues():
    """Parts arrive as separate messages in separate turns, so while a recipient
    holds only part one its ending has to be readable as deliberate: a clause
    boundary plus an explicit marker, not a bare character cut."""
    body = ("first clause of the notice. second clause carries the detail; "

            "third clause closes it. and the tail runs on for a while. " * 8).strip()
    parts = wall.plan_chunks(body, max_chunk_chars=300, max_chunks=20)
    assert len(parts) > 1
    assert all(len(part) <= 300 for part in parts)
    for index, part in enumerate(parts[:-1], start=1):
        assert part.startswith(f"({index}/{len(parts)}) ")
        assert part.endswith(f"[continues in {index + 1}/{len(parts)}]")
        assert _text_of(part).endswith((".", ";", "!"))
    assert "[continues in" not in parts[-1]


def test_a_hard_wrapped_body_splits_on_the_sentence_not_on_the_wrap():
    """Newlines in a hand-wrapped body sit mid-sentence and render as a space in
    the delivered message, so a wrap is not a safe split point."""
    body = ("the notice starts here and keeps going for a long while, deliberately wide enough\n"
            "to need more than one part. the second sentence is also long enough to carry the\n"
            "rest of the body over the limit. and a third one follows it. " * 4).strip()
    parts = wall.plan_chunks(body, max_chunk_chars=300, max_chunks=20)
    assert len(parts) > 1
    for part in parts[:-1]:
        assert _text_of(part).endswith((".", ";", "!", ":"))


def test_a_body_too_large_to_chunk_is_refused():
    body = " ".join(f"w{i:03d}" for i in range(3000))
    with pytest.raises(wall.WallRefused) as refusal:
        wall.plan_chunks(body, max_chunk_chars=200, max_chunks=3)
    assert "parts" in str(refusal.value)


# --------------------------------------------------------------------------- #
# what the transport reported -> the state the operator reads
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(("result", "expected"), [
    ("delivered", wall.STATE_DELIVERED),
    ("receipt", wall.STATE_QUEUED),
    ("failed", wall.STATE_FAILED),
    ("refused", wall.STATE_REFUSED),
    ("unknown", wall.STATE_UNKNOWN),
])
def test_transport_result_enum_maps_onto_wall_states(result, expected):
    state, _detail = wall._state_from_attempt(0, _envelope(result), "")
    assert state == expected


def test_a_missing_route_is_reported_unroutable_not_unreachable():
    state, detail = wall._state_from_attempt(
        1, None, "Peer 'fleet' rejected the request (HTTP 404): profile not served")
    assert state == wall.STATE_UNROUTABLE
    assert "404" in detail


def test_the_ledger_can_only_confirm_a_delivery_never_invent_one(tmp_path):
    assert wall.ledger_state(tmp_path, "") == ""
    assert wall.ledger_state(tmp_path, "0" * 32) == ""


def test_the_peer_dm_command_carries_the_target_and_the_json_envelope(monkeypatch):
    seen: dict = {}

    class _Proc:
        returncode = 0
        stdout = "Peer 'fleet'\n{\"result\": \"delivered\"}\n"
        stderr = ""

    def _fake_run(cmd, **_kwargs):
        seen["cmd"] = list(cmd)
        return _Proc()

    monkeypatch.setattr(wall.subprocess, "run", _fake_run)
    code, envelope, _stderr = wall._run_peer_dm("fleet/alfa", "hi", 30.0, None)
    assert code == 0
    assert envelope == {"result": "delivered"}
    cmd = seen["cmd"]
    start = cmd.index("peer")
    assert cmd[start:start + 5] == ["peer", "dm", "fleet/alfa", "hi", "--json"]
    assert cmd[-2:] == ["--wait", "30.0"]


# --------------------------------------------------------------------------- #
# the roster
# --------------------------------------------------------------------------- #

def test_every_agent_reached_exits_zero_with_one_roster_line_each():
    agents = _agents("alfa", "bravo", "charlie")
    transport = _Transport({a.target: [(0, _envelope("delivered", reply="ok"), "")] for a in agents})
    code, out, _err = _run("maintenance in five minutes", agents, transport)
    assert code == 0
    lines = [line for line in out.splitlines() if line.startswith("wall target=")]
    assert len(lines) == 3
    for name in ("alfa", "bravo", "charlie"):
        assert any(f"target=fleet/{name} state=delivered" in line for line in lines), out
    assert "reached=3/3" in out
    assert "exit=0" in out


def test_an_unreachable_agent_fails_the_wall_and_is_named():
    agents = _agents("alfa", "bravo", "charlie")
    transport = _Transport({
        "fleet/alfa": [(0, _envelope("delivered"), "")],
        "fleet/bravo": [(1, None, "Peer 'fleet' is unreachable: connection refused")],
        "fleet/charlie": [(0, _envelope("delivered"), "")],
    })
    code, out, _err = _run("switching model pins", agents, transport)
    assert code == 1
    assert "target=fleet/bravo state=unreachable" in out
    assert "target=fleet/alfa state=delivered" in out  # one dead lane, not the fleet
    assert "reached=2/3" in out


def test_an_unserved_registered_profile_is_reported_not_skipped():
    agents = _agents("alfa")
    transport = _Transport({"fleet/alfa": [(0, _envelope("delivered"), "")]})
    out, err = io.StringIO(), io.StringIO()
    code = wall.run_wall("note", out=out, err=err, runner=transport, sleep=lambda _s: None,
                         registry_fn=lambda: (agents, ["ghost"]))
    assert code == 1
    assert "target=ghost state=unroutable" in out.getvalue()
    assert "unroutable=1" in out.getvalue()


# --------------------------------------------------------------------------- #
# verify the settle
# --------------------------------------------------------------------------- #

def test_an_unsettled_receipt_is_retried_once_then_reported_not_reached():
    agents = _agents("alfa")
    transport = _Transport({
        "fleet/alfa": [(0, _envelope("receipt", status="queued", detail="target busy"), "")],
    })
    code, out, _err = _run("heads up", agents, transport)
    assert code == 1
    assert len(transport.calls) == 2          # exactly one retry, never a loop
    assert "state=queued attempts=2" in out
    assert "reached=0/1" in out


def test_a_receipt_that_settles_in_the_ledger_counts_as_reached(monkeypatch):
    agents = _agents("alfa")
    transport = _Transport({
        "fleet/alfa": [(0, _envelope("receipt", status="queued", delivery_id="a" * 32), "")],
    })
    monkeypatch.setattr(wall, "ledger_state",
                        lambda home, delivery_id: wall.STATE_DELIVERED
                        if delivery_id == "a" * 32 else "")
    code, out, _err = _run("heads up", agents, transport)
    assert code == 0
    assert len(transport.calls) == 1          # settled: no retry needed
    assert "state=delivered" in out


def test_a_refused_body_is_not_sent_at_all():
    agents = _agents("alfa")
    transport = _Transport({})
    code, _out, err = _run("attempted notice ...[truncated]", agents, transport)
    assert code == 2
    assert transport.calls == []
    assert "refused" in err


# --------------------------------------------------------------------------- #
# chunking reaches every agent, in order
# --------------------------------------------------------------------------- #

def test_a_long_body_reaches_every_agent_as_ordered_parts():
    agents = _agents("alfa", "bravo")
    transport = _Transport({a.target: [(0, _envelope("delivered"), "")] for a in agents})
    body = " ".join(f"w{i:03d}" for i in range(300))
    code, out, _err = _run(body, agents, transport)
    assert code == 0
    for name in ("alfa", "bravo"):
        chunks = transport.chunks_for(f"fleet/{name}")
        assert len(chunks) > 1
        assert chunks[0].startswith("(1/")
        assert chunks == sorted(chunks)
    assert len(transport.chunks_for("fleet/alfa")) == len(transport.chunks_for("fleet/bravo"))


def test_reason_labels_the_broadcast_without_touching_the_body():
    agents = _agents("alfa")
    transport = _Transport({"fleet/alfa": [(0, _envelope("delivered"), "")]})
    code, out, _err = _run("body text", agents, transport, reason="deploy")
    assert code == 0
    assert "reason: deploy" in out
    assert transport.calls[0]["chunk"] == "body text"


def test_json_result_carries_the_roster_and_the_exit_code():
    agents = _agents("alfa")
    transport = _Transport({"fleet/alfa": [(0, _envelope("delivered"), "")]})
    code, out, _err = _run("note", agents, transport, as_json=True)
    payload = json.loads(out)
    assert payload["exit_code"] == 0 == code
    assert payload["reached"] == 1
    assert payload["roster"][0]["target"] == "fleet/alfa"
    assert payload["roster"][0]["state"] == wall.STATE_DELIVERED
