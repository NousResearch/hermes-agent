"""C17 prompt-cache prefix byte-stability + usage accounting across REAL processes and surfaces.

One durable session is driven for 10+ turns through fresh processes of the real entrypoints that
can continue it — the ``tui_gateway`` stdio JSON-RPC server (what the TUI/Desktop spawn) and the
``hermes chat -q --resume`` oneshot CLI — switching process and working directory between turns,
with a parallel tool batch and (gateway journeys) one manual ``session.compress`` on the way.

Invariants over the fake provider's recorded request stream:

* every main request is a byte-identical extension of the previous one — same tools array, same
  system prompt, same earlier messages — across resume, process restart and cwd change; the ONLY
  sanctioned break is the first request after the compaction, and it breaks exactly once;
* every process boundary replays exactly the persisted model view (nothing re-rendered);
* each logical message is persisted once (storage rows, not reader output); rows never shrink;
* ``sessions`` / ``session_model_usage`` equal what the provider billed, request by request.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.core.history._helpers import (
    NO_BACKGROUND_REVIEW,
    OFFLINE_CONFIG,
    Script,
    Spawned,
    TuiGateway,
    assert_replay_equals_persisted,
    assert_rows_exactly_once,
    assert_usage_matches,
    big,
    child_env,
    integrity_ok,
    lineage,
    prefix_breaks,
    row_counts,
    run_oneshot,
    tools_breaks,
    views,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

# A hop is (surface, cwd, turns); a turn is its prompt, "TOOLS:<prompt>" (parallel tool batch
# first) or "/compress" (gateway only). Every hop is a FRESH process.
Hop = tuple[str, str, list[str]]

JOURNEYS: dict[str, list[Hop]] = {
    # Same surface, four gateway processes, two cwds, compaction mid-way, resume after it.
    "tui_gateway_restarts": [
        ("gw", "a", ["gw1 first turn", "TOOLS:gw1 runs a parallel batch", "gw1 third turn"]),
        ("gw", "b", ["gw2 resumed turn", "gw2 another turn"]),
        ("gw", "a", ["gw3 turn before compress", "/compress", "gw3 turn after compress", "gw3 more"]),
        ("gw", "b", ["gw4 resumed after compaction", "gw4 last turn"]),
    ],
    # Same surface, eight oneshot processes, cwd flips between them.
    "oneshot_restarts": [
        ("cli", "a", ["cli1 first turn"]),
        ("cli", "a", ["TOOLS:cli2 runs a parallel batch"]),
        ("cli", "b", ["cli3 from another cwd"]),
        ("cli", "a", ["cli4 back home"]),
        ("cli", "b", ["TOOLS:cli5 tools away again"]),
        ("cli", "a", ["cli6 turn"]),
        ("cli", "b", ["cli7 turn"]),
        ("cli", "a", ["cli8 last turn"]),
    ],
    # Surface switch on one session: TUI <-> oneshot, cwd change, compaction, resume after it.
    "surface_switch": [
        ("gw", "a", ["sw1 first turn", "TOOLS:sw1 runs a parallel batch", "sw1 third turn"]),
        ("cli", "a", ["sw2 oneshot turn"]),
        ("cli", "b", ["sw3 oneshot from another cwd"]),
        ("gw", "b", ["sw4 gateway resumes", "sw4 another turn"]),
        ("cli", "a", ["sw5 oneshot turn"]),
        ("gw", "a", ["sw6 before compress", "/compress", "sw6 after compress", "sw6 more"]),
        ("cli", "b", ["sw7 oneshot after compaction"]),
    ],
}

KNOWN_BROKEN: dict[str, str] = {}


class ToolsArrayDrift(Exception):
    """The tools array changed between requests of one session. Not an AssertionError: the
    known-bug xfail matches only this, so every other invariant still fails the test."""


@pytest.fixture
def world(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    hermes_home = Path(os.environ["HERMES_HOME"])  # hermetic per-test tmp dir from the conftest
    cwds = {"a": tmp_path / "work-a", "b": tmp_path / "work-b"}
    for d in cwds.values():
        d.mkdir()
    script = Script()
    spawned = Spawned()
    with FakeLLMServer(script) as srv:
        write_hermes_home(hermes_home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        yield {"srv": srv, "script": script, "home": home, "hermes_home": hermes_home,
               "cwds": cwds, "spawned": spawned}
        leaked = spawned.reap()
        assert not leaked, f"subprocesses outlived the test: {leaked}"


def _queue_turn(script: Script, prompt: str) -> str:
    """Script the model for one turn; returns the text the user types."""
    if prompt.startswith("TOOLS:"):
        prompt = prompt[len("TOOLS:"):]
        tag = prompt.split()[0]
        script.actions.append(ToolCall("terminal", {"command": f"echo {tag}-a"},
                                       parallel=[("terminal", {"command": f"echo {tag}-b"})]))
    # Bulky answers so the later /compress genuinely shrinks the context; distinct usage per
    # answer so a double-counted or dropped request is visible.
    n = len(prompt)
    script.actions.append(Text(big(f"answer-{prompt.replace(' ', '-')}", 12000),
                               prompt_tokens=2000 + n, completion_tokens=40 + n, cached_tokens=1500))
    return prompt


def run_journey(world: dict, hops: list[Hop]) -> tuple[str, list[tuple[int, str]], int | None]:
    """Drive the hops; returns (durable sid, [(first request idx, hop label)], compaction idx)."""
    srv, script, home = world["srv"], world["script"], world["hermes_home"]
    sid: str | None = None
    openings: list[tuple[int, str]] = []
    compaction_idx: int | None = None
    rows_total = 0
    for n, (surface, cwd_key, turns) in enumerate(hops, 1):
        cwd = world["cwds"][cwd_key]
        label = f"hop {n} ({surface} in work-{cwd_key})"
        persisted = views(home, sid)[0] if sid else None
        opened_at = len(srv.main_requests())
        env = child_env(world["home"], home, cwd)
        if surface == "gw":
            with TuiGateway(env, cwd, world["spawned"]) as gw:
                live = gw.resume(sid) if sid else gw.create()
                for turn in turns:
                    if turn == "/compress":
                        res = gw.call_when_idle("session.compress", {"session_id": live}) or {}
                        summary = res.get("summary") or {}
                        flags = {k: summary.get(k) for k in ("noop", "aborted", "refused_would_grow")}
                        assert res.get("status") == "compressed" and not any(flags.values()) and (
                            (res.get("after_messages") or 0) < (res.get("before_messages") or 0)), (
                            f"{label}: session.compress did not compact: "
                            f"{ {k: v for k, v in res.items() if k != 'messages'} }")
                        compaction_idx = len(srv.main_requests())
                        continue
                    gw.submit(live, _queue_turn(script, turn))
                sid = sid or gw.stored[live]
        else:
            for turn in turns:
                _out, got = run_oneshot(env, cwd, _queue_turn(script, turn), world["spawned"], resume=sid)
                assert sid is None or got == sid, f"{label}: --resume {sid} continued {got}"
                sid = got
        main = srv.main_requests()
        assert len(main) > opened_at, f"{label}: sent no model request"
        if persisted is not None:
            assert_replay_equals_persisted(main[opened_at], persisted, f"{label} opening request")
        assert_rows_exactly_once(home, sid, label)
        total = row_counts(home, sid)["total"]
        assert total >= rows_total, f"{label}: durable rows shrank {rows_total} -> {total}"
        rows_total = total
        openings.append((opened_at, label))
    assert not script.actions, f"scripted responses never consumed: {script.actions}"
    return sid, openings, compaction_idx


@pytest.mark.parametrize("journey", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=ToolsArrayDrift, reason=KNOWN_BROKEN[name]))
    if name in KNOWN_BROKEN else name
    for name in JOURNEYS
])
def test_request_prefix_is_byte_stable_across_processes(world, journey):
    sid, openings, compaction_idx = run_journey(world, JOURNEYS[journey])
    srv, home = world["srv"], world["hermes_home"]
    main = srv.main_requests()
    assert len(main) >= 10

    def where(i: int) -> str:
        return f"request {i} (in {max((o for o in openings if o[0] <= i), default=(0, '?'))[1]})"

    # System prompt + messages first; the tools array is checked last, on its own, so a known
    # tools-drift xfail cannot mask a message-prefix, usage or integrity regression.
    breaks = prefix_breaks(main, tools=False)
    unexpected = [(i, why) for i, why in breaks if i != compaction_idx]
    assert not unexpected, "prompt-cache prefix broke outside the compaction boundary:\n" + "\n".join(
        f"  {where(i)}: {why}" for i, why in unexpected)
    if compaction_idx is not None:
        assert [i for i, _ in breaks] == [compaction_idx], (
            f"the manual /compress must break the prefix exactly once, at request {compaction_idx}: {breaks}")

    assert_usage_matches(home, lineage(home, sid), srv.requests, f"after journey {journey}")
    integrity_ok(home)

    drift = [(i, why) for i, why in tools_breaks(main) if i != compaction_idx]
    if drift:
        raise ToolsArrayDrift("tools array changed within one session:\n" + "\n".join(
            f"  {where(i)}: {why}" for i, why in drift))
