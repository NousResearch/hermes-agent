"""C4 compaction with the warm handoff (``compression.warm_handoff: off | on | auto``), end to end.

A compaction first sends the last ordinary request of the session once more, with the history rows that
came after it and one appended user instruction. The fake provider sees it as a main request (it carries
the tools). Checks:

* the warm request keeps the exact message prefix and settings of the last ordinary request;
* an accepted handoff replaces the auxiliary summary call, for ``/compress`` and for automatic compaction;
* a rejected or failed handoff falls back to one auxiliary summary call in the same attempt;
* a request or execution middleware that changes the captured part sends no warm request;
* the mode is off by default; ``auto`` needs a reported prompt cache;
* every C4 invariant still holds after the compaction (same helpers as the manual and automatic suites).
"""

from __future__ import annotations

import dataclasses

import pytest

from agent.context_compressor_warm import WARM_HANDOFF_HEADINGS, WARM_HANDOFF_INSTRUCTION
from tests.e2e.core.compaction._helpers import GOOD_SUMMARY_TOKEN, drive, generate_transcript
from tests.e2e.core.compaction.test_compaction_manual import (
    HIGH_THRESHOLD,
    POST_TURNS,
    PRE_TURNS,
    _compress_via_tui,
    _run_after,
)
from tests.fakes.fake_llm_provider import Error, Text, ToolCall

WARM_TOKEN = "WARM-HANDOFF-OK"
WARM_HANDOFF = "\n\n".join(f"{heading}\n- {WARM_TOKEN} line." for heading in WARM_HANDOFF_HEADINGS)
WARM_CONFIG = "  warm_handoff: on\n"
WARM_MARK = WARM_HANDOFF_INSTRUCTION.splitlines()[0]
AUTOMATIC_KINDS = ("tools", "parallel", "chat")
# Only these keys may differ between the final ordinary request and the warm request. The handoff has its own
# reply limit (checked apart).
_LIMIT_KEYS = ("max_tokens", "max_completion_tokens")
_REQUEST_ONLY_KEYS = ("messages", "stream", "stream_options", *_LIMIT_KEYS)


def _session(make_scenario, tmp_path, seed, *, extra=WARM_CONFIG):
    sc = make_scenario("good", threshold_tokens=HIGH_THRESHOLD, extra=extra)
    specs = generate_transcript(seed, PRE_TURNS + POST_TURNS + 1, tmp_path / "work",
                                kinds=("tools", "parallel", "chat"))
    for spec in specs[:PRE_TURNS]:
        sc.run_turn(spec)
    return sc, specs


def _compress_with_main_reply(sc, reply):
    """Queue the provider's answer for a main request, run ``/compress``; return the new main request bodies."""
    first = len(sc._snapshots)
    with sc._qlock:
        sc._queue = [reply]
    removed = _compress_via_tui(sc, "")
    with sc._qlock:
        unconsumed, sc._queue = len(sc._queue), []
    return removed, [body for body, _ in sc._snapshots[first:]], unconsumed


def _settings(body):
    return {key: value for key, value in body.items() if key not in _REQUEST_ONLY_KEYS}


@pytest.mark.parametrize("seed", (5, 17))
def test_manual_compress_uses_the_warm_handoff(make_scenario, tmp_path, seed):
    sc, specs = _session(make_scenario, tmp_path, seed)
    final = sc._snapshots[-1][0]
    final_answer = specs[PRE_TURNS - 1].steps[-1].text
    calls_before = len(sc.summary_calls)

    removed, bodies, unconsumed = _compress_with_main_reply(sc, Text(WARM_HANDOFF, cached_tokens=777))

    assert removed > 0 and sc.committed_compactions() > 0, "/compress did not compact"
    assert len(sc.summary_calls) == calls_before, "an accepted warm handoff must replace the aux summary call"
    assert unconsumed == 0 and len(bodies) == 1, "expected exactly one warm request"
    warm, n = bodies[0], len(final["messages"])
    assert warm["messages"][:n] == final["messages"], "the warm request changed the request prefix"
    assert warm["messages"][n] == {"role": "assistant", "content": final_answer}
    assert warm["messages"][n + 1]["role"] == "user" and len(warm["messages"]) == n + 2
    assert all(heading in warm["messages"][n + 1]["content"] for heading in WARM_HANDOFF_HEADINGS)
    assert warm.get("stream") is False and "stream_options" not in warm
    assert _settings(warm) == _settings(final), "the warm request changed a request setting"
    if not any(final.get(key) for key in _LIMIT_KEYS):
        assert sorted(key for key in _LIMIT_KEYS if warm.get(key) == 8192) in (["max_tokens"], ["max_completion_tokens"])
    assert sc.agent.context_compressor._last_warm_handoff == {
        "used": True, "reason": "accepted", "elapsed_s": sc.agent.context_compressor._last_warm_handoff["elapsed_s"],
        "prompt_tokens": sc.server.requests[-1]["usage"]["prompt_tokens"], "cache_read_tokens": 777}
    assert any(WARM_TOKEN in str(m.get("content")) for m in sc.history), "the handoff is not in the history"
    assert not any(GOOD_SUMMARY_TOKEN in str(m.get("content")) for m in sc.history)
    _run_after(sc, specs[PRE_TURNS:], "after warm /compress")


@pytest.mark.parametrize(("reply", "reason"), [
    (Text("Plain text without the handoff headings."), "refused:heading_missing"),
    (Error(500, "warm upstream exploded"), "unavailable:provider_error"),
])
def test_a_failed_warm_handoff_falls_back_to_the_aux_summary(make_scenario, tmp_path, reply, reason):
    sc, specs = _session(make_scenario, tmp_path, 5)
    calls_before = len(sc.summary_calls)

    removed, bodies, unconsumed = _compress_with_main_reply(sc, reply)

    assert removed > 0 and sc.committed_compactions() > 0, "/compress did not compact"
    assert unconsumed == 0 and len(bodies) == 1, "expected exactly one warm request and no retry"
    assert len(sc.summary_calls) - calls_before == 1, "the fallback must be one aux summary call"
    assert sc.agent.context_compressor._last_warm_handoff["reason"] == reason
    assert any(GOOD_SUMMARY_TOKEN in str(m.get("content")) for m in sc.history), "aux summary missing"
    assert not any(WARM_TOKEN in str(m.get("content")) for m in sc.history)
    _run_after(sc, specs[PRE_TURNS:], f"after fallback /compress ({reason})")


def _rewrite_prefix_row(request):
    request["messages"][1]["content"] = "A different captured request."


def _rewrite_tools(request):
    request["tools"] = []


def _rewrite_model(request):
    request["model"] = "another-model"


def _register(monkeypatch, kind, rewrite):
    """A real middleware of the given kind that changes the captured part of the warm request only."""
    import copy

    from hermes_cli.plugins import _delivery_manager

    def request_middleware(request=None, **context):
        if context.get("purpose") != "context_prefix_request":
            return None
        changed = copy.deepcopy(request)
        rewrite(changed)
        return {"request": changed}

    def execution_middleware(request=None, next_call=None, **context):
        if context.get("purpose") != "context_prefix_request":
            return next_call()
        changed = copy.deepcopy(request)
        rewrite(changed)
        return next_call(changed)

    callback = request_middleware if kind == "llm_request" else execution_middleware
    monkeypatch.setitem(_delivery_manager()._middleware, kind, [callback])


@pytest.mark.parametrize("rewrite", (_rewrite_prefix_row, _rewrite_tools, _rewrite_model))
@pytest.mark.parametrize("kind", ("llm_request", "llm_execution"))
def test_a_middleware_rewrite_of_the_captured_part_falls_back(make_scenario, tmp_path, monkeypatch, kind, rewrite):
    # The model would summarize a request that is not the history (and the cache would miss): no warm request
    # reaches the provider, and the same attempt makes one aux summary call.
    sc, specs = _session(make_scenario, tmp_path, 5)
    calls_before = len(sc.summary_calls)
    _register(monkeypatch, kind, rewrite)

    removed, bodies, _unconsumed = _compress_with_main_reply(sc, Text(WARM_HANDOFF))

    assert removed > 0 and sc.committed_compactions() > 0, "/compress did not compact"
    assert bodies == [], "a rewritten warm request reached the provider"
    assert len(sc.summary_calls) - calls_before == 1, "the fallback must be one aux summary call"
    assert sc.agent.context_compressor._last_warm_handoff["reason"] == "unavailable:middleware_rewrite"
    assert not any(WARM_TOKEN in str(m.get("content")) for m in sc.history)
    _run_after(sc, specs[PRE_TURNS:], f"after a {kind} rewrite")


def test_manual_compress_without_the_option_sends_no_warm_request(make_scenario, tmp_path):
    sc, specs = _session(make_scenario, tmp_path, 17, extra="")
    calls_before = len(sc.summary_calls)

    removed, bodies, _unconsumed = _compress_with_main_reply(sc, Text(WARM_HANDOFF))

    assert removed > 0 and bodies == [], "the default compressor must not send a warm request"
    assert len(sc.summary_calls) - calls_before == 1
    assert sc.agent.context_compressor._last_warm_handoff == {"used": False, "reason": "skipped:off"}
    _run_after(sc, specs[PRE_TURNS:], "after default /compress")


def _answer_warm_requests(sc, *, cached_tokens=0):
    """Answer every warm request with a handoff before the scripted main replies see it.

    ``cached_tokens`` > 0 makes the scripted text and tool-call replies report a prompt cache, as a caching
    server does: an automatic compaction can come right after a tool-call turn.
    Returns the list that collects the warm request bodies."""
    warm_bodies = []
    scripted = sc._main

    def main(rec):
        last = rec["body"]["messages"][-1]
        # The instruction is the last block of the last user row: a trailing user row comes in front of it.
        text = str(last.get("content") or "")
        if last.get("role") == "user" and (text.startswith(WARM_MARK) or "\n\n" + WARM_MARK in text):
            warm_bodies.append(rec["body"])
            return Text(WARM_HANDOFF, cached_tokens=777)
        reply = scripted(rec)
        if cached_tokens and isinstance(reply, (Text, ToolCall)) and not reply.cached_tokens:
            reply = dataclasses.replace(reply, cached_tokens=cached_tokens)
        return reply

    sc._main = main
    return warm_bodies


def _handoff_reached_the_model(sc):
    return any(WARM_TOKEN in str(m.get("content")) for turn in sc.turns for body in turn.main_requests
               for m in body["messages"])


@pytest.mark.parametrize("seed", (11, 23))
def test_automatic_compaction_uses_the_warm_handoff(make_scenario, tmp_path, seed):
    sc = make_scenario("good", extra=WARM_CONFIG)
    warm_bodies = _answer_warm_requests(sc)
    drive(sc, seed, tmp_path / "work", kinds=AUTOMATIC_KINDS)
    assert sc.committed_compactions() > 0, "the seeded session never compacted"
    assert warm_bodies, "automatic compaction sent no warm request"
    for body in warm_bodies:
        assert body.get("stream") is False and "stream_options" not in body
        assert body["tools"], "the warm request lost the tools of the main request"
    assert _handoff_reached_the_model(sc), "an accepted handoff never reached the model"


def test_auto_mode_uses_the_warm_handoff_when_the_server_reports_a_cache(make_scenario, tmp_path):
    sc = make_scenario("good", extra="  warm_handoff: auto\n")
    warm_bodies = _answer_warm_requests(sc, cached_tokens=64)
    drive(sc, 11, tmp_path / "work", kinds=AUTOMATIC_KINDS)
    assert sc.committed_compactions() > 0
    assert warm_bodies, "auto mode sent no warm request although the server reported a cache"
    assert _handoff_reached_the_model(sc)


def test_auto_mode_without_a_reported_cache_keeps_the_aux_summary(make_scenario, tmp_path):
    sc = make_scenario("good", extra="  warm_handoff: auto\n")
    warm_bodies = _answer_warm_requests(sc)
    drive(sc, 11, tmp_path / "work", kinds=AUTOMATIC_KINDS)
    assert sc.committed_compactions() > 0 and sc.summary_calls
    assert warm_bodies == [], "auto mode must not send a warm request without a reported cache"
