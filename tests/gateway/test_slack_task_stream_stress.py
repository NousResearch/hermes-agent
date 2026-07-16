"""Stress tests for the reasoning-dedup + Slack task-stream paths (#59009/#59010).

These go beyond the targeted regression tests: randomized provider
misbehavior against ``normalize_reasoning_delta`` (the reconstruction
invariant must hold under any mix of clean tokens, cumulative echoes, and
overlap re-deliveries), and concurrency/rollover stress against
``SlackTaskStream`` (ordered per-card transitions under parallel event
storms; reasoning-card replay across rollover).

Seeded RNG — failures reproduce exactly.
"""
from __future__ import annotations

import asyncio
import random

from agent.chat_completion_helpers import (
    _MIN_REASONING_OVERLAP,
    normalize_reasoning_delta,
)
from gateway.slack_task_stream import SlackTaskStream

from tests.gateway.test_slack_task_stream import FakeSlackClient, _make_stream


# ---------------------------------------------------------------------------
# normalize_reasoning_delta — randomized reconstruction invariant
# ---------------------------------------------------------------------------

_WORDS = (
    "the quick brown fox jumps over lazy dog while thinking about "
    "merge conflicts and rebasing the feature branch onto main again"
).split()


def _token_stream(rng: random.Random, n_tokens: int) -> list[str]:
    """A realistic incremental token stream (short word fragments)."""
    out = []
    for _ in range(n_tokens):
        w = rng.choice(_WORDS)
        out.append(w if rng.random() < 0.5 else f" {w}")
    return out


def _consume(chunks: list[str]) -> str:
    """Feed chunks through the normalizer exactly like the streaming loop."""
    parts: list[str] = []
    for chunk in chunks:
        text = normalize_reasoning_delta("".join(parts), chunk)
        if text:
            parts.append(text)
    return "".join(parts)


class TestNormalizerFuzz:
    def test_clean_streams_are_passthrough(self):
        """No corruption → byte-identical reconstruction. 200 random streams."""
        rng = random.Random(59009)
        for _ in range(200):
            tokens = _token_stream(rng, rng.randint(1, 80))
            assert _consume(tokens) == "".join(tokens)

    def test_cumulative_echo_injection_always_deduped(self):
        """Inject full-accumulated echoes at random positions — the echo must
        never change the reconstruction. 200 random streams. Echoes are only
        injected once the accumulated text has cleared the dedup gate
        (mirrors reality: providers echo late-stream, never on token two)."""
        rng = random.Random(62829)
        for _ in range(200):
            tokens = _token_stream(rng, rng.randint(2, 60))
            clean = "".join(tokens)
            corrupted: list[str] = []
            acc = ""
            for t in tokens:
                corrupted.append(t)
                acc += t
                if len(acc) >= _MIN_REASONING_OVERLAP and rng.random() < 0.15:
                    corrupted.append(acc)  # cumulative echo of everything so far
            assert _consume(corrupted) == clean, (
                f"echo injection changed reconstruction (seed case): {corrupted[:5]}…"
            )

    def test_overlap_redelivery_injection_always_deduped(self):
        """Inject reconnect-style re-deliveries (tail of accumulated + new
        tokens) — the overlapped head must be trimmed, the new tail kept.
        Injected overlaps clear the ≥24-char gate (the documented contract:
        below-gate overlaps are deliberately treated as legitimate text;
        real reconnect re-deliveries overlap by hundreds of chars)."""
        rng = random.Random(59010)
        for _ in range(200):
            tokens = _token_stream(rng, rng.randint(4, 60))
            clean = "".join(tokens)
            # Build the corrupted stream: at one random point, replace a token
            # with (long tail of accumulated so far) + that token. Skip cases
            # where the accumulated text hasn't cleared the gate yet.
            idx = rng.randint(2, len(tokens) - 1)
            acc_before = "".join(tokens[:idx])
            if len(acc_before) < _MIN_REASONING_OVERLAP:
                continue
            tail_len = rng.randint(_MIN_REASONING_OVERLAP, len(acc_before))
            tail = acc_before[-tail_len:]
            corrupted = tokens[:idx] + [tail + tokens[idx]] + tokens[idx + 1:]
            assert _consume(corrupted) == clean

    def test_mixed_corruption_storm(self):
        """Both corruption modes interleaved in one stream, repeatedly.
        Corruption only injected past the dedup gate, per the contract."""
        rng = random.Random(4990)
        for _ in range(100):
            tokens = _token_stream(rng, rng.randint(6, 50))
            clean = "".join(tokens)
            corrupted: list[str] = []
            acc = ""
            for i, t in enumerate(tokens):
                if len(acc) >= _MIN_REASONING_OVERLAP and rng.random() < 0.1:
                    corrupted.append(acc)  # echo
                if len(acc) > _MIN_REASONING_OVERLAP and rng.random() < 0.1:
                    corrupted.append(acc[-(_MIN_REASONING_OVERLAP + 5):] + t)  # overlap
                    acc += t
                    continue
                corrupted.append(t)
                acc += t
            assert _consume(corrupted) == clean

    def test_legit_repetition_never_dropped(self):
        """A stream that legitimately repeats the same short phrase many times
        (models do this) must reconstruct in full — the anti-overcorrection
        guarantee the reviewer asked for."""
        phrase = " check the list"
        chunks = [phrase] * 50
        assert _consume(chunks) == phrase * 50


# ---------------------------------------------------------------------------
# SlackTaskStream — concurrency + rollover stress
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestStreamConcurrencyStress:
    def test_parallel_event_storm_keeps_per_card_order(self):
        """40 tools × (start, finish) scheduled as concurrent coroutines —
        the send lock must serialize appends so every card still transitions
        in_progress → complete with no error and exactly one stream open."""
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def one_tool(i: int):
            await stream.task_started(i, "terminal", preview=f"job {i}")
            await asyncio.sleep(random.Random(i).random() * 0.01)
            await stream.task_finished(i, "terminal", duration=0.01)

        async def scenario():
            await asyncio.gather(*(one_tool(i) for i in range(1, 41)))
            await stream.stop()

        _run(scenario())
        assert stream.disabled is False
        assert client.stream_opens() == 1
        chunks = [c for c in client.appended_chunks() if c.get("type") == "task_update"]
        for i in range(1, 41):
            statuses = [c["status"] for c in chunks if c.get("id") == f"t{i}"]
            assert statuses, f"card t{i} missing"
            # Per-card: never complete before in_progress; ends complete.
            assert statuses[0] == "in_progress" and statuses[-1] == "complete", (
                f"t{i} out of order: {statuses}"
            )

    def test_storm_with_interleaved_reasoning_and_subagents(self):
        """Tools + reasoning bursts + subagent events all interleaved
        concurrently — nothing raises, stream survives, cards consistent."""
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            jobs = []
            for i in range(1, 11):
                jobs.append(stream.task_started(i, "terminal", preview=f"j{i}"))
                jobs.append(stream.reasoning_update(
                    f"thinking burst {i} with enough characters to clear the "
                    f"minimum threshold for card creation easily"
                ))
                jobs.append(stream.subagent_event(
                    "subagent.start", f"k{i}", goal=f"goal {i}", number=i
                ))
            await asyncio.gather(*jobs)
            await asyncio.gather(*(
                stream.task_finished(i, "terminal", duration=0.1) for i in range(1, 11)
            ))
            await asyncio.gather(*(
                stream.subagent_event("subagent.complete", f"k{i}", goal=f"goal {i}")
                for i in range(1, 11)
            ))
            await stream.stop()

        _run(scenario())
        assert stream.disabled is False
        ids = {c.get("id") for c in client.appended_chunks() if c.get("type") == "task_update"}
        assert {f"t{i}" for i in range(1, 11)} <= ids
        assert {f"sub_k{i}" for i in range(1, 11)} <= ids


class TestRolloverStress:
    def test_size_threshold_rollover(self):
        """Exceeding ROLLOVER_MAX_CHARS proactively opens a fresh stream."""
        client = FakeSlackClient()
        stream = _make_stream(client, rollover_chars=500)

        async def scenario():
            for i in range(1, 8):
                await stream.task_started(
                    i, "terminal",
                    preview="x" * 100,
                    details="y" * 200,  # each card ~300+ chars of net content
                )

        _run(scenario())
        assert stream.disabled is False
        assert client.stream_opens() >= 2, "size threshold never triggered rollover"

    def test_rollover_replays_open_reasoning_card_with_full_details(self):
        """The open 💭 card's tracked chunk only carries the last-sent delta
        (details append server-side) — on rollover the replay must substitute
        the FULL locally-accumulated burst so the fresh card isn't truncated."""
        client = FakeSlackClient()
        stream = _make_stream(client)
        burst1 = "first portion of the thought with plenty of substance here"
        burst2 = " second portion appended later with more substance still"

        async def scenario():
            await stream.task_started(1, "terminal", preview="warm up")
            await stream.reasoning_update(burst1)
            await stream.reasoning_update(burst2)
            # Force a reactive rollover on the next append.
            client.fail_appends_with = "message_not_in_streaming_state"
            await stream.task_started(2, "read_file", preview="next")

        _run(scenario())
        assert stream.disabled is False
        assert client.stream_opens() == 2
        # Find replayed 💭 chunks (appended after the second startStream).
        start_indices = [i for i, (n, _) in enumerate(client.calls) if n == "startStream"]
        post_rollover = [
            chunk
            for name, kw in client.calls[start_indices[1]:]
            if name == "appendStream"
            for chunk in kw.get("chunks", [])
        ]
        think_replays = [
            c for c in post_rollover if str(c.get("id", "")).startswith("think")
        ]
        assert think_replays, "open reasoning card not replayed after rollover"
        replay_details = think_replays[0].get("details", "")
        assert burst1 in replay_details and burst2.strip() in replay_details, (
            "rollover replay lost accumulated reasoning: "
            f"{replay_details[:120]!r}"
        )

    def test_repeated_reactive_rollovers_within_budget(self):
        """Several mid-turn stream deaths in sequence — each recovers on a
        fresh stream until MAX_ROLLOVERS, with every card still consistent."""
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            for i in range(1, 6):
                if i > 1:
                    client.fail_appends_with = "message_not_in_streaming_state"
                await stream.task_started(i, "terminal", preview=f"wave {i}")
                await stream.task_finished(i, "terminal", duration=0.1)

        _run(scenario())
        assert stream.disabled is False
        assert client.stream_opens() == 5  # 1 original + 4 reactive rollovers
        chunks = [c for c in client.appended_chunks() if c.get("type") == "task_update"]
        for i in range(1, 6):
            statuses = [c["status"] for c in chunks if c.get("id") == f"t{i}"]
            assert statuses and statuses[-1] == "complete"
