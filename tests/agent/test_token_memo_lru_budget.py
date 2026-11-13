"""The per-message token-estimate memo sheds least-recently-used idle entries only: a running conversation keeps
its hits however many dead sessions or large bodies stream past it, even when it alone is larger than the
budget, and idle bodies stop counting forever."""

import hermes_agent.agent.model_metadata as mm
import hermes_agent.agent.model_metadata_token_memo as memo


def _history(tag: str, n: int, size: int = 200) -> list[dict]:
    return [{"role": "user" if i % 2 else "assistant", "content": f"{tag}-{i}-" + "x" * size} for i in range(n)]


def _count_computes(monkeypatch) -> list[int]:
    computed = [0]
    real = mm._estimate_message_tokens_without_images

    def counting(msg):
        computed[0] += 1
        return real(msg)

    monkeypatch.setattr(mm, "_estimate_message_tokens_without_images", counting)
    return computed


def test_two_large_running_conversations_both_keep_their_memo_hits(monkeypatch):
    mm._MSG_TOKENS_CACHE.clear()
    computed = _count_computes(monkeypatch)
    first, second = _history("a", 2500), _history("b", 2500)  # each the size of a real 1M-token history
    expected = {id(first): mm.estimate_messages_tokens_rough(first), id(second): mm.estimate_messages_tokens_rough(second)}
    computed[0] = 0
    for _ in range(3):  # interleaved agent-loop iterations, fresh shallow copies like api_messages
        for history in (first, second):
            assert mm.estimate_messages_tokens_rough([dict(m) for m in history]) == expected[id(history)]
    assert computed[0] == 0


def _fake_clock(monkeypatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(memo, "_clock", lambda: now[0], raising=False)
    return now


def test_dead_bodies_past_the_byte_budget_go_first_and_estimates_never_change(monkeypatch):
    mm._MSG_TOKENS_CACHE.clear()
    monkeypatch.setattr(memo, "_MSG_TOKENS_CACHE_MAX_BYTES", 3_500_000, raising=False)
    now = _fake_clock(monkeypatch)
    active = {"role": "user", "content": "active " + "y" * 100_000}
    active_tokens = mm.estimate_messages_tokens_rough([active])
    computed = _count_computes(monkeypatch)
    dead_bodies = []
    for i in range(10):
        now[0] += 3600  # each dead body goes idle before the next arrives
        dead_bodies.append(f"dead {i} " + "z" * 1_000_000)
        mm.estimate_messages_tokens_rough([{"role": "tool", "content": dead_bodies[-1]}])
        computed[0] -= 1  # the dead body itself is a miss
        assert mm.estimate_messages_tokens_rough([dict(active)]) == active_tokens  # the running turn touches it
    assert computed[0] == 0  # every active lookup was a hit, never an evict-and-recompute

    # One message bigger than the whole budget is not memoized and does not flush what fits (all idle now).
    now[0] += 3600
    mm.estimate_messages_tokens_rough([{"role": "tool", "content": "huge " + "w" * 4_000_000}])
    pinned = {id(pin) for pins, _text, _images in mm._MSG_TOKENS_CACHE.values() for pin in pins}
    assert id(active["content"]) in pinned
    assert sum(id(body) in pinned for body in dead_bodies) <= 3
    assert id(dead_bodies[-1]) in pinned


def test_a_running_conversation_larger_than_both_bounds_never_misses_then_sheds_when_idle(monkeypatch):
    mm._MSG_TOKENS_CACHE.clear()
    monkeypatch.setattr(memo, "_MSG_TOKENS_CACHE_MAX_BYTES", 1_000_000, raising=False)
    monkeypatch.setattr(memo, "_MSG_TOKENS_CACHE_MAX", 1000, raising=False)
    now = _fake_clock(monkeypatch)
    computed = _count_computes(monkeypatch)
    history = _history("big", 5000, size=1000)  # ~6 MB retained and 5000 entries: over both bounds alone
    expected = mm.estimate_messages_tokens_rough(history)
    computed[0] = 0
    for _ in range(3):  # agent-loop iterations seconds apart, each appending a message
        now[0] += 5
        history.append({"role": "user", "content": f"turn {len(history)} " + "q" * 1000})
        expected = mm.estimate_messages_tokens_rough([dict(m) for m in history])
    assert computed[0] == 3  # only each pass's new message is ever computed
    assert mm.estimate_messages_tokens_rough([dict(m) for m in history]) == expected

    now[0] += 3600  # the conversation goes idle; the next insert sheds it back under the bounds
    mm.estimate_messages_tokens_rough([{"role": "user", "content": "next " + "w" * 1000}])
    assert len(mm._MSG_TOKENS_CACHE) <= 1000 and memo.retained_bytes() <= 1_000_000


def test_a_hit_racing_eviction_never_makes_it_drop_a_different_active_entry(monkeypatch):
    """A writer checks the oldest entry is idle, then pops the oldest. A concurrent hit on that entry in
    between must not reorder the memo under the writer, or the pop takes the next, recently used entry."""
    import threading
    from collections import OrderedDict

    reader_done = threading.Event()

    class HitDuringPop(OrderedDict):
        armed = False

        def popitem(self, last=True):
            if self.armed:  # the writer has checked the oldest entry is idle and is about to pop it
                self.armed = False
                threading.Thread(target=lambda: (memo.lookup_message_tokens("idle"), reader_done.set()),
                                 daemon=True).start()
                reader_done.wait(1.0)
            return super().popitem(last=last)

    cache = HitDuringPop()
    monkeypatch.setattr(memo, "_MSG_TOKENS_CACHE", cache)
    monkeypatch.setattr(memo, "_MSG_TOKENS_CACHE_MAX", 2)
    now = _fake_clock(monkeypatch)
    memo.remember_message_tokens("idle", (memo.MemoPins(["idle text"]), 1, 0))
    now[0] += 3600
    memo.remember_message_tokens("active", (memo.MemoPins(["active text"]), 1, 0))
    cache.armed = True
    memo.remember_message_tokens("new", (memo.MemoPins(["new text"]), 1, 0))
    assert reader_done.wait(5)
    assert "active" in cache and "new" in cache


def test_idle_entries_never_outnumber_the_memo_main_always_had(monkeypatch):
    """Closed sessions' entries (idle) stay within the original 4,096-entry footprint; only entries a running
    conversation touched in the last minute may exceed it."""
    mm._MSG_TOKENS_CACHE.clear()
    now = _fake_clock(monkeypatch)
    for i in range(6000):
        now[0] += 61  # each message's session is idle before the next one arrives
        mm.estimate_messages_tokens_rough([{"role": "tool", "content": f"closed-{i} " + "q" * 300}])
    assert len(mm._MSG_TOKENS_CACHE) <= 4096
