"""The per-message token-estimate memo sheds least-recently-used entries only: a running conversation keeps
its hits however many dead sessions or large bodies stream past it, and dead bodies stop counting forever."""

import agent.model_metadata as mm
import agent.model_metadata_token_memo as memo


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


def test_dead_bodies_past_the_byte_budget_go_first_and_estimates_never_change(monkeypatch):
    mm._MSG_TOKENS_CACHE.clear()
    monkeypatch.setattr(memo, "_MSG_TOKENS_CACHE_MAX_BYTES", 3_500_000, raising=False)
    active = {"role": "user", "content": "active " + "y" * 100_000}
    active_tokens = mm.estimate_messages_tokens_rough([active])
    dead_bodies = []
    for i in range(10):
        dead_bodies.append(f"dead {i} " + "z" * 1_000_000)
        mm.estimate_messages_tokens_rough([{"role": "tool", "content": dead_bodies[-1]}])
        assert mm.estimate_messages_tokens_rough([dict(active)]) == active_tokens  # the running turn touches it

    pinned = {id(pin) for pins, _text, _images in mm._MSG_TOKENS_CACHE.values() for pin in pins}
    assert id(active["content"]) in pinned
    assert sum(id(body) in pinned for body in dead_bodies) <= 3
    assert id(dead_bodies[-1]) in pinned
