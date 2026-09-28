# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""A credential the user deliberately activates is adopted by an ALREADY-OPEN session.

Reordering the pool (``hermes auth priority``, the account-switch chip) only steers sessions that
resolve their credential afterwards. A live chat kept billing its init-time account until a 429/402
rotated it off, so "switch account" silently meant "switch account, eventually". ``move_entry`` and
the reset paths stamp ``activated_at``; ``adopt_activated_credential`` reads it at the turn boundary
(before the turn's first API call, so nothing in flight is disturbed).

The negative cases matter as much: without an activation nothing rebinds, because rotating accounts
mid-conversation throws away the provider-side prompt cache.
"""

import time

from agent.agent_runtime_helpers import adopt_activated_credential
from agent.credential_pool import CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"


def _entry(entry_id, label, *, priority, token, base_url=_BASE):
    return PooledCredential.from_dict("anthropic", {
        "id": entry_id, "label": label, "auth_type": "api_key", "priority": priority,
        "access_token": token, "base_url": base_url, "source": "manual",
    })


def _pool(*entries):
    return CredentialPool(provider="anthropic", entries=list(entries))


class _LiveAgent:
    """Open chat stand-in: real pool + real adoption hook, no client build."""

    provider = "anthropic"
    model = "claude-opus-5"
    base_url = _BASE
    swap_refuses = False
    _credential_pool_revert_id: "str | None" = None

    def __init__(self, pool):
        self._credential_pool = pool
        first = pool.select()
        self._credential_pool_entry_id = first.id
        self.api_key = first.runtime_api_key
        self.statuses = []

    def _swap_credential(self, entry):
        if self.swap_refuses:
            return False
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))


def _seed_pool(monkeypatch, pool):
    """Route ``load_pool`` (and the runtime key resolution it hangs off) at *pool*."""
    import agent.agent_runtime_helpers as arh
    import agent.credential_pool as cp

    monkeypatch.setattr(cp, "load_pool", lambda _key: pool)
    monkeypatch.setattr(arh, "resolve_runtime_pool_key", lambda _p, base_url=None: "anthropic")


def test_promote_moves_an_open_chat_at_the_next_turn(monkeypatch):
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert agent._credential_pool_entry_id == "acct0001"

    # First look only establishes the baseline: pre-existing stamps are history, not a choice
    # made during this conversation.
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"

    pool.move_entry("acct0002", 0)  # the user promotes the second account in the chip
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"
    assert agent.api_key == "«redacted:sk-…»-two"
    assert agent.statuses and "conta-2" in agent.statuses[0]

    # Idempotent: the same activation must not re-swap (and re-pay the prompt cache) every turn.
    agent.statuses.clear()
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0002" and not agent.statuses


def test_no_activation_keeps_the_session_and_its_prompt_cache(monkeypatch):
    """The hook is activation-driven, never a per-turn re-select: silence means stay put."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    for _ in range(3):
        assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"


def test_bulk_reset_does_not_yank_the_session_onto_a_benched_account(monkeypatch):
    """``reset_statuses`` lifts cooldowns; it is not an account choice, so nothing rebinds."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    assert pool.reset_statuses() >= 1
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"


def test_targeted_reset_is_an_account_choice_and_rebinds(monkeypatch):
    """``reset_status(id)`` names one credential: that IS "use this one", so the session moves."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    assert pool.reset_status("acct0002") is not None
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_activation_on_a_foreign_endpoint_is_not_adopted(monkeypatch):
    """A same-provider entry pointing elsewhere (proxy/gateway) must not move this session's route."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("proxy001", "gateway", priority=1, token="«redacted:sk-…»-proxy",
               base_url="https://llm-proxy.internal"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    pool.move_entry("proxy001", 0)
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"


def test_activation_of_a_benched_account_waits_for_its_window(monkeypatch):
    """Promoting a rate-limited account must not hand the next turn a guaranteed 429.

    The baseline is NOT advanced in that case: once the cooldown lifts, a later turn adopts the
    account the user already chose, without asking them to click promote again.
    """
    import agent.credential_pool as cp

    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    pool.move_entry("acct0002", 0)  # promoted while still cooling down
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"

    real_time = time.time
    monkeypatch.setattr(
        cp.time, "time", lambda: real_time() + cp.EXHAUSTED_TTL_429_SECONDS + 120,
    )
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_a_refused_swap_leaves_the_session_exactly_as_it_was(monkeypatch):
    """``_swap_credential`` refuses when the entry's route cannot serve the conversation's model."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    original_pool = agent._credential_pool
    agent.swap_refuses = True
    pool.move_entry("acct0002", 0)
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"
    assert agent._credential_pool is original_pool


def test_activation_clears_a_pending_automatic_revert(monkeypatch):
    """An explicit account choice outranks the queued revert to a quota-benched credential."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    agent._credential_pool_revert_id = "acct0009"
    pool.move_entry("acct0002", 0)
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_revert_id is None


def test_activation_stamp_survives_a_pool_round_trip():
    """``activated_at`` rides in ``extra``, so it must survive to_dict -> from_dict."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    before = time.time()
    promoted = pool.move_entry("acct0002", 0)
    assert promoted is not None
    assert isinstance(promoted.activated_at, float) and promoted.activated_at >= before
    revived = PooledCredential.from_dict("anthropic", promoted.to_dict())
    assert revived.activated_at == promoted.activated_at


def _count_pool_reads(monkeypatch, pool):
    """Route ``load_pool`` at *pool* and count how often the turn boundary pays for it."""
    import agent.agent_runtime_helpers as arh
    import agent.credential_pool as cp

    reads = []
    monkeypatch.setattr(cp, "load_pool", lambda _key: (reads.append(1), pool)[1])
    monkeypatch.setattr(arh, "resolve_runtime_pool_key", lambda _p, base_url=None: "anthropic")
    return reads


def _stub_store(monkeypatch, fingerprint):
    """Pin the credential store's (mtime, size) so the test drives the short-circuit."""
    import agent.agent_runtime_helpers as arh

    box = {"value": fingerprint}
    monkeypatch.setattr(arh, "_auth_store_fingerprint", lambda: box["value"])
    return box


def test_an_untouched_store_is_not_reloaded_every_turn(monkeypatch):
    """An activation always writes auth.json, so an unchanged store means nothing to adopt.

    ``load_pool`` is expensive (on macOS it shells out to ``security`` for the Claude Code
    keychain entry), and this hook runs at every turn of every live session.
    """
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)

    assert adopt_activated_credential(agent) is False  # baseline: reads once, arms the fingerprint
    assert len(reads) == 1

    for _ in range(5):
        assert adopt_activated_credential(agent) is False
    assert len(reads) == 1, "an unchanged store must not be re-read"


def test_a_written_store_is_read_again_and_adopted(monkeypatch):
    """The short-circuit must not swallow a real activation: a new (mtime, size) reopens the path."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    store = _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    pool.move_entry("acct0002", 0)
    store["value"] = (333, 444)  # the promote wrote auth.json

    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"
    assert len(reads) == 2


def test_a_cooling_down_activation_keeps_being_retried(monkeypatch):
    """A cooldown expires with the clock, not with a write, so the fingerprint must NOT be armed.

    Arming it there would strand the activation until something else touched auth.json — the
    user would have to promote a second time to get the account they already chose.
    """
    import agent.credential_pool as cp

    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    store = _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    pool.move_entry("acct0002", 0)
    store["value"] = (333, 444)  # the promote wrote auth.json

    assert adopt_activated_credential(agent) is False  # benched: stays put
    assert adopt_activated_credential(agent) is False  # and keeps looking, store unchanged
    assert len(reads) == 3, "a pending activation must survive until its window reopens"

    real_time = time.time
    monkeypatch.setattr(
        cp.time, "time", lambda: real_time() + cp.EXHAUSTED_TTL_429_SECONDS + 120,
    )
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_a_store_that_cannot_be_stated_falls_back_to_reading(monkeypatch):
    """No fingerprint (missing or unreadable store) must degrade to the old always-read path."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    _stub_store(monkeypatch, None)
    agent = _LiveAgent(pool)

    assert adopt_activated_credential(agent) is False  # baseline
    pool.move_entry("acct0002", 0)
    assert adopt_activated_credential(agent) is True
    assert len(reads) == 2


def test_equal_stamps_break_the_tie_by_priority(monkeypatch):
    """Two entries stamped in the same tick must resolve deterministically, not by dict order."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
        _entry("acct0003", "conta-3", priority=2, token="«redacted:sk-…»-three"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    stamp = time.time() + 1
    for entry_id in ("acct0003", "acct0002"):
        entry = next(e for e in pool.entries() if e.id == entry_id)
        pool._adopt(entry, persist=False, extra={**(entry.extra or {}), "activated_at": stamp})

    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002", "lowest priority wins an exact tie"
