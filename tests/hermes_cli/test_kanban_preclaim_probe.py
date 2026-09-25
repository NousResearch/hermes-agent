"""Pre-claim model/credential probe for pinned cards (#122703).

A card that pins ``model_override``/``provider_override`` dies at worker
startup when the pin is unreachable — unknown provider, quota-walled 403, no
credential — but as a clean ``rc=0`` exit AFTER the dispatcher already spent
the claim, indistinguishable from success at the dispatch layer (two burns:
t_85821c09, t_4dc09492 — respawn_guarded loops burned 3 runs each). The probe
resolves the pin the way the worker would, BEFORE the claim is consumed; a
failed probe releases the claim through the spawn-failure path.

The dead-provider shape is hermetic: ``resolve_runtime_provider`` raises
``AuthError`` locally for unknown provider names (verified against the real
CLI: ``hermes -z ... --provider nonexistent-provider-xyz`` exits rc=1 in ~0.4s
without touching the network), so these tests never need a live endpoint.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

DEAD_PROVIDER = "nonexistent-provider-xyz"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and no ambient probe knobs."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "alice").mkdir(parents=True)
    # Identity marker so resolve_profile_env()/profile_exists treat alice as live.
    (home / "profiles" / "alice" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_PRECLAIM_PROBE", raising=False)
    monkeypatch.delenv("HERMES_BIN", raising=False)
    cache = getattr(kbd, "_preclaim_probe_cache", None)
    if cache is not None:
        cache.clear()
    kb.init_db()
    return home


def _spawn_recorder():
    calls: list[str] = []
    return calls, lambda task, _workspace, board=None: (calls.append(task.id), 4242)[1]


def _pinned_card(conn, **overrides):
    kwargs: dict = dict(
        title="pinned to a dead provider",
        assignee="alice",
        model_override="test-model",
        provider_override=DEAD_PROVIDER,
    )
    kwargs.update(overrides)
    tid = kb.create_task(conn, **kwargs)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    conn.commit()
    return tid


def test_dead_provider_pin_never_claims(kanban_home):
    """The core invariant (#122703): a card pinned to an unreachable provider
    must not have its claim consumed. Verdict: card back in ready with the
    claim released, no worker spawned, and the event chain carries an
    observable reason — instead of a clean rc=0 worker exit after the claim
    was spent."""
    with kbc.connect() as conn:
        tid = _pinned_card(conn)
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        task = kb.get_task(conn, tid)
        assert task is not None
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        failure = task.last_failure_error or ""
    # Invariants first so the RED on base is the behavioral claim itself
    # (worker spawned / claim spent), not a missing-attribute error.
    assert spawns == [], "worker must not be spawned on a dead pin"
    assert task.status == "ready", "claim must be released back to the source phase"
    assert task.claim_lock is None
    assert res.spawned == []
    # Feature surface (exists only with the patch):
    assert res.preclaim_probe_failed == [tid]
    assert "preclaim probe" in failure
    assert "preclaim_probe_failed" in kinds
    assert "spawn_failed" in kinds


def test_probe_failure_counts_toward_circuit_breaker(kanban_home):
    """The release rides the existing spawn-failure accounting: one probe
    failure counts one failure; the breaker (default limit 2) parks the card
    on the second — no new state machine, and a permanently dead pin cannot
    retry-storm."""
    with kbc.connect() as conn:
        tid = _pinned_card(conn)
        spawns, spawn_fn = _spawn_recorder()
        kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.consecutive_failures == 1
        res2 = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        task = kb.get_task(conn, tid)
    assert spawns == []
    assert res2.preclaim_probe_failed == [tid]
    assert task.status == "blocked", "second dead-pin failure must trip the breaker"


def test_probe_verdict_cached_across_cards(kanban_home, monkeypatch):
    """A fan-out of same-pinned cards probes ONCE: the second dispatch reads
    the cached verdict instead of re-probing per claim."""
    calls = {"n": 0}

    def counting_resolve(*args, **kwargs):
        calls["n"] += 1
        from hermes_cli.auth import AuthError

        raise AuthError(f"Unknown provider '{DEAD_PROVIDER}'.")

    import hermes_cli.runtime_provider as rp

    monkeypatch.setattr(rp, "resolve_runtime_provider", counting_resolve)
    with kbc.connect() as conn:
        t1 = _pinned_card(conn, title="pin 1")
        t2 = _pinned_card(conn, title="pin 2")
        spawns, spawn_fn = _spawn_recorder()
        res1 = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        res2 = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert calls["n"] == 1, "static resolve must run once, then hit the cache"
    assert res1.preclaim_probe_failed == [t1, t2], "same tick: t2 hits the verdict t1 just cached"
    assert res2.preclaim_probe_failed == [t1, t2], "next tick: both reuse the cached verdict, zero re-probes"
    assert spawns == []


def test_disabled_probe_keeps_upstream_behavior(kanban_home, monkeypatch):
    """HERMES_KANBAN_PRECLAIM_PROBE=0 reverts to upstream behavior: the dead
    pin card is claimed and spawned exactly as on main — the probe is
    skippable, never a mandatory network tax on the open path."""
    monkeypatch.setenv("HERMES_KANBAN_PRECLAIM_PROBE", "0")
    with kbc.connect() as conn:
        tid = _pinned_card(conn)
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert res.preclaim_probe_failed == []
    assert spawns == [tid], "disabled probe must not block the claim"
    assert res.spawned and res.spawned[0][0] == tid


def test_unpinned_card_never_probed(kanban_home, monkeypatch):
    """Only pinned cards pay the probe: an unpinned card claims and spawns
    with the probe machinery completely absent (even a broken resolver must
    not matter)."""
    import hermes_cli.runtime_provider as rp

    def explode(*args, **kwargs):
        raise AssertionError("unpinned card must not reach the resolver")

    monkeypatch.setattr(rp, "resolve_runtime_provider", explode)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no pin", assignee="alice")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert res.preclaim_probe_failed == []
    assert spawns == [tid]


def _write_fake_hermes(dir_: Path, body: str) -> str:
    bin_path = dir_ / "fake-hermes"
    bin_path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)
    return str(bin_path)


@pytest.fixture
def patched_static_resolve(monkeypatch):
    """Layer 1 passes (well-formed pin); layer 2 (live child) is under test."""
    import hermes_cli.runtime_provider as rp

    monkeypatch.setattr(
        rp, "resolve_runtime_provider",
        lambda *a, **k: {"provider": "openrouter", "api_key": "sk-test", "api_mode": "chat"},
    )


def test_live_probe_dead_child_releases_claim(kanban_home, monkeypatch, tmp_path, patched_static_resolve):
    """Layer 2 (real one-shot child, sealed via HERMES_BIN): a well-formed pin
    whose child exits non-zero (the 403/quota-wall shape) is caught before the
    claim is spent."""
    monkeypatch.setenv(
        "HERMES_BIN", _write_fake_hermes(tmp_path, 'echo "hermes -z: agent failed: 403 quota exceeded" >&2\nexit 1'),
    )
    with kbc.connect() as conn:
        tid = _pinned_card(conn)
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        task = kb.get_task(conn, tid)
    assert spawns == []
    assert res.preclaim_probe_failed == [tid]
    assert task.status == "ready"
    assert "quota" in (task.last_failure_error or "")


def test_live_probe_timeout_fails_open(kanban_home, monkeypatch, tmp_path, patched_static_resolve):
    """A probe that hangs is inconclusive, not fatal: fail-open, card spawns.
    The probe must never become a new way to lose a healthy claim."""
    monkeypatch.setenv("HERMES_KANBAN_PRECLAIM_PROBE", "1")  # 1s timeout
    monkeypatch.setenv("HERMES_BIN", _write_fake_hermes(tmp_path, "sleep 30"))
    with kbc.connect() as conn:
        tid = _pinned_card(conn)
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert res.preclaim_probe_failed == []
    assert spawns == [tid], "timed-out probe must fail open"


def test_live_probe_healthy_child_claims(kanban_home, monkeypatch, tmp_path, patched_static_resolve):
    """A healthy one-shot child proves the pin: the claim proceeds to spawn."""
    monkeypatch.setenv("HERMES_BIN", _write_fake_hermes(tmp_path, "echo PROBE_OK"))
    with kbc.connect() as conn:
        tid = _pinned_card(conn)
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert res.preclaim_probe_failed == []
    assert spawns == [tid]


def test_review_lane_pinned_card_also_gated(kanban_home):
    """Review-lane spawns share the same _dispatch_lane_task chokepoint: a
    reviewer card pinned to a dead provider is released the same way (its run
    restores source_status=review)."""
    with kbc.connect() as conn:
        tid = _pinned_card(conn, title="review pin", model_override="m", provider_override=DEAD_PROVIDER)
        # Move the card through the review handoff the way an implementer does.
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        conn.commit()
        spawns, spawn_fn = _spawn_recorder()
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        task = kb.get_task(conn, tid)
    assert res.preclaim_probe_failed == [tid]
    assert spawns == []
    assert task.status == "review", "review lane restores its own source phase"
