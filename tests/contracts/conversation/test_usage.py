"""Contract cases for ``persistence.conversation.usage.UsageRepository``.

Runs against every registered backend from the shared fixture. Sessions are minted through the sessions capability (``store.sessions.create``), never through the seeding seam, because the lineage walk and the model and ``model_config`` promises are about real session rows. The session row's accounting columns are read through ``SessionRepository.get``. Each case asserts one outcome: queued deltas apply in queue order (the last queued route is the recent route), deltas flushed once equal the same deltas flushed one by one, an empty flush writes nothing, two callers queueing into one session sum, an apply failure is logged and dropped; ``set_billing_route`` replaces provider and base URL, keeps the mode on ``None``, leaves the model, ``model_config`` and stored prompt alone, is not undone by later deltas, and ignores an unknown id; ``auxiliary_usage_by_task`` sums over the lineage chain cut at a branch, excludes main-loop rows, keys by task, and never crosses scopes; a per-turn route ``task`` counts in the session but never becomes the recent route.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence.conversation.usage import (
    AuxiliaryTaskUsage,
    BillingRoute,
    TokenCounts,
    UsageDelta,
)

from tests.contracts._scopes import MEDIA_ROOT

ROUTE_A = BillingRoute(model="model-a", billing_provider="prov-a", billing_base_url="https://a.example", billing_mode="plan")
ROUTE_B = BillingRoute(model="model-b", billing_provider="prov-b", billing_base_url="https://b.example", billing_mode="included")


@dataclass(frozen=True)
class Accounting:
    """The accounting columns of one session row."""

    tokens: TokenCounts
    api_call_count: int
    estimated_cost_usd: Optional[float]
    actual_cost_usd: Optional[float]
    cost_status: Optional[str]
    cost_source: Optional[str]
    pricing_version: Optional[str]


def accounting(store: Any, session_id: str) -> Optional[Accounting]:
    """The session row's counters and cost fields through ``SessionRepository.get``; ``None`` for an unknown session."""
    record = store.sessions.get(SessionId(session_id))
    if record is None:
        return None
    return Accounting(
        tokens=TokenCounts(
            input_tokens=record.input_tokens, output_tokens=record.output_tokens, cache_read_tokens=record.cache_read_tokens,
            cache_write_tokens=record.cache_write_tokens, reasoning_tokens=record.reasoning_tokens,
        ),
        api_call_count=record.api_call_count,
        estimated_cost_usd=record.estimated_cost_usd,
        actual_cost_usd=record.actual_cost_usd,
        cost_status=record.cost_status,
        cost_source=record.cost_source,
        pricing_version=record.pricing_version,
    )


def billing_route(store: Any, session_id: str) -> Optional[tuple[Optional[str], Optional[str], Optional[str]]]:
    """The session row's stored (provider, base URL, mode) through ``SessionRepository.get``; ``None`` for an unknown session."""
    record = store.sessions.get(SessionId(session_id))
    return None if record is None else (record.billing_provider, record.billing_base_url, record.billing_mode)


def delta(route: BillingRoute, *, inp: int = 10, out: int = 5, calls: int = 1, cost: float | None = 0.01, **fields: Any) -> UsageDelta:
    return UsageDelta(
        tokens=TokenCounts(input_tokens=inp, output_tokens=out),
        api_call_count=calls,
        estimated_cost_usd=cost,
        route=route,
        **fields,
    )


def _new(session_id: str, **kwargs: Any) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source="cli", **kwargs)


def race_on_threads(attempts: list[Callable[[], Any]]) -> list[Any]:
    """Run each call on its own OS thread, released together."""
    barrier = threading.Barrier(len(attempts))

    def run(make: Callable[[], Any]) -> Any:
        barrier.wait(timeout=30.0)
        return make()

    with ThreadPoolExecutor(max_workers=len(attempts)) as pool:
        return list(pool.map(run, attempts))


# ── queue_usage / flush_usage ─────────────────────────────────────────────────


def test_deltas_flushed_once_equal_the_same_deltas_flushed_one_by_one(backend, scope_a):
    """CONTRACT: N queued deltas (route switches, differing cost status, an all-None cost run) flushed once leave the counters and the recent route exactly as the same N each followed by a flush; a merge of adjacent deltas is unobservable."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("uq-batched"))
    store_a.sessions.create(_new("uq-sequential"))
    deltas = [
        delta(ROUTE_A, inp=10, out=1, cost=0.10, cost_status="estimated", cost_source="table"),
        delta(ROUTE_A, inp=20, out=2, cost=None, cost_status="estimated", cost_source="table"),
        delta(ROUTE_A, inp=30, out=3, cost=0.30, cost_status="final", cost_source="api", pricing_version="pv-1"),
        delta(ROUTE_B, inp=40, out=4, cost=None, cost_status="final", cost_source="api"),
        delta(ROUTE_B, inp=50, out=5, cost=None, cost_status="final", cost_source="api"),
        delta(ROUTE_A, inp=60, out=6, calls=2, cost=0.60, actual_cost_usd=0.55, cost_status="final", cost_source="api"),
    ]
    for d in deltas:
        usage.queue_usage(SessionId("uq-batched"), d)
    assert usage.flush_usage() is True
    for d in deltas:
        usage.queue_usage(SessionId("uq-sequential"), d)
        assert usage.flush_usage() is True
    batched = accounting(store_a, "uq-batched")
    assert batched == accounting(store_a, "uq-sequential")
    assert batched.tokens == TokenCounts(input_tokens=210, output_tokens=21)
    assert batched.api_call_count == 7
    assert batched.estimated_cost_usd == pytest.approx(1.0)
    assert batched.actual_cost_usd == pytest.approx(0.55)
    assert (batched.cost_status, batched.cost_source, batched.pricing_version) == ("final", "api", "pv-1")
    recent_batched = usage.recent_route(SessionId("uq-batched"))
    assert recent_batched == usage.recent_route(SessionId("uq-sequential"))
    # Route A's attribution row carries the exact call count of every A delta (1+1+1+2).
    assert recent_batched.route == ROUTE_A and recent_batched.api_call_count == 5


def test_queued_deltas_apply_in_queue_order_so_the_last_queued_route_is_recent(backend, scope_a):
    """CONTRACT: order is queue order, not route or size order; the same two deltas queued the other way round flip the recent route."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("uq-order-ab"))
    store_a.sessions.create(_new("uq-order-ba"))
    usage.queue_usage(SessionId("uq-order-ab"), delta(ROUTE_A, inp=1000))
    usage.queue_usage(SessionId("uq-order-ab"), delta(ROUTE_B, inp=1))
    usage.queue_usage(SessionId("uq-order-ba"), delta(ROUTE_B, inp=1))
    usage.queue_usage(SessionId("uq-order-ba"), delta(ROUTE_A, inp=1000))
    assert usage.flush_usage() is True
    assert (usage.recent_route(SessionId("uq-order-ab"))).route == ROUTE_B
    assert (usage.recent_route(SessionId("uq-order-ba"))).route == ROUTE_A


def test_one_queue_spans_sessions_and_keeps_each_in_order(backend, scope_a):
    """CONTRACT: the store's queue carries deltas for several sessions; each session sees only its own, in order."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("uq-multi-1"))
    store_a.sessions.create(_new("uq-multi-2"))
    usage.queue_usage(SessionId("uq-multi-1"), delta(ROUTE_A, inp=1))
    usage.queue_usage(SessionId("uq-multi-2"), delta(ROUTE_B, inp=2))
    usage.queue_usage(SessionId("uq-multi-1"), delta(ROUTE_B, inp=4))
    usage.queue_usage(SessionId("uq-multi-2"), delta(ROUTE_A, inp=8))
    assert usage.flush_usage() is True
    one = accounting(store_a, "uq-multi-1")
    two = accounting(store_a, "uq-multi-2")
    assert (one.tokens.input_tokens, one.api_call_count) == (5, 2)
    assert (two.tokens.input_tokens, two.api_call_count) == (10, 2)
    assert (usage.recent_route(SessionId("uq-multi-1"))).route == ROUTE_B
    assert (usage.recent_route(SessionId("uq-multi-2"))).route == ROUTE_A


def test_empty_flush_returns_true_and_a_fresh_session_stays_zero(backend, scope_a):
    """CONTRACT: a flush with nothing queued returns ``True`` at once and writes nothing."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("uq-empty"))
    assert store_a.usage.flush_usage() is True
    assert store_a.usage.flush_usage(timeout=0.0) is True
    read = accounting(store_a, "uq-empty")
    assert read.tokens == TokenCounts() and read.api_call_count == 0
    assert store_a.usage.recent_route(SessionId("uq-empty")) is None


def test_two_callers_queueing_into_one_session_sum(backend, scope_a):
    """CONTRACT: two callers on two OS threads queue into one session and flush; nothing is lost: the counters are the exact sum and each attribution row the exact call count."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("uq-two-callers"))
    rounds = 25

    def caller(lane: int) -> bool:
        for i in range(rounds):
            # Alternate routes so nothing merges and every delta is its own write.
            usage.queue_usage(SessionId("uq-two-callers"), delta(ROUTE_A if i % 2 == lane else ROUTE_B, inp=1, out=1, cost=0.01))
        return usage.flush_usage()

    assert race_on_threads([lambda lane=lane: caller(lane) for lane in range(2)]) == [True, True]
    read = accounting(store_a, "uq-two-callers")
    assert read.tokens == TokenCounts(input_tokens=2 * rounds, output_tokens=2 * rounds)
    assert read.api_call_count == 2 * rounds
    assert read.estimated_cost_usd == pytest.approx(0.01 * 2 * rounds)
    # Whichever route was recorded last, its row holds exactly its own calls: rounds in total per route.
    assert usage.recent_route(SessionId("uq-two-callers")).api_call_count == rounds


def test_a_delta_that_fails_to_apply_is_logged_and_dropped(backend, scope_a, caplog):
    """CONTRACT: the writer logs an apply failure at WARNING and drops the delta, never raising into the caller; the deleted session is not recreated beside its tombstone, and later deltas still apply."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("uq-deleted"))
    store_a.sessions.create(_new("uq-alive"))
    store_a.sessions.delete(SessionId("uq-deleted"), media_root=MEDIA_ROOT)
    with caplog.at_level(logging.WARNING):
        store_a.usage.queue_usage(SessionId("uq-deleted"), delta(ROUTE_A, inp=2))
        store_a.usage.queue_usage(SessionId("uq-alive"), delta(ROUTE_A, inp=3))
        assert store_a.usage.flush_usage() is True
    assert any("uq-deleted" in record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)
    assert store_a.sessions.get(SessionId("uq-deleted")) is None
    assert accounting(store_a, "uq-alive").tokens.input_tokens == 3


def test_queued_delta_backfills_a_missing_row_with_its_source(backend, scope_a):
    """CONTRACT: a delta for a session with no row creates the row with the delta's ``source`` before accounting; without a source the row is the ``unknown`` placeholder, which a later real source replaces."""
    store_a = backend.open(scope_a)
    store_a.usage.queue_usage(SessionId("uq-ghost-src"), delta(ROUTE_A, source="telegram"))
    store_a.usage.queue_usage(SessionId("uq-ghost"), delta(ROUTE_A))
    assert store_a.usage.flush_usage() is True
    assert store_a.sessions.get(SessionId("uq-ghost-src")).source == "telegram"
    assert store_a.sessions.get(SessionId("uq-ghost")).source == "unknown"
    store_a.usage.queue_usage(SessionId("uq-ghost"), delta(ROUTE_A, source="cli"))
    assert store_a.usage.flush_usage() is True
    ghost = store_a.sessions.get(SessionId("uq-ghost"))
    assert ghost.source == "cli" and ghost.api_call_count == 2


def test_turn_route_task_counts_in_the_session_but_never_becomes_the_recent_route(backend, scope_a):
    """CONTRACT: a delta carrying a per-turn route ``task`` adds to the session counters, lands in its own attribution row summed by ``auxiliary_usage_by_task``, and never becomes the recent route."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("uq-task"))
    store_a.usage.queue_usage(SessionId("uq-task"), delta(ROUTE_B, inp=7, calls=2, task="voice_chat"))
    store_a.usage.queue_usage(SessionId("uq-task"), delta(ROUTE_A, inp=1))
    assert store_a.usage.flush_usage() is True
    assert accounting(store_a, "uq-task").api_call_count == 3
    assert store_a.usage.recent_route(SessionId("uq-task")).route == ROUTE_A
    # The task delta is not the first accounted route, but it still fills the unset route columns, and its calls end the first-accounted window.
    assert billing_route(store_a, "uq-task") == ("prov-b", "https://b.example", "included")
    voice = store_a.usage.auxiliary_usage_by_task(SessionId("uq-task"))["voice_chat"]
    assert (voice.api_call_count, voice.tokens.input_tokens) == (2, 7)


# ── set_billing_route ─────────────────────────────────────────────────────────


def test_set_billing_route_replaces_provider_and_base_url_unconditionally(backend, scope_a):
    """CONTRACT: the stored route is replaced even though the delta path filled it; the model, ``model_config`` and the stored prompt stay as created."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    config = {"provider": "prov-a", "yolo_mode": True}
    store_a.sessions.create(_new("ubr-switch", model="model-a", model_config=config, system_prompt="stored prompt"))
    created = store_a.sessions.get(SessionId("ubr-switch"))
    usage.queue_usage(SessionId("ubr-switch"), delta(ROUTE_A))
    usage.set_billing_route(SessionId("ubr-switch"), provider="prov-b", base_url="https://b.example", billing_mode="subscription_included")
    assert billing_route(store_a, "ubr-switch") == ("prov-b", "https://b.example", "subscription_included")
    after = store_a.sessions.get(SessionId("ubr-switch"))
    assert after.model == created.model == "model-a"
    assert dict(after.model_config) == dict(created.model_config)
    assert after.system_prompt == created.system_prompt == "stored prompt"
    # The main-loop attribution and the counters are untouched by a route switch.
    assert (usage.recent_route(SessionId("ubr-switch"))).route == ROUTE_A
    assert accounting(store_a, "ubr-switch").api_call_count == 1


def test_set_billing_route_mode_none_keeps_the_stored_mode(backend, scope_a):
    """CONTRACT: ``billing_mode=None`` keeps whatever mode is stored while provider and base URL still replace; an explicit mode replaces it."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("ubr-mode"))
    usage.set_billing_route(SessionId("ubr-mode"), provider="p1", base_url="u1", billing_mode="plan")
    usage.set_billing_route(SessionId("ubr-mode"), provider="p2", base_url="u2")
    assert billing_route(store_a, "ubr-mode") == ("p2", "u2", "plan")
    usage.set_billing_route(SessionId("ubr-mode"), provider="p3", base_url="u3", billing_mode="included")
    assert billing_route(store_a, "ubr-mode") == ("p3", "u3", "included")


def test_set_billing_route_survives_later_deltas_on_another_route(backend, scope_a):
    """CONTRACT: once usage is accounted, deltas after the switch attribute to their own route but never overwrite the set session route (the delta path fills only unset columns)."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("ubr-sticky"))
    usage.queue_usage(SessionId("ubr-sticky"), delta(ROUTE_A))
    usage.set_billing_route(SessionId("ubr-sticky"), provider="prov-set", base_url="https://set.example", billing_mode="plan")
    usage.queue_usage(SessionId("ubr-sticky"), delta(ROUTE_B))
    usage.queue_usage(SessionId("ubr-sticky"), delta(ROUTE_A))
    assert billing_route(store_a, "ubr-sticky") == ("prov-set", "https://set.example", "plan")
    assert (usage.recent_route(SessionId("ubr-sticky"))).route == ROUTE_A


def test_set_route_before_any_usage_yields_to_the_first_accounted_delta_but_not_to_an_unaccounted_one(backend, scope_a):
    """CONTRACT (first-accounted-route rule): a route set before any accounted usage survives a delta with zero counters, zero calls and a 0.0 actual cost (not accounted), and is replaced by the first delta that really accounts usage."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("ubr-early"))
    usage.set_billing_route(SessionId("ubr-early"), provider="prov-set", base_url="https://set.example", billing_mode="plan")
    empty = UsageDelta(tokens=TokenCounts(), api_call_count=0, estimated_cost_usd=None, actual_cost_usd=0.0, route=ROUTE_B)
    usage.queue_usage(SessionId("ubr-early"), empty)
    assert billing_route(store_a, "ubr-early") == ("prov-set", "https://set.example", "plan")
    usage.queue_usage(SessionId("ubr-early"), delta(ROUTE_B))
    assert billing_route(store_a, "ubr-early") == ("prov-b", "https://b.example", "included")


def test_set_billing_route_on_unknown_session_creates_nothing(backend, scope_a):
    """CONTRACT: an unknown id is left unknown: no row is minted and nothing raises."""
    store_a = backend.open(scope_a)
    store_a.usage.set_billing_route(SessionId("ubr-ghost"), provider="p", base_url="u", billing_mode="m")
    assert store_a.sessions.get(SessionId("ubr-ghost")) is None


def test_billing_route_is_scope_isolated(backend, scope_a, scope_b):
    """CONTRACT: a route set in one scope is invisible in the other, id collision or not."""
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("ubr-iso"))
    store_b.sessions.create(_new("ubr-iso"))
    store_a.usage.set_billing_route(SessionId("ubr-iso"), provider="pa", base_url="ua", billing_mode="ma")
    store_b.usage.set_billing_route(SessionId("ubr-iso"), provider="pb", base_url="ub", billing_mode="mb")
    assert billing_route(store_a, "ubr-iso") == ("pa", "ua", "ma")
    assert billing_route(store_b, "ubr-iso") == ("pb", "ub", "mb")


# ── auxiliary_usage_by_task ───────────────────────────────────────────────────


def aux(route: BillingRoute, *, inp: int, out: int = 0, calls: int = 1, cost: float = 0.0) -> dict[str, Any]:
    return {
        "model": route.model, "billing_provider": route.billing_provider, "billing_base_url": route.billing_base_url,
        "tokens": TokenCounts(input_tokens=inp, output_tokens=out), "api_call_count": calls, "estimated_cost_usd": cost,
    }


def build_lineage(backend, scope) -> None:
    """root -> child (compression continuation) -> grandchild; branch (explicit /branch copy of root) -> branch-child."""
    store = backend.open(scope)
    store.sessions.create(_new("ua-root"))
    store.sessions.create(_new("ua-child", parent_session_id=SessionId("ua-root")))
    store.sessions.create(_new("ua-grandchild", parent_session_id=SessionId("ua-child")))
    store.sessions.create(_new("ua-branch", parent_session_id=SessionId("ua-root"), lineage_kind=SessionLineageKind.BRANCH))
    store.sessions.create(_new("ua-branch-child", parent_session_id=SessionId("ua-branch")))
    usage = store.usage
    usage.record_auxiliary_usage(SessionId("ua-root"), "vision", **aux(ROUTE_A, inp=100, out=10, calls=2, cost=0.5))
    usage.record_auxiliary_usage(SessionId("ua-root"), "title_generation", **aux(ROUTE_B, inp=7, calls=1, cost=0.01))
    usage.record_auxiliary_usage(SessionId("ua-child"), "vision", **aux(ROUTE_B, inp=1000, out=100, calls=3, cost=1.5))
    usage.record_auxiliary_usage(SessionId("ua-grandchild"), "compression", **aux(ROUTE_A, inp=20, calls=1, cost=0.2))
    usage.record_auxiliary_usage(SessionId("ua-branch"), "vision", **aux(ROUTE_A, inp=5, calls=1, cost=0.05))
    usage.record_auxiliary_usage(SessionId("ua-branch-child"), "compression", **aux(ROUTE_A, inp=3, calls=1, cost=0.03))
    # Main-loop usage on every node except ua-branch-child: never part of the auxiliary sum.
    for sid in ("ua-root", "ua-child", "ua-grandchild", "ua-branch"):
        usage.queue_usage(SessionId(sid), delta(ROUTE_A, inp=10_000, calls=9))
    assert usage.flush_usage() is True


def test_auxiliary_usage_sums_the_lineage_chain_keyed_by_task(backend, scope_a):
    """CONTRACT: read from the compression tip, the sum covers the tip and every ancestor (root, child, grandchild), one entry per task keyed by task name; main-loop rows never count."""
    store_a = backend.open(scope_a)
    build_lineage(backend, scope_a)
    rows = store_a.usage.auxiliary_usage_by_task(SessionId("ua-grandchild"))
    assert dict(rows) == {
        "compression": AuxiliaryTaskUsage(task="compression", api_call_count=1, tokens=TokenCounts(input_tokens=20), estimated_cost_usd=pytest.approx(0.2)),
        "title_generation": AuxiliaryTaskUsage(task="title_generation", api_call_count=1, tokens=TokenCounts(input_tokens=7), estimated_cost_usd=pytest.approx(0.01)),
        "vision": AuxiliaryTaskUsage(task="vision", api_call_count=5, tokens=TokenCounts(input_tokens=1100, output_tokens=110), estimated_cost_usd=pytest.approx(2.0)),
    }
    # Read from the root: only the root's own rows (descendants are not ancestors).
    root_rows = store_a.usage.auxiliary_usage_by_task(SessionId("ua-root"))
    assert {task: (r.api_call_count, r.tokens.input_tokens) for task, r in root_rows.items()} == {"title_generation": (1, 7), "vision": (2, 100)}


def test_auxiliary_usage_walk_is_cut_at_an_explicit_branch(backend, scope_a):
    """CONTRACT: a branch owns its spend: read from the branch or its child, the root's rows are excluded while the branch's own rows still sum into its child."""
    store_a = backend.open(scope_a)
    build_lineage(backend, scope_a)
    branch = store_a.usage.auxiliary_usage_by_task(SessionId("ua-branch"))
    assert {task: (r.api_call_count, r.tokens.input_tokens) for task, r in branch.items()} == {"vision": (1, 5)}
    branch_child = store_a.usage.auxiliary_usage_by_task(SessionId("ua-branch-child"))
    assert {task: (r.api_call_count, r.tokens.input_tokens) for task, r in branch_child.items()} == {"compression": (1, 3), "vision": (1, 5)}


def test_auxiliary_usage_unknown_or_empty_id_reads_empty_and_scopes_never_mix(backend, scope_a, scope_b):
    """CONTRACT: an unknown id and an empty id read as empty; rows recorded in another scope under the same ids never count."""
    store_b = backend.open(scope_b)
    build_lineage(backend, scope_a)
    usage = backend.open(scope_a).usage
    assert dict(usage.auxiliary_usage_by_task(SessionId("ua-nobody"))) == {}
    assert dict(usage.auxiliary_usage_by_task(SessionId(""))) == {}
    store_b.sessions.create(_new("ua-grandchild"))
    assert dict(store_b.usage.auxiliary_usage_by_task(SessionId("ua-grandchild"))) == {}
    store_b.usage.record_auxiliary_usage(SessionId("ua-grandchild"), "vision", **aux(ROUTE_A, inp=1))
    assert {task: r.tokens.input_tokens for task, r in store_b.usage.auxiliary_usage_by_task(SessionId("ua-grandchild")).items()} == {"vision": 1}
    # scope_a's read is unchanged by scope_b's row.
    a_rows = usage.auxiliary_usage_by_task(SessionId("ua-grandchild"))
    assert sum(r.tokens.input_tokens for r in a_rows.values()) == 1127


def test_auxiliary_usage_without_a_route_still_sums_per_task(backend, scope_a):
    """CONTRACT: an auxiliary call that names no model or provider is still billed to its task (stored as the ``unknown`` route); the per-task sum is route-agnostic, and the session counters and recent route stay untouched."""
    store_a = backend.open(scope_a)
    usage = store_a.usage
    store_a.sessions.create(_new("ua-routeless"))
    usage.record_auxiliary_usage(SessionId("ua-routeless"), "vision", tokens=TokenCounts(input_tokens=17), estimated_cost_usd=0.7)
    usage.record_auxiliary_usage(SessionId("ua-routeless"), "vision", **aux(ROUTE_A, inp=3, cost=0.3))
    rows = usage.auxiliary_usage_by_task(SessionId("ua-routeless"))
    assert {task: (r.api_call_count, r.tokens.input_tokens) for task, r in rows.items()} == {"vision": (2, 20)}
    assert rows["vision"].estimated_cost_usd == pytest.approx(1.0)
    assert accounting(store_a, "ua-routeless").api_call_count == 0
    assert usage.recent_route(SessionId("ua-routeless")) is None


def test_auxiliary_usage_with_an_empty_id_or_task_writes_nothing(backend, scope_a):
    """CONTRACT: an empty session id or task returns before any write: no row is minted and nothing is attributed."""
    store_a = backend.open(scope_a)
    store_a.usage.record_auxiliary_usage(SessionId("ua-no-task"), "", **aux(ROUTE_A, inp=4))
    store_a.usage.record_auxiliary_usage(SessionId(""), "vision", **aux(ROUTE_A, inp=4))
    assert store_a.sessions.get(SessionId("ua-no-task")) is None
    assert dict(store_a.usage.auxiliary_usage_by_task(SessionId("ua-no-task"))) == {}


def test_auxiliary_usage_for_a_missing_row_mints_the_unknown_placeholder(backend, scope_a):
    """CONTRACT: an auxiliary call billed to a session with no row inserts the ``unknown`` placeholder (the creator later replaces it); the counters stay zero and the default counts one call."""
    store_a = backend.open(scope_a)
    store_a.usage.record_auxiliary_usage(SessionId("ua-early"), "title_generation", tokens=TokenCounts(input_tokens=2))
    row = store_a.sessions.get(SessionId("ua-early"))
    assert row is not None and row.source == "unknown" and row.api_call_count == 0
    assert store_a.usage.auxiliary_usage_by_task(SessionId("ua-early"))["title_generation"].api_call_count == 1
