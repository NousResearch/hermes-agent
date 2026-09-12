"""Budget reservation and reconciliation mixin for the Phase 2 authority store.

Cumulative, fence-scoped reservations with exact integer token ceilings and
exact ``Decimal`` USD accounting.  Mixed into ``Phase2AuthorityStore``; never
instantiated alone.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

from agent.phase2_envelope import (
    _budget_decimal,
    _normalized_reservation_metadata,
    _tokens_max_integer,
    _utc,
)
from agent.phase2_errors import AuthorityError
from agent.phase2_sqlite import _DB_LOCK


class Phase2BudgetMixin:
    """Budget methods; the concrete store supplies connection and gate hooks."""

    @staticmethod
    def _budget_state(
        conn: sqlite3.Connection,
        graph_id: str,
        node_id: str,
        fence: int | None = None,
    ) -> dict[str, Any]:
        query = """SELECT kind, event_id, payload_json FROM authority_events
                   WHERE graph_id = ? AND node_id = ?
                     AND kind IN ('BUDGET_RESERVED', 'BUDGET_RECONCILED')"""
        params: tuple[Any, ...] = (graph_id, node_id)
        if fence is not None:
            query += " AND fence = ?"
            params += (fence,)
        rows = conn.execute(query + " ORDER BY id", params).fetchall()
        reservations: dict[str, dict[str, Any]] = {}
        reconciled: set[str] = set()
        charged_tokens = 0
        charged_usd = Decimal(0)
        cost_unknown = False
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["kind"] == "BUDGET_RESERVED":
                reservations[row["event_id"]] = payload
            else:
                reservation_id = payload["reservation_id"]
                reconciled.add(reservation_id)
                charged_tokens += int(payload["actual_tokens"])
                if payload["actual_usd"] is None:
                    cost_unknown = True
                else:
                    charged_usd += _budget_decimal(payload["actual_usd"], "actual_usd")
        reserved_tokens = sum(
            int(payload["tokens"])
            for event_id, payload in reservations.items()
            if event_id not in reconciled
        )
        reserved_usd = sum(
            (
                _budget_decimal(payload["usd"], "usd")
                for event_id, payload in reservations.items()
                if event_id not in reconciled
            ),
            Decimal(0),
        )
        return {
            "reserved_tokens": reserved_tokens,
            "reserved_usd": reserved_usd,
            "charged_tokens": charged_tokens,
            "charged_usd": charged_usd,
            "cost_unknown": cost_unknown,
        }

    def reserve_budget_allocations(
        self,
        envelope: Mapping[str, Any],
        allocations: list[Mapping[str, Any]],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """Atomically reserve a bounded parent allocation for every child."""

        if not allocations:
            raise AuthorityError("budget allocations must be non-empty")
        normalized: list[dict[str, Any]] = []
        for allocation in allocations:
            tokens = allocation.get("tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise AuthorityError("tokens must be a non-negative integer")
            usd = allocation.get("usd")
            _budget_decimal(usd, "usd")
            normalized.append({
                "tokens": tokens,
                "usd": usd,
                "metadata": _normalized_reservation_metadata(
                    allocation.get("metadata"), "budget allocation metadata"
                ),
            })
        current = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id, node_id, fence, latest_envelope = self._load_live_authority(
                    conn, envelope, current, action="reserve budget"
                )
                state = self._budget_state(conn, graph_id, node_id)
                if state["cost_unknown"]:
                    raise AuthorityError("budget cost is unknown")
                limits = latest_envelope["budgets"]
                tokens_total = sum(item["tokens"] for item in normalized)
                usd_total = sum(
                    (_budget_decimal(item["usd"], "usd") for item in normalized),
                    Decimal(0),
                )
                tokens_max = _tokens_max_integer(
                    limits.get("tokens_max"), "sealed budgets.tokens_max"
                )
                usd_max = _budget_decimal(
                    limits.get("usd_max"), "sealed budgets.usd_max"
                )
                if (
                    state["charged_tokens"] + state["reserved_tokens"] + tokens_total
                    > tokens_max
                ):
                    raise AuthorityError("token budget reservation exceeds node budget")
                if state["charged_usd"] + state["reserved_usd"] + usd_total > usd_max:
                    raise AuthorityError("USD budget reservation exceeds node budget")
                reservation_ids: list[str] = []
                for item in normalized:
                    reservation_id = f"res-{uuid.uuid4().hex}"
                    payload: dict[str, Any] = {
                        "tokens": item["tokens"],
                        "usd": item["usd"],
                    }
                    if item["metadata"] is not None:
                        payload["metadata"] = item["metadata"]
                    self._append_event(
                        conn,
                        kind="BUDGET_RESERVED",
                        graph_id=graph_id,
                        node_id=node_id,
                        attempt_id=str(latest_envelope["attempt_id"]),
                        fence=fence,
                        payload=payload,
                        now=current,
                        event_id=reservation_id,
                    )
                    reservation_ids.append(reservation_id)
                conn.execute("COMMIT")
                return reservation_ids
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def reserve_budget(
        self,
        envelope: Mapping[str, Any],
        *,
        tokens: int,
        usd: float,
        metadata: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> str:
        """Reserve spend against the node budget under exact live authority."""

        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise AuthorityError("tokens must be a non-negative integer")
        usd_value = usd
        _budget_decimal(usd_value, "usd")
        metadata_payload = _normalized_reservation_metadata(metadata)
        current = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id, node_id, fence, latest_envelope = self._load_live_authority(
                    conn, envelope, current, action="reserve budget"
                )
                state = self._budget_state(conn, graph_id, node_id)
                if state["cost_unknown"]:
                    raise AuthorityError("budget cost is unknown")
                limits = latest_envelope["budgets"]
                tokens_max = _tokens_max_integer(
                    limits.get("tokens_max"), "sealed budgets.tokens_max"
                )
                usd_max = _budget_decimal(
                    limits.get("usd_max"), "sealed budgets.usd_max"
                )
                if (
                    state["charged_tokens"] + state["reserved_tokens"] + tokens
                    > tokens_max
                ):
                    raise AuthorityError("token budget reservation exceeds node budget")
                if (
                    state["charged_usd"]
                    + state["reserved_usd"]
                    + _budget_decimal(usd_value, "usd")
                    > usd_max
                ):
                    raise AuthorityError("USD budget reservation exceeds node budget")
                reservation_id = f"res-{uuid.uuid4().hex}"
                self._append_event(
                    conn,
                    kind="BUDGET_RESERVED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=str(latest_envelope["attempt_id"]),
                    fence=fence,
                    payload={
                        "tokens": tokens,
                        "usd": usd_value,
                        **(
                            {"metadata": metadata_payload}
                            if metadata_payload is not None
                            else {}
                        ),
                    },
                    now=current,
                    event_id=reservation_id,
                )
                conn.execute("COMMIT")
                return reservation_id
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def reconcile_budget(
        self,
        reservation_id: str,
        *,
        actual_tokens: int,
        actual_usd: float | None,
        now: datetime | None = None,
    ) -> None:
        if (
            isinstance(actual_tokens, bool)
            or not isinstance(actual_tokens, int)
            or actual_tokens < 0
        ):
            raise AuthorityError("actual_tokens must be a non-negative integer")
        if actual_usd is not None:
            _budget_decimal(actual_usd, "actual_usd")
        current = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                reservation = conn.execute(
                    "SELECT * FROM authority_events WHERE event_id = ? AND kind = 'BUDGET_RESERVED'",
                    (reservation_id,),
                ).fetchone()
                if reservation is None:
                    raise AuthorityError("unknown budget reservation")
                duplicate = conn.execute(
                    """SELECT payload_json FROM authority_events
                       WHERE kind = 'BUDGET_RECONCILED'
                         AND json_extract(payload_json, '$.reservation_id') = ?""",
                    (reservation_id,),
                ).fetchone()
                if duplicate:
                    prior = json.loads(duplicate["payload_json"])
                    if (
                        int(prior.get("actual_tokens", -1)) == actual_tokens
                        and prior.get("actual_usd") == actual_usd
                    ):
                        conn.execute("COMMIT")
                        return
                    raise AuthorityError("budget reservation already reconciled")
                self._append_event(
                    conn,
                    kind="BUDGET_RECONCILED",
                    graph_id=reservation["graph_id"],
                    node_id=reservation["node_id"],
                    attempt_id=reservation["attempt_id"],
                    fence=reservation["fence"],
                    payload={
                        "reservation_id": reservation_id,
                        "actual_tokens": actual_tokens,
                        "actual_usd": actual_usd,
                    },
                    now=current,
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def budget_usage(
        self, graph_id: str, node_id: str, *, fence: int
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            state = self._budget_state(conn, graph_id, node_id, fence)
            return {
                **state,
                "reserved_usd": float(state["reserved_usd"]),
                "charged_usd": float(state["charged_usd"]),
            }
        finally:
            conn.close()
