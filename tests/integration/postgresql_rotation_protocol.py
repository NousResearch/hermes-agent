"""Test-only PG18 transaction protocol for the future rotation adapter.

This module is deliberately not imported by production.  It gives the acceptance
gate a real database protocol with durable receipts, server-clock fences and
phase barriers before the production adapter exists.  A production publisher
must replace this helper, not reuse it.
"""
from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PHASES = ("lease-revalidated", "child-row-inserted", "handoff-inserted", "parent-close-issued", "commit-returned")


def psycopg() -> Any:
    return importlib.import_module("psycopg")


@dataclass(frozen=True)
class RotationReceipt:
    parent_id: str
    child_id: str
    owner: str
    fence: int
    request_id: str


def install(dsn: str, schema: str) -> None:
    q = f'"{schema}"'
    with psycopg().connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(f"""
            CREATE TABLE {q}.rotation_harness_parents (
              id text PRIMARY KEY, tenant text NOT NULL, prompt text NOT NULL,
              model text NOT NULL, config jsonb NOT NULL, title text NOT NULL,
              lineage text NOT NULL, visible boolean NOT NULL, activity double precision NOT NULL,
              cooldown double precision NOT NULL, fallback integer NOT NULL, ineffective integer NOT NULL,
              watermark bigint NOT NULL, generation integer NOT NULL, closed boolean NOT NULL DEFAULT false);
            CREATE TABLE {q}.rotation_harness_leases (
              parent_id text PRIMARY KEY REFERENCES {q}.rotation_harness_parents(id),
              owner text NOT NULL, fence integer NOT NULL, expires_at double precision NOT NULL);
            CREATE TABLE {q}.rotation_harness_children (
              id text PRIMARY KEY, parent_id text NOT NULL REFERENCES {q}.rotation_harness_parents(id),
              tenant text NOT NULL, prompt text NOT NULL, model text NOT NULL, config jsonb NOT NULL,
              title text NOT NULL, lineage text NOT NULL, visible boolean NOT NULL, activity double precision NOT NULL,
              cooldown double precision NOT NULL, fallback integer NOT NULL, ineffective integer NOT NULL,
              watermark bigint NOT NULL, generation integer NOT NULL, UNIQUE(parent_id));
            CREATE TABLE {q}.rotation_harness_messages (
              child_id text NOT NULL REFERENCES {q}.rotation_harness_children(id), ordinal integer NOT NULL,
              role text NOT NULL, content text NOT NULL, PRIMARY KEY(child_id, ordinal));
            CREATE TABLE {q}.rotation_harness_receipts (
              request_id text PRIMARY KEY, parent_id text NOT NULL, child_id text NOT NULL,
              owner text NOT NULL, fence integer NOT NULL, state text NOT NULL, backend_pid integer NOT NULL,
              UNIQUE(parent_id, child_id));
        """)


def seed(dsn: str, schema: str, parent: str, tenant: str = "tenant-a") -> None:
    q = f'"{schema}"'
    with psycopg().connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(f"INSERT INTO {q}.rotation_harness_parents VALUES (%s,%s,%s,%s,%s,%s,%s,true,1234,2345,3,2,9,4,false)",
                       (parent, tenant, "exact cached prompt", "oracle/model", json.dumps({"max_tokens": None}), "Oracle title", "root/parent"))
        connection.commit()


def acquire(dsn: str, schema: str, parent: str, owner: str, ttl: float = 60) -> int | None:
    q = f'"{schema}"'
    with psycopg().connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{schema}:rotation-harness:{parent}",))
        cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
        now = float(cursor.fetchone()[0])
        cursor.execute(f"SELECT owner, fence, expires_at FROM {q}.rotation_harness_leases WHERE parent_id=%s FOR UPDATE", (parent,))
        row = cursor.fetchone()
        if row and row[0] != owner and float(row[2]) > now:
            connection.rollback(); return None
        fence = 1 if row is None else int(row[1]) + (row[0] != owner)
        cursor.execute(f"INSERT INTO {q}.rotation_harness_leases(parent_id,owner,fence,expires_at) VALUES(%s,%s,%s,%s) ON CONFLICT(parent_id) DO UPDATE SET owner=EXCLUDED.owner,fence=EXCLUDED.fence,expires_at=EXCLUDED.expires_at", (parent, owner, fence, now + ttl))
        connection.commit()
        return fence


def expire(dsn: str, schema: str, parent: str) -> None:
    q = f'"{schema}"'
    with psycopg().connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(f"UPDATE {q}.rotation_harness_leases SET expires_at=EXTRACT(EPOCH FROM clock_timestamp())-1 WHERE parent_id=%s", (parent,))


def publish(dsn: str, schema: str, receipt: RotationReceipt, phase: Callable[[str, int], None] | None = None) -> RotationReceipt:
    """One real transaction; phase callback runs before the next SQL mutation."""
    q = f'"{schema}"'
    with psycopg().connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()"); backend_pid = int(cursor.fetchone()[0])
        cursor.execute(f"SELECT child_id, owner, fence, state FROM {q}.rotation_harness_receipts WHERE request_id=%s", (receipt.request_id,))
        prior = cursor.fetchone()
        if prior:
            if tuple(prior[:3]) != (receipt.child_id, receipt.owner, receipt.fence):
                raise RuntimeError("idempotency receipt conflicts with a different publication")
            connection.commit(); return receipt
        cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
        now = float(cursor.fetchone()[0])
        cursor.execute(f"SELECT owner,fence FROM {q}.rotation_harness_leases WHERE parent_id=%s AND expires_at>%s FOR UPDATE", (receipt.parent_id, now))
        lease = cursor.fetchone()
        if lease != (receipt.owner, receipt.fence):
            raise RuntimeError("stale or expired rotation lease")
        if phase: phase("lease-revalidated", backend_pid)
        cursor.execute(f"""INSERT INTO {q}.rotation_harness_children
          SELECT %s,id,tenant,prompt,model,config,title,lineage,visible,0,0,0,0,watermark,generation+1
          FROM {q}.rotation_harness_parents WHERE id=%s AND closed=false""", (receipt.child_id, receipt.parent_id))
        if cursor.rowcount != 1: raise RuntimeError("parent is already closed")
        if phase: phase("child-row-inserted", backend_pid)
        cursor.execute(f"INSERT INTO {q}.rotation_harness_messages VALUES(%s,0,'assistant','[CONTEXT COMPACTION] deterministic summary'),(%s,1,'user','deterministic live tail')", (receipt.child_id, receipt.child_id))
        if phase: phase("handoff-inserted", backend_pid)
        cursor.execute(f"UPDATE {q}.rotation_harness_parents SET closed=true WHERE id=%s AND closed=false", (receipt.parent_id,))
        if cursor.rowcount != 1: raise RuntimeError("parent close lost")
        if phase: phase("parent-close-issued", backend_pid)
        cursor.execute(f"INSERT INTO {q}.rotation_harness_receipts VALUES(%s,%s,%s,%s,%s,'committed',%s)", (receipt.request_id, receipt.parent_id, receipt.child_id, receipt.owner, receipt.fence, backend_pid))
        connection.commit()
    if phase: phase("commit-returned", backend_pid)
    return receipt


def audit(dsn: str, schema: str, parent: str) -> dict[str, Any]:
    q = f'"{schema}"'
    with psycopg().connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT closed FROM {q}.rotation_harness_parents WHERE id=%s", (parent,)); closed = cursor.fetchone()[0]
        cursor.execute(f"SELECT id,tenant,prompt,model,config,title,lineage,visible,activity,cooldown,fallback,ineffective,watermark,generation FROM {q}.rotation_harness_children WHERE parent_id=%s", (parent,)); children = cursor.fetchall()
        cursor.execute(f"SELECT request_id,state FROM {q}.rotation_harness_receipts WHERE parent_id=%s", (parent,)); receipts = cursor.fetchall()
        messages = [] if not children else _messages(cursor, q, children[0][0])
    return {"closed": closed, "children": children, "receipts": receipts, "messages": messages}


def _messages(cursor: Any, q: str, child: str) -> list[tuple[str, str]]:
    cursor.execute(f"SELECT role,content FROM {q}.rotation_harness_messages WHERE child_id=%s ORDER BY ordinal", (child,))
    return list(cursor.fetchall())
