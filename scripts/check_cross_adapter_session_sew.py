#!/usr/bin/env python3
"""Detect cross-adapter session sews (#64934).

Read-only diagnostic for ``state.db.gateway_routing``. The #64934 bug merged
multiple distinct Feishu-app routing keys (e.g. ``adapter=feishu%3Acli_aad581a8``
and ``adapter=feishu%3Acli_aad58273d``) onto ONE ``session_id`` during async-
delegation completion, fusing Tony/Sam into Pete's transcript. After the fix,
no ``session_id`` should be the target of routing keys from more than one
adapter (with one documented exception: CLI continuity handoff, which is an
intentional single cross-adapter rebind, not a multi-adapter fan-out).

Exit codes:
  0 — no multi-adapter sew detected
  1 — at least one session_id is shared across adapters (printed with details)

Usage:
  python scripts/check_cross_adapter_session_sew.py            # $HERMES_HOME/state.db
  python scripts/check_cross_adapter_session_sew.py --db PATH
  python scripts/check_cross_adapter_session_sew.py --selftest
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote


def adapter_id_from_key(session_key: str) -> Optional[str]:
    """Extract the ``adapter=<id>`` segment from a session key.

    Namespace-agnostic (does not assume ``parts[1]=='main'``), so multi-profile
    keys still resolve. Returns None when the key carries no adapter segment.
    Mirrors ``gateway.session._adapter_id_from_key``.
    """
    if not session_key:
        return None
    for part in session_key.split(":"):
        if part.startswith("adapter="):
            return unquote(part[len("adapter="):]) or None
    return None


# A row: (session_key, session_id, updated_at_epoch_or_None)
Row = Tuple[str, Optional[str], Optional[float]]


def detect_sews(rows: Iterable[Row]) -> List[dict]:
    """Group routing rows by session_id and flag any shared across adapters.

    Returns one dict per offending session_id: ``session_id``, ``adapters``
    (sorted distinct list), ``keys`` (the routing keys), and the max
    ``updated_at`` among them (so callers can tell a stale historical sew from
    a fresh post-fix one).
    """
    by_sid_adapters: dict = defaultdict(set)
    by_sid_keys: dict = defaultdict(list)
    by_sid_ts: dict = defaultdict(list)
    for session_key, sid, updated_at in rows:
        aid = adapter_id_from_key(session_key)
        if not aid or not sid:
            continue
        by_sid_adapters[sid].add(aid)
        by_sid_keys[sid].append(session_key)
        if updated_at is not None:
            by_sid_ts[sid].append(updated_at)
    sews: List[dict] = []
    for sid, adapters in by_sid_adapters.items():
        if len(adapters) > 1:
            sews.append(
                {
                    "session_id": sid,
                    "adapters": sorted(adapters),
                    "keys": by_sid_keys[sid],
                    "updated_at": max(by_sid_ts[sid]) if by_sid_ts[sid] else None,
                }
            )
    sews.sort(key=lambda s: (s["updated_at"] or 0.0), reverse=True)
    return sews


def _default_db_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "state.db"
    except Exception:
        return Path.home() / ".hermes" / "state.db"


def load_rows(db_path: Path) -> List[Row]:
    """Read all adapter-bearing gateway_routing rows (read-only)."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        # schema: gateway_routing(scope TEXT, session_key TEXT, entry_json TEXT, updated_at REAL)
        rows = conn.execute(
            """SELECT session_key,
                      json_extract(entry_json, '$.session_id'),
                      updated_at
               FROM gateway_routing
               WHERE session_key LIKE '%adapter=%'"""
        ).fetchall()
    finally:
        conn.close()
    return [(k, s, ts) for k, s, ts in rows]


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "?"
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat(timespec="seconds")


def report(sews: Sequence[dict], total_rows: int) -> str:
    if not sews:
        return (
            f"PASS: scanned {total_rows} adapter-bearing routing rows; "
            "no session_id is shared across multiple adapters (#64934 clean)."
        )
    lines = [
        f"FAIL: {len(sews)} session_id(s) shared across multiple adapters "
        f"(of {total_rows} adapter-bearing rows):",
    ]
    for s in sews:
        lines.append("")
        lines.append(f"  session_id   = {s['session_id']}")
        lines.append(f"  updated_at   = {_fmt_ts(s['updated_at'])}")
        lines.append(f"  adapters     = {', '.join(s['adapters'])}  ({len(s['adapters'])})")
        for k in s["keys"]:
            lines.append(f"    key = {k}")
    lines.append("")
    lines.append(
        "NOTE: a single historical sew is expected to persist in state.db from "
        "before the fix. A sew whose updated_at is AFTER the fix deploy indicates "
        "the bug is still firing — re-run the failing delegation scenario."
    )
    return "\n".join(lines)


def _selftest() -> int:
    """Exercise the pure detection logic on synthetic rows."""
    # 1. No sew — two adapters on two distinct sessions.
    clean = [
        ("agent:main:feishu:adapter=feishu%3AappA:group:oc_c:omt_t", "sess_A", 1.0),
        ("agent:main:feishu:adapter=feishu%3AappB:group:oc_c:omt_t", "sess_B", 2.0),
    ]
    assert detect_sews(clean) == [], "distinct sessions must not flag"

    # 2. Sew — two adapters onto one session (the #64934 shape).
    sewn = [
        ("agent:main:feishu:adapter=feishu%3AappA:group:oc_c:omt_t", "sess_X", 1.0),
        ("agent:main:feishu:adapter=feishu%3AappB:group:oc_c:omt_t", "sess_X", 2.0),
    ]
    found = detect_sews(sewn)
    assert len(found) == 1, found
    assert found[0]["session_id"] == "sess_X"
    assert sorted(found[0]["adapters"]) == ["feishu:appA", "feishu:appB"]
    assert found[0]["updated_at"] == 2.0

    # 3. Same adapter, two keys onto one session — NOT a sew (legitimate alias).
    alias = [
        ("agent:main:feishu:adapter=feishu%3AappA:dm:oc_c", "sess_Y", 1.0),
        ("agent:main:feishu:adapter=feishu%3AappA:dm:oc_d", "sess_Y", 2.0),
    ]
    assert detect_sews(alias) == [], "same-adapter alias must not flag"

    # 4. namespace-agnostic: profile namespace must not blind the parser.
    assert adapter_id_from_key("agent:coder:feishu:adapter=feishu%3AappA:group:x") == "feishu:appA"
    assert adapter_id_from_key("agent:main:feishu:group:x") is None

    print("selftest OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect cross-adapter session sews (#64934).")
    parser.add_argument("--db", default=None, help="state.db path (default: $HERMES_HOME/state.db)")
    parser.add_argument("--selftest", action="store_true", help="run built-in detection selftest")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    db_path = Path(args.db) if args.db else _default_db_path()
    if not db_path.exists():
        print(f"state.db not found at {db_path}", file=sys.stderr)
        return 2
    try:
        rows = load_rows(db_path)
    except sqlite3.Error as exc:
        print(f"failed to read {db_path}: {exc}", file=sys.stderr)
        return 2

    sews = detect_sews(rows)
    print(report(sews, total_rows=len(rows)))
    return 1 if sews else 0


if __name__ == "__main__":
    raise SystemExit(main())
