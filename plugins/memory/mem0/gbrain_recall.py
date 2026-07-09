"""gbrain document-recall leg for the mem0 plugin (Phase 2b, 2026-07-08).

A flag-gated ALTERNATIVE to the QMD leg (qmd_recall): when the `mem0_gbrain` config
block sets `enabled: true`, the prefetch + mem0_search document lanes call the warm
gbrain serve (loopback MCP-over-HTTP, OAuth 2.1 client_credentials) instead of the
QMD daemon. One retrieval leg per turn — never both.

Design mirrors qmd_recall on purpose:
- stdlib-only (http.client / urllib), no `mcp` SDK, no new pip dependency;
- a hard wall-clock deadline enforced by a watchdog that shuts the live socket
  (interrupts an SSE keepalive trickle hang);
- degraded-safe: ANY failure -> [] — a down gbrain must never break a turn;
- returns the exact QMD pointer shape [{file, title, score, line, docid}] so the
  hits ride the existing rendering / budget / intent-gate machinery unchanged.
  `file` carries the gbrain page slug (the brain's stable identifier — resolvable
  via `rail.sh graph <slug>` / the `get_page` op); `docid` is "gbrain:<page_id>".

OAuth: client credentials are read from a 600-mode env file (default
~/gbrain/.gbrain/rail-client.env — GBRAIN_RAIL_CLIENT_ID / _SECRET). The bearer is
cached at module level (process-wide, thread-safe) and refreshed <120s from expiry,
so steady-state turns pay ZERO auth round-trips (one mint per process per ~hour).
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)

# ---- config defaults --------------------------------------------------------
# Mirrors qmd_recall.QMD_DEFAULTS' shape/gating semantics. Loaded from the
# `mem0_gbrain` (fallback `gbrain`) block of mem0.json — the SAME config surface
# _qmd_cfg comes from. `enabled` defaults False: deploy is inert until flipped.
GBRAIN_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    # Sub-lane gates (meaningful only when `enabled`). Both default true so flipping
    # only `enabled` swaps BOTH lanes from QMD to gbrain in one move.
    "prefetch_enabled": True,
    "search_enabled": True,
    "url": "http://127.0.0.1:8199",           # warm serve base (launchd ai.gbrain.serve)
    "creds_path": "~/gbrain/.gbrain/rail-client.env",
    # Whole-operation wall-clock deadline (token mint, when needed, included).
    # gbrain `search` measured warm p50 1.14s / p95 1.58s (2026-07-08) — 4.0s
    # matches the QMD budget precedent and leaves headroom for a cold token mint.
    "total_deadline_s": 4.0,
    "mem0_budget_s": 6.0,      # same INV-4a semantics as the QMD leg
    # gbrain scores are calibrated-ish vector/RRF blends (top hits ~0.8+); 0.5
    # trims junk-tail rows without clipping legitimate rank-2/3 results.
    "min_score": 0.5,
    "prefetch_limit": 3,
    "search_limit": 5,
    "intent_min_tokens": 1,    # reuses qmd_recall.is_lookup_intent (same gate)
}

_TOKEN_REFRESH_MARGIN_S = 120.0

# Process-wide token cache: one auth per process, not per turn. Keyed by
# (base_url, creds_path) so tests / multi-instance configs can't cross-feed.
_token_lock = threading.Lock()
_token_cache: Dict[Any, Dict[str, Any]] = {}
_mint_locks: Dict[Any, threading.Lock] = {}  # per-key mint serialization (stampede guard)


def load_gbrain_config(raw: Optional[dict]) -> Dict[str, Any]:
    """Merge a `mem0_gbrain` sub-block over the defaults. Missing block -> defaults
    (enabled False)."""
    cfg = dict(GBRAIN_DEFAULTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in cfg and v is not None:
                cfg[k] = v
    return cfg


def _read_client_creds(creds_path: str) -> Optional[Dict[str, str]]:
    """Parse the KEY=VALUE env file holding the OAuth client credentials."""
    path = os.path.expanduser(creds_path or "")
    try:
        out: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip().strip('"').strip("'")
        cid = out.get("GBRAIN_RAIL_CLIENT_ID", "")
        sec = out.get("GBRAIN_RAIL_CLIENT_SECRET", "")
        if not cid or not sec:
            return None
        return {"client_id": cid, "client_secret": sec}
    except Exception as e:
        logger.debug("gbrain creds read failed: %s", e)
        return None


def _http_post(base_url: str, path: str, body: bytes, headers: Dict[str, str],
               timeout_s: float, state: Optional[Dict[str, Any]] = None,
               lock: Optional[threading.Lock] = None):
    """One POST via http.client. When (state, lock) are provided, the live
    connection/socket handles are published for the caller's watchdog to shut."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8199
    conn = http.client.HTTPConnection(host, port, timeout=max(0.05, float(timeout_s)))
    if state is not None and lock is not None:
        with lock:
            if state.get("fired"):
                conn.close()
                raise TimeoutError("gbrain deadline tripped")
            state["conn"] = conn
    try:
        conn.request("POST", path, body=body, headers=headers)
        if state is not None and lock is not None:
            with lock:
                state["sock"] = getattr(conn, "sock", None)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", "replace")
        return resp.status, data
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_token(base_url: str, creds_path: str, timeout_s: float) -> Optional[str]:
    """Cached bearer for the warm serve. One /token round-trip per process per
    token lifetime (server TTL ~3600s); refreshed <120s from expiry. Never raises.

    Stampede-safe: a per-key mint lock serializes concurrent minters; each minter
    re-checks the cache after acquiring it, so N threads racing an empty/expired
    cache produce exactly ONE /token round-trip (Greptile #248)."""
    key = (base_url, os.path.expanduser(creds_path or ""))

    def _cached() -> Optional[str]:
        ent = _token_cache.get(key)
        if ent and ent.get("token") and float(ent.get("expires_at", 0)) > time.time() + _TOKEN_REFRESH_MARGIN_S:
            return ent["token"]
        return None

    with _token_lock:
        tok = _cached()
        if tok:
            return tok
        mint_lock = _mint_locks.setdefault(key, threading.Lock())

    with mint_lock:
        # Re-check under the mint lock: a racing thread may have minted while we waited.
        with _token_lock:
            tok = _cached()
            if tok:
                return tok
        creds = _read_client_creds(creds_path)
        if not creds:
            return None
        try:
            body = urlencode({
                "grant_type": "client_credentials",
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
            }).encode("utf-8")
            st, data = _http_post(
                base_url, "/token", body,
                {"Content-Type": "application/x-www-form-urlencoded"},
                timeout_s,
            )
            if st != 200:
                return None
            d = json.loads(data)
            tok = d.get("access_token")
            if not tok:
                return None
            expires_at = time.time() + float(d.get("expires_in", 3600) or 3600)
            with _token_lock:
                _token_cache[key] = {"token": tok, "expires_at": expires_at}
            return tok
        except Exception as e:
            logger.debug("gbrain token mint failed: %s", e)
            return None


def _invalidate_token(base_url: str, creds_path: str) -> None:
    key = (base_url, os.path.expanduser(creds_path or ""))
    with _token_lock:
        _token_cache.pop(key, None)


def _extract_json(raw: str) -> Optional[Any]:
    """Parse a possibly-SSE-framed JSON body (collect `data:` lines, else raw)."""
    data_lines = [ln[5:].lstrip() for ln in raw.splitlines() if ln.startswith("data:")]
    body = "".join(data_lines) if data_lines else raw
    try:
        return json.loads(body)
    except Exception:
        return None


def parse_gbrain_results(payload: Any, min_score: float) -> List[Dict[str, Any]]:
    """Pure: map a `search` tools/call response to the QMD pointer shape.

    GROUND-TRUTHED response shape (v0.42.x, 2026-07-08): the MCP result carries
    result.content[0].text = a JSON ARRAY of rows, each with slug / page_id /
    title / score (float) / chunk_text / ... . Pointers only — chunk_text is
    DROPPED (same INV-5 posture as the QMD leg: never inject document bodies).
    """
    try:
        result = (payload or {}).get("result", {})
        if result.get("isError"):
            return []
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        rows = json.loads(text) if text else []
    except (AttributeError, IndexError, ValueError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            score = float(r.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        if score < float(min_score):
            continue
        slug = str(r.get("slug", "") or "")
        if not slug:
            continue
        page_id = r.get("page_id")
        out.append({
            "file": slug,                     # brain-relative identifier (page slug)
            "title": str(r.get("title", "") or ""),
            "score": round(score, 4),
            "line": 0,                        # gbrain chunks carry no line numbers
            "docid": f"gbrain:{page_id}" if page_id is not None else "gbrain:",
        })
    return out


def render_gbrain_block(hits: List[Dict[str, Any]]) -> str:
    """Pure renderer, same contract as qmd_recall.render_qmd_block: empty -> ""
    (no header). The header names the backend so a transcript reader can tell
    which leg served the turn."""
    if not hits:
        return ""
    lines = ["## Local Docs (gbrain)"]
    for h in hits:
        pct = f"{round(float(h.get('score', 0)) * 100)}%"
        title = h.get("title") or h.get("file", "")
        lines.append(f"- {h.get('file','')} — {title} ({pct})")
    return "\n".join(lines)


def gbrain_search(query: str, *, limit: int, min_score: float, deadline_s: float,
                  url: str = "http://127.0.0.1:8199",
                  creds_path: str = "~/gbrain/.gbrain/rail-client.env",
                  ) -> List[Dict[str, Any]]:
    """Hard-cancellable `search` op against the warm gbrain serve. ANY failure -> [].

    The wall-clock deadline covers the WHOLE operation including a token mint when
    the cache is cold (steady state: zero auth round-trips). A watchdog timer trips
    at `deadline_s` and shuts down the live socket so a blocked read — including an
    SSE keepalive trickle — raises immediately instead of hanging the turn.
    Degraded-safe: never raises; the watchdog is always cancelled.
    """
    t0 = time.monotonic()
    deadline_s = max(0.05, float(deadline_s))

    state: Dict[str, Any] = {"conn": None, "sock": None, "fired": False}
    lock = threading.Lock()

    def _trip() -> None:
        with lock:
            state["fired"] = True
            conn = state.get("conn")
            sock = state.get("sock")
        for s in (sock, getattr(conn, "sock", None) if conn is not None else None):
            if s is None:
                continue
            try:
                s.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    timer = threading.Timer(deadline_s, _trip)
    timer.daemon = True
    timer.start()
    try:
        token = _get_token(url, creds_path, timeout_s=deadline_s)
        if not token:
            return []
        remaining = deadline_s - (time.monotonic() - t0)
        if remaining < 0.2:
            return []
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search",
                       "arguments": {"query": query, "limit": int(limit)}},
        }).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Connection": "close",
        }
        st, data = _http_post(url, "/mcp", body, headers, remaining, state, lock)
        if st in (401, 403):
            # Token revoked/expired server-side (e.g. serve restarted with a fresh
            # key). Invalidate and retry ONCE within the remaining budget; the next
            # turn then rides the fresh cached token.
            _invalidate_token(url, creds_path)
            token = _get_token(url, creds_path,
                               timeout_s=max(0.05, deadline_s - (time.monotonic() - t0)))
            remaining = deadline_s - (time.monotonic() - t0)
            if not token or remaining < 0.2:
                return []
            headers["Authorization"] = f"Bearer {token}"
            st, data = _http_post(url, "/mcp", body, headers, remaining, state, lock)
        if st != 200:
            return []
        payload = _extract_json(data)
        if payload is None:
            return []
        return parse_gbrain_results(payload, min_score)
    except Exception as e:  # degraded-safe: ANY failure -> []
        logger.debug("gbrain_search degraded: %s", e)
        return []
    finally:
        timer.cancel()
