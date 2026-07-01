"""QMD document-recall fold-in for the mem0 plugin (Unified Recall, spec v0.3, 2026-06-28).

Folds the LOCAL QMD hybrid document index into mem0's two recall paths — prefetch (every
turn, intent-gated, mem0-first) and the explicit mem0_search tool (additive `docs` key) —
WITHOUT merging the stores. QMD stays read-only document SEARCH; nothing here ever writes
to mem0 or QMD (INV-1).

Lives in the mem0 plugin package (a sibling submodule, imported by __init__.py) so the pure
functions are unit-testable without the network-heavy mem0 client. No new pip dependency, no
`mcp` SDK — a tiny stdlib http.client MCP client with a hard wall-clock deadline enforced by
a watchdog that shuts the socket (interrupts an SSE keepalive trickle hang — INV-4).
"""

from __future__ import annotations

import fnmatch
import http.client
import json
import logging
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---- config defaults (D-5) -------------------------------------------------
QMD_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    # Sub-lane gates (only meaningful when `enabled` is true). Let an operator kill the
    # every-turn PREFETCH lane (cost + noise) while keeping the explicit mem0_search fan-out —
    # or vice versa — without turning the whole integration off. Both default true so flipping
    # only `enabled` behaves exactly as before.
    "prefetch_enabled": True,       # the automatic per-turn QMD leg in prefetch
    "search_enabled": True,         # the QMD `docs` fan-out on an explicit mem0_search call
    "url": "http://[::1]:8181/mcp",
    "qmd_total_deadline_s": 6.0,   # whole-operation wall-clock deadline (INV-4); 6s catches
                                   # real warm hybrid+rerank latency (measured 2.3-5.3s live,
                                   # 2026-06-30) with margin, still bounded under the 10s join
    "mem0_budget_s": 6.0,          # the mem0 leg's own budget (INV-4a); +deadline <= join 10s
    "min_score": 0.45,            # RRF/rerank score is POSITIONAL not calibrated-relevance
                                  # (rank1~0.9, rank2~0.5, tail 0.34-0.47 — identical for a real
                                  # and a nonsense query; measured 2026-06-30). So this floor TRIMS
                                  # the low-rank tail, it does NOT gate relevance. Relevance is
                                  # protected by prefetch_limit + the intent gate. 0.45 keeps the
                                  # legit rank-2 hit that 0.5 flakily clipped.
    "prefetch_limit": 3,
    "search_limit": 5,
    # allowlist — sessions & memories EXCLUDED by default (egress-aware, INV-5)
    "collections": ["obsidian", "skills", "plans", "projects"],
    "exclude_path_globs": [],      # client-side post-filter on `file` (INV-5/N3)
    "intent_min_tokens": 1,  # the LEADER-word set (ok/yes/fix/run/ship...) is the real gate and catches affirmations at any length; the token-count floor only ever dropped legit SHORT lookups (a bare hostname "blocked.local.ace", a skill name "unifi") so it is set to 1. Proven safe 2026-06-30: 0 single-word non-lookups pass, 0 short lookups skipped.
    "prefetch_rerank": True,
    # Default-off semantic floor (depends on the local/upstream QMD rerankScore bridge).
    # When enabled, qmd_query requests explain:true, lowers daemon-side blended minScore to
    # 0.0, and filters client-side on the pure reranker score. Missing/null rerankScore is
    # dropped, never silently treated as blended score (no fake-green on an unpatched daemon).
    "use_rerank_score_floor": False,
    "rerank_score_min": 0.50,
}

# leading tokens that mark a NON-lookup turn (affirmation / imperative-action) — D-9/INV-7
_NON_LOOKUP_LEADERS = {
    "yes", "yep", "yeah", "ok", "okay", "sure", "thanks", "thank", "thx", "ty",
    "ship", "do", "go", "fix", "run", "add", "delete", "remove", "make", "build",
    "create", "commit", "push", "merge", "send", "post", "stop", "cancel", "no",
    "nope", "yup", "sounds", "great", "perfect", "good", "nice", "cool",
}


# ---- background-review telemetry (digest BG/FG split spec) ------------------
# The prefetch QMD leg hits the daemon directly (not the `qmds` wrapper), so it writes
# NOTHING to `.qmds-usage.log` (foreground lane) by design. To make the prefetch lane
# visible in the QMD digest, it emits its OWN text-free line to a separate log, mirroring
# the wrapper's schema. INV-1: NEVER any query text — only flags/counts/latency.
_PREFETCH_USAGE_LOG = os.path.expanduser(
    os.environ.get("QMD_PREFETCH_USAGE_LOG",
                   "~/.hermes/state/qmd-corpus/.qmd-prefetch-usage.log"))
_PREFETCH_LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate (rename, NOT truncate) past 5 MiB

# The ONLY permitted shape. Any new/leaking field fails this match by construction (INV-1).
_USAGE_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?"
    r"\tmode=(?:warm|cold)\tms=\d+\tn=\d+\tqlen=\d+"
    r"\ttyped=(?:hybrid|lex|vec|hyde)\tscope=(?:[\w-]+|all)\tlane=prefetch$"
)


def _usage_suppressed() -> bool:
    """E2E/benchmark harnesses set QMD_PREFETCH_NO_USAGE_LOG=1 so synthetic traffic
    doesn't poison the organic adoption/fire-rate signal (INV-7)."""
    return str(os.environ.get("QMD_PREFETCH_NO_USAGE_LOG", "")).strip() not in ("", "0", "false", "False")


def format_usage_line(*, mode: str, ms: int, n: int, qlen: int,
                      typed: str = "hybrid", scope: str = "all") -> str:
    """Pure, text-free telemetry line (no query text — INV-1). Validated against the
    allowlist regex before return; an out-of-domain field raises (caller swallows)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mode = mode if mode in ("warm", "cold") else "warm"
    typed = typed if typed in ("hybrid", "lex", "vec", "hyde") else "hybrid"
    scope = scope if (scope == "all" or re.fullmatch(r"[\w-]+", scope or "")) else "all"
    line = (f"{ts}\tmode={mode}\tms={int(ms)}\tn={int(n)}\tqlen={int(qlen)}"
            f"\ttyped={typed}\tscope={scope}\tlane=prefetch")
    if not _USAGE_LINE_RE.match(line):
        raise ValueError("usage line failed allowlist validation")
    return line


def rotate_prefetch_log(path: Optional[str] = None, max_bytes: Optional[int] = None) -> bool:
    """Rename-rotation, called BY THE MONITOR at digest time AFTER it has read+counted the
    window (NOT in the emit hot path — rotating on write with a single `.1` loses lines when
    a window produces more than cap-worth of data; rotating after-read guarantees the live
    file + at most one `.1` cover exactly one un-read window). Rename to `.1` + fresh empty
    file; NEVER truncate (a byte-offset HWM double-counts a kept tail). Returns True if rotated."""
    target = path or _PREFETCH_USAGE_LOG
    cap = _PREFETCH_LOG_MAX_BYTES if max_bytes is None else max_bytes
    try:
        if os.path.getsize(target) < cap:
            return False
        os.replace(target, target + ".1")  # atomic; fresh file created on next append
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.debug("prefetch usage rotate skipped: %s", e)
        return False


def emit_prefetch_usage(*, mode: str, ms: int, n: int, qlen: int,
                        typed: str = "hybrid", scope: str = "all",
                        path: Optional[str] = None) -> None:
    """Append ONE text-free telemetry line. Degrade-safe + atomic: format the full line
    first, then a single append-write; ANY failure is swallowed and NEVER raises into or
    delays the prefetch path (INV-2). Suppressed by QMD_PREFETCH_NO_USAGE_LOG. Does NOT
    rotate (the monitor owns rotation, at digest-time-after-read — see rotate_prefetch_log)."""
    if _usage_suppressed():
        return
    target = path or _PREFETCH_USAGE_LOG
    try:
        line = format_usage_line(mode=mode, ms=ms, n=n, qlen=qlen, typed=typed, scope=scope)
    except Exception as e:
        logger.debug("prefetch usage format skipped: %s", e)
        return
    try:
        d = os.path.dirname(target)
        if d:
            os.makedirs(d, exist_ok=True)
        # 0600: privacy posture, defense in depth even for a text-free file.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as e:
        logger.debug("prefetch usage emit skipped: %s", e)


def load_qmd_config(raw: Optional[dict]) -> Dict[str, Any]:
    """Merge a `qmd` sub-block over the defaults. Missing block -> defaults (enabled False)."""
    cfg = dict(QMD_DEFAULTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in cfg and v is not None:
                cfg[k] = v
    return cfg


def is_lookup_intent(query: str, min_tokens: int) -> bool:
    """Pure intent gate (D-9). False for short or imperative/affirmation turns → skip QMD."""
    if not query or not query.strip():
        return False
    toks = query.strip().lower().split()
    if len(toks) < max(1, int(min_tokens)):
        return False
    first = "".join(ch for ch in toks[0] if ch.isalpha())
    if first in _NON_LOOKUP_LEADERS:
        return False
    return True


def parse_qmd_results(payload: Any, min_score: float,
                      exclude_globs: Optional[List[str]] = None, *,
                      use_rerank_score_floor: bool = False,
                      rerank_score_min: Optional[float] = None) -> List[Dict[str, Any]]:
    """Pure: extract the pointer list from a tools/call `query` response.

    GROUND-TRUTHED shape: result["structuredContent"]["results"] (top-level result keys are
    `content` + `structuredContent`; there is NO bare result["results"]). Pointers only — the
    `snippet`/`context`/`content` body fields are dropped (INV-5).
    """
    try:
        results = (payload or {}).get("result", {}).get("structuredContent", {}).get("results", [])
    except AttributeError:
        return []
    if not isinstance(results, list):
        return []
    globs = exclude_globs or []
    out: List[Dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            score = float(r.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        if use_rerank_score_floor:
            # QMD's blended `score` is positional (RRF + rerank blend). If the daemon exposes
            # pure `rerankScore`, use it as the relevance floor; if it is absent/null/non-numeric
            # (older/unpatched daemon, or rerank:false sentinel), DROP the row rather than falling
            # back to the positional score and claiming semantic gating worked.
            raw_semantic = r.get("rerankScore")
            if raw_semantic is None:
                continue
            try:
                semantic_score = float(raw_semantic)
            except (TypeError, ValueError):
                continue
            floor = float(min_score if rerank_score_min is None else rerank_score_min)
            if semantic_score < floor:
                continue
        elif score < min_score:
            continue
        f = str(r.get("file", "") or "")
        if any(fnmatch.fnmatch(f, g) for g in globs):
            continue
        out.append({
            "file": f,
            "title": str(r.get("title", "") or ""),
            "score": round(score, 4),
            "line": int(r.get("line", 0) or 0),
            "docid": str(r.get("docid", "") or ""),
        })
    return out


def render_qmd_block(hits: List[Dict[str, Any]]) -> str:
    """Pure renderer. Empty -> "" (no header, INV-7/m2). Pointers only (INV-5)."""
    if not hits:
        return ""
    lines = ["## Local Docs (QMD)"]
    for h in hits:
        pct = f"{round(float(h.get('score', 0)) * 100)}%"
        title = h.get("title") or h.get("file", "")
        line = h.get("line", 0)
        loc = f" :{line}" if line else ""
        lines.append(f"- {h.get('file','')} — {title} ({pct}){loc}")
    return "\n".join(lines)


def join_blocks(mem0_block: str, qmd_block: str) -> str:
    """Join the two recall blocks. Skip the separator when a side is empty (INV-6/m2 byte-guard)."""
    a = mem0_block or ""
    b = qmd_block or ""
    if a and b:
        return a + "\n\n" + b
    return a or b


def _extract_json(raw: str) -> Optional[Any]:
    """Parse a possibly-SSE-framed JSON body (collect `data:` lines, else the raw body)."""
    data_lines = [ln[5:].lstrip() for ln in raw.splitlines() if ln.startswith("data:")]
    body = "".join(data_lines) if data_lines else raw
    try:
        return json.loads(body)
    except Exception:
        return None


def build_qmd_query_args(query: str, *, limit: int, min_score: float,
                         collections: Optional[List[str]], rerank: bool,
                         use_rerank_score_floor: bool = False) -> Dict[str, Any]:
    """Pure request builder for QMD's `query` tool.

    Default path is byte/behavior-compatible with the original integration: no `explain`
    field and the blended QMD `minScore` is sent as-is. The semantic-floor path is explicit
    and default-off: ask QMD for `explain:true` / `rerankScore`, and do not let the blended
    positional `minScore` pre-drop candidates before the client-side pure-rerank floor runs.
    """
    args: Dict[str, Any] = {
        "searches": [{"type": "vec", "query": query}, {"type": "lex", "query": query}],
        "limit": int(limit),
        "minScore": 0.0 if use_rerank_score_floor else float(min_score),
        "rerank": bool(rerank),
    }
    if use_rerank_score_floor:
        args["explain"] = True
    if collections:
        args["collections"] = list(collections)
    return args


def qmd_query(query: str, *, limit: int, min_score: float,
              collections: Optional[List[str]], rerank: bool, deadline_s: float,
              url: str = "http://[::1]:8181/mcp",
              exclude_globs: Optional[List[str]] = None,
              use_rerank_score_floor: bool = False,
              rerank_score_min: Optional[float] = None) -> List[Dict[str, Any]]:
    """Hard-cancellable MCP `query` against the local QMD daemon. ANY failure -> [].

    A watchdog timer trips at `deadline_s` (wall-clock, whole operation incl. the 3-POST
    handshake) and `sock.shutdown(SHUT_RDWR)+close()`s the live connection, so a blocked
    read — including an SSE keepalive trickle — raises immediately instead of hanging the
    turn (INV-4). Degraded-safe: never raises, the watchdog is always cancelled (INV-3).
    """
    parsed = urlparse(url)
    host = parsed.hostname or "::1"
    port = parsed.port or 8181
    path = parsed.path or "/mcp"

    state: Dict[str, Any] = {"conn": None, "sock": None, "fired": False}
    lock = threading.Lock()

    def _trip() -> None:
        with lock:
            state["fired"] = True
            conn = state["conn"]
            sock = state.get("sock")
        # http.client can hand the raw socket to HTTPResponse.read(); closing only
        # HTTPConnection is not enough on that path. Shut down/close both handles.
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

    timer = threading.Timer(max(0.05, float(deadline_s)), _trip)
    timer.daemon = True
    timer.start()

    # background-review telemetry (single chokepoint, INV-9): time the whole op + count
    # results, emit ONE text-free line at the end regardless of success/failure.
    _t0 = time.monotonic()
    _qlen = len(query or "")
    _results: List[Dict[str, Any]] = []

    common = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Connection": "close",
    }

    def _post(body: dict, headers: dict):
        with lock:
            if state["fired"]:
                raise TimeoutError("qmd deadline tripped")
            conn = http.client.HTTPConnection(host, port, timeout=float(deadline_s))
            state["conn"] = conn
        conn.request("POST", path, body=json.dumps(body), headers=headers)
        with lock:
            state["sock"] = getattr(conn, "sock", None)
        resp = conn.getresponse()
        with lock:
            # Keep the raw socket handle alive for the watchdog while HTTPResponse.read()
            # blocks; HTTPConnection may no longer expose it once the response owns it.
            state["sock"] = getattr(resp, "fp", None) and getattr(resp.fp, "raw", None) and getattr(resp.fp.raw, "_sock", None) or state.get("sock")
        data = resp.read().decode("utf-8", "replace")
        sid = resp.getheader("mcp-session-id")
        try:
            conn.close()
        except Exception:
            pass
        return resp.status, data, sid

    try:
        _st, _d, sid = _post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "mem0-recall", "version": "1"}},
        }, common)
        if not sid:
            return []
        h2 = dict(common)
        h2["mcp-session-id"] = sid
        _post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, h2)

        args = build_qmd_query_args(
            query,
            limit=limit,
            min_score=min_score,
            collections=collections,
            rerank=rerank,
            use_rerank_score_floor=use_rerank_score_floor,
        )
        st, data, _sid = _post({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "query", "arguments": args},
        }, h2)
        if st != 200:
            return []
        payload = _extract_json(data)
        if payload is None:
            return []
        _results = parse_qmd_results(
            payload,
            min_score,
            exclude_globs,
            use_rerank_score_floor=use_rerank_score_floor,
            rerank_score_min=rerank_score_min,
        )
        return _results
    except Exception as e:  # degraded-safe: ANY failure -> [] (INV-3)
        logger.debug("qmd_query degraded: %s", e)
        return []
    finally:
        timer.cancel()
        # emit AFTER the result is in hand (latency-safe; never delays the caller's return
        # value because this runs in the same finally and is itself degrade-safe).
        try:
            _ms = int((time.monotonic() - _t0) * 1000)
            _scope = collections[0] if (collections and len(collections) == 1) else "all"
            emit_prefetch_usage(mode="warm", ms=_ms, n=len(_results),
                                qlen=_qlen, typed="hybrid", scope=_scope)
        except Exception:
            pass
