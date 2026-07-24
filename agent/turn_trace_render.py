"""Waterfall renderer for per-turn traces emitted by :mod:`agent.turn_trace`.

Reads the JSONL sink (one record per turn) and renders each selected trace as
an indented terminal waterfall — nesting derived from interval containment,
bars drawn over the turn timeline with sub-cell unicode blocks, model time
(``llm.call``) separated from hermes overhead. Also provides an aggregate
``--summary`` table and a self-contained interactive ``--html`` export.

Usage::

    python -m agent.turn_trace_render [--file PATH] [--last N] [--trace ID]
        [--min-ms F] [--width N] [--summary] [--html OUT] [--no-color] [--demo]

Stdlib only; pure reader — never touches the live tracing hot path.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import turn_trace

# Parent = smallest span fully containing this one, within this slack (ms).
_SLACK_MS = 1.0

# Left-anchored partial blocks for the trailing bar cell (~1/8 .. 8/8).
_BLOCKS = "▏▎▍▌▊█"

_DUR_W = 10  # "  1234.5ms"


# --- data model ---------------------------------------------------------------


class Node:
    __slots__ = ("name", "t0", "d", "tags", "thread", "parent", "children", "depth")

    def __init__(self, wire: Dict[str, Any]):
        self.name = str(wire.get("n", "?"))
        self.t0 = _f(wire.get("t0", 0.0))
        self.d = max(0.0, _f(wire.get("d", 0.0)))
        tags = wire.get("tags")
        self.tags: Dict[str, Any] = tags if isinstance(tags, dict) else {}
        self.thread = str(wire.get("th", ""))
        self.parent: Optional["Node"] = None
        self.children: List["Node"] = []
        self.depth = 0

    @property
    def end(self) -> float:
        return self.t0 + self.d

    @property
    def self_ms(self) -> float:
        # Subtract the UNION of child intervals, not their sum: concurrent
        # children overlap, and summing would zero out the parent's real
        # dispatch/collection overhead.
        covered = 0.0
        cursor = self.t0
        for c in sorted(self.children, key=lambda c: c.t0):
            lo, hi = max(c.t0, cursor), min(c.end, self.end)
            if hi > lo:
                covered += hi - lo
                cursor = hi
        return max(0.0, self.d - covered)

    def is_hot(self) -> bool:
        return self.name == "tools.delay" or bool(self.tags.get("error"))


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_tree(span_wires: List[Any]) -> List[Node]:
    """Return all nodes sorted by start, with parent/children/depth assigned.

    Parent = smallest (by duration) other span that contains this one within
    ``_SLACK_MS``; equal-duration ties resolve to the earlier-sorted span so
    identical intervals cannot parent each other cyclically.
    """
    nodes = [Node(w) for w in span_wires if isinstance(w, dict)]
    nodes.sort(key=lambda n: (n.t0, -n.d))
    for i, s in enumerate(nodes):
        best: Optional[Node] = None
        for j, p in enumerate(nodes):
            if j == i or p.d < s.d or (p.d == s.d and j > i):
                continue
            if p.t0 - _SLACK_MS <= s.t0 and s.end <= p.end + _SLACK_MS:
                if best is None or p.d < best.d:
                    best = p
        s.parent = best
    for n in nodes:
        if n.parent is not None:
            n.parent.children.append(n)
    for n in nodes:
        d, p = 0, n.parent
        while p is not None:
            d, p = d + 1, p.parent
        n.depth = d
    return nodes


def visible_rows(nodes: List[Node], min_ms: float) -> List[Node]:
    """Depth-first row order (children already sorted by start), filtered."""
    roots = [n for n in nodes if n.parent is None]
    out: List[Node] = []

    def walk(n: Node) -> None:
        if n.d >= min_ms:
            out.append(n)
        for c in n.children:
            walk(c)

    for r in roots:
        walk(r)
    return out


# --- loading -------------------------------------------------------------------


def load_traces(path: str) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and isinstance(rec.get("spans"), list):
                        traces.append(rec)
                    else:
                        raise ValueError("not a trace record")
                except Exception:
                    sys.stderr.write(f"turn_trace_render: skipped corrupt line {i} in {path}\n")
    except OSError as e:
        sys.stderr.write(f"turn_trace_render: cannot read {path}: {e}\n")
    return traces


def select_traces(traces: List[Dict[str, Any]], trace_id: Optional[str], last: int) -> List[Dict[str, Any]]:
    if trace_id:
        return [t for t in traces if str(t.get("trace_id", "")).startswith(trace_id)]
    return traces[-max(1, last):]


# --- terminal rendering ---------------------------------------------------------


class _Colors:
    def __init__(self, on: bool):
        self.model = "\x1b[34m" if on else ""
        self.over = "\x1b[32m" if on else ""
        self.hot = "\x1b[31;1m" if on else ""
        self.dim = "\x1b[2m" if on else ""
        self.bold = "\x1b[1m" if on else ""
        self.reset = "\x1b[0m" if on else ""


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _fmt_tags(tags: Dict[str, Any], limit: int = 0) -> str:
    s = " ".join(f"{k}={_fmt_val(v)}" for k, v in tags.items())
    if limit and len(s) > limit:
        s = s[: max(0, limit - 1)] + "…"
    return s


def _bar(t0: float, d: float, total: float, width: int) -> Tuple[str, str, str]:
    """(lead spaces, block chars, trailing pad) for one timeline row."""
    scale = width / max(total, 0.001)
    x0, x1 = t0 * scale, (t0 + d) * scale
    i0 = min(width - 1, max(0, int(round(x0))))
    cells = x1 - i0
    if cells <= 0:
        blocks = _BLOCKS[0]
    else:
        full = min(width - i0, int(cells))
        frac = cells - full
        blocks = _BLOCKS[-1] * full
        if frac > 0 and i0 + full < width:
            blocks += _BLOCKS[min(len(_BLOCKS) - 1, int(frac * len(_BLOCKS)))]
        if not blocks:
            blocks = _BLOCKS[0]
    pad = max(0, width - i0 - len(blocks))
    return " " * i0, blocks, " " * pad


def _trace_stats(nodes: List[Node], duration_ms: float) -> Tuple[float, float]:
    """(model_ms, overhead_ms) — model = sum of llm.call durations."""
    model = sum(n.d for n in nodes if n.name == "llm.call")
    return model, max(0.0, duration_ms - model)


def render_waterfall(rec: Dict[str, Any], min_ms: float, width: int, colors: _Colors, out=None) -> None:
    out = out or sys.stdout
    c = colors
    nodes = build_tree(rec.get("spans", []))
    total = max(_f(rec.get("duration_ms", 0.0)), max((n.end for n in nodes), default=0.0), 0.001)
    started = _f(rec.get("started_at", 0.0))
    when = datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S") if started else "?"
    tags = rec.get("tags") if isinstance(rec.get("tags"), dict) else {}

    out.write(
        f"{c.bold}── trace {rec.get('trace_id', '?')}{c.reset}  key={rec.get('key', '?')}"
        f"  {when}  total {total:.1f}ms\n"
    )
    if tags:
        out.write(f"   {c.dim}{_fmt_tags(tags)}{c.reset}\n")

    rows = visible_rows(nodes, min_ms)
    if not rows:
        out.write("   (no spans" + ("" if nodes else " recorded") + f" >= {min_ms:g}ms)\n")
    else:
        name_w = min(40, max(16, max(len("  " * r.depth + r.name) for r in rows)))
        tag_strs = [_fmt_tags(r.tags, 32) for r in rows]
        tag_w = min(32, max(len(s) for s in tag_strs))
        bar_w = max(16, width - name_w - _DUR_W - tag_w - 5)
        for r, ts in zip(rows, tag_strs):
            label = "  " * r.depth + r.name
            if len(label) > name_w:
                label = label[: name_w - 1] + "…"
            lead, blocks, pad = _bar(r.t0, r.d, total, bar_w)
            col = c.hot if r.is_hot() else (c.model if r.name == "llm.call" else c.over)
            out.write(
                f" {label:<{name_w}} {lead}{col}{blocks}{c.reset}{pad}"
                f" {r.d:>{_DUR_W - 2}.1f}ms {c.dim}{ts}{c.reset}\n"
            )

    model, over = _trace_stats(nodes, total)
    mp, op = 100.0 * model / total, 100.0 * over / total
    out.write(
        f"   total {total:.1f}ms   {c.model}model {model:.1f}ms ({mp:.1f}%){c.reset}"
        f"   {c.over}hermes overhead {over:.1f}ms ({op:.1f}%){c.reset}\n"
    )
    top = sorted((n for n in nodes if n.name != "llm.call"), key=lambda n: n.self_ms, reverse=True)[:5]
    if top:
        parts = "  ".join(f"{n.name} {n.self_ms:.1f}" for n in top)
        out.write(f"   top overhead self-time (ms): {parts}\n")
    out.write("\n")


# --- summary --------------------------------------------------------------------


def _pctl(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_name: Dict[str, Dict[str, Any]] = {}
    model_pcts: List[float] = []
    turn_ms: List[float] = []
    for rec in traces:
        nodes = build_tree(rec.get("spans", []))
        total = max(_f(rec.get("duration_ms", 0.0)), max((n.end for n in nodes), default=0.0), 0.001)
        turn_ms.append(total)
        model, _ = _trace_stats(nodes, total)
        model_pcts.append(100.0 * model / total)
        self_by_name: Dict[str, float] = {}
        for n in nodes:
            st = per_name.setdefault(n.name, {"count": 0, "durs": [], "shares": [], "self_total": 0.0})
            st["count"] += 1
            st["durs"].append(n.d)
            self_by_name[n.name] = self_by_name.get(n.name, 0.0) + n.self_ms
        for name, self_ms in self_by_name.items():
            per_name[name]["shares"].append(100.0 * self_ms / total)
            per_name[name]["self_total"] += self_ms
    return {
        "n_traces": len(traces),
        "turn_ms": turn_ms,
        "model_pcts": model_pcts,
        "per_name": per_name,
    }


def summary_rows(agg: Dict[str, Any]) -> List[Tuple[str, float, float, float, float]]:
    """(name, count/turn, p50, p95, mean self%) sorted by pooled self-time."""
    n = max(1, agg["n_traces"])
    rows = []
    for name, st in agg["per_name"].items():
        durs = sorted(st["durs"])
        shares = st["shares"]
        rows.append(
            (
                name,
                st["count"] / n,
                _pctl(durs, 0.5),
                _pctl(durs, 0.95),
                sum(shares) / len(shares) if shares else 0.0,
            )
        )
    rows.sort(key=lambda r: agg["per_name"][r[0]]["self_total"], reverse=True)
    return rows


def render_summary(traces: List[Dict[str, Any]], colors: _Colors, out=None) -> None:
    out = out or sys.stdout
    c = colors
    agg = summarize(traces)
    if not agg["n_traces"]:
        out.write("no traces selected\n")
        return
    rows = summary_rows(agg)
    name_w = max(4, max((len(r[0]) for r in rows), default=4))
    out.write(f"{c.bold}summary over {agg['n_traces']} trace(s){c.reset}\n")
    hdr = f" {'span':<{name_w}} {'n/turn':>7} {'p50 ms':>9} {'p95 ms':>9} {'self%':>7}"
    out.write(hdr + "\n")
    out.write(" " + "-" * (len(hdr) - 1) + "\n")
    for name, cnt, p50, p95, share in rows:
        out.write(f" {name:<{name_w}} {cnt:>7.1f} {p50:>9.1f} {p95:>9.1f} {share:>6.1f}%\n")
    mean_turn = sum(agg["turn_ms"]) / agg["n_traces"]
    mean_model = sum(agg["model_pcts"]) / agg["n_traces"]
    out.write(
        f"\n mean turn {mean_turn:.1f}ms   {c.model}model {mean_model:.1f}%{c.reset}"
        f"   {c.over}hermes overhead {100.0 - mean_model:.1f}%{c.reset}\n"
    )


# --- html export ------------------------------------------------------------------

_HTML_CSS = """
:root { --bg:#ffffff; --fg:#1b1f24; --muted:#5b6470; --track:#eceff3; --border:#d7dce2;
        --model:#3b82f6; --over:#10b981; --hot:#ef4444; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#101418; --fg:#e4e8ec; --muted:#8b96a3; --track:#1d232b; --border:#2c343e; }
}
body { background:var(--bg); color:var(--fg); font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
       max-width:1100px; margin:2rem auto; padding:0 1rem; }
h1 { font-size:1.15rem; } h2 { font-size:1rem; margin:1.6rem 0 .4rem; }
.meta { color:var(--muted); margin:.15rem 0 .6rem; }
.row { display:grid; grid-template-columns:280px 1fr 90px; gap:8px; align-items:center; padding:1px 0; }
.row.haskids .nm { cursor:pointer; }
.nm { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding-left:calc(var(--depth)*14px); }
.tw { display:inline-block; width:1em; color:var(--muted); }
.track { position:relative; height:12px; background:var(--track); border-radius:3px; }
.bar { position:absolute; top:0; height:100%; border-radius:2px; background:var(--over); min-width:2px; }
.bar.model { background:var(--model); } .bar.hot { background:var(--hot); }
.dur { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
.footer { color:var(--muted); margin:.4rem 0 1.2rem; }
.legend span { margin-right:1.2rem; }
.dot { display:inline-block; width:.7em; height:.7em; border-radius:2px; margin-right:.35em; vertical-align:baseline; }
table { border-collapse:collapse; margin:.6rem 0 1.5rem; }
th, td { padding:3px 12px; border-bottom:1px solid var(--border); text-align:right; }
th:first-child, td:first-child { text-align:left; }
"""

_HTML_JS = """
document.querySelectorAll('.rows').forEach(function (root) {
  var rows = Array.prototype.slice.call(root.querySelectorAll('.row'));
  function kids(id) { return rows.filter(function (r) { return r.dataset.parent === id; }); }
  function hideAll(id) {
    kids(id).forEach(function (r) { r.style.display = 'none'; hideAll(r.dataset.id); });
  }
  function showKids(id) {
    kids(id).forEach(function (r) {
      r.style.display = '';
      if (!r.classList.contains('collapsed')) showKids(r.dataset.id);
    });
  }
  rows.forEach(function (r) {
    if (!r.classList.contains('haskids')) return;
    r.addEventListener('click', function () {
      var closed = r.classList.toggle('collapsed');
      r.querySelector('.tw').textContent = closed ? '\\u25b8' : '\\u25be';
      if (closed) hideAll(r.dataset.id); else showKids(r.dataset.id);
    });
  });
});
"""


def render_html(traces: List[Dict[str, Any]], min_ms: float, path: str) -> None:
    e = _html.escape
    parts: List[str] = []
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("<h1>hermes turn traces</h1>")
    parts.append(
        '<p class="legend"><span><i class="dot" style="background:var(--model)"></i>model (llm.call)</span>'
        '<span><i class="dot" style="background:var(--over)"></i>hermes overhead</span>'
        '<span><i class="dot" style="background:var(--hot)"></i>tools.delay / error</span>'
        "<span>click a row to collapse its children</span></p>"
    )
    for rec in traces:
        nodes = build_tree(rec.get("spans", []))
        total = max(_f(rec.get("duration_ms", 0.0)), max((n.end for n in nodes), default=0.0), 0.001)
        started = _f(rec.get("started_at", 0.0))
        when = datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S") if started else "?"
        tags = rec.get("tags") if isinstance(rec.get("tags"), dict) else {}
        model, over = _trace_stats(nodes, total)
        parts.append(f"<h2>trace {e(str(rec.get('trace_id', '?')))} &mdash; key={e(str(rec.get('key', '?')))}</h2>")
        parts.append(f'<div class="meta">{e(when)} &middot; total {total:.1f}ms &middot; {e(_fmt_tags(tags))}</div>')
        rows = visible_rows(nodes, min_ms)
        ids = {id(n): str(i) for i, n in enumerate(rows)}
        parts.append('<div class="rows">')
        for n in rows:
            rid = ids[id(n)]
            pid = ids.get(id(n.parent), "") if n.parent is not None else ""
            has_kids = any(c.d >= min_ms for c in n.children)
            cls = "model" if n.name == "llm.call" else ("hot" if n.is_hot() else "")
            left = 100.0 * n.t0 / total
            w = max(0.05, 100.0 * n.d / total)
            tip = f"{n.name}  {n.d:.1f}ms  @{n.t0:.1f}ms"
            if n.tags:
                tip += "\n" + "\n".join(f"{k}={_fmt_val(v)}" for k, v in n.tags.items())
            tw = "▾" if has_kids else "&nbsp;"
            parts.append(
                f'<div class="row{" haskids" if has_kids else ""}" data-id="{rid}" data-parent="{pid}"'
                f' style="--depth:{n.depth}" title="{e(tip)}">'
                f'<span class="nm"><span class="tw">{tw}</span>{e(n.name)}</span>'
                f'<span class="track"><span class="bar {cls}" style="left:{left:.3f}%;width:{w:.3f}%"></span></span>'
                f'<span class="dur">{n.d:.1f}ms</span></div>'
            )
        parts.append("</div>")
        mp = 100.0 * model / total
        parts.append(
            f'<div class="footer">model {model:.1f}ms ({mp:.1f}%) &middot; '
            f"hermes overhead {over:.1f}ms ({100.0 - mp:.1f}%)</div>"
        )
    agg = summarize(traces)
    if agg["n_traces"]:
        parts.append(f"<h2>summary over {agg['n_traces']} trace(s)</h2>")
        parts.append("<table><tr><th>span</th><th>n/turn</th><th>p50 ms</th><th>p95 ms</th><th>self%</th></tr>")
        for name, cnt, p50, p95, share in summary_rows(agg):
            parts.append(
                f"<tr><td>{e(name)}</td><td>{cnt:.1f}</td><td>{p50:.1f}</td>"
                f"<td>{p95:.1f}</td><td>{share:.1f}%</td></tr>"
            )
        parts.append("</table>")
        mean_model = sum(agg["model_pcts"]) / agg["n_traces"]
        parts.append(
            f'<div class="footer">mean split: model {mean_model:.1f}% / overhead {100.0 - mean_model:.1f}%</div>'
        )
    parts.append(f"<script>{_HTML_JS}</script>")
    doc = "<!doctype html><meta charset='utf-8'><title>turn traces</title>" + "".join(parts)
    with open(os.path.expanduser(path), "w", encoding="utf-8") as fh:
        fh.write(doc)


# --- demo data --------------------------------------------------------------------


def demo_traces(n: int = 3, seed: int = 7) -> List[Dict[str, Any]]:
    """Synthetic-but-realistic traces so the renderer is testable without data."""
    rng = random.Random(seed)
    out = []
    base = time.time() - 3600.0
    for k in range(n):
        spans: List[Dict[str, Any]] = []

        def add(name: str, t0: float, d: float, **tags: Any) -> None:
            w: Dict[str, Any] = {"n": name, "t0": round(t0, 2), "d": round(d, 2), "th": "demo"}
            if tags:
                w["tags"] = tags
            spans.append(w)

        # gateway ingress (~400ms)
        add("gateway.session_resolve", 4, 38)
        add("gateway.transcript_load", 46, rng.uniform(90, 150))
        add("gateway.hygiene", 205, 32)
        add("gateway.agent_setup", 242, rng.uniform(110, 150), rebuild=(k == 0))
        add("gateway.ingest", 0, 400, platform="telegram", session_key=f"demo:{k}")

        # prologue (~180ms)
        t = 402.0
        add("prologue.system_prompt", t + 2, 55, rebuilt=(k == 0))
        add("prologue.persist_early", t + 60, 26)
        add("prologue.compression_preflight", t + 88, 9, compressed=False)
        add("prologue.pre_llm_hook", t + 99, 11)
        add("prologue.memory_prefetch", t + 112, 62, decision="skip")
        add("turn.prologue", t, 180)
        t += 184

        tool_calls = 0
        for i in range(1, 4):
            it0 = t
            ca = rng.uniform(18, 55)
            add("iteration.context_assemble", t, ca)
            t += ca + 2
            add("iteration.request_setup", t, 8)
            t += 10
            add("llm.client_create", t, 3)
            t += 4
            llm = rng.uniform(2000, 6000)
            add(
                "llm.call", t, llm,
                api_request_id=f"req_{k}{i}{rng.randrange(16**6):06x}",
                ttft_ms=round(rng.uniform(400, 1200)),
                cache_pct=round(rng.uniform(40, 95), 1),
                model="claude-opus-4",
            )
            t += llm + 1
            add("llm.accounting", t, 5)
            t += 7
            if i < 3:
                b0 = t
                ncalls = rng.choice((1, 2))
                for _ in range(ncalls):
                    ex = rng.uniform(120, 700)
                    add(
                        "tools.call", t, ex + 40, tool=rng.choice(("terminal", "web_search", "read_file")),
                        checkpoint_ms=round(rng.uniform(2, 15), 1), execute_ms=round(ex, 1),
                        flush_ms=round(rng.uniform(5, 25), 1),
                    )
                    t += ex + 42
                    tool_calls += 1
                add("tools.delay", t, 1000)
                t += 1001
                add("tools.batch", b0, t - b0 - 1, count=ncalls, mode="sequential")
            add("iteration", it0, t - it0, i=i)
            t += 2
        add("turn.verify_gate", t, 12, nudge_fired=False)
        t += 14

        f0 = t
        add("finalize.trajectory", t + 2, 80)
        add("finalize.resource_cleanup", t + 85, 28)
        add("finalize.persist", t + 116, 120)
        add("finalize.post_hooks", t + 240, 18)
        add("finalize.memory_dispatch", t + 260, 38)
        add("turn.finalize", f0, 300)
        t = f0 + 302
        add(
            "turn", 400, t - 400,
            turn_id=f"turn_{k}", model="claude-opus-4", iterations=3,
            tool_calls=tool_calls, exit_reason="end_turn",
        )
        add("gateway.persist", t + 2, 250)
        add("transport.delivery", t + 256, rng.uniform(40, 90))
        end = t + 350

        out.append(
            {
                "schema": 1,
                "trace_id": "".join(rng.choice("0123456789abcdef") for _ in range(12)),
                "key": f"demo:{k}",
                "started_at": round(base + k * 120.0, 3),
                "duration_ms": round(end, 2),
                "tags": {
                    "platform": "telegram", "model": "claude-opus-4", "iterations": 3,
                    "tool_calls": tool_calls, "exit_reason": "end_turn",
                },
                "spans": sorted(spans, key=lambda s: s["t0"]),
            }
        )
    return out


# --- cli ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent.turn_trace_render", description=__doc__.split("\n")[0])
    ap.add_argument("--file", default=None, help="trace jsonl (default: the turn_trace sink)")
    ap.add_argument("--last", type=int, default=1, help="render the last N traces (default 1)")
    ap.add_argument("--trace", default=None, help="select by trace_id (prefix match)")
    ap.add_argument("--min-ms", type=float, default=1.0, help="hide spans shorter than this (default 1.0)")
    ap.add_argument("--width", type=int, default=0, help="output width (default: terminal width)")
    ap.add_argument("--summary", action="store_true", help="aggregate table instead of waterfalls")
    ap.add_argument("--html", default=None, metavar="PATH", help="write a self-contained interactive HTML file")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI color")
    ap.add_argument("--demo", action="store_true", help="render embedded synthetic traces (no sink needed)")
    args = ap.parse_args(argv)

    if args.demo:
        traces = demo_traces()
    else:
        traces = load_traces(os.path.expanduser(args.file) if args.file else turn_trace.sink_path())
    selected = select_traces(traces, args.trace, args.last)
    if not selected:
        sys.stderr.write("turn_trace_render: no matching traces\n")
        return 1

    width = args.width or shutil.get_terminal_size((120, 24)).columns
    color_on = not args.no_color and "NO_COLOR" not in os.environ and sys.stdout.isatty()
    colors = _Colors(color_on)

    if args.html:
        render_html(selected, args.min_ms, args.html)
        print(f"wrote {args.html} ({len(selected)} trace(s))")
    if args.summary:
        render_summary(selected, colors)
    elif not args.html:
        for rec in selected:
            render_waterfall(rec, args.min_ms, width, colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
