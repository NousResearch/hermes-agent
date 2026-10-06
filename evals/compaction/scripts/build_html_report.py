#!/usr/bin/env python3
"""Build a self-contained HTML report comparing compaction runs.

Usage: build_report.py <runs_dir> <out_html>
Expects runs/<checkout>_<session>.json pairs from run_compaction.py.
"""
import html
import json
import sys
from pathlib import Path

RUNS = Path(sys.argv[1])
OUT = sys.argv[2]

pairs = {}
for f in sorted(RUNS.glob("*.json")):
    co, sid = f.stem.split("_", 1)
    if co == "main":
        co, sid = "main-co", f.stem[len("main-co_"):]
    elif co == "pr":
        co, sid = "pr-co", f.stem[len("pr-co_"):]
    data = json.loads(f.read_text(encoding="utf-8"))
    pairs.setdefault(sid, {})[co] = data

E = html.escape

def msg_class(m):
    role = m.get("role", "?")
    c = m.get("content") or ""
    if isinstance(c, str):
        if "[CONTEXT COMPACTION" in c or "[CONTEXT SUMMARY" in c:
            return "summary"
        if "SKILL_PRUNED" in c:
            return "skillpruned"
        if "SKILL POLICY DIGEST" in c or "SKILL_POLICY_DIGEST" in c:
            return "digest"
        if "preserved across context compression" in c:
            return "todosnap"
    return role

def render_msg(m, idx):
    role = m.get("role", "?")
    c = m.get("content")
    if not isinstance(c, str):
        c = json.dumps(c, default=str)[:2000]
    tool = m.get("tool_name") or ""
    tcs = m.get("tool_calls") or []
    tc_names = ", ".join(
        (t.get("function", {}) or {}).get("name", "?") for t in tcs if isinstance(t, dict)
    )
    cls = msg_class(m)
    nchars = len(c)
    label = role
    if tool:
        label += f" · {tool}"
    if tc_names:
        label += f" → {tc_names}"
    preview = c[:180].replace("\n", " ")
    full = c if nchars <= 20000 else c[:20000] + f"\n…[{nchars-20000:,} more chars]"
    return (
        f'<details class="msg {cls}"><summary><span class="idx">#{idx}</span>'
        f'<span class="role">{E(label)}</span>'
        f'<span class="chars">{nchars:,}ch</span>'
        f'<span class="preview">{E(preview)}</span></summary>'
        f"<pre>{E(full)}</pre></details>"
    )

def render_column(title, data, key):
    meta = data["meta"]
    msgs = data[key]
    body = "".join(render_msg(m, i) for i, m in enumerate(msgs))
    return (
        f'<div class="col"><div class="colhead"><h3>{E(title)}</h3>'
        f'<div class="stats">{meta[key.replace("before","before_msgs").replace("after","after_msgs")] if False else len(msgs)} msgs · '
        f'~{(meta["before_tokens_est"] if key=="before" else meta["after_tokens_est"]):,} tok</div></div>'
        f'<div class="msgs">{body}</div></div>'
    )

def survival_stats(before, after):
    after_texts = set()
    for m in after:
        c = m.get("content")
        if isinstance(c, str) and c:
            after_texts.add(c[:400])
    kept = sum(1 for m in before if isinstance(m.get("content"), str) and (m.get("content") or "")[:400] in after_texts)
    return kept

sections = []
toc = []
for sid, versions in pairs.items():
    if "main-co" not in versions or "pr-co" not in versions:
        continue
    main_d, pr_d = versions["main-co"], versions["pr-co"]
    title = main_d["meta"].get("title") or sid
    mm, pm = main_d["meta"], pr_d["meta"]

    def count_markers(msgs, needle):
        return sum((m.get("content") or "").count(needle) for m in msgs if isinstance(m.get("content"), str))

    rows = []
    def stat(name, mv, pv):
        cls = "diff" if mv != pv else ""
        rows.append(f"<tr class='{cls}'><td>{E(name)}</td><td>{E(str(mv))}</td><td>{E(str(pv))}</td></tr>")

    stat("Messages after", mm["after_msgs"], pm["after_msgs"])
    stat("Est. tokens after", f"{mm['after_tokens_est']:,}", f"{pm['after_tokens_est']:,}")
    stat("Reduction", f"{100-100*mm['after_tokens_est']//max(1,mm['before_tokens_est'])}%", f"{100-100*pm['after_tokens_est']//max(1,pm['before_tokens_est'])}%")
    stat("Compress time", f"{mm['elapsed_s']}s", f"{pm['elapsed_s']}s")
    stat("SKILL_PRUNED markers", count_markers(main_d["after"], "SKILL_PRUNED"), count_markers(pr_d["after"], "SKILL_PRUNED"))
    stat("Policy digest blocks", count_markers(main_d["after"], "SKILL POLICY DIGEST") + count_markers(main_d["after"], "SKILL_POLICY_DIGEST"), count_markers(pr_d["after"], "SKILL POLICY DIGEST") + count_markers(pr_d["after"], "SKILL_POLICY_DIGEST"))
    stat("Todo snapshot present", "yes" if count_markers(main_d["after"], "preserved across context compression") else "no", "yes" if count_markers(pr_d["after"], "preserved across context compression") else "no")
    stat("Kept-verbatim msgs", survival_stats(main_d["before"], main_d["after"]), survival_stats(pr_d["before"], pr_d["after"]))
    stat("Summary error", mm.get("summary_error") or "—", pm.get("summary_error") or "—")

    todo_html = ""
    for label, d in (("main", mm), ("PR #87090", pm)):
        tb = d.get("todo_injection_block")
        if tb:
            todo_html += f"<h4>Todo injection block — {E(label)}</h4><pre class='todoblock'>{E(tb)}</pre>"

    anchor = f"s-{sid}"
    toc.append(f'<a href="#{anchor}">{E(title)} <span class="dim">({sid})</span></a>')
    sections.append(f"""
<section id="{anchor}">
<h2>{E(title)} <span class="dim">{sid}</span></h2>
<table class="stats-table"><tr><th></th><th>main</th><th>PR #87090</th></tr>{"".join(rows)}</table>
{todo_html}
<div class="cols">
{render_column("BEFORE (original transcript)", main_d, "before")}
{render_column("AFTER — main", main_d, "after")}
{render_column("AFTER — PR #87090", pr_d, "after")}
</div>
</section>""")

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Compaction comparison — main vs PR #87090</title>
<style>
:root {{ color-scheme: dark; }}
body {{ background:#0d1117; color:#c9d1d9; font:14px/1.45 -apple-system,Segoe UI,sans-serif; margin:0; padding:24px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:18px; border-bottom:1px solid #30363d; padding-bottom:6px; margin-top:48px; }}
.dim {{ color:#8b949e; font-weight:normal; font-size:12px; }}
nav a {{ display:block; color:#58a6ff; margin:2px 0; text-decoration:none; }}
.legend span {{ display:inline-block; padding:2px 10px; margin-right:8px; border-radius:4px; font-size:12px; }}
.stats-table {{ border-collapse:collapse; margin:12px 0; }}
.stats-table td, .stats-table th {{ border:1px solid #30363d; padding:4px 12px; text-align:left; font-size:13px; }}
.stats-table tr.diff td {{ background:#1c2a1c; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }}
.col {{ min-width:0; }}
.colhead {{ position:sticky; top:0; background:#161b22; padding:8px; border:1px solid #30363d; border-radius:6px 6px 0 0; z-index:2; }}
.colhead h3 {{ margin:0; font-size:13px; }} .colhead .stats {{ color:#8b949e; font-size:12px; }}
.msgs {{ border:1px solid #30363d; border-top:none; max-height:80vh; overflow-y:auto; }}
.msg {{ border-bottom:1px solid #21262d; }}
.msg summary {{ cursor:pointer; padding:3px 6px; display:flex; gap:6px; align-items:baseline; white-space:nowrap; overflow:hidden; }}
.msg summary::-webkit-details-marker {{ display:none; }}
.idx {{ color:#484f58; font-size:11px; min-width:34px; }}
.role {{ font-size:11px; font-weight:600; min-width:110px; overflow:hidden; text-overflow:ellipsis; }}
.chars {{ color:#8b949e; font-size:11px; min-width:52px; }}
.preview {{ color:#8b949e; font-size:11px; overflow:hidden; text-overflow:ellipsis; flex:1; }}
.msg pre {{ white-space:pre-wrap; word-break:break-word; font-size:11px; background:#161b22; margin:0; padding:8px; max-height:400px; overflow-y:auto; }}
.msg.user summary {{ background:#0d2137; }} .msg.user .role {{ color:#58a6ff; }}
.msg.assistant .role {{ color:#d2a8ff; }}
.msg.tool .role {{ color:#7ee787; }}
.msg.system summary {{ background:#21262d; }} .msg.system .role {{ color:#8b949e; }}
.msg.summary summary {{ background:#3d2e00; }} .msg.summary .role {{ color:#e3b341; }}
.msg.skillpruned summary {{ background:#3d1418; }} .msg.skillpruned .role {{ color:#ff7b72; }}
.msg.digest summary {{ background:#1b3d2e; }} .msg.digest .role {{ color:#56d364; }}
.msg.todosnap summary {{ background:#2d1b3d; }} .msg.todosnap .role {{ color:#d2a8ff; }}
.todoblock {{ background:#1b1230; border:1px solid #6e40c9; padding:10px; white-space:pre-wrap; font-size:12px; }}
</style></head><body>
<h1>Compaction comparison — current main (7619564fb) vs PR #87090 (41fd511f6)</h1>
<p class="dim">Real sessions from state.db (copy), replayed through each checkout's ContextCompressor with force=True. Real LLM summaries. Click any row to expand the full message.</p>
<div class="legend">
<span style="background:#3d2e00;color:#e3b341">compaction summary</span>
<span style="background:#3d1418;color:#ff7b72">SKILL_PRUNED marker</span>
<span style="background:#1b3d2e;color:#56d364">policy digest</span>
<span style="background:#2d1b3d;color:#d2a8ff">todo snapshot</span>
<span style="background:#0d2137;color:#58a6ff">user</span>
</div>
<nav>{"".join(toc)}</nav>
{"".join(sections)}
</body></html>"""

Path(OUT).write_text(page, encoding="utf-8")
print(f"wrote {OUT} ({len(page):,} bytes, {len(sections)} sessions)")
