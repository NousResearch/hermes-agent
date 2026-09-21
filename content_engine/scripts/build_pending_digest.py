#!/usr/bin/env python3
"""Build a lightweight pending-approvals STATUS/SUMMARY index for #blog-management.

Design (per Sahil's 20:03 format):
  1. This script emits ONE small summary/status HTML (title list + counts +
     per-item !approve/!reject). No inlined article bodies — keeps it tiny
     (~tens of KB) so it always delivers as a MEDIA attachment.
  2. The per-article openable previews (with embedded images) are generated
     separately by blog_approval._generate_preview() into
     ~/.hermes/reports/blog-previews/<slug>.html. The cron attaches each of
     those as its own MEDIA file. So Sahil gets one summary + one openable
     HTML per article, each with its images.

Deterministic: no LLM HTML generation, so no drift or broken path links.
Prints the absolute summary path on stdout (cron attaches it as MEDIA:).
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
TRACKER = ENGINE / "blog_topics" / "pending_approvals.jsonl"
PREVIEW_DIR = Path.home() / ".hermes" / "reports" / "blog-previews"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _blog_items() -> list[dict]:
    rows = _read_jsonl(TRACKER)
    items = []
    for e in rows:
        if e.get("status") != "pending":
            continue
        slug = e.get("slug", "")
        preview = e.get("preview_path") or ""
        items.append({
            "slug": slug,
            "title": e.get("title", slug),
            "stream": e.get("stream", e.get("tier", "ai")),
            "preview": preview,
        })
    return items


def _x_items() -> list[dict]:
    items = []
    for p in sorted(PREVIEW_DIR.glob("xarticle-*.html")):
        stem = p.stem
        parts = stem.split("-", 2)
        date_part = parts[1] if len(parts) > 1 else ""
        title_guess = parts[2].replace("-", " ").title() if len(parts) > 2 else stem
        items.append({
            "slug": stem,
            "title": f"{title_guess} ({date_part})",
            "stream": "x",
            "preview": str(p),
        })
    return items


_STREAM_CLASS = {
    "ai": "stream-ai", "builder": "stream-builder", "pm": "stream-pm", "x": "stream-ai",
}


def _row(item: dict, idx: int) -> str:
    cls = _STREAM_CLASS.get(str(item.get("stream", "")).lower(), "stream-ai")
    slug = _esc(item.get("slug", ""))
    title = _esc(item.get("title", slug))
    stream = _esc(str(item.get("stream", "")).upper() or "AI")
    has_prev = "yes" if item.get("preview") else "no"
    return f"""<tr>
  <td class="num">{idx}</td>
  <td><span class="item-title">{title}</span><br><span class="slug"><code>{slug}</code></span></td>
  <td><span class="stream-tag {cls}">{stream}</span></td>
  <td class="prev">{has_prev}</td>
  <td class="acts"><code>!approve {slug}</code><br><code>!reject {slug}</code></td>
</tr>"""


def build() -> str:
    blog = _blog_items()
    x = _x_items()
    today = date.today().strftime("%d/%m/%Y")
    ddmmyy = date.today().strftime("%d-%m-%y")

    blog_rows = "\n".join(_row(it, i + 1) for i, it in enumerate(blog)) or '<tr><td colspan="5"><em>None pending.</em></td></tr>'
    x_rows = "\n".join(_row(it, i + 1) for i, it in enumerate(x)) or '<tr><td colspan="5"><em>None pending.</em></td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en" data-color-scheme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pending Approvals — {today}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #11100f; color: #f5f5f4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; padding: 1.5rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: #fbbf24; font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #a8a29e; font-size: 0.9rem; margin-bottom: 1.25rem; }}
  h2 {{ color: #fbbf24; font-size: 1.1rem; margin: 1.5rem 0 0.5rem; padding-bottom: 0.25rem; border-bottom: 1px solid #34302c; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; color: #a8a29e; font-weight: 600; padding: 0.4rem 0.5rem; border-bottom: 1px solid #34302c; }}
  td {{ padding: 0.5rem; border-bottom: 1px solid #1f1d1b; vertical-align: top; }}
  .num {{ color: #a8a29e; width: 2rem; }}
  .item-title {{ font-weight: 600; }}
  .slug {{ font-size: 0.75rem; color: #a8a29e; }}
  .slug code, .acts code {{ background: #2c2a28; padding: 1px 5px; border-radius: 3px; font-size: 0.72rem; }}
  .acts code {{ color: #fbbf24; }}
  .prev {{ color: #86efac; }}
  .stream-tag {{ display: inline-block; font-size: 0.7rem; padding: 1px 6px; border-radius: 3px; text-transform: uppercase; }}
  .stream-ai {{ background: #3b1f1f; color: #fca5a5; }}
  .stream-builder {{ background: #1f3b2f; color: #86efac; }}
  .stream-pm {{ background: #1f2a3b; color: #93c5fd; }}
  .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #34302c; color: #a8a29e; font-size: 0.8rem; }}
  .footer code {{ background: #2c2a28; padding: 1px 5px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>📝 Pending Approvals</h1>
<p class="subtitle">{today} · {len(blog)} blog posts + {len(x)} X/Twitter articles awaiting review</p>

<h2>Blog Posts</h2>
<table>
  <thead><tr><th>#</th><th>Article</th><th>Stream</th><th>Preview</th><th>Actions</th></tr></thead>
  <tbody>
{blog_rows}
  </tbody>
</table>

<h2>X/Twitter Articles</h2>
<table>
  <thead><tr><th>#</th><th>Article</th><th>Stream</th><th>Preview</th><th>Actions</th></tr></thead>
  <tbody>
{x_rows}
  </tbody>
</table>

<div class="footer">
  Each article has its own openable HTML preview (with images) attached separately.<br>
  Approve: <code>!approve &lt;slug&gt;</code> · Reject: <code>!reject &lt;slug&gt;</code><br>
  Batch: <code>!approve all</code> or <code>!approve 1,2,3,6-10</code>
</div>
</body>
</html>"""

    out_path = PREVIEW_DIR / f"pending-digest-{ddmmyy}.html"
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = build()
    print(path)
