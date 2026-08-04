#!/usr/bin/env python3
"""Single tabbed HTML review bundle for Discord #blog-management approvals.

Collapses the old "one openable HTML per article" flow (which silently dropped
most attachments at scale) into ONE self-contained HTML: a left sidebar
(SUMMARY + per-platform article list) and one pane per article, every image
base64-embedded. If the combined file would exceed SIZE_CAP it auto-splits into
one file per platform. Prints the exact MEDIA: line(s) to stdout so the cron
echoes them verbatim — no LLM path-listing.

Design deliberately reproduces the validation-*.html look Sahil confirmed
"PERFECT" (Inter / #111 / #ffd166), kept here as a committed constant so the
approved aesthetic is deterministic and can't drift.

Output: writes into ~/.hermes/reports/blog-previews/ (an allowlisted MEDIA
delivery root). Prints a summary line + MEDIA: lines, or [SILENT] when nothing
is pending.
"""
from __future__ import annotations

import json
import re
from datetime import date
from html import escape as _esc
from pathlib import Path

# Reused low-level helpers (import only — do not modify the source modules).
from blog.preview import md_to_html, _image_to_data_uri, parse_frontmatter, _read_mdx
from scripts.build_xarticle_previews import embed_image_resized
import database

ENGINE = Path(__file__).resolve().parent.parent
TRACKER = ENGINE / "blog_topics" / "pending_approvals.jsonl"
X_BUNDLES = ENGINE / "output" / "articles"
PREVIEW_DIR = Path.home() / ".hermes" / "reports" / "blog-previews"

# Stay safely under Discord's upload limit (a prior 13.6 MB combined file was
# rejected). Multi-file delivery is unreliable in this system (attachments drop
# silently past the first), so we do everything possible to ship ONE file:
# images are compressed progressively until the whole bundle fits under this
# cap. Only if even the most aggressive level overflows do we split.
SIZE_CAP = 7_500_000

# (max_width_px, jpeg_quality) tried in order — first that fits wins. The user
# wants an exact preview, so we start at full fidelity and only step down as
# far as needed to keep it a single downloadable file.
COMPRESSION_LEVELS = [(800, 80), (680, 72), (560, 64), (460, 55)]

BLOG_GROUP = "SAHILSBLOG"
X_GROUP = "X/TWITTER"
LINKEDIN_GROUP = "LINKEDIN"


# ── Design: the approved validation-*.html palette + minimal layout CSS ──────

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif;
       background: #111; color: #eee; display: flex; min-height: 100vh; }
a { color: #ffd166; }

/* sidebar */
.sidebar { width: 300px; flex: none; background: #161616; border-right: 1px solid #333;
           padding: 20px 14px; position: sticky; top: 0; height: 100vh; overflow-y: auto; }
.sidebar-head { display: flex; align-items: center; justify-content: space-between;
                gap: 8px; margin-bottom: 12px; }
.brand { color: #ffd166; font-weight: 700; font-size: 18px; }
.collapse-btn, .reopen-btn { background: #1b1b1b; border: 1px solid #333; color: #ffd166;
    border-radius: 8px; padding: 4px 10px; cursor: pointer; font-size: 15px; line-height: 1; flex: none; }
.collapse-btn:hover, .reopen-btn:hover { background: #262626; }
.reopen-btn { position: fixed; top: 14px; left: 14px; z-index: 50; display: none; }
body.nav-collapsed .sidebar { display: none; }
body.nav-collapsed .reopen-btn { display: block; }
body.nav-collapsed .content { padding-left: 60px; }
.group { color: #f5c84c; font-size: 12px; letter-spacing: 1px; text-transform: uppercase;
         margin: 18px 0 6px; opacity: .85; }
.group.muted { opacity: .4; }
.group .hint { display: block; text-transform: none; letter-spacing: 0; color: #888;
               font-size: 11px; margin-top: 2px; }
.navlink { display: block; color: #ddd; text-decoration: none; padding: 8px 10px;
           border-radius: 10px; cursor: pointer; font-size: 14px; line-height: 1.35; margin: 2px 0; }
.navlink:hover { background: #1b1b1b; }
.navlink.active { background: #1b1b1b; color: #ffd166; box-shadow: inset 3px 0 0 #ffd166; }

/* content */
.content { flex: 1; min-width: 0; padding: 28px; overflow-x: hidden; }
.pane { display: none; } .pane.active { display: block; }
.wrap { max-width: 1180px; margin: auto; }

/* validation article aesthetic */
h1 { font-size: 34px; line-height: 1.05; margin: 0 0 8px; }
.deck { color: #bbb; font-size: 17px; line-height: 1.45; }
.source { margin: 12px 0 20px; color: #aaa; font-size: 13px; word-break: break-all; }
dl { display: grid; grid-template-columns: 130px 1fr; gap: 8px 14px; background: #181818;
     border: 1px solid #333; border-radius: 16px; padding: 16px; margin: 18px 0; }
dt { color: #f5c84c; } dd { margin: 0; color: #ddd; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 20px 0; }
.card { background: #1b1b1b; border: 1px solid #333; border-radius: 18px; padding: 16px; }
.card.missing { border-color: #a33; }
.card h2 { margin: 0 0 12px; color: #f5c84c; font-size: 15px; }
.card img { width: 100%; border-radius: 12px; border: 1px solid #333; background: #000; }
.card p { font-size: 12px; color: #999; word-break: break-all; margin-top: 8px; }

.actions { margin: 14px 0 4px; }
.actions code { background: #1b1b1b; border: 1px solid #333; color: #ffd166; padding: 4px 10px;
                border-radius: 8px; font-size: 13px; margin-right: 8px; display: inline-block; }

/* rendered markdown body */
.body { margin-top: 22px; font-size: 16px; line-height: 1.7; color: #e6e6e6; }
.body h1 { font-size: 26px; margin: 26px 0 10px; }
.body h2 { font-size: 21px; color: #f5c84c; margin: 24px 0 10px; }
.body h3 { font-size: 17px; color: #ffd166; margin: 18px 0 8px; }
.body p { margin: 12px 0; }
.body ul, .body ol { margin: 12px 0 12px 22px; }
.body li { margin: 5px 0; }
.body a { color: #ffd166; }
.body code { background: #1b1b1b; border: 1px solid #333; color: #f0883e; padding: 1px 6px;
             border-radius: 5px; font-size: 14px; }
.body pre { background: #1b1b1b; border: 1px solid #333; border-radius: 12px; padding: 14px;
            overflow-x: auto; margin: 14px 0; }
.body pre code { border: 0; padding: 0; background: none; }
.body blockquote { border-left: 3px solid #f5c84c; padding-left: 14px; color: #bbb; margin: 14px 0; }
.body img { width: 100%; border-radius: 12px; border: 1px solid #333; margin: 16px 0; }

/* summary table */
table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 18px 0; }
th { text-align: left; color: #aaa; font-weight: 600; padding: 8px; border-bottom: 1px solid #333; }
td { padding: 8px; border-bottom: 1px solid #222; vertical-align: top; }
td.num { color: #777; width: 2.5rem; }
.pill { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 20px;
        background: #1b1b1b; border: 1px solid #333; color: #f5c84c; }
td code { background: #1b1b1b; border: 1px solid #333; color: #ffd166; padding: 1px 6px;
          border-radius: 5px; font-size: 12px; }
.slug { color: #888; font-size: 12px; }
.rowtitle { color: #eee; text-decoration: none; cursor: pointer; }
.rowtitle:hover { color: #ffd166; }
"""

_JS = """
function show(n){
  document.querySelectorAll('.pane').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.navlink').forEach(function(a){a.classList.remove('active');});
  var pane=document.getElementById('pane-'+n); if(pane){pane.classList.add('active');}
  var link=document.querySelector('.navlink[data-pane="'+n+'"]'); if(link){link.classList.add('active');}
  window.scrollTo(0,0);
}
function toggleNav(){ document.body.classList.toggle('nav-collapsed'); }
"""


# ── Small render helpers ─────────────────────────────────────────────────────

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


def _meta_dl(**fields) -> str:
    rows = []
    for key, val in fields.items():
        if isinstance(val, (list, tuple)):
            val = ", ".join(str(v) for v in val) if val else "—"
        val = str(val) if val not in (None, "") else "—"
        rows.append(f"<dt>{_esc(key)}</dt><dd>{_esc(val)}</dd>")
    return f"<dl>{''.join(rows)}</dl>"


def _actions(slug: str) -> str:
    return (
        f'<div class="actions">'
        f'<code>!approve {_esc(slug)}</code>'
        f'<code>!reject {_esc(slug)}</code>'
        f"</div>"
    )


def _x_markdown_to_html(md: str, bundle: Path, max_width: int, quality: int) -> str:
    """X-article markdown → HTML with images embedded at the given compression.

    Mirrors scripts.build_xarticle_previews.markdown_to_html but routes image
    embedding through embed_image_resized(max_width, quality) so the whole
    bundle can be shrunk to fit a single file.
    """
    def replace_img(m):
        img_path = bundle / "imgs" / m.group(2)
        uri = embed_image_resized(img_path, max_width=max_width, quality=quality)
        if uri:
            return f'<img src="{uri}" alt="{_esc(m.group(1))}" />'
        return '<div class="card missing" style="padding:20px;text-align:center;color:#c66;">[IMAGE MISSING]</div>'

    md = re.sub(r"!\[([^\]]*)\]\(imgs/([^)]+)\)", replace_img, md)

    out, in_list = [], False
    for line in md.split("\n"):
        if line.startswith("# "):
            out.append(f"<h1>{_esc(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{_esc(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{_esc(line[4:])}</h3>")
        elif line.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
            if line.strip():
                out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _img_card(label: str, uri: str, meta: str) -> str:
    if uri.startswith("data:"):
        cls, inner = "card", f'<img src="{uri}" alt="{_esc(label)}" />'
    else:
        cls = "card missing"
        inner = '<div style="padding:28px;text-align:center;color:#c66;">[IMAGE MISSING]</div>'
    return (
        f'<section class="{cls}"><h2>{_esc(label)}</h2>{inner}'
        f"<p>{_esc(meta)}</p></section>"
    )


# ── Gather + render articles ─────────────────────────────────────────────────

def _blog_items(max_width: int = 800, quality: int = 80) -> list[dict]:
    """Render a pane per pending blog post from pending_approvals.jsonl."""
    items = []
    for entry in _read_jsonl(TRACKER):
        if entry.get("status") != "pending":
            continue
        items.append(_render_blog_pane(entry, max_width, quality))
    return items


def _render_blog_pane(entry: dict, max_width: int = 800, quality: int = 80) -> dict:
    slug = entry.get("slug", "")
    mdx = entry.get("mdx_path", "")
    title = entry.get("title", slug)
    stream = entry.get("stream", entry.get("tier", "ai"))
    tier = entry.get("tier", "ai")
    description = pub_date = ""
    hero_src = ""
    section_images: list[str] = []
    tags: list[str] = []
    body_md = ""

    p = Path(mdx) if mdx else None
    if p and p.exists():
        fm, body = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        description = fm.get("description", "")
        pub_date = str(fm.get("pubDate", ""))
        hero_src = fm.get("heroImage", "")
        tier = fm.get("tier", tier)
        stream = "builder" if fm.get("tier") == "builder" else fm.get("tier", stream)
        tags_raw = fm.get("tags", "[]")
        try:
            tags = (
                json.loads(tags_raw)
                if tags_raw.startswith("[")
                else [t.strip().strip("\"'") for t in tags_raw.strip("[]").split(",") if t.strip()]
            )
        except Exception:
            tags = []
        section_images = [m[1] for m in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", body)[:6]]
        body_md = body
    else:
        post = _read_mdx(slug)
        if post:
            description = post["description"]
            pub_date = post["pub_date"]
            hero_src = post["hero_src"]
            section_images = post["section_images"]
            tags = post["tags"]
            stream = post["stream"]
            tier = post["tier"]
            body_md = post["body_md"]
            title = post["title"]

    body_html = md_to_html(body_md)[0] if body_md else "<p><em>Body unavailable.</em></p>"

    cards = []
    if hero_src:
        uri, meta = _image_to_data_uri(hero_src, max_width=max_width, jpeg_quality=quality)
        cards.append(_img_card("Hero image", uri, meta))
    for i, src in enumerate(section_images):
        if not str(src).strip():
            continue
        uri, meta = _image_to_data_uri(str(src).strip(), max_width=max_width, jpeg_quality=quality)
        cards.append(_img_card(f"Section {i + 1}", uri, meta))
    grid = f'<div class="grid">{"".join(cards)}</div>' if cards else ""

    source = (
        f'<p class="source">MDX: <a href="file://{_esc(mdx)}">{_esc(mdx or slug)}</a></p>'
        if mdx
        else ""
    )
    deck = f'<p class="deck">{_esc(description)}</p>' if description else ""
    meta_dl = _meta_dl(Stream=str(stream).upper(), Tier=tier, Date=pub_date, Slug=slug, Tags=tags)

    pane = (
        f"<h1>{_esc(title)}</h1>{deck}{source}{meta_dl}{_actions(slug)}"
        f"{grid}"
        f'<div class="body">{body_html}</div>'
    )
    return {"slug": slug, "title": title, "pane": pane, "group": BLOG_GROUP}


def _article_items(records: list[dict], max_width: int = 800, quality: int = 80) -> tuple[dict[str, list[dict]], list[str]]:
    """Render recorded pending articles and diagnose broken durable pointers."""
    groups = {X_GROUP: [], LINKEDIN_GROUP: []}
    diagnostics: list[str] = []
    for record in records:
        bundle = Path(record["bundle_path"])
        article_path = bundle / "article.md"
        if not article_path.is_file():
            diagnostics.append(f"missing bundle for article {record['article_id']}: {bundle}")
            continue
        platform = str(record.get("platform", "")).lower()
        if platform in {"twitter", "x"}:
            group = X_GROUP
        elif platform == "linkedin":
            group = LINKEDIN_GROUP
        else:
            diagnostics.append(f"unsupported platform for article {record['article_id']}: {platform or 'unknown'}")
            continue
        groups[group].append(_render_article_pane(record, bundle, group, max_width, quality))
    return groups, diagnostics


def _pending_article_items(max_width: int = 800, quality: int = 80) -> tuple[dict[str, list[dict]], list[str]]:
    database.init_db()
    migration = database.migrate_article_approvals(X_BUNDLES)
    groups, diagnostics = _article_items(
        database.list_article_approvals(status="pending"), max_width, quality
    )
    diagnostics.extend(
        f"article draft has no matching bundle: {article_id}"
        for article_id in migration["missing_bundles"]
    )
    diagnostics.extend(
        f"article bundle has no matching DB record: {bundle_path}"
        for bundle_path in migration["orphan_bundles"]
    )
    return groups, diagnostics


def _render_article_pane(record: dict, bundle: Path, group: str, max_width: int = 800, quality: int = 80) -> dict:
    slug = record["article_id"]
    md = (bundle / "article.md").read_text(encoding="utf-8", errors="replace")
    first_line = md.split("\n", 1)[0]
    title = first_line.replace("# ", "").strip() or slug
    body_html = _x_markdown_to_html(md, bundle, max_width, quality)
    # markdown_to_html renders the leading "# Title" as an <h1>; drop it since we
    # show the title in the pane header already.
    body_html = re.sub(r"^\s*<h1>.*?</h1>", "", body_html, count=1, flags=re.S)

    m = re.match(r"(\d{4}-\d{2}-\d{2})", slug)
    pub_date = m.group(1) if m else ""
    meta_dl = _meta_dl(Platform=group, Brand=record.get("brand", ""), Date=pub_date, ID=slug)

    pane = (
        f"<h1>{_esc(title)}</h1>{meta_dl}{_actions(slug)}"
        f'<div class="body">{body_html}</div>'
    )
    return {"slug": slug, "title": title, "pane": pane, "group": group}


# ── Shell assembly ───────────────────────────────────────────────────────────

def _summary_pane(sections: list[tuple[str, list[dict]]], indexed: list[tuple[int, dict]]) -> str:
    today = date.today().strftime("%d/%m/%Y")
    counts = {label: len(items) for label, items in sections}
    nblog = counts.get(BLOG_GROUP, 0)
    nx = counts.get(X_GROUP, 0)
    nlinkedin = counts.get(LINKEDIN_GROUP, 0)
    total = nblog + nx + nlinkedin

    rows = []
    for idx, it in indexed:
        group = {BLOG_GROUP: "Blog", X_GROUP: "X/Twitter", LINKEDIN_GROUP: "LinkedIn"}[it["group"]]
        rows.append(
            f"<tr><td class='num'>{idx}</td>"
            f"<td><a class='rowtitle' onclick='show({idx})'>{_esc(it['title'])}</a>"
            f"<br><span class='slug'>{_esc(it['slug'])}</span></td>"
            f"<td><span class='pill'>{_esc(group)}</span></td>"
            f"<td><code>!approve {_esc(it['slug'])}</code><br><code>!reject {_esc(it['slug'])}</code></td></tr>"
        )
    table = (
        "<table><thead><tr><th>#</th><th>Article</th><th>Platform</th><th>Actions</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=4><em>None pending.</em></td></tr>'}</tbody></table>"
    )
    return (
        f"<h1>Pending Review</h1>"
        f'<p class="deck">{today} · {nblog} blog posts + {nx} X/Twitter articles + {nlinkedin} LinkedIn articles awaiting review '
        f"(total {total})</p>"
        f"{_meta_dl(Blog=nblog, **{'X/Twitter': nx, 'LinkedIn': nlinkedin}, Total=total)}"
        f"{table}"
        f'<p class="source">Approve: <code>!approve &lt;slug&gt;</code> · '
        f"Reject: <code>!reject &lt;slug&gt;</code> · Batch: <code>!approve all</code></p>"
    )


def render(sections: list[tuple[str, list[dict]]], doc_title: str) -> str:
    """Compose one self-contained tabbed document from the given sections."""
    indexed: list[tuple[int, dict]] = []
    idx = 0
    for _label, items in sections:
        for it in items:
            idx += 1
            indexed.append((idx, it))

    nav = ['<div class="sidebar-head"><span class="brand">📝 Pending Review</span>'
           '<button class="collapse-btn" onclick="toggleNav()" title="Hide sidebar">◀</button></div>',
           '<a class="navlink active" data-pane="0" onclick="show(0)">SUMMARY</a>']
    counter = 0
    for label, items in sections:
        muted = " muted" if not items else ""
        hint = ""
        nav.append(f'<div class="group{muted}">{_esc(label)}{hint}</div>')
        for it in items:
            counter += 1
            nav.append(
                f'<a class="navlink" data-pane="{counter}" onclick="show({counter})">{_esc(it["title"])}</a>'
            )

    summary_html = _summary_pane(sections, indexed)
    panes = [f'<section class="pane active" id="pane-0"><div class="wrap">{summary_html}</div></section>']
    for i, it in indexed:
        panes.append(f'<section class="pane" id="pane-{i}"><div class="wrap">{it["pane"]}</div></section>')

    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        f"<title>{_esc(doc_title)}</title><style>{_CSS}</style></head><body>"
        f'<nav class="sidebar">{"".join(nav)}</nav>'
        '<button class="reopen-btn" onclick="toggleNav()" title="Show sidebar">☰</button>'
        f'<main class="content">{"".join(panes)}</main>'
        f"<script>{_JS}</script></body></html>"
    )


def _write(name: str, html_doc: str) -> str:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = PREVIEW_DIR / name
    path.write_text(html_doc, encoding="utf-8")
    return str(path)


def _chunk_by_size(items: list[dict]) -> list[list[dict]]:
    """Greedily pack panes so each rendered file stays under SIZE_CAP.

    Measures each pane's byte size once; the shell (CSS/JS/nav) is small and
    covered by the headroom below.
    """
    budget = SIZE_CAP - 40_000  # headroom for shell + nav + summary table
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_bytes = 0
    for it in items:
        size = len(it["pane"].encode("utf-8"))
        if cur and cur_bytes + size > budget:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(it)
        cur_bytes += size
    if cur:
        chunks.append(cur)
    return chunks or [[]]


def build() -> list[str]:
    """Build the bundle and return the written file path(s).

    Strategy: prefer ONE file (what the user asked for, and the only reliably
    delivered shape). Try each compression level until the combined document
    fits under SIZE_CAP. Only if the most aggressive level still overflows do we
    fall back to per-platform, size-chunked files.
    """
    ddmmyy = date.today().strftime("%d-%m-%y")

    blog0 = _blog_items(*COMPRESSION_LEVELS[0])
    article_groups0, diagnostics = _pending_article_items(*COMPRESSION_LEVELS[0])
    x0, linkedin0 = article_groups0[X_GROUP], article_groups0[LINKEDIN_GROUP]
    if diagnostics:
        raise RuntimeError("article approval state mismatch: " + "; ".join(diagnostics))
    if not blog0 and not x0 and not linkedin0:
        return []

    last_blog, last_x, last_linkedin = blog0, x0, linkedin0
    for i, (width, quality) in enumerate(COMPRESSION_LEVELS):
        blog = blog0 if i == 0 else _blog_items(width, quality)
        article_groups, diagnostics = (article_groups0, diagnostics) if i == 0 else _pending_article_items(width, quality)
        if diagnostics:
            raise RuntimeError("article approval state mismatch: " + "; ".join(diagnostics))
        x, linkedin = article_groups[X_GROUP], article_groups[LINKEDIN_GROUP]
        last_blog, last_x, last_linkedin = blog, x, linkedin
        combined = render(
            [(BLOG_GROUP, blog), (X_GROUP, x), (LINKEDIN_GROUP, linkedin)],
            "Pending Review",
        )
        if len(combined.encode("utf-8")) <= SIZE_CAP:
            return [_write(f"pending-review-{ddmmyy}.html", combined)]

    # Even the most aggressive level overflows — split per platform, then chunk
    # any platform still too big so no single file exceeds the cap.
    outputs = []
    for group, slug, items in ((BLOG_GROUP, "blog", last_blog), (X_GROUP, "x", last_x), (LINKEDIN_GROUP, "linkedin", last_linkedin)):
        if not items:
            continue
        chunks = _chunk_by_size(items)
        multi = len(chunks) > 1
        for n, chunk in enumerate(chunks, 1):
            suffix = f"-{n}" if multi else ""
            part = f" (part {n}/{len(chunks)})" if multi else ""
            doc = render([(group, chunk)], f"Pending Review — {group}{part}")
            outputs.append(_write(f"pending-review-{slug}{suffix}-{ddmmyy}.html", doc))
    return outputs


def main() -> None:
    blog_n = len([e for e in _read_jsonl(TRACKER) if e.get("status") == "pending"])
    outputs = build()
    pending_articles = database.list_article_approvals(status="pending")
    x_n = sum(str(row.get("platform", "")).lower() in {"x", "twitter"} for row in pending_articles)
    linkedin_n = sum(str(row.get("platform", "")).lower() == "linkedin" for row in pending_articles)
    if not outputs:
        print("[SILENT]")
        return
    print(
        f"{blog_n} blog posts + {x_n} X/Twitter articles + {linkedin_n} LinkedIn articles awaiting review "
        f"(total {blog_n + x_n + linkedin_n})"
    )
    for path in outputs:
        print(f"MEDIA:{path}")


if __name__ == "__main__":
    main()
