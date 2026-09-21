#!/usr/bin/env python3
"""Minimal self-contained HTML reports for approval-only LinkedIn manager runs.

One clean, selectable HTML attachment per run, showing each recommendation
alongside its original source/evidence and (for the PM insight) a direct link
to the full post. Mirrors the proven X manager report pattern but is a
separate, LinkedIn-only renderer: it never imports the X manager or the blog
pipeline, and it never posts anywhere.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

REPORT_DIR = Path('/home/kensei/.hermes/document_cache/li-manager')
CARD_ACCENTS = ('#0ea5e9', '#f59e0b', '#a78bfa', '#34d399', '#fb7185')

KIND_LABELS = {
    'pm_insight': 'PM Insight summary',
    'ai_pm_post': 'AI/PM post',
    'user_material': 'User-material post',
}


def _e(value: object) -> str:
    return html.escape(str(value or ''))


def _paragraphs(value: str) -> str:
    chunks = [c.strip() for c in (value or '').split('\n\n') if c.strip()]
    return ''.join(f'<p>{_e(c)}</p>' for c in chunks)


def render_report(artifacts: list, *, lane: str, title: str) -> Path:
    """Write one clean report containing every recommendation and its source."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = REPORT_DIR / f'{lane}-{stamp}.html'

    cards = []
    for index, artifact in enumerate(artifacts, 1):
        accent = CARD_ACCENTS[(index - 1) % len(CARD_ACCENTS)]
        context = artifact.pack.context or {}
        source_text = (
            context.get('source_text')
            or context.get('source')
            or context.get('evidence_source')
            or 'Source text unavailable'
        )
        source_url = artifact.source_url or context.get('source_url') or ''
        author = context.get('author') or context.get('signal_type') or 'Source'
        kind_label = KIND_LABELS.get(artifact.kind, artifact.kind)
        source_link = (
            f'<a class="source-link" href="{_e(source_url)}" target="_blank" rel="noopener">Open original ↗</a>'
            if source_url else ''
        )
        cards.append(f'''<article class="card" id="recommendation-{index}" style="--card-accent:{accent}">
  <div class="accent-bar"></div>
  <header class="card-head">
    <span class="number">{index:02d}</span>
    <div><div class="eyebrow">{_e(kind_label)} · {_e(author)}</div><h2>Recommendation {index}</h2></div>
    <span class="status">Pending</span>
  </header>
  <div class="comparison">
  <section class="source">
    <div class="section-label">Original / evidence</div>
    <blockquote>{_e(source_text)}</blockquote>
    {source_link}
  </section>
  <section class="recommendation">
    <div class="section-label">Recommended draft</div>
    <div class="draft">{_paragraphs(artifact.body)}</div>
  </section>
  </div>
  <details>
    <summary>Why this angle</summary>
    <dl>
      <dt>Claim</dt><dd>{_e(artifact.pack.claim)}</dd>
      <dt>Evidence</dt><dd>{_e(artifact.pack.evidence)}</dd>
      <dt>Mechanism</dt><dd>{_e(artifact.pack.mechanism)}</dd>
      <dt>Position</dt><dd>{_e(artifact.pack.position)}</dd>
    </dl>
  </details>
  <footer><code>{_e(artifact.id)}</code></footer>
</article>''')

    generated = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    index_links = ''.join(
        f'<a href="#recommendation-{i}" style="--chip-accent:{CARD_ACCENTS[(i - 1) % len(CARD_ACCENTS)]}">{i:02d}</a>'
        for i in range(1, len(artifacts) + 1)
    )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>
<style>
:root{{color-scheme:dark;--bg:#080a0d;--panel:#111419;--panel2:#171b22;--text:#f4f5f7;--muted:#929ba8;--line:#303641;--accent:#38bdf8;--green:#86efac}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{width:min(1080px,calc(100% - 32px));margin:0 auto;padding:48px 0 72px}} .top{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:22px;padding-bottom:22px;border-bottom:1px solid var(--line)}}
h1{{font-size:clamp(28px,5vw,46px);line-height:1.05;margin:6px 0 0;letter-spacing:-.04em}} .meta{{color:var(--muted);text-align:right;white-space:nowrap}} .eyebrow,.section-label{{color:var(--accent);font:700 11px/1.2 ui-monospace,SFMono-Regular,monospace;text-transform:uppercase;letter-spacing:.12em}}
.jump{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 28px}} .jump a{{width:40px;height:34px;display:grid;place-items:center;text-decoration:none;color:#e9edf3;background:color-mix(in srgb,var(--chip-accent) 14%,#12151a);border:1px solid color-mix(in srgb,var(--chip-accent) 55%,#303641);border-radius:8px;font:700 12px ui-monospace,monospace}}
.grid{{display:grid;gap:30px}} .card{{position:relative;background:var(--panel);border:1px solid color-mix(in srgb,var(--card-accent) 38%,var(--line));border-radius:14px;overflow:hidden;box-shadow:0 18px 54px rgba(0,0,0,.26)}} .accent-bar{{height:4px;background:var(--card-accent)}} .card-head{{display:grid;grid-template-columns:auto 1fr auto;gap:16px;align-items:center;padding:20px 22px;border-bottom:1px solid var(--line);background:linear-gradient(90deg,color-mix(in srgb,var(--card-accent) 11%,var(--panel)),var(--panel) 62%)}}
.number{{font:800 13px ui-monospace,monospace;color:var(--card-accent)}} h2{{font-size:19px;margin:3px 0 0;letter-spacing:-.02em}} .card .eyebrow,.card .section-label{{color:var(--card-accent)}} .status{{color:var(--green);background:#102519;border:1px solid #21462e;border-radius:999px;padding:5px 9px;font-size:11px;font-weight:700}}
.comparison{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}} section{{padding:24px}} .source{{background:color-mix(in srgb,var(--card-accent) 7%,var(--panel2));border-right:1px solid var(--line)}} .recommendation{{background:#0e1116}} blockquote{{margin:12px 0 16px;padding-left:16px;border-left:3px solid var(--card-accent);font-size:16px;color:#d8dde5;white-space:pre-wrap}} .source-link{{color:var(--card-accent);text-decoration:none;font-size:13px;font-weight:650}}
.draft{{font-size:17px;line-height:1.72;color:#f6f7f9}} .draft p{{margin:10px 0}} details{{border-top:1px solid var(--line);padding:16px 22px;color:var(--muted);background:#0c0f13}} summary{{cursor:pointer;color:#c4cad3;font-weight:650}} dl{{display:grid;grid-template-columns:90px 1fr;gap:8px 16px;margin:16px 0 0}} dt{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} dd{{margin:0;color:#cbd1da}} footer{{padding:12px 22px;border-top:1px solid var(--line);color:#68717e;font-size:11px;background:#0a0d11}}
@media(max-width:760px){{main{{width:min(100% - 20px,1080px);padding-top:24px}}.top{{display:block}}.meta{{text-align:left;margin-top:10px}}.card-head{{grid-template-columns:auto 1fr}}.status{{grid-column:2}}.comparison{{grid-template-columns:1fr}}.source{{border-right:0;border-bottom:1px solid var(--line)}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header class="top"><div><div class="eyebrow">LinkedIn manager · Approval review</div><h1>{_e(title)}</h1></div><div class="meta">{len(artifacts)} recommendation{'s' if len(artifacts) != 1 else ''}<br>{generated}</div></header>
<nav class="jump" aria-label="Jump to recommendation">{index_links}</nav>
<div class="grid">{''.join(cards)}</div>
</main></body></html>'''
    path.write_text(document, encoding='utf-8')
    return path
