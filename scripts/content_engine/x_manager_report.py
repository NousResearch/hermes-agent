"""Approval review artifact generation. No network or publishing surface."""
import html
import json
import uuid
from datetime import datetime, timezone

from hermes_constants import get_hermes_home

REPORT_DIR = get_hermes_home() / 'document_cache' / 'x-manager'


def render_report(artifacts, *, lane, title):
    import x_manager as xm
    from x_delivery import prepare_delivery, expiring_report_bytes
    artifacts = [prepare_delivery(a.id) if a.pack.context.get('staged_at') else a
                 for a in artifacts]
    for artifact in artifacts:
        xm._validate_artifact(artifact)
    generated = datetime.now(timezone.utc)
    cards = []
    esc = lambda value: html.escape(str(value))
    for artifact in artifacts:
        context = artifact.pack.context
        sources = context['sources']
        rows = ''.join(
            f'<li><a href="{esc(s["url"])}">Open original</a> · {esc(s["origin"])} · '
            f'created {esc(s["created_at"])} · age '
            f'{(generated-datetime.fromisoformat(s["created_at"].replace("Z","+00:00"))).total_seconds()/3600:.2f}h</li>'
            for s in sources
        )
        grounding = context.get('grounding', {})
        references = ''.join(
            f'<li><a href="{esc(r.get("url", ""))}">Own published reference</a>: {esc(r.get("text", "")[:180])}</li>'
            for r in grounding.get('own_posts', [])
            if str(r.get('url', '')).startswith(('https://x.com/', 'https://twitter.com/'))
        )
        knowledge = (
            f'<h3>Why this angle</h3><p>{esc(context.get("selection_reason", ""))}</p><ul>{references}</ul>'
            f'<p>Relevant blog excerpts: {len(grounding.get("blog", []))}. '
            f'Repository references: {len(grounding.get("repositories", []))}. '
            f'Memory hints: {len(grounding.get("memory", []))}.</p>'
            '<p>Private knowledge informs the thought, not permission to disclose it. '
            'Memory is a hint, not proof of an outcome.</p>'
        )
        status = context.get('context_status', {})
        complete = isinstance(status, dict) and status.get('complete') is True
        context_note = ('Conversation context available as a bounded capture.' if complete else
                        'Conversation context is incomplete. Review this draft against the visible original; do not assume the full thread was read.')
        diagnostics = {
            'conversation': context.get('thread_context', []),
            'context_status': status,
            'feed_coverage': context.get('feed_coverage', 'coverage unavailable'),
            'additional_references': context.get('blog_refs', []),
        }
        detail = '<details><summary>Conversation, collection and source details</summary><pre>' + esc(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str)) + '</pre></details>'
        argument = '' if not artifact.pack.is_complete() else '<details data-section="argument"><summary>Argument and evidence</summary><pre>' + esc(json.dumps({k: getattr(artifact.pack, k) for k in xm.REQUIRED_PACK_FIELDS}, indent=2)) + '</pre></details>'
        cards.append(
            f'<article class="card"><h2>{esc(context.get("recommended_action", artifact.lane))} · pending approval</h2>'
            f'<ul>{rows}</ul><h3>Original</h3><pre>{esc(context.get("source_text", "Source text unavailable"))}</pre>'
            f'<h3>Draft</h3><pre>{esc(artifact.body)}</pre><p class="notice">{esc(context_note)}</p>'
            f'{knowledge}{detail}{argument}</article>'
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f'review-{uuid.uuid4().hex}.expiry.html'
    style = '<style>*{box-sizing:border-box}body{max-width:1000px;margin:auto;padding:16px;font:16px system-ui;background:#10141b;color:#eee;overflow-wrap:anywhere}article{border:1px solid #678;padding:24px;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#7cf}details{margin:16px 0}summary{cursor:pointer}.notice{color:#edc}</style>'
    body = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(title) + '</title>' + style + '<h1>' + esc(title) + '</h1><p>Approval only. Delivery expires six hours after the oldest source. Generated ' + generated.isoformat() + '</p>' + ''.join(cards)
    path.write_bytes(expiring_report_bytes(body, artifacts, generated=generated))
    return path
