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
    cards=[]
    esc=lambda value: html.escape(str(value))
    for artifact in artifacts:
        context=artifact.pack.context
        sources=context['sources']
        rows=''.join(f'<li><a href="{esc(s["url"])}">{esc(s["id"])}</a> · {esc(s["origin"])} · created {esc(s["created_at"])} · age {(generated-datetime.fromisoformat(s["created_at"].replace("Z","+00:00"))).total_seconds()/3600:.2f}h</li>' for s in sources)
        grounding = context.get('grounding', {})
        references = ''.join(f'<li><a href="{esc(r.get("url", ""))}">Own published reference</a>: {esc(r.get("text", "")[:180])}</li>' for r in grounding.get('own_posts', []) if str(r.get('url', '')).startswith(('https://x.com/', 'https://twitter.com/')))
        knowledge = f'<h3>Why this angle</h3><ul>{references}</ul><p>Relevant blog excerpts: {len(grounding.get("blog", []))}. Repository references: {len(grounding.get("repositories", []))}. Memory hints: {len(grounding.get("memory", []))}.</p><p>Private knowledge informs the draft; it is not permission to disclose private details. Memory is not proof of an outcome.</p>'
        argument = '' if not artifact.pack.is_complete() else f'<h3>Argument</h3><pre>{esc(json.dumps({k:getattr(artifact.pack,k) for k in xm.REQUIRED_PACK_FIELDS},indent=2))}</pre>'
        cards.append(f'<article><h2>{esc(context.get("recommended_action",artifact.lane))} · pending approval</h2><ul>{rows}</ul><h3>Original</h3><pre>{esc(context.get("source_text","Source text unavailable"))}</pre><h3>Draft</h3><pre>{esc(artifact.body)}</pre><h3>Conversation context</h3><pre>{esc(context.get("thread_context",[]))}</pre><p>{esc(context.get("context_status","source only; full thread unavailable"))}</p><h3>Feed coverage</h3><pre>{esc(context.get("feed_coverage","coverage unavailable"))}</pre>{knowledge}{argument}<pre>{esc(context.get("blog_refs",[]))}</pre></article>')
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    path=REPORT_DIR/f'review-{uuid.uuid4().hex}.expiry.html'
    body = '<!doctype html><meta charset="utf-8"><title>'+esc(title)+'</title><style>body{max-width:1000px;margin:auto;font:16px system-ui;background:#10141b;color:#eee}article{border:1px solid #678;padding:24px;margin:24px 0}pre{white-space:pre-wrap}a{color:#7cf}</style><h1>'+esc(title)+'</h1><p>Approval only. Delivery expires six hours after the oldest source. Generated '+generated.isoformat()+'</p>'+''.join(cards)
    path.write_bytes(expiring_report_bytes(body, artifacts, generated=generated))
    return path
