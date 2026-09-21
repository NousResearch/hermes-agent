"""Telegram digest delivery — v2 summary + HTML attachment.

Replaces per-card delivery with a single summary message and a detailed HTML
file showing all drafts numbered for easy approval/rejection by reference number.
"""
import json as _json
import os
import requests
from datetime import datetime
from typing import List, Optional

# Default to Topic 22 "Content & Social" in Kensei Workspace
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CONTENT_CHAT_ID", "-1003922682700")
TELEGRAM_TOPIC_ID = os.getenv("TELEGRAM_CONTENT_TOPIC_ID", "22")


def _send_payload(endpoint: str, payload: dict) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=30)
        return r.status_code == 200
    except Exception:
        return False


def _base_payload() -> dict:
    base = {"chat_id": TELEGRAM_CHAT_ID, "parse_mode": "HTML"}
    if TELEGRAM_TOPIC_ID:
        base["message_thread_id"] = int(TELEGRAM_TOPIC_ID)
    return base


def send_message(text: str) -> bool:
    payload = _base_payload()
    payload["text"] = text
    return _send_payload("sendMessage", payload)


def send_document(doc_path: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        payload = _base_payload()
        payload["caption"] = caption
        payload["parse_mode"] = "HTML"
        with open(doc_path, "rb") as f:
            r = requests.post(url, data=payload, files={"document": f}, timeout=60)
        return r.status_code == 200
    except Exception:
        return False


def _brand_emoji(brand: str) -> str:
    mapping = {
        "matchdaymaestro": "⚽",
        "plenishd": "🍽️",
        "coachos": "📋",
        "sahil_linkedin": "💼",
        "sahil_twitter": "🧵",
    }
    return mapping.get(brand.lower().replace(" ", "_"), "📝")


def _content_type_icon(content_type: str) -> str:
    mapping = {
        "text": "📝",
        "text+image": "🖼️",
        "voice": "🎙️",
        "video": "🎬",
        "text+video": "📹",
    }
    return mapping.get(content_type, "📝")


def generate_draft_html(drafts: List[dict], title: str = "Content Engine Drafts") -> str:
    """Generate a detailed HTML digest showing all drafts numbered for approval."""
    today = datetime.now().strftime("%a %d %b %Y")

    # Brand counts
    brand_counts = {}
    for d in drafts:
        b = d.get("brand", "unknown").replace("_", " ").title()
        brand_counts[b] = brand_counts.get(b, 0) + 1

    # Content type counts
    ct_counts = {}
    for d in drafts:
        ct = d.get("content_type", "text")
        ct_counts[ct] = ct_counts.get(ct, 0) + 1

    brand_meta = " · ".join(
        f'{_brand_emoji(b)} {b}: {c}' for b, c in sorted(brand_counts.items())
    )
    ct_meta = " · ".join(
        f'{_content_type_icon(ct)} {ct}: {c}' for ct, c in sorted(ct_counts.items())
    )

    lines = []
    lines.append("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { color-scheme: dark; --bg: #11100f; --card: #1c1a18; --muted: #a8a29e; --text: #f5f5f4; --accent: #fbbf24; --line: #34302c; --green: #22c55e; --red: #ef4444; --blue: #3b82f6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.6 system-ui, sans-serif; }
main { max-width: 800px; margin: 0 auto; padding: 24px 16px 48px; }
header { margin-bottom: 24px; }
.eyebrow { color: var(--accent); font-weight: 700; letter-spacing: .06em; text-transform: uppercase; font-size: 12px; }
h1 { font-size: 24px; margin: 6px 0 6px; }
.summary { color: var(--muted); font-size: 13px; }
.meta { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
.meta span { font-size: 12px; color: var(--muted); }
.draft { border: 1px solid var(--line); border-radius: 12px; margin-bottom: 12px; overflow: hidden; background: var(--card); }
.draft-header { padding: 12px 16px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.draft-num { font-weight: 700; color: var(--accent); min-width: 28px; }
.brand-tag { font-weight: 600; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 500; }
.badge-platform { background: #1e3a5f; color: #93c5fd; }
.badge-type { background: #3d3a36; color: #a8a29e; }
.badge-pillar { background: #1a2e1a; color: #86efac; }
.draft-body { padding: 16px; white-space: pre-wrap; font-size: 14px; line-height: 1.6; }
.draft-id { color: var(--muted); font-family: monospace; font-size: 11px; }
.visual { padding: 12px 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; font-style: italic; }
.instruction { background: color-mix(in srgb, var(--accent) 15%, var(--bg)); border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--line)); border-radius: 12px; padding: 16px; margin-bottom: 20px; }
.instruction h2 { margin: 0 0 8px; font-size: 14px; }
.instruction p { margin: 0; color: var(--muted); font-size: 13px; }
code { background: #2a2826; padding: 1px 6px; border-radius: 3px; font-size: 12px; color: var(--accent); }
</style></head><body>
<main>
<header>
<div class="eyebrow">KENSEI Content Engine — Drafts for Review</div>
""")

    lines.append(f'<h1>{title}</h1>')
    lines.append(f'<div class="summary">{today} · {len(drafts)} draft(s) pending across {len(brand_counts)} brand(s)</div>')
    lines.append('<div class="meta">')
    lines.append(f'<span>Brands: {brand_meta}</span>')
    lines.append(f'<span>Types: {ct_meta}</span>')
    lines.append("</div></header>")

    # Approval instructions
    lines.append('<div class="instruction">')
    lines.append('<h2>How to approve or reject</h2>')
    lines.append("<p>Reply in Telegram with your decision by reference number:</p>")
    lines.append("<p><code>A:1,2,3,6-28</code> = Approve drafts 1,2,3 and 6 through 28</p>")
    lines.append(f"<p><code>R:4,5,11,29</code> = Reject drafts 4,5,11,29</p>")
    lines.append(f"<p><code>A:all</code> = Approve all {len(drafts)} drafts</p>")
    lines.append(f'<p>Or write a longer message like: <code>All approved except 1, 4, 5, 11, 29. Number 4 needs a different angle.</code></p>')
    lines.append("</div>")

    # Draft cards
    for i, d in enumerate(drafts, 1):
        brand = d.get("brand", "").replace("_", " ").title()
        plat = d.get("platform", "twitter")
        ct = d.get("content_type", "text")
        pillar = d.get("pillar", "")
        body = d.get("body_text", "")
        visual = d.get("visual_description", "")
        visual_path = d.get("visual_path", "")
        draft_id = d["id"]

        body_html = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        visual_bits = []
        if visual:
            visual_bits.append(visual)
        if visual_path:
            visual_bits.append(f"Free preview: {visual_path}")
        visual_text = " · ".join(visual_bits)
        visual_html = visual_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if visual_text else ""

        lines.append('<div class="draft">')
        lines.append('<div class="draft-header">')
        lines.append(f'<span class="draft-num">{i}</span>')
        lines.append(f'<span class="brand-tag">{_brand_emoji(d.get("brand", ""))} {brand}</span>')
        lines.append(f'<span class="badge badge-platform">{plat}</span>')
        lines.append(f'<span class="badge badge-type">{_content_type_icon(ct)} {ct}</span>')
        lines.append(f'<span class="badge badge-pillar">{pillar.replace("_", " ").title()}</span>')
        lines.append(f'<span class="draft-id">{draft_id}</span>')
        lines.append("</div>")
        lines.append(f'<div class="draft-body">{body_html}</div>')
        if visual_html:
            lines.append(f'<div class="visual">{visual_html}</div>')
        lines.append("</div>")

    lines.append("</main></body></html>")
    return "\n".join(lines)


def deliver_digest(drafts: List[dict]) -> bool:
    """Send a single summary message + HTML attachment to Telegram.

    Follows the cron-output-contract:
    - One summary message (headline + count + signal)
    - One HTML file attached via MEDIA: tag
    - Full detail in HTML, numbered for easy approval/rejection
    """
    if not drafts:
        return False

    today = datetime.now().strftime("%a %d %b · %H:%M")
    total = len(drafts)

    # Brand counts
    brand_counts = {}
    for d in drafts:
        b = d.get("brand", "unknown").replace("_", " ").title()
        brand_counts[b] = brand_counts.get(b, 0) + 1

    # Content type counts
    ct_counts = {}
    for d in drafts:
        ct = d.get("content_type", "text")
        ct_counts[ct] = ct_counts.get(ct, 0) + 1

    brand_summary = " · ".join(
        f"{_brand_emoji(b.lower().replace(' ', '_'))} {b}: {c}"
        for b, c in sorted(brand_counts.items())
    )

    ct_summary = " · ".join(f"{_content_type_icon(ct)} {ct}: {c}" for ct, c in sorted(ct_counts.items()))

    # Save HTML file first
    html = generate_draft_html(drafts)
    out_dir = "/home/kensei/.hermes/runbooks/content-digest"
    os.makedirs(out_dir, exist_ok=True)
    html_path = f"{out_dir}/drafts-{datetime.now().strftime('%Y%m%d-%H%M')}.html"
    
    with open(html_path, 'w', encoding="utf-8") as f:
        f.write(html)

    # Send summary message
    message = (
        f"📝 <b>Content drafts</b> · {today}\n"
        f"{total} drafts · {brand_summary}\n"
        f"\n"
        f"<b>Content types</b>\n"
        f"• {ct_summary}\n"
        f"\n"
        f"<b>Full detail</b>\n"
        f"<code>{html_path}</code>\n"
        f"\n"
        f"Approve: <code>A:1,2,3,6-28</code> or <code>A:all</code>\n"
        f"Reject: <code>R:4,5,11,29</code> or write a note"
    )

    # Try to attach HTML as document
    doc_sent = send_document(html_path, caption="KENSEI Content Drafts — numbered for approval/rejection")

    # Also attach via MEDIA tag for Hermes gateway file handling
    send_message(message)

    # If document delivered via API, also send MEDIA line via Hermes routing
    if doc_sent:
        media_msg = f"MEDIA:{html_path}"
        send_message(media_msg)
    
    return True
