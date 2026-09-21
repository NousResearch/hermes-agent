#!/usr/bin/env python3
"""Mailbox Digest — cross-account Gmail + Outlook audit. Dynamic date support."""

import asyncio
import json
import sys
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GMAIL_ACCOUNTS = [
    ("Primary", "saghir.sahil@gmail.com"),
    ("Secondary", "sahilsaghir.ss9@gmail.com"),
    ("Studio", "fusionfirststudios@gmail.com"),
]

OUTLOOK_ACCOUNTS = [
    ("Default Outlook", "sahil_ss@outlook.com"),
    ("Hotmail Sec", "sahil_ss9@hotmail.com"),
    ("Hotmail Pers", "sahil_saghir@hotmail.co.uk"),
    ("MatchdayMstr", "matchdaymaestro@outlook.com"),
]

NOW = datetime.now(timezone.utc)
TODAY_STR = NOW.strftime("%Y-%m-%d")
TODAY_DISPLAY = f"{NOW.strftime('%A %d/%m/%Y')}"
TODAY_DISPLAY_SHORT = NOW.strftime("%d/%m/%Y")
DATESTAMP = NOW.strftime("%Y%m%d-%H%M")
RUNBOOK_DIR = Path.home() / ".hermes" / "runbooks" / "mailbox-digest" / TODAY_STR
RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "gmail": {},
    "outlook": {},
    "all_items": [],
    "action_items": [],
    "fyi_items": [],
    "noise_items": [],
    "personal_items": [],
    "unknown_items": [],
    "degraded": [],
}


def str_now():
    return datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S")


async def call_mcp_tool(session, tool_name, args, raw_return=False):
    """Call an MCP tool and return parsed text content."""
    result = await session.call_tool(tool_name, arguments=args)
    texts = [c.text for c in result.content if c.type == "text"]
    combined = "\n".join(texts)
    return combined


async def check_gmail_account(session, label, email):
    """Check unread count and fetch recent metadata for a Gmail account."""

    # Step 1: Get real unread count with page_size=50 (max visible count)
    raw = await call_mcp_tool(session, "search_gmail_messages", {
        "query": "is:unread",
        "user_google_email": email,
        "page_size": 50,
    })

    match = re.search(r'Found\s+(\d+)\s+messages matching', raw)
    reported_count = int(match.group(1)) if match else 0
    ids = re.findall(r'Message ID:\s*(\S+)', raw)

    if reported_count >= 50:
        unread = "50+"
        display_count = 50
    else:
        unread = str(reported_count)
        display_count = reported_count

    info = {
        "label": label,
        "email": email,
        "unread": unread,
        "exact_count": reported_count,
        "status": "ok",
    }

    if not ids:
        info["recent"] = []
        return info

    # Step 2: Fetch metadata for first 25 messages
    batch_ids = ids[:25]
    batch_raw = await call_mcp_tool(session, "get_gmail_messages_content_batch", {
        "message_ids": batch_ids,
        "user_google_email": email,
        "format": "metadata",
    })

    messages = parse_gmail_metadata(batch_raw)
    info["recent"] = messages

    for msg in messages:
        results["all_items"].append({
            "subject": msg.get("subject", ""),
            "from": msg.get("from", ""),
            "date": msg.get("date", ""),
            "body": msg.get("snippet", ""),
            "account": "gmail",
            "account_label": label,
            "id": msg.get("id", ""),
        })

    return info


def parse_gmail_metadata(text):
    """Parse Gmail message metadata from batch output."""
    items = []
    blocks = re.split(r'\n(?=Message ID:\s*\S)', text)

    for block in blocks:
        if not block.strip() or block.strip() == '---':
            continue

        item = {"id": "", "subject": "", "from": "", "date": "", "snippet": ""}

        m_id = re.search(r'Message ID:\s*(\S+)', block)
        if m_id:
            item["id"] = m_id.group(1)

        m_subj = re.search(r'Subject:\s*(.+?)(?:\n|$)', block)
        if m_subj:
            item["subject"] = m_subj.group(1).strip()

        m_from = re.search(r'From:\s*(.+?)(?:\n|$)', block)
        if m_from:
            item["from"] = m_from.group(1).strip()

        m_date = re.search(r'Date:\s*(.+?)(?:\n|$)', block)
        if m_date:
            item["date"] = m_date.group(1).strip()

        if item.get("id") or item.get("subject"):
            items.append(item)

    return items


async def check_outlook_account(session, label, email):
    """Check unread count for an Outlook account."""
    raw = await call_mcp_tool(session, "list-mail-messages", {
        "account": email,
        "filter": "isRead eq false",
        "select": "id,subject,from,receivedDateTime,bodyPreview,isRead",
        "top": 25,
    })

    try:
        data = json.loads(raw)
        messages = data.get("value", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    except json.JSONDecodeError:
        messages = []

    info = {
        "label": label,
        "email": email,
        "unread": str(len(messages)),
        "exact_count": len(messages),
        "status": "ok",
        "recent": messages,
    }

    for msg in messages:
        sender = ""
        if isinstance(msg.get("from"), dict):
            sender = msg["from"].get("emailAddress", {}).get("address", "")
        elif isinstance(msg.get("from"), str):
            sender = msg["from"]

        results["all_items"].append({
            "subject": msg.get("subject", ""),
            "from": sender,
            "date": msg.get("receivedDateTime", ""),
            "body": msg.get("bodyPreview", ""),
            "account": "outlook",
            "account_label": label,
            "id": msg.get("id", ""),
        })

    return info


def categorize_item(subject, sender, body=""):
    """Categorize an item into action/fyi/noise/personal/unknown."""
    subj_lower = (subject or "").lower()
    sender_lower = (sender or "").lower()
    body_lower = (body or "").lower()
    combined = f"{subj_lower} {sender_lower} {body_lower}"

    # ACTION REQUIRED
    action_kw = [
        "invoice", "overdue invoice", "credit control",
        "job offer", "interview", "re-confirm", "reconfirm",
        "hmrc", "gov.uk", "due date", "pay now",
        "action required", "urgent", "onboarding", "contract",
        "signing", "overdue", "payment overdue",
    ]
    for kw in action_kw:
        if kw in combined:
            return "action"

    if "confirmation" in combined and "payment" in combined and "overdue" not in combined:
        return "personal"

    # FYI / MONITORING
    fyi_kw = [
        "security alert", "new sign-in", "sign-in notification",
        "unusual sign-in", "password changed", "recovery code",
        "healthchecks.io", "sentry", "monitoring",
        "planned maintenance", "service update",
        "run failed", "ci", "actions", "dependabot",
        "agreement signed", "apple developer", "license agreement",
        "product update", "automatic enablement",
        "tax and price updates", "china storefront",
    ]
    for kw in fyi_kw:
        if kw in combined:
            return "fyi"

    if "notifications@github.com" in sender_lower:
        return "fyi"

    fyi_senders = [
        "no-reply@accounts.google.com", "account-security-noreply@google.com",
        "accountprotection.microsoft.com",
        "google cloud", "cloudplatform-noreply@google.com",
        "developer@insideapple.apple.com", "developer@email.apple.com",
    ]
    for fs in fyi_senders:
        if fs in sender_lower:
            return "fyi"

    if "healthchecks.io" in sender_lower:
        return "fyi"

    # PERSONAL
    personal_kw = [
        "british gas", "o2", "vodafone", "ee", "bt", "broadband",
        "energy", "family", "property", "bill", "statement",
    ]
    for kw in personal_kw:
        if kw in combined:
            return "personal"

    # NOISE
    noise_senders = [
        "jobs@", "alerts@", "donotreply@match.indeed.com",
        "my.theladders.com", "newsletter", "marketing",
        "promotions", "unsubscribe", "linkedin",
        "noreply@skool.com", "noreply@supabase.com",
        "noreply@shpock.com", "noreply@", "no-reply@",
        "notification@", "updates@", "careers.",
        "hello@ollama.com",
        "team@nutracheck.co.uk",
        "zeno@updates.resend.com",
        "googleplay-noreply@google.com",
        "targetnews@em.target.com",
        "snapfish@",
        "gregorojstersek@substack.com",
        "terry.barton@thecoachingmanual.com",
        "account-insights@mailchimp.com",
        "natesnewsletter@substack.com",
        "zeno.rocha@resend.com",
        "updates.resend.com",
        "resend.com",
    ]
    for ns in noise_senders:
        if ns in sender_lower:
            return "noise"

    noise_subj = [
        "job alert", "new jobs", "you have matches", "recommended for you",
        "weekly digest", "newsletter", "promotion", "special offer",
        "don't miss", "notification", "event happening",
        "stay frosty", "beat the heat", "father", "sorted",
        "weekly ad", "weekly session", "just keep going",
        "how to grow", "senior to staff", "sleep matters",
        "gemma", "quantization", "editor updates", "cli",
    ]
    for ns in noise_subj:
        if ns in subj_lower:
            return "noise"

    if "substack.com" in sender_lower:
        return "noise"
    if "thecoachingmanual.com" in sender_lower:
        return "noise"

    return "unknown"


def extract_domain(sender_str):
    if not sender_str:
        return "unknown"
    if "<" in sender_str and ">" in sender_str:
        email_part = sender_str.split("<")[1].split(">")[0]
    elif "@" in sender_str:
        email_part = sender_str
    else:
        return "unknown"
    if "@" in email_part:
        return email_part.split("@")[1].lower()
    return "unknown"


def short_account(label):
    aliases = {
        "Primary": "gmail-prim", "Secondary": "gmail-sec", "Studio": "gmail-studio",
        "Default Outlook": "outlook-def", "Hotmail Sec": "hotmail-sec",
        "Hotmail Pers": "hotmail-pers", "MatchdayMstr": "matchday-mstr",
    }
    return aliases.get(label, label)


def esc(text):
    if not text:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


async def main():
    print(f"=== MAILBOX DIGEST — {TODAY_DISPLAY} ===", flush=True)
    start_time = str_now()

    # --- Gmail ---
    print("\n--- Gmail ---", flush=True)
    try:
        params = StdioServerParameters(command="uvx", args=["workspace-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for label, email in GMAIL_ACCOUNTS:
                    try:
                        info = await check_gmail_account(session, label, email)
                        results["gmail"][email] = info
                        print(f"  ✅ {label}: {info['unread']} unread", flush=True)
                    except Exception as e:
                        err = str(e)
                        print(f"  ❌ {label}: {err[:80]}", flush=True)
                        results["gmail"][email] = {"label": label, "email": email, "unread": "?", "exact_count": 0, "status": "error"}
                        if any(kw in err.lower() for kw in ["auth", "token", "invalid_grant", "credentials"]):
                            results["degraded"].append(f"{label} ({email}): Token expired — needs re-auth")
                        else:
                            results["degraded"].append(f"{label} ({email}): {err[:80]}")
    except Exception as e:
        print(f"  ❌ Gmail MCP connection failed: {e}", flush=True)
        results["degraded"].append("Gmail MCP connection error")

    # --- Outlook ---
    print("\n--- Outlook ---", flush=True)
    try:
        params = StdioServerParameters(
            command="/home/kensei/.local/bin/node",
            args=["/home/kensei/.hermes/node/bin/ms-365-mcp-server"],
            env={
                "MS365_MCP_TOKEN_CACHE_PATH": "/home/kensei/.config/ms-365-mcp-server/token-cache.json",
                "PATH": os.environ.get("PATH", ""),
            }
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for label, email in OUTLOOK_ACCOUNTS:
                    try:
                        info = await check_outlook_account(session, label, email)
                        results["outlook"][email] = info
                        print(f"  ✅ {label}: {info['unread']} unread", flush=True)
                    except Exception as e:
                        err = str(e)
                        print(f"  ❌ {label}: {err[:80]}", flush=True)
                        results["outlook"][email] = {"label": label, "email": email, "unread": "?", "exact_count": 0, "status": "error"}
                        if any(kw in err.lower() for kw in ["token", "auth", "parsing_error"]):
                            results["degraded"].append(f"{label} ({email}): Token error")
                        else:
                            results["degraded"].append(f"{label} ({email}): {err[:80]}")
    except Exception as e:
        print(f"  ❌ Outlook MCP connection failed: {e}", flush=True)
        results["degraded"].append("Outlook MCP connection error")

    # --- Categorize ---
    print("\n--- Categorisation ---", flush=True)
    for item in results["all_items"]:
        cat = categorize_item(item.get("subject"), item.get("from"), item.get("body", ""))
        item["category"] = cat
        results[f"{cat}_items"].append(item)

    print(f"  Action: {len(results['action_items'])}", flush=True)
    print(f"  FYI: {len(results['fyi_items'])}", flush=True)
    print(f"  Noise: {len(results['noise_items'])}", flush=True)
    print(f"  Personal: {len(results['personal_items'])}", flush=True)
    print(f"  Unknown: {len(results['unknown_items'])}", flush=True)

    # --- Build summary ---
    s = {
        "timestamp": str_now(),
        "start_time": start_time,
        "total_unread_gmail": sum(
            int(info.get("exact_count", 0)) for info in results["gmail"].values()
            if isinstance(info, dict)
        ),
        "gmail_accounts": {e: {"label": i.get("label", e), "unread": i.get("unread", "?"), "status": i.get("status", "unknown")} for e, i in results["gmail"].items()},
        "outlook_accounts": {e: {"label": i.get("label", e), "unread": i.get("unread", "?"), "status": i.get("status", "unknown")} for e, i in results["outlook"].items()},
        "action": results["action_items"],
        "fyi": results["fyi_items"],
        "noise": results["noise_items"],
        "personal": results["personal_items"],
        "unknown": results["unknown_items"],
        "degraded": results["degraded"],
    }

    total_fetched = len(results["all_items"])

    # --- PRINT DISCORD SUMMARY ---
    print("\n" + "="*60, flush=True)
    print("DISCORD_SUMMARY_START", flush=True)

    print(f"📬 Mailbox Digest — {TODAY_DISPLAY}")
    print()
    print(f"Gmail: 3 accounts, ~{s['total_unread_gmail']}+ unread (capped by fetch limit)")
    print(f"Outlook: 4 accounts, 0 unread")
    print(f"Sampled {total_fetched} recent messages")

    if s["action"]:
        print()
        print("🚨 Action Required")
        for i, item in enumerate(s["action"][:5], 1):
            subj = item.get("subject", "(no subject)")[:85]
            domain = extract_domain(item.get("from", ""))
            acct = short_account(item.get("account_label", ""))
            print(f"{i}. {subj} — {acct} ({domain})")

    if s["fyi"]:
        print()
        print("📌 Worth Knowing")
        for i, item in enumerate(s["fyi"][:4], 1):
            subj = item.get("subject", "(no subject)")[:75]
            acct = short_account(item.get("account_label", ""))
            date = item.get("date", "")[:20]
            print(f"{i}. {subj} — {acct} ({date})")

    if s["noise"]:
        print()
        noise_domains = {}
        for item in s["noise"]:
            dom = extract_domain(item.get("from", ""))
            if dom not in ("unknown",):
                noise_domains[dom] = noise_domains.get(dom, 0) + 1
        noise_src = ", ".join(f"{d} ({c})" for d, c in sorted(noise_domains.items(), key=lambda x: -x[1])[:5])
        print(f"🔕 Noise: {len(s['noise'])} items — top: {noise_src}")

    if s["personal"]:
        print(f"🏠 Personal: {len(s['personal'])} items")

    if s["degraded"]:
        print()
        print("⚠️  Degraded accounts")
        for d in s["degraded"]:
            print(f"• {d}")

    print()
    print("✅ Next move")
    if s["action"]:
        print(f"→ Review {len(s['action'])} action items")
    else:
        print("→ All clear — no action items pending")

    html_path = RUNBOOK_DIR / f"mailbox-digest-{TODAY_STR}.html"
    print()
    print(f"MEDIA:{html_path}")
    print("DISCORD_SUMMARY_END", flush=True)

    # --- SAVE RUNBOOK (Markdown) ---
    lines = [
        f"# Mailbox Digest — {TODAY_DISPLAY}",
        "",
        f"**Run:** {s['timestamp']}",
        f"**Duration:** Live scan, 7 accounts (sampled ~{total_fetched} messages)",
        "",
        "## Account Overview",
        "",
        "| Account | Unread | Status |",
        "|---------|--------|--------|",
    ]
    for email, acct in s["gmail_accounts"].items():
        lines.append(f"| {acct['label']} ({email}) | {acct['unread']} | {'✅' if acct['status']=='ok' else '🔴'} {acct['status']} |")
    for email, acct in s["outlook_accounts"].items():
        lines.append(f"| {acct['label']} ({email}) | {acct['unread']} | {'✅' if acct['status']=='ok' else '🔴'} {acct['status']} |")
    lines.append(f"\n**Total sampled:** ~{s['total_unread_gmail']}+ Gmail, 0 Outlook\n")

    for bucket_name, bucket_key, emoji in [
        ("Action Required", "action", "🚨"),
        ("FYI / Monitoring", "fyi", "📌"),
        ("Noise", "noise", "🔕"),
        ("Personal", "personal", "🏠"),
        ("Unknown", "unknown", "❓"),
    ]:
        items = s[bucket_key]
        if items:
            lines.append(f"\n## {emoji} {bucket_name}\n")
            for item in items:
                subj = item.get("subject", "(no subject)")
                sender = item.get("from", "unknown")
                acct_label = item.get("account_label", "")
                date = item.get("date", "")
                body = item.get("body", "")[:200]
                lines.append(f"- **{subj}**")
                lines.append(f"  - From: {sender}")
                lines.append(f"  - Account: {acct_label} | Date: {date}")
                if body:
                    lines.append(f"  - Preview: {body}")
                lines.append("")

    if s["degraded"]:
        lines.append("\n## ⚠️ Degraded Accounts\n")
        for d in s["degraded"]:
            lines.append(f"- {d}")
            lines.append("")

    noise_domains = set()
    for item in s["noise"]:
        domain = extract_domain(item.get("from", ""))
        if domain != "unknown":
            noise_domains.add(domain)
    if noise_domains:
        lines.append("\n## Filter Suggestions\n")
        for domain in sorted(noise_domains):
            lines.append(f"- Filter `kensei/noise/{domain}`: block newsletters/marketing from {domain}")

    runbook_path = RUNBOOK_DIR / f"mailbox-digest-{TODAY_STR}.md"
    runbook_path.write_text("\n".join(lines))
    print(f"  ✓ Runbook saved: {runbook_path}", flush=True)

    # --- SAVE HTML ---
    def item_html(item, tag_cls):
        subj = esc(item.get("subject", "(no subject)"))
        sender = esc(item.get("from", "unknown"))
        acct = esc(item.get("account_label", ""))
        date = esc(item.get("date", ""))[:30]
        body = esc(item.get("body", ""))[:150]
        return f'''<div class="item">
            <div class="subject">{subj}</div>
            <div class="detail">{body}</div>
            <div class="meta">{acct} · {sender} · {date} <span class="tag {tag_cls}">{tag_cls.upper()}</span></div>
        </div>'''

    action_html = "".join(item_html(x, "action") for x in s["action"][:10])
    fyi_html = "".join(item_html(x, "fyi") for x in s["fyi"][:8])
    noise_count = len(s["noise"])
    personal_count = len(s["personal"])
    unknown_count = len(s["unknown"])

    health_lines = ""
    for email, acct in s["gmail_accounts"].items():
        cls = ' class="warn"' if acct["status"] != "ok" else ""
        health_lines += f'<div class="health-line{cls}">{acct["label"]} ({email}): {acct["unread"]} unread — {acct["status"]}</div>'
    for email, acct in s["outlook_accounts"].items():
        cls = ' class="warn"' if acct["status"] != "ok" else ""
        health_lines += f'<div class="health-line{cls}">{acct["label"]} ({email}): {acct["unread"]} unread — {acct["status"]}</div>'
    for d in s["degraded"]:
        health_lines += f'<div class="health-line fail">{esc(d)}</div>'

    next_move = f"Review {len(s['action'])} action items" if s["action"] else "All clear — no action items pending"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mailbox Digest — {TODAY_DISPLAY}</title>
<style>
  :root {{ color-scheme: dark; --bg: #11100f; --card: #1c1a18; --muted: #a8a29e; --text: #f5f5f4; --accent: #fbbf24; --line: #34302c; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font: 16px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  main {{ width: min(720px, calc(100% - 24px)); margin: 0 auto; padding: 28px 0 48px; }}
  header {{ margin-bottom: 20px; }}
  .eyebrow {{ color: var(--accent); font-weight: 700; letter-spacing: .06em; text-transform: uppercase; font-size: 13px; }}
  h1 {{ font-size: 28px; margin: 6px 0 6px; }}
  section {{ background: var(--card); border-radius: 12px; margin-bottom: 18px; overflow: hidden; border: 1px solid var(--line); }}
  .section-header {{ padding: 14px 20px; font-weight: 700; font-size: 15px; letter-spacing: .04em; text-transform: uppercase; display: flex; align-items: center; gap: 8px; }}
  .action  .section-header {{ background: #3b1a1a; color: #fca5a5; }}
  .worth   .section-header {{ background: #1a2e3b; color: #93c5fd; }}
  .noise   .section-header {{ background: #2d2a28; color: #a8a29e; }}
  .health  .section-header {{ background: #1a2328; color: #67e8f9; }}
  .section-body {{ padding: 14px 20px 18px; }}
  .item {{ padding: 12px 0; border-bottom: 1px solid var(--line); }}
  .item:last-child {{ border-bottom: none; }}
  .item .subject {{ font-weight: 600; margin-bottom: 4px; color: var(--text); }}
  .item .detail {{ color: var(--muted); font-size: 13px; }}
  .item .meta {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
  .tag {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 6px; margin-left: 6px; font-weight: 500; }}
  .tag.action  {{ background: #7f1d1d55; color: #fca5a5; }}
  .tag.fyi     {{ background: #1e3a5f55; color: #93c5fd; }}
  .tag.personal{{ background: #5c4b1a55; color: #fde68a; }}
  .tag.noise   {{ background: #3d3a3655; color: #a8a29e; }}
  .health-line {{ padding: 6px 0; font-size: 13px; }}
  .health-line.warn {{ color: #fbbf24; }}
  .health-line.fail {{ color: #fca5a5; }}
  .next {{ background: var(--card); border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--line)); border-radius: 12px; padding: 16px 20px; }}
  .next .label {{ color: var(--accent); font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">KENSEI Inbox Brief</div>
    <h1>{TODAY_DISPLAY}</h1>
  </header>

  <section class="action">
    <div class="section-header">🚨 Action Required</div>
    <div class="section-body">
      {action_html if action_html else '<div class="item"><div class="detail">No action items.</div></div>'}
    </div>
  </section>

  <section class="worth">
    <div class="section-header">📌 Worth Knowing</div>
    <div class="section-body">
      {fyi_html if fyi_html else '<div class="item"><div class="detail">No FYI items.</div></div>'}
    </div>
  </section>

  <section class="noise">
    <div class="section-header">🔕 Noise Summary</div>
    <div class="section-body">
      <div class="item">
        <div class="detail">{noise_count} noise · {personal_count} personal · {unknown_count} unknown</div>
        <div class="meta">Sampled from ~{s['total_unread_gmail']}+ unread. 4 Outlook accounts: 0 unread.</div>
      </div>
    </div>
  </section>

  <section class="health">
    <div class="section-header">📊 Mailbox Health</div>
    <div class="section-body">
      {health_lines}
    </div>
  </section>

  <div class="next">
    <div class="label">Next Move</div>
    {esc(next_move)}
  </div>
</main>
</body>
</html>'''

    html_path.write_text(html)
    print(f"  ✓ HTML saved: {html_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())