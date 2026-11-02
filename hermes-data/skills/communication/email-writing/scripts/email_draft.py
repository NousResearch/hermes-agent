#!/usr/bin/env python3
"""email_draft.py — the ONLY sanctioned way to put an email into Gmail Drafts.

Takes a Markdown body and produces a multipart/alternative message
(text/plain WITHOUT hard wraps + styled text/html), then creates or updates a
Gmail draft. Never sends.

Why this exists: hand-writing a body string is what produced the
"poem format" (a newline every 4-7 words) and the plain-text emails NDR
rejected. The renderer makes hard-wrapping structurally impossible: every
paragraph in the plain part is ONE line, and the HTML part carries the
formatting.

Usage
  python3 email_draft.py --md body.md --to "A <a@x.com>" --subject "..." \
      [--cc "B <b@x.com>"] [--bcc ""] [--style standard|full] [--no-html] \
      [--thread THREAD_ID] [--in-reply-to MSGID] [--references "MSGID MSGID2"] \
      [--attach /path/file.pdf ...] [--update DRAFT_ID] \
      [--service google-draas] [--from "Nishant Ranka <ndr@draas.com>"] \
      [--dry-run] [--outdir /tmp/emailbuild]

Prints one JSON object: {action, draft_id, message_id, thread_id, parts,
line_stats, html_path, text_path, warnings}. Non-zero exit on verification fail.

Markdown supported: # / ## / ### headings, **bold**, *italic*, `code`,
- bullets (nested with 2 spaces), 1. numbered, > quote, | tables |, --- rules,
[links](url).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from email.message import EmailMessage
from pathlib import Path

# ---------------------------------------------------------------- stylesheets
BASE_CSS = """
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.62;
  color: #1f2328;
""".strip()

STYLE_CSS = {
    "standard": f"""
body {{ {BASE_CSS} }}
p {{ margin: 0 0 14px 0; }}
strong {{ font-weight: 600; color: #111418; }}
h1, h2, h3 {{ font-size: 15px; font-weight: 600; color: #111418;
             margin: 26px 0 8px 0; line-height: 1.4; }}
h1 {{ font-size: 17px; }}
h2 {{ padding-bottom: 4px; border-bottom: 1px solid #e4e8ed; }}
ul, ol {{ margin: 0 0 14px 0; padding-left: 22px; }}
li {{ margin: 0 0 6px 0; }}
li > ul, li > ol {{ margin-top: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 4px 0 18px 0; font-size: 14px; }}
th, td {{ border: 1px solid #d8dee4; padding: 7px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; font-weight: 600; }}
blockquote {{ margin: 0 0 14px 0; padding: 2px 0 2px 14px; border-left: 3px solid #d0d7de; color: #57606a; }}
hr {{ border: 0; border-top: 1px solid #e4e8ed; margin: 24px 0; }}
a {{ color: #0b5cad; }}
code {{ font-family: SFMono-Regular, Consolas, monospace; font-size: 13px; background: #f2f4f7; padding: 1px 4px; border-radius: 3px; }}
""",
    "full": f"""
body {{ {BASE_CSS} background: #ffffff; }}
.wrap {{ border: 1px solid #e4e8ed; border-radius: 8px; overflow: hidden; }}
.head {{ background: #0f2d4a; color: #ffffff; padding: 16px 22px; }}
.head .kicker {{ font-size: 11px; letter-spacing: 1.4px; text-transform: uppercase; opacity: .78; }}
.head .title {{ font-size: 18px; font-weight: 600; margin-top: 4px; line-height: 1.35; }}
.body {{ padding: 20px 22px 8px 22px; }}
p {{ margin: 0 0 14px 0; }}
strong {{ font-weight: 600; color: #111418; }}
h1, h2, h3 {{ font-size: 15px; font-weight: 600; color: #0f2d4a;
             margin: 26px 0 8px 0; line-height: 1.4; }}
h1 {{ font-size: 17px; }}
h2 {{ padding-bottom: 4px; border-bottom: 1px solid #e4e8ed; }}
ul, ol {{ margin: 0 0 14px 0; padding-left: 22px; }}
li {{ margin: 0 0 6px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 4px 0 18px 0; font-size: 14px; }}
th, td {{ border: 1px solid #d8dee4; padding: 7px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; font-weight: 600; }}
blockquote {{ margin: 0 0 14px 0; padding: 2px 0 2px 14px; border-left: 3px solid #d0d7de; color: #57606a; }}
hr {{ border: 0; border-top: 1px solid #e4e8ed; margin: 24px 0; }}
a {{ color: #0b5cad; }}
code {{ font-family: SFMono-Regular, Consolas, monospace; font-size: 13px; background: #f2f4f7; padding: 1px 4px; border-radius: 3px; }}
""",
}


# ------------------------------------------------------------- md -> html/text
def md_to_html(md: str) -> str:
    import markdown

    return markdown.markdown(
        md,
        # deliberately NO 'nl2br': a soft newline inside a paragraph must JOIN, not
        # become <br>, otherwise a source file written at 80 cols re-creates the
        # narrow-ribbon rendering in the HTML part too.
        extensions=["tables", "sane_lists", "fenced_code"],
        output_format="html5",
    )


_INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _strip_inline(s: str) -> str:
    s = _INLINE_LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s)
    return s.strip()


def md_to_text(md: str) -> str:
    """Markdown -> plain text with ONE LINE PER PARAGRAPH (never hard-wrapped)."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush():
        if para:
            out.append(_strip_inline(" ".join(x.strip() for x in para)))
            out.append("")
            para.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush()
            i += 1
            continue

        # table block
        if stripped.startswith("|") and stripped.endswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [
                [_strip_inline(c.strip()) for c in r.strip("|").split("|")]
                for r in block
                if not re.fullmatch(r"\|[\s:|-]+\|", r)
            ]
            if rows:
                header, *body = rows
                for r in body:
                    if len(r) > 1 and r[0]:
                        rest = " | ".join(x for x in r[1:] if x)
                        out.append(f"- {r[0]}: {rest}" if rest else f"- {r[0]}")
                    else:
                        out.append("- " + " | ".join(x for x in r if x))
                out.append("")
            continue

        # horizontal rule
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush()
            out.append("")
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            out.append(_strip_inline(m.group(2)).upper() if len(m.group(1)) <= 2
                       else _strip_inline(m.group(2)))
            out.append("")
            i += 1
            continue

        # blockquote
        if stripped.startswith(">"):
            flush()
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(_strip_inline(" ".join(q)))
            out.append("")
            continue

        # bullets
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m:
            flush()
            indent = "  " * (len(m.group(1)) // 2)
            out.append(f"{indent}\u2022 {_strip_inline(m.group(2))}")
            i += 1
            # keep consecutive bullets tight (no blank line between)
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                mm = re.match(r"^(\s*)[-*+]\s+(.*)$", lines[i])
                out.append(f"{'  ' * (len(mm.group(1)) // 2)}\u2022 {_strip_inline(mm.group(2))}")
                i += 1
            out.append("")
            continue

        # numbered
        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if m:
            flush()
            out.append(f"{m.group(2)}. {_strip_inline(m.group(3))}")
            i += 1
            while i < len(lines) and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                mm = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", lines[i])
                out.append(f"{mm.group(2)}. {_strip_inline(mm.group(3))}")
                i += 1
            out.append("")
            continue

        para.append(line)
        i += 1

    flush()
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def wrap_html(inner: str, style: str, title: str = "", kicker: str = "") -> str:
    css = STYLE_CSS.get(style, STYLE_CSS["standard"])
    if style == "full":
        head = (
            f'<div class="head"><div class="kicker">{kicker or "DRA Group"}</div>'
            f'<div class="title">{title}</div></div>'
            if title
            else ""
        )
        inner = f'<div class="wrap">{head}<div class="body">{inner}</div></div>'
    return (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
        f"<style>\n{css}\n</style></head>\n<body>\n{inner}\n</body></html>\n"
    )


# ------------------------------------------------------------------ verification
def line_stats(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return {"lines": 0, "max": 0, "suspect": 0}
    suspect = 0
    for l in lines:
        s = l.strip()
        if s.startswith(("\u2022", "-", "*")) or re.match(r"^\d+[.)]\s", s):
            continue
        if s.isupper():
            continue
        if 55 <= len(s) <= 90 and not s.rstrip().endswith((".", "!", "?", ":", ";", ")", '"')):
            suspect += 1
    return {"lines": len(lines), "max": max(len(l) for l in lines), "suspect": suspect}


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="markdown body file, or '-' for stdin")
    ap.add_argument("--to", required=True)
    ap.add_argument("--cc", default="")
    ap.add_argument("--bcc", default="")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--from", dest="sender", default="Nishant Ranka <ndr@draas.com>")
    ap.add_argument("--style", choices=["standard", "full"], default="standard")
    ap.add_argument("--title", default="", help="header title for --style full")
    ap.add_argument("--kicker", default="", help="header kicker for --style full")
    ap.add_argument("--no-html", action="store_true",
                    help="plain-only. Use ONLY when NDR explicitly asks for plain text.")
    ap.add_argument("--thread", default="")
    ap.add_argument("--in-reply-to", dest="in_reply_to", default="")
    ap.add_argument("--references", default="")
    ap.add_argument("--attach", action="append", default=[])
    ap.add_argument("--update", default="", help="existing draft id to revise in place")
    ap.add_argument("--service", default="google-draas")
    ap.add_argument("--user", default="ndr", help="HERMES_SESSION_USER_ID")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default="/tmp/emailbuild")
    ap.add_argument("--state", default="")
    a = ap.parse_args()

    md = sys.stdin.read() if a.md == "-" else Path(a.md).read_text(encoding="utf-8")
    if not md.strip():
        print(json.dumps({"error": "empty body"}))
        return 2

    inner = md_to_html(md)
    html = wrap_html(inner, a.style, a.title, a.kicker)
    text = md_to_text(md)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "body.html").write_text(html, encoding="utf-8")
    (outdir / "body.txt").write_text(text, encoding="utf-8")

    stats = line_stats(text)
    warnings = []
    if stats["suspect"] >= 3:
        warnings.append(
            f"plain part looks hard-wrapped ({stats['suspect']} lines at 55-90 chars "
            "with no sentence end) - fix the markdown source"
        )

    msg = EmailMessage()
    msg["From"] = a.sender
    msg["To"] = a.to
    if a.cc:
        msg["Cc"] = a.cc
    if a.bcc:
        msg["Bcc"] = a.bcc
    msg["Subject"] = a.subject
    if a.in_reply_to:
        msg["In-Reply-To"] = a.in_reply_to.strip()
    if a.references:
        msg["References"] = re.sub(r"\s+", " ", a.references).strip()

    if a.no_html:
        msg.set_content(text)
    else:
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")

    for p in a.attach:
        data = Path(p).read_bytes()
        ctype = "application/octet-stream"
        ext = Path(p).suffix.lower()
        ctype = {
            ".pdf": ("application", "pdf"),
            ".png": ("image", "png"),
            ".jpg": ("image", "jpeg"),
            ".jpeg": ("image", "jpeg"),
            ".xlsx": ("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ".docx": ("application", "vnd.openxmlformats-officedocument.wordprocessingml.document"),
        }.get(ext, ("application", "octet-stream"))
        msg.add_attachment(data, maintype=ctype[0], subtype=ctype[1], filename=Path(p).name)

    result = {
        "action": "dry-run",
        "parts": ["text/plain"] if a.no_html else ["text/plain", "text/html"],
        "line_stats": stats,
        "html_path": str(outdir / "body.html"),
        "text_path": str(outdir / "body.txt"),
        "warnings": warnings,
    }
    if a.dry_run:
        print(json.dumps(result, indent=2))
        return 0

    os.environ.setdefault("HERMES_SESSION_USER_ID", a.user)
    sys.path.insert(0, "/opt/hermes")
    from tools.gws_auth import build_service

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc = build_service("gmail", "v1", service_name=a.service)

    body: dict = {"message": {"raw": raw}}
    if a.thread:
        body["message"]["threadId"] = a.thread

    if a.update:
        d = svc.users().drafts().update(userId="me", id=a.update, body=body).execute()
        result["action"] = "updated"
    else:
        d = svc.users().drafts().create(userId="me", body=body).execute()
        result["action"] = "created"

    result["draft_id"] = d["id"]
    result["message_id"] = d["message"]["id"]
    result["thread_id"] = d["message"].get("threadId", "")

    # ---- verify by reading the stored draft back
    got = svc.users().drafts().get(
        userId="me", id=d["id"], format="full").execute()["message"]

    def walk(p, acc):
        if p.get("mimeType"):
            acc.append(p["mimeType"])
        for c in p.get("parts", []) or []:
            walk(c, acc)
        return acc

    stored = walk(got["payload"], [])
    result["stored_types"] = stored
    if a.no_html:
        ok = "text/html" not in stored
    else:
        ok = "text/html" in stored and "text/plain" in stored
    if not ok:
        warnings.append(f"stored structure unexpected: {stored}")
    if "DRAFT" not in got.get("labelIds", []):
        warnings.append(f"draft not labelled DRAFT: {got.get('labelIds')}")

    hdrs = {h["name"].lower(): h["value"] for h in got["payload"].get("headers", [])}
    result["subject"] = hdrs.get("subject", "")
    result["to"] = hdrs.get("to", "")

    if a.state:
        Path(a.state).write_text(f"{d['id']}\n{d['message']['id']}\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0 if not warnings else 1


if __name__ == "__main__":
    sys.exit(main())
