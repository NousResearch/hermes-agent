#!/usr/bin/env python3
"""Create a Gmail DRAFT that replies into an existing thread AND carries a file.

The draft tools (bridge draft_create/draft_reply_create, native gws_gmail_draft_create /
gws_gmail_reply_draft) accept no attachment, so this builds the multipart MIME itself and calls
users().drafts().create(). It NEVER sends -- there is no send code path in this file.

Run with the Hermes venv and no PYTHONPATH leakage:

    env -u PYTHONPATH /opt/hermes/.venv/bin/python3 \
        scripts/draft_reply_with_attachment.py \
          --anchor  19cad1f04abda1ef \
          --thread  19cacba9e0cf52ec \
          --to      vishwas@vantagepointadvisors.in \
          --cc      "dharmesh.ranka1@gmail.com, mdr@draas.com" \
          --body-file /tmp/body.txt \
          --attach  "/data/hermes/tmp/jobs/Vishwas Rao - fee split shared 26 Oct 2026.pdf" \
          --service google-draas

Prints JSON: draft_id, message_id, threadId, subject, to, cc, attachment, attachment_bytes.
Then verifies the draft by reading it back and comparing the attachment part size to the file.
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sys.path.insert(0, "/opt/hermes")


def strip_prefixes(subject: str) -> str:
    s = re.sub(r"^\s*(re\s*:|fwd?\s*:|fw\s*:)\s*", "", subject or "", flags=re.I)
    return "Re: " + s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, help="message id of the thread tail you reply to")
    ap.add_argument("--thread", required=True, help="threadId the draft must stay in")
    ap.add_argument("--to", required=True)
    ap.add_argument("--cc", default="")
    ap.add_argument("--bcc", default="")
    ap.add_argument("--subject", default="", help="override; default = 'Re: ' + de-prefixed anchor subject")
    ap.add_argument("--body-file", required=True)
    ap.add_argument("--attach", action="append", required=True, help="repeatable")
    ap.add_argument("--service", default="google-draas")
    ap.add_argument("--html", action="store_true")
    args = ap.parse_args()

    from tools.gws_auth import build_service  # sanctioned vault path; never raw credentials

    svc = build_service("gmail", "v1", service_name=args.service)

    orig = svc.users().messages().get(
        userId="me", id=args.anchor, format="metadata",
        metadataHeaders=["Subject", "Message-ID", "References"],
    ).execute()
    h = {x["name"].lower(): x["value"] for x in orig["payload"].get("headers", [])}
    msgid = h.get("message-id", "")
    refs = h.get("references", "")
    subject = args.subject or strip_prefixes(h.get("subject", ""))

    body = open(args.body_file, encoding="utf-8").read()
    msg = MIMEMultipart()
    msg["To"] = args.to
    if args.cc:
        msg["Cc"] = args.cc
    if args.bcc:
        msg["Bcc"] = args.bcc
    msg["Subject"] = subject
    if msgid:
        msg["In-Reply-To"] = msgid
        msg["References"] = (refs + " " + msgid).strip() if refs else msgid
    msg.attach(MIMEText(body, "html" if args.html else "plain"))

    sizes = {}
    for path in args.attach:
        data = open(path, "rb").read()
        sizes[os.path.basename(path)] = len(data)
        ctype, _ = mimetypes.guess_type(path)
        subtype = (ctype or "application/octet-stream").split("/", 1)[1]
        part = MIMEApplication(data, _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
        msg.attach(part)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    draft = svc.users().drafts().create(
        userId="me", body={"message": {"raw": raw, "threadId": args.thread}}
    ).execute()
    did = draft["id"]

    # ---- verify: headers + every attachment part present at its real byte size ----
    got = svc.users().drafts().get(userId="me", id=did, format="full").execute()
    m = got["message"]
    gh = {x["name"]: x["value"] for x in m["payload"].get("headers", [])}

    def walk(p):
        out = []
        if p.get("filename"):
            out.append((p["filename"], p.get("body", {}).get("size")))
        for c in p.get("parts", []) or []:
            out += walk(c)
        return out

    parts = walk(m["payload"])
    ok = (m.get("threadId") == args.thread
          and gh.get("Subject") == subject
          and sorted(f for f, _ in parts) == sorted(sizes)
          and all(sz == sizes[f] for f, sz in parts))

    print(json.dumps({
        "draft_id": did,
        "message_id": m.get("id"),
        "threadId": m.get("threadId"),
        "subject": gh.get("Subject"),
        "to": gh.get("To"),
        "cc": gh.get("Cc"),
        "in_reply_to": gh.get("In-Reply-To"),
        "attachments": parts,
        "verified": ok,
    }, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
