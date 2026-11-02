#!/usr/bin/env python3
"""Create a properly THREADED Gmail reply draft (draft only - never sends) and verify it.

Built for the email-drafter skill. Handles the pitfalls that have bitten before:
  * Message-ID must come from format="raw" (format="metadata" can return it empty)
  * threadId MUST be passed to drafts().create() or Gmail mints a NEW thread
  * the HTML part must stay declared text/html (a demoted part shows literal markup
    to the recipient - the failure NDR actually reports)

Run with the Hermes venv, NOT execute_code (the sandbox lacks gws_fetch_token):
    /opt/hermes/.venv/bin/python3 scripts/create_threaded_reply_draft.py ...

Example:
    /opt/hermes/.venv/bin/python3 create_threaded_reply_draft.py \
      --service google-draas \
      --reply-to-id 1a143287bd9ac2d9 \
      --to "Prakash Singh <psingh@draas.com>" \
      --body-text /tmp/body.txt --body-html /tmp/body.html

Exit 0 only if every verification assertion passes.
"""
import argparse
import base64
import email as email_parser
import re
import sys

sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service  # noqa: E402
from email.message import EmailMessage  # noqa: E402


def clean(v):
    return re.sub(r"\s+", " ", v or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="google-draas",
                    help="vault service_name (resolve with gws_resolve_account; never a raw email)")
    ap.add_argument("--reply-to-id", required=True,
                    help="Gmail message id being replied to (the message whose headers we thread on)")
    ap.add_argument("--to", required=True, help='e.g. "Prakash Singh <psingh@draas.com>"')
    ap.add_argument("--cc", default="", help="omit entirely for a single-recipient reply")
    ap.add_argument("--subject", default="", help="override; default = Re: <original subject>")
    ap.add_argument("--body-text", required=True, help="path to plain-text body")
    ap.add_argument("--body-html", required=True, help="path to HTML body")
    ap.add_argument("--thread-id", default="", help="override master threadId (default: source message's)")
    args = ap.parse_args()

    svc = build_service("gmail", "v1", service_name=args.service)

    raw = base64.urlsafe_b64decode(
        svc.users().messages().get(userId="me", id=args.reply_to_id, format="raw").execute()["raw"]
    ).decode("utf-8", errors="replace")
    parsed = email_parser.message_from_string(raw)
    msg_id_header = clean(parsed.get("Message-ID", ""))
    references = clean(parsed.get("References", ""))
    orig_subject = clean(parsed.get("Subject", ""))
    thread_id = args.thread_id or parsed.get("X-GM-THREADID", "")

    # threadId is not a header - read it from the API resource itself
    meta = svc.users().messages().get(userId="me", id=args.reply_to_id, format="metadata").execute()
    thread_id = thread_id or meta.get("threadId", "")

    assert msg_id_header, "no Message-ID on the source message -> threading would break"
    assert thread_id, "no threadId resolved -> draft would land in a NEW thread"

    subject = args.subject or (
        orig_subject if orig_subject.lower().startswith("re:") else "Re: " + orig_subject
    )

    reply = EmailMessage()
    reply["To"] = args.to
    if args.cc:
        reply["Cc"] = args.cc
    reply["Subject"] = subject
    reply["In-Reply-To"] = msg_id_header
    reply["References"] = references if references else msg_id_header
    reply.set_content(open(args.body_text, encoding="utf-8").read())
    reply.add_alternative(open(args.body_html, encoding="utf-8").read(), subtype="html")

    draft = svc.users().drafts().create(
        userId="me",
        body={"message": {
            "raw": base64.urlsafe_b64encode(reply.as_bytes()).decode(),
            "threadId": thread_id,
        }},
    ).execute()

    got = svc.users().drafts().get(userId="me", id=draft["id"], format="full").execute()
    h = {x["name"]: x["value"] for x in got["message"]["payload"]["headers"]}
    parts = [p.get("mimeType") for p in got["message"]["payload"].get("parts", [])]

    print("DRAFT ID   :", draft["id"])
    print("threadId   :", got["message"].get("threadId"), "(expected", thread_id + ")")
    print("To         :", h.get("To"))
    print("Cc         :", h.get("Cc"))
    print("Subject    :", h.get("Subject"))
    print("In-Reply-To:", h.get("In-Reply-To"))
    print("parts      :", parts)
    print("labels     :", got["message"].get("labelIds"))

    ok = (
        got["message"].get("threadId") == thread_id
        and "text/html" in parts
        and "text/plain" in parts
        and h.get("In-Reply-To") == msg_id_header
    )
    in_list = draft["id"] in [
        d["id"] for d in svc.users().drafts().list(userId="me").execute().get("drafts", [])
    ]
    print("in drafts().list():", in_list)
    ok = ok and in_list

    if not ok:
        print("VERIFICATION FAILED - fix before telling the user it is ready")
        return 1
    print("VERIFIED - threaded reply draft, not sent. The user sends it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
