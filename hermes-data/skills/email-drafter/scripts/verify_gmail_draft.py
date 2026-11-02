#!/usr/bin/env python3
"""Read-back verifier for a Gmail draft created on an ndr@ account (or any vault account).

Usage:
    python3 verify_gmail_draft.py <message_id> [--service google-draas]
                                  [--local-dir DIR] [--keyword KW]...

Checks, in order:
  1. every draft currently in the mailbox (so a duplicate or a leftover is visible);
  2. the message's labelIds -- ['DRAFT'] means still unsent, ['SENT'] means NDR already
     sent it from his own client and recreating it would duplicate the send;
  3. decoded From / To / Cc / Subject / Date read from the RAW MIME
     (the JSON draft summary does not show Cc at all, so a Cc list is unverified until
     it is read here);
  4. every attachment's name, size and sha256 -- compared against --local-dir when given
     (a filename in the part metadata is NOT proof the recipient gets the same bytes);
  5. keyword presence in both the text/plain and text/html parts.

Creates nothing, sends nothing. Verification only.
"""
import argparse
import base64
import hashlib
import os
import sys

sys.path.insert(0, "/opt/hermes")  # NOT /opt/data -- that raises ModuleNotFoundError: tools
from tools.gws_auth import build_service  # noqa: E402

from email import message_from_bytes  # noqa: E402
from email.header import decode_header, make_header  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message_id", help="message id returned by drafts().create()")
    ap.add_argument("--service", default="google-draas",
                    help="vault service_name from gws_resolve_account (default google-draas)")
    ap.add_argument("--local-dir", help="dir of source files to sha256-compare attachments against")
    ap.add_argument("--keyword", action="append", default=[],
                    help="keyword that must appear in the body (repeatable)")
    args = ap.parse_args()

    svc = build_service("gmail", "v1", service_name=args.service)

    # 1. what else is sitting in Drafts
    drafts = svc.users().drafts().list(userId="me", maxResults=25).execute().get("drafts", [])
    print(f"DRAFTS IN MAILBOX: {len(drafts)}")
    for d in drafts:
        md = svc.users().messages().get(
            userId="me", id=d["message"]["id"], format="metadata",
            metadataHeaders=["To", "Cc", "Subject"]).execute()
        h = {x["name"]: x["value"] for x in md["payload"]["headers"]}
        print(f"    {d['id']} | {h.get('To')} | {(h.get('Subject') or '')[:70]}")

    # 2 + 3. raw read-back
    m = svc.users().messages().get(userId="me", id=args.message_id, format="raw").execute()
    print(f"\nLABELS  : {m.get('labelIds')}   (['DRAFT'] == unsent)")
    parsed = message_from_bytes(base64.urlsafe_b64decode(m["raw"]))
    for hdr in ("From", "To", "Cc", "Subject", "Date"):
        val = parsed[hdr]
        print(f"{hdr:8s}:", make_header(decode_header(val)) if val else "(none)")

    # 4. attachments
    local = {}
    if args.local_dir:
        for f in os.listdir(args.local_dir):
            p = os.path.join(args.local_dir, f)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    local[f] = hashlib.sha256(fh.read()).hexdigest()

    n = 0
    print("\nATTACHMENTS:")
    for part in parsed.walk():
        fn = part.get_filename()
        if not fn:
            continue
        fn = str(make_header(decode_header(fn)))
        data = part.get_payload(decode=True) or b""
        n += 1
        line = f"  {n:2d}. {fn}  {len(data) // 1024} KB"
        if local:
            line += f"  sha256 match: {hashlib.sha256(data).hexdigest() == local.get(fn)}"
        print(line)
    print(f"attachment count: {n}   local files: {len(local)}")
    # NB: do not accumulate this comparison into a flag initialised to False --
    # `ok = ok and match` prints False even when every file matched.

    # 5. body keyword check on BOTH parts
    if args.keyword:
        for part in parsed.walk():
            ctype = part.get_content_type()
            if ctype not in ("text/plain", "text/html"):
                continue
            body = part.get_payload(decode=True).decode("utf-8", "replace")
            print(f"\n--- {ctype} ({len(body)} chars) ---")
            for kw in args.keyword:
                print(f"   {kw:30s}: {kw in body}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
