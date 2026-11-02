#!/usr/bin/env python3
"""Revise an EXISTING Gmail draft in place, preserving the draft id and the user's own edits.

Why this exists: NDR routinely edits a draft himself and then says "update the draft with
this". Naive MIME surgery silently breaks the message (empty From/To/Subject, MIME headers
rendered as body text, stray boundary lines). This script does it correctly and verifies.

Usage:
  python3 revise_gmail_draft.py --draft-id r-3344798209455834652 \
      --find "6. Any other blockers" \
      --replace-file /tmp/new_item_5.txt \
      --service google-draas

  # several edits at once: a JSON list of {"find":..., "replace":...} pairs
  python3 revise_gmail_draft.py --draft-id r-... --edits edits.json

  # read-only: show the current draft as the recipient will see it
  python3 revise_gmail_draft.py --draft-id r-... --show

Notes:
  * Edits are applied to BOTH text/plain and text/html parts. For the HTML part pass
    --html-find/--html-replace when the markup differs from the plain text.
  * --find must match exactly once in the plain part; the script aborts otherwise, so a
    silent double-insert is impossible.
  * Nothing is ever sent. drafts().update only.
"""
import argparse
import base64
import json
import quopri
import re
import sys
from datetime import datetime, timedelta, timezone
from email.header import Header

sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service  # noqa: E402

BOUNDARY = "=_hermes_revise_boundary"


def _svc(name):
    return build_service("gmail", "v1", service_name=name)


def _raw(svc, msg_id):
    m = svc.users().messages().get(userId="me", id=msg_id, format="raw").execute()
    return base64.urlsafe_b64decode(m["raw"]).decode("utf-8", "replace")


def _find_message_id(svc, draft_id):
    d = svc.users().drafts().get(userId="me", id=draft_id).execute()
    return d["message"]["id"]


def split_parts(raw):
    """Return (plain, html) decoded text, whether the raw is multipart or flattened."""
    i_plain = raw.find("Content-Type: text/plain")
    i_html = raw.find("Content-Type: text/html")
    if i_plain == -1 or i_html == -1:
        # single-part message: treat the whole body as plain text
        body = raw.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in raw else raw
        return body, body
    plain_block = raw[i_plain:i_html]
    html_block = raw[i_html:]
    plain = quopri.decodestring(
        plain_block.split("quoted-printable", 1)[1].split("\r\n\r\n", 1)[1].encode()
    ).decode("utf-8", "replace")
    html = quopri.decodestring(
        html_block.split("quoted-printable", 1)[1].split("\r\n\r\n", 1)[1].encode()
    ).decode("utf-8", "replace")
    return plain, html


def clean_tail(t):
    """Drop trailing blank lines and any leaked MIME boundary token."""
    lines = t.replace("\r\n", "\n").split("\n")
    while lines and (not lines[-1].strip() or re.match(r"^--[0-9_=A-Za-z]{8,}", lines[-1].strip())):
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def apply_edits(text, edits, label):
    for e in edits:
        find = e.get("find") if label == "plain" else e.get("html_find", e.get("find"))
        repl = e.get("replace") if label == "plain" else e.get("html_replace", e.get("replace"))
        if not find:
            continue
        n = text.count(find)
        if n != 1:
            sys.exit("ABORT: %s anchor matched %d times (need exactly 1): %r" % (label, n, find[:70]))
        text = text.replace(find, repl)
    return text


def build_raw(subject, to_hdr, from_hdr, plain, html):
    """Build multipart/alternative with EXPLICIT top-level headers.

    Never re-join the old raw: folded header continuations and boundary spacing are the
    two things that silently corrupt the draft.
    """
    ist = timezone(timedelta(hours=5, minutes=30))
    date_hdr = datetime.now(ist).strftime("%a, %d %b %Y %H:%M:%S +0530")
    top = "\r\n".join([
        "Date: " + date_hdr,
        "Subject: " + Header(subject, "utf-8").encode(),
        "From: " + from_hdr,
        "To: " + to_hdr,
        "MIME-Version: 1.0",
        'Content-Type: multipart/alternative; boundary="%s"' % BOUNDARY,
    ])
    q1 = ('Content-Type: text/plain; charset="UTF-8"\r\n'
          'Content-Transfer-Encoding: quoted-printable\r\n\r\n'
          + quopri.encodestring(plain.encode()).decode("ascii"))
    q2 = ('Content-Type: text/html; charset="UTF-8"\r\n'
          'Content-Transfer-Encoding: quoted-printable\r\n\r\n'
          + quopri.encodestring(html.encode()).decode("ascii"))
    return (top + "\r\n\r\n--" + BOUNDARY + "\r\n" + q1
            + "\r\n--" + BOUNDARY + "\r\n" + q2
            + "\r\n--" + BOUNDARY + "--\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-id", required=True)
    ap.add_argument("--service", default="google-draas")
    ap.add_argument("--show", action="store_true", help="print current draft, change nothing")
    ap.add_argument("--find")
    ap.add_argument("--replace-file")
    ap.add_argument("--edits", help="JSON file: [{\"find\":..,\"replace\":..,\"html_find\":..,\"html_replace\":..}]")
    a = ap.parse_args()

    svc = _svc(a.service)
    msg_id = _find_message_id(svc, a.draft_id)
    raw = _raw(svc, msg_id)
    plain, html = split_parts(raw)
    plain, html = clean_tail(plain), clean_tail(html)

    if a.show:
        print(plain)
        return

    edits = []
    if a.edits:
        edits = json.load(open(a.edits))
    if a.find:
        repl = open(a.replace_file).read() if a.replace_file else ""
        edits.append({"find": a.find, "replace": repl})
    if not edits:
        sys.exit("nothing to do: pass --edits or --find/--replace-file")

    plain = apply_edits(plain, edits, "plain")
    html = apply_edits(html, edits, "html")

    # reuse the existing headers' Subject / To / From so nothing is re-typed by hand
    subj = re.search(r"^Subject: (.*(?:\r\n[ \t].*)*)", raw, re.M)
    to_h = re.search(r"^To: (.*(?:\r\n[ \t].*)*)", raw, re.M)
    from_h = re.search(r"^From: (.*(?:\r\n[ \t].*)*)", raw, re.M)
    from email.header import decode_header, make_header
    subject = str(make_header(decode_header(subj.group(1).replace("\r\n", " ")))) if subj else ""
    to_hdr = re.sub(r"\s+", " ", to_h.group(1)).strip() if to_h else ""
    from_hdr = re.sub(r"\s+", " ", from_h.group(1)).strip() if from_h else ""

    new_raw = build_raw(subject, to_hdr, from_hdr, plain, html)
    b64 = base64.urlsafe_b64encode(new_raw.encode("utf-8")).decode("ascii")
    res = svc.users().drafts().update(
        userId="me", id=a.draft_id, body={"message": {"raw": b64}}).execute()
    print("updated draft", res["id"], "-> message", res["message"]["id"])

    # VERIFY through the rendered view, not the raw
    got = svc.users().messages().get(
        userId="me", id=res["message"]["id"], format="metadata",
        metadataHeaders=["From", "To", "Subject"]).execute()
    hdrs = {h["name"].lower(): h["value"] for h in got["payload"].get("headers", [])}
    print("From   :", hdrs.get("from"))
    print("To     :", hdrs.get("to"))
    print("Subject:", hdrs.get("subject"))
    if not all(hdrs.get(k) for k in ("from", "to", "subject")):
        sys.exit("BROKEN: headers missing — the MIME rebuild failed, inspect before reporting")


if __name__ == "__main__":
    main()
