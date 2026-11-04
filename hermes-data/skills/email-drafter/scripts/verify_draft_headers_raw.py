#!/usr/bin/env python3
"""Verify a Gmail DRAFT via the raw MIME header block.

Why this exists: `gws_gmail_get` does NOT return the Cc header, so a Cc set with
`gws_gmail_draft_create(cc=...)` cannot be confirmed from the native tool output.
This reads the raw envelope instead and asserts the things that matter.

Usage:
    cd /opt/hermes && python3 /data/hermes/skills/email-drafter/scripts/verify_draft_headers_raw.py \
        <message_id_or_draft_id> [service_name] [--expect-to a@x.com] [--expect-cc b@y.com]

`service_name` defaults to google-draas. Run `gws_resolve_account` first and pass the
exact key — never guess it, and never pass a raw email address.
"""
import base64
import quopri
import sys

sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service  # noqa: E402

WANT = ("from", "to", "cc", "bcc", "subject", "in-reply-to", "references", "message-id")


def fetch_raw(svc, ident):
    """Accept either a draft id (r...) or a message id (hex) and return the raw bytes."""
    if ident.startswith("r"):
        d = svc.users().drafts().get(userId="me", id=ident).execute()
        ident = d["message"]["id"]
    m = svc.users().messages().get(userId="me", id=ident, format="raw").execute()
    return m, base64.urlsafe_b64decode(m["raw"]).decode("utf-8", "replace")


def main(argv):
    if not argv:
        print(__doc__)
        return 2

    ident = argv[0]
    service = "google-draas"
    expect_to = expect_cc = None
    rest = argv[1:]
    i = 0
    while i < len(rest):
        if rest[i] == "--expect-to":
            expect_to = rest[i + 1].lower()
            i += 2
        elif rest[i] == "--expect-cc":
            expect_cc = rest[i + 1].lower()
            i += 2
        else:
            service = rest[i]
            i += 1

    svc = build_service("gmail", "v1", service_name=service)
    who = svc.users().getProfile(userId="me").execute()["emailAddress"]
    print("AUTHENTICATED AS:", who)

    m, raw = fetch_raw(svc, ident)
    hdr, _, body = raw.partition("\n\n")
    h = {}
    for line in hdr.split("\n"):
        if ":" in line:
            k, _, v = line.partition(":")
            if k.strip().lower() in WANT:
                h[k.strip().lower()] = v.strip()

    print("LABELS:", m.get("labelIds"), "(must contain DRAFT; SENT means it already went out)")
    print("THREAD:", m.get("threadId"))
    for k in WANT:
        if k in h:
            print(f"{k:>12}: {h[k]}")

    text = quopri.decodestring(body).decode("utf-8", "replace")
    print("\n--- decoded body (first 600 chars) ---")
    print(text.strip()[:600])
    print("--- checks ---")
    # Literal HTML entities in a plain-text body mean html= was left false while
    # entity-escaped copy was passed in. See references/native-gws-draft-tools-pitfalls.md
    print("literal '&amp;' in body:", "&amp;" in text, "(must be False for plain-text drafts)")

    ok = True
    if "DRAFT" not in (m.get("labelIds") or []):
        print("FAIL: not labelled DRAFT — do not report it as an unsent draft")
        ok = False
    if expect_to and expect_to not in (h.get("to") or "").lower():
        print(f"FAIL: To does not contain {expect_to} -> {h.get('to')!r}")
        ok = False
    if expect_cc and expect_cc not in (h.get("cc") or "").lower():
        print(f"FAIL: Cc does not contain {expect_cc} -> {h.get('cc')!r}")
        ok = False
    if "&amp;" in text:
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
