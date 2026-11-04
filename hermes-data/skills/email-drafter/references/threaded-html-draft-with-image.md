# Threaded HTML draft with an image attachment (raw MIME, verified 2026-09-29)

Use when NDR dictates a STRONG follow-up email that must (a) reply on the SAME thread,
(b) carry HTML+CSS highlighting (red "critical" banners, amber call-outs, navy boxes),
and (c) attach an image (often a screenshot/annotation NDR just sent in Telegram).

Worked case: "Oasis 50 cents Area planning" thread (`1a0b9536ac51197e`) — follow-up to
Sinchana demanding ZERO odd plots, cc royalty (Bharat, Anbu, Kantesh, Gowri) with a Gowri
"FYI only — hold" box, attaching NDR's annotated layout image
(`/data/hermes/image_cache/img_d6864455e58c.jpg`).

## Why not the existing scripts
- `scripts/create-draft-with-attachments.py` builds ONLY `text/plain` + PDF parts —
  no HTML body, and `Content-Type: application/pdf` hardcoded for every attachment.
- `gws_gmail_draft_create` supports HTML but no attachments.
- The verified path is hand-built raw MIME: `multipart/mixed` wrapping
  `multipart/alternative` (text/plain + text/html) + one image part.

## Non-negotiable ingredients
1. `HERMES_SESSION_USER_ID=ndr` + `/opt/hermes/.venv/bin/python3` + `build_service('gmail','v1',service_name='google-draas')`; ALWAYS print `MAILBOX:` from `users().getProfile` and confirm `ndr@draas.com`.
2. Threading: pass `threadId` in the draft body AND set `In-Reply-To`/`References` from the
   parent message's actual `Message-ID` header (`format='metadata', metadataHeaders=['Message-ID','References']`).
3. Recipients: resolve via `contact_resolver` and cross-check Gmail `from:` search for the
   CURRENT work mailbox (contacts sheet can carry stale/personal addresses — Gowri Singh =
   gsingh@draas.com found via `from:gowri newer_than:2y`, NOT her yahoo/gmail rows).
4. Attach the EXACT image NDR sent: `Content-Type: image/jpeg; name="<basename>"`,
   `Content-Disposition: attachment`, base64 body. It lives at the path from the Telegram
   attachment block (e.g. `/data/hermes/image_cache/…`).
5. HTML per `ndr-email-presentation-2026.md`: Arial 15px, max-width 860px, navy #1F3864
   headers; for a "very concerning/critical" tone use a red `#c00000` banner box at top,
   amber `#fff8e1` call-out for the rework ask, navy box for FYI-only addresses.

## Verify after create (mandatory)
```python
got = svc.users().drafts().get(userId='me', id=draft_id, format='full').execute()
# assert labelIds == ['DRAFT']
# assert threadId == original
# walk payload parts collecting filenames -> image basename present
# read To/Cc/Subject headers
```

## Draft-spec shape (what worked)
```
TO      = primary recipient
CC      = all others (Bharat, Anbu, Kantesh, Gowri) — reply-all semantics carried manually
SUBJECT = "Re: <original subject>" (keep exact)
THREAD_ID, IN_REPLY_TO, REFERENCES from parent
```
RAW layout: head (MIME-Version/From/To/Cc/Subject/In-Reply-To/References/Message-Id) +
`multipart/mixed; boundary=B` + inner `multipart/alternative; boundary=A` (plain, html) +
`--B` image part + `--B--`. Base64 via `urlsafe_b64encode(raw.encode('utf-8'))`.