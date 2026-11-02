# Revising an Existing Draft, Internal-Request Tone, Link Verification

## Tone rule — when WE are the ones asking another team for work

Bharat's correction (17-Oct-2026). A hoarding-design request to Gowri Singh (creative head) carried:

> "To be clear on the sequence — preparing these posters was not our scope, but we have prepared them anyway… Please treat the attached posters as a reference only."

His verdict: **"it looks rude"** — because the whole point of the mail is that we are *requesting her* to do the design. Scope language turns a request into a complaint.

- Delete any "not our scope" / "not our job" / "we did it anyway" framing from a draft that asks for work.
- Reframe as preparation + credit: *"We have prepared X ourselves, purely so that [the inputs] are settled before design begins. Please treat these as a working reference — the design itself is entirely in your hands."*
- Add one courtesy line before the sign-off: *"Happy to walk you through any of the locations on a call before you begin."*
- Keep the ask explicit and complete (the data table, the confirmation points) — polite ≠ vague.
- Applies to any internal request: creative, legal, accounts, contractors, site teams.

## Revising a draft in place (with attachments) — Gmail

The tool surface here exposes create / list / delete for drafts, **not** update. Working sequence:

1. `drafts().get(userId="me", id=<draft_id>)` → message id.
2. `messages().get(userId="me", id=<mid>, format="full")` — **pass `format="full"` explicitly.** Without it the response can come back with `payload: None` and part-walking raises `TypeError: 'NoneType' object is not iterable`.
3. Flatten `payload["parts"]` recursively (a helper that appends then recurses); take the `text/plain` part as the body and download each attachment with `messages().attachments().get(...)` → base64 → bytes.
4. Replace only the offending paragraph(s) in the body text (assert the old text is present before writing), rebuild the MIME: `MIMEMultipart` + `MIMEText(new_body, "plain", "utf-8")` + `MIMEApplication(bytes, _subtype="pdf")` with a `Content-Disposition` filename, then `drafts().create(...)`.
5. **Delete the superseded draft** (`drafts().delete`) so exactly one live draft exists — Bharat's stated preference; two near-identical drafts invite sending the wrong one.
6. Read the new draft back and show the user the **changed lines verbatim (before → after)**, plus what was left untouched — not the whole body.

## Verify every link is publicly openable BEFORE it goes to a customer

A link shared to a lead must open with no Google sign-in. Anonymous probe (no cookies):

```bash
curl -sI -L --max-time 8 -o /dev/null -w '%{http_code}|%{url_effective}' \
  "https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download"
```

- Redirect to `accounts.google.com/ServiceLogin` → **PRIVATE** — do not send; fix sharing first.
- `200` → **PUBLIC**, safe to include in the message.
- Google **Sheets** can return `500` on that endpoint regardless of sharing. Test the real link instead: `https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>` returning CSV content anonymously means the `/edit` link is openable. Get the `gid` values from `spreadsheets().get(fields="sheets.properties(sheetId,title)")`.
- **Run the checks in a thread pool.** Sequential curls across 6+ files are slow enough that the turn gets interrupted before you see results.

When the message carries a price/cost sheet, state the rate basis you are quoting and ask the user to confirm it before they tap the link.
