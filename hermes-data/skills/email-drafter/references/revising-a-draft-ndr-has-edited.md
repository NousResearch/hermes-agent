# Revising a draft NDR has already edited (in place, draft id preserved)

Trigger: "I did not send out the draft email; I have updated it. Please review that."
"...please add this ... and update the draft for me with this."

## 0. The rule

His rewrite is authoritative. Review it, then ADD to it. Never restore anything he
deleted, never "improve" his wording, never regenerate the email from your own earlier
version. If his edit dropped a point you had written (e.g. he deleted the "Sartak has
nothing for tomorrow" agenda line), that deletion stands — do not put it back.

## 1. Confirm he actually edited it

```
gws_gmail_draft_list(service_name="google-draas")
```
The DRAFT id is stable (`r-3344798209455834652`); the underlying **message id changes**
on every save (`1a18ce46…` → `1a18d0e2…`). A different message id under the same draft id
is proof of his edit. Then read his version:

```
gws_gmail_get(message_id=<current message id>, service_name="google-draas")
```

## 2. Update, don't recreate

Call `users().drafts().update(userId="me", id=DRAFT_ID, body={"message": {"raw": b64}})`
with the **same draft id**. Never delete-and-recreate: a new draft id loses his place,
his read state and the appearance of continuity in his Drafts folder.

Any body edit must be applied to **both** the `text/plain` and the `text/html` parts,
or the two versions disagree depending on the recipient's client.

## 3. MIME surgery — four traps that all produce a broken draft

Use `scripts/revise_gmail_draft.py` (it does all of this). If hand-rolling:

1. **Do not split `raw` on `"--"+boundary` and rejoin with `"--"+boundary+"\r\n"+part`**
   when the part still carries its own leading `\r\n`. You get a blank line immediately
   after the boundary, the part loses its headers, and Gmail stores the whole MIME as
   the message *body*.
   Symptom: `gws_gmail_get` returns `from`/`to`/`subject` **empty**, and `body` starts
   `Content-Type: text/html; charset="UTF-8"…`.
2. **Never filter header lines with `startswith("Received:")`** and keep the rest.
   `Received:` and `To:` are folded — their continuation lines start with a TAB, survive
   the filter, and terminate the header block early. Same symptom as (1).
   Correct fix: **build the top-level headers explicitly** —
   `Date`, `Subject` (via `email.header.Header(subject, "utf-8").encode()`),
   `From`, `To`, `MIME-Version: 1.0`, `Content-Type: multipart/alternative; boundary="…"`.
3. **When pulling a quoted-printable part out of a flattened body**, split on
   `"quoted-printable"` and then on the FIRST `"\r\n\r\n"`. Do not `.lstrip("\r\n")`
   first — you then split on a paragraph break *inside* the body and truncate the email.
4. **Strip leftover boundary tokens** (`^--[0-9_=A-Za-z]{8,}`) from the decoded text
   before rebuilding, or the draft ends with a stray `--0000000000004434f4065f0eff98`
   line visible to the recipient.

## 4. Verify — read it back, twice

After `drafts().update`, call `gws_gmail_get` on the NEW message id and check all four:

- `from`, `to`, `subject` are populated (not empty).
- `body` begins with his first line, not with `Content-Type:`.
- the added block is present and numbering is contiguous (inserting agenda item 5 into a
  1,2,3,4,6 list fixes the gap — keep "Any other blockers" as 6).
- `body` does not end with a boundary token.

Reading the raw (`format="raw"`) and decoding by hand is NOT verification — that is how
the broken build passed its own check. Verify through `gws_gmail_get`, which renders what
the recipient will see.

## 5. Reporting back

State the draft id, that nothing was sent, what was added verbatim, and what he had
changed that you deliberately left alone. Flag dictation artefacts rather than silently
"correcting" them (e.g. "Bhadscape" → written as "hardscape", with the offer to change it
if it is a firm's name).
