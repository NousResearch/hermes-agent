# Re-sending a mail with a named template — "formatting from mail A, content from mail B"

Trigger shape (NDR): *"the content is from my 7 October email to Prakash of Giraaf, Nishant Prakash and Sarthak,
but the formatting I want is like the earlier 'Ranka deal documents — Giraaf DD pack' email I sent to Prakash
Singh, Roshni, Nishant Prakash, Bharat and Sartak."* That is **two different sent mails**. Reuse the HTML of the
first, the content of the second.

## Why one of them needs rebuilding
Detailed mails he sends himself frequently lose their styling in transit (re-pasted → sent as flat text). So the
**newest** mail is usually the flat one and the **older** mail is the template. Never assume the latest is the
styled one — check both.

## Step 1 — identify the two mails (never guess from the subject)
He names them by **recipient list + rough timing**. Two versions usually share the same subject line, so:
- `gws_gmail_search` on the subject stem, then read each candidate's **Cc**.
- **Cc is the discriminator.** The bridge's `gws_gmail_get` output shows `to` only; fetch headers with
  `format='full'` and read `hdrs['cc']` and `hdrs['message-id']`.
- Confirm sender is ndr@draas.com and the date matches what he said ("Wednesday the 7th at 2044").
- If a name doesn't appear, remember he Cc's colleagues — Roshni = rnr@draas.com, Sarthak = admin3.blr@draas.com,
  Bharat = sales1.blr@draas.com.

## Step 2 — pull the REAL html part
`gws_gmail_get` / `gws_gmail_thread_get` return the `text/plain` rendering only. Literal markdown asterisks
(`*Project*`, `*Total cost incurred*`) appear in that text for **every** HTML mail and are NOT evidence the mail
was sent plain. Never judge formatting from the bridge body.

```python
# /opt/hermes/.venv/bin/python3
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")   # resolve service_name via gws_resolve_account
m = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
hdrs = {h["name"].lower(): h["value"] for h in m["payload"]["headers"]}
# walk m["payload"]["parts"] recursively; where mimeType == "text/html":
#   base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", "replace")
```

## Step 3 — fingerprint both so you can prove which is which
Byte length, md5, sorted unique `#rrggbb` colours, `font-family`, presence of a `<style>` block, counts of
`<table>` / `<h2>` / `<h3>`, marker strings (e.g. "FOR PRAKASH", "Open folder 07"). A styled mail shows colours +
a font-family + high h2/h3 counts; a flat one shows none of these. **Report the pair back to NDR before
drafting** — that's how he confirms you grabbed the right two.

## Step 4 — rebuild and stage
- Save both HTML parts to disk (`/tmp/<topic>/sent_<date>.html`) and record the **message IDs in the governing
  skill/reference** — /tmp is ephemeral and the mails may need re-extraction months later.
- Build the new body from the template's markup with the newer content swapped in; drop any internal-only block
  (e.g. a "FOR PRAKASH — items to confirm" yellow box) from the outbound version.
- Reply in-thread with `gws_gmail_reply_draft` so In-Reply-To/References stay on the original thread, and
  **delete any stale draft sitting on the same thread** so only one survives.
- Draft only. Never send.

## Recipient discipline
When he says "reply all … only to X, and Y and Z in Cc", the **explicit list wins** over reply-all semantics.
Restate your reading before creating the draft: "To Nishant Prakash; Cc Prakash Singh and Sarthak Sharma; nobody
else — confirm if you want Roshni/Bharat added."

## Delivery discipline for staged edits
When he supplies new data for only some sections: apply them, then present **only the changed sections** back for
review, and draft the mail only after his approval. If other threads carry newer numbers that contradict the ones
he quoted, flag them before he signs off.
