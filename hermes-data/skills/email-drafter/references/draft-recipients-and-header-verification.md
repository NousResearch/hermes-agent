# Draft recipients, Cc verification, and dictated-name resolution

Session origin: DRA Projects "MD mandate" email, 28 Oct 2026. NDR dictated the body
and said: "send a detailed email to Dharmesh and Manish, **marking DCC and Roshini**".

## 1. Dictation conventions for recipients

- **"marking X" = put X in Cc.** It does not mean "address it to X", and it is not a
  distribution list. Seen in the 26 Oct 2026 minutes ("marking Mamta") and again as
  "marking DCC and Roshini".
- **"new email as a follow-up" = a fresh message with its own subject**, not a reply in
  the old thread — even when the substance follows the previous note. Create it new, and
  say in the delivery summary that it is new and offer to re-issue it threaded if he
  wants it in the record of that chain.
- Where the new email refers back to an earlier note or minutes, name the note and the
  date in the body ("items 3 and 4 of the note of 26 October ... apply as recorded there")
  rather than restating or contradicting it.

## 2. An unresolvable dictated name is flagged, never guessed

If a dictated recipient ("DCC", an initials-only token, an unfamiliar surname) does not
resolve, do the resolution work first, in this order:

1. `contact_resolver` on the token, then on the plausible first/last-name expansions.
2. Gmail search `in:sent "<TOKEN>"`, `cc:<resolved-person> newer_than:180d`, and a
   metadata read of recent sent mail to the same counterparties (see §3) to learn who
   NDR habitually copies on this class of mail.
3. If it still does not resolve: **do not invent an address, do not insert a
   placeholder, do not drop the name silently.** Create the draft with the recipients
   you did resolve, and state plainly in the delivery summary: "<TOKEN> could not be
   matched to any contact or address; the draft copies X only — tell me who it is and
   I will add them."

A draft is safe to leave incomplete in this one respect as long as the gap is stated.
A guessed Cc is not — it puts a board email in a stranger's inbox.

## 3. Reading the Cc/To of an existing message

`gws_gmail_get` / `gws_gmail_thread_get` return `to`, `from`, `subject`, but **no `cc`**.
To see the real header set, call the API with metadata headers:

```python
svc.users().messages().get(userId="me", id=MSG_ID, format="metadata",
    metadataHeaders=["To","Cc","Bcc","Subject","Date"]).execute()
```

Useful for "who does he normally copy on board mail" before drafting.

## 4. Verifying a draft you just created — including Cc

`gws_gmail_get` on a **draft** returns `to` and `subject` but an **empty `body`** for an
HTML-only draft, and no `cc` at all. The `gws_gmail_draft_create` response therefore proves
only that a draft exists, not that it is addressed correctly.

Read the raw MIME. This must run **inside `execute_code`**, because the vault token bridge
(`hermes_tools.gws_fetch_token`, which `tools.gws_auth.load_credentials` imports) is a stub
that exists only in that sandbox — a `subprocess` call to
`/opt/hermes/.venv/bin/python3` does not have it, and the `gws` CLI on the box is not
authenticated for this path.

```python
import sys, base64
sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")
raw = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
mime = base64.urlsafe_b64decode(raw["message"]["raw"]).decode("utf-8", "replace")
print(mime.split("\n\n")[0])   # to / cc / subject / from / message-id
```

Confirm `to`, `cc`, and the decoded subject line against what was asked for. For the body,
walk `payload.parts` with `format="full"` and base64url-decode the `text/html` part.

Pitfall: `drafts().get(format="full")` returns `payload.headers` with `To`/`Cc` as `None` —
only the `raw` MIME carries them. Do not read that `None` as "no recipients".
