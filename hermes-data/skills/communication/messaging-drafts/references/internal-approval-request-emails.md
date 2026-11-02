# Internal approval-request emails (staff → NDR)

The shape for "we want to do X, please approve" mail from a DRA staff member to
**Nishant (NDR, `ndr@draas.com`)**. Worked cleanly 2026-10-19 (Bharat Hawaldar →
NDR, approval of a Diwali coupon concept). Rich HTML, per `format-and-tone-two-modes.md`
— the email surface is always Mode A, whoever the recipient is.

## The six blocks, in this order

1. **Callout box at the top that disambiguates WHAT is being approved.** This is the
   block that prevents a wrong-shaped approval. Verbatim shape that worked:
   *"Request: approval of a concept and its offer parameters … What we seek now is your
   approval on the **concept** and the **commercial terms**, not on the artwork."* It
   stops the approver reviewing a first-pass design as if it were final.
2. **What we have spotted** — the opportunity, in plain business terms, ending on the one
   sentence that carries the argument (e.g. *"a printed coupon is the only vehicle we get
   in that moment"*). No adjectives, no build-up.
3. **The parameters as a table** — one row per decision (format, offer, scope, validity,
   how it is given, response route, compliance), each figure/date/name bold.
4. **A short section defending the commercially risky choice** — the decision the approver
   is most likely to question, stated as reasoning, not as a request for permission. In the
   coupon case: why the money is a reduction on the booking receipt and **not** a payment
   after the sale (customer tax position, agreement value, protecting the recorded rate).
5. **Numbered "approval sought" list** — the discrete things you need a yes on, one line
   each, so he can reply "1 and 3 yes, 4 change".
6. **Links to the artifacts + the next step after approval** (who does what, and the
   quantity/cost that will come back once approved). Hand over ownership of the design
   explicitly if a designer is involved ("the visual will be finalised with <name>").

Close with a short sign-off block: name, company, phone, email.

## Sender / recipient plumbing

- Create the draft in the **requester's own account** — for Bharat that is
  `service_name="google-draas"` (`sales1.blr@draas.com`), not NDR's mailbox. Confirm the
  recipient with `gws_dwd_resolve(target="ndr@draas.com")` → `Nishant Ranka` rather than
  assuming.
- **Never send.** Draft only.
- `gws_gmail_get` returns an **empty body for an HTML-only draft** — that is not a failure.
  Verify by decoding the stored draft instead:

```python
svc = build_service("gmail", "v1", service_name="google-draas")
got = svc.users().drafts().get(userId="me", id=draft_id, format="full").execute()
# walk payload.parts, base64.urlsafe_b64decode(part["body"]["data"]) for text/html,
# strip tags and assert the key facts are present
```

  Assert on **content**, not on success: the figures (e.g. `25,000`), the load-bearing
  clause (`no cash redemption`), the asks, the artifact links, and any compliance numbers
  (RERA) — a link typo or a dropped clause is invisible otherwise.
- **Delete the superseded draft when you recreate it after an edit** — two drafts of the
  same mail in the sender's folder is a real hazard. `users().drafts().delete(userId="me", id=...)`.
- Send the requester a **PNG preview of the rendered email** so they can read it on
  Telegram before sending: `weasyprint /tmp/email.html /tmp/email.pdf` then
  `pdftoppm -png -r 110 -singlefile`. Wrap the email HTML with a
  `@page { size: 900px 2100px }` so it renders as one tall page.

## Pitfalls

- Do not offer a **cc list** the user has not asked for. Suggest it in one line in your
  reply and let them decide.
- **Personal names you cannot resolve get flagged, not invented.** "Kaori ma'am" was not in
  the contacts sheet and had no Drive identity — the name was used as dictated *and* called
  out in the reply for confirmation. Never silently "correct" a dictated name into a
  plausible-looking one.
- Voice dictation garbles: revisit the reading before acting on a phrase like
  "the content you prepared for gift to Chandra" (it meant the gift **voucher**). State the
  interpretation in one line rather than blocking on a question.
