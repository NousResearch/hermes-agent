# Native draft tool: real HTML in, raw MIME out to verify

Applies to `gws_gmail_draft_create` / `gws_gmail_reply_draft` (the native, draft-only tools),
as distinct from the `EmailMessage`-built raw MIME route in the parent skill.

## 1 · `html=true` means the body string is sent VERBATIM as `text/html`

There is no newline-to-`<br>` conversion and no pipe-table rendering. A body written as

```
Unit | Floor | Market value
601 | 6th | Rs 40 crore
```

arrives at the recipient as one run-on paragraph — every line break gone. This is the exact
mistake made in the 29-Oct-2026 Sarthak mail and it had to be deleted and recreated.

- For a **table**, write real `<table>` HTML with **inline** border/padding styles.
  Gmail strips `<style>` blocks and classes, so `style="border:1px solid #bfbfbf"` on each
  `<td>` is what actually survives. `cellpadding`/`cellspacing` attributes still work.
- Use HTML entities for typography — `&middot;` `&mdash;` `&#8377;` — rather than literal
  Unicode, so the body is safe to pass through a shell or JSON layer unchanged.
- If the mail genuinely is plain text, pass `html=false` and rely on newlines. Do not mix:
  plain-text-with-pipes + `html=true` is the failure mode above.

## 2 · Correcting a bad draft = delete, then recreate

These tools cannot patch an existing draft's body. `gws_gmail_draft_delete(draft_id)` then
`gws_gmail_draft_create(...)` again. Keep the same subject so the recreated draft is obvious.

The create call returns **both** handles: `draft_id` (e.g. `r-1836553301263656547`) and
`message_id`. Use `draft_id` for delete/verify-by-draft; `message_id` for `messages().get`.

## 3 · Verifying: `To`/`Subject` read as `None` until you look at raw

`drafts().get(format="metadata")` and `messages().get(format="full")` came back with
`To: None` and `Subject: None` on a perfectly correct draft — the headers are MIME-encoded
(`=?utf-8?q?...?=`) and the naive header dict misses them. **This is not a failed draft.**

Verify with `format="raw"`, base64url-decode the RFC 822, and read the header lines:

```python
d = gm.users().drafts().get(userId="me", id=draft_id, format="raw").execute()
raw = base64.urlsafe_b64decode(d["message"]["raw"]).decode("utf-8", "replace")
print(raw[:1200])          # to: / subject: / From: are plain here
```

That read also confirms the HTML part's byte length matches the string you composed — the
cheapest proof that the body you intended is the body that landed.

Also: the mailbox normally holds other drafts. A draft count of 3 after creating one and
deleting one is expected — do not read a non-zero count as "my delete failed".

## 4 · Which route to use

- Native `gws_gmail_draft_create` — fine for a short note, a fact sheet, a table of figures.
- `EmailMessage` + raw MIME (parent skill) — needed when you want a **plain-text alternative
  part** alongside the HTML, attachments, or In-Reply-To/References threading.
