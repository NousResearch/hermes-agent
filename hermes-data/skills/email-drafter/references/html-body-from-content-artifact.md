# Present a DRA Content artifact's HTML inside a Gmail draft (not as plain text)

**Trigger.** NDR says some version of: *"there's a draft in my email about X, and the same thing is in
the content system as neat clean HTML — I want the email presented using that HTML rendered itself,
not this difficult-to-read text only. Update the draft accordingly."*

This is a recurring request shape: the deliverable already exists twice — a long plain-text Gmail
draft and a formatted HTML artifact — and he wants the **draft body replaced with the rendered HTML**,
keeping the draft itself (same draft id, same recipients, same subject).

---

## 1. Find both halves first

```python
svc = gws_auth.build_service('gmail','v1', service_name='google-draas')   # resolve via gws_resolve_account
for x in svc.users().drafts().list(userId='me', maxResults=50).execute().get('drafts', []):
    m = svc.users().messages().get(userId='me', id=x['message']['id'], format='metadata',
                                   metadataHeaders=['Subject','To','Date']).execute()
    print(x['id'], {h['name']: h['value'] for h in m['payload']['headers']})
```

In job/cron sessions with no session user, export
`HERMES_SESSION_USER_ID=ndr GWS_VAULT_SOCKET=/run/gws-vault/vault.sock PYTHONPATH=/opt/hermes`
and call `tools.gws_auth.build_service(...)` directly — the native `gws_*` tools fail closed there.

The draft body usually names the artifact (`https://content.ahfl.in/a/DRA-ART-...`). Do **not** assume
that ID still resolves — see §5.

## 2. Get the HTML source

Order of preference:

1. The build directory the note was generated in — frequently still on disk:
   `/tmp/<note_name>/index.html` (grep for a distinctive sentence to locate it:
   `grep -rl "Note of Monday" /data/hermes /tmp`).
2. `content_get` / `content_list_versions` for the artifact id.
3. The session DB, which records the original `write_file` paths even when the artifact is gone:
   `select id, role, content from messages where content like '%Note of Meeting%'`.

Prefer the source that matches the draft's own version narrative — if the draft says *"this version
supersedes the one I sent earlier … car-park pages now embedded"*, use the build that contains those
edits (the `/tmp/<name>_v2` sibling is usually the later one).

## 3. Convert the document HTML into email-safe HTML

Gmail strips or ignores what a browser happily renders. Non-negotiables:

| Browser-OK | Must become |
|---|---|
| `<style>` block, class selectors | every rule inlined into `style="..."` on the element |
| CSS custom properties `var(--ink)` | the literal hex value |
| `display:flex` (e.g. a signature row) | a `role="presentation"` 3-column `<table>` |
| `tr:nth-child(even) td` zebra | explicit `background:#…` on the even row's `<td>` |
| relative `<img src="x.png">`, `data:` URIs, Drive view links | `src="cid:…"` + inline MIME part (see §4) |
| `box-shadow`, `page-break-*` | drop |
| `max-width:900px` sheet | `<table width="760">` wrapper on a `#e9e5dc` background |

`scripts/html_email_from_artifact.py` in this skill does all of the above from a source `index.html`
and writes `<out>.html` + `<out>.txt` (plain-text alternative). Run it, then eyeball the output:

```
python3 scripts/html_email_from_artifact.py /tmp/note_v2/index.html /tmp/out/minutes_email.html
```
It prints `unresolved var() left / style blocks left / flex left / cid images` — the first three must
be `0`.

**Keep the email's own front matter.** The artifact is the *note*; the *email* still needs NDR's
greeting, a two-line "what this is", any explicit asks ("reply with a date/time for the 31 October
meeting", "reply with the date you will deliver your items"), and — when the draft he is replacing was
itself a revision — a short "what changed in this version" block. Style those with the same palette
(gold `#c9a227` flag box for asks, `#f7f4ed` note box for changes, `#8a1c1c` accent, Georgia body).

**Drop dead links.** Never carry a `content.ahfl.in/...` link into the email unless you verified it
resolves a minute ago. The rendered body supersedes it anyway.

## 4. Build the MIME and update the draft IN PLACE

Structure that renders inline images without Gmail's "display images below" prompt:

```
multipart/related
├── multipart/alternative
│   ├── text/plain        (the stripped-down fallback — always include it)
│   └── text/html
├── image/png   Content-ID: <annexI@note>   Content-Disposition: inline
└── image/png   Content-ID: <clause841@note>
```

```python
res = svc.users().drafts().update(userId='me', id=DRAFT_ID,
                                  body={'message': {'raw': b64url(root.as_bytes())}}).execute()
```

- `drafts().update` **keeps the draft id, To, Subject and From** — no need to delete and recreate.
- Set `To`/`From`/`Subject` explicitly on the root MIME; `update` replaces the whole message.
- **Pitfall:** the thread id can change on update (a one-message draft thread is re-threaded). Harmless
  when the draft was the only message in it; check `res['message']['threadId']` and confirm the old
  thread is gone rather than assuming.
- Do **not** set `In-Reply-To` to the draft's own Message-Id when replacing a standalone draft.

## 5. Verify by reading the message back — structure, not vibes

```python
m = svc.users().messages().get(userId='me', id=new_msg_id, format='full').execute()
# expect: ['multipart/related','multipart/alternative','text/plain','text/html','image/png','image/png']
# html part must contain 'cid:…' and must NOT contain '<style', 'var(--', 'display:flex'
```

Then render a local preview (`sed` the `cid:` refs to `file://` paths) in the browser and check layout
**numerically**, not by asking a vision model for adjectives:

```js
[...document.images].map(i => [i.naturalWidth, i.clientWidth])   // client must be ~692, not 2220
[...document.querySelectorAll('table')].map(t => t.getBoundingClientRect().width)
[...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > innerWidth+1).length  // must be 0
```

An unscaled 2220px scan is the classic break — it silently blows the sheet past the viewport.
`browser_vision` prose about an email screenshot is unreliable (it has hallucinated missing yellow
boxes and invented headings); trust computed styles and the accessibility tree instead.

## 6. Known content-system gotcha

`content_get` / `content_list_versions` can return **`404 no such artifact`** for an artifact that a
past session genuinely published and linked (IDs are never reused, and archived items can 404 too).
`content_find` may also return 0 for an obvious term while a near-adjacent id exists — the search
index is sparse, so do not read 0 results as "it was never created".

When that happens: build from the local source, strip the dead link, and **tell NDR plainly** that the
artifact no longer resolves. Do not silently re-publish a fresh copy under a new id (that creates a
duplicate of a document that may be restored), and do not fabricate the artifact's contents.

## 7. Report

State: draft id, recipients/subject unchanged, what the body is now, that the scans are embedded inline,
that a plain-text alternative exists, that nothing was sent, and any dead link you removed. Attach the
HTML, the text fallback and a screenshot of the rendered top so he can see it without opening Gmail.
