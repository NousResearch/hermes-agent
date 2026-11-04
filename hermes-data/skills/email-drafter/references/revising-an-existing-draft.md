# Revising an EXISTING draft (not writing a new one)

Trigger: "update the draft", "amend the draft", "add X to that email",
"the rest of the email is fine, just change Y". The draft already exists on the
Gmail server from an earlier turn or an earlier job. Do not start from scratch.

## 1. Find the draft and its thread first

```python
import os, sys
os.environ['HERMES_SESSION_USER_ID'] = 'ndr'   # BEFORE importing gws modules
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
g = build_service('gmail', 'v1', service_name='google-draas')
for dr in g.users().drafts().list(userId='me', maxResults=100).execute().get('drafts', []):
    i = g.users().drafts().get(userId='me', id=dr['id'], format='metadata').execute()
    print(dr['id'], i['message'].get('threadId'),
          [h['value'] for h in i['message']['payload']['headers'] if h['name'] == 'Subject'])
```

Confirmed live draft = one row whose subject matches the matter. Keep both
`draft_id` and `threadId` — you will assert both after re-creating.

**Job / api_server sessions have no session user.** The native `gws_*` tools fail
closed with `service_name 'google-draas' is not one of YOUR accounts`. That is not
an auth problem and not a reason to re-OAuth. Use a script file run through
`terminal`, with `os.environ['HERMES_SESSION_USER_ID'] = 'ndr'` executed *inside*
the script before the `tools.*` imports (`setdefault` does not work — the var
already exists as an empty string). Same pattern for `tools.gws_skill_bridge`.

## 2. The bridge has no draft_update — revision is delete + re-create

Bridge surface in `tools/gws_skill_bridge`: `draft_create`, `draft_reply_create`,
`draft_delete`, `draft_get`, `draft_list`. There is **no** `draft_update`.

Correct sequence for a threaded reply draft:

1. `call("draft_delete", service_name="google-draas", draft_id=<old>)`
2. `call("draft_reply_create", service_name="google-draas",
        message_id=<the same message you replied to before>,
        to=..., cc=..., body=<new html>, html=True)`

Re-using the **same `message_id`** is what keeps `threadId`, `In-Reply-To` and
`References` intact. Confirmed working: threadId after re-create == threadId before.
Delete-then-create (never create-then-delete) so there is never a window with two
live drafts in the thread, and never leave the stale draft behind — a duplicate
reply-all draft going out twice is the failure mode that matters here.

## 3. Preserve-and-extend editing rule

When the user says "the rest of the email is fine":

- Open the existing `body.html` from the prior job dir
  (`/data/hermes/tmp/jobs/<job_id>/body.html`) and edit it. Do not regenerate prose.
- Insert the new content as a **new numbered section** and renumber the rest
  (1..n). Do not append it to the bottom, and do not fold it into an unrelated
  section where it loses weight.
- Every figure, table row, UTR and name from v1 must survive verbatim. Re-check
  them programmatically, not by eye (see `scripts/verify_threaded_draft.py`).
- Keep the same To/Cc set from v1 — reply-all membership does not change because
  the body changed.

## 4. Formatting a point the user calls CRITICAL

When NDR says "this is critical", "highlight this specifically", "otherwise we
will get into trouble", the words alone are not enough — the email must look like
it. Pattern that worked:

- A single-cell `<table>` with `border:2px solid #b00020; background:#fdf1f2`
  (red rule, pale red fill) placed **second**, immediately after the opening
  permission/agreement paragraph — high in the email, not at the end.
- Heading in UPPERCASE bold with the number: `2. CRITICAL — KYC AND BLANK CHEQUE
  MUST COME FROM THE PURCHASER BEFORE THE SALE DEED IS EXECUTED`.
- Sub-points lettered (a), (b) — one requirement each, in the user's own phrasing.
- A paragraph explaining **why it cannot wait**, in consequence terms: what we lose
  if it is skipped, expressed from the company's exposure ("once the deed is
  executed we have given away the asset and hold no hold left").
- A **stop condition**: "if <bad case>, stop and tell me before doing anything
  further — do not execute on an explanation, do not accept an indemnity in place
  of the document."
- Restate the point in the closing paragraph and in the final call-to-action line.
  NDR's critical points get said twice.

## 5. Cross-link the new requirement to the finding it answers

If the requirement exists because of something you established earlier (e.g. the
ledger carries the payer's name only, with no remitter account or UTR), say so in
the section that discusses that finding — "that is precisely the gap the blank
cheque in point 2 is meant to close". This turns a checklist item into a reasoned
instruction and is what NDR actually wants to see. It also stops the recipient
treating the new ask as bureaucracy.

## 6. Verify before reporting, and deliver a readable artifact

Run `scripts/verify_threaded_draft.py <draft_id> <thread_id>` — it asserts thread
match, exactly one live draft in the thread, header To/Cc, and a keyword list.

Then produce a plain-text rendering of the HTML next to it
(`draft_body_v2.txt`) and deliver both, because the chat surface is plain text and
NDR reviews the words, not the markup. Report honestly that nothing was sent.

## Pitfalls

- Reporting "draft updated" without re-reading it from the server. Always fetch the
  new draft with `format='raw'` and decode.
- Re-creating on a *different* `message_id` (e.g. the thread's first message rather
  than the latest inbound) — the draft then stops being a reply in the thread.
- Answering the native `gws_gmail_draft_create` path in a job session when a
  threaded reply-all is required. Native draft_create does not thread.
- Trusting the HTML source for the final wording check; strip tags and search the
  rendered text for each figure.
