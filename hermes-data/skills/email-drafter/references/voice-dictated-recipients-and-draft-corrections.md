# Voice-dictated recipients, and correcting a draft that is already created

## 1. Never let a dictated name reach To:/Cc: or a greeting unverified
NDR dictates replies while walking; names get mangled or invented by STT. Two live cases:

- "hi Nabeel and **Rakesh**" — there was no Rakesh: not on the thread, not at PropLion in
  contacts. NDR's follow-up: *"There is no Rakesh. That might have been a transcription
  error... Prakash is my person."*
- "his colleague, **Sririni**" — the real person is **Shrinidhi GV**, Arvind's office
  (`shrinidhi.a.j.architects@gmail.com`).

Rule: before a dictated name goes into the body or headers, check it against (a) the thread's
existing participants, and (b) `contact_resolver`. If it matches neither, **do not add it**.
Greet only the confirmed party, flag the uncertain name to NDR in the reply ("no Rakesh on
this thread or in contacts — send me his email if he should be on copy"), and let him decide.
A stray greeting name in a commercial mail is worse than a missing one.

Corollary: a dictated name that *does* match an internal colleague is usually a CC, not a
name to greet. Don't invent a second addressee to explain it.

## 2. There is no draft-edit tool — correct by delete + recreate
`gws_gmail_reply_draft` / `gws_gmail_draft_create` only create. To fix wording in a draft
you already created:

```python
svc.users().drafts().delete(userId="me", id=old_draft_id).execute()   # remove the superseded one
# then create the corrected draft and keep only the new id
```

Then list drafts to prove exactly one remains open on that thread (`drafts().list()`), so NDR
isn't left with two near-identical drafts to choose between. Report the superseding
explicitly ("old one deleted, only this remains").

## 3. Verify every draft by reading it back
`drafts().get(userId="me", id=<draft_id>, format="full")` → check `To`, `Cc`, `Subject`,
`In-Reply-To` before telling NDR it is ready. For a reply, `In-Reply-To` must be the message
being replied to (the *latest* inbound), not the thread root — that is what keeps it in the
same conversation.
