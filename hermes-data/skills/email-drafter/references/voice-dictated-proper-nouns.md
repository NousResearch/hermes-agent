# Voice-dictated proper nouns in an email body — verify against the referenced chain, then flag

Verified 2026-10-29 (NDR → Manish Ranka, Bcc Roshini: "MD selection - need your vote between Path 1
and Path 2").

NDR dictates emails by voice. Every **name, case name, project and entity** in the dictation is an
unverified token, and the failure mode is silent: the draft reads perfectly and names the wrong
person. The rule is NOT "keep it as dictated" (that is the rule for a *project name* in an
intro-connection mail, where nothing else can adjudicate it) and NOT "quietly fix it". It is:

> **The email chain the mail is about is the authority on names.** Read it, adopt its spelling, and
> list every change back to NDR in the completion summary with the evidence.

## 1. The procedure

1. Identify the chain the mail refers to. NDR says *"there is a chain of emails between X and me"* —
   that chain is sitting in the mailbox. Find it by the **job**, not the person:
   `gws_gmail_search(query='Dharmesh OR "flow plan" OR <distinctive phrase>')`, then
   `gws_gmail_thread_get` on the top hit and read the **tail** (the other side's latest message is
   what NDR is answering).
2. Diff every proper noun in the dictation against the thread. The other side's own email is the
   strongest evidence available — it is the same conversation, in writing, by the person being
   described.
3. Confirm the identity with `contact_resolver` (mailbox + org). Contacts hold the canonical
   spelling; Gmail headers hold the address they actually mail from.
4. Anything still unresolved: **keep as dictated, and flag it as unverified** — do not substitute a
   plausible-looking alternative. An invented case name in a board mail is far worse than a flagged
   one.

## 2. What it caught in this session

| Dictated | Correct | Evidence |
|---|---|---|
| "Kushal" (said 3×) | **Khushroo** — Khushroo Engineer, `khushroo@draas.com`, DRA Group, handles compliances | Dharmesh's 29-Oct email: *"I would like to propose Khushroo to be the outside party"*; contacts row 1627 |
| "these are heritage and equity matters" | **"Hermitage and Aquagreen matters"** | Dharmesh: *"On legal, it is Hermitage cases and Aquagreen matters"*; Prestige Hermitage khata thread; Ranka Aqua Green Owners Assn v. DRA, CC 1230/2016 (NCDRC) |
| "the ongoing Gandhinagar case" | **left as dictated, flagged** | no Gandhinagar matter anywhere in the mailbox; company-level litigation on record is Hermitage + Aquagreen only |

Note the asymmetry that decided it: "Kushal" is a *different plausible person* — leaving it in would
have made the mail incoherent to a recipient who had just read Dharmesh naming Khushroo. A name that
resolves to a real but wrong person must be corrected; a name that resolves to nothing must be
flagged.

## 3. How to report it (the summary is part of the deliverable)

Two explicit lists, never prose-mush:

- **Corrections I made — please confirm:** `dictated → used`, plus the one-line evidence for each.
- **Names I could not verify:** the exact phrase, and what the record *does* show instead.

NDR's standing rule applies: never assert a fact you did not read this turn. "I could not verify X"
is the correct sentence; "X is a typo for Y" is not, unless the record shows it.

## 4. Two build pitfalls that cost drafts in this session

- **A post-processing `str.replace` that does not match is a silent no-op.** The body was rebuilt to
  turn `" - "` into em dashes; the edit script's target string carried no blank line, the replace
  matched nothing, and the draft was created with the old text. It was caught only because the
  read-back asserted `plain.count("\u2014") == 0` → wrong. **After any string surgery on the body,
  assert the artefact (count of the character, presence of the phrase) in the read-back, not just
  that the script exited 0.**
- **Decide body typography before the first `drafts().create`.** Em dash vs hyphen is not worth a
  rebuild cycle: each rebuild is delete-old → create → re-read → re-verify, and it leaves
  `drafts().list()` needing a count check to prove no duplicate survived. Choose once, build once.

## 5. Recipient shape for a family/board vote mail

- NDR names the To and the **Bcc** explicitly ("Bcc Roshini"). Do not promote a Bcc to Cc and do not
  add the person being disagreed with to Cc because it "feels" like a three-way thread — the Bcc is
  the point of the mail.
- Subject: keep pure ASCII even when NDR writes an en dash — `MD selection – need your vote between
  Path 1 and Path 2` goes out as `MD selection - need your vote between Path 1 and Path 2` (plain
  hyphen), per §3 of `new-email-html-attachment-nonascii.md`.
- Body: NDR's dictated prose, paragraph for paragraph, in his voice. No bullet-point rewrite, no
  added headings, no editorial tightening — a board vote mail is an argument, and the argument is
  his. Em dashes in the body are fine (base64 parts carry them; only *headers* must stay ASCII).
