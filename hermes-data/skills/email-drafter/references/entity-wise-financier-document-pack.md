# Entity-wise document pack for lender / financier onboarding

Worked: V-Gro Finserv (vamshiraj@vgrofinserv.com) requesting promoter + project
documentation for DRA Realty Pvt Ltd, Sevaganapalli Land Partners, DRA Thindlu
Land Partners — Sep 2026.

## When this fires

A financier / NBFC / bank / lender's BD person mails asking for a standard
onboarding bundle — typically some subset of:

- 3-years audited balance sheet / financials
- Company & promoter profile
- Debt profile (usually as THEIR blank Excel format)
- Ongoing & upcoming project details (their blank Excel format)
- Existing sanction letter

…and names several **entities**. The deliverable is ONE threaded reply with a
**consolidated, entity-wise** document set — not five separate mails.

## ⚠️ CHECK THE PARENT FOLDER'S EXISTING SHARES BEFORE BUILDING A PACK INSIDE IT

Hit 2026-10-05 building the Ranka Oasis pack for Jiraaf. The pack parent was picked
as an existing project folder (`Ranka Oasis`, id `1WVcodN-Zub4AM5sMAsdPbJHyRakVBEXC`)
because it looked like the natural project home. **It already carried `reader`
grants for a DIFFERENT lender — three `vgrofinserv.com` users from the earlier
V-Gro pack.** Every folder and every file created inside it inherited those
grants, so the whole Jiraaf pack was quietly readable by V-Gro.

Verify empirically, do not assume My Drive does not inherit — it did here:

```python
# probe BEFORE building: create a throwaway folder in the intended parent
t = drive.files().create(body={'name':'__perm_probe_tmp',
      'mimeType':'application/vnd.google-apps.folder','parents':[PARENT]},
      fields='id').execute()
print([ (p.get('emailAddress') or p.get('type'))
        for p in drive.permissions().list(fileId=t['id'],
             fields='permissions(id,emailAddress,role)').execute()['permissions'] ])
drive.files().delete(fileId=t['id']).execute()   # always clean up the probe
```

Any address you do not recognise is a **pre-existing external share** — build the
pack somewhere clean (a fresh folder with no external grants) instead.

Fix if it already happened, in this order:

1. `files().update(fileId=ROOT, removeParents=<shared parent>)` — move the pack out.
2. Walk the tree (root → subfolders → files) and `permissions().delete()` every
   offending `permissionId`. Moving alone does NOT remove grants already applied.
3. Re-list permissions on the root, the subfolders **and a sample of files** —
   the files are what actually leak.
4. **Re-grant your OWN people afterwards.** The move drops every permission that
   was coming from the parent — including the internal reviewer's. Here
   `psingh@draas.com` (the reviewer) silently lost access the moment the pack was
   moved out, and the pack would have gone out with the reviewer unable to open
   it. Re-create the internal reader grants on the root **and every subfolder**
   after the move, then verify.

Confirmed working end state (2026-10-05): 293 nodes checked, zero foreign grants,
root carrying only `ndr@draas.com` (owner), `psingh@draas.com` (reader, internal)
and the two deal-intermediary readers with a 90-day expiry.

Also worth knowing: `permissions().create` on a `*.jiraaf.com` target fails with
`400 ... "The administrator for jiraaf.com has disabled the ability to receive
items from outside their organisation."` Other domains fail with `"there is no
Google Account associated with this email address, you must tick the Notify
people box"`. Both mean the same thing operationally — **flag it, do not silently
switch the pack to link-sharing**; whether to open the links is the user's call.

## Core pattern: share folders, don't attach

The instinct is to hunt down every file and attach them. Wrong here.

- N entities × ~10 docs each = 30–50 files. Blows the 25 MB Gmail cap instantly.
- The lender wants a **browsable pack they can forward to credit internally**, not
  an attachment blob.
- Build **one parent Drive folder + one subfolder per entity**, share the whole
  tree as `reader` with every recipient (plus `ndr@draas.com` where instructed),
  and put the folder links in an HTML body.

Sharing mechanics (permission body, `reader` not `viewer`, expiry rules) →
`references/drive-link-share-expiry.md`.

### Folder shape that worked

```
<Parent> — Consolidated Document Pack (<Mon YYYY>)
├── 1. <Entity A>            ← financials + firm/profile docs
├── 2. <Entity B>
├── 3. <Entity C>
├── 4. <Individual promoter 1>   ← identity + ITR + photo
├── 5. <Individual promoter 2>
└── 6. Detailed Project Reports
    ├── <Project 1>          ← DPR doc + slide deck + editable slides
    ├── <Project 2>
    └── <Project 3>
```

Numbered prefixes sort correctly and read as an index in the email. Individual
promoters get their own folder — they are not sub-items of a company folder.

## Build recipe

```python
# create folder
c = drive.files().create(body={'name': name,
        'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent]},
        fields='id,name', supportsAllDrives=True).execute()

# COPY the source file in — do NOT move it
c = drive.files().copy(fileId=src_id, body={'parents': [dest]},
        fields='id,name', supportsAllDrives=True).execute()
```

**Always `files().copy()`, never `files().update(addParents=...)`.** The source
documents live in project/firm folders that other workstreams depend on; moving
them breaks those. The pack is a view, not a relocation.

Then grant:
```python
drive.permissions().create(fileId=fid,
    body={'type':'user','role':'reader','emailAddress':u},
    sendNotificationEmail=False, supportsAllDrives=True,
    fields='id,role,emailAddress').execute()
```
Grant on the **parent AND every subfolder** — a recipient clicking a subfolder
link directly needs the grant on that folder id, not just the parent.

Sanity-check the grant: `ndr@draas.com` usually returns `role: owner` (it is the
folder owner, not a newly-added reader) — that is success, not a failure.

## Data-integrity rules (these are the ones that bite)

**Verify every source file is what its name claims before it goes in the pack.**
Pull the PDF and extract text; do not trust the filename.

- The DRA Realty FY2024-25 statement was an 11-page **scan with zero text layer**
  while the ITR sets had a text layer — report accurately rather than describing
  a scan as an audited financial statement.
- Individual ITR files named "…Ack…" are **acknowledgements only**, not the
  returns. Say so in the email if that is all that exists.
- Files whose name encodes a year need the year confirmed from the document body
  (acknowledgement number / AY line), not the filename.

**Partnership firms are usually not statutorily audited.** Do not silently
substitute nothing, and do not claim audited accounts exist. Enclose the ITR
filings and state the reason explicitly in the email so the counterparty cannot
come back later alleging an incomplete submission.

**Flag genuine gaps in the email rather than omitting them.** The pack that
worked had three, all surfaced to the user before sending:
- no live bank sanction letter for the entities → asked whether a group-entity
  sanction serves, or whether fresh credit is proposed
- their Debt Profile format sheet ambiguous on scope → asked whether they want
  it at entity / individual-promoter / group level
- their Ongoing & Upcoming Projects Excel not yet compiled → said it follows
  separately

Never volunteer **group-level** debt or financials when the ask names specific
entities. Scope creep on disclosure is the counterparty's call to make, not
ours — ask.

## User correction: entity attribution of firm documents

**A document can sit in the wrong entity's Drive folder.** In this session the
`20231121 Reconstitution Deed DRA Realty & NDR & Srinivas Krishnappa` was filed
under DRA Realty's directory but belongs to a **different** entity/firm. The
user caught it and asked for its removal.

Rules that follow:

1. **Read the deed's own party list, not its folder.** A reconstitution/partnership
   deed names the partners; the folder it happens to live in proves nothing.
   A deed naming "NDR & Srinivas Krishnappa" as the reconstituting parties is not
   automatically a DRA Realty document just because DRA Realty appears in the title.
2. **Look-alike names across entities.** Here three separate entities all carry
   near-identical wording: `DRA Realty Pvt Ltd`, `Sevaganapalli Land Partners`,
   `DRA Thindlu Land Partners`. Each has its OWN Partnership/Reconstitution Deed,
   PAN, TAN, GST and ITR series. Match the document to the entity whose partners
   and PAN appear inside it.
3. **When consolidating for a third party, verify each doc's entity attribution
   before it goes in** — a misfiled deed in a lender pack misrepresents the
   borrower's constitution.
4. Removing a file from a *pack* folder is a plain `files().delete()` on the
   **copy** — the original in the source folder is untouched. (This is the payoff
   for copying rather than moving.)

## Do NOT link the master investor/portfolio workbook into a lender pack

Mid-session the user asked to add the `DRA Group Investor Portfolio — All
Projects` spreadsheet (per-project tabs + a Project Summary tab) to the pack "for
project related detailed sheets." He then said **"Stop, lets remove this
spreadsheet from the email."**

The workbook was a reasonable-looking fit but was wrong to share: it is the
internal investor/equity portfolio model (`ndr@draas.com`-adjacent, owner
`nishantranka@gmail.com`) covering projects **wider** than the three being
financed — it carried a **Ranka NorthStar** tab that was NOT part of the
financing ask. Linking it would (a) disclose a fourth project's economics, and
(b) invite the counterparty to treat NorthStar as a financed project, which was
flagged to the user at the time and is what he then reverted.

Rules:

- **Share project DPRs, not the investor portfolio model.** DPRs are written for
  a third party; the portfolio workbook is internal equity/IRR working.
- **Scope-match every artifact to the entities/projects named in the ask.** If a
  workbook spans more projects than the financing request, it does not belong in
  the pack even if a tab matches.
- When you add an artifact that covers MORE than the ask, say so explicitly in
  your report and name the extra scope. Here the extra NorthStar tab was called
  out before the user sent — that is why the correction was cheap.

**Two-sided cleanup when the user says "remove this".** Undo both halves, and
say so:

1. Remove the body section and re-verify the HTML length returns to the
   pre-addition value (byte-identical is the strongest check — 9,359 here).
2. Undo the **Drive** side too — trash the shortcut/copy you created **and**
   revoke the `reader` grants you issued on the linked file. Removing text from
   an email while leaving an external party holding access to the underlying
   workbook is an incomplete revert.

```python
drive.files().update(fileId=shortcut_id, body={'trashed': True}).execute()
for p in drive.permissions().list(fileId=target_id,
        fields='permissions(id,emailAddress,role)').execute()['permissions']:
    if p.get('emailAddress','').endswith('<counterparty-domain>'):
        drive.permissions().delete(fileId=target_id, permissionId=p['id']).execute()
```

## A My Drive file cannot have two parents — shortcut, don't move

`files().update(fileId=..., addParents=<folder_id>)` on a file already parented
in My Drive fails:

```
HttpError 403 ... "Increasing the number of parents is not allowed"
(reason: cannotAddParent)
```

The file also shows `parents: []` under a different account's token, which makes
it look unparented and eligible for a move — it is not. To place an
existing-owned file inside a pack folder, create a **shortcut**:

```python
drive.files().create(body={
    'name': '<label>',
    'mimeType': 'application/vnd.google-apps.shortcut',
    'parents': [dest_folder_id],
    'shortcutDetails': {'targetId': target_file_id},
}, fields='id,name', supportsAllDrives=True).execute()
```

A shortcut keeps one underlying file (live edits propagate, no duplicate) and
the link in the email should still point at the **original file URL**. Do not
use `shortcutedFileTargetId` as a Drive query field — it is not queryable and
returns `400 Invalid Value`; list the folder and filter on
`shortcutDetails.targetId` client-side instead.

## Replying to a DIFFERENT participant on a pack thread — rewrite the body, don't just change To/Cc

After the pack went out, a second person on the thread (Sudheer, the ED) replied
with two concrete asks. The temptation is to reuse the sent body, swap the
salutation and repoint `In-Reply-To`. That produces a **self-contradicting
mail** and the user will catch it.

Scan the reused body for every sentence that describes the pack as being sent
**now**, and rewrite them as referring to the earlier mail:

| Stale (wrong on a follow-up) | Rewrite |
|---|---|
| "As requested, I am sharing the documentation…" | "…the documentation shared earlier on this thread is set out below." |
| "Each folder has been shared with you, Mr. Praveen and Mr. Sudheer with viewer access" | "The folders carry viewer access for you, Mr. Praveen and Mr. Vamshi" |
| "the detailed project reports **enclosed above**" | keep, or "shared earlier on this thread" |

Two specific traps:

- **Third-person self-reference.** The sent version named Sudheer in the third
  person inside a sentence about the pack. As a reply *addressed to Sudheer*
  that reads badly — and the recipient list changes too (on the follow-up the
  original asker moves from `To` to `Cc`; the replier takes `To`).
- **Change the salutation.** `Dear Vamshi` → `Dear Sudheer`. Trivial, and the
  easiest thing to forget.

Then anchor the reply correctly: `In-Reply-To` = the **replier's** Message-ID;
`References` = their existing References chain **plus** their Message-ID
appended. Both belong on the draft or it lands in its own thread.

## A follow-up on a pack thread carries ONLY the answer — never re-send the pack

**User rule (explicit, 2026-09-21):** *"as we had already sent the folders and
other requirements in the courtesy mail response. In this email lets reply with
only what is asked for."*

Once the pack has gone out on the thread, a later reply to a participant is a
**narrow answer**, not a re-issue. Cut the whole document apparatus:

| Remove | Keep |
|---|---|
| The consolidated-folders summary table + all folder links | The financing requirement table (it is context for the ask) |
| The per-entity `<h3>`/`<ul>` document listings | The answer box, one bullet per question asked |
| The DPR sub-list with its three folder links | A single refer-back sentence |
| "The Google Drive folders carry all documents listed above…" | The closing courtesy line |
| Any "To summarise" restatement | — |

Target is **zero Drive links and no attachments** in the body. Verify by
counting (`html.count("drive.google.com") == 0`) — that counter is the cheapest
proof you did not re-send anything.

The trap this rule fixes: trimming the *table* but leaving the prose restating
what was sent. Every sentence describing the pack must go, not just the links.
Scan for `"enclosed above"`, `"I am sharing"`, `"shared with you"`,
`"Google Drive folders carry"` and rewrite or delete each — a document that is
not in *this* mail must never be described as enclosed.

Related correction in the same session, same principle: the pack's standalone
one-off lines (an uncompiled Excel promised "separately") should collapse into
**"the detailed project reports for each project, already shared, may be
referred to for the project particulars"** — the previously-sent DPRs *are* the
answer; do not re-promise a separate deliverable you have already substantively
delivered.

## Use the user's actual financing facts — do not reason them out

The debt-profile/sanctions position is **bundled information only the user
holds**. When the reply needed it, the user supplied the substance directly:
financing is **project-wise**; the project entities and the parent hold **no
existing bank sanctions** (so no sanction letter exists, and fresh credit is the
ask); **APF (project pre-approval) is held for two of the three projects**
(Udaya, Amber — not Oasis) and can be furnished on request.

Do **not** infer or soften this. Observable guardrails:

- Never state or imply a sanction letter exists if the user has not said so —
  and never claim a project *has* an APF you were not told about.
- When the user gives you a position, encode it verbatim as the bullet; the
  phrasing here is close to their own words on purpose.
- Distinguish the **entity-level** position ("the entities hold no sanctions")
  from a **group-level** one — they are not interchangeable to a lender, and
  conflating them misrepresents the borrower.
- The parent-entity relationship (**DRA Realty Pvt Ltd as parent for the
  projects**) is itself a fact the counterparty needs; lead the debt-profile
  bullet with it.

## Record the APF / sanctions answer shape for future lender replies

Reusable skeleton for a lender follow-up once the pack is out:

```
<li><b>Debt Profile — Project-wise.</b> Confirm the financing is project-wise;
    name the parent entity; say the profile will be furnished in that format.</li>
<li><b>Existing Sanctions.</b> The project entities hold no sanctions of their
    own, and neither does <parent entity>. No sanction letter to furnish;
    the requirement is to arrange fresh credit.</li>
<li><b>APF (Project Pre-Approval).</b> Name which projects hold APF and which
    do not; offer to furnish the approvals on request.</li>
```

Keep unrelated open points (financial statements position, project details) in
the same box but as their own bullets — a partly-answered box is correct.

## A follow-up answer replaces the open question it closes

The pack's amber "Points to note" box asked the counterparty to choose a
debt-profile scope. The reply answered it — so the ask must be **replaced by the
answer**, not left standing beneath it:

- Retitle the box from "Points to note on the documentation" to something
  answering-shaped, e.g. **"Debt Profile & Sanctions — our position"**.
- Convert each ask into a statement of position.
- Keep an untouched gap (the uncompiled Excel) as-is — a partly-answered box is
  correct; do not delete the whole box.
- Add a one-line closing summary restating the resolved position. The user's
  counterparty skims; the summary is what they forward internally.

## Record the substance of a counterparty's reply before drafting

When a participant answers mid-thread, read their mail from the thread with
`threads().get(format='full')` and extract: the decision they made, their exact
wording, and their Message-ID for threading. Report both a **typo/diction flag**
and an **open item** separately to the user — e.g. Sudheer wrote "**debtor**
profile" where he plainly meant **debt** profile (matching the original ask);
draft on the intended meaning but tell the user the word used. If he genuinely
wanted a receivables/debtor schedule that is a different document, and only the
user can judge.

## Email body shape

HTML, inline CSS only. Structure that worked:

1. One-line thanks; skip echoing the counterparty's marketing boilerplate.
2. **Summary table** — `Entity | Documents folder` with each folder as a
   clickable link, plus a parent-folder link underneath.
3. One `<h3>` + `<ul>` per entity, listing what is in that folder.
4. **Financing requirement as a highlighted callout** when the ask is
   credit-driven — a coloured box (`#e8f0fe` bg / `#0b57d0` left border) with a
   project-wise table and the consolidated figure in a `#fff3cd` highlight span.
   Put it **after** the DPR/report section so the reports read as substantiation.
5. **Amber "Points to note" box** (`#fff8c5` bg / `#bf8700` border) carrying the
   gaps-and-questions list.
6. Close with an offer of alternate formats + a request for an early discussion.

Never use a bare URL in the body — every Drive reference is an `<a href>`.

## Threading and mailbox

Two traps hit in this session, both worth pre-empting:

**The thread may only exist in the forwarder's mailbox, not the original
recipient's.** Prakash forwarded Vamshi's mail to NDR, so NDR's copy had a
thread id and message id that **404 on Prakash's mailbox**. The original Vamshi
message (with the real thread) was separately in Prakash's own mailbox. Search
the drafting mailbox itself for the original sender before anchoring:

```python
svc.users().messages().list(userId='me', q='<sender-domain> OR <sender-name>')
```
Then anchor the draft to the **original sender's own message id** in that
mailbox, not the forward.

**Draft in the mailbox the user asked for.** The signature-derived default is
`ndr@draas.com`, but this request explicitly said the draft is Prakash's. Set
`HERMES_SESSION_USER_ID=psingh` and **assert the mailbox** before writing:

```python
who = svc.users().getProfile(userId='me').execute()['emailAddress']
assert who == 'psingh@draas.com', f'WRONG MAILBOX: {who}'
```
A draft landing in the wrong mailbox looks like success and is a data-isolation
breach. Also cross-check the OTHER mailbox for stray copies afterwards — the
correct outcome is exactly ONE draft in the target account and zero elsewhere.

## Updating the draft across revision rounds

Expect several rounds (add a folder, remove a doc, add figures). Rebuild cleanly:

1. Delete the previous draft by its **draft resource id** (`r-…`), not the
   message id.
2. Re-create from the original sender's message id with `threadId` preserved.
3. Verify: `labelIds == ['DRAFT']`, correct `From`/`To`/`Cc`, HTML length, and
   that every Drive link in the body resolves to a folder that actually exists.

Leaving the superseded draft behind creates a duplicate the user may send twice —
always delete before recreate, then confirm the account holds exactly one draft.

## Closing report to the user

State the folder tree with links, what changed this round, and — separately and
plainly — **the gaps and the open questions**, since those are what the user has
to act on before sending. Do not bury them in the success summary.
