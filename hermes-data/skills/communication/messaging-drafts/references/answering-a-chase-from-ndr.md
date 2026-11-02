# Answering a chase from NDR on behalf of a reportee (Anbu)

## Trigger

NDR is chasing a reportee (usually Anbu, pm2.blr@draas.com) for a **firm date**,
**what is pending and why**, or **a position on closing / walking away** on some
matter. The reportee pastes NDR's WhatsApp message verbatim into Telegram with
no instruction of their own.

That paste means: *"help me answer this today."* Do not reply with a summary of
the message — reply with the answer.

## Rule zero — never invent the date, the pending list, or the position

The chase message contains no facts. Session history usually contains only the
chase itself (the cron that sent it), not the substance. **Do not**:

- pick a date out of thin air and present it as committed,
- assert what is "pending" from the wording of the chase,
- state a walk-away position as if it were NDR's decision already taken.

Reconstruct from the reportee's **own** mailbox and Drive, then label clearly:
`FACTS FROM RECORD` vs `RECOMMENDATION`. Where you cannot evidence an item,
leave one obviously-marked placeholder for the human — do not fill it.

## Step 1 — find out what the matter actually is

The chase uses a codename ("AHFL matters", "Oasis closure"). Resolve the codename
against the reportee's own Gmail before anything else:

```bash
cd /opt/hermes && HERMES_SESSION_USER_ID=pm2_blr env -u PYTHONPATH \
  /opt/hermes/.venv/bin/python3 -c "
import sys; sys.path.insert(0,'/opt/hermes')
from tools.gws_auth import build_service
svc=build_service('gmail','v1',service_name='google-draas')
for q in ['AHFL','AHFL landlord','AHFL society']:
    print('=== QUERY:',q)
    r=svc.users().messages().list(userId='me',q=q,maxResults=10).execute()
    for m in r.get('messages',[]):
        d=svc.users().messages().get(userId='me',id=m['id'],format='metadata',
            metadataHeaders=['Subject','From','Date']).execute()
        h={x['name']:x['value'] for x in d['payload']['headers']}
        print(' -',h.get('Date','')[:31],'|',h.get('From','')[:38],'|',h.get('Subject','')[:80])
"
```

Then, once a subject line is known, pull the whole thread and read the bodies
(the part-walker below handles multipart correctly):

```python
def walk(p, out):
    if p.get('parts'):
        for q in p['parts']: walk(q, out)
    else:
        data = p.get('body', {}).get('data')
        if data and p.get('mimeType') == 'text/plain':
            out.append(base64.urlsafe_b64decode(data).decode('utf-8', 'ignore'))
```

`format='metadata'` for triage, `format='full'` + `walk()` only for the 3–5
messages that matter. Never dump 200 bodies into context.

Also worth one pass each, same account:

- Drive: `name contains '<codename>'`, `fullText contains '<matter> settlement'`
  — finds the consolidated document folder, the document index sheet, and draft
  settlement deeds.
- Sheets: read the document index (`values().get`) and filter rows on keywords
  (`jda|landlord|association|parking|noc|occupancy|handover`) to prove what
  exists rather than guessing.

## Step 2 — the deliverable shape Anbu expects

Two artefacts, not one:

1. **Short WhatsApp copy-paste block** in a code fence — 6–10 lines, NDR's
   register, the date first, then the pending items, then the position. Anbu
   pastes this himself; do not generate a wa.me link for it.
2. **Threaded Gmail draft in Anbu's own inbox** — because NDR's standing
   instruction on matters like this is *"keep all communication on this email
   chain"*. Full detail lives here, threaded on NDR's most recent message.

Plus, in the Telegram reply itself: the recommendation with its reasoning, and
**one explicit closing ask** — the specific facts only Anbu holds.

## Step 3 — create the threaded draft in the REPORTEE's account

The native `gws_gmail_reply_draft` / `gws_gmail_draft_create` tools resolve to
the session's default account, which for Anbu's session is **not** necessarily
his. Scope it explicitly:

```bash
cd /opt/hermes && HERMES_SESSION_USER_ID=pm2_blr env -u PYTHONPATH \
  /opt/hermes/.venv/bin/python3 -c "
import sys; sys.path.insert(0,'/opt/hermes')
from tools.gws_skill_bridge import call
body=open('/opt/data/tmp_reply.txt').read()
print(call('draft_reply_create', service_name='google-draas',
           message_id='<id of the message being replied to>', body=body, html=False))
"
```

Write the body to a file first (`write_file`) — avoids shell-quoting damage on
em-dashes, ₹ and apostrophes.

**Verify before reporting.** `draft_created` alone is not proof:

```python
svc.users().getProfile(userId='me').execute()['emailAddress']   # must be pm2.blr@draas.com
d = svc.users().drafts().get(userId='me', id=draft_id, format='metadata').execute()
# check To == the intended recipient, Subject startswith 'Re:',
# In-Reply-To/References present, threadId == the original thread,
# labelIds == ['DRAFT']  (i.e. not sent)
```

Report the draft id, the account, the thread and the DRAFT label. Never say an
email was sent — it is a draft.

## Step 4 — pick the date defensibly

Do not offer "next week". Two constraints decide it:

1. **The counterparty's calendar** — an association/managing committee will not
   sit during a festival week. Check the Indian festival dates before proposing
   anything (Diwali 2026 = Sun 8 Nov; Dhanteras Fri 6 Nov; the 5-day window runs
   6–11 Nov). That collapses "sometime in November" into a real either/or:
   **before the festival, or after it** — state both windows explicitly and say
   which one you are holding and which is the fallback.
2. **Travel** — Dharwad/Hubli from Bangalore is a one-day trip (BLR→Hubli ~1h10
   flight, Hubli→Dharwad ~40 min). Propose: fly the evening before, meet late
   morning, return same evening.
3. **Prep days** — count the working days between today and the proposed date
   and state whether they are enough to close the pre-conditions. A date you
   cannot prep for is a repeat of the last failed meeting.

Check the reportee's own calendar (`calendar','v3', service_name='google-draas'`
with the same env var) for clashes before proposing.

## Step 5 — the position on closing / F&F / walking away

Structure it as three lines, no hedging:

- **Closable?** — yes/no, and only if specific pre-conditions are met by a named date.
- **Mandate** — recommend a ceiling and an opening number, marked as a
  recommendation for NDR to approve, not as a fact.
- **Walk away?** — recommend against it where a statutory/contractual obligation
  is still open (e.g. OC already granted ⇒ handover is the remaining obligation),
  and give the trigger at which to pause and revert instead of walking.

## Pitfalls

- **`entity_resolver` / `contact_resolver` can be unavailable in a non-NDR
  session** (no `gws_service` on the profile, or a handler kwarg error). Don't
  burn turns on them — go straight to Gmail/Drive search in the reportee's own
  account, which is the authoritative source anyway.
- **A near-miss document will mislead you.** `Stello B25 Full and Final
  Settlement` in Drive is a *buyer's flat-cancellation* letter (Mohammed Kaleem,
  ₹7,36,955), not the association settlement. Read the body before citing a file
  whose name matches the query.
- **"Landlords" ≠ "society".** On a JDA project these are two separate closures —
  the JDA landowners and the flat-owners' association. If the record only
  evidences the association side, say so and leave the landlord item as a marked
  placeholder.
- An unresolved cheque question from months back (NDR asked twice, never
  answered) is a real open item, not noise — surface it as the reportee's own
  lapse rather than burying it.
- NDR's standing instruction on such matters is often **verbal-only with the
  counterparty**. Reflect that in the draft ("I will confirm on phone today —
  nothing in writing to them") and never propose emailing the association.

## Reference facts — Ranka Stelo (AHFL, Dharwad)

- **AHFL = Associated Housing Finance & Investment Ltd** — the JV/development
  partner, not a landlord. "AHFL matters" = the Ranka Stelo file.
- Project: Ranka Stelo, Barakotri Road / Saptapur, Ward 14, Dharwad.
  R.S. No. 61A/1A/1/1. RERA PRM/KA/RERA/1259/410/PR/171016/000370.
- Sanction: HDMC No. 51/KAV/450/BID-22204/2015 dated 07-04-2015; Annex A site
  plan (plot 3,352.49 sqm, FAR 2.25, G+6 + cellar); OC 16-11-2019.
- Legal set: AHFL FILE 1–11 (JDAs 2013-07-01 and No. 2940 dated 2013-07-06,
  agreements for sale, deeds, declarations, ECs, RTCs); our JDA share registered
  2019-06-14.
- Consolidation: NDR's approvals folder + 225-row document index shared
  22/23-Sep-2026 (`AHFL Ranka Stelo Approvals - Document Index`, sheet id
  `1NiljVmFzetVpdGT9jy4k8BWMxXvX-lTR8_OobGtSn6I`).
- Open as at Oct 2026: no approved settlement ceiling (₹10 L offered, rejected;
  reportee estimates ₹20–25 L, no counter given); ₹10 L cheque unaccounted for
  3–6 months; physical originals unconfirmed (index is scans) and the 17-Jul
  handover list never delivered; car parking allocation letter not obtained;
  association resolution / legal letter not done.
- "Data sir" in NDR's dictation = **Datta Dorle**, the Ranka Stelo
  association-side contact. "Rajan" handed over the document set to Anbu.

## See also

`point-wise-reply-to-multi-item-review.md` — the companion case where the chase is instead a long
multi-item review mail (e.g. NDR's 13-invoice mail) that has to be answered item by item, in the
sender's own numbering. Same session pattern: reconstruct from the reportee's own mailbox, draft
in his account, verify the draft, never send.

