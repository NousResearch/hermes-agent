# Deadline + capacity pushback replies (colleague accepts extra scope, with a constraint)

**Trigger.** NDR asked a colleague (architect / coordinator) to take on more than their
nominal role. They replied *agreeing*, but hedged with a stated constraint ("happy to help,
but within my four hours a day"). At the same time a hard deadline has slipped and a **second**
person must be put on notice. NDR dictates a single reply covering all of it, and says
"reply all".

Verified working shape: Sinchana Gowda, thread *OASIS Site - Revised Saucer Drain Design and
Section Updates*, 19 Oct 2026 (drafted; NDR sends). HTML, full width, five banner sections.

## The five moves, in NDR's dictated order

1. **Thank + accept, without softening the ask.** "Thank you for your positive reply" — then
   restate what was actually asked, so the acceptance is not later re-readable as "I only
   agreed to review".
2. **Respect the constraint, then redirect INSIDE it.** "I understand the constraint on your
   hours and I respect that. I am not asking you to breach it. What I am asking is that within
   those hours the bias shifts towards closing deliverables rather than reviewing." NDR never
   argues with the constraint — he re-points the effort inside it. Do not propose a remedy that
   requires more hours.
3. **Turn his rhetorical question into a real question for the recipient.** When he says
   "hopefully it doesn't consume too many hours ... am I right?" he wants *her* to answer it.
   Write it as a question, and name the counterparties you actually found in the mailbox so she
   can confirm or correct the list rather than answer in the abstract.
4. **Escalate the deadline to the second person in a visually distinct block** (below).
5. **Daily-target discipline + one consolidated tracker** (below).

## Restating a slipped deadline ("Bhuvanesh please take note of this")

Make it its own coloured callout block mid-mail — never a sentence buried in a paragraph.
Red banner header + red left-border box containing only the deadline fact, three parts:

- the original target date, named ("our target date was last Friday")
- where we actually are now, named ("we have already crossed over into Monday")
- the new hard ceiling, non-negotiable ("I cannot, under any condition, go past this Friday")

Then immediately the mechanism: each morning name the items closing that day; each evening
report what actually closed and what slipped, with the reason; nothing open-ended, no item
without a named closing day.

## The tracking-sheet ask — resolve it BEFORE drafting

NDR says "the sheet where we are tracking deliverables ... from the master list that I had
created". Do not ask him which sheet, and do not ask the recipient blindly. Find it, then let
the ask carry the finding:

```
gws_drive_list(service_name="google-draas",
               query="name contains 'Deliverables' and mimeType='application/vnd.google-apps.spreadsheet'")
```

NDR-owned trackers show `owners: ["ndr@draas.com"]`. The tracker a colleague is working from is
often already linked in an earlier thread — subject-search it rather than guessing:

```
gws_gmail_search(service_name="google-draas", query="Deliverables checklist <project> <project>")
```

Read the tab list and the first rows before writing a word of the ask:

```python
svc  = build_service("sheets", "v4", service_name="google-draas")
meta = svc.spreadsheets().get(spreadsheetId=sid).execute()          # tab names
svc.spreadsheets().values().get(spreadsheetId=sid, range="'<tab>'!A1:J12").execute()
```

Then state what you actually found — e.g. "from what I can see the tracker currently carries
only Ranka Oasis and Ranka Amber" — and ask for links to the consolidated view. This turns a
vague request into a precise one and surfaces the real gap: a tracker with no ETA/owner column
cannot carry a day-by-day target. Report that gap to NDR in the delivery message; ask for the
links in the email.

## Recipients

Reply-all across the **whole thread**, not the last message's To — see
`reply-all-and-to-cc.md` ("latest message addressed narrowly").

## Quoting the constraint back

Write the colleague's number back exactly as *she* gave it ("four hours a day", "within those
hours"). Never estimate, round, or editorialise about someone's working hours in a mail that
they will read.

## Materials

`templates/html-reply-with-attachment.py` is the hand-built raw-MIME reply pattern used here
(minus the attachment). Subject carrying an en-dash: see `threaded-reply-raw-api.md`
("Non-ASCII subjects in hand-built MIME").
