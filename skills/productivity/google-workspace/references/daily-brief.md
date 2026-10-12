# Daily Brief (Gmail + Calendar)

Produce an action-oriented start-of-day or next-day brief from Gmail and Google Calendar. Load this reference when the user asks for a morning brief, "what's on my calendar and what email needs attention," meeting preparation, or tomorrow's deadlines and conflicts. The main SKILL.md owns the commands; this reference owns the brief-composition procedure.

Credit: workflow contributed by Ben Barclay (benbarclay).

## Procedure

### 1. Resolve day and identity

Confirm Google account, timezone, and target local day. Use an explicit half-open window `[day_start, next_day_start)` in the account's timezone rather than vague "today" filters — the account timezone and the machine timezone are frequently different. Done when the exact UTC and local window are stated.

### 2. Fetch calendar events

Retrieve all calendars in scope, including accepted and tentative meetings, all-day events, travel/holds, location/video links, organizers, and attendee status. Detect overlaps and unrealistic travel gaps between consecutive events. Done when pagination is complete and declined/cancelled events are excluded intentionally.

### 3. Fetch relevant Gmail threads

Search a bounded recent window plus messages connected to meeting participants, subjects, projects, and explicit deadlines (see `gmail-search-syntax.md` for operators). Read full relevant threads. Do not dump every unread newsletter into the brief. Done when each included email changes preparation, priority, or follow-up.

### 4. Link mail to meetings

Match by thread references, participant addresses, company/domain, event title, and project context. Treat fuzzy matches as suggestions, not facts — one shared keyword is not an association. Extract promised documents, unanswered questions, pre-read links, and decisions needed. Done when each meeting has either preparation items or an explicit "no preparation found."

### 5. Build the brief

Use this order:

1. Schedule at a glance
2. Conflicts and tight transitions
3. Meetings requiring preparation
4. Urgent mail and deadlines
5. Follow-ups owed by the user
6. Waiting on others
7. Data coverage or connector failures

Rank by consequence and time, not message count. Done when each included item has a clear preparation, deadline, conflict, or follow-up reason.

### 6. Offer bounded actions

Draft replies, create calendar holds, or add tasks only after presenting them — a brief request is not authorization to mutate. Apply approved actions with the main skill's commands, then read them back. Done when every approved mutation has a Google object ID/link and correct time/recipient.

## Pitfalls

- Mixing account timezone with the machine timezone.
- Hiding all-day commitments below timed meetings.
- Treating tentative meetings as confirmed.
- Associating an email to a meeting from one shared keyword alone.
- Creating calendar events while the user only requested a brief.

## Verification

- [ ] The day window and calendars covered are stated, or gaps are named.
- [ ] Every prep item, deadline, and conflict traces to a specific event or thread.
- [ ] No mutation happened without presentation and approval; approved writes were read back.
