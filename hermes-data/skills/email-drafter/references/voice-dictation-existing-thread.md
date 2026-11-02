# Voice dictation → "on the existing thread" → resolve the thread FIRST

Verified 2026-09-15 (Ranka North Star email confusion cascade).

## The failure mode

NDR dictated a long project-timeline message. The assistant:
1. Guessed project = Palya, recipient = Arvind Jain → created a fresh draft.
2. NDR corrected: "no, for Prakash and Bharat, Ranka Northstar" → created ANOTHER fresh draft.
3. NDR corrected again: "this email was supposed to be a REPLY-ALL to the Sinchana email, addressing Alvin (Arvind), keeping Sinchana in the room."

Only then was the correct artifact discovered: the reply-all had **already been SENT the same day at 13:36 IST** (likely by a parallel api_server session), and the Bharat/JDA-registration email **already existed as a draft**. Two wrong drafts + one duplicate creation before the truth surfaced.

## Rules (in order)

1. **Search Gmail threads BEFORE composing.** When the dictation references an existing conversation — "reply all to the Sinchana email", "the above email", "on the existing thread", "the original September JDA" — find the thread first:
   - `threads().list` / `messages().list` with subject and participant queries
   - Read the full thread (`threads().get`) including To/Cc/Date/Message-ID per message
   - The thread title + participants decide the project AND the recipient set.

2. **Check Sent mail and Drafts before creating.** Parallel Hermes sessions (Telegram gateway + api_server) can already have created/sent the requested artifact. List recent drafts (`drafts().list`) and search Sent for today's messages on the suspect thread. If you find the exact artifact, STOP — report it to NDR rather than duplicating.

3. **Draft as a REPLY on the found thread**, never a fresh standalone email, unless NDR explicitly said "new email". Reply-all semantics: keep original To + participants, add any newly named CCs. Use In-Reply-To/References + threadId on `drafts().create`.

4. **Resolve voice-garbled terms from the actual thread, not memory.** Observed in this session:
   - "JDS" → **JDA** (Joint Development Agreement) — the addendum-to-JDA registration
   - "flow plan" → **floor plan**
   - "Alvin" / "Irwin" → **Arvind Jain** (architect, AJ Architects)
   - "Rampa Nortstar" → **Ranka North Star** (Allalsandra)
   Confirm the reading against the thread's subject/body before composing.

## Work emails for DRA colleagues — People API surname/org search beats the contacts sheet

The NDR DRAAS contacts sheet (1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g) often holds only
personal/Gmail addresses for employees. When NDR says "she has a @draas.com email, use her work
email", do `people().searchContacts()` by **surname + organization** — the work address surfaces there:
- Gowri Singh → content lead → `gsingh@draas.com` (mobile +91 98454 30607)
- Sai Neha Vaddadi ("Neha", content creator) → `nvaddadi@draas.com` (mobile +91 78993 98273; personal esotericarts.ani@gmail.com)
The People API result (`searchContacts(query='Vaddadi'...)`) carried org "DRA Realty Private Limited" + the @draas.com address while the sheet had no company email for her row.

## Terminal `&`-guard gotcha when creating HTML-entity drafts

Running a Gmail draft script inline as a heredoc fails when the HTML body contains entity
references (`&mdash;`, `&amp;`, `&nbsp;`): the foreground terminal guard treats any `&` as
shell backgrounding and kills the run (exit 130 / "uses '&' backgrounding").
**Fix:** write the Python script to a file (`write_file`) and run `python3 /tmp/script.py` —
never inline-heredoc an HTML body containing raw `&` entities.

## Checklist before delivering a dictated email

- [ ] Thread found? (search Gmail by subject/participants first)
- [ ] Already exists in Sent/Drafts? (report instead of duplicating)
- [ ] Recipients = the thread's participants + dictated additions; work emails verified via People API
- [ ] Voice terms (JDS/JDA, floor plan, aliases) match the thread content
- [ ] Draft created as reply on thread (threadId) where NDR said "existing thread"/"reply all"