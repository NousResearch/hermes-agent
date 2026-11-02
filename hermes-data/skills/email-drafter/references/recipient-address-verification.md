# Recipient Address Verification — NDR Rule (2026-09-23)

## The rule (verbatim intent, set after a bounce)

ALL emails must:
1. Be looked up — recipient must be **available in contacts** (contact sheet or Google
   Contacts via contact_resolver / GWS People API).
2. Have **previous emails sent to that person from that exact address** — i.e. seen in
   prior sent-mail on the thread or mailbox, NOT guessed.
3. **NOT have bounced.** If a previous send to that address bounced (550 address not
   found), NEVER reuse it. Re-lookup the contact and get the correct address from there.

The rule exists because a guessed address (arvindsarchitects@gmail.com) produced a hard
bounce on the Ranka North Star planning thread and the draft went out to a dead inbox.

## Procedure before composing ANY draft

1. `gws_resolve_account` → confirm the sending mailbox (default ndr@draas.com).
2. `contact_resolver` for each recipient — if not found, search Gmail history for that
   person's actual address in a non-bounced message (`from:<name>` or `to:<name>` scans).
3. If a thread exists: read the thread tail and copy the recipient addresses from the
   last messages IN that thread — never from memory.
4. Addresses in a draft must be verifiable: contacts entry OR prior non-bounced mail.
   If neither exists, flag it to NDR instead of guessing.

## Known-correct architect addresses (23 Sep 2026 verification)

- Arvind Jain (architect, AJ Architects) → `arch_arvind2000@yahoo.co.in` ✅
- Shrinidhi GV (Arvind's team, sends sanction files) → `shrinidhi.a.j.architects@gmail.com` ✅
  (contacts row 4249; voice often transcribes her as "Sriniti" / "Sri Niti" / "Sridevi")
- ❌ `arvindsarchitects@gmail.com` — DEAD (550 bounce 27 Sep 2026). Never reuse. Not in contacts.

## Bounce handling

- On ANY 550 / "address not found" / "mailbox does not exist" error: do not retry the same
  address. Re-resolve the contact; if the resolver has no email, search prior mail for a
  working address; if none, tell NDR the person has no verified address.

## Verification of the draft itself

After creating with draft_create, ALWAYS read the draft back (draft_get / drafts().list)
and confirm: recipients, subject, and that the body is HTML-not-dropped. Never report a
draft as "in Drafts" from the create call alone.