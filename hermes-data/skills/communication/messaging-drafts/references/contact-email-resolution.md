# Email-Address Resolution — HARD RULE

**NDR's standing rule (stated verbatim 2026-10-14): "any email must be looked up through
contact."** Never type an address from context, memory, or a plausible-looking pattern.

An invented recipient address is worse than a missing one: it silently sends a family /
financial / Kelsa document to a stranger, and on a **calendar invite** it also fails to
reach the person you were told to notify.

## The two sources — check BOTH, online Contacts wins

| Source | How | Trust |
|---|---|---|
| **Online Google Contacts** (People API) | `build_service('people','v1', service_name=...)` → `people().searchContacts(query=…)` or `people().connections().list()` | **Authoritative for emails.** This is what the phone and Gmail autocomplete use. |
| **DRAAS contacts Google Sheet** | Sheet `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, tab `NDR DRAAS Google contacts.csv` — see `google-contacts-sheet-access.md` | Useful, but **carries stale and phantom rows**. Never treat a sheet hit alone as proof. |

Report what each source returns, and **name the source** in your reply. If they disagree,
say so — that disagreement is itself the finding.

## Verified failure case (2026-10-14) — Roshini Ranka

A calendar invite was created with Roshini as an attendee at **`rivaanranka@gmail.com`**,
an address invented from a half-remembered note. NDR's correction:

> *"Please note Roshni Ranka at gmail is not her email address at all. Always we've said
> any email must be looked up through contact. She does not have a Roshni Ranka at
> gmail.com address. So please remove that completely."*

**Her actual email set — three addresses, and only three:**

- `rnr@draas.com`
- `rnr@drahomes.in`
- `rmurjani@gmail.com`

*Online Google Contacts (People API, `ndr@draas.com`) returned exactly those three and
nothing else.* The **Sheet row 2989 ("Roshini Ranka") was the source of the pollution** —
it additionally carried `roshiniranka@gmail.com` and `roshini.ranka@gmail.com`, neither of
which exists in Google Contacts and neither of which is hers.

That sheet row is also structurally corrupted, which is how three addresses ended up in
one cell:

- `E-mail 2 - Label` = `rnr@drahomes.in ::: rmurjani@gmail.com` (a *label* cell holding
  concatenated *addresses*)
- `E-mail 2 - Value` = `rnr@drahomes.in ::: roshiniranka@gmail.com ::: roshini.ranka@gmail.com`
- phone columns shifted: `Phone 1 - Label` = `Mobile` but `Phone 1 - Value` = `Phuket`;
  `Phone 2 - Label` = `+66 95 985 7012`

**Lesson:** a `:::`-joined cell in that sheet is a corruption marker. Do not read any
address out of it as authoritative — cross-check People API and, when in doubt, ask.

## Anagrams are a real hazard — confirm, don't infer

The two near-anagram addresses in this family are:

| Person | Real address |
|---|---|
| **Rivaan Ranka** (son, Std 7) | `rankarivaan@gmail.com` |
| **Ruhaan Ranka** (son) | `pebblyshark69@gmail.com` (also `ruhaanranka@gmail.com`, `ruhaan.ranka@gsuite.aditi.edu.in`) |

`rivaanranka@gmail.com` is **not** an address for anyone. Whenever two addresses differ
only by word order, state which is whose explicitly before using either.

## Recipe — resolve before you address anything

```python
import sys, os, json
sys.path.insert(0, "/opt/hermes"); os.chdir("/opt/hermes")
from tools.gws_auth import build_service

# Try spelling variants — the API is spelling-sensitive.
for acct in ("google-draas", "google-ahfl", "google-gmail"):
    try:
        ppl = build_service('people', 'v1', service_name=acct)
    except Exception as e:
        print(acct, "no token:", str(e)[:120]); continue
    for q in ("Roshni", "Roshini", "Ranka", "rmurjani"):   # name AND local-part
        res = ppl.people().searchContacts(
            query=q, readMask="names,emailAddresses").execute().get("results", [])
        for r in res:
            p = r["person"]
            print(acct, q,
                  [n.get("displayName") for n in p.get("names", [])],
                  [e.get("value") for e in p.get("emailAddresses", [])])
```

### Pitfalls in that recipe

- **`searchContacts(query="Roshni")` returned 0 hits while `query="Roshini"` worked.** A
  zero-hit search is NOT "no such contact" — retry with spelling variants, the last name,
  and the email local-part.
- **`people()` is not in the native `gws_*` tool list**, so this needs `build_service`.
  `service_name` is the vault key — resolve it via `gws_resolve_account`, never guess.
- When the person is not on the relevant account at all (e.g. `ndr@ahfl.in` had no
  Roshini Ranka contact, only an email-less "Roshni Mama"), say that plainly per account
  rather than reporting a single merged answer.
- Watch for **duplicate/near-duplicate contacts** across accounts — `nishantranka@gmail.com`
  carried a stray contact literally named **"Woh"** holding `rmurjani@gmail.com`.

## If a wrong address has already been used — fix it at the source

1. **Calendar invite:** `cal.events().patch(calendarId="primary", eventId=…, body={"attendees": [...]}, sendUpdates="all")` — replaces the whole attendee list, so pass the complete corrected list, and `sendUpdates="all"` also notifies the removed address that it's off the invite.
2. **Email draft/thread:** correct the To/Cc in the draft; if already sent, say so — do not silently re-send.
3. **Then fix the cause:** update memory AND the governing skill/reference so the bad
   address cannot resurface. For the sheet itself, offer NDR a cleanup (strip the phantom
   addresses, de-shift the label/value/phone columns) rather than editing his contact
   sheet unilaterally.
4. **Report the correction explicitly** — what was wrong, what it is now, and what was
   reversed. NDR tracks reversals (he reversed a Kelsa stage move the day before this).
