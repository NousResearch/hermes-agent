# Sharing a DRA Content artifact with an EXTERNAL reviewer

Validated 2026-09-28 (AJ Architects Earth House × Tapovana study, `DRA-ART-2026-000034`;
and the Tapovana DNA study, `DRA-ART-2026-000035`, shared to Aamir Khan + Arvind Jain).

This is the email-side counterpart to `dra-content-artifacts`. The artifact is the
deliverable; the email's job is to hand over ONE link and get the recipient in.

## The rule that costs a round-trip if you skip it

**A DRA Content artifact is viewed through Google sign-in. The address you granted is
the address they must sign in WITH.** A client whose contact email is a Yahoo / company
domain often signs in with a different Google account entirely — and then the grant
looks correct in the ACL while the recipient sees an access-denied page.

So the draft must ask. Wording that worked:

> The study is viewed through Google sign-in. Could you confirm this is also the email
> your Google account uses to sign in? If your Google sign-in differs from this address,
> sign-in will fail — send me that address and I'll grant access to it.

**Do not guess the Google identity from the contact email.** On this project the
architect's contact address was `arch_arvind2000@yahoo.co.in`, but he replied asking to
add **`arch.arvind2000@gmail.com`** — the Yahoo grant was live and useless for sign-in.
Grant BOTH when in doubt (the ACL tolerates it; a failed sign-in does not).

## Order of operations

1. **Resolve every recipient address first** — contact lookup, then confirm against a
   real message header. Never hand-type the domain.
2. **Grant the artifact access BEFORE creating the draft**, so the draft can state
   access is already in place. Default VIEWER / 365 days; never ADMIN and never a
   permanent grant unless explicitly asked.
3. **Read the permissions back** (`content_access_list`) and confirm the role + expiry
   on the record. The grant call returning success is not the verification.
4. **Create the draft** with the canonical URL (never a signed render URL), a one-line
   description of what the study contains, the sign-in confirmation ask, and what you
   want back from them.
5. **Verify the draft by listing drafts back** — and on this mailbox the body may be
   stored base64 inside `multipart/alternative`, so string-checking the raw MIME
   reports everything false. Read it via `format='full'` per SKILL.md.

## Sharing with more than one reviewer

When a later request comes in as *"give Aamir access as well"*, treat it as a fresh
grant on the SAME artifact — do not rebuild or republish. Resolve the person, grant
VIEWER, read the ACL back, and report the full access list so the user can see who
currently holds access. If a contact matches more than one plausible person, check the
project's own email thread for the name before granting to a guess.

## Related

- `dra-content-artifacts` — publishing, updating, the 200-file cap and sprite packing,
  and the live-render verification that must pass before any link goes out.
- `architect-design-partner-briefing.md` — the engagement-model brief itself.
