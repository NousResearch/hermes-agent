# Resolving voice-transcribed recipient names (ambiguous contact lookup)

When NDR dictates an email recipient (voice → text), the transcription can
diverge from the canonical contact-sheet name, and `contact_resolver`'s top
hit can be the WRONG person. Confirmed 2026-09-07 with the Riverstone
farmhouse presentation email.

## Confirmed failures

| Spoken | contact_resolver top hit | ACTUAL recipient | How we found them |
|--------|--------------------------|------------------|-------------------|
| "Amir" | Amir Ahmadi (Vanguard, US, +1 phones) | **Aamir Khan** (khan.hussain.aamir@gmail.com, Direct LLP — Katenahalli/Riverstone) | Session search ("farmhouse", "Riverstone") → entity is Direct LLP → Gmail thread "Proposed farmhouse presentation" |
| "Salman" | Salman Sayeed (STERLING construction) | **Salman Khalid** (salman@redificedevelopers.com, Direct LLP / TerraGreens LLP) | Gmail thread "Proposed farmhouse presentation" — address NOT in contacts sheet at all |
| "Manisha" (Indus Law) | Manisha Chajjer / manisha jain (old unrelated contacts) | **Maneesha Kongovi** (maneesha.kongovi@cms-induslaw.com, Partner CMS INDUSLAW) | Gmail thread "Re: Succession Certificate" (Sep 2026) — CMS INDUSLAW, not Indus Law |
| "Varvindh" | — | **Arvind Krishna G** (arvind.krishna@cms-induslaw.com, Senior Associate CMS INDUSLAW) | Same thread — he forwarded the matter to Maneesha; voice conflated two people into one name |
| "Justine" | — | **Jasleen Kaur** (jasleen@smritilegal.com, Smriti Legal) | Gmail thread "Sabyasachi Behera - RERA Case CMP/200628/0006093" (Sep 2026) |
| "Pingol Khan" | — | **Pingal Khan** (pingal.khan@ashlarlaw.com, Ashlar Law) — OUTSIDE the KHR Chambers firm | Akwagreen case thread — name his own counsel firm says; he must be EXCLUDED from event invites per NDR's same-domain rule |
| "Pujol, Kiribati Raji" (dictated BCC list right after "BCC Joel") | Pooja candidates — none school-related | **Joel Kribairaj only** (joel@aditi.edu.in) — "Pujol" was a *second garble of the same name*; no Pooja exists | User clarified "Not Pujol it's Joel". Dictated recipient lists can duplicate an earlier name — ask before inventing a second person when contact search comes up empty |
| "Manisha from Indus Law replied on behalf of Varvindh" | (no candidate) | **Maneesha Kongovi** (maneesha.kongovi@cms-induslaw.com, Partner, CMS INDUSLAW — Indus Law is now CMS INDUSLAW); "Varvindh" = **Arvind Krishna G** (arvind.krishna@cms-induslaw.com, Senior Associate) | Read the actual reply's From header + the quoted "Fw:" block inside her reply shows Arvind forwarded the thread to her |
| "Pujol" + "Kiribati Raji" (one BCC list) | Pooja Chandak (poojachandakk11@gmail.com — NOT school-related) | **Joel Kribairaj** (joel@aditi.edu.in) — BOTH dictated tokens were ONE person | NDR's one-word correction "Not Pujol it's Joel". He had named the same person twice via two mangled transcriptions; no Pooja existed anywhere near Aditi (checked contact sheet, People API, Gmail) |
| "BCC Pujol, Kiribati Raji" | — (both unresolved) | **No Pooja. BCC was Joel Kribairaj only** (joel@aditi.edu.in) | The SAME BCC name dictated twice, garbled: "Pujol, Kiribati Raji" = "Joel, Kribairaj". Contacts/Gmail/People search for a school "Pooja" returned nothing; NDR corrected "Not Pujol it's Joel" |
| "Manish" (dictated: "forward it to Manish and Mantha") | Manish Gulecha (Snapooh), Manish Chandok, Manish Jain — all unrelated | **Manish Ranka** (mdr@draas.com / mdr@drahomes.in — family, DRA) | People API search "Ranka" → full family list; the live email thread's Cc header already contained "Manish Ranka <mdr@draas.com>" |
| "Mantha" / "Mamtha" (same dictation) | none (Soma Samantha noise) | **Mamata (Mamta) Rathod** (maadi32@gmail.com / mamatadr@gmail.com — family; signs the Dinesh Ranka succession petition) | session_search "Mamtha" → 01-Sep succession dictation ("I will get Mamtha to sign it… she is ready to sign both the petition and the GPA"); the thread's Cc literally read "Mamta Rathod <maadi32@gmail.com>" — use THAT address variant, not the People API primary |

## Reply-all on a FORWARDED email

See `reply-all-forwarded-email-recipients.md` — reconstruct the real To/Cc from the embedded quoted headers ("Begin forwarded message:" block), rebuild Cc from earlier thread messages the forward dropped, thread to the forward's threadId with In-Reply-To = forward Message-ID and References = forward's References + Message-ID.

## Working recipe

1. **Never accept the top contact_resolver hit when multiple candidates exist
   and the name is a common/ambiguous one** — check `auto_selected: false` and
   the candidate org/phones for plausibility (a +1 US phone for a
   Bangalore farmhouse email is a red flag).
2. **Use `session_search` first** with the PROJECT/entity term from the voice
   note ("farmhouse", "Riverstone", "Katenahalli"). Past sessions reveal who
   is actually involved in that project and their canonical spelling.
3. **Search Gmail for a related thread** — `gmail.users().messages().list(
   userId='me', q='"<project term>" <name>')` and read From/To headers of the
   most recent thread. The existing correspondence is the ground truth for
   recipient emails, especially when the contact sheet has none (Salman's
   case).
4. **Re-run contact_resolver with the corrected full name + project context**
   to confirm the right row (e.g. "Aamir Khan" with context "Katenahalli
   Riverstone Direct LLP" scores 100).
5. **When the task is a reply/forward on an EXISTING thread, dump the current
   message's To/Cc headers FIRST — they are the ground truth for both identity
   and address variant.** Manish Ranka and Mamata Rathod were both already CC'd
   on the Behera thread as `mdr@draas.com` and `maadi32@gmail.com`, while People
   API lists alternate aliases (`mdr@drahomes.in`, `mamatadr@gmail.com`). Use
   the exact variant the thread uses for consistency — recipients who are
   already on the chain should never be re-addressed from a different alias.
6. Only then compose the draft. If ambiguity persists, ask NDR before
   drafting — do not guess.

## Pitfalls

- Voice often drops/duplicates vowels: "Amir" → Aamir (two A's). Also note
  the DRAAS contacts sheet has BOTH "Aamir Khan" AND "Aamir Khan Khan" rows;
  the Riverstone person is "Aamir Khan" (Direct LLP, khan.hussain.aamir@gmail.com
  primary; aamirkhan@icloud.com work).
- Salman Khalid's sheet row (row 3081) contains NO email column; his only
  working address comes from email history. Always check Gmail before
  declaring "no email found."
- Do NOT use the Contact resolver's "Other" candidates like "Aamir Khan Khan"
  (row 36, Direct LLP) just because org matches — verify against the actual
  thread's From/To headers.
- **Dictated recipient LIST can contain the same person twice, garbled (confirmed
  2026-09-09, Aditi speaker-session email):** when the first name is already
  verified (e.g. Joel Kribairaj) and a second dictated name ("Pujol", "Kiribati
  Raji") suspiciously resembles a mangled repeat of it, treat it as a possible
  duplicate BEFORE scoping out a new person. Empty results across contacts sheet +
  Gmail + People API is the signal to ask — but say "could this be the same person
  as X?" rather than hunting a phantom third recipient.