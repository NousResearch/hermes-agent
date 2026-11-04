# Voice-dictation names — resolve before you type them

NDR dictates by voice and the transcriber mangles Indian names, initials and titles. A dictated
name is NOT a fact. Never copy one straight into a To/Cc field or into the body of a draft.

## Resolution ladder (cheapest → most authoritative)

1. **NDR's own SENT mail headers — authoritative for addresses.** Search his mailbox for the
   real spelling/address, then read the thread. The contacts sheet is often stale or wrong
   (classic example: the sheet says `pm2.blr@draas.com` for Anbu, but he SENDS as
   `anbarasan@draas.com`).
   `gws_gmail_search(query="<garble> OR <guess>", service_name="google-draas")` → `gws_gmail_get` / `gws_gmail_thread_get`.
2. **Re-read the thread tail first.** Unresolved names are very often garbles of people who are
   ALREADY on the thread in front of you (a cc'd participant, the person being discussed).
   Hunt the wider world only after the thread is exhausted.
3. **Contacts spreadsheet — local grep, not the resolver.** `NDR DRAAS Google contacts`
   = `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`; tabs: contacts csv, projects, entities,
   land_proposals, topics, vocab_corrections, employees. `contact_resolver` returns only the
   top ~5 hits, so a row can exist and still come back empty — fetch `A1:Z5000` in Python and
   grep locally (4,263 rows).
4. **People API** — `people().searchContacts(query=..., readMask="names,emailAddresses,phoneNumbers,organizations")`
   as a fuzzy second opinion.
5. **Drive file/folder names** — project folders, family documents (e.g. a family-tree or
   succession-certificate doc names heirs).

If a lookup tool in this ladder errors out, fall through to the next rung — do not stall, and
do not conclude the person doesn't exist until rungs 1–5 have all come back empty.

## Confirmed garble → canonical (DRA, as at Oct 2026)

| Dictated | Actually | Notes |
|---|---|---|
| "Fyush" | **Piyush Ranka** | `piyush@draas.com` (also `piyush.r@drahomes.in`) |
| "Ambu" | **Anbu / Anbarasan Murugaperumal** | sends as `anbarasan@draas.com`, sheet says `pm2.blr@draas.com` |
| "Sanju uncle" | **Sanjeev Ranka** | `rankasanjeev@gmail.com` — how NDR refers to him in mail |
| "Heywan sir" / "him and sir" | **Hemant** | employee of DRA Developers & Projects Pvt Ltd (the Oct-2026 termination thread). The transcriber renders "Hemant sir" as "him and sir" |
| "Manish Mutkal" | **Munish Moudgil** | Special Commissioner (Revenue & IT), GBA/BBMP — `spcomm-rev@bbmp.gov.in`. The real name comes from NDR's own sent khata-intervention mails |
| "jelly" | **jelly work in the transformer yard** | Ranka Iris electrical scope |
| "Manisha" / "Manish" | both exist | transcriber conflates the two; Manish Ranka = `mdr@draas.com` (brother, Director DRA Projects), Manisha Chajjer = unrelated vendor contact |
| "Manisha Vivan", "Sanjay Akal" | **UNRESOLVED** (Oct 2026) | nothing by either name in contacts, People, Drive or any past mail — do not guess a near-match |
| "Vinodh" | **Vinod Kumar Das (Rahul / VK Das)** | NDR's **advocate**. `vkdas@draas.com` / `vkdas@drahomes.in` / `c2vdas@gmail.com`, +91 99000 93813. The look-alike `vinodhatadvocate.in` is a **DIFFERENT Vinodh and a WRONG address** — NDR mis-sent to it once (2026-10-18) and asked for it never to be reused. When NDR says "Vinodh" in the Anbu/Prakash context he means Vinodkumardas unless he names someone else |

## Protocol when a dictated name resolves nowhere

1. Do **not** invent it, do **not** guess the nearest-looking contact, do **not** silently drop
   it into an unqualified "as we discussed".
2. Build the deliverable with the resolvable parts (the other To/Cc recipients, the rest of the
   body). Omit the unresolved name from the sentence, or use a short bracketed placeholder.
3. Put a **short numbered flag list of unresolved names at the top of your final reply** — who
   they are, that they were left out, and what you need back. That list IS the ask.
4. `clarify` is not available in every execution context (API-server/web sessions). If it is
   unavailable, do not stall waiting for it — deliver the draft plus the flag list, so NDR can
   fix one line and send. A draft is never sent, so a flagged draft is strictly better than no
   draft.

## Related pitfall — ampersands in the Subject

`gws_gmail_draft_create` HTML-escapes the Subject, so a `&` reaches the header as `&amp;` and
displays literally in the recipient's inbox. Write subjects with `and` or `/` instead, then
verify with `drafts().get(format="metadata")` — `drafts().get(format="full")` on a draft can
return an empty `body` field, so verify the HTML part by walking the payload for `text/html`
and checking its length and tail.
