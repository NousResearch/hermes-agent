# Enriching a DRA contact "fully" (People API + contacts sheet)

**Trigger:** NDR says *"I have added a contact called X — can you enrich his contact
fully"*, *"add a note of the call we had with him"*, or drops a contact name with the
expectation that the whole history (company, deck review, call date, links) gets
attached. Validated 2026-09-28 (Ashish Thesia / Hesthetic Life Pvt Ltd).

"Enrich fully" means **all three**, in this order:

1. **Google People contact** (the store the user actually added him to)
2. **DRA contacts sheet row** (the store `contact_resolver` reads)
3. **The note itself** — reconstructed from real artefacts, never from memory alone

---

## Step 0 — Locate the person (People API first)

`contact_resolver` and a direct sheet scan can BOTH come back empty for a contact the
user has just added, because he added it to Google Contacts / his phone. Do not stop at
the sheet:

```bash
/opt/hermes/.venv/bin/python \
  "/data/hermes/skills/communication/messaging-drafts/scripts/find_contact.py" "Thesia"
# → "=== People API matches ===" prints the name + phone; "=== Sheet 1XbSRA ===" prints (no matching rows)
```

Then pin the exact resource:

```python
svc = ga.build_service("people", "v1", service_name="google-draas")   # NOT google-gmail
res = svc.people().searchContacts(
    query="Thesia",
    readMask="names,emailAddresses,phoneNumbers,organizations,biographies,addresses,urls,memberships",
).execute()
```

Search **all** accounts (`google-draas`, `google-gmail`, `google-ahfl`) — a contact can
live on any of them; only one will return the hit.

### Which account holds the contacts sheet

The DRA contacts sheet (`1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, tab
`NDR DRAAS Google contacts.csv`) is **`google-draas`**. Loading it with
`google-gmail` credentials returns **403 Forbidden** — switch the service, don't retry.

## Step 1 — Reconstruct the history BEFORE writing the note

The note is only as good as the sources. Sweep in this order:

1. `session_search` on the **person's name AND the company name** — the richest source by
   far. On this task a single search surfaced the whole thread: company registry data
   (CIN, incorporation date, directors), the pitch-deck review, the three-model
   investment thesis, the call-script doc, and the calendar/Meet invite.
2. `gws_gmail_search` on the person and the company (`Ashish Thesia OR Aesthetics`) — gets
   you the **full email signature**: mobile, designation, website, socials, and the
   company's exact legal entity name ("Hesthetic Life (P) Ltd").
3. Drive, for the deliverables worth citing in the note (deck, thesis, call script).

**Decode the garble.** NDR said *"Ashish Thesia of Aesthetics"* — there is no company
called Aesthetics; it is **Hesthetic**, a 100%-cold-pressed-oils brand in Surat. Put the
garble→real mapping in the note (`"Aesthetics" = Hesthetic`) so the next session resolves
it instantly, and resolve the entity's registered name from the signature, not from the
spoken name.

## Step 2 — Write the People API enrichment

`updateContact` needs the person's **`etag`**, so GET first, then PATCH. Only the fields
listed in `updatePersonFields` are touched — existing name and phone are preserved.

```python
res_name = "people/c6179664239932436745"
person = svc.people().get(resourceName=res_name,
    personFields="names,emailAddresses,phoneNumbers,organizations,urls,biographies").execute()

body = {
    "resourceName": res_name,
    "etag": person["etag"],                      # REQUIRED
    "emailAddresses": [{"value": "ashish@hesthetic.com", "type": "work"}],
    "organizations": [{"name": "Hesthetic Life Pvt Ltd", "title": "Co-founder & CEO",
                       "location": "Surat, Gujarat", "type": "work"}],
    "urls": [{"value": "https://www.hesthetic.com", "type": "work"}],
    "biographies": [{"value": NOTE, "contentType": "TEXT_PLAIN"}],
}
svc.people().updateContact(resourceName=res_name,
    updatePersonFields="emailAddresses,organizations,urls,biographies", body=body).execute()
```

A ~900-character plain-text biography is fine and is the right home for the call date,
the deck-review headline numbers, and the pending next step.

## Step 3 — Complete the sheet row (so `contact_resolver` finds him)

Headers are looked up by **name**; build a fixed-width row (81 columns in this sheet) and
fill only the indices that matter. For Ashish:

| index | header | value |
|---|---|---|
| 0 | First Name | `Ashish` |
| 2 | Last Name | `Thesia` |
| 9 | File As | `Ashish Thesia` |
| 10 / 11 | Organization Name / Title | `Hesthetic Life Pvt Ltd` / `Co-founder & CEO` |
| 14 | Notes | the full note |
| 17 / 18 | E-mail 1 - Label / Value | `Work` / `ashish@hesthetic.com` |
| 27 / 28 | Phone 1 - Label / Value | `Mobile` / `+91 99305 44141` |
| 39–46 | Address 1 | `Work` / `Surat, Gujarat, India` / … / `Surat` / … / `Gujarat` / … / `India` |

Then **immediately fix the phone cell with `valueInputOption=RAW`** — under USER_ENTERED a
`+91 …` value becomes `#ERROR!` (formula parsing). See
`references/google-contacts-sheet-access.md`. Read the row back and confirm the phone and
label sit in the right columns; a miscounted column letter silently swaps label and value.

## Step 4 — Reporting back

State plainly: which account/store each write landed in, the phone and email you recorded,
what the note contains, and — if a WhatsApp/E-mail follow-up was also drafted — that the
link/draft is **generated only**, not sent. NDR reads this as the audit trail for the
contact.

## Pitfalls

- **Don't create a second contact** when the person already exists in People API — enrich
  the existing `resourceName`; a duplicate splits the history.
- **Don't put the note only in People API.** `contact_resolver` reads the sheet, so a
  People-only note is invisible to every future lookup.
- **Don't "fix" the voice garble in the sheet's First/Last name** — the canonical name NDR
  typed (`Ashish Thesia`) is correct; the garble belongs in the note.
- **A contact row written near the end of a 4,000-row sheet may not appear in
  `contact_resolver` immediately** — the People API hit plus the read-back of the new row
  are the proof of completion; don't re-append.
