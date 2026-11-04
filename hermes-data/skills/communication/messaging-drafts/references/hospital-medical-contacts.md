# Hospital / Medical Contacts (NDR context)

Recurring contacts cluster for Nishant's mother KDR's surgery and follow-ups. These are stored in Google Contacts under ndr@draas.com (google-draas) but may not be findable via `searchContacts` due to naming conventions — always fall back to `connections().list()` iteration when needed.

## Trustwell Hospital ecosystem

| Name in Contacts | Role | Phone | Searchable? |
|---|---|---|---|
| Dr. Deepak Haldipur | ENT Surgeon, Trustwell | +91 80 45666789 / +91 80 45666851 | ✅ Found via searchContacts('Haldipur') |
| Shridhar, operations, Dr. Haldipur, operation coordinator | OR coordinator for Dr. Haldipur's surgeries | ❓ (ask user) | ❌ NOT found via searchContacts — name is comma-heavy and not indexed. Scan connections list for "Shridhar" / "operations" / "Haldipur" in displayName. |
| Elumali Dr Haldipur Trustwell | OR staff / assistant at Trustwell | +91 99020 12550 | ✅ Found via searchContacts('Haldipur') — named "Elumali Dr Haldipur Trustwell" |
| Yashree Dharnath Srinivas | Contact on Dr. Haldipur's prescription pad (circled name/number) | 9449784569 | ❓ Not confirmed stored in contacts |

## Manipal Hospital ecosystem (verified 16-Aug-2026)

| Name in Contacts | Role | Phone | Notes |
|---|---|---|---|
| Juliet Sunita EA Ballal | EA to Dr. Sudarshan Ballal (Chairman), Manipal Hospital | +91 70224 29829 | ⚠️ 16-Aug-2026: NDR reported this number NOT on WhatsApp — check alternates before sending (see lookup pitfall below) |
| Sudarshan Ballal | Chairman, Manipal | +91 80493 60314 / +91 98804 00644 / +91 94496 49026 | the chairman himself |
| Dr. H S Ballal | MD, DMRD (Manipal University) | +91 82029 22498 / +91 82025 71911 / +91 82025 70062 | legacy/family |
| Dr. Shantala Kurtkoti | listed as "(Sudarshan Ballal)" in contacts | +91 99805 12330 | — |
| Dr. Amit Rauthan | Manipal (oncology) | not stored | NDR considered messaging him re: a patient — only name on file |
| Sandhiya | Customer Care Coordinator, Manipal Hospitals, HAL Airport Rd | 080 40119000 / 080 25023344 (ext 2633) / +91 91482 40487 | in "NDR CONTACTS" sheet (id `1fYa-t2RY1siy2qBgAH8uu_Jd2chjJ716BbcpxilpOK0`, tab `NDR CONTACTS` — only 16 rows; most contacts live in the 1XbSRA sheet) |

Related Manipal rows in the 1XbSRA contacts sheet (name-only, no stored phone): Amaan 21 ICU, Dr Faizal, Dr Niranjan Rai/Shetty, Dr Srinivasan (Haematology Lab), Dr Subba, Geetha Radiology, Jiten, Sunil Karanth (ICU Head), Suja, Nagarjuna Ambati, Reshmi Menon (Hearing), Seetha, Satyanarayan, Manipal Home Care (John Mathew), Mod Manipal hospitals.

**WhatsApp number not on WhatsApp — alternate lookup pitfall (16-Aug-2026):** when NDR says a stored number "doesn't seem to be on WhatsApp", do NOT just re-send the link. Check in order: (1) People API `connections().list()` scan for the person + close family/colleague variants (same surname, same org); (2) the 1XbSRA contacts sheet full-scan for related rows; (3) the `NDR CONTACTS` sheet (1fYa-t2RY...) for org-level customer-care numbers; (4) session_search for any alternate number used in past conversations. If nothing surfaces, report what related numbers DO exist (incl. the org's landline/customer care) and let NDR choose — never invent or web-search a number.

## Contact lookup strategy for hospital contacts

When the user says "send a message to [role] at [hospital/doctor]":

1. **Try `searchContacts` with the person's actual name** — ask user if you don't know it
2. **If searchContacts returns nothing**, use the `connections().list()` approach:
   ```python
   service = build_service('people', 'v1', service_name='google-draas')
   connections = service.people().connections().list(
       resourceName='people/me',
       pageSize=1000,
       personFields='names,phoneNumbers'
   ).execute()
   # Scan for partial name matches or role keywords
   for person in connections.get('connections', []):
       name = person.get('names', [{}])[0].get('displayName', '')
       if 'shridhar' in name.lower() or 'haldipur' in name.lower():
           phone = person.get('phoneNumbers', [{}])[0].get('value', '')
   ```
3. **Search session history** via `session_search` for the phone number if the contact was previously discussed in a conversation
4. **Still not found?** Ask the user — do not invent or web-search the number

## Tone for medical-context messages

- Hospital coordinators: direct, respectful, no pleasantries. Lead with the ask.
- Doctors/medical staff: use "Dr." prefix in message body, but address by first name in the conversation context if the user does.
- Insurance coordinators (Charan, etc.): factual, timeline-driven. They need dates and document names.
- "Mom" / "KDR" / "Kanta Ranka" — use the variant the user uses in the current message. The user switches between these freely.
