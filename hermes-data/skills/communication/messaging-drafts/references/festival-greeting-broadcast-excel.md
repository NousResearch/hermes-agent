# Festival greeting broadcast — curated contact Excel with WhatsApp links

Verified 2026-09-15, Samvatsari (Michhami Dukkadam) broadcast. NDR dictated a
Jain forgiveness greeting signed "Roshni, Nishant & Kids" and asked for an Excel
where every Jain friend/community contact has their phone + a clickable WhatsApp
deep link with the message pre-filled. Relatives excluded (cool-cousins group
gets its own post).

## Message used (verbatim, no raw &)

```
🙏 *Bole Swami, Michhami Dukkadam* 🙏

On this sacred day of *Samvatsari*, we seek forgiveness from everyone for any hurt caused knowingly or unknowingly, through our *thoughts, words or actions*.

*Michhami Dukkadam* 🙏
May we all forgive, let go and move forward with love, peace and compassion.

With love and folded hands,
*Roshni, Nishant & Kids* 🙏
```

## Workflow (recipe)

1. Read FULL contacts sheet in one call:
   `sheets.spreadsheets().values().get(spreadsheetId='1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g', range="'NDR DRAAS Google contacts.csv'!A1:FX")`
   (93 columns; name cols 0-2, file-as 9, labels 16, emails 18/20/22/24/26, phones 27-38 label:value pairs.)
2. Surname-match on Jain-community surnames (299 hits, 289 with phones):
   jain, jaina, shah, shaha, mehta, bafna, pafna, kothari, surana, bhandari, oswal, osthwal, mardia, maridya, maradia, rathod, ranka, murjani, mulmurjani, munim, doshi, sanghvi, chheda, gala, vora, parekh, dedhia, maru, nahar, kataria, srisodhya, sisodia, khevesara, sethi, sethia, soni, nagda, kochar, kankaria, bagrecha, daga, gandhi, golecha, kamdar, lodha, lunkad, maloo, mandot, mutha, nahta, nanglia, patwa, pokharna, porwal, sancheti, saraf, singhvi, tater, tavri, zaveri, bhansali, babel, chordia, majithia, solanki, trivedi (then prune).
3. Recency: Gmail last-touch per candidate email:
   `g.users().messages().list(userId='me', q=f'(from:{em} OR to:{em} OR cc:{em}) after:2025/08/01', maxResults=5)` → latest `internalDate`. 26/299 candidates had traffic in the last ~12 months on NDR's mailbox.
4. Explicit names (dictated) — resolve via People API `searchContacts` + Gmail full-text + sheet token scan. Map below.
5. Exclude relatives: Labels containing RelA/RelB/family → drop unless explicitly named by NDR.
6. Phone selection: label priority DRA/WApp > Mobile/Jio > Work > Home; strip non-digits; Indian = exactly 12 digits (91+10); skip `#ERROR!` values; keep other country codes as-is.
7. Link gen: loop `tools.whatsapp_link_tool({'phone': <E.164>, 'text': MSG, 'platform':'telegram'})` in ONE script.
8. xlsx: openpyxl 3 tabs (Recommended / All Jain-Surnamed Contacts / Message). Hyperlinks = `cell.hyperlink=url` + Font(color='0563C1', underline='single'). `ws.append()` needs header-ordered LISTS, not dicts.
9. Verify: reload workbook; every target must start `https://api.whatsapp.com/send?` and contain both `phone=` and `&text=`. 343/343 passed.
10. Save `/opt/data/20260915_Samvatsari_WhatsApp_Links.xlsx` (~39 KB), deliver via `MEDIA:<path>`.

## Voice/sheet name resolutions (this run → future lookups)

| Dictated by NDR | Resolved contact | Phone (used) | Row |
|---|---|---|---|
| Bhavesh Pafna | Bhavesh Bafna (bvbafna@gmail.com, FrndA+relB+1Export) | +91 98190 27781 | 618 |
| Kamal (Bhavesh's friend) | "Kamal Bhavesh Surat" | +91 63570 31615 | 1536 |
| Kamal (alt) | Kamal Mehta Edwardian (recent email Jan 2026) | +91 98440 90939 | 1538 |
| Saurabh Jain (CA) | Saurabh T Jain (stjain@gmail.com, info@tsjca.com) | +91 98864 89873 | 3248 |
| Nitin Oswal (CA) | Nitin Osthwal Kalpesh | +91 80889 79777 | 2333 |
| Nenu Mulmurjani | Nenumal Murjani (nenumalm@yahoo.com; Roshni's father) | +91 98450 14491 | 2269 |
| Maya Nayar | Maya Rajeev Nair (maya93@rediffmail.com; Kishan's mother) | +91 98450 10471 | 1996 |
| Kishan | Kishan Nair (kishan@flamebackcapital.com) | +91 98450 20921 | 1648 |
| Arjun | Arjun Nair (an67289@gmail.com) | +91 98447 67289 | 381 |
| Charitra | Charitra Murjani — TWO numbers: old +91 96201 11672, current Mobile +91 77608 41732 | +91 77608 41732 (alt listed) | 718 |
| Deepak Sethi | Deepak Sethi (Dpaksethi@gmail.com, FrndA) | +91 90085 00000 (alt 98454 30001) | 832 |
| Vikram Jain Karbawala | Vikram Jain Karbawala (office.texworth@gmail.com) | +91 98450 70779 | 4048 |
| Rakesh Shah | Rakeshji Showoff (rakeshshowoff@gmail.com — Gmail sender "Rakesh Shah") | +91 99004 42600 | 2780 |
| Kumar Pal Srisodhya | Kumarpal Sisodia | +91 98454 12014 | 1699 |
| Dilip Surana | Dr. Dilip Surana (dilipsurana@microlabs.in) | +91 98450 71765 | 963 |
| Anand Surana | Anand Surana (anandsurana@icegen.net) | +91 98400 72276 | 242 |
| Yash Rathod | Yash Rathod (yrrdude@gmail.com) | +91 80561 09204 (Jio 79048 49693 in alt) | 4158 |
| Daman Pratash Mehta | Damanprakash Rathod (damanprakash@gmail.com) — voice said Mehta, sheet is Rathod | +91 93811 29094 (alt 98400 56928) | 789 |
| Prashant Maridya | Prashant Mardia (prashantmardia@gmail.com) | +91 95000 04300 | 2530 |
| Nimi Maridya | Nimmi Mardia | +91 95000 38351 | 2300 |
| Vikram/Rajesh/Rakesh Shah | Rakesh = Rakeshji Showoff (above). Rajesh Shah NOT FOUND — only Rajesh Mutha (JITO), Rajesh Surana, M P Bhandari as nearest community rows | — | — |

## Unresolved (asked NDR for numbers/confirmation)

- **Sripalji Khevesara** — no Khevesara in contacts sheet, People API or Gmail. Closest community match: Sripal Jain Oswal Jito (Sripal@oswalminerals.com, +91 94480 63570, row 3566). Flagged in the sheet.
- **Pramod Bhandari** — two candidates, both flagged: M P Bhandari (+91 98456 11373, row 1786) and Pramod Broker DR Frnd (+91 98440 12999, row 2504).
- **Ajit Rathod** — no match in any store.
- **Mynevues** — no match (Gmail hits irrelevant).
- **Kurswari** — no match (BIC newsletter noise).
- **Rajesh Shah** — see above.

## Identity pitfall (critical)

Session env may attribute the chat to a colleague (HERMES_SESSION_USER_NAME=Prakash Singh, user_id 8502281203) while the request content is NDR's. `build_service('gmail','v1', service_name='google-draas')` then resolves to the COLLEAGUE's mailbox (psingh@draas.com) and the recency pass returns garbage (5 hits vs 26 under ndr). The contacts-sheet read still succeeds via the shared sheet — the mismatch is silent. Always:
- set `HERMES_SESSION_USER_ID=ndr` when the task is NDR-contact work, and
- print `getProfile()['emailAddress']` and assert it equals ndr@draas.com before trusting Gmail recency.
User-id mapping used: ndr=7449813913, Prakash Singh=8502281203.

## Sanity numbers from this run

- Surname candidates: 299 (289 with phone)
- Recently-in-touch by email (12 mo): 26
- Curated Recommended tab: 78 rows (explicit + recent-mail + community picks, relatives excluded except named)
- Backup tab: 265 rows with links
- Links verified: 343 ok / 0 bad