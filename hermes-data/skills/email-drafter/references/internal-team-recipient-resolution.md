# Internal-team recipient resolution (voice-dictated distribution lists)

Verified 2026-09-30 while drafting the DRA weekly-team-review email. NDR dictates
"marking Gowri, Anbu, Bharat, Prakash, Kantesh, Sinchana, Sartak..." and expects a
draft addressed correctly.

## Why not to guess
- `contact_resolver` returns ambiguous or **email-less** rows for partial first names
  (multiple Gowris/Prakashs/Bharats; many rows have no email field at all).
- Guessing `firstname@draas.com` and probing Gmail returns 0 hits for most guesses
  (`to:gauri@draas.com`, `to:tantesh@draas.com`, `to:siddhira@draas.com` all empty,
  checked 2026-09-30). Gmail `to:` probe: `gm.users().messages().list(q=f'to:{e}', maxResults=2)`.
- The contacts Sheets read can **429 per-user** (`ReadRequestsPerMinutePerUser` = 60/min,
  consumer project 353166898892) right when you need it — don't burn quota re-reading
  the whole contacts sheet during a recipient lookup.

## Known internal email map (tracker/memory-verified — use FIRST)
| Person | Email |
|---|---|
| Bharat Hawaldar | sales1.blr@draas.com |
| Prakash Singh | psingh@draas.com |
| Anbu (PM2) | pm2.blr@draas.com |
| Sinchana Gowda | sgowda@draas.com |
| Sarthak Sharma | admin3.blr@draas.com |
| Roshni Ranka | rnr@draas.com |
| Vinod (Rahul) | vkdas@draas.com |
| Gowri Singh | gowrisingh1341@gmail.com (per contacts; confirm if a @draas.com variant exists) |

## Rule
Resolve everything you can from the map + gmail probe. Then **ask NDR to spell the
garbled voice names** before creating the draft — recipients are his to send; a clean
draft is the deliverable. Never silently drop or invent recipients. List what you
resolved, ask for the rest. (2026-09-30 outcome: Anbu/Bharat/Prakash/Sinchana/Sarthak/
Roshni resolved; asked for Gowri's exact email, Tantesh, Siddhira, and the garbled
"Ten, Tali, Aangara, Riveona, Sondajawai".)