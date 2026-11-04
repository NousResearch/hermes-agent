# Verified DRA team sending addresses

Derived from **Gmail headers** on ndr@draas.com, not from the contacts sheet (sheet holds the
mailbox, headers hold the address the person actually mails *from*). Re-verify with
`gws_gmail_search(query="from:<candidate>")` + a metadata read whenever a new name appears.

| Person | Role | Send to |
|---|---|---|
| Anbarasan M ("Anbu") | PM-2, projects | `anbarasan@draas.com` — sheet says `pm2.blr@draas.com`; older mail came from pm2, current mail comes from anbarasan@. Use anbarasan@. |
| Sinchana Gowda ("Sanchana") | Architect coordinator, custodian of design/BOQ trackers | `sgowda@draas.com` — **not** `sinchana@kelsa.io`, which is "Sinchana Kelsa", a different person. |
| Gowri Singh | Marketing content + creative approvals | `gsingh@draas.com` |
| Bharat Hawaldar | Sales head | `sales1.blr@draas.com` — display name `BHARAT H`. He *sends as* the shared sales mailbox, so that is the correct personal To. `sales1.blr@drahomes.in` is the other-entity variant. |
| Roshni Ranka | Director | `rnr@draas.com` (company mail). `rnr@drahomes.in` / `rmurjani@gmail.com` only for personal/DRA Homes context. |
| Nishant Ranka (NDR) | — | `ndr@draas.com` for company mail; `ndr@drahomes.in` / `nishantranka@gmail.com` only when asked. |

## Method that produced this

```python
import re, collections
r = g.users().messages().list(userId="me", q="<name>", maxResults=40).execute()
# for each hit, read format="metadata", metadataHeaders=["From","To","Cc"]
# counter over every address found in those headers, then read the display names on the top hits
```

The counter finds the address; the display name on a `From` header proves the person actually sends
from it. A single appearance in a `To:` is weak evidence — it may be a shared or obsolete alias.

Beware the noisy contact-sheet rows: Bharat Hawaldar's sheet row carries a mash-up cell of five
unrelated `@draas.com` addresses. Never paste from the sheet.
