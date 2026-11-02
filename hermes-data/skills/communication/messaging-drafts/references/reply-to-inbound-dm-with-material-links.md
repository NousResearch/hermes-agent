# Reverting to an inbound NDR chase DM — with the material links inline

Fires when a reportee (Bharat, Prakash, …) says something like *"we got a DM from
Nishant on Telegram at 16 Oct 8:59, I want to revert — show me the draft first"*.

## 1. Recover NDR's ACTUAL questions — never draft from the paraphrase

The reportee names only a timestamp and answers question 1 of 3. The inbound DM's
full question set is recoverable:

- The tracker cron (`personal-task-tracker`) logs every NDR instruction with spool
  line numbers and records exactly which DMs Hermes sent on his behalf, e.g.
  *"Bharat chat 8717455402 msg 73293, Prakash chat 8502281203 msg 73294"*.
- `session_search(query="expo", sort="newest")` → the cron session → scroll around
  the match. The cron's own row-note text quotes NDR's questions verbatim.
- Verify the timestamp: NDR's 16-Oct **8:59 IST** DM was the 03:24 UTC spool line.
  IST = UTC + 5:30 — do the conversion before claiming a match.

## 2. Find the prepared collateral instead of asking for it

When the revert says *"whatever documents / standees we prepared, all the links are
ready"*, that means a Drive folder of print collateral. Locate by name **and**
recency:

- `gws_drive_list(service_name="google-draas", query="name contains 'Standee'")`,
  also `'Stall'`, `'Expo'`, `'Dasara'`.
- `query="modifiedTime > '<event week>T00:00:00' and mimeType = 'application/vnd.google-apps.folder'"`
  surfaces the folder created for that event in one call.
- Resolve `service_name` with `gws_resolve_account` first (Bharat =
  `google-draas` / sales1.blr@draas.com). Never pass a raw email.
- Link formats: folder `https://drive.google.com/drive/folders/<id>`,
  file `https://drive.google.com/file/d/<id>/view`.

Log what exists so the next event is a lookup, not a dig. Dasara 2026 set —
folder `DASARA_STALL` id `10Y7OaSlR8HMiMEAre3OB7-NaawrxGfG2` (all owned by
gsingh@draas.com, DRAAS marketing):

| File | id |
|---|---|
| Standees.pdf | 1bNwLRhOd5JSFOS5uAAbWDLgfRRYxq-s0 |
| Stall_panels.pdf | 14R7kJFjR_mKcQSW0IKDlZ4h14QikxQZ- |
| Stall_side_panels.pdf | 1JuTD2FwvlBEz5afrJs7M7uVzkz9rSs4a |
| Stall_DRA_branding.pdf | 1IrITYMvZvLADKhSdvilVU34oUoabpP38 |
| Table_branding.pdf | 1kDJ6Tg1QU03eIMHx5LxZpbjFKe9kBJHu |
| RANKA_UDAYA_A5_leaflet.pdf | 18L1Jv4VncFdYLeK0K9z4z4pvYVABlK4K |
| RANKA_AMBER_A5_leaflet.pdf | 1_mLubBLny2Gl1bMxrP3xXU_tXTlji6Mi |
| RANKA_OASIS_A5_leaflet.pdf | 1EUzPo4DahbjWohlRnxGajQwzpHoJm0Zm |

Related event artefacts: `Durga Puja 2026 — Stall & Sponsorship (Sukriti Cultural
Association, opp. TISB)` (folder `16Zn1OQ1Pb9b_32E8-L5ZMHunWFuXctsb`, holds the SCA
sponsorship deck); SCA sponsorship proforma invoice PINV/2026-27/019 of 13-Oct-26.

## 3. Deliver the WHOLE message, links inline

Bharat's ask is literally *"share me the entire message with links and all"*. Return
ONE copy-pasteable block:

1. Salutation + sign-off as he signs.
2. Numbered answers **mirroring NDR's own numbering** — one line per question, in
   his order.
3. A `Material links:` list — folder link first, then one labelled link per file.
4. Keep it Telegram-legible; no pipe tables.

A prose draft with "I can add the links if you want" is a miss — it costs a round
trip. Then list separately, before he sends, the facts worth confirming rather than
inventing: the **event's own name** (NDR names no event — ask, don't guess) and the
**dates** (voice dictation garbles digits).

## Pitfall — a viewer cannot verify the recipient's access

`drive_permissions_list` on a file owned by someone else returns
`403 insufficientFilePermissions` even when the session user can see and read the
file. So never claim *"NDR can open these"*. State the owner, and **offer** to add
the recipient as a viewer from the session user's account, or hand the check back.

## Delivery channel

Reply on the channel the inbound came in on — a Telegram DM gets a Telegram text.
Offer a tap-to-open link only if asked; do not auto-generate a WhatsApp deep link
for a Telegram inbound.
