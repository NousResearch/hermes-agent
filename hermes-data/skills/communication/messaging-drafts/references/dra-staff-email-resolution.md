# Resolving DRA staff email addresses (verified 2026-10-13)

When drafting an email to **DRA/teammate staff** (Sinchana, Gauri, Prakash,
Kantesh, Anbarasan, Roshini, Bhuvanesh, etc.):

## DRA staff are NOT in the contacts sheet — use the mailbox
`contact_resolver` is tuned for external contacts (vendors, landowners, brokers).
DRA staff either don't resolve or return wrong external lookalikes (e.g. "Gauri"
→ "Gauri Broker Dubai"; "Prakash" → 5 unrelated externals). **Do not rely on it
for staff.**

Instead, `gws_gmail_search` on the sender's own mailbox (service_name from
`gws_resolve_account`) with the first name as query → the staff's canonical
address appears in `from:`/`to:` fields of real threads within 1-2 calls.

Known staff addresses (canonical, from mail trails):
- Sinchana Gowda (design/collaterals) — `sgowda@draas.com`
- Gowri Singh (Gauri mam, marketing) — `gsingh@draas.com`
- Prakash Singh (sales) — `psingh@draas.com`
- Nishant Ranka (NDR) — `ndr@draas.com` (also ndr@drahomes.in / ndr@ahfl.in)
- Bhuvanesh S Krishnan (architect, external) — `bl@fnw.homes`
- Anbarasan / Anbu (PM) — `pm2.blr@draas.com`
- Kantesh B G — `kanteshbg@draas.com`
- Roshini Ranka — `rnr@draas.com`

## contact_resolver errors with "no gws_service configured" (Bharat sessions)
For session users whose profile lacks `gws_service` (e.g. sales1.blr / Bharat's
telegram id 8717455402), `contact_resolver` fails with
`Sheets connection failed: User '<id>' has no gws_service configured`.
**Fix:** pass `account_email="sales1.blr@draas.com"` (the user's real account)
explicitly — resolves and works normally. Same shape applies to any user:
resolve their account first via `gws_resolve_account`, then pass it.

## Draft flow that landed (2026-10-13, Bharat → Sinchana + CC)
1. `gws_resolve_account()` → author's token/key (google-draas = sales1.blr).
2. `gws_gmail_search(query="<staff first name>", service_name=...)` → exact
   addresses from thread metadata. One search per staff name is enough.
3. `gws_gmail_draft_create(to=..., cc=..., subject=..., body=..., service_name=...)`
   — draft only, user sends. Never `.send()`.
4. State the resolved recipients back to the user in chat + flag any assumed
   project name in the subject so they can correct before sending.