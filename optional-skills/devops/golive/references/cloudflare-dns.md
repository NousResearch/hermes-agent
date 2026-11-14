# Cloudflare DNS: notes for Hermes

Load this when the plan uses `dns=cloudflare`.

## 1. Logging in: a zone-scoped API token

Cloudflare is **token-only**. `wrangler login` can't be used: its browser login has no DNS-write
scope, so every DNS write would fail with 403. Don't send the human there.

Walk the human through creating the token (it goes into the credentials file, never this chat):
1. Cloudflare dashboard → **My Profile → API Tokens** → **Create Token**.
2. Start from the **"Edit zone DNS"** template (Zone → DNS → Edit).
3. **Add** a permission: **Zone → Zone → Read** (lets golive find the zone by name).
4. **Zone Resources → Include → Specific zone → their domain.** This limits the token to that one
   domain, so a leak can't touch anything else.
5. Optional but good: set an expiry. Create the token; Cloudflare shows it **once**.
6. On macOS run `credentials --prompt CLOUDFLARE_API_TOKEN --json`; they enter the value in the
   private native dialog. Their own editor is the fallback if unavailable, unsupported, or preferred;
   follow [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
   replacement and cancellation. Never put the value in chat or arguments. A token exported in their
   own terminal doesn't reach Hermes's shell.

Notes:
- `CF_API_TOKEN` is accepted as an older alias. The **Global API Key** (`CLOUDFLARE_API_KEY`) is
  refused: it can do anything on the account. Ask for a scoped token instead.
- Account-owned tokens only verify under their account: also set `CLOUDFLARE_ACCOUNT_ID` (not a
  secret; Hermes's environment or the credentials file).

## 2. What golive does vs. what stays with the human

golive automates (DNS writes always need `--confirm-dns`):
- Finds the domain's **active** zone in the human's Cloudflare account through the API. If that lookup
  fails (token permissions, rate limit, network), `plan` warns and plans no DNS step.
- For each record it needs, looks at the existing records at that name, then creates it, adopts an
  identical record as-is, or updates one it created earlier. Records golive creates carry a comment
  starting `golive:`, so later runs know they're its own. Changes are applied **one record at a time**
  (not one batch): if a step stops midway, records already written stay, and re-running `apply`
  continues.
- Sets **proxy off** on records it creates or owns.
- **SPF:** merges the sender's mechanisms into the one existing `v=spf1` record. Mechanisms are
  compared without qualifiers, so nothing is duplicated. If the existing SPF already has the sender's
  mechanism with a `~`, `-` or `?` qualifier, golive stops and asks the human to change that term to a
  plain `include:` (SPF could never pass otherwise). It also stops past 10 DNS lookups.
- **One-per-name records:** a DKIM key TXT at `<selector>._domainkey.<domain>` and a return-path MX
  (`feedback-smtp.<region>.amazonses.com`). A stale one golive owns is updated in place (e.g. after
  the Resend domain was recreated or its region changed). One someone else created stops golive with
  a conflict error and nothing is changed. Other TXT/MX records sit alongside existing values.
- **CNAME exclusivity:** golive never puts a CNAME on a name that also holds TXT/MX/CAA records, or
  those records next to a CNAME, except at the zone apex. It stops with an error listing the
  records; for mail records it suggests deleting them or recreating the Resend domain with a different
  custom return path (e.g. `bounce`).
- Doesn't write DMARC (the `email-dns` check suggests one).
- Retries: record updates are retry-safe; a record create is not re-sent after a timeout or 5xx
  (re-running `apply` adopts it if it landed).
- Verifies: `domain:dns:records` (the zone holds the records), then `domain-live`, `email-dns`,
  `email-verified` from outside.

Stays with the human (and why):
- **Nameservers.** If the zone is `pending`, the domain's registrar still points elsewhere. The human
  sets the two Cloudflare nameservers shown in the dashboard at their registrar.
- **Records golive didn't create that conflict with what's needed** (e.g. an old `www` pointing at a
  previous host, a proxied record with the right value, a CNAME clash). golive never overwrites or
  deletes them: the error names them, and the human edits or deletes them in the dashboard (or sets
  their comment to start with `golive:` to let golive manage them), then re-runs.
- Buying the domain; domains that live in someone else's Cloudflare account.
- SPF over 10 DNS lookups after merging: a human decision about which senders to keep.

## 3. Explain these in plain words

- **Orange cloud off.** Cloudflare's proxy in front of Vercel breaks certificates and verification,
  and proxied email CNAMEs never verify. Records the human made earlier may be orange; golive leaves
  those alone and warns. The human switches them to "DNS only" (grey).
- **Only one SPF record.** Two `v=spf1` records at the same name make SPF fail completely. New senders
  get merged into the existing record.
- **Propagation takes time.** Cloudflare updates within about a minute, but other resolvers cache. A
  name looked up *before* it existed can stay "not found" for up to **30 minutes** (negative caching).
  Don't test a name by hand before golive creates it, and let `verify` retry.
- **Root domain for Vercel:** use the A record from `plan` (Vercel's verifier expects it).
- **A name can't have both a CNAME and other records** (except at the apex, where Cloudflare flattens).

## 4. Troubleshooting

| Symptom | What to do |
|---|---|
| `doctor`: token missing or not active | Create/recreate the token (§1) and put it in the credentials file. |
| Zone not found | Token isn't scoped to this domain, lacks Zone Read, the zone is `pending`, or it's in another account. Fix the token. |
| Zone status `pending` | Nameservers not changed at the registrar yet. Human updates them; can take hours. |
| 403 on writes | Token lacks DNS Edit (e.g. a wrangler token). Use the "Edit zone DNS" token. |
| "DNS conflict at <name>" | A record golive didn't create is in the way (A/CNAME clash, CNAME next to TXT/MX, foreign DKIM/return-path). Follow the error: edit/delete it in the dashboard, then re-run. |
| `domain:dns`: "the DNS records … requires … changed since the plan was approved" | The host now wants different records than the human approved; nothing was written. Run `plan` again and get re-approval with `--confirm-dns`. |
| "SPF … already has `~include:…`" | Change that term to plain `include:…` in the dashboard, then re-run. |
| Several SPF records at one name | Merge them into one in the dashboard, then re-run. |
| Error 81058 "An identical record already exists" | Already there; golive adopts it. Harmless. |
| Warning "recovering a stale zone id for …" | The zone id in `.golive/state.json` was rejected (zone deleted/re-created, or the token re-scoped); golive re-found the zone and continued, naming both zones. If the domain doesn't go live, confirm the zone golive used is the one that serves the domain, then re-run `plan`. |
| Error "The cached Cloudflare zone … was rejected … no active zone … anymore" | The zone was deleted or re-created, the nameservers changed, or the token was re-scoped/revoked. Check the Cloudflare dashboard (zone exists and is active, token still scoped to it), then re-run `plan`. |
| 429 | Rate limit (1,200 requests / 5 minutes); wait and re-run. |
| Vercel says misconfigured; lookups show Cloudflare IPs (104.16.x, 172.64.x) | Record is proxied. Turn the orange cloud off. |
| Email domain won't verify (Cloudflare "Code 1004") | Proxied email CNAME. Set it to DNS only. |
| Still "not found" right after creating | Negative caching; wait up to 30 minutes and re-run `verify`. |

## Unverified

- Whether a token with only DNS Edit (no Zone Read) can still find its zone by name. The safe setup
  above includes Zone Read.
- "Dashboard proxies new records by default" comes from community reports, not current official docs.
