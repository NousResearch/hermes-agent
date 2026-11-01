# Resend (email): notes for Hermes

Load this when the plan uses `email=resend`.

Status: the disposable live run passed end to end — domain created through the CLI transport, records
written to an automated DNS provider, domain verified, sending-scoped keys issued into the app's env,
and a send using that environment key was delivered (spam folder; fresh subdomain, no DMARC). Auth
SMTP is wired from a sending key golive issues (`auth:smtp`, with `auth.smtp: resend`) and that write
is live-validated (2026-09-24: the settings and the raised auth email rate limit read back; the
password itself is write-only); bounce handling remains open. A sending domain's `verified` flag can
be stale — a zone cleaned up, moved or restored leaves it reading `verified` with its records gone
([#52](https://github.com/mikehasa/golive-skill/issues/52)) — so `email-verified` resolves the records
Resend itself lists before passing, and the plan keeps writing them when they are missing.

## 1. Logging in (least friction first)

1. **`resend login` (preferred).** Install (`npm i -g resend-cli` or `brew install resend/cli/resend`),
   then the human runs `resend login` in a **separate terminal window** (the Terminal app or their
   IDE's terminal; not the Hermes `terminal` tool, which has no interactive TTY for its picker) and
   picks **"Login with Resend (opens browser)"**. They click Authorize; nothing is copied. golive uses
   this login first, even if a `RESEND_API_KEY` is also around.
   - Never suggest `resend login --key …`: it puts the key on the command line.
2. **Full access API key (alternative).** Resend dashboard → API Keys → create a key with **Full
   access** (golive creates domains and keys, which sending-only keys can't). On macOS run
   `credentials --prompt RESEND_API_KEY --json` for private native entry. Their own editor is the
   fallback if unavailable, unsupported, or preferred; follow
   [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
   replacement and cancellation. Never put the value in chat or arguments; a key exported in their
   own terminal doesn't reach Hermes's shell.
   - A **sending-only** key there (common from local app development) fails with
     `restricted_api_key`: replace it with a Full access key, or use `resend login`.

## 2. What golive does vs. what stays with the human

golive automates (after plan approval; DNS writes need `--confirm-dns`):
- `email:domain`: finds and **adopts** the sending domain in Resend, or adds it (in `email.region`
  from `golive.yaml` if set), with open/click tracking off. Its changes list the DNS records Resend
  asks for (they differ by domain age and region; never hard-code them).
- `email:dns`: writes exactly those records through an automated DNS provider (e.g.
  `cloudflare-dns`), or they become an `email:dns` handoff for a guided DNS host. Record names
  relative to the apex are resolved correctly, also under short-SLD ccTLDs (e.g. `send.notify.app`
  for `notify.app.hey.io` becomes `send.notify.app.hey.io`). golive does **not** add a DMARC record;
  `email-dns` warns when none exists and suggests one for the human to add.
- `email:verify`: asks Resend to verify (safe to retry); `pending` is not a failure.
- `email:key:<target>`: **mints a new sending-only key scoped to that domain, per environment**, and
  writes it straight into the host env name the code reads (usually `RESEND_API_KEY`). Resend shows a
  key once, so "adopt" means "issue a new one". Older golive keys are **left active**: the change log
  names them, and the human revokes them in Resend once nothing uses them.
- Verifies: `email-dns` (the exact records Resend lists for the domain, plus DMARC, in public DNS; a
  missing DKIM record fails) and `email-verified` (Resend marks the domain verified **and** the records
  it lists for that domain resolve in public DNS — a stale `verified` whose records are gone fails
  with the records named, #52).

Not automated yet: the from-address env var (the human sets it if the code reads one) and a test send.
Resend as Supabase Auth's SMTP server is written by the `auth:smtp` step when `auth.smtp: resend` is
set (host `smtp.resend.com`, port 465, user `resend`): it takes the SMTP password from the key the
email journey issued in the same run, or issues `golive-<app>-smtp` for that purpose alone and records
it like every other key, and it raises Supabase's own auth email rate limit to 30 per hour (or
`auth.emailRateLimitPerHour`) in the same write, since the provider keeps that limit with custom SMTP.
See `supabase.md` for the step's read-back limit (the provider never returns the password).

Stays with the human (and why):
- **A domain already registered by another Resend team.** Claiming it needs a TXT record and gives
  the domain new DKIM keys. Ask before starting a claim.
- **DNS at a provider golive doesn't automate.** The human adds the exact records from `plan`/`handoff`.
- Revoking old keys, plan upgrades (free plan: 100 emails/day, 3,000/month, 3 domains).

## 3. Explain these in plain words

- **Proxy off.** On Cloudflare, Resend's CNAME records must be "DNS only" (grey cloud). Proxied ones
  never verify.
- **One SPF record per name.** Resend usually puts SPF on a subdomain (`send.<domain>`), so the main
  domain's SPF often isn't touched. If a merge is needed, it's merged into the existing record, never
  added as a second one.
- **`send.` already in use?** A CNAME can't share a name with other records. golive stops with a
  conflict instead of overwriting. Resend supports a different custom return path (e.g. `bounce`),
  chosen when the domain is added; raise it with the human.
- **Recreated domain or changed region:** the DKIM key and return-path MX change. With Cloudflare,
  golive replaces the stale records it created; records someone else created stop it with a conflict.
- **Verification can take minutes up to 72 hours.** After 72 hours without records Resend marks the
  domain `failed`. `partially_verified` can send but has no fallback; finish the missing record.
- **Tracking off for auth emails.** Open/click tracking rewrites links and breaks Supabase magic
  links. Domains golive creates have it off.
- **`onboarding@resend.dev`** only sends to the account owner's own address. Production needs the
  verified domain.
- **Never expose the key to the browser.** `NEXT_PUBLIC_RESEND_API_KEY` (or any public prefix) is a
  leak; `detect` flags it and golive won't write the key there.

## 4. Troubleshooting

| Symptom | What to do |
|---|---|
| `doctor`: not authenticated | Human runs `resend login` in a real terminal window (not `!`), browser option. Never `--key`. |
| `401 restricted_api_key` on setup | A sending-only key is in use. Use `resend login` or a Full access key in the credentials file (§1). |
| `403 validation_error` "has been registered already" | Another Resend team owns the domain. Claim flow, with the human's OK. |
| Domain stuck `pending` / `failed` | Records missing, proxied, or entered with the domain twice (`send.example.com.example.com`). Compare with `verify`'s `email-dns` evidence. |
| Domain reads `verified` but mail fails SPF/DKIM | The flag is stale: the records Resend lists are not in DNS (a cleaned-up zone, a domain moved between teams or a restored backup). `email-verified` fails on exactly this, naming the records. Re-run `plan` + `apply --confirm-dns` to write them, or add them by hand at your DNS host; the verified flag itself stays as it is. |
| "multiple-regions" verification error | MX records on the `send` host point at different regions. Keep only the one Resend listed. |
| Cloudflare conflict at `send.<domain>` or a `_domainkey` name | Another record holds that name. Delete it if unused, or recreate the Resend domain with another return path. |
| `403` sending from `onboarding@resend.dev` | Only the owner's address works. Send from the verified domain. |
| `429 daily_quota_exceeded` / `monthly_quota_exceeded` | Plan quota hit. Wait for the reset or the human upgrades. |
| `429 rate_limit_exceeded` | 10 requests/second per team; retry after a moment. |
| Supabase auth emails not arriving | With `auth.smtp: resend`, check the `auth:smtp` step's changes and `auth-policy`'s `custom SMTP via Resend` evidence, that tracking is off, and the auth email rate limit (the step raises it to 30 per hour, or `auth.emailRateLimitPerHour`). Without that opt-in the project still uses Supabase's built-in mailer, which is rate-limited. |

## Unverified

- Exact names of the newer CNAME-style SPF records (seen as `send` / `rsend`); golive reads them from
  Resend's API rather than assuming.
- Supabase's email rate limit right after enabling custom SMTP (docs say 30/hour, Resend says 25).
