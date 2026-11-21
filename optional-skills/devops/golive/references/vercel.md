# Vercel (hosting): notes for Hermes

Load this when the plan uses `hosting=vercel`. Read it before explaining `doctor`, `plan`, or `verify`
output for Vercel. (Vercel DNS is not
automated: `dns=vercel-dns` is a guided provider, see `guided.md`.)

## 1. Logging in (least friction first)

**Install the Vercel CLI either way** (`npm i -g vercel`; running that install through the Hermes
`terminal` tool is fine). golive always deploys through the `vercel` binary; a token doesn't replace the CLI.

1. **`vercel login` (preferred).** The human runs it in a **separate terminal window** (the Terminal
   app or their IDE's terminal). It is a browser device-code flow: it shows a code, they approve in the
   browser, and the CLI stores the login where golive can use it. Nothing gets copied. Whether it works
   through the Hermes `terminal` tool (no interactive TTY, output shown only at the end) is unverified, so
   don't run it there.
   - CLI old: `npm i -g vercel@latest` first. The old `--github` / `--gitlab` / email logins were
     removed in 2026; use plain `vercel login`.
   - `doctor`'s `via` shows the account and team (`vercel CLI (logged in as alice, team acme)`). Check
     it's the team they expect. `plan.targets` shows the effective team/account of the project;
     an explicit `VERCEL_ORG_ID` can differ from the CLI's default team. Before approval, show
     the frontend destination's display name, project, and new/existing status in chat. The
     approved scope is bound to the plan; changing it requires a new plan and explicit approval.
2. **Token (alternative to `vercel login`, not to the CLI).** The human creates a **team-scoped token with an expiry** at
   https://vercel.com/account/tokens. On macOS, run `credentials --prompt VERCEL_TOKEN --json` so
   they enter it in the private native dialog. Use their own editor only if unavailable, unsupported,
   or preferred; follow [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts)
   for fallback, replacement and cancellation. Never put the value in chat or arguments. A token exported in their own terminal
   doesn't reach Hermes's shell. `doctor` then shows `via: VERCEL_TOKEN (user …)`, but reports
   Vercel as not ready ("install it: npm i -g vercel") while the CLI is missing. golive hands the token
   to `vercel deploy` through the child's `VERCEL_TOKEN` env var, never `--token` (argv shows up in
   process lists). Never suggest `--token` to the human either.

## 2. What golive does vs. what stays with the human

golive automates (after plan approval):
- **Picks the project** (`project:hosting`): the linked one, else `projects.hosting` from `golive.yaml`,
  else a same-named one, else a Create step that lists existing projects the human could use instead
  (`init --project hosting=<name>`).
- Writes env vars **by name** for Production and Preview separately: secrets as Vercel's **Sensitive**
  type (write-only), public values as Encrypted. Local development is left to `.env.local`. It checks
  every target before writing any, so a refusal (hidden production env, an integration-owned var, or
  one var shared with other environments) never leaves a var half-written.
- Deploys production with the CLI when needed (see `plan-and-verify.md`: production env changed, a
  pending or failed deploy, or never deployed by golive).
- Finds the **public production URL**: the verified custom domain, else the alias Vercel actually
  assigned (exact `<project>.vercel.app`, then a custom domain, then the shortest non-branch
  `*.vercel.app` alias). Before the first successful production deploy there is no URL, and golive
  never guesses `https://<project>.vercel.app` (that name may belong to someone else).
- **Custom domain:** `domain:attach`, then the **project-specific** DNS records Vercel wants go to an
  automated DNS provider (`--confirm-dns`) or into a handoff, then `domain:verify` asks Vercel to verify
  ownership (`POST /v9/projects/{id}/domains/{domain}/verify`). That call is a write, because success
  moves the domain away from another Vercel account/team, so it runs only as an approved step and
  never re-verifies a verified domain. A missing or mismatched TXT record gives `pending` (DNS
  propagating: re-run `plan` / `apply` later). While the domain isn't live, each plan re-sends
  `domain:attach` (idempotent), so a hosting-project switch or a domain removed in the dashboard is
  re-attached.
- Verifies: `env-parity` (names per environment, never values), `bundle-secrets`, `webhook-unsigned`,
  `domain-live` (passes only when Vercel reports the domain `ok`; `pending` warns).

Stays with the human (and why):
- **Deployment Protection settings.** golive doesn't switch protection off: it's a security downgrade.
  Its probes go to the public production URL, which Standard Protection leaves public. If
  `bundle-secrets` warns that production redirects to `vercel.com/sso-api`, "All Deployments"
  protection is on; the human decides whether to make production public.
- **Env vars owned by a Vercel integration** (Vercel's integrations store; e.g. Supabase or Stripe installed through Vercel).
  golive adopts them and never overwrites them. If the code expects a different name, change the code.
- **Values golive can't source** (`unmappedEnv`, e.g. `OPENAI_API_KEY`). The human types them straight
  into the Vercel dashboard (Project → Settings → Environment Variables). Never through chat.
- **Team roles.** If the login's role can't see some production vars, golive can't verify or write
  production env (§4).
- Buying a domain, upgrading plans, adding payment methods.

## 3. Explain these in plain words

- **"Your deployment link asks for a Vercel login."** New projects start with Standard Protection. It
  protects every generated URL, *including the long unique URL a production deploy prints*
  (`<project>-<hash>-<team>.vercel.app`). The public addresses are the production alias and custom
  domains. Share those, and point webhooks at those.
- **Webhooks to preview deployments fail.** Stripe can't log in to Vercel. golive registers webhooks for
  production only.
- **Env changes need a redeploy.** Existing deployments keep the old values. golive redeploys production
  after it writes production env; preview-only changes apply on the next preview deploy. After a
  dashboard edit, the human redeploys.
- **Secret vars can't be read back**, even in the dashboard. That's intended. `verify` checks names
  only.
- **Browser-visible names** (`NEXT_PUBLIC_`, `VITE_`, names inlined by the framework config, …) ship to
  every visitor. golive won't write a server secret into them; `detect` flags it as critical.
- **DNS values are per project.** Blog posts say `76.76.21.21` or `cname.vercel-dns.com`; don't use
  those. Use exactly the records `plan` prints. On a project never deployed by golive, the first
  production deploy runs before the domain is attached.
- **Auth redirect URLs for previews:** preview hostnames are unpredictable. golive adds preview
  wildcards to the (production) auth allowlist only with `auth.previewRedirects: true`.
- The Vercel CLI sometimes suggests `--value "<value>"` in its hints. **Don't follow that for
  secrets**: it puts the value on the command line.

## 4. Troubleshooting

| Symptom | What to do |
|---|---|
| `doctor`: not logged in | Human runs `vercel login` in a real terminal window (not `!`), then re-run `doctor`. |
| `doctor`: "install it: npm i -g vercel" (even with `VERCEL_TOKEN`) | The CLI is missing; deploys need it. Install, re-run `doctor`. Plans stay blocked on `login:vercel` until then. |
| Deploy fails with code `cli_missing` | Same: install the CLI and re-run `apply`. |
| Login fails or asks for a removed method | `npm i -g vercel@latest`, then plain `vercel login`. |
| "VERCEL_TOKEN is set but Vercel rejected it" | Remove that line from the credentials file (or Hermes's env) and use `vercel login`, or replace it with a new token. |
| `vercel api` not found / unknown command | CLI too old (`vercel api` is beta). Update the CLI (`npm i -g vercel@latest`), or add a `VERCEL_TOKEN` to the credentials file (API calls then skip `vercel api`; deploys still use the CLI). |
| A var is refused: "split per environment in the Vercel dashboard" | One Vercel var row covers environments golive wasn't asked to write. The human splits it into one per environment (Project → Settings → Environment Variables), then `plan` again. For the webhook secret this shows as a blocking `stripe:webhook-env` handoff and no endpoint is created. |
| `cannot verify production env: N … hidden from this Vercel login's role` (code `hidden_env`) | The role can't see some production vars, so golive won't report them missing and refuses to write production env. Have a team Owner run golive, or grant a role that can read production env. |
| `bundle-secrets` warns the page redirects to `vercel.com/sso-api` or a login | Production is behind protection or an auth wall; its scripts weren't scanned. Make production public, re-run `verify`. |
| A check says `cannot confirm … belongs to your project yet` | No confirmed production URL yet (never deployed, or domain not verified at Vercel). Apply the deploy / domain steps, then re-run. |
| App says a var is undefined after wiring | Redeploy. Check the var targets the right environment. Client-side vars need the framework prefix. |
| `production_secret_must_be_separate` | Team policy: one Secret can't span production and preview. golive already writes them separately. |
| Env var skipped as "managed by a Vercel … integration" (the CLI names Vercel's integrations store) | Adopt it; rename in code if needed. |
| Adding the domain returns 400 about the latest production deployment | Fix the failing deploy first, then re-run `apply`. |
| Domain already assigned to another project / account | The human removes it there (Vercel dashboard → Domains), or adds the TXT record from `plan` so `domain:verify` can move it. |
| "Vercel is already verifying <domain> for another project" | The human removes the domain from that project in the Vercel dashboard, then re-run `apply`. |
| `domain:verify` pending | TXT record not visible yet. Wait for DNS, run `plan` / `apply` again. |
| Domain `misconfigured` | DNS record not created yet, still propagating, or proxied (Cloudflare orange cloud). See `cloudflare-dns.md`. |
| Deploy output isn't a URL | Normal in agent mode: the CLI prints JSON. golive parses both shapes. |

## Unverified

- Whether `vercel login` (device-code flow) completes under the Hermes `terminal` tool, which has no
  interactive TTY. That path is unverified, so these notes always say "separate terminal".
- Whether the production alias stays public under Standard Protection in every case (docs imply yes).
- Exactly how env "upsert" behaves when an existing var has different targets or type. golive lists
  first and plans explicitly instead of relying on it.
