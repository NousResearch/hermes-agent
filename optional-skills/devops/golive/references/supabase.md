# Supabase (database + auth): notes for Hermes

Load this when the plan uses `db=supabase` or `auth=supabase`.

## 1. Logging in

**Preferred: one `supabase login` in the human's own terminal.** It needs a TTY, so use the Terminal
app or the IDE terminal, not the Hermes `terminal` tool (no interactive TTY). For a new account, the human
signs up and creates/selects a Free organization first; SDK imports do not prove a project exists.
Then golive checks the account and presents the exact project destination for approval.

On supported credential stores, golive reuses that browser login for the **complete flow**: creation,
API keys, pooled database URLs, Auth redirects, RLS queries and advisors. It captures the credential
internally as `Secret` and uses HTTPS; it never prints it or puts it on argv. A second manually
created Management API token is not required. Do not inspect or print the CLI token store yourself.

Supported credential stores (Supabase CLI v2.117.0 or later within v2):
- **macOS:** the official Keychain service/profile, with the documented private-file fallback when
  its items are absent. macOS may raise a **SecurityAgent dialog** asking whether the read-only
  `security` helper (golive's reader) may read the item "Supabase CLI" — the Supabase CLI itself is on
  the item's allow list, which is why its own commands never ask. golive only reads that item and
  never changes the Keychain. Tell the human which dialog is theirs to answer and that in it:
  **"Allow"** answers this read (the dialog returns on the next run) and **"Always Allow"** records the
  permission permanently for that item, so it stops asking. golive waits 15 s, then one longer
  attended window (120 s) while it says the dialog is coming; if it still goes unanswered, the
  reusable credential read fails closed with that same instruction, and the reads the CLI performs
  itself (project discovery, API keys, the RLS query) keep working through the CLI while anything that
  needs the Management API is handed over. A refused/locked read is never retried and never falls
  back to another account.
- **Linux/WSL:** the CLI's private token file with keyring disabled, or on WSL. On Linux use
  `SUPABASE_NO_KEYRING=1 supabase login --profile supabase` in the human's terminal and preserve
  `SUPABASE_NO_KEYRING=1` when running golive. This is still browser login, not manual PAT creation.
- **Windows/native Linux keyrings:** safe reuse is not implemented. Use the explicit-token fallback
  below. Unknown CLI major versions require review; update an older CLI before retrying.

Only the production **`supabase` profile** is supported. `SUPABASE_PROFILE` overrides the CLI's
persisted profile file; staging, local, Snap and custom API profiles are rejected. Use
`supabase login --profile supabase` to select the intended production account. Stores are read-only;
golive never migrates them or changes permissions. Wrong owners, unsafe permissions or redirected
paths fail closed. A native macOS read-only test on 2026-09-23 reused the existing login with the
explicit-token path disabled: profile/projects/organizations reads passed and the CLI/API project
inventories agreed. The later auth live run (2026-09-23) created one project and wrote its auth
policy through the same reused login, with no explicit token, no `SUPABASE_*` variable in the shell
and no Keychain prompt; the hosting E2E before it used an explicit token, and first-login UX is still
untested. Do not describe either run as a fresh login.

**Alternative: `SUPABASE_ACCESS_TOKEN` for CI or unsupported stores.** The human creates it at
https://supabase.com/dashboard/account/tokens with the permissions below. On macOS, run
`credentials --prompt SUPABASE_ACCESS_TOKEN --json` for private native entry. Their own editor is
the fallback if unavailable, unsupported, or preferred; follow
[How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
replacement and cancellation. Never put the value in chat or command arguments; exporting it in an unrelated human
terminal does not reach Hermes's shell. Never run `supabase login --token …`.

An explicit token takes precedence over CLI login for every call. If it is invalid, golive fails;
it never silently switches accounts. The human may replace it or remove that explicit setting to
use the CLI login. A rejected stored login is refreshed with browser login, without generating a PAT.

If using the **explicit-token alternative for an existing throwaway project**, choose only that project and a short expiry. Stage 1 permissions, using
the [current Supabase UI labels](https://supabase.com/docs/guides/platform/personal-access-tokens):

| Task | Permission / access |
|---|---|
| Project details + health | Project Settings: Read |
| Reveal API keys | API Keys + API Key Secrets: Read |
| Inspect auth (site URL, redirects, policy) | Auth Config: Read |
| Set site URL / redirects / policy | Auth Config + Project Settings: Read-write |
| Read Data API schemas | Data API Config: Read |
| RLS SQL check | Database: Read |
| Security advisors | Advisors: Read |
| DB URLs if requested | Connection Pooling: Read |
| Approved password recovery for an golive-created project only | Database Config: Read-write |

For **a new project**, the human signs up and creates/selects a Free organization first. If using the token alternative, the docs list Organizations
Read, Organization Settings Read, Projects (account-wide) Read and Organization Projects Read-write for the
account/org flow. If the user's scoped-token UI does not expose these, the account-management token was under
the experimental-token dropdown in Stage 1; that is an observed rollout detail, not a universal UI guarantee.
Alternatively the human creates the throwaway project first, then supplies a token scoped to that project.

`doctor` / `plan` report the reused CLI login and the API-confirmed account identity. A missing,
unsupported or unsafe store gives a blocking `login:supabase` handoff with a specific remedy. When
only the limited legacy CLI fallback is usable, it covers project discovery, keys and the RLS query
(those reads shell out to the CLI, which reads its own store without a prompt and never exposes the
credential); creation, the pooled database URL, auth settings and advisors use the Management API,
and stay blocked until the login can be reused or the human supplies the alternative. An unanswered
macOS Keychain dialog is exactly that case: the CLI-covered reads continue, the rest fails closed
with the dialog instruction, and the run warns once that the reusable credential could not be read.

A `/profile` 403 does not mean the token is invalid: project-scoped tokens can still access their selected project.
golive verifies an explicitly selected visible existing project (exact ref or unambiguous configured name). With no
selected project, it checks organization discovery/details for one eligible Free destination instead. That confirms
organization read access only; creation write permission remains unverified until the operation. An empty/denied
organization list blocks this path. API operations check permissions independently and plan/apply revalidates targets.

## 2. What golive does vs. what stays with the human

golive automates (after plan approval):
- **Picks the project** (`project:db`): the linked one (state → `projects.db` in `golive.yaml` →
  `supabase link` file), else a same-named one it may adopt (names compare case-insensitively), else
  a **Create** step. It adopts a
  same-named project automatically only if it is in this app's org (the account's only org, else its
  single free org) and not paused or failed. A same-named project in another org, a paused
  (`INACTIVE`) one, or several orgs with no single free one means no auto-adoption: create refuses (no
  duplicate is made) and names the ref, org or status. The human chooses with
  `init --project db=<ref>`, or restores a paused project in the dashboard. golive never restores one;
  selecting a paused project fails with restore instructions. A `COMING_UP` / `RESTORING` project is
  waited on until `ACTIVE_HEALTHY`.
- **Creating costs money:** free orgs allow 2 active projects; on paid orgs each extra project is about
  $10/month of compute. Confirm with the human, and offer the existing projects the Create step
  lists. The region comes from `supabase.region` in `golive.yaml` (else Supabase's default group).
- For a new project, golive generates the database password and sends it only inside the create
  request. That request is **never re-sent** (90s timeout, no retry). If the response is lost, golive
  lists projects again and adopts the one new project with that name in that org, keeping the
  password it sent. If none appears, it fails and asks you to check
  `https://supabase.com/dashboard/org/<org>` before re-running (a re-run adopts it by name once it
  appears). Then it waits until the project is healthy (usually 1–3 minutes, up to 15). On timeout
  it asks you to re-run later: golive picks the project up (and resets its password if needed, below).
- A project golive creates is recorded as `supabase.createdByGolive=<ref>` in `.golive/state.json`
  (not a secret). The generated password lives only in memory. If a later run has lost it and no DB
  URL from that project was ever written, and the app actually requests a database URL, `plan` warns and the env write step sets a **new generated
  password** for it (Supabase Management API, using the reused login or explicit token) instead of a `db:password` handoff.
  Never for an adopted project, without a reusable credential, or once a DB URL was written for any target.
  Apps needing only Supabase client URL/API keys do not collect database URLs or reset the database password.
- Writes the project URL and API keys to the host under **the names the code uses** (publishable key
  for the browser, secret key server-only).
- When it knows the password (projects it created) writes:
  - `DATABASE_URL` (and `POSTGRES_URL`, `POSTGRES_PRISMA_URL`, …): the shared Supavisor pooler in
    **transaction mode, port 6543** (uses the same login). Prisma repos get `?pgbouncer=true` (a
    `schema.prisma`, `prisma/schema.prisma` or `prisma/schema/` folder, a `prisma.config.*` or custom
    `prisma.schema` path, a Prisma dependency, or `POSTGRES_PRISMA_URL` referenced). Otherwise the
    log warns: postgres.js and Drizzle need `prepare: false`; node-postgres needs nothing.
  - `DIRECT_URL` (and `DATABASE_URL_UNPOOLED`, `POSTGRES_URL_NON_POOLING`, `DIRECT_DATABASE_URL`): the
    shared pooler in **session mode**, `postgres.<ref>@<region>.pooler.supabase.com:5432`. It is
    IPv4, so `prisma migrate deploy` works from Vercel. Without a reusable credential (or when Supabase lists no
    shared pooler) it falls back to `db.<ref>.supabase.co:5432` with a warning: that host is
    IPv6-only unless the project has the IPv4 add-on, so it won't work from most serverless hosts.
- Sets auth **Site URL** and the **redirect allowlist** to the production origin with a surgical API
  update (not `supabase config push`). Preview-deployment wildcards are added only with
  `auth.previewRedirects: true` in `golive.yaml` (flagged as a risk: it widens the production
  allowlist). Otherwise `plan` warns that sign-in on preview URLs won't work.
- Sets the auth **policy** from `auth` in `golive.yaml` — `signup`, `requireEmailConfirm`,
  `passwordMinLength` — in the separate `auth:settings` step (so a policy change doesn't re-run the
  redirect work). It writes only the values that differ, then re-reads them: the step's changes show
  `before → after`, the `auth:settings:applied` result confirms them, and anything the API does not
  report back appears as `not confirmed:` instead of a silent success. Only the settings this API is
  known to return are ever read or written; `smtp_pass` is write-only (the API answers a hash), so an
  SMTP write can never be confirmed from the read-back. The live run confirmed that the Management API
  does echo signup, email confirmation, the password minimum length, the SMTP-configured flag and the
  email rate limit back.
- Writes the **custom SMTP** in its own `auth:smtp` step when `auth.smtp: resend` is set: `smtp_host`
  (`smtp.resend.com`), `smtp_port` (465), `smtp_user` (`resend`), `smtp_admin_email` and
  `smtp_sender_name` from the sender `email.from` already uses, and the write-only `smtp_pass` from a
  sending key golive issued — the one the email journey issued in this run, otherwise one it issues for
  SMTP alone (`golive-…-smtp`, recorded as `<provider>.keyId@smtp` so teardown can revoke it); never a
  paste and never a chat prompt. Configuring the mailer is only half of it: Supabase keeps its **own
  auth-email rate limit** with custom SMTP in place (the live project's `rate_limit_email_sent: 2`
  refused a recovery request with HTTP 429 in the same run that wired the SMTP), and one run of the
  auth journeys needs four accepted sends, so the step writes `rate_limit_email_sent` too:
  **30 per hour**, or `auth.emailRateLimitPerHour` from `golive.yaml` when the human wants another
  value. The change is named in the plan and the step's changes
  (`auth email rate limit: 2 → 30 per hour`). Unlike an SMTP field, a rate limit the API keeps at
  another value does not fail the step — it is the provider's own setting — so it warns instead:
  `auth:smtp:applied:rate-limit` (medium) names the value that holds, and one the API never reports
  back is named as unconfirmed like any other field. `smtp_port` is the one field this API documents
  and validates as a string: golive sends `"465"`, because a number is the live 400
  `smtp_port: Invalid input: expected string, received number` (the port stays a number in the plan,
  the changes and everything golive reads back — the read accepts either type). The step re-reads the
  non-secret fields (`auth:smtp:applied`) and the `auth-policy` check reports
  `custom SMTP via Resend` instead of the built-in-mailer warning, and the journeys that send real
  mail run after it. The password is never compared, because the API never returns it: the read-back
  confirms the settings, and a real auth email arriving is the only full proof. Live-validated on
  2026-09-24 on a disposable project: the approved apply wrote the mailer and the rate limit in one
  request and read both back (`SMTP host: (not set) → smtp.resend.com`, `SMTP port: (not set) → 465`,
  `SMTP user: (not set) → resend`, `sender address: (not set) → auth@mail.trytofu.xyz`, `auth email
  rate limit: 2 → 30 per hour`), issuing one sending key for SMTP alone and recording it as
  `resend.keyId@smtp` (revoked in that run's teardown), and `auth-policy` then read `custom SMTP via
  Resend` with `rate limit: 30 auth emails/hour`. What that evidence is not: the password (write-only —
  reported `not confirmed`) and delivery, which needs an inbox golive cannot read.
- **Runs the signup journey** when the human opted in with `auth.e2e: true` (see below): the
  `auth:test-user` step seeds one test account through the project's own `/auth/v1` signup endpoint,
  `auth:confirm-email` hands the inbox click over, and the `auth-signup`/`auth-session` checks prove
  the rest.
- **Rotates that account's password through recovery** when the human also opted in with
  `auth.recovery: true` (see below): the `auth:recovery` step asks for a real recovery email, mints its
  own recovery link through the Auth admin API, exchanges the token for a session and sets the new
  password with that session. `auth-recovery-email` hands the inbox click over and `auth-recovery`
  proves the outcome.
- **Seeds a second account and reads the app's own routes** when the human opted in with
  `auth.isolation: true` (see below): the `auth:isolation` step signs up a second account
  (`testEmail` plus `+gl-isolation`) and confirms it through the Auth admin API, and `auth-isolation`
  signs in as both accounts and checks that neither can read the other's identity or rows through
  `auth.identityPath` / `auth.isolationPath`.
- Verifies: `rls-probe` (tables not readable with the public key, plus security advisors, read-only),
  `auth-redirects` (production URLs, no `localhost`), `auth-policy` (signup/confirmation/password
  policy and the mailer, with the effective values as evidence), `auth-signup`/`auth-session` (the
  journey above, when opted in), `auth-recovery` (the recovery journey above, when opted in),
  `auth-isolation` (the two-account journey above, when opted in), `env-parity` (names on the host).

Stays with the human (and why):
- **The database password of an existing project.** Supabase only reveals it at creation, so a
  `db:password` handoff appears: the human copies the connection string from the Supabase dashboard
  straight into the host's dashboard (never into chat), or resets the password in the Supabase
  dashboard (breaks anything else using the old one). Or the app skips a DB URL if it only uses the
  Supabase client.
- **Fixing RLS findings.** You (Hermes) write the migration/policy change; the human approves it.
  golive never write-probes your application's data tables. With `auth.e2e: true` it does create auth
  **test users** (see below) — that opt-in is exactly what covers them; with `auth.recovery: true` it
  also sets a new password on that same recorded test account through the recovery path; with
  `auth.isolation: true` it creates a **second** test account, confirms it through the admin API, and
  the check stores one small marker row per test account through the app's own
  `auth.isolationPath` route.
- **The inbox click** for a confirmation or recovery email (see the journeys below): golive never
  reads an inbox, and with `auth.smtp: resend` that message comes from the app's own email provider
  instead of the built-in mailer.
- Restoring a paused project, plan upgrades, billing, creating OAuth apps (e.g. Google sign-in).

### The signup journey (`auth.e2e`)

Opt-in, three keys in `golive.yaml`:

```yaml
auth:
  e2e: true                      # accept that this journey writes real auth users
  testEmail: you+go-live@example.com   # the human's own inbox (plus-addressing is fine)
  protectedPath: /dashboard      # an app route that must require a session
```

What it proves, with the evidence to match:

- **Signup sends mail.** `auth-signup` signs up a fresh probe address (`testEmail` plus a random
  `+gl-…` tag, so it always reaches the same inbox) and requires a confirmation email.
- **Confirmation is enforced.** The check immediately tries a password login for that same
  unconfirmed address and requires `email_not_confirmed`. An account that can sign in before its
  address is confirmed is a failure, not a warning.
- **The human's click is visible.** After they click, the seeded account reads back with
  `email_confirmed_at` set through the admin API.
- **A session works.** `auth-session` signs in as the seeded account, requires
  `GET /auth/v1/user` to return that same user, requires an anonymous `GET /auth/v1/user` to be 401,
  and — only with `protectedPath` — requires an anonymous GET of the confirmed production URL plus
  that path to redirect to sign-in or answer 401/403. A 200 there is a failure; a 404 only warns
  (the path is probably wrong). A 401/403 is corroborated with one more anonymous GET of the
  production root: when that public route answers normally the whole origin is not walled, so the
  refusal is scoped to this path, and when the root is walled or unreadable too the leg is
  **inconclusive** (warn, naming the wall, not a pass) — a WAF, edge rule, visitor access or a
  maintenance page answers 401/403 without your app involved. Live-validated
  on 2026-09-24 against a deployed Vercel fixture: the declared route answered `401` anonymously
  (the pre-corroboration wording, `protected without a session`); the root comparison itself has
  mocked coverage only.
- **The app can read its own tables.** The checks probe the exposed tables AS the signed-in user
  (anonymity stays `rls-probe`'s job). Every table refusing the `authenticated` role warns: new
  projects no longer `GRANT` new tables automatically, so the app may be missing a migration.
  The probe names the tables it read, schema-qualified and grouped by verdict (`reachable:`,
  `denied:`, `undecided:`), up to four per line with `+N more` when a schema has more, so the
  count can be audited back to a table. Live-validated on 2026-09-24: the probe read the project's
  one exposed RLS-protected table (`1 reachable, 0 denied, 0 undecided`), which also live-exercises
  the `supabaseAuthedProbe` bearer fix; that run's line was a count without the table's name.

What stays human, and why the evidence says so:

- **Email delivery and the click.** golive cannot read an inbox. It never claims delivery: it reports
  the provider's own confirmation state. `auth:confirm-email` is closed only by `auth-signup` passing,
  and until then the report says the address is not confirmed yet.
- **The generated password.** `auth:test-user` generates one per run (32 random characters, carrying one
  character of every class a project's password policy can require, so a policy that demands a digit or a
  symbol never refuses it) and keeps it in that run's memory only — never in state, a report or evidence. A
  later run re-runs the step with a new password for the same account (recorded as `supabase.testUserId` +
  the address in `.golive/state.json`). A `verify`-only run holds no password, so `auth-session` skips with
  `blocked by: no password for the test account in this run`; `auth-signup` still passes on the
  provider reads alone (probe signup, its refused login, the account's `email_confirmed_at`, with an
  evidence line naming where the confirmed login itself is exercised), so `handoff` reports the
  confirmation handoff done without it. Run `plan` + `apply` after the human clicks, then re-run
  `verify`: the step rotates the password, and both checks run against the live account.

Caveats to pass on before enabling it:

- **It writes real users.** One test account per project, plus one throwaway probe account on every
  run of these two checks (every `verify` with `auth.e2e: true`, and the `apply` that carries the
  step). Both live in the project's user list until someone deletes them. `auth:test-user` carries
  `--confirm-live` for exactly this, and nothing here is a purchase. With `auth.isolation: true` there
  is a second seeded account (`+gl-isolation`) that `auth:isolation` creates and confirms the same way,
  and the `auth-isolation` check stores one small marker row per test account through the app's own
  `auth.isolationPath` route on every run — those two rows stay in the project's data too.
- **The auth email limit is the project's own, and the built-in mailer is rate-limited on top of it**
  (a handful of auth emails per hour; the live project's `rate_limit_email_sent: 2` behaved closer to
  one accepted send per window). Supabase then answers HTTP 429: `auth-signup` warns, `auth:test-user`
  fails with instructions, and one run of the journeys can exceed what that limit fits. The fix is the
  `auth:smtp` step (`auth.smtp: resend`), which now raises the limit to 30 per hour (or
  `auth.emailRateLimitPerHour`) as well as wiring the mailer — a limit that already spent its window
  still has to reset, so a run that hit 429 should wait before re-running. Check the spam folder — a
  project without DMARC often lands there.
- **A captcha on signup** (`hcaptcha`/`turnstile`) makes a scripted journey impossible: the step
  fails and `auth-signup` skips, never fails. Turn the auth captcha off for this project, or accept
  that the journey stays manual.
- **One account per address.** If the address already has a Supabase account, signup sends nothing
  and answers with an obfuscated user: the step adopts that account (rotating its password) and says
  so. Delete the test account in the dashboard to start over, or use another `auth.testEmail`.
- **Recovery spends auth emails too.** With `auth.recovery: true`, `auth:recovery` asks for one
  recovery email and `auth-recovery` asks for up to two more (the recorded address, then a fresh
  address with no account) every time it runs, against the same throttled mailer. A 429 warns instead
  of failing for exactly that reason.

### The password-recovery journey (`auth.recovery`)

A second opt-in, on top of the signup journey's confirmed account — the account the recovery rotation
touches is that same recorded test account, and no other:

```yaml
auth:
  recovery: true                # rotate the recorded test account's password through recovery
```

What the `auth:recovery` step (`--confirm-live`) does, in the app's own order:

- **Asks for the reset.** `POST /auth/v1/recover` for the recorded address, so a real recovery email
  lands in that inbox. An address with no account is answered the same way — Supabase refuses to
  reveal which addresses exist.
- **Mints its own link.** `POST /auth/v1/admin/generate_link` with `type: recovery` answers the token
  the email would have carried; it stays a `Secret` in that run's memory. That is what makes the
  journey provable without reading an inbox.
- **Exchanges it and sets the password.** `POST /auth/v1/verify` with `type: recovery` and the token
  returns a session, and `PUT /auth/v1/user` with that session sets the new password — exactly the
  calls a recovery page makes. Each of those requests carries `Bearer <token>`; a scheme-less header
  was a live-found defect (#25), and the adapter tests now assert the scheme on every one of them.
- **Keeps the rest of the run working.** The new password is stored under the key `auth:test-user`
  uses, so `auth-signup`/`auth-session` prove the same run with it; the replaced password and the spent
  token stay in that run's memory for the check to use.

What `auth-recovery` proves, and what stays human:

- **The request is accepted.** A 429 from the project's auth email limit warns, never fails: the mail
  throttle decides what a run can prove (roughly one accepted send per window on the built-in mailer).
- **No account enumeration.** The same request for an address with no account (a fresh
  `+gl-recovery-…` plus-tag) must get the same answer. A different status or acceptance is a finding:
  it answers "does this address have an account here?" for anyone who asks.
- **The token is one-time.** Replaying the token this run spent must be refused.
- **The password actually changed.** The new password signs in; the one it replaced is refused.
- **The window is named, not assumed.** When the project reports `mailer_otp_exp` (golive's
  `otpExpirySeconds`), the evidence says how long such a link stays usable; a project that does not
  report it is named as not reporting it.
- **The click stays human.** `auth:recovery-email` is non-blocking and closed by `auth-recovery`
  passing; golive never reads the inbox. A captcha on the project blocks the scripted request, so the
  check skips instead of claiming a pass.

Live-validated on 2026-09-24 on a disposable project, in one apply that also carried the SMTP write
above: the request was accepted for sending (HTTP 200), the admin-minted link was exchanged for a
session, the new password was set with it, and `auth-recovery` passed every leg — an address with no
account answered identically (HTTP 200 both, no enumeration), the spent token refused on replay (403
`otp_expired`), the new password signing in, the replaced one refused (`invalid_credentials`), and the
provider's 3600 s OTP window (`otpExpirySeconds`) named. The row recording it is in
`docs/VALIDATION.md`. What that run cannot show: the click in the inbox (golive never reads one), and
whether the message was ever delivered — its HTTP 200s are the provider accepting the send, and the
run's sending domain was the subject of [#52](https://github.com/mikehasa/golive-skill/issues/52).
A separate `verify` still skips the check, by design: the token and both passwords exist only in the
rotating run's memory.

### The account-isolation journey (`auth.isolation`)

The third opt-in, and the only one that answers a question about YOUR app rather than about Supabase:
can the account that just signed in read anything that belongs to another one? Supabase holds the two
accounts; your routes hold the answer.

```yaml
auth:
  isolation: true               # seed a second real account and read the app's own routes
  identityPath: /api/me         # GET: the signed-in caller's OWN id as JSON; 401/403 or a redirect without a session
  isolationPath: /api/notes     # GET: the caller's OWN rows; POST {"marker": "…"}: store one row for the caller; both refuse anonymous callers
```

`auth.e2e: true` and `auth.testEmail` are prerequisites: the first account is the one `auth:test-user`
seeds, and its password for this run comes from that same step. A plan warns and waits when the first
account is missing or not yet confirmed.

What the `auth:isolation` step (`--confirm-live`) does:

- **Seeds or rotates the second account** through the project's own `/auth/v1/signup` at
  `you+gl-isolation@example.com` (derived from `auth.testEmail`, so a later run finds the same
  account). Only the user id and the address are recorded (`supabase.isolationUserId` /
  `supabase.isolationUserEmail`); the generated password stays in that run's memory under the same
  per-user key `auth:test-user` uses — so `auth-signup`/`auth-session` keep working on the first
  account and `auth-isolation` can sign in as both.
- **Confirms it through the admin API.** `PUT /auth/v1/admin/users/{id}` with
  `{"email_confirm": true}`, re-read afterwards. This is deliberately not an inbox leg: the journey is
  about the app's data, and a second click would spend the project's throttled mail budget. The
  confirmation email the signup sends is a side effect, not a step.
- **Never touches an account golive did not seed.** It re-reads the recorded account first, refuses
  when it is gone, and its risk is `{ writes, live, replayable }` — the same discipline as its
  siblings, so a failed attempt from an older release may resume.

What `auth-isolation` proves, and what stays with the app:

- **Both routes refuse anonymous callers.** A 200 on either declared route is a **critical** finding.
- **Each account sees its own identity.** With each session, `identityPath` must answer with that
  account's own user id and never the other's (`user.id` from `/auth/v1/user` is what golive compares).
- **Each account reads only its own rows.** The check writes one unique marker row per account through
  `isolationPath` (**through the app, with that account's session**) and reads both routes back: a
  response carrying the other account's marker is a cross-account read and fails **critical**.
- **Sessions are real but not evidence of delivery.** Both accounts sign in with the passwords that
  run generated; nothing here is a claim about the inbox.
- **Skips, never passes:** the opt-in is off, either route is not declared (`auth:isolation-routes`
  carries the app-code task when they are missing), a route answers 404 or refuses the session token
  golive holds (the skip names the exact app-code task), the host cannot confirm the production URL,
  or the provider or the app rate-limits a request. A route that answers without the caller's own id
  or marker only **warns**: the absence of the other account's data is then not attributable.

The app-side contract, in one place: `identityPath` answers `GET` with the caller's own identity as
JSON and refuses (401/403 or a redirect) without a session; `isolationPath` answers `GET` with only
the caller's own rows and stores one row for the caller for a `POST` body `{"marker": "…"}`, refusing
both without a session. golive sends the account's session as an `Authorization: Bearer <token>`
header — the same token the app already gets from `supabase.auth.getSession()`. RLS with
`auth.uid() = user_id` is the usual way to satisfy the rows half; the identity half is a route the app
already has to have (or a two-line handler).

This is **implemented and mock-covered, not live-validated yet** on a real Supabase project: the live
run that exercises it (and the `docs/VALIDATION.md` row recording it) comes separately. Never present
account isolation as proven on a human's project until a live report says `pass` for `auth-isolation`.

## 3. Explain these in plain words

- **New API keys.** `sb_publishable_…` is meant for the browser and is safe *only* if row-level
  security (RLS) is on. `sb_secret_…` bypasses RLS and must stay on the server. The old `anon` /
  `service_role` keys don't exist on projects created after 2025-11-01 and are being removed in late
  2026. If the code reads `SUPABASE_ANON_KEY`, golive fills it with the publishable key; suggest a
  rename.
- **RLS in one sentence:** "the publishable key is public, so the database itself must decide who can
  read each row." A table readable with the public key is a real leak unless it's meant to be public.
- **Default grants changed.** New projects (since 2026-05-30) and all projects from 2026-10-30 no
  longer let the public roles reach new tables automatically. After a migration the app may get
  "permission denied" (`42501`) until the migration adds a `GRANT` plus RLS policies.
- **Edge Functions and `verify_jwt`.** By default functions demand a Supabase login token (JWT). Stripe
  webhooks don't send one, so they get 401. Set `verify_jwt = false` for that function in
  `supabase/config.toml` (`[functions.<name>]`) and verify the Stripe signature in code. `detect`
  notes this. Env names used only under `supabase/functions/` are Edge Function secrets, not host env
  vars; golive doesn't write them.
- **Don't push `supabase/config.toml` to production blindly.** It usually has
  `site_url = "http://127.0.0.1:3000"` and localhost redirects; pushing that breaks sign-in.
- **Transaction pooler (port 6543)** has no prepared statements: Prisma needs `?pgbouncer=true` (golive
  adds it when it detects Prisma), postgres.js / Drizzle need `prepare: false` in code.
- **Free projects pause** after about a week without activity. Probes fail until the human restores it.
- **Vercel integration (from Vercel's integrations store):** if installed, Supabase vars are already synced into Vercel.
  golive adopts them rather than writing duplicates.

## 4. Troubleshooting

| Symptom | What to do |
|---|---|
| "Cannot use automatic login flow inside non-TTY environments" | The human runs `supabase login` in a real terminal window (Terminal app / IDE terminal), not with `!`. Not `--token`: use the credentials file. |
| CLI credential cannot be reused | Follow the specific version/profile/store remedy in `doctor`; check §1 platform support. Do not print or copy the vendor store. A manual PAT is the alternative only for unsupported setups. |
| Keychain denied, locked or timed out | Nothing was clicked in time: re-run and answer the dialog macOS raises for the read-only `security` helper — **"Allow"** for this read, **"Always Allow"** to record it permanently for the item so it stops asking. golive never changes the Keychain and never falls back to another credential; a refusal or a locked Keychain is not retried. If the dialog cannot appear (CI, no desktop session), use the explicit-token alternative. Meanwhile the CLI-covered reads (project discovery, keys, the RLS query) still run through the CLI. |
| Credential invalid / expired (401) | If `via` is CLI login, refresh it with `supabase login --profile supabase`. If explicitly supplied, replace or remove that setting privately. No automatic account fallback. |
| `/profile` 403 but `/projects` succeeds | Project access can be valid. Explicitly select that existing project; creating needs organization/account management access. |
| 403 reading API keys | Check selected project, API Keys Read and API Key Secrets Read. A provider bug has also been reported; do not assume all scoped tokens fail or automatically broaden access. |
| Project paused (`INACTIVE`) | The human restores it in the dashboard (golive never does), waits until active, re-runs. |
| `INIT_FAILED` / `RESTORE_FAILED` | The human checks the project in the dashboard, or picks another with `init --project db=<ref>`. |
| Create refuses: same name in another org / several orgs | The human picks: `init --project db=<ref>` for an existing project, or creates it in the right org. |
| "no new project … is listed yet" after a create | Check the org's dashboard page; re-run apply once it appears (it's adopted by name). |
| Free-plan project limit on create | Human pauses/deletes an unused project or upgrades. Or adopt an existing project. |
| App gets `42501` / "permission denied" | Missing `GRANT` for the table (see default grants). Add grant + RLS policy in a migration. |
| `/rest/v1/` returns 403 "Access to schema is forbidden" | Expected since 2026: listing tables with the public key is blocked. `verify` uses the Management API / CLI instead. |
| Prepared-statement errors in production | Transaction pooler: Prisma `?pgbouncer=true`, postgres.js / Drizzle `prepare: false`. |
| Can't reach `db.<ref>.supabase.co` from the host | IPv6-only direct connection (fallback without a reusable credential). Repair the CLI login/store or use the explicit-token alternative and re-run so `DIRECT_URL` uses the session pooler; use `DATABASE_URL` at runtime. |
| Create timed out waiting for the project | Re-run `plan` + `apply` later; golive adopts the project it created and resets the password if it was lost. |
| Stripe webhook to an Edge Function returns 401 | `verify_jwt` is still on for that function. |
| Sign-in redirects to localhost or "redirect not allowed" | Auth Site URL / allowlist not updated. Re-run `plan` + `apply`, then `verify --only auth-redirects`. |
| Sign-in fails on preview URLs | Expected unless `auth.previewRedirects: true` (a risk) or a separate preview auth project. |
| `auth-policy` says signup is closed / confirmation off / password too short | Write the intended policy under `auth` in `golive.yaml` (`signup`, `requireEmailConfirm`, `passwordMinLength`), then `plan` + `apply` (the `auth:settings` step) and re-run verify. |
| `auth:settings` step fails with "is X after the write, not Y" | Supabase accepted the PATCH but reports another value: check Auth Config write permission for this token and the setting in the dashboard, then re-run `apply`. |
| `auth:settings` changes say `not confirmed: …` | Supabase does not return that setting through the API, so golive cannot confirm it. Confirm it in the dashboard; the rest of the write is unaffected. |
| `auth-policy` warns about the built-in mailer | Supabase's default SMTP is rate-limited; set `auth.smtp: resend` in `golive.yaml` and run `plan` + `apply` (the `auth:smtp` step writes the custom SMTP from a sending key golive issues, and raises the auth email rate limit), configure it by hand, or accept the built-in mailer with `auth.smtp: provider`. Turn off link tracking at the email provider. |
| `auth-policy` warns the rate limit is below a run's sends | The project's own auth email limit (`rate_limit_email_sent`) is under the four accepted sends one run of the journeys needs. Set `auth.emailRateLimitPerHour` (30 is Supabase's suggested starting point) or raise it in the dashboard, then `plan` + `apply` (the `auth:smtp` step writes it when `auth.smtp: resend`) and re-run verify. |
| `auth-policy` says `custom SMTP via Resend` but mail still fails | The read-back only proves the settings: `smtp_pass` is write-only, so a wrong or revoked key looks the same. Confirm the sender domain is verified (`email-verified`), the key exists at Resend, then re-run `plan` + `apply` (a fresh key) and send a real auth email. |
| Magic-link emails broken or slow | Supabase's default SMTP is rate-limited; the `auth:smtp` step replaces it with the app's own email provider (§2). Turn off link tracking at the email provider. |
| `auth.e2e` journey | Start with `auth.e2e: true`, `auth.testEmail` and `auth.protectedPath` in `golive.yaml`, then `plan` + `apply --confirm-live` (the `auth:test-user` step creates a real account). Click the link in that inbox, then `plan` + `apply` again and re-run `verify`. |
| `auth-signup` skips with `blocked by: auth:test-user` | No test account is seeded yet: run `plan` + `apply` with `auth.e2e: true` first. |
| `auth-session` reports `blocked by: no password for the test account in this run` | The generated password exists only in the run that seeded or rotated it, so a `verify`-only run cannot sign in. `auth-signup` still passes on the provider reads (`email_confirmed_at`) and closes the handoff; `auth-session` needs the password. Run `plan` + `apply` again (the step re-runs with a new password), then re-run `verify`. |
| `auth-signup` says the test account is not confirmed yet | The human has not clicked that link. golive cannot read an inbox; the `auth:confirm-email` handoff stays open until `auth-signup` passes. Check spam (the built-in mailer is rate-limited and new domains often land there). |
| `auth-signup` warns "rate-limited (HTTP 429)" | Supabase's built-in mailer limit (or a per-project email rate limit) refused the send. Wait for it to reset, then let the `auth:smtp` step configure custom SMTP and raise `rate_limit_email_sent` (`auth.emailRateLimitPerHour` overrides the 30 per hour it writes), or raise it in the dashboard, and re-run verify. |
| `auth:test-user` fails with "wants a captcha" | Auth captcha (hcaptcha/turnstile) is on for the project: turn it off for a test journey, or keep the journey manual. A scripted signup cannot pass a captcha. |
| `auth-signup` fails "accepted without sending a confirmation email" | `mailer_autoconfirm` is on (users are confirmed automatically): set `auth.requireEmailConfirm: true`, `plan` + `apply`, and re-run. If the address already had an account, that is why nothing was sent — see the next row. |
| `auth:test-user` says the address already has an account | Supabase answers a duplicate signup without sending mail. golive adopts that account and rotates its password; delete it in the dashboard (Authentication → Users) or set another `auth.testEmail` to start clean. |
| `auth-session` fails on the declared protected path (HTTP 200) | The route is served without a session. Make it redirect to sign-in or answer 401/403; if it renders a sign-in page with 200, choose a path that redirects in `auth.protectedPath`. A 404 there only warns: the path is probably wrong or not deployed. |
| `auth-session` warns the protected-path answer is inconclusive | The production root answered 401/403 (or something that is not a normal page) to the same anonymous request, so an edge rule, WAF, visitor access or maintenance page is refusing the whole origin: golive cannot tell that wall from your app's own refusal, and the leg is not a pass. Make the root publicly readable, keep the declared path refused, and re-run verify. |
| `auth-session` warns the signed-in user is denied by every table | The `authenticated` role has no `GRANT` (new projects stopped granting new tables automatically); the `denied:` evidence line names the tables. Add the grant plus RLS policies in a migration, then re-run verify. |
| `auth.recovery` check | Start with `auth.e2e: true`, `auth.testEmail` and `auth.recovery: true` in `golive.yaml`, seed and confirm the test account, then `plan` + `apply --confirm-live` (the `auth:recovery` step sends a real recovery email and rotates that account's password) and re-run `verify`. |
| `plan` warns `auth.recovery is on … no test account is recorded yet` | The recovery rotation only touches the account `auth:test-user` seeds. Apply the plan that seeds it (`auth.e2e: true`, `auth.testEmail`), click the confirmation link, then run `plan` again. |
| `plan` warns the test account `is not confirmed yet` (with `auth.recovery`) | A recovery of an unconfirmed address sends a confirmation, not a recovery link, so the rotation waits. Click the confirmation link in that inbox, then run `plan` again. |
| `auth-recovery` skips with `this run holds none of what the recovery check needs` | The new password and the spent token exist only in the run that carries the `auth:recovery` step. Run `plan` + `apply --confirm-live`, then re-run verify: a plain `verify` cannot prove a rotation it did not perform. |
| `auth-recovery` skips with `blocked by: auth:test-user` | No test account is recorded yet: run `plan` + `apply` with `auth.e2e: true` first, then the recovery journey. |
| `auth-recovery` fails: an address with no account was answered differently | Something in front of `/auth/v1/recover` (a proxy, WAF, edge function or cached response) is leaking whether an address has an account. Answer an unknown address exactly like a known one. |
| `auth-recovery` fails: the spent token resolved again | The verification endpoint accepted a one-time token twice. Check for anything answering `/auth/v1/verify` ahead of the project, then re-run verify. |
| `auth-recovery` fails: the password set through recovery cannot sign in | The project's password policy may reject the generated password, or the account changed during the run. Check the user in the dashboard, then run `plan` + `apply` again (a fresh rotation) and re-run verify. |
| `auth:recovery` or `auth-recovery` warns/errors with HTTP 429 | The project's auth email limit refused the send. Wait for the window to reset, then configure custom SMTP (§2) — the `auth:smtp` step also raises `rate_limit_email_sent` (30 per hour, or `auth.emailRateLimitPerHour`) — and re-run. |
| `auth.isolation` journey | Start with `auth.e2e: true`, `auth.testEmail`, `auth.isolation: true`, `auth.identityPath` and `auth.isolationPath` in `golive.yaml`, have the app expose both routes (the `auth:isolation-routes` handoff holds the contract), then `plan` + `apply --confirm-live` (the `auth:isolation` step creates and confirms the second account) and re-run `verify`. |
| `plan` warns `the FIRST account comes from auth.e2e …` | `auth.isolation` needs the first test account too: set `auth.e2e: true` and `auth.testEmail`, then run `plan` again. |
| `plan` warns the test account `is not confirmed yet` (with `auth.isolation`) | The isolation check signs in as both accounts, so the first one has to be confirmed: click its confirmation link, then run `plan` again. |
| `auth-isolation` skips with `no app route is declared` | `auth.identityPath` / `auth.isolationPath` are not set. Get the app's code changed (the `auth:isolation-routes` handoff), name both routes in `golive.yaml`, then re-run `verify`. |
| `auth-isolation` skips with `no password for … in this run` | Both passwords exist only in the run that seeds or rotates them. Run `plan` + `apply --confirm-live`, then re-run `verify`. |
| `auth-isolation` skips with `HTTP 404: the app does not implement the declared … route` | The route is not deployed (or the path is wrong). Deploy it as the skip's app-code task describes, then re-run `verify`. |
| `auth-isolation` skips with `the route refused the session token golive holds` | The app is not reading the caller's session from `Authorization: Bearer <token>` (a cookie-only route, or a proxy in front). Fix the route or accept that golive cannot exercise isolation there. |
| `auth-isolation` fails `the declared … route is served without a session` | That route answers anonymous callers: make it 401/403 or redirect to sign-in when there is no session. |
| `auth-isolation` fails `carried the OTHER account's id` / `carried …'s row` | A cross-account read: the route answered with another account's data. Scope it to the caller (RLS `auth.uid() = user_id`, or the same filter in the route) and check anything that widens it (shared cache, service-role client, join), then re-run verify. |
| `auth-isolation` warns `was not in its own rows` | The route said 200 but did not return the row golive just wrote for that caller, so the read-back proves nothing either way. Return the caller's own rows from `auth.isolationPath` and re-run verify. |

## Unverified

- Whether publishable keys are blocked from `/rest/v1/` exactly like anon keys (assumed yes).
- The exact enforcement date for removing legacy keys ("late 2026", not final).
- Custom SMTP writes: **live-validated on 2026-09-24** — the accepted write and its read-back are
  observed (`smtp.resend.com`, port 465, user `resend`, sender `auth@mail.trytofu.xyz`) together with
  the auth email rate limit raised in the same request (`auth email rate limit: 2 → 30 per hour`,
  `rate_limit_email_sent`), after an earlier run's 400 on the port (`smtp_port: Invalid input: expected
  string, received number`) was fixed by sending the string. What remains unverified: `smtp_pass` is
  write-only (the API answers
  a hash), so the read-back
  confirms only host/port/user/sender and the limit — never that the key in effect is the recorded one,
  or that mail leaves the project; the rate limit is still the one setting whose refusal golive only
  warns about (the provider is free to keep its own value); and the auth email throttle's exact
  behaviour is also unconfirmed — this run's project accepted every send it made at 30/hour, while an
  earlier project's `rate_limit_email_sent: 2` accepted one send and refused the next 25 seconds later
  rather than allowing a clean two per window.
- GoTrue answer shapes still modelled from its documented behaviour: an obfuscated duplicate signup
  and a captcha refusal. The disposable live run (2026-09-23) exercised an accepted signup, the
  confirmation email request, the `email_not_confirmed` login refusal and the 429 rate-limit refusal.
- What the 2026-09-24 app-side evidence shows, and what has changed since: that run's signed-in probe
  read the project's one table as a count, and any 401/403 on the declared `auth.protectedPath` read
  as protection — a WAF, edge rule or maintenance page would have read the same. Both were evidence
  limits, not wrong results; the change tracked as #30 fixed the text (the probe names the tables per
  verdict, and a refused path is corroborated against the public root, so an origin-wide wall makes
  the leg inconclusive). Neither fix has been live-exercised yet: they are mock-covered. Inbox
  delivery and the human's click stay human-confirmed by design, and the confirmations in both runs
  were applied through the Auth Admin API (`PUT /auth/v1/admin/users/<id>` with `email_confirm:
  true`) rather than by clicking the seeded account's own email.
