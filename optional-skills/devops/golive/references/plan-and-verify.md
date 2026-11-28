# Detect, plan, apply, verify: details

Load this when you need to explain a detect finding, a plan step, a handoff, why a check skipped, or
what a `status` drift item means. Provider-specific notes live in the other references.

## 1. Detect

`detect --json` returns `detect` (framework, `providers`, `envRefs`, `webhooks`, `notes`, …),
`env.mapped` / `env.unmapped`, `findings` and `suggestedStack`. Exit code 2 = a critical finding.

**Browser-exposed names.** Besides framework prefixes (`NEXT_PUBLIC_`, `VITE_`, `PUBLIC_`, …), detect
reads the framework config. Names inlined by next.config `env: {}`, by Vite/Astro `define` or webpack
`DefinePlugin` (`process.env.X` / `import.meta.env.X` keys, and env reads inside define values), or
matching a custom Vite `envPrefix` or SvelteKit `kit.env.publicPrefix`, count as client-exposed. golive
refuses to write a server secret into them (`secret-in-client-env`) and adds a note naming them.

**`config-inlines-all-env` (critical).** Raised when the config defines or spreads the whole
`process.env` / `import.meta.env`, a whole `loadEnv(mode, dir, '')` object, computed
`process.env.${k}` keys, or uses an empty public prefix. Every web-side env name is then treated as
public, so no server secret is written to that app's host until the config is fixed. Names used
only by Supabase Edge Functions (`supabase/functions/`) are not host env vars and are unaffected.
This also covers a `define` / `env` / `DefinePlugin` value that isn't an object literal: an
identifier, a call, shorthand `{ define }`, a spread identifier, and later mutations of the object.
If such a value (or a `'process.env'` define) can't be resolved and doesn't inline the whole env, a
**note** says "<file> <via> is set from an expression golive cannot read; make sure it holds no server
secrets". Read the config and confirm.

**Other detect output.** `configs` lists config files found, including Prisma schemas
(`schema.prisma`, `prisma/schema.prisma`, the `prisma/schema` directory, `prisma.config.ts`, a custom
package.json `prisma.schema` path). A `@prisma/client` or `prisma` dependency adds the note
"Prisma detected".

**Webhooks.** Each `webhooks[]` entry has `provider`, `path`, `file`, `verifiesSignature` and `events`:
the Stripe event types the handler references as string literals (case labels, `===` comparisons,
arrays, handler maps), sorted. `init` subscribes the endpoint to exactly these unless you pass
`--events`. If `events` is `[]`, a note says so: read the handler and pass `--events a,b`. Check
detected events against the handler the same way you check the path.

## 2. Plan

`plan --json` → `planId`, `steps[]` (`id`, `title`, `kind`, `writes`, `needs`, `preview`,
`dependsOn`), `handoffs[]`, `unmappedEnv`, `warnings`, `findings`. `targets[]` lists only the steps
that carry a structured `destination` (the project steps), so a teardown plan's `targets` is empty:
read the destinations from `steps[].preview` (and the step's `needs` confirm flags) instead. Previews
are deterministic: the same state gives the same `planId`. `teardown --json` returns the same shape;
its steps carry `kind: 'destroy'` and `needs` includes `--confirm-destroy`.

**Step intent.** A step may also carry a secret-free `intent`: what it writes beyond its preview text
(source project ids, key fingerprints, endpoint ids, the hosting project, a previous attempt's time).
It is part of the step's hash and the `planId`. `apply` skips a completed step only when preview,
intent, risk, dependencies and kind are all unchanged, so a step with the same preview as last time
but a different intent runs again: a db project switch, a Stripe key rotation or a domain re-attach
really lands. The deploy step's intent includes the env writes it picks up.

**Teardown.** `teardown` enumerates only golive-created resources with their ownership proofs
(provider markers, state fingerprints, the host project's creation marker): DNS records golive owns,
recorded webhook endpoints, issued sending keys and the created host project. A removal re-checks
ownership before deleting, and every removal treats "already gone" as done. A delete is confirmed by
a provider read, never by the delete response alone: the zone is re-read for a DNS record, the
project for the host project, and the provider's own endpoint list for a webhook endpoint — a
resource the provider still lists fails the step, while a read it cannot answer (auth, network) warns
instead of passing. No read exists for an issued sending key (the provider offers issue and revoke
only), so a revoked key stays **unverified**: the step reports `warn` naming the provider dashboard,
never a pass. A removal the provider's answer says is gone also forgets that resource's recorded
baseline — the `dns:<zone>|<type>|<name>` entry for a record, the endpoint id for a webhook, the key
id for a sending key — and removing the host project forgets its deploy facts (the `deployed:…` time
marker, the recorded deployment identity and the completed deploy step), so a later `golive status`
does not report golive's own teardown as drift and a project created again in the same repo is
deployed again rather than inheriting "production was deployed". A resource the provider still lists
or refused to remove keeps its record. Resources it cannot remove — adopted projects, Supabase/Neon
projects, the Resend sending domain — appear as non-blocking `manual` handoffs, and so does anything
the inventory could not even read: a DNS zone whose provider golive cannot use, cannot tell
golive-owned records apart in, or cannot delete from, and a linked host project golive cannot reach or
whose host exposes no project deletion. Each such row names what remains (from state, never a guess),
why golive cannot remove it and the exact fix — reconnect the provider and run `golive teardown`
again, name that provider in `golive.yaml` again, or delete it in the provider's dashboard — so a
teardown never goes quiet about records or a project left behind. Records or endpoints a human created
are never deleted.

**Which project.** Every plan has a step `project:hosting` / `project:db` naming the provider,
project name (id), team/org if known, where the choice came from, and the logged-in account.
- Already linked: a zero-write pin. At apply time it refuses if the linked project changed since
  approval, then pins it in `.golive/state.json`. After a full apply, `plan` shows only these pins,
  and applying them is a no-op (`skipped`).
- `golive.yaml` `projects.<axis>` disagrees with the linked project: `plan` warns and uses the linked
  one.
- Nothing linked: a configured (`init --project`) or same-named project is selected; otherwise a
  **Create** step whose preview lists existing projects the human could use instead. Ask them, and
  use `init --project <axis>=<name>` to adopt one.

**Deploys.** Production is (re)deployed when this plan writes production env, when a production env
write is still waiting for a deploy, when the last deploy failed, or when golive has never deployed
production. Preview-only env changes don't trigger a production deploy. If production was never
deployed by golive, `deploy:production` runs **before** `domain:attach`, and
`deploy:production:final` redeploys after env writes that need the domain (the webhook secret). Once
deployed, attaching a domain neither waits for nor triggers a deploy. A production deploy planned
while state records no successful production deploy for that target carries `risk.live`, so `apply`
also needs `--confirm-live`: the plan approval covers what the deploy writes, and this flag is the
human's separate yes to writing production there for the first time — explain that when you present
the plan (`deploy:production`, and `deploy:production:final` when it runs as part of that first
deploy, both carry it, and each step's preview says so). A failed attempt records no deploy, so the
gate stays on the next plan; once a successful production deploy is recorded for the target, later
deploys carry `risk: { writes: true }` alone and need no extra flag.

A successful deploy records the deployment the provider reported: the `deployed:production` time
marker plus, when the provider gives one, its own identity under `deployed:production:id` as
`<provider>|<deployment id>|<url>|<time>` in `.golive/state.json` — the name a later promotion or
rollback of exactly that deployment would use. A provider that reports no identity records the
marker alone; golive never derives one from the URL.

**Preview deployments (`release: { preview: true }` in `golive.yaml`).** With the opt-in — and
`preview` in `targets` — `plan` adds two steps at the end (a stack without the opt-in is unchanged),
and `apply` needs `--confirm-live` too when a live-mode value fills a preview env name:

- `preview:deploy` is a **create** (`risk: { writes: true }`, never `replayable`): it deploys the
  current working tree — the preview names the branch when local `git` reports one, because both hosts
  build what is on disk, not a commit — to the host's preview target. It depends on `project:hosting`
  and `env:preview`, and records the provider's own identity for the deployment it made as
  `deployed:preview:id` (the same shape production records, so a promotion can name it). Its
  preview names the provider and project, the env target, the preview URL the provider reports per
  deployment, and whether the preview shares production's source project: golive fills preview env from
  the same db/auth project as production, so a preview reads and writes production's data. It is
  planned for the same reasons a production deploy is (no preview deployed yet, preview env changes in
  this plan, the last preview deploy failed) plus a failed release check — a re-planned preview deploy
  always makes a new deployment, so the gate never re-checks a bundle golive did not replace.
- `release:check` writes nothing (`risk: { writes: false }`) and declares `preview:deploy` as its
  prerequisite — a declared edge, not an ordering accident: the plan's dependency helper keeps only step
  ids already tracked, so a gate built before its deploy declares no prerequisite and `apply --only
  release:check` runs the check against a deployment the plan never made. With the edge, that `--only`
  call is refused while `preview:deploy` has no completed evidence for the plan. It runs two checks as
  its inline verification and **fails the step when one fails**, and the runner stops the plan there.
  What that stops turns on the plan, and the step's own text says which: in a **cut** plan the gate is
  the last step, so it stops nothing emitted before it — that plan's own production deploy included —
  and it gates the promotion, whose plan re-runs the check before writing; in the **release** plan the
  gate is `promote:production`'s prerequisite, and a red gate stops the re-point. Its intent is the
  deploy's intent plus the previous attempt, so a re-plan checks again (the `domain:verify` idiom). It
  has a second mode: as the promotion's prerequisite it depends only on `project:hosting` and re-reads
  the preview deployment golive already recorded — the exact deployment `promote:production` would make
  production.

Adding these step ids changes a plan's id, so an approval that was not applied must be re-planned.

**Promotion and rollback (`release: { preview: true, promote: true }`, or `release: { rollback: true }`).**
Two production re-points, both **implemented and mock-covered, not live-validated**, and both planned
only when the human opted in:

- `promote:production` (`kind: 'deploy'`, `risk: { writes: true }`, `dependsOn: ['release:check']`)
  re-points production at the preview deployment golive recorded and that check just re-read. Its
  preview names the provider's own deployment id and URL, when golive recorded it, the env target it
  was built for, what production serves before, the gate, and that production will change. There is
  **no extra confirmation flag**: the plan id, the named deployment and the fresh gate are the
  approval. Because the provider reports a deployment's id only once the deployment is made, the plan
  that can name it is a different plan from the one that deploys it — with the opt-in set, a plan is
  either **cut** (`preview:deploy` + `release:check` at the end of the plan) or **release**
  (`release:check` + `promote:production`), and its preview says which. A **cut** plan changes nothing
  about the rest of the plan: its other steps, a production deploy included, are emitted before the
  preview steps, so the gate stops nothing that came before it; the gate's grip is the promotion. While
  `release.promote` is set, every plan asks for a release; remove the flag to stop planning releases.
  `run` re-reads the target deployment and
  what production serves before writing, refuses when the provider cannot answer either read (or the
  deployment is gone/not ready), then re-reads production after the write and records nothing unless
  the provider confirms the switch. Production already serving the target is a no-op.
- `release:rollback` (`kind: 'deploy'`, `risk: { writes: true }`) re-points production at an earlier
  deployment from golive's own trail (`deployed:history`). Its target is never a deployment golive did
  not create: a dashboard, Git or PR-built one stays with that provider. While `release.rollback` is
  set, a plan contains the rollback and no preview steps, and it stops planning one once golive has
  rolled production back to that deployment. It is not a `destroy` step
  (nothing is deleted) and not `replayable` (a production re-point keeps the cross-release stop, so a
  rollback recorded under an older release is refused and reconciled instead of repeated). It is never
  automatic — a failed check never triggers one — and once golive has rolled production back, a later
  plan reports that instead of planning the same rollback again.
- Both hosts differ: **Netlify** re-reads `published_deploy` and can restore an earlier deploy, so both
  steps work there; **Vercel** has no production-deployment read and no promote/rollback call, so no
  promotion or rollback is planned and a warning names the missing capability.

**Production URL before the first deploy.** Without a custom domain, the host's production URL is used
for the webhook, the auth site URL and `SITE_URL`-style vars only after golive has deployed production
once. Until then those are left out with a warning; run `plan` again after the first deploy.

**Env steps.** Each env-writing step is verified by a step-scoped check `<step-id>:env-written` (only
the names that step wrote, on its target). Vars golive manages are rewritten when the provider, the
mode, the db/auth project behind them, or the payment account/key behind them changes.

**Domain.** `domain:attach` → `domain:dns` (DNS writes, `--confirm-dns`; verified against the zone's
records) → `domain:verify` (asks the host to verify ownership; on Vercel this can move the domain
from another account of the same host). `pending` there is not a failure: DNS is propagating, and
each new `plan` asks again (its preview shows `previous request: <time>`). While the domain isn't
live, each plan also re-sends `domain:attach` (idempotent). If at apply time the host requires
different records than the approved plan listed, `domain:dns` refuses and writes nothing: run
`plan` again and get re-approval with `--confirm-dns`. `domain-live` is left to `verify`. If the DNS zone lookup fails (token
permissions, rate limit, network), `plan` warns and plans no DNS step.

**Auth redirects.** Preview-deployment wildcards go into the (production) auth redirect allowlist only
with `auth.previewRedirects: true` in `golive.yaml`, and each such line is flagged as a risk.
Otherwise `plan` warns that sign-in on preview URLs won't work.

**Auth policy.** `auth:settings` writes only the values from `golive.yaml` `auth` that differ from
what the provider reports (`signup`, `requireEmailConfirm`, `passwordMinLength`), then re-reads them;
the step's own `auth:settings:applied` result and the `auth-policy` check carry that evidence. A
setting the provider does not report back shows as `not confirmed:` in the step's changes and does
not fail it; a value the provider keeps reporting differently fails the step. Changing a value in
`golive.yaml`, or changing it back in the provider dashboard, changes the step's intent, so it runs
again.

**Auth SMTP (`auth.smtp: resend`).** A separate step (`auth:smtp`, `risk: { writes }`, no new flag)
points the auth project's custom SMTP at Resend: `smtp.resend.com:465`, the user `resend`, the sender
`email.from` already uses, and an SMTP password that is a sending key — the one the email journey
issued in this run, otherwise one golive issues for SMTP alone (`golive-…-smtp`, state key
`<provider>.keyId@smtp`, so `teardown` can revoke it). The same write raises the project's **auth email
rate limit** (`rate_limit_email_sent`) to 30 per hour, or to `auth.emailRateLimitPerHour` from
`golive.yaml`: the provider keeps its own limit with custom SMTP in place and one run of the journeys
needs four accepted sends. Unlike an SMTP field, a limit the provider keeps at another value does not
fail the step — the provider's own setting, reported by `auth:smtp:applied:rate-limit` (medium) — and
one it never reports back is named as unconfirmed. Otherwise it writes only the fields that differ and
plans nothing once they hold; its own `auth:smtp:applied` result plus the `auth-policy` check carry the
evidence. The password is write-only (`smtp_pass` answers a hash, never the value), so what is
confirmed is the settings golive can read back and the write itself — a real auth email arriving is
the only full proof, and the journeys below run after this step so their sends are the one that
counts. Auth emails still going through the provider's built-in mailer warn (its limit can refuse the
sends a journey needs, HTTP 429, roughly one accepted send per window) unless `auth.smtp: provider`
accepts that deliberately.

**Auth signup journey (`auth.e2e: true`).** The `auth:test-user` step creates ONE real account in the
project through the provider's own signup endpoint (`risk: { writes, live }`, so it needs
`--confirm-live`) and records the user id and the address in `.golive/state.json`; the generated
password lives only in that run's memory. Its intent carries the previous attempt, so a fresh
`plan` + `apply` re-runs it — as a password rotation on the same account — which is how a later run
can prove login again. `auth:confirm-email` (non-blocking, verified by `auth-signup`) is the human's
click in the inbox. Both checks are opt-in:

- `auth-signup` signs up a fresh probe address (`auth.testEmail` plus a random `+gl-…` tag) and
  requires a confirmation email, requires an immediate login refusal (`email_not_confirmed`) and
  requires the seeded account to read back as `email_confirmed_at` after the click. Those three
  provider reads are the whole pass rule, so a `handoff` — or a `verify` outside the seeding apply —
  can report a complete handoff as done. The confirmed account's own sign-in is added as extra
  evidence when this run holds that account's password (the apply that seeded or rotated it); when it
  does not, an evidence line says so and names where the login is exercised instead of skipping.
  golive cannot read an inbox: delivery and the click stay human-confirmed, and the evidence says so.
- `auth-session` requires a session for the seeded account, requires `GET /auth/v1/user` to return
  the same user, requires an anonymous request to be 401, and — with `auth.protectedPath` — requires
  an anonymous GET of the host-confirmed production URL plus that path to redirect or answer
  401/403 (a 200 fails; a 404 warns). A 401/403 there is corroborated with one more anonymous GET of
  the production root: an edge wall (a WAF, edge rule, visitor access, a maintenance page) refuses
  the root too, so the leg reports **inconclusive** (warn, naming the wall) instead of protection
  whenever the public route is not readable. Its table probe uses the session token — anonymity
  stays `rls-probe`'s job — and names the schema-qualified tables it read per verdict (up to four
  names each, then `+N more`), so the count line can be audited back to a table.

Both checks write when they run (one throwaway account per run) and skip, never fail, without the
opt-in, without `auth.testEmail`, without a usable provider credential, or when a captcha blocks the
scripted signup. `auth-session` also skips when this run holds no password for the seeded account:
without one there is no session to inspect.

**Auth password recovery (`auth.recovery: true`).** `auth:recovery` sends a real recovery email for the
recorded test account, mints a recovery link through the provider's admin API (so golive never needs to
read the inbox), exchanges its token for a session and sets a new password with that session, then
keeps that password under the key `auth.e2e` uses — so `auth-signup`/`auth-session` keep working in the
same run. Its risk is `{ writes, live, replayable }`: `--confirm-live` is required, it re-reads the
recorded account and its provider state before acting (and waits at plan time until that account reads
back confirmed, warning instead of planning), and it only ever touches that one account.
`auth:recovery-email` (non-blocking, verified by `auth-recovery`) is the human's click. The check needs
what that step left in the run's memory — the spent token and the two passwords — so a `verify` outside
that run skips with `this run holds none of what the recovery check needs`; it asserts that an address
with no account is answered like a known one (a different answer is account enumeration and fails),
that the spent token is refused on replay, and that the new password signs in while the replaced one
does not. A 429 anywhere in it warns, never fails: the provider's mail throttle decides what a run can
prove.

**Auth account isolation (`auth.isolation: true`).** The second half of the journey: with TWO real
accounts, golive can ask whether one signed-in account can read the other's data through the app. The
`auth:isolation` step (risk `{ writes, live, replayable }`, so `--confirm-live`) seeds or rotates a
SECOND account beside the one `auth:test-user` seeds — the same provider signup, the address derived
from `auth.testEmail` (`you+gl-isolation@example.com`), the password again only in that run's memory —
and confirms it through the provider's **admin API**, re-reading it before the step reports done: a
second inbox click would spend the throttled mail budget on a journey whose subject is the app's data,
not delivery. The step reads the recorded account and its provider state before acting and only ever
touches accounts golive created and recorded. `auth.e2e: true` and `auth.testEmail` are prerequisites
(the first account's password comes from `auth:test-user` in the same run); a plan says so and waits
when the first account is missing or unconfirmed.

The app has to answer for itself: `auth.identityPath` and `auth.isolationPath` in `golive.yaml` name
two routes — one that returns the signed-in caller's OWN identity (its provider user id) as JSON, one
that returns ONLY the caller's own rows (a GET) and stores one row for the caller for a POST body
`{"marker": "…"}`. Both must refuse an anonymous request (401/403 or a redirect). `auth:isolation-routes`
(non-blocking, verified by `auth-isolation`) hands that app-code task to Hermes or the human when
either route is not declared.

`auth-isolation` needs both accounts' passwords, so it only passes in the run that seeds or rotates
them; otherwise it skips with `blocked by: no password for … in this run`. With both sessions it reads
both routes anonymously (**a 200 is a critical finding**, whoever the caller is), then reads the
identity route as each account (each must answer with its own id, never the other's) and writes one
unique marker row per account **through the app's own rows route**, then reads it back as each: a
response carrying the other account's marker is a cross-account read and fails critically. It skips —
never passes — when the opt-in is off, either route is not declared, the app answers 404 (the app-code
task is named), the route refuses the session token it was given, the host cannot confirm the
production URL, or the provider or the app rate-limits a request. A route that answers without the
caller's own id or marker only warns: the absence of the other account's data is then not attributable.
It never probes a table anonymously — that stays `rls-probe`'s job.

**Email.** `email:verify` is re-sent on each plan while the domain is pending; its preview shows
`previous request: <time>`.

**Secrets exposed.** A critical detect finding (or a client-prefixed secret name) blocks secret writes
for the affected names: env, payments keys, email key, and the whole webhook step. A blocking
handoff `secrets:exposed` tells the human to fix the code first. It disappears once detect stops
reporting the finding.

**Webhook.** The preview names exactly the endpoint apply will adopt, including one the human created
at the same URL: whether its events are kept, whether it will be re-enabled, and whether the old
endpoint is deleted or left for the human to delete. Apply refuses without writing if the endpoint
changed since approval. When the production URL changed, it also says the golive-created endpoint for
the old URL is deleted once the new secret is stored. If the host would refuse the secret
(`EnvStore.canSet`), a blocking `<adapter>:webhook-env` handoff replaces the step and no endpoint is
created. With a guided host, a blocking `<adapter>:webhook-guided` handoff (verified by
`webhook-registered`) asks the human to create it. See `stripe.md`.

## 3. Handoffs

Each handoff has `id`, `why`, `action`, `blocking`, and optionally `verifiedBy` (a check id) or
`manual: true`. `handoff --json` adds `done` and `evidence`:
- `done: true`: its check passed. Only a passing check closes a handoff.
- `done: false`: open. Its check ran and did not pass, or it has no check and isn't `manual` (it is in
  the plan only because its condition still holds, e.g. `secrets:exposed`, `stripe:webhook-env`).
  Blocking ones count in `apply`'s open handoffs and the report's `blocking`.
- `done: null`: golive cannot verify it (a `manual` item, or its check skipped). Confirm it with the
  human and list it as **not verified by golive** in your summary. `unverified` lists them. When a
  check can only skip outside the run that did the work (e.g. `auth-recovery`, whose token and
  passwords live in that run's memory), its evidence leads with the recorded outcome of the plan step
  that check verifies — a step recorded `done` with its plan id and time, or a failed one with its
  recorded error — so a skip never reads as if the work never happened; the skip itself is still named.

Common ones: `login:<provider>`, `project:<axis>` (golive can't create it), `secrets:exposed`,
`db:password`, `stripe:activate`, `stripe:publishable-key:<mode>` (ask for the `pk_` key in chat,
then `init --stripe-publishable <mode>=pk_<mode>_…`; or the human adds it to the host dashboard),
`stripe:secret-key:<mode>` (the credentials-file instructions in its `action`; never chat),
`email:dns` / `domain:dns` / `domain:attach` (records or setup at a provider golive can't write),
`guided:<axis>`, `env:<target>` for a guided host (per target: the human sets the listed names in its
dashboard; `env-parity` can't read a guided host, so it stays `done: null`),
`stripe:webhook-env` / `stripe:webhook-guided` (see Webhook above), `auth:confirm-email` (non-blocking,
verified by `auth-signup`: the human clicks the confirmation link in their own inbox, which golive
cannot read), `auth:recovery-email` (non-blocking, verified by `auth-recovery`: the same for the
recovery link — golive requests it and mints its own copy, the human clicks theirs),
`auth:isolation-routes` (non-blocking, verified by `auth-isolation`: the app must expose the two
declared routes, which only Hermes or the human can add — golive names exactly what they answer and
drops the handoff once both paths are declared), and
`auth:redirects` for a guided auth provider (manual, non-blocking: confirm it with
the human, name it as unverified).

**Ownership document.** `handoff --write --json` also writes `GOLIVE_HANDOVER.md` and
`.golive/handover.json` (paths are reported as `handoverPaths`; `--force` replaces a file golive did
not generate — a file without the "Generated by golive" marker is never overwritten, and a symlink is
never followed). It is built from recorded state, `golive.yaml`, the cheap provider reads
`doctor`/`verify` already make (`auth().via`, project scope and URLs) and the teardown inventory —
never from `state.secrets` or an `Outputs` value, never from a billing endpoint. Sections: accounts and
login route; resources created (id, public URL, ownership proof, created-by-golive vs adopted); costs
and recurrence (no figures at all: golive reads no plan, quota or usage data); what is manual (the open
handoffs plus recurring jobs — DMARC tightening, key rotation, backups, domain renewal); if it breaks
(per-subject `doctor` / `verify --only` commands, `plan` → `apply` → `verify`, and where evidence
lives); retirement (the teardown inventory with its `--confirm-destroy` / `--confirm-dns` gates); and a
provenance footer. Every row carries `[verified by golive]`, `[recorded <date>, not re-checked]`,
`[not verifiable by golive]` or `[unknown]`. Treat the last three as unverified: this document is not
drift detection. It holds no secret values, but it names accounts and resources — review it before
sharing it. Recommend adding `.golive/`, `GOLIVE_REPORT.md` and `GOLIVE_HANDOVER.md` to the app's own
`.gitignore`: state, report and handover carry resource ids and account names, while credential
values live outside the repo in the private credentials file.

## 4. Verify

`verify --json` runs every check. `--only id,id` writes a partial report containing only results
from this invocation; omitted checks are listed and previous results are not reused. Neither scope
establishes whole-app readiness: review pending plan steps and app functional acceptance. Verification
writes `.golive/report.json` and `GOLIVE_REPORT.md` (old `SHIP_REPORT.md` files are preserved).
Exit 2 when any check fails. `summary` counts
`pass`, `fail`, `warn`, `skip`, `blocking` (open blocking handoffs) and `manual` (blocking ones golive
can't verify).

**`skip` = blocked or not applicable, never passed.** Evidence `blocked by: <id>` names what's
missing: `login:<adapter>`, `project:hosting`, `project:db`, `deploy:production`, `email:domain`, or a
plain reason (e.g. `no publishable/anon key`, `the hosting token's role cannot read production env
vars`). Only `accounts` fails for login problems; fix it first, then re-run `verify`.

**Active probes** (`bundle-secrets`, `webhook-unsigned`, `auth-session`'s protected-path GET and the
public-root GET that corroborates it, `auth-isolation`'s route reads and its one marker row per test
account, and the key `rls-probe` takes from the bundle) only target the production URL the hosting
adapter reports
for the linked project, never `config.domain` directly. If the host can't confirm it, the check skips with `cannot confirm <url>
belongs to your project yet`. If the host reports another origin than `config.domain` (e.g. the domain
isn't verified at Vercel yet), `webhook-unsigned` probes the host's URL and says so. `domain-live`
does resolve and GET `config.domain`. `preview-bundle` is the one probe outside production: it scans
the preview URL the hosting adapter reports for the linked project — never a URL golive only has in
state — and a protected preview skips instead of being reported as scanned.

| id | passes when | skips when |
|---|---|---|
| `accounts` | every automated provider authenticates | nothing chosen |
| `env-parity` | every referenced name exists per target (names only; unmapped missing names only warn) | guided host; a source provider not logged in (`blocked by: login:<id> (NAME@target, …)`); the role can't read production env |
| `domain-live` | automated host reports the domain `ok` (`pending` warns), it resolves, HTTPS answers 2xx/3xx; guided/no host: DNS + HTTPS only, evidence says the attachment isn't confirmed | no domain; `blocked by: login:<host>` / `project:hosting`; the host's status lookup errors (`cannot confirm <d> is attached …`) |
| `bundle-secrets` | no known credential patterns in the complete bounded fetch set | production URL not confirmed; **warns** on asset fetch failures, scan limits or off-origin production redirects |
| `rls-probe` | tables in exposed schemas aren't readable with the publishable key; advisors clean | `blocked by: project:db`; no publishable/anon key |
| `db-connection` | the selected Neon compute accepts a fixed read-only query and returns the expected database and role; no schema/Auth/app-isolation claim | no connection-probe capability; `blocked by: login:<db>` / `project:db` |
| `auth-redirects` | site URL and allowlist point at production, no localhost | guided auth; `blocked by: deploy:production` |
| `auth-policy` | the reported signup/confirmation/password policy matches golive.yaml `auth` (below 12 characters, a built-in mailer, an unapplied `auth.smtp: resend` or an auth email rate limit below a run's four sends only warn); the mailer is reported as the provider's built-in one or as custom SMTP via Resend, and the SMTP password is never read back; evidence lists the effective values | guided auth; `blocked by: login:<id>` / `project:<axis>`; the provider reports no policy fields |
| `auth-signup` | a fresh probe address got a confirmation email, could not sign in before confirming, and the seeded account reads back confirmed (`email_confirmed_at`) — the confirmed account's own sign-in is extra evidence when this run holds its password (delivery stays human-confirmed) | `auth.e2e` off; no `auth.testEmail`; guided auth; `blocked by: login:<id>` / `auth:test-user`; a captcha blocks signup; **warns** on a 429 or while the account is still unconfirmed |
| `auth-session` | the seeded account's session is accepted for the same user, an anonymous request is 401, and a declared `auth.protectedPath` is refused while the production root still answers (each refusal is corroborated against that public route); the signed-in table probe names the tables it read per verdict | `auth.e2e` off; guided auth; `blocked by: login:<id>` / `auth:test-user` / `no password for the test account in this run`; **warns** on a 429, an unconfirmed account, every exposed table denying the signed-in user, or an inconclusive protected-path answer (the root is walled or unreadable too) |
| `auth-recovery` | the recorded account's recovery request is accepted for sending, an address with no account gets the same answer (no account enumeration), the token this run spent is refused on replay, the new password signs in and the replaced one is refused, and the token window is named from `otpExpirySeconds` when reported | `auth.recovery` off; guided auth; `blocked by: login:<id>` / `auth:test-user`; no rotation in this run (`this run holds none of what the recovery check needs`); a captcha blocks a scripted request; **warns** on a 429 for either request or a login leg, never fails |
| `auth-isolation` | two accounts golive seeded and recorded sign in, both declared routes refuse an anonymous request, each account's identity route answers with its own id (never the other's), and each account's rows route returns its own marker row and none of the other's | `auth.isolation` off; no `auth.identityPath`/`auth.isolationPath` declared; guided auth; `blocked by: login:<id>` / `auth:test-user` / `auth:isolation` / `no password for … in this run`; production URL not confirmed; a route answers 404 or refuses the session token (the app-code task is named); a route does not accept the marker write; a 429 from the provider or the app. **Fails critical** on an anonymous 200, a crossed id or another account's marker; **warns** while an account is unconfirmed, on an inconclusive status, or when nothing in the answer is attributable |
| `webhook-unsigned` | an unsigned POST gets 4xx from the handler (a non-HTML 401/403 only warns — ambiguous between a rejection and an auth wall) | production URL not confirmed |
| `webhook-registered` | an enabled endpoint for the production URL covers the configured events | guided payments; no production URL |
| `stripe-live-ready` | the account has `charges_enabled` | production isn't live mode |
| `email-dns` | the provider's listed records (or common locations) and DMARC are in public DNS | no sending domain |
| `email-verified` | the provider marks the domain verified **and** the records it lists for that domain resolve in public DNS (a provider that cannot list them, lists none, or a lookup that failed, warns or skips — never a pass) | guided email; `blocked by: email:domain`; the provider exposes no record list, or lists none golive can resolve |
| `preview-deploy` | the hosting provider's own read confirms the preview deployment golive recorded (`deployed:preview:id`) is ready, belongs to the project this repo links and is not the production deployment | no recorded preview deployment; the recording belongs to another provider; a guided or logged-out host; a host with no per-deployment preview read (Vercel). **Warns** when the host reports a different preview deployment than the recorded one; **fails** when the recorded "preview" is the production deployment |
| `preview-bundle` | the HTML/JavaScript served by the provider-confirmed preview URL is scanned completely and holds no known credential patterns | no provider-confirmed preview URL; **skips** a 401/403 protection wall (a private preview is normal and is never a pass); **warns** on an incomplete scan or a page that did not load; **fails critical** on a leaked pattern |
| `production-release` | the provider's own read of what production serves is the deployment golive promoted or rolled back to (`deployed:release`), with what production served before named. Runs while `release.promote`/`release.rollback` is set, and afterwards for as long as a release is recorded (the opt-in can be removed and the evidence stays readable) | no recorded release; a guided or logged-out host; a host with no read of what production serves (Vercel); the provider reports no production deployment; another provider's recording. **Warns** when the provider read fails, or when production serves a deployment golive never recorded (a dashboard/Git/PR-built one — a handoff for the human); **fails** when production serves another deployment golive recorded (something moved production after the release) |

Details that trip people up:
- `env-parity` doesn't require `STRIPE_WEBHOOK_SECRET` (or other webhook-secret names) outside
  production: the webhook is registered and its secret written for production only. It doesn't
  require site-URL vars for a target with no stable URL (Vercel preview; production before the first
  deploy without a domain). These show as `not required` evidence lines. If the app needs e.g.
  `NEXT_PUBLIC_SITE_URL` in preview, the human sets it.
- `domain-live` warns "not propagated yet" (instead of failing) only right after a `domain:dns` step.
- `email-dns` checks exactly the records the provider lists once a sending domain id is recorded; a
  missing provider DKIM record fails. A missing DMARC record only warns and suggests one; golive does
  not write DMARC. Without that list it uses each provider's usual layout: a missing SPF fails only
  for Resend's `send.<domain>`; Postmark / SES DKIM selectors can't be found over DNS, so not finding
  one is a low warning (confirm DKIM in the provider dashboard). Details in `guided.md`.
- `email-verified` corroborates the flag instead of trusting it: it resolves exactly the records the
  provider lists for the domain and **fails** when they are gone — a zone cleaned up, moved between
  accounts or restored from a backup leaves the domain reading `verified` while nothing in DNS
  carries its SPF/DKIM ([#52](https://github.com/mikehasa/golive-skill/issues/52)). It **warns** when
  a record golive wrote is still inside the 48 h propagation window (a cached answer or a zone
  wildcard can answer first) or when a lookup failed; it **skips** when the provider exposes no record
  list, lists none, or the read failed. A warn or a skip is never a pass. The email link keeps its DNS
  work for such a domain too: the `email:dns` step (or the blocking handoff when golive cannot write
  DNS) stays planned, and the step's intent carries the unresolved records, so `apply` writes them
  again rather than skipping a step it recorded done when they matched.

## 5. Status (drift): what changed behind golive's back

`status --json` is the only command that asks whether the world still matches what golive **recorded**
— which is why no check can answer it: a check's report is release evidence for this invocation, while
drift needs the recorded side (state resources and step evidence, or a marker the provider assigned).
Run it once the app is live, before a release and after a run that changed providers or settings. It
writes nothing: no report, no state change, no provider write, and exit `2` means at least one item has
an `action` other than `none`.

Each item pairs `expected (recorded by golive <time>)` with `observed (read now)` and says who can act:
`verify` (re-run the named `checkId`), `reconcile` (an approved `plan` → `apply` restores it; DNS steps
still need `--confirm-dns`) or `human` (only the human decides, e.g. the credential now reads a
different account, or a project cannot be read at all). Severity: `high` = the app is broken (a record
deleted, an endpoint gone, a domain detached, a project unreadable); `medium` = hygiene or teardown
safety, or a change that may be deliberate; `info` = nothing to act on. Subjects: DNS records golive
wrote (checked against the zone, and against public DNS only while the zone still matches the baseline
— a record written inside the propagation window may legitimately differ publicly and is `info`), public
name-server delegation, golive-managed env **names** (never values: hosts hide sensitive values and
golive stores fingerprints, so a rotated value is outside the comparison), the recorded webhook endpoint
(gone, disabled, or missing events; a replacement at the same URL names the now-stale signing secret),
the domain attachment plus the records the host now requires, the db project and its
branch/database/role selectors, the sending domain, issued sending keys (no provider read exists, so
they are reported as unverifiable rather than checked off), the payment account and mode behind the
app's keys (a 403 is unverifiable, never drift), the host project identity and its creation marker, and
unfinished release state (a production env write no deploy picked up, a failed step).

A failed step is compared with the plan this release would run now, because that decides what an
operator can do: when the recorded step belongs to another (or an unknown) release and declares
neither `destroy` nor `risk.replayable`, `apply` refuses to replay the write, so the item is `human`
and points at the reviewed reconciliation path in [flow.md → Approval and resume boundary](flow.md#approval-and-resume-boundary) instead of an impossible
`apply --plan <planId>`. A step this release recorded, or one that declares the exemption, keeps the
plain re-run advice.

A provider that cannot be read yields `unverifiable: true` with `action: 'none'` and appears in
`notChecked`: never drift, and never "clean". `verified` lists the subjects read and found unchanged —
the only thing a "nothing changed" statement may cover; `limits` names what this comparison can never
see. Drift is never a gate: `plan` and `apply` do not consult it, and nothing is re-baselined except by
an approved write.
