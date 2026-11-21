# Guided providers

Use this when the human picked a provider golive doesn't automate yet (`menu` shows `automated: false`,
or a provider that isn't listed at all). Help execute a documented path where possible, then guide
the human through the parts that need them. This is best-effort assistance, not a promise that an
adapter exists or that this provider can take the app live.

Keep providers the app already uses. Check framework/runtime compatibility before suggesting a new
one; changing hosting does not migrate the app's database or Auth. Use the menu's exact provider id
when present. Otherwise choose a descriptive lowercase letters/digits/hyphens id, for example
`init --stack hosting=example-host,db=supabase --json`. Replace `example-host` with the provider's
id, not `other`. This only records the choice; no new capability is installed.

`doctor` may return `ok: false` / exit code 2 for guided providers because no adapter exists.
Do not repeatedly request credentials to fix that. The `accounts` check covers automated providers
only; its passing result does not prove a guided provider's login or account ownership.

## How to guide

1. **Find a supported path in current official documentation.** Check the specific operation,
   required permissions, account/project scope, price and verification method. Link the relevant
   documentation. Do not guess command flags, API endpoints or dashboard controls.
2. **Choose tools by capability.** Prefer the official CLI when it supports the operation and a
   usable account login. An available official MCP or the official API can fill a gap, subject to
   the secret rules below. No MCP installation is required; do not mechanically try every tool.
   If automation lacks the needed capability, guide the provider's dashboard flow. Use a direct
   provider integration when it safely connects the two services without exposing credentials.
3. **Establish identity before writes.** Use read-only, non-secret observations to identify the
   selected account/team, project and any existing resources. The human completes signup, login,
   purchases and identity checks. Interactive logins, prompts and pickers need a **separate terminal
   window**: the Hermes `terminal` tool has no interactive TTY. Verify the installed CLI is on PATH, then resume the
   same stage. Never use a `--token` / `--key` flag: argv exposes the secret.
4. **Get approval for the concrete external operations.** Add existing account/team/project IDs
   and proposed new resource names/settings, intended changes, permissions, costs and verification
   steps to `docs/GOLIVE-<stage>-PLAN.md`, and summarize
   them in chat before requesting approval. The CLI `planId` approves only its listed steps; it does
   not cover separate CLI/MCP/API/dashboard writes. State which operations are outside `apply`.
   Resolve unknown scope or cost first; never automate purchases. Live payments and exact DNS
   changes need explicit category approval. A changed destination or operation needs a new plan.
5. **Execute or guide one bounded step at a time.** After approval, perform supported operations
   with safe tool outputs. For human steps, give the exact action and expected result, then wait
   for confirmation. If an operation times out, inspect its remote outcome before retrying; do not
   create duplicates. Follow `troubleshooting.md` for a concrete repair and resume the same stage.
6. **Verify and record what the evidence establishes.** Run applicable CLI checks as described
   below. For external tooling, confirm the exact project/resource and record the observation,
   source, time and any newly assigned resource IDs in `docs/GOLIVE-<stage>-RESULT.md`, without
   credentials. An HTTP 200 alone does not
   prove the intended app or workflow works. Do not actively probe a project URL without confirming
   its ownership; do not bypass the CLI's confirmed-origin guard to make a skipped check run.
7. **Stop when there is no safe documented path.** If the required capability is unavailable,
   secret output cannot be avoided, permissions or cost remain unresolved, or a failed operation
   cannot be reconciled, state what blocks this step and the next human action. Preserve progress;
   do not invent a command, repeatedly try speculative alternatives or claim deployment succeeded.

## Keep secrets outside the Hermes context

- Before calling any tool, check whether its output can contain credentials. **Do not call MCP
  endpoints that return secret keys, connection strings or one-time credentials into the Hermes
  context.** Redacting after receiving a result is too late. Prefer metadata-only operations.
- Official API calls are acceptable only through a reviewed local path that loads credentials
  privately, transports them through stdin or HTTPS headers/body, and returns only allowlisted
  non-secret metadata. Apply the same rule to CLI output and errors. Never put secrets in argv,
  chat, logs, screenshots or a file Hermes will read. If safe output cannot be guaranteed, use
  a human dashboard handoff instead.
- Prefer provider-to-provider integrations. When a value must be entered manually, the human
  enters it **directly into the destination dashboard** using their own browser. Do not inspect,
  capture or ask them to relay it. Resume Hermes inspection only after secret fields are closed.

**DNS records:** use the exact records supplied by the selected host/email provider, or by
`plan` / `handoff`. Approve their zone, type, name and content before editing. Follow the provider's
proxy guidance; merge SPF into the single existing `v=spf1` record instead of adding a second one.

## Keep CLI checks and external evidence distinct

After each piece, run `verify --only <check-id> --json` with an applicable id below (or from SKILL.md).
This creates a partial report for that invocation, not a cumulative acceptance report. A CLI check
counts as passed only when it actually **passes**; a `skip` is not a pass.

Record external evidence separately as **verified by Hermes; not verified by the golive CLI**,
naming the tool or observation, scope and limitations. Human confirmation alone is
**human-confirmed; not independently verified**. Never rewrite `.golive/state.json`, generated
reports or handoff status to turn these into CLI passes. Run full CLI verification at the end and
summarize both sets of evidence without merging their claims.

`handoff --json` can retain a generic `guided:<axis>` item as `done: false` because it has no closing
check. Manual items and skipped checks stay `done: null`, including guided Auth settings or a guided
host's environment variables. Explain those open/unverified items even if separate observations
were recorded; do not claim the CLI has closed them.

**Guided host with automated Stripe.** golive can't store the webhook signing secret in a guided
host, and Stripe reveals it once, so a blocking `stripe:webhook-guided` handoff (closed by
`webhook-registered`) tells the human to create the endpoint in the Stripe dashboard and copy its
`whsec_…` straight into the host's **Production** env. The `env:<target>` handoffs are per target:
no webhook secret outside production, and Stripe keys are labelled with that target's mode (test in
preview, live in production by default).

## What golive can still check

| Check id | Works with a guided provider? |
|---|---|
| `accounts` | Covers only automated providers; guided entries are not authenticated or verified by this check. |
| `domain-live` | Partly: public DNS-over-HTTPS plus an HTTPS request. With a guided host (or no host) its evidence says the attachment is **not confirmed**: a pass doesn't prove the domain is attached to that host. |
| `email-dns` | Yes: SPF/DKIM/DMARC looked up in public DNS at the usual locations for the provider (see below). |
| `bundle-secrets`, `webhook-unsigned` | Only when the **hosting** provider is automated: they probe only the production URL the host confirms belongs to the project, so with a guided host they skip. |
| `rls-probe` | Only with `db=supabase`. |
| `db-connection` | Only with the implemented Neon database capability; not a generic guided DB test. |
| `env-parity`, `auth-redirects`, `auth-policy`, `auth-signup`, `auth-session`, `webhook-registered`, `email-verified` | No: they need the provider's API, so they skip for a guided provider on that axis. |

Every skipped check is something golive did not verify. Name them in your final summary.

**`email-dns` for guided email providers** (no record list from the provider API, so it uses each
provider's usual layout):
- **Postmark:** no SPF required; a `pm-bounces.<domain>` CNAME is accepted as the custom return path
  and is optional (a note). Its DKIM selector (`<timestamp>pm`) can't be found over DNS.
- **SES:** no SPF required (the default MAIL FROM is amazonses.com; a custom MAIL FROM isn't checked).
  Its Easy DKIM tokens can't be found over DNS.
- **SendGrid:** SPF is inferred from its automated-security `s1._domainkey` CNAME into sendgrid.net,
  otherwise a low warning.
- **Other providers:** a missing SPF warns (medium). Only Resend's `send.<domain>` layout fails on a
  missing SPF.
- For Postmark and SES a DKIM record not found is a **low warning**, not a failure: ask the human to
  confirm DKIM shows verified in the provider's dashboard, and name it as unverified by golive.
