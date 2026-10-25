---
name: stripe-link-cli
description: Agent payments via Stripe Link — cards, SPT, approvals.
version: 0.2.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Payments, Stripe, Link, Checkout, MPP]
    related_skills: [mpp-agent, stripe-projects]
---

# Stripe Link CLI Skill

Wraps [@stripe/link-cli](https://github.com/stripe/link-cli) so Hermes can complete purchases on the user's behalf using one-time-use virtual cards or Shared Payment Tokens (SPT). Every spend is gated by an in-app approval in the Link mobile/web app — Hermes cannot self-approve.

US-only at the moment (Link account requirement). Works on Linux, macOS, and Windows (verified with link-cli 0.27.3 on Windows 11, Node 26; see the Windows notes under Pitfalls).

## When to Use

Trigger phrases:

- "buy X", "pay for X", "make a purchase", "complete checkout"
- "get me a card", "I need a payment method"
- "log in to Link", "connect my Link wallet"
- HTTP 402 response from a merchant API with `www-authenticate: ... method="stripe"`

If the user wants a paid API call (HTTP 402, no checkout form), the `card` path is wrong — use SPT via this same skill, or hand off to the `mpp-agent` skill.

## Prerequisites

- Node.js 20+ available on `PATH` (`node --version`)
- US-based (Link account requirement)

The Link account, payment method, and spend-approval app do NOT need to be set up before Hermes attempts to pay — the CLI walks the user through them on first run:

- A Link account at https://app.link.com — created/linked during first `link-cli` auth
- At least one payment method — added during first run at https://app.link.com/wallet
- The Link mobile/web app — opened to approve the first spend request when it's made

No env vars required — auth state is stored locally by the CLI under its own config directory.

## Install

Install once, globally:

```
npm install -g @stripe/link-cli
```

Or invoke ad-hoc via `npx @stripe/link-cli`. The skill below uses the installed `link-cli` form.

npm may warn `install-scripts ... not yet covered by allowScripts` for `@stripe/link-cli`. That is harmless: the skipped postinstall only runs `npx skills add stripe/link-cli` to install a skill for other agents, so the CLI works without it.

## How to Run

All commands run through the `terminal` tool. The CLI auto-detects non-TTY callers and emits compact `toon` output by default — fine for the model. Pass `--format json` if a step needs structured fields.

Discover commands: `link-cli --llms-full`.
Get a command's schema before invoking: `link-cli <command> --schema`.

## Procedure

### 1. Check / establish auth

```
link-cli auth status
```

If not authenticated, log in with a clear client name (this label shows in the user's Link app):

```
link-cli auth login --client-name "Hermes" --interval 5 --timeout 300
```

The `--interval`/`--timeout` form polls inline so the agent doesn't need to manage a `_next` step. Run it as a background process, read the `verification_url` + `phrase` from its first output lines, give them to the user, and wait for the CLI to return.

**The device code expires after ~10 minutes regardless of `--timeout`.** When it expires, the CLI clears the pending login and keeps polling, then exits `0` with `authenticated: false` and no `pending:` field. That is not an error. Tell the user the link is good for about 10 minutes, and if it lapses start a fresh `auth login` and send the new URL.

**Do not proceed past this step until `auth status` confirms login.**

### 2. Evaluate the merchant before creating a spend request

Decide the credential type:

| Merchant surface | `--credential-type` |
|---|---|
| Standard web checkout form / Stripe Elements | `card` (default) |
| Returns HTTP 402 with `method="stripe"` in `www-authenticate` | `shared_payment_token` |
| Returns HTTP 402 without `method="stripe"` | unsupported — stop |

For 402 responses, do NOT decode the challenge manually. Pass the raw header:

```
link-cli mpp decode --challenge '<full WWW-Authenticate header>'
```

This validates the challenge and extracts the network ID + decoded request body.

### 3. List payment methods + shipping

```
link-cli payment-methods list
link-cli shipping-address list
```

Use the first entry unless the user specifies otherwise. The `id` from `payment-methods list` is the `--payment-method-id` in the next step.

### 4. Create the spend request

Confirm the final total with the user before issuing this command. Amounts are in cents.

```
link-cli spend-request create \
  --payment-method-id <pm_id> \
  --merchant-name "<name>" \
  --merchant-url "<url>" \
  --context "<what is being purchased, from whom, and why; at least 100 characters>" \
  --amount <cents> \
  --line-item "name:<item>,unit_amount:<cents>,quantity:1" \
  --total "type:total,display_text:Total,amount:<cents>" \
  --request-approval
```

For MPP merchants add `--credential-type shared_payment_token`.

`--context` must be at least 100 characters, because the user reads it on the approval screen. `--merchant-name` and `--merchant-url` are required for `card` requests. Itemize with repeated `--total` entries (`subtotal`, `shipping`, `tax`, `total`) so the approval screen matches the merchant's checkout.

With `--request-approval`, the CLI (0.27.x) creates the request and returns right away with `status: pending_approval`, an `approval_url`, and a `_next` command:

```
link-cli spend-request retrieve <lsrq_id> --interval 2 --max-attempts 300
```

Give the user the `approval_url`, then run `_next` (in the background if your harness allows) to wait until the status leaves `pending_approval`. **The push notification does not always appear in the Link mobile app.** If the user can't see the request, have them open the `approval_url` directly while signed in to Link. The approval window is about 30 minutes (`approval_window_expires_at`). `_next` gives up after its attempt budget (about 10 minutes), so re-run it if the window is still open.

### 5. Retrieve the credential — SECURELY

**Do not print card details to stdout.** Use `--output-file` so the PAN never enters the agent's transcript or logs:

```
link-cli spend-request retrieve <lsrq_id> \
  --include card \
  --output-file <scratch-dir>/link-card.json \
  --format json
```

Use an absolute path for `<scratch-dir>`: `~/.hermes/cache/scratch` on Linux/macOS, `C:/Users/<user>/AppData/Local/hermes/cache/scratch` on Windows (see the Windows pitfall below).

The file is written with `0600` perms; stdout shows only redacted fields (brand, last4, expiry) plus a `card_output_file` path.

### 6. Use the credential

- For web checkout: hand the file path to the user, OR pass it to a browser-driving tool that fills the form directly from disk. Never `read_file` or `cat` the card file into the agent's reasoning context.
- For MPP merchants:

  ```
  link-cli mpp pay <merchant-url> \
    --spend-request-id <lsrq_id> \
    --method POST \
    --data '<json body>'
  ```

### 7. Clean up

Delete the card file as soon as the purchase is done:

```
rm -f <scratch-dir>/link-card.json
```

## Optional: run as an MCP server instead

`@stripe/link-cli --mcp` exposes the same commands as MCP tools over stdio. To register it with Hermes' native MCP:

```
hermes mcp add stripe-link --command "npx" --args "@stripe/link-cli --mcp"
```

Then `hermes mcp list` should show `stripe-link`. The same approval rules apply — MCP doesn't bypass the Link app approval step.

## Pitfalls

- **US-only.** Outside the US, `auth login` will fail. Tell the user, don't keep retrying.
- **Card PAN must never enter agent context.** Use `--output-file` every time. If you've already retrieved without it, immediately `link-cli auth logout` is not enough — the card is one-time-use but rotate hygiene matters.
- **`--request-approval` does not block.** It returns at once with `status: pending_approval`; the wait happens in the `_next` retrieve poll (Section 4). Run that poll in the background, re-run it if it exhausts its ~10-minute attempt budget while the ~30-minute approval window is still open, and tell the user the request expires if they don't act.
- **Multi-step `_next` commands.** Some commands return `_next.command` that must be executed to continue. When in doubt, prefer the inline-polling flags (`--interval`/`--timeout`).
- **Windows: `--output-file` needs an absolute native path.** Under git-bash, POSIX paths are not translated for Node: `/tmp/...` resolves to `C:\tmp\...` and `~/...` expands to `/c/Users/...`, and both fail with `OUTPUT_FILE_WRITE_ERROR ... ENOENT`. Pass a forward-slash native path such as `C:/Users/<user>/AppData/Local/hermes/cache/scratch/link-card.json`.
- **Output format defaults to `toon`** in non-TTY mode. Fine for prose, but if a downstream step needs to parse a specific field, pass `--format json`.
- **Don't default to `card`.** The merchant-evaluation step (Section 2) exists because picking the wrong credential type fails the purchase silently or leaks more data than needed.
- **Shopify storefront checkout resists browser automation.** Filling a one-time card into a Shopify checkout with a driven browser hit bot protection: a headless browser's "Pay now" was silently ignored, and a headed one got a CAPTCHA plus a Shop Pay prompt. Never try to solve CAPTCHAs. For Shopify merchants, prefer the `shop` optional skill (Shop CLI, UCP checkout with Shop Pay), or hand the final click to the user.
- **`link-cli ucp ...` is account-gated.** `ucp catalog search` and `ucp checkout create` (even with `--test`) can return `403 Forbidden`, and `approval-policy retrieve` returns `403: This feature is not enabled on your account.` `auth upgrade` can't fix this, because the CLI defines no UCP scope (only `userinfo:read` and `payment_methods.agentic`). Report it to the user rather than retrying; access comes from Stripe (agent-spend@stripe.com).
- **Report blocked checkouts.** `link-cli report --domain <host> --outcome blocked --spend-request-id <lsrq_id> --tag captcha --attempt-trace "<numbered steps>"` sends Stripe a structured outcome report. Keep buyer PII out of the trace.

## Verification

```
link-cli --version && link-cli auth status
```

Exit code 0 means installed and logged in.
