# Pact Broker Webhooks (PactFlow → GitHub)

## Principle

Configure PactFlow webhooks to trigger provider verification in GitHub Actions via a dedicated GitHub machine user, a long-lived classic Personal Access Token (PAT), and a PactFlow-stored secret. Monitor for silent webhook failures so an expired/revoked token does not quietly block deployments for days.

## Rationale

### Why webhooks matter

- PactFlow's `contract_requiring_verification_published` webhook is the mechanism that notifies a provider repo (via `repository_dispatch`) that a consumer has published a contract needing verification.
- Without a working webhook, `can-i-deploy` in the consumer CI **times out** (900s) and eventually fails with `There is no verified pact between <consumer-version> and the version of <provider> currently in <env>` — even though nothing is wrong in either codebase.
- Webhook failures are **silent by default**: PactFlow keeps emitting requests, GitHub keeps returning `401 Unauthorized`, but nothing alerts the team until a PR is blocked.

### Why a dedicated GitHub machine user (not a personal PAT)

- Personal PATs die when the person leaves the company, rotates laptops, or revokes credentials during a security review. The contract test pipeline then breaks for reasons unrelated to any code change.
- A dedicated machine user (e.g., `pactflow-<org>`) is owned by the org, has only the repos it needs, and the PAT lifecycle is controlled by the security/platform team.
- GitHub **billing does not count** machine users added as outside collaborators to the specific repos they need — confirm with the org owner before assuming it's free.

### Why classic PAT with `repo` scope and no expiration

- PactFlow's webhook calls the GitHub REST API's `repository_dispatch` endpoint. This endpoint requires the **`repo` scope** on a classic PAT (fine-grained PATs work for many flows but have edge cases with `repository_dispatch` that are not universally supported at time of writing — verify with current GitHub docs).
- Classic PATs support "No expiration" — required to avoid the silent-failure trap every 90 days. GitHub warns against this for human users; for a locked-down machine-user PAT stored in PactFlow's secret vault, the security trade-off is documented and accepted.
- The alternative — rotating a PAT every 30/60/90 days — requires tooling and coordination most teams don't yet have. Long-lived + monitored + machine-user-owned is the pragmatic default.

## Pattern Examples

### Example 1: Webhook URL, Headers, and Body

```json
{
  "description": "Notify <provider-repo> when a consumer contract requires verification",
  "events": [{ "name": "contract_requiring_verification_published" }],
  "provider": { "name": "<provider-pacticipant-name>" },
  "request": {
    "method": "POST",
    "url": "https://api.github.com/repos/<org>/<provider-repo>/dispatches",
    "headers": {
      "Accept": "application/vnd.github+json",
      "Authorization": "Bearer ${user.githubToken}",
      "Content-Type": "application/json",
      "User-Agent": "PactFlow",
      "X-GitHub-Api-Version": "2022-11-28"
    },
    "body": {
      "event_type": "contract_requiring_verification_published",
      "client_payload": {
        "pact_url": "${pactbroker.pactUrl}",
        "sha": "${pactbroker.providerVersionNumber}",
        "branch": "${pactbroker.providerVersionBranch}",
        "consumer_name": "${pactbroker.consumerName}",
        "consumer_version_number": "${pactbroker.consumerVersionNumber}",
        "consumer_version_tags": "${pactbroker.consumerVersionTags}",
        "consumer_version_branch": "${pactbroker.consumerVersionBranch}"
      }
    }
  }
}
```

**Key Points**:

- `${user.githubToken}` references a PactFlow **secret** stored in `Settings → Secrets` (web UI: `/settings/secrets`). The secret holds the classic PAT — never inline the token in the webhook body.
- `${pactbroker.*}` are PactFlow-injected template variables; the provider workflow reads them from `github.event.client_payload`.
- Use the `contract_requiring_verification_published` event (not `contract_published`) — the former fires only when a new pact _content_ change needs verification; the latter fires on every publish, including no-op republishes.

### Example 2: Provider GitHub Actions Workflow (Triggered by Webhook)

```yaml
# .github/workflows/contract-test-provider.yml
name: contract-test-provider

on:
  repository_dispatch:
    types: [contract_requiring_verification_published]
  push:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    env:
      PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
      PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
      # Pulled from webhook client_payload when triggered by PactFlow:
      PACT_PAYLOAD_URL: ${{ github.event.client_payload.pact_url }}
      GITHUB_SHA: ${{ github.event.client_payload.sha || github.sha }}
      GITHUB_BRANCH: ${{ github.event.client_payload.branch || github.head_ref || github.ref_name }}
    steps:
      - uses: actions/checkout@v4
        with:
          # Check out the provider version known to the broker — this is the provider SHA PactFlow wants verified.
          ref: ${{ github.event.client_payload.sha || github.sha }}
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - name: Run provider verification
        run: npm run test:pact:provider
      - name: Can I deploy provider?
        if: github.event_name == 'push'
        run: npm run can:i:deploy:provider
```

**Key Points**:

- `repository_dispatch` is the event type emitted by GitHub when the webhook's REST call hits `/repos/<org>/<repo>/dispatches`.
- The `types` filter must match the webhook's `event_type` (`contract_requiring_verification_published` here).
- Checking out the provider version known to the broker (`providerVersionNumber`) ensures verification runs against the exact provider commit PactFlow registered — not whatever is on main.
- `PACT_PAYLOAD_URL` makes `buildVerifierOptions` verify only the triggering pact (see `pactjs-utils-provider-verifier.md` Example 1).

### Example 3: Secret Rotation Runbook

**Trigger**: `can-i-deploy` in a consumer repo times out with `There is no verified pact between <consumer-version> and the version of <provider> currently in <env>` — AND the provider's `contract-test-provider` workflow shows no recent `repository_dispatch` runs.

**Diagnosis**:

1. In PactFlow UI: `Settings → Webhooks → <webhook-id> → Test`. A `401 Unauthorized` from GitHub confirms the token is dead.
2. In PactFlow UI: the webhook's "Last executed at" is hours/days stale while consumer pacts are actively being published.

**Rotation**:

1. Log in to GitHub as the dedicated machine user (e.g., `pactflow-<org>`). **Do not use a personal account** — the whole point of the machine user is that the token outlives any individual.
2. `Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token (classic)`.
3. Configure the token:
   - Name: `pactflow-webhook-<yyyy-mm-dd>`
   - Expiration: **No expiration** (accepted trade-off for a locked-down machine-user token stored in PactFlow's secret vault)
   - Scopes: **`repo`** (full repo scope is required by `repository_dispatch`; `public_repo` alone is insufficient for private repos)
4. Copy the new token value (shown only once).
5. In PactFlow UI: `Settings → Secrets → <secret-name>` (e.g., `githubToken`). Paste the new token into the **value** field and save. The webhook does not need to be edited — it references the secret by name via `${user.<secret-name>}`.
6. Re-test the webhook: `Settings → Webhooks → <webhook-id> → Test`. Expect `HTTP/1.1 204 No Content` (GitHub's success response for `repository_dispatch`).
7. In the provider repo: watch `Actions → contract-test-provider` for the newly dispatched run. Re-run the original consumer CI to confirm `can-i-deploy` now passes.
8. Revoke the old token: in the machine user's GitHub settings, delete the previous `pactflow-webhook-*` token so a leaked copy can't be reused.

**Why no expiration**: A token with a 90-day expiry rotates 4× per year. Each rotation is a silent-failure window if the runbook isn't executed proactively. With monitoring (Example 4) + a locked-down machine-user-owned PAT that is only stored in PactFlow, long-lived is safer than short-lived-but-forgotten.

### Example 4: Staleness Monitoring (Detect Silent Webhook Failures)

**Goal**: Alert the team if verification results haven't been published for a pacticipant pair in the last N hours, so an expired PAT or network issue doesn't silently block `can-i-deploy` for days.

Pick one of these (in increasing order of investment):

**Option A — Daily sanity CI job (cheapest)**:

```yaml
# .github/workflows/pact-staleness-check.yml
name: pact-staleness-check
on:
  schedule:
    - cron: '0 9 * * 1-5' # weekdays 09:00 UTC
  workflow_dispatch:
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - name: Fail if latest verification for <pair> is older than 24h
        env:
          PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
        run: |
          # Query broker matrix for newest verification timestamp for consumer/provider pair.
          # Exit 1 if > 24h old; team gets an email on the failed scheduled run.
          ./scripts/assert-recent-verification.sh <consumer> <provider> 86400
```

**Option B — PactFlow metrics endpoint**: Use the SmartBear MCP `get_metrics` / `get_team_metrics` tool (see `pact-mcp.md`) to surface verification freshness in a dashboard or Slack digest.

**Option C — Webhook delivery log**: PactFlow logs every webhook execution. Ship those logs to your SIEM / observability stack and alert on sustained 4xx responses from `api.github.com`.

**Key Points**:

- The point is not "which option you pick" — it's that **you pick at least one**. Without monitoring, the first time you learn the webhook is dead is when a release is blocked.
- Alert threshold should match your consumer-publish cadence: if consumers publish daily, alert after 24–48h of silence; if hourly, after 3–6h.
- Keep the alert noise-free: page only on sustained staleness, not a single missed run.

## Key Points

- **Dedicated machine user owns the PAT** — never a personal PAT. Name it `pactflow-<org>` or similar; give it outside-collaborator access only to the specific provider repos.
- **Classic PAT, `repo` scope, no expiration** — required for `repository_dispatch`. The "no expiration" trade-off is accepted in exchange for machine-user ownership + PactFlow-secret storage + staleness monitoring.
- **Store the PAT as a PactFlow secret** at `/settings/secrets`, reference it from the webhook via `${user.<secret-name>}`. Never inline the token.
- **Monitor for silence** — at minimum, a daily scheduled CI job that asserts a recent verification timestamp exists for each critical consumer/provider pair.
- **Rotation is a runbook, not an emergency** — document it (see Example 3), keep it in the repo, and do a practice rotation once a year so it stays fresh.
- **Symptom to remember**: "consumer `can-i-deploy` timeout after 900s with `There is no verified pact...`" + "provider's `contract-test-provider` workflow has no recent runs" = expired/revoked PAT. Start with Example 3.

## Related Fragments

- `pactjs-utils-provider-verifier.md` — how `PACT_PAYLOAD_URL` from the webhook's `client_payload.pact_url` is consumed by `buildVerifierOptions`
- `pact-consumer-framework-setup.md` — consumer CI flow that issues `can-i-deploy` and silently times out when the webhook is dead
- `pact-mcp.md` — SmartBear MCP tools (`Matrix`, `Metrics - All`) useful for staleness monitoring dashboards
- `contract-testing.md` — foundational CDC patterns and resilience coverage

## Anti-Patterns

### Wrong: Using a human's personal PAT

```
# ❌ PactFlow secret githubToken stores the lead engineer's personal classic PAT
# When they leave / rotate / revoke → all provider verifications stop silently
```

### Right: Dedicated machine user owns the PAT

```
# ✅ Machine user `pactflow-<org>` generates the PAT; secret is owned by the org
# PAT lifecycle is decoupled from any individual's employment or laptop state
```

### Wrong: No staleness monitoring

```
# ❌ No scheduled check for verification recency
# First signal that the webhook is dead: a blocked release PR, several days later
```

### Right: Daily scheduled sanity check

```
# ✅ Scheduled workflow fails if latest verification > 24h old
# Team gets email alert on failed scheduled run → rotate PAT before anyone is blocked
```

### Wrong: Short-expiration PAT with no rotation tooling

```
# ❌ 90-day expiry PAT, no calendar reminder, no runbook
# Breaks every 90 days for a day or two until someone notices
```

### Right: No-expiration PAT on machine user + monitoring + documented runbook

```
# ✅ Long-lived PAT, scoped narrowly, stored in PactFlow, monitored for staleness
# Rotation is intentional (security review, suspected leak) not calendar-driven
```

_Source: PactFlow webhook documentation, GitHub `repository_dispatch` REST API, seon-mcp-server / seon-admin-panel production incident April 2026_
