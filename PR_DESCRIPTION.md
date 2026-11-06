# Add qbo-ops — QuickBooks Online operations plugin

## What the plugin does

qbo-ops connects a Hermes agent to QuickBooks Online with reconciliation
tradecraft built in — the judgment layer, not just API plumbing:

- **`qbo_health`** — connection health; doubles as the setup wizard
  (unconfigured state returns a guided walkthrough with exact URLs, never a
  raw error)
- **`qbo_pnl`** — P&L for a date range (verified against real October
  actuals during development)
- **`qbo_udf_audit`** — close audit flagging backdated entries
  (created-after-transaction-date) for review
- **`qbo_deposit_trace`** — deposit tracing with duplicate-deposit (wash)
  detection
- **`qbo_query`** — guarded read-only SQL against any QBO entity
  (SELECT-only, no statement stacking, maxresults clamped to 200 — enforced
  in code, not requested)

Plus a **desktop status pane** (connection chip + month summary) for
local-mode desktop installs — on remote-backend connections the agent
tools carry the full experience — and a **bundled skill**
(`qbo-ops-reconciliation`) carrying the interpretation layer: close-audit
reading, deposit-wash analysis, month-end sanity chain, and a failure
catalog.

## Why

The catalog had zero bookkeeping/QuickBooks entries despite QBO being the
default books of the small-business demographic that runs self-hosted
agents. Every figure in the plugin's development was verified against a
real production QBO company.

## Hermes surfaces used

- `ctx.register_tool` × 5 (agent tools)
- `ctx.register_skill` × 1 (bundled reconciliation skill)
- Desktop plugin SDK: `statusBar.right` chip + `panes` surface via the
  plugin's own dashboard backend (`dashboard/plugin_api.py`)

## Disclosure (rule 13)

- **Network calls:** to Intuit only — `quickbooks.api.intuit.com` (data)
  and `oauth.platform.intuit.com` (token refresh). No telemetry, no other
  endpoints.
- **Credentials:** flow exclusively through Hermes secret scope
  (`requires_env` names in the manifest). The plugin never reads `.env`
  files, auth.json, or any other token store directly.
- **Token refresh:** the BYO-refresh-token path refreshes Intuit tokens via
  the plugin's flock-guarded refresher (rotation-race safe). Disclosed
  tradeoff: pure `hermes auth add` OAuth needs upstream `redirect_port`
  support for Intuit's exact-match redirect URIs — proposed upstream in
  #149642; the environment-variable path is the current default and is
  fully functional.
- **Read-only by design:** no tool creates or modifies QBO data. A
  per-user `allow_writes` config switch (default **off**) reserves the
  config space for a future write tool; writes, when they ship, will also
  route through Hermes's approval surface.
- **No shell commands, no background processes, no reads outside the
  plugin's own data.**

## Validation

`hermes plugins validate . --install-deps` passes 16/16 checks at the
pinned commit (one `caution` warning, disclosed above: the fallback token
refresher names Intuit's token URL — the disclosed tradeoff while
#149642 is pending). `hermes plugins doctor` loads the plugin with a
temporary HERMES_HOME and confirms 5 tool registrations.

Tests: 17 offline tests (`pytest tests/`) covering the token cache,
lock contention, P&L report walking (fixture captured from a real QBO
report), wizard paths, query guardrails, and the JSON-string handler
contract.

## Docs

- README: https://github.com/blackghostosint/qbo-ops#readme
- Complete beginner setup walkthrough (every click and URL):
  https://github.com/blackghostosint/qbo-ops/blob/main/docs/SETUP.md
