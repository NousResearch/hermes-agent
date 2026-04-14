# Codex worker-pool isolation runbook

Purpose: keep Codex worker lanes deterministic, boring, and easy to verify.

## Canonical contract

- `codex-orchestrator-pro` = active orchestrator lane
- `codex-orchestrator-plus` = prepared backup orchestrator lane
- `codex-worker-pro` + `codex-worker-plus-1..11` = active coding worker pool
- `codex-worker` and `codex-executioner` are retired legacy lanes; do not use them for login, dispatch, or normalization
- worker model standard = `openai-codex/gpt-5.3-codex`
- each active lane should resolve to exactly one desired profile id:
  - `openai-codex:codex-orchestrator-pro`
  - `openai-codex:codex-orchestrator-plus`
  - `openai-codex:codex-worker-pro`
  - `openai-codex:codex-worker-plus-N`

Manifest: `ops/openclaw/codex_worker_pool_manifest.json`
Authoritative lane mapping: `ops/openclaw/codex_lane_account_map.json`
Reconcile script: `ops/openclaw/codex_lane_reconcile.py`
Lane map setter: `ops/openclaw/codex_lane_set_account.py`
Drift watchdog: `ops/openclaw/codex_drift_watchdog.py`

If two active lanes intentionally share one account, mark both lanes with
`"allowSharedAccount": true` in `codex_lane_account_map.json` so dispatch health
does not raise a false `shared_account_binding` quarantine.

## Audit current state

Quick preflight before any new login or lane change:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_preflight_audit.py
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_drift_watchdog.py
```

Detailed worker-pool audit:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_worker_pool_audit.py
```

## Mandatory pre-dispatch gate (fail-closed)

Before assigning work to a worker lane, run a target-scoped dispatch gate and require a healthy verdict.

Target-scoped preflight (preferred):

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_preflight_audit.py --agent <worker-agent> --lookback-hours 24
```

Direct dispatch-health gate (equivalent fail-closed check):

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_dispatch_health.py \
  --lookback-hours 24 \
  --include-orchestrator \
  --require-healthy-agent <worker-agent>
```

Fleet-wide strict audit (optional, broader than target dispatch):

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_preflight_audit.py --agent <worker-agent> --all-lanes
```

## Preferred login path (alias-aware wrapper)

For all remaining worker-pool logins, prefer the alias-aware wrapper. It gives us the practical equivalent of upstream `--profile-alias` without patching the installed OpenClaw package.

Example:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_login_alias.py --agent codex-worker-plus-2
```

This will:
1. run the interactive OpenClaw Codex OAuth login for that agent,
2. normalize the landed credential to `openai-codex:<agent>`,
3. run authoritative lane reconciliation against `codex_lane_account_map.json`,
4. pin auth order to the stable lane alias,
5. verify the result + usage report.

If the wrapper reports **Global-only OAuth landing detected**, stop retrying that lane blindly. That means OpenClaw only updated `~/.openclaw/openclaw.json` metadata and did not write a usable lane-local credential.

If the store is multi-profile / ambiguous, provide an explicit selector for an active lane:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_login_alias.py --agent codex-worker-plus-11 --alias codex-worker-plus-11 --from-email grinedibattista26@outlook.com
```

## Normalize an agent after login

Single-profile store:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_auth_normalize.py codex-worker-plus-2
```

Multi-profile store, explicit source profile id:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_auth_normalize.py codex-worker-plus-10 --from-profile-id openai-codex:keartetzlaff50@outlook.com --profile-id openai-codex:codex-worker-plus-10
```

Multi-profile store, explicit email:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_auth_normalize.py codex-worker-plus-11 --from-email grinedibattista26@outlook.com --profile-id openai-codex:codex-worker-plus-11
```

Order-only repin when the desired profile id already exists:

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_auth_normalize.py codex-worker-plus-1 --profile-id openai-codex:codex-worker-plus-1 --set-order-only
```

## Required verification after every login

```bash
python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/codex_auth_inspect.py <agent>
openclaw models auth order get --agent <agent> --provider openai-codex --json
```

Stop if the landed email/account is wrong.

## Safe rollout order

1. `codex-worker-pro`
2. `codex-worker-plus-1..11`
3. `codex-orchestrator-plus`
4. `codex-orchestrator-pro` last

Reason: orchestrator relogin is the most disruptive live lane.
