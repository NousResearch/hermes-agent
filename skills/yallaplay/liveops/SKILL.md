---
name: liveops
description: Use for live account state, backend config, support tickets, pushes, ranking/game-engine, clienttwin, CockroachDB, and production operational reads.
version: 1.1.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [liveops, support, config, clienttwin, ranking, onesignal, helpshift, yallaplay]
    related_skills: [analytics, collaboration, knowledge]
---

# LiveOps Skill

## Overview

Use this skill for current operational state: individual accounts, support tickets, backend config, runtime flags, pushes, ranking/game-engine records, and incident triage. Default to read-only workflows. Any user-visible or production write needs exact confirmation and a documented gated command.

This is the Hermes equivalent of the legacy `personas/liveops.md` workflow, adapted to the Hermes pilot where many legacy scripts are not yet ported.

## When to Use

- Specific-user debugging, account state, purchase/support investigation.
- Helpshift tickets, OneSignal pushes, backend config, global KV, ranking, game-engine records, clienttwin state.
- Live incident triage involving config, operational logs, support signals, or live DB state.
- Questions that combine support/account state with warehouse metrics; load Analytics too.

## Required Context

Inspect relevant wiki pages before non-trivial tool use:

- `yallaplay-wiki/operations/index.md`
- `yallaplay-wiki/operations/liveops_runbooks.md`
- `yallaplay-wiki/operations/customer_support.md`
- `yallaplay-wiki/tools/product/helpshift.md`
- `yallaplay-wiki/tools/product/onesignal.md`
- `yallaplay-wiki/tools/product/app_insights.md`
- Product feature pages under `yallaplay-wiki/products/**/facts.md`, `engineering.md`, and `findings.md`.

Use `python3 tools/wiki_search.py <terms> --domain liveops` to route into the wiki.

## Identity and Joins

- Warehouse user id = `EVENTS.INDEX_USER.USER_ID` = Helpshift `meta_user_id` = twin/ranking user id.
- Cockroach `index_user.id` maps to profile display name and handle in the legacy workflow.
- Named teammates belong to Collaboration; resolve Slack identities with the Slack lookup flow before messaging or attributing.
- For user/account IDs, preserve exact strings. Do not "fix" an ID without evidence.

## Tool Routing

Legacy commands from `yallaplay-analytics-agent-gpt/scripts/` are the reference for what still needs porting. In this Hermes pilot, use ported wrappers when available and otherwise say the wrapper is not yet ported instead of inventing output.

Read-only targets from the legacy repo:

- Cockroach/live DB: `scripts/cockroach_query.py` for current account state not in Snowflake.
- Clienttwin: `scripts/twin_client.py` for current player state/devices.
- Config: `scripts/config_fetch.py`, `scripts/config_history.py`, `scripts/dump_config.py` for read-only config snapshots/history.
- Global KV: `scripts/global_kv.py get/list` for runtime flags; writes are gated.
- Helpshift: `scripts/helpshift_fetch.py` and local sync/mirror workflows for support tickets.
- OneSignal: `scripts/onesignal_client.py user/view/list` for read-only push state.
- Ranking/game-engine: `scripts/ranking_client.py` read paths and `scripts/game_json.py` for canonical per-match records.
- Observability: App Insights and Embrace tools belong at the LiveOps/Infrastructure/Analytics boundary; load those skills too when metrics or backend telemetry are part of the answer.

## Workflow

1. **Scope.** Identify app, environment, user/account identifiers, time window, and whether the action is read or write. Completion: ambiguity that changes tools is resolved.
2. **Load context.** Open relevant wiki/tool docs and prior findings before live calls. Completion: source-specific caveats are known.
3. **Prefer read-only/local mirrors.** Use local mirrors or read-only APIs before live calls. Completion: least-invasive source was tried first.
4. **Separate observation from inference.** Report exact facts observed, then possible root cause. Completion: user can see what was measured vs inferred.
5. **Gate writes.** For any visible/live mutation, draft the exact command/body/target and wait for explicit confirmation. Completion: confirmation names the exact side effect.
6. **Capture durable learning.** Incidents, config gotchas, and reusable investigations go to `yallaplay-wiki/operations/`, `tools/product/`, or product findings. Completion: future Claudio can find it.

## Write Safety

Require explicit confirmation before:

- Helpshift replies.
- OneSignal sends/deletes.
- Slack posts/uploads/reactions/pins.
- Jira or Google Workspace writes.
- Backend config, global KV, clienttwin, ranking, game-engine, or live account mutations.

Never write production config directly in this pilot. Dev config writes require the documented dev-only gated flow and `--yes` confirmation.

## Common Pitfalls

1. **Treating support data as analytics truth.** Helpshift is sampled by user behavior; use it for concrete cases and signals, not population rates.
2. **Conflating live and warehouse lag.** Warehouse facts can lag; current purchases/account state may need live DB or support sources.
3. **Guessing app/environment.** Spades vs Rummy and prod vs dev materially change tools and risk.
4. **Silent user-visible writes.** Draft first; post/send/reply only after confirmation.
5. **Over-broad incident actions.** Refuse broad deletes, pushes, config wipes, or mass user impact without a narrow approved plan.

## Verification Checklist

- [ ] App, environment, identifiers, and time window are explicit.
- [ ] Read-only source was used unless a confirmed gated write was required.
- [ ] User-visible/production writes were not performed without exact confirmation.
- [ ] Facts are separated from inference.
- [ ] Durable incident/config/support knowledge was captured when reusable.
