# Cron failure recovery verification — 2026-09-22

## Verdict

**Partial acceptance.** Eight non-GitHub jobs have new durable `completed` replacement runs from the repaired runtime. Both GitHub PR-feedback jobs remain blocked by the dedicated `mrkillbobbot` credential: the plugin policy parses, but its live viewer probe returns `authentication`. No cron incident was acknowledged.

The safe default multiplex gateway is active from exact source commit `c330fbe02e0ebe3350402e20e0007e78d4564d59`. One child process owns both the gateway and embedded dispatcher locks. The restart completed only after the in-flight `lunabot-failure-scan` reached a durable `completed` state.

## Runtime identity and ownership

- `hermes --version-local` reported source `/Users/mikedemott/.codex/worktrees/e89e/Hermes-agent`, local commit `c330fbe0`, and the corresponding worktree runtime.
- `launchctl print gui/501/ai.hermes.gateway` reported one running LaunchAgent, supervisor PID `34154`, program `/Users/mikedemott/.codex/worktrees/e89e/Hermes-agent/.venv/bin/python`, and `runs = 2` after the graceful restart.
- The sole gateway child was PID `34155`.
- `lsof /Users/mikedemott/.hermes/gateway.lock /Users/mikedemott/.hermes/kanban/.dispatcher.lock` showed PID `34155` as the only owner of both locks.
- Fresh startup logs at `08:17:00` recorded `kanban dispatcher: holding singleton dispatcher lock`, `Cron scheduler will tick 98 profile(s) under multiplex`, and `kanban dispatcher: embedded in gateway`.
- `launchctl print-disabled gui/501` showed `ai.hermes.gateway => enabled`, `ai.hermes.gateway-task-intake-router => disabled`, and the fleet supervisor disabled. The removed task-orchestrator service was absent.

## Required replacement runs

Commands were run through the supported `hermes cron run <job-id>` path. Durable receipts were read back with `hermes cron runs <job-id>`.

| Job | Durable run | Source | Result | Started (America/Los_Angeles) |
|---|---|---|---|---|
| Lunar City asset generation | `504bfb514e104846a9ad77ae8f2c7419` | direct | completed | `2026-09-22T08:02:18.280320-07:00` |
| Lunar City media render | `414129a1b1fb47e1a2b127886fbd3e5d` | direct | completed | `2026-09-22T08:02:30.075717-07:00` |
| LunaBot research cycle | `a08a944af52a49a2857dbb1548b450ce` | direct | completed | `2026-09-22T08:02:46.245206-07:00` |
| Research experiment execution | `e91a62c5dfea46bea23068cdfb092314` | direct | completed | `2026-09-22T08:03:04.111970-07:00` |
| Research promotion gate | `a2055bc98b1a4e3d835039aa315a786b` | direct | completed | `2026-09-22T08:03:14.105576-07:00` |
| Federation resource budget rebalance | `23c162ac1acf49d6970acca0cda5cef0` | direct | completed | `2026-09-22T08:03:23.908080-07:00` |
| R&D adversarial fuzz | `01b893844c4d4a44b3a13616b312ec7a` | direct | completed | `2026-09-22T08:03:36.965440-07:00` |
| R&D dependency stress | `b736cb5c8e7749b7a065368a62c365da` | direct | completed | `2026-09-22T08:03:48.951782-07:00` |

## GitHub PR-feedback blocker

The plugin discovery/configuration defects were repaired before the live probes:

- `~/.hermes/plugins/github-pr-feedback` now links to the plugin in the active exact-source worktree.
- Both Hermes repository entries use the existing canonical checkout `/Users/mikedemott/Hermes-agent`.
- The Luna repository entry uses the existing governed checkout `/Users/mikedemott/.codex/lunabot-support/worktrees/hermes-conversation-base`.
- `not_before` is the required string `2026-08-25T06:03:40+00:00`.
- The policy requests `expected_login: mrkillbobbot` through the dedicated `HERMES_GITHUB_BOT_TOKEN` boundary.

`hermes github-pr-feedback doctor` then returned:

```json
{"checks":{"assignee":"ok","board":"ok","gh_executable":"ok","github_identity":"failed","hermes_executable":"ok","ledger_access":"ok","repository_worktree":"ok","worker_completion_policy":"ok"},"status":"degraded"}
```

The latest authoritative job receipts are failures:

- LunaBot PR feedback `e3753541e2fa`: run `9bfccc2611d5415792f31d72a6cad954`, `failed`, `2026-09-22T08:00:48.237007-07:00`.
- Hermes-agent PR feedback `def3474fce41`: run `cf31a8d27656439ea9a3c7c6b5a0c35e`, `failed`, `2026-09-22T08:17:04.289844-07:00`.

Both terminate in the governed client at `gh api user` with `GitHubClientError: GitHub command failed (authentication)`. Core `hermes doctor` only proved that the dedicated token and login fields are present; it did not validate the token against GitHub. A new fine-grained token owned by `mrkillbobbot` must replace the rejected credential before either scan is rerun. No secret was read, copied, printed, or moved into the human `mrkillbob` GitHub identity.

## Stale deleted-worktree paths found by final doctor

The first post-recovery `hermes cron doctor` found six definitions whose workdir still named the deleted `hermes-update-20260902-bot-integration-sparse` checkout. Supported `hermes cron edit <id> --workdir ...` commands moved only those definitions to the active exact-source worktree:

- `482c250c9086` Hermes profile and workflow health
- `4754bb0bd539` Hermes White Knight issue intake
- `5143a736175c` Research agent capability feedstock
- `9ef5ab8c34ff` Federation department discovery and librarian intake
- `35f6826b9d32` Hermes-agent worktree cleanup
- `649fe8ed77dc` cron-health-monitor

Three referenced helper scripts were repaired narrowly:

- `~/.hermes/scripts/federation-discovery.sh`
- `~/.hermes/scripts/library-vault-catalog.py`
- `~/.hermes/scripts/worktree-cleanup.py`

Their Hermes source and interpreter paths now resolve in `/Users/mikedemott/.codex/worktrees/e89e/Hermes-agent`; `bash -n` and Python byte-compilation passed. The final doctor no longer reports missing workdirs. It still reports the prior failed rows for the cleanup/catalog jobs until replacement scheduled executions supersede them, along with late/catch-up warnings.

## Worker capacity

The live configuration includes `ollama-launch/devstral-small-2:24b: 1`. A runtime construction of `WorkerCapacity` against that configuration returned:

```text
{'devstral_local_limit': 1, 'openai_cloud_limit': 10, 'openrouter_cloud_limit': 4}
```

Devstral therefore remains capped at one local worker. Cloud model caps remain independently configured above one and are not reduced by the local-model guard.

## Fresh error audit and incident state

For gateway log lines from `2026-09-22 08:00:00` onward, a case-insensitive search found no fresh `malformed database`, retired-WAL, missing-plugin, provider-auth, or duplicate-ticker signatures. The separate durable PR-run stderr contains the GitHub credential authentication failure documented above. Startup also rejects duplicate Discord/Photon profile credentials; those are credential ownership guards, not duplicate ticker owners.

The final `hermes cron doctor` reported 14 issues across 11 jobs. The live issues comprise:

- both PR-feedback jobs failing the dedicated GitHub identity probe;
- historical last-run failures for cleanup/catalog jobs whose paths are now repaired but not yet superseded by a new scheduled run;
- late and catch-up timing warnings.

`hermes cron incidents list` still shows detected and resolved history. **No `hermes cron incidents ack` command was run.** Even incidents backed by successful non-GitHub replacement runs were left unacknowledged because the two required PR scans have not succeeded.

## Remaining acceptance boundary

Task 6 cannot be fully accepted until a valid dedicated `mrkillbobbot` fine-grained token is installed through the Hermes secret boundary and both PR-feedback jobs produce new durable `completed` runs. After that, rerun `hermes github-pr-feedback doctor` and `hermes cron doctor`, then decide which resolved incidents are obsolete from the new receipts.
