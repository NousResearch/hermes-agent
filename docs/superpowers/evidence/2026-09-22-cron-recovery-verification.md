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

## Review round 1: cleanup script restoration

Review found that `~/.hermes/scripts/worktree-cleanup.py` was byte-identical to
`library-vault-catalog.py`, only 31 lines long, and ignored
`HERMES_WORKTREE_CLEANUP_SCOPE`. The incorrect files shared SHA-256
`01d8bddff999da39ebb84b3be365ffdca9e12ed7a6dd0061b1d6cf97872ea058`.

The cleanup script was restored from the preserved pre-rebind implementation at
`~/.hermes/path-repair-backups/20260910-pre-lunabot-rebind/scripts/worktree-cleanup.py`.
That 337-line implementation retains the 48-hour activity buffer, dirty-worktree and
open-file guards, captured-work verification, bounded removal count, and per-scope logs.
Its sole path repair changed the retired Hermes checkout to the canonical
`/Users/mikedemott/Hermes-agent` checkout. The restored live script has SHA-256
`76d71a9c765d1de71109656010736c3accbe05f1d853f4f45640f8b8b24aa7d4` and is no
longer identical to the library catalog script.

Validation receipts:

- Python byte-compilation passed.
- A non-executing scope probe resolved `lunabot` to `/Users/mikedemott/LunaBot` and
  `worktree-cleanup-lunabot-log.jsonl`.
- A non-executing scope probe resolved `hermes-agent` to
  `/Users/mikedemott/Hermes-agent` and `worktree-cleanup-hermes-agent-log.jsonl`.
- The first LunaBot direct attempt, run `f013e2f3c7444213a6c1a461d9ac14c3`, failed on a
  transient read-only SQLite open at `08:29:35`. A retry proceeded through the full guarded
  scan and completed.

Authoritative successful replacement runs:

| Job | Durable run | Result | Started (America/Los_Angeles) |
|---|---|---|---|
| LunaBot worktree cleanup `a0f646ee7160` | `42a440f356a3442580a5a7a4d2427307` | completed | `2026-09-22T08:30:43.706584-07:00` |
| Hermes-agent worktree cleanup `35f6826b9d32` | `9b1d7b31072c4036bd9e95b4a4c15be5` | completed | `2026-09-22T08:29:43.688481-07:00` |
| Incremental library Vault catalogue `2047820efb33` | `c75e41d2db3e4b64a2f2544d2d08339b` | completed | `2026-09-22T08:30:00.901175-07:00` |

The LunaBot cleanup inspected 380 registered worktrees, removed 51 worktrees whose guards
passed, freed 23,489,996,527 bytes, skipped 40 uncaptured worktrees, and skipped 271 for
other safety reasons. The Hermes cleanup inspected 29 registered worktrees, removed one
guard-approved worktree, freed 4,027,248,008 bytes, and skipped 14 for safety reasons.

The follow-up `hermes cron doctor` dropped from 14 issues across 11 jobs to 9 issues across
9 jobs. It no longer reports failed last runs for either cleanup job or the library catalog;
the remaining cleanup findings are timing warnings. No incident acknowledgement command was
run.

## Review round 2: configured-worktree protection and restoration

The successful LunaBot cleanup receipt exposed a critical policy gap: the script protected
worktree-name patterns but did not protect stable checkout paths consumed by Hermes config.
It therefore removed these two configured clean worktrees after judging their commits
reachable from a remote branch:

- `/Users/mikedemott/.codex/lunabot-support/worktrees/hermes-conversation-base`, configured
  as both `conversation_worktree.source_worktree` and the LunaBot GitHub-feedback
  `local_path`; the cleanup receipt reported branch `hermes/conversation-base`, reason
  `HEAD reachable from a remote branch`, and `1,938,149,555` bytes.
- `/Users/mikedemott/.hermes/github-pr-feedback/post-merge-deploy-worktree`, configured as
  the LunaBot merge maintainer's `deployment_path`; the receipt reported detached HEAD,
  reason `HEAD reachable from a remote branch`, and `540,443,987` bytes.

The cleanup script now reads the active Hermes `config.yaml` with the standard library and
fails closed if it cannot read it. Absolute `source_worktree`, `local_path`, and
`deployment_path` values are normalized through `realpath` and rejected by `is_protected`
before candidate evaluation. The installed script and the verified candidate have identical
SHA-256 `c0d9abb255743c1032a42d6fb624c472e037e372d0821a401593a07b6889b616`;
Python compilation passed. Installed runtime probes returned `True` for both restored paths.

The script's GitHub repository is also scope-specific now:

| Cleanup scope | Main repository checkout | GitHub repository passed to `load_pr_states` |
|---|---|---|
| `lunabot` | `/Users/mikedemott/LunaBot` | `mrkillbob/luna-bot` |
| `hermes-agent` | `/Users/mikedemott/Hermes-agent` | `mrkillbob/hermes-agent` |

Non-executing `main()` probes replaced the destructive and external functions and captured
the exact `load_pr_states` argument. They returned `['mrkillbob/luna-bot']` for the LunaBot
scope and `['mrkillbob/hermes-agent']` for the Hermes Agent scope.

The conversation checkout was restored without changing or deleting any other worktree:

- registered path: `/Users/mikedemott/.codex/lunabot-support/worktrees/hermes-conversation-base`
- branch: `refs/heads/hermes/conversation-base`
- HEAD: `009b7859622ea867bee802e7fbdb0e75113a809e`
- origin: `https://github.com/mrkillbob/luna-bot.git`
- status: clean
- runtime link: `.venv -> /Users/mikedemott/LunaBot/.venv`, Python `3.13.6`

The deployment checkout was restored from its configured repository and base branch as a
detached registered worktree:

- registered path: `/Users/mikedemott/.hermes/github-pr-feedback/post-merge-deploy-worktree`
- configured repository/base: `mrkillbob/luna-bot`, `stable`
- source ref: local `refs/remotes/origin/stable`
- HEAD: `ebf2aed27d4a4574f73e871be657651199a93dc0`
- origin: `https://github.com/mrkillbob/luna-bot.git`
- status: clean and detached, matching the prior deployment-worktree shape

After both restorations, `hermes github-pr-feedback doctor` no longer returned
`invalid_configuration`. It parsed the policy and returned only the known external identity
blocker:

```json
{"checks":{"assignee":"ok","board":"ok","gh_executable":"ok","github_identity":"failed","hermes_executable":"ok","ledger_access":"ok","repository_worktree":"ok","worker_completion_policy":"ok"},"status":"degraded"}
```

Supported direct reruns reached the governed GitHub client and failed only at its live
authentication probe:

| Job | Durable run | Result | Started (America/Los_Angeles) | Terminal cause |
|---|---|---|---|---|
| LunaBot PR feedback `e3753541e2fa` | `0da5cc32646b4617928d8a1b3c930f84` | failed | `2026-09-22T08:48:40.972691-07:00` | `GitHubClientError: GitHub command failed (authentication)` |
| Hermes-agent PR feedback `def3474fce41` | `4aa217ba624146dcbb5710e8049ccf43` | failed | `2026-09-22T08:48:53.362320-07:00` | `GitHubClientError: GitHub command failed (authentication)` |

The final cron doctor still reported 9 issues across 9 jobs: these two authentication
failures plus timing warnings. The newest incidents for both PR jobs are `detected` auth
incidents. No incident acknowledgement command was run, and no secret was read or changed.
