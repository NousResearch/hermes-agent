---
name: cursor-cloud-agents
description: Launch and manage Cursor Cloud Agents from Hermes.
version: 1.0.0
author: Finn763
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Cursor, Cloud-Agents, Delegation, GitHub, API]
    related_skills: [claude-code, codex, hermes-agent]
---

# Cursor Cloud Agents Skill

Delegate repository work to [Cursor Cloud Agents](https://cursor.com/docs/cloud-agent/api/endpoints) — ephemeral coding agents that run in Cursor-hosted VMs against a GitHub repository. This skill covers the public Cloud Agents HTTP API only: launching agents, sending follow-ups, polling status, cancelling runs, and listing models. It does not drive the local `cursor-agent` CLI and it never opens a shell inside the agent VM.

## When to Use

- Long-running repo work that should not occupy the Hermes host (large refactors, dependency migrations, test-suite repair).
- Several independent workstreams at once — each agent gets its own VM, its own branch, and its own PR.
- Work that must outlive the current session: the agent keeps running after Hermes exits.
- Delegating to a model that only exists inside Cursor's runtime (`composer-*`).

Not for: quick edits to the local checkout (use `patch`), interactive pair-programming (that is the local CLI), or repos outside GitHub.

## Prerequisites

- **API key:** create one at https://cursor.com/dashboard (Settings → API Keys) and export it as `CURSOR_API_KEY`. In a gateway/service install put it in the profile `.env`; never paste the key into a prompt, a commit, or a file in the repo.
- **GitHub App:** Cursor's GitHub App must be installed for the target repository, otherwise the launch fails with `400`. Confirm with the `repos` command.
- **Python 3.10+** on the host running Hermes. The helper uses the standard library only — nothing to install.
- **Host is fixed.** The helper always talks to `https://api.cursor.com`; there is no host override by design, so the key cannot be redirected to a third party.

## How to Run

Copy the script path once, then drive it through `terminal`:

```bash
CCA=~/.hermes/skills/autonomous-ai-agents/cursor-cloud-agents/scripts/cursor_cloud.py
```

```bash
python "$CCA" models
```

Every command prints tab-separated `label<TAB>value` lines, so results are readable and easy to parse.

## Quick Reference

| Task | Command |
|------|---------|
| Check the key | `python "$CCA" me` |
| List models | `python "$CCA" models` |
| List visible repos | `python "$CCA" repos` |
| Launch an agent | `python "$CCA" launch --prompt "<task>" --repo owner/name` |
| Follow up | `python "$CCA" followup <agent-id> --prompt "<task>"` |
| Latest status | `python "$CCA" status <agent-id>` |
| Poll to completion | `python "$CCA" watch <agent-id> --run <run-id>` |
| Cancel | `python "$CCA" cancel <agent-id> --run <run-id>` |
| List agents | `python "$CCA" list --limit 20` |

| Flag | Effect |
|------|--------|
| `--repo` | `owner/name`, an HTTPS URL, or an SSH URL (normalised to `https://github.com/...`) |
| `--ref` | Starting branch or commit SHA; requires `--repo` |
| `--model` | Model id returned by `models`; omit to use the account default |
| `--mode agent\|plan` | `plan` explores before coding, `agent` implements directly |
| `--work-on-current-branch` | Commit to the starting ref instead of a new `cursor/...` branch |
| `--auto-create-pr` | Have Cursor open a pull request when the run finishes |
| `--interval` / `--timeout` | Poll cadence and give-up window for `watch` |

Endpoint shapes and status codes: `references/api.md` in this skill.

## Procedure

1. **Verify credentials and model.** Run the `models` command; a `401` means the key is wrong or unset, and the printed ids are what `--model` accepts.
2. **Confirm repo access.** Run the `repos` command when unsure. It is rate-limited to one request per minute, so cache the answer instead of calling it in a loop.
3. **Launch.** Pass the task text verbatim in `--prompt`, the repo in `--repo`, and a base branch in `--ref`. Add `--auto-create-pr` when you want a PR, `--model` for a specific model, and `--mode plan` when you want a plan before code.
4. **Record both ids.** The launch output carries a durable `agent` id and a per-run `run` id. Follow-ups and cancellation need the agent id; polling needs both.
5. **Follow the run.** Run `watch` in the background for long tasks and read output with `process(action="poll")`:

   ```bash
   python "$CCA" watch <agent-id> --run <run-id> --interval 30 --timeout 3600
   ```

   `watch` exits `0` at the first terminal status and `1` on timeout, printing the final `result`, branches, and PR URLs.
6. **Report.** Quote the `result` text and the PR URL back to the user. Do not paraphrase a `FINISHED` result as a success until the PR or branch listing confirms it.
7. **Iterate or stop.** Send another `followup` to continue the same conversation, or `cancel` the active run. Review the diff in the PR before merging anything the agent produced.

## Pitfalls

- **One active run per agent.** A follow-up while a run is in flight returns `409`. Wait for a terminal status, or cancel first.
- **Branching default.** Without `--work-on-current-branch`, commits land on a new `cursor/...` branch. That is usually what you want; only opt in when the task is explicitly meant to push to the base branch.
- **`EXPIRED` is not success.** A sandbox that gets reclaimed terminates the run as `EXPIRED` with no result. Treat only `FINISHED` as a completed run.
- **`--ref` without `--repo`** is rejected locally; the API cannot infer a base branch.
- **`envVars` are off-limits for secrets you care about.** They are session-scoped, names cannot start with `CURSOR_`, and the field is beta (silently ignored on some accounts). Keep long-lived credentials in the environment that runs Hermes.
- **Streaming endpoints are not used here.** `watch` polls REST instead of the SSE stream, which avoids `410 stream_expired` after the retention window and needs no reconnect bookkeeping.
- **Never echo the key.** Error messages from the helper carry the HTTP status and response body only.

## Verification

- `python "$CCA" models` prints at least one model id — proves the key and the network path.
- `python "$CCA" launch --prompt "Add a CONTRIBUTING.md" --repo owner/name` prints `agent`, `run`, and `status` lines — proves request construction against the live API.
- `python "$CCA" status <agent-id>` shows the latest run's `status`, and after completion the `result`, `branch`, and `pr` lines.
- Offline contract tests (no API key, no network): `scripts/run_tests.sh tests/skills/test_cursor_cloud_agents_skill.py -q` — pins the payload shape, the Basic auth header, the fixed host, and the polling loop.
