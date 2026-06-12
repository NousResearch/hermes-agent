# Simplicio Runtime Final Report

Run: `run-1781282786814-66145`
Status: `completed_skeleton`
Repo: `/Users/wesleysimplicio/Projetos/ai/hermes-agent`
Task: `fix issue #45039: Dashboard "Default project directory" setting is ignored (even Dashboard sessions)`

## Decision

- Mode: `run`
- Profile: `research`
- Kind: `edit`
- Model: `remote escalation allowed with event log`
- Logical agents: `20`
- Model workers: `4`
- Selected packs: `repo-intelligence, tdd-verification`
- Capability decisions: `capability-selection.json`
- Task state: `task-state.json` when sprint mode is used
- PR handoff: `pr-handoff.md` when sprint/PR evidence is present

## Resume State

- Resume status: `not resumed; state.json can be used by simplicio resume`
- State file: `state.json`
- Resume plan: `resume-plan.json` when requested
- Cache and evidence are reused unless repo fingerprint or unsafe task review invalidates them.

## Cost And Cache

```json
{"schema":"simplicio.cost-ledger/v1","local_tokens":154,"remote_prompt_tokens":0,"remote_completion_tokens":0,"cache_hits":0,"prompts_reused":0,"commands_avoided":0,"estimated_paid_tokens_saved":0,"client":"","model":"openai/gpt-5-mini","task_link":"fix issue #45039: Dashboard \"Default project directory\" setting is ignored (even Dashboard sessions)"}
```

## Agent Lifecycle

```json
{"schema":"simplicio.agent-lifecycle-summary/v1","agents_source":"agent-store","store_status":"unavailable: agent store not initialized; run simplicio agent-store init --json","note":"no live agents","logical_agents":20,"active_agents":8,"queued_agents":12,"supervisor_queued_agents":0,"model_workers":4,"spawned_agents":0,"reused_agents":0,"killed_agents":0,"replaced_agents":0,"completed_agents":0,"idle_agents":0,"stuck_agents":0,"average_wait_ms":0,"idle_ttl_seconds":300,"stuck_timeout_seconds":900,"warm_agent_reuse":true,"resource_savings":"20 logical slots share 8 active agent(s) and 4 model worker(s)","states":["queued","starting","running","idle","stuck","killed","completed"],"events":["agent_spawned","agent_reused","agent_idle_checked","agent_throttled","agent_killed","agent_completed"],"agents":[]}
```

## Evidence

- `agent-lifecycle.json`
- `cost-ledger.json`
- `dev-cli-contract.json`
- `events.jsonl`
- `evidence/access-policy.json`
- `evidence/attachment-manifest.json`
- `evidence/curl-preflight.txt`
- `evidence/hermes-agent-benchmark.json`
- `evidence/hermes-agent-benchmark.md`
- `evidence/index.json`
- `evidence/index.md`
- `evidence/logs/events.snapshot.jsonl`
- `evidence/playwright/trace.zip`
- `evidence/pr-summary.md`
- `evidence/screenshots/screenshot-fixture.png`
- `evidence/test-summary.json`
- `final-report.md`
- `ledger.jsonl`
- `map-result.json`
- `prompt-envelope.json`
- `validation-plan.json`