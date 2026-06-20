# AgentCyber Working Edition Plan

> For Hermes: use this as the implementation checklist to make the AgentCyber fork usable as a real Cyber Edition profile, not just a preserved patchset.

## Goal

Make `breakingcircuits1337/hermes-agentcyber` work end-to-end for authorized BC cyber operations: profile setup, cyber tool availability, sensitive-task local/open-weight routing, asset-scoped execution gates, live smoke tests, and a human break-glass workflow for high-impact actions.

## Current evidence

Checked on local repo `/home/kbun/Desktop/hermes-agentcyber`.

- Local `main` is clean and synced to `origin/main` at `d3acc91c1`.
- After fetching upstream, the fork is currently `17` commits behind `NousResearch/hermes-agent` and `43` commits ahead.
- CLI import/version smoke works:
  - `uv run --frozen hermes --version` reports Hermes Agent v0.17.0.
- Local model endpoints are reachable:
  - `http://127.0.0.1:11434/api/tags` responds.
  - `http://192.168.1.120:11434/api/tags` responds and includes `qwen3-coder:30b`.
- Current user config contains `agent_cyber.routing.enabled: true`, `require_local_for_sensitive: true`, local provider `ollama`, model `qwen3-coder:30b`, and a base URL.
- Existing cyber preservation tests recently pass:
  - `96 passed` for `tests/cyber`, routing guard, cyber routing, cyber audit hook, and route capture tests.
- Existing skill-index tests recently pass:
  - `8 passed` for skills index/diff checks.

## Main blockers / missing pieces

1. **Upstream drift must be handled before feature work.** The fork is already behind upstream again. Continue the guarded upstream-sync workflow before new runtime changes.
2. **No first-class AgentCyber setup/doctor path.** Defaults include `agent_cyber`, but there is no obvious `agentcyber setup/status/doctor` flow that validates local model, assets, cyber toolsets, and gateway hooks.
3. **Cyber toolsets are not exposed in `hermes tools list`.** `toolsets.py` defines `cyber` and `live_usb`, and tool modules register correctly, but `hermes tools list` did not show them for the CLI surface.
4. **Classifier has gaps.** Example found: `recover access to VM 112` currently routes as `general` instead of `ir_breakglass` or `credentials_sensitive`.
5. **Execution-gate tool-name coverage is incomplete.** Actual tool names include `extract_iocs` and `ir_incident`, while `agent/cyber_policy.py` still lists older names like `ioc_extractor` and `ir_playbook` in `_SECURITY_TOOLS`.
6. **Live local-routing smoke is missing.** Unit tests prove the switch path with stubs, but there is no live or hermetic integration test proving a sensitive request routes to Ollama/open-weight and does not egress to hosted models.
7. **S4/S5 break-glass approval is not implemented.** Destructive/high-impact actions are hard-blocked. That is safe, but it means incident recovery workflows cannot complete when a legitimate approved action crosses S4/S5.
8. **AgentCyber docs are not yet a runnable operator guide.** `docs/CYBER_EDITION.md` documents surfaces, but needs a short “do this on a fresh machine/profile” runbook and acceptance checks.

## Implementation lanes

### Lane 0: Guarded upstream sync baseline

Objective: Start from a clean, current fork before changing AgentCyber runtime behavior.

Steps:
1. Create a new guarded upstream-sync branch from clean `main`.
2. Merge `upstream/main` with `--no-commit`.
3. Audit deletion drift:
   - `git diff --cached --name-status --diff-filter=D`
   - cyber deletion guard for `agent/cyber*`, `docs/CYBER*`, `gateway/builtin_hooks/cyber*`, `live-usb/`, `skills/cybersecurity/`, `tests/cyber/`, `tools/cyber*`, and `HERMES_CYBER_EDITION_TODO.md`.
4. Resolve conflicts preserving Cyber Edition files.
5. Run focused tests and commit sync.

Verification:
```bash
uv run --frozen python -m pytest tests/cyber tests/agent/test_agentcyber_routing_guard.py tests/agent/test_cyber_routing.py tests/gateway/test_cyber_audit_hook.py tests/run_agent/test_cyber_route_capture.py -q -o addopts= --tb=short
git diff --check
```

### Lane 1: Add an AgentCyber doctor/status command

Objective: Make setup failures visible before an operator trusts the edition.

Preferred shape:
- Add `hermes agentcyber status` or `hermes security agentcyber status` if that fits current CLI command conventions better.
- Keep it read-only.

Checks to report:
- `agent_cyber.routing.enabled`.
- `require_local_for_sensitive`.
- local/open-weight provider, model, base URL presence without printing secrets.
- live local model endpoint health.
- configured asset registry source and asset count.
- cyber/live_usb toolset availability and whether enabled for current platform.
- gateway cyber audit hook presence.
- upstream drift count.

Tests:
- Add hermetic CLI tests with temp `HERMES_HOME`.
- Mock local model health; do not require live Ollama in CI.

### Lane 2: Expose and enable Cyber toolsets properly

Objective: Make `cyber` and `live_usb` selectable/visible like other built-in toolsets.

Steps:
1. Find the CLI/tool configuration source that feeds `hermes tools list`.
2. Add display metadata for `cyber` and `live_usb` without enabling them globally by default.
3. Add tests that `hermes tools list` includes them and that `hermes tools enable cyber` persists correctly.
4. Add docs showing `hermes tools enable cyber` and optional `live_usb` enablement.

Verification:
```bash
uv run --frozen hermes tools list
uv run --frozen python -m pytest tests/hermes_cli/test_tools_config.py tests/tools/test_registry.py -q -o addopts= --tb=short
```

### Lane 3: Fix route classification coverage

Objective: Route common BC operator language into the correct Cyber Edition lanes.

Immediate failing examples:
- `recover access to VM 112` should not be `general`.
- `restore VM112 access` should route to `ir_breakglass`.
- `use qwen locally for malware triage` should preserve local preference.

Steps:
1. Add failing tests in `tests/agent/test_cyber_routing.py`.
2. Extend `_BREAKGLASS_TERMS`, `_CREDENTIAL_TERMS`, and lab host detection in `agent/cyber_routing.py`.
3. Keep general release/coding tasks on `general`.

Verification:
```bash
uv run --frozen python -m pytest tests/agent/test_cyber_routing.py tests/agent/test_agentcyber_routing_guard.py -q -o addopts= --tb=short
```

### Lane 4: Fix execution-gate tool-name coverage

Objective: Ensure every registered cyber tool is classified intentionally.

Findings:
- Registered cyber tools: `threat_intel`, `extract_iocs`, `vuln_triage`, `ir_incident`, `network_scan`.
- Current security-tool list includes stale names: `ioc_extractor`, `ir_playbook`.

Steps:
1. Add tests for all registered cyber tool names.
2. Decide intended gates:
   - `extract_iocs`: S1 read-only local analysis.
   - `threat_intel`: S1 read-only unless active query target rules require S2.
   - `vuln_triage`: S1 read-only.
   - `ir_incident`: S3 local incident-store mutation.
   - `network_scan`: S2 scoped recon.
3. Update `agent/cyber_policy.py` names and reasons.
4. Keep local-only incident mutations allowed while blocking external risky actions.

Verification:
```bash
uv run --frozen python -m pytest tests/agent/test_agentcyber_routing_guard.py tests/cyber -q -o addopts= --tb=short
```

### Lane 5: Add live/local-routing smoke

Objective: Prove sensitive work uses local/open-weight runtime and fails closed when unavailable.

Steps:
1. Add a hermetic fake OpenAI-compatible local server test for CI.
2. Add an optional live smoke script for BC lab:
   - checks `http://192.168.1.120:11434/api/tags`;
   - confirms `qwen3-coder:30b` exists;
   - runs a minimal classification/routing action without printing secrets.
3. Add `hermes agentcyber status --json` output so automation can assert health.

Verification:
```bash
uv run --frozen python -m pytest tests/agent/test_agentcyber_routing_guard.py -q -o addopts= --tb=short
python scripts/agentcyber_live_smoke.py --base-url http://192.168.1.120:11434 --model qwen3-coder:30b
```

### Lane 6: Implement S4/S5 human approval / break-glass

Objective: Keep destructive actions blocked by default, but provide an auditable operator-approved path for owned assets.

Design constraints:
- No autonomous S5 execution.
- Approval must be explicit, scoped, expiring, and logged.
- Approval must include asset match, command/action fingerprint, operator, timestamp, and reason.
- Never print secrets in approval records.

Steps:
1. Define approval token format and storage location under `HERMES_HOME`.
2. Add tests that S5 remains blocked without a valid token.
3. Add tests that a valid token allows only the exact scoped action.
4. Wire approval metadata into gateway audit hook.
5. Add CLI command to create/inspect/revoke break-glass approvals.

Verification:
```bash
uv run --frozen python -m pytest tests/agent/test_agentcyber_routing_guard.py tests/gateway/test_cyber_audit_hook.py -q -o addopts= --tb=short
```

### Lane 7: Operator docs and acceptance checklist

Objective: Make it obvious how to run AgentCyber from a fresh profile.

Docs to update:
- `docs/CYBER_EDITION.md`
- `README.md` quickstart section
- possibly website docs if the fork publishes them

Must include:
- Create/use an `agentcyber` profile.
- Enable `cyber` toolset.
- Configure local/open-weight runtime.
- Configure asset registry.
- Run doctor/status.
- Run a safe smoke command.
- Explain S4/S5 approval boundaries.

Acceptance checklist:
```bash
uv run --frozen hermes --version
uv run --frozen hermes tools list
uv run --frozen hermes agentcyber status --json
uv run --frozen python -m pytest tests/agent/test_cyber_routing.py tests/agent/test_agentcyber_routing_guard.py tests/cyber -q -o addopts= --tb=short
```

## Recommended order

1. Lane 0: guarded upstream sync.
2. Lane 3 and Lane 4: fix correctness gaps found during review.
3. Lane 1 and Lane 2: make the edition discoverable/configurable.
4. Lane 5: prove live local-routing health.
5. Lane 6: add break-glass approval.
6. Lane 7: write the operator runbook.

## Definition of done

AgentCyber “works” when:

- `hermes agentcyber status` is green on the BC machine/profile.
- `cyber` tools are visible and can be enabled.
- Sensitive cyber tasks route to local/open-weight runtime or fail closed.
- Asset-scoped S2/S3 actions are allowed only for approved assets.
- S5 actions are blocked unless a scoped human break-glass approval exists.
- Local smoke and CI tests pass.
- README/docs tell a new operator how to run it without guessing.
