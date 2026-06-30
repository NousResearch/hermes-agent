# Upstream-delta behavior audit (Phase 0b / INV-9)

**Scope:** `c6b0eb4..929dd9c0d` (1,734 commits, the frozen parity target). Code-surface scan
(NOT commit-message grep — pass-2 fix). Purpose: know what inherited behavior the merge ships
unreviewed, and flag any AGENTS.md violation for Ace's knowing-deploy decision.

## Findings

### 1. New core tools on the every-call schema — **NONE** ✅
- `toolsets.py` changed (+52/-24, 74 lines) but added **zero new `"name":` tool entries** — the
  changes are modifications to existing tool wiring, not new always-on model tools.
- `acp_adapter/tools.py`: 2-line change (not a new tool).
- Base vs frozen `toolsets.py` top-level tool-name count: 1 → 1 (unchanged).
- **Verdict:** the project's sacred narrow-waist constraint is not widened by this merge. No new
  per-API-call cost/cache/alternation surface.

### 2. New outbound calls / telemetry / analytics — **NONE of concern** ✅
- 113 added lines match outbound-call patterns (`requests`/`httpx`/`urllib`/`socket`), but **zero**
  match telemetry/analytics SDKs (`telemetry|analytics|track|segment|posthog|mixpanel|amplitude|
  sentry|datadog`).
- **Zero** hardcoded `http(s)://` destination hosts added in the delta — the outbound calls are to
  already-configured provider/API endpoints, not new exfil/analytics sinks.
- **Verdict:** no un-gated telemetry inherited. AGENTS.md "outbound telemetry without opt-in" rule
  not tripped.

### 3. New `HERMES_*` env vars for non-secret config — **internal plumbing only** ✅
Added env reads are session/path/runtime plumbing, all internal bridge vars (AGENTS.md permits
bridging to an internal env var; the rule bans *user-facing* `HERMES_*` config knobs):
`HERMES_HOME`, `HERMES_SESSION_ID`, `HERMES_SESSION_SOURCE`, `HERMES_SESSION_KEY`, `HERMES_PLATFORM`,
`HERMES_MACHINE_ID`, `HERMES_MAX_ITERATIONS`, `HERMES_WRITE_SAFE_ROOT`, `HERMES_VERIFY_ON_STOP`,
`HERMES_PET_IMAGE_PROVIDER`, `HERMES_PYTHON_SRC_ROOT`, `HERMES_NOUS_MIN_KEY_TTL_SECONDS`.
- These are runtime/session-context bridges (the kind AGENTS.md explicitly allows), not new
  user-facing behavioral config that should live in `config.yaml`. One notable comment in the delta
  even shows upstream *deliberately NOT* writing `HERMES_SESSION_KEY` — they're being careful here.
- **Verdict:** no AGENTS.md violation. None are credentials, none are user-facing config knobs.

### 4. Config-version migration — **29 → 32** ⚠️ (handled by Phase 5 gate)
- Upstream bumped `_config_version` 29 → 32 (3 migrations). NOT a violation, but the load-bearing
  reason Phase 5 (config-migration dry-run against real fleet configs) exists. Tracked there.

### 5. Major dependency bumps — to enumerate in Task 11
- Lockfile delta to be reviewed when regenerating in Task 11 (INV-9 runtime-compat note). Resolution
  ≠ runtime-compat; any major bump (pydantic etc.) gets watched in host-1 soak.

## Net deploy posture
**No AGENTS.md violation inherited that needs an Ace stop-and-decide.** The two real items —
config migration (29→32) and major dep bumps — are already gated by Phase 5 + Task 11 + host-1
soak. The merge does not widen the core tool schema, add un-gated telemetry, or introduce
user-facing `HERMES_*` config. Safe to proceed through the build; the config migration is the one
thing to prove live before any host restart.
