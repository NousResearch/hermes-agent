# Hermes Local Subprocess Wrapper Foundation Design

Date: 2026-05-17
Status: Approved for implementation planning after user review
Scope: A-slice only, local foundation, no activation

## 1. Purpose

Build a local, test-backed foundation for the Hermes architecture pivot without activating new live authority. This slice prepares Hermes to support Claude Code and Codex CLI subprocess routing later, while preserving all CMH safety gates.

The design intentionally does not implement Telegram sends, gateway changes, launchd, cron, Cowork-headless dispatch, Codex auto-dispatch, routing-table activation, R109 fire, git merge, deploy, or production mutation.

## 2. Source inputs

This design is based on these current inputs:

- Cowork checklist pasted by Christopher: Hermes Architecture Pivot Implementation Checklist, created 2026-05-17T18:45-07:00.
- Canonical strategic plans in the vault:
  - `00-OS/Strategic-Plans/Hermes-Tier-Routing-Recommendation.md`
  - `00-OS/Strategic-Plans/Hermes-Escalation-Routing-Via-CLI-Subprocesses.md`
  - `00-OS/Strategic-Plans/Codex-Auto-Dispatch-From-Hermes-Architecture.md`
- Hermes Agent skill guidance for CLI subprocess wrappers and slash-command development.
- Local verification performed in this session.

Current dependency findings:

- `claude` is available at `/Users/Millson/.local/bin/claude`.
- `codex` is not on PATH in this shell.
- The expected Codex Wave 1.16.E verified flag docs are not present yet:
  - `/Users/Millson/Desktop/operate-report-service/docs/architecture/hermes-claude-print-invocation-verified.md`
  - `/Users/Millson/Desktop/operate-report-service/docs/architecture/hermes-codex-print-invocation-verified.md`
- Local Claude help showed `--max-budget-usd`, not draft flag `--max-cost-usd`.
- Local Claude help did not show draft flags `--max-turns` or `--no-mcp` in the captured output.

## 3. Design decision

Implement a local foundation inside the Hermes repo first. Do not place active wrappers under `~/.hermes/wrappers/` or `~/.hermes/bin/` in this slice.

Recommended module boundary:

- `agent/cmh_subprocess/flags.py`
  - Parse and validate verified CLI flag documents or captured help text.
  - Represent required and optional flags for Claude and Codex.
  - Return clear missing-flag diagnostics.

- `agent/cmh_subprocess/envelope.py`
  - Read and write envelope state in a profile-aware path under `get_hermes_home()/state/envelope_tracking.json`.
  - Track `anthropic_max` and `chatgpt_pro` windows.
  - Reset usage after a 5-hour rolling window.
  - Enforce configured allocation thresholds without invoking any external CLI.

- `agent/cmh_subprocess/halt_flags.py`
  - Read and write halt flags in a profile-aware path under `get_hermes_home()/state/cmh_halt_flags.json`.
  - Support class flags: `cowork_headless`, `codex_auto_dispatch`, `hermes_telegram_acks`, and `all`.
  - Fail closed on malformed state by returning halted for unsafe classes.

- `agent/cmh_subprocess/result.py`
  - Shared result types for wrapper preflight and eventual subprocess invocation.
  - Keep disabled, halted, budget-blocked, missing-binary, missing-flag, timeout, and success states distinct.

- `agent/cmh_subprocess/wrappers.py`
  - Provide disabled-by-default `prepare_claude_print_invocation()` and `prepare_codex_print_invocation()` helpers.
  - Perform binary checks, flag checks, halt checks, and envelope checks.
  - Return command argv and diagnostics but do not run model subprocesses by default.
  - Only support actual subprocess execution behind an explicit function parameter used by tests with harmless local commands or by later approved phases.

- `tests/agent/test_cmh_subprocess_*.py`
  - TDD coverage for flags, envelope state, halt flags, and wrapper preflight behavior.

This keeps the first implementation reversible, testable, and isolated from live CMH surfaces.

## 4. Behavior requirements

### 4.1 CLI flag verification

The flag verifier must support two evidence sources:

1. Verified docs from Codex Wave 1.16.E when present.
2. Local captured help text as a temporary evidence source for tests and diagnostics.

The verifier must not silently assume draft flags. It must explicitly report:

- `claude` present or missing.
- `codex` present or missing.
- Required flag present or missing.
- Draft or deprecated flag mismatch when a spec requests a flag not found in help output.

Claude initial required flag set for this slice:

- `--print`
- `--max-budget-usd`
- `--output-format`
- `--no-session-persistence`

Claude optional flags recognized if present:

- `--plugin-dir`
- `--model`
- `--permission-mode`
- `--tools`
- `--bare`

Codex required flags remain unresolved until Codex Wave 1.16.E verified docs exist or `codex --print --help` becomes available locally. The Codex wrapper must therefore return a disabled or missing-binary result in this slice.

### 4.2 Envelope tracking

Envelope state path:

`{get_hermes_home()}/state/envelope_tracking.json`

Initial schema:

```json
{
  "anthropic_max": {
    "envelope_total_messages_per_5h": 225,
    "envelope_allocation_hermes_pct": 85,
    "envelope_messages_used_5h": 0,
    "window_start_iso": null,
    "last_invocation_iso": null,
    "halt_flag_active": false
  },
  "chatgpt_pro": {
    "envelope_total_messages_per_5h": 200,
    "envelope_allocation_hermes_pct": 85,
    "envelope_messages_used_5h": 0,
    "window_start_iso": null,
    "last_invocation_iso": null,
    "halt_flag_active": false
  }
}
```

Rules:

- If state file is missing, create defaults in memory and write only when the caller explicitly requests persistence.
- If `window_start_iso` is missing, start a new window on first increment.
- If current time is 5 hours or more after `window_start_iso`, reset `envelope_messages_used_5h` to zero and start a new window.
- Allocation cap is `floor(total * allocation_pct / 100)`.
- If usage is at or above cap, return budget-blocked for non-priority work.
- This slice does not detect Christopher interactive use automatically. It leaves that as a later integration point.

### 4.3 Halt flags

Halt flag state path:

`{get_hermes_home()}/state/cmh_halt_flags.json`

Initial schema:

```json
{
  "cowork_headless": false,
  "codex_auto_dispatch": false,
  "hermes_telegram_acks": false,
  "all": false
}
```

Rules:

- `all: true` halts every wrapper class.
- `cowork_headless: true` halts Claude/Cowork wrapper class.
- `codex_auto_dispatch: true` halts Codex wrapper class.
- Missing file means no halt flags are active.
- Malformed JSON means fail closed for subprocess wrapper preflight and report the state file path.

### 4.4 Wrapper preflight

Wrapper preflight must run checks in this order:

1. Halt flag check.
2. Binary availability check.
3. Verified flag surface check.
4. Envelope budget check.
5. Command assembly.

This order prevents a halted class from probing or invoking external CLIs.

The wrapper must return structured results rather than raising for expected operational states. Expected states include missing binary, missing verified flag docs, budget cap reached, halted, and disabled.

### 4.5 Logging

This A slice does not write Hermes-Learning-Ledger entries by default because it does not perform real model invocations. It may expose a pure formatter that later phases can use to write ledger rows after approved activation.

Test logs may use temporary Hermes home directories only. Tests must not write to Christopher's real `~/.hermes/state`.

## 5. Local CLI command surface

This slice may add a local-only helper function or internal command handler for budget formatting, but it should not add Telegram verbs.

If a CLI slash command is included in implementation, it must be limited to `/budget` in local CLI mode and must read only local envelope state. It must not send messages, restart gateway, alter config, create cron, or invoke subprocesses.

A safe `/budget` output shape:

```text
Cowork envelope: 0/191 used, 191 available, window not started
Codex envelope: 0/170 used, 170 available, window not started
Halt flags: all=false, cowork_headless=false, codex_auto_dispatch=false
```

Adding `/cowork-budget` and `/codex-budget` as aliases is acceptable only if they remain local read-only command aliases in this slice. Telegram exposure belongs to Phase 6 and is out of scope.

## 6. Error handling

- Missing `claude`: return `missing_binary` and setup guidance.
- Missing `codex`: return `missing_binary` and keep Codex path disabled.
- Missing verified docs: return `missing_verified_flags` unless local help text is explicitly supplied for diagnostic mode.
- Missing required flag: return `missing_required_flag` with the exact flag name.
- Halt active: return `halted` with the active flag name.
- Budget cap reached: return `budget_blocked` with used, cap, and reset time if known.
- Malformed state: return `state_error` and fail closed.

No error path should print secrets or full environment variables.

## 7. Testing strategy

Use TDD. No production code should be written before a failing test exists.

Minimum tests:

1. Flag parser accepts Claude help containing `--print` and `--max-budget-usd`.
2. Flag parser rejects draft `--max-cost-usd` as missing when local help lacks it.
3. Codex wrapper returns `missing_binary` when binary path is absent.
4. Halt flag `all: true` prevents later binary or flag checks.
5. Class halt flag prevents the matching wrapper class only.
6. Envelope defaults calculate 85 percent caps: 225 to 191, 200 to 170.
7. Envelope increments within window.
8. Envelope resets after 5 hours.
9. Envelope blocks non-priority work at cap.
10. Wrapper preflight assembles Claude argv only when halt, binary, flags, and envelope checks pass.
11. State paths are rooted in `get_hermes_home()` and tests use temporary Hermes homes.
12. Local `/budget` formatting, if implemented, reads state and prints no secrets.

Target test commands:

```bash
python -m pytest tests/agent/test_cmh_subprocess_flags.py -q -o 'addopts='
python -m pytest tests/agent/test_cmh_subprocess_envelope.py -q -o 'addopts='
python -m pytest tests/agent/test_cmh_subprocess_halt_flags.py -q -o 'addopts='
python -m pytest tests/agent/test_cmh_subprocess_wrappers.py -q -o 'addopts='
```

If slash command support is included:

```bash
python -m pytest tests/test_cmh_budget_command.py -q -o 'addopts='
```

## 8. Out of scope

This slice must not implement or activate:

- Telegram alerts or Telegram budget verbs.
- Gateway callbacks, gateway restart, or gateway config changes.
- `~/.hermes/wrappers/` runtime scripts.
- `~/.hermes/bin/spawn-cowork-headless.sh`.
- launchd, daemon, cron, webhook, MCP, AgentMail, or provider changes.
- Live Claude or Codex model subprocess calls as part of normal Hermes routing.
- `tier_4_substantive_default` route activation.
- `tier_2_code_reasoning` route activation.
- Codex auto-dispatch folder writes.
- R109 fire.
- Git push, merge, deploy, or production mutation.

## 9. Rollback

Because this slice is local repo code only, rollback is simple:

- Revert the implementation commit or remove the new `agent/cmh_subprocess/` package and tests.
- Delete any temporary test state under test-created Hermes homes.
- Do not remove or edit real `~/.hermes/state` unless Christopher explicitly asks.
- No gateway, cron, launchd, MCP, or provider rollback should be required because none are activated.

## 10. Acceptance criteria

The A-slice is complete when:

- All tests listed in the implementation plan pass.
- `codex` absence is handled as a clean disabled or missing-binary result.
- Claude flag verification reflects local reality and rejects draft-only flags.
- Envelope and halt state are profile-aware and fail closed on malformed state.
- No real model subprocess is invoked in normal tests.
- No Telegram, gateway, cron, launchd, MCP, provider, send, deploy, merge, R-audit, or production surface changed.
- A closeout clearly states that this is foundation-only and not architecture-pivot activation.

## 11. Open dependency for later phases

Before Phase 1 activation from the Cowork checklist, Codex Wave 1.16.E must land verified invocation docs. If those docs are still missing, Hermes may continue building local tests and disabled foundations but must not activate subprocess routing.

When verified docs arrive, later work should reconcile their flag names against the local verifier. If docs conflict with live local help output, implementation should fail closed and report the discrepancy rather than guessing.
