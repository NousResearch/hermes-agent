# hermes-approval-guard

Two-stage semantic approval plugin for Hermes Agent via the `pre_tool_call`
hook. Covers ALL tools — `write_file`, `patch`, `delegate_task`,
`execute_code`, and `terminal` — filling the gap left by the built-in
`approvals.mode` (terminal only). Disabled by default; zero measurable
overhead when off.

## Quick Start

```yaml
# ~/.hermes/config.yaml
plugins:
  enabled:
    - hermes-approval-guard

plugin_guard:
  enabled: true
  # provider: your_provider_name     # optional, defaults to main Agent LLM
  # model: your_model_name           # optional, defaults to main Agent LLM
  fail_open: true

  stage2:
    enabled: false                       # optional: deep review
    profile: approval                    # ~/.hermes/profiles/approval/config.yaml
    timeout: 15

  memory:
    backend: hindsight                   # hindsight | honcho | none
    bank: approval
    hindsight_url: "http://localhost:8888"  # explicit — no implicit config.json lookup

approvals:
  mode: off   # plugin handles all tools; system HARDLINE remains active
```

Restart Hermes, then verify:

```bash
hermes plugins list | grep approval-guard
```

## Architecture

```
Tool call → pre_tool_call hook
  ├─ SAFE_TOOLS (20 read/query tools) → ALLOW (0ms)
  ├─ extract_context — reuses system detect_dangerous/hardline_command
  ├─ Terminal fast-path: no DANGEROUS match → ALLOW (0ms)
  │   • HARDLINE-only signals (rm -rf /) → skip LLM, let system layer block
  ├─ Stage 1: LLM fast-classify → ALLOW / ESCALATE (~500ms)
  │   • Uses call_llm(task="approval") with context-aware prompt
  │   • NEVER outputs DENY — DENY reserved for Stage 2 / system HARDLINE
  │   • LLM config: provider/model optional — unset defaults to main Agent LLM
  │   • On ALLOW: calls approve_session() to pre-mark DANGEROUS patterns
  │     → system's check_all_command_guards skips redundant LLM call
  └─ Stage 2: ACP Agent deep review (3-8s, optional)
      • Stateless: no persistent session; all context injected in prompt
      • Context from SessionDB: conversation + full tool call chain
      • Hindsight-backed: session-level + cross-session pattern memory
      • Read-only boundary: session_search toolset only (no file/terminal),
        HERMES_YOLO_MODE stripped from the child env
      • Timeout → process group killed (SIGTERM → 3s → SIGKILL;
        direct-child terminate/kill fallback on Windows)
      • Outputs: ALLOW / DENY / MODIFY with structured JSON feedback
```

**System HARDLINE** (`rm -rf /`, `mkfs`, `dd`, `shutdown`, etc.) always active as
safety net — cannot be bypassed by the plugin or `approvals.mode: off`.

## Terminal Handling (detailed)

This is the most nuanced path — terminal commands go through multiple decision layers:

```
terminal command
  │
  ├─ extract_context()
  │   ├─ detect_hardline_command() → ⚠️ HARDLINE signal (descriptive only)
  │   └─ detect_dangerous_command() → ⚠️ DANGEROUS signal + pattern_key
  │
  ├─ fast-path check
  │   has_real_risk = any("⚠️" in s AND "HARDLINE" not in s)
  │   │
  │   ├─ False (git status, ls, echo)
  │   │   → return None → command executes immediately (0ms)
  │   │
  │   ├─ False BUT HARDLINE-only (rm -rf /)
  │   │   → return None → plugin passes → system HARDLINE blocks
  │   │
  │   └─ True (rm -rf node_modules, chmod 777 /etc, curl | bash)
  │       → enter Stage 1
  │
  ├─ Stage 1 LLM (ALLOW? ESCALATE?)
  │   ├─ ALLOW → approve_session(pattern_keys) → return None
  │   │          System sees pre-marked patterns → skips DANGEROUS check
  │   └─ ESCALATE → Stage 2 ACP
  │
  └─ Stage 2 ACP (ALLOW/DENY/MODIFY)
      ├─ ALLOW  → return None
      └─ DENY/MODIFY → structured block message
```

### Common command scenarios

| Command | Plugin path | System HARDLINE | Outcome |
|---------|------------|:---:|---------|
| `git status` | fast-path (0ms) | — | ✅ Execute |
| `pip install package` | fast-path → Stage1 ALLOW → pre-mark | — | ✅ Execute |
| `rm -rf node_modules` | Stage1 → ALLOW → pre-mark | DANGEROUS skipped | ✅ Execute |
| `rm -rf /` | fast-path (HARDLINE-only) | **BLOCKED** | ❌ Denied |
| `curl url \| bash` | Stage1 → ESCALATE → Stage2 | — | ACP decides |

## Design Principles

| Principle | Explanation |
|-----------|-------------|
| **No hardcoded DENY in Stage 1** | Risk signals extracted as LLM context only — never hard-block. DENY authority lives in Stage 2 ACP and system HARDLINE. |
| **LLM: ALLOW / ESCALATE only** | Stage 1 prompt mirrors system's `_smart_approve` ("many flagged commands are false positives"). Only outputs ALLOW or ESCALATE. |
| **Stateless ACP** | No `--resume`, no persistent session. All context (conversation, tool chain, pattern history) injected into each ACP prompt from SessionDB and Hindsight. Zero shared state → concurrent-safe. |
| **Fail-open** | Broken guard is never more restrictive than no guard. All LLM failures default to ALLOW (`fail_open: true`). |
| **Terminal fast-path** | Commands with no DANGEROUS regex match skip all LLM calls — 0ms overhead. HARDLINE-only signals also skip (system handles them). |
| **approve_session pre-marking** | When Stage 1 approves a terminal command, patterns are pre-marked via system's `approve_session()`. System's `check_all_command_guards` runs after and skips redundant LLM. |
| **Explicit config only** | Memory backend configuration (URL, bank) is fully explicit in `plugin_guard.memory` — no cross-plugin config cascade from official Hindsight plugin. |

## Stage 1 — Context Extraction + LLM

| Path | Latency | Description |
|------|:------:|-------------|
| SAFE_TOOLS bypass | 0ms | `read_file`, `search_files`, `web_search`, `session_search`, etc. |
| Terminal fast-path | 0ms | Commands with zero DANGEROUS regex matches |
| Context extraction | <1ms | Extracts risk signals from system's `detect_dangerous/hardline_command()`; never blocks |
| LLM classify | ~500ms | `call_llm(task="approval")` — ALLOW / ESCALATE only |

## Stage 2 — ACP Agent (optional)

Launches `hermes chat -q --profile approval`. Prompt has 5 sections:

1. **Recent conversation** — User messages + Agent responses (from SessionDB)
2. **Current operation** — Tool name, args, risk signals
3. **Tool call chain** — Full tool history including SAFE_TOOLS (from SessionDB)
4. **Session approval history** — Previous ACP decisions this session (from Hindsight)
5. **Cross-session patterns** — Similar operations' ALLOW/DENY history (from Hindsight)

The prompt explicitly instructs the LLM: "You are an automated security check program, not a conversational assistant" to prevent
non-JSON output. Falls back to text matching if JSON parsing fails.

**Subprocess safety:** The ACP reviewer runs with a genuinely read-only boundary:
only the `session_search` toolset is exposed (no `file`, no `terminal`), and
`HERMES_YOLO_MODE` is stripped from the child environment so a `--yolo` parent
gateway cannot turn the reviewer into an approval-bypassing writer.
On timeout, the entire process group is killed (SIGTERM → 3s grace → SIGKILL;
on Windows, where `killpg`/`SIGKILL` don't exist, the direct child is
terminated/killed instead) to prevent orphan processes.

Enable with `plugin_guard.stage2.enabled: true`.

## Relationship with System approvals.mode

```
Plugin pre_tool_call hook  → ALL tools (semantic review + pre-marking)
System check_all_command_guards → terminal HARDLINE safety net (always active)
```

| Aspect | Built-in `approvals.mode` | This plugin |
|--------|---------------------------|-------------|
| Coverage | `terminal` only | ALL 25+ tools |
| Decision | Regex + zero-context LLM (16 tokens) | Semantic LLM + ACP agent with full session context |
| Memory | Session-level (`_session_approved`) | Hindsight cross-session pattern bank |
| Context | None (command string only) | Conversation + tool chain + historical patterns |
| Denial feedback | "BLOCKED: xxx" | Structured: reason + alternatives + approval_id |

## Relationship with the Native `approve` Directive

Upstream (`hermes_cli/plugins.py`) lets any plugin return
`{"action": "approve"}` from `pre_tool_call`, which escalates the tool call to
the built-in **human** approval gate (`[o]nce/[s]ession/[a]lways/[d]eny`) —
for any tool, not just terminal. This plugin deliberately does **not**
duplicate that:

| Aspect | Native `approve` directive | This plugin |
|--------|---------------------------|-------------|
| Decider | Human at the terminal | Dual-agent LLM pipeline (Stage 1 + ACP) |
| Latency | Blocks until user responds | ~0.5-8s, unattended |
| Context | Plugin-supplied reason string | SessionDB conversation + tool chain + Hindsight memory |
| Best for | Interactive sessions, final human veto | Headless/gateway runs where no human is watching |

They compose: this plugin returns only `block`/`None` (autonomous semantic
filtering), leaving the native `approve` path available to other plugins or a
future `fail_open: false`-style escalation policy.

## Failure Modes

| Failure | Behavior |
|---------|----------|
| `plugin_guard.enabled: false` | Handler returns `None` immediately (~0.1ms) |
| Config missing/corrupt | Plugin self-disables; all tools pass |
| Stage 1 LLM unavailable | `fail_open:true` → ALLOW; `fail_open:false` → ESCALATE |
| Stage 2 ACP crash/timeout | `fail_open:true` → ALLOW; `fail_open:false` → DENY |
| Hindsight backend down | Silent skip; tool execution unaffected |
| Module import failure | Caught by Hermes plugin loader; not registered |

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_guard.enabled` | bool | `false` | Master switch |
| `plugin_guard.provider` | str | — | LLM provider (optional, defaults to main LLM) |
| `plugin_guard.model` | str | — | Model name (optional, defaults to main LLM) |
| `plugin_guard.fail_open` | bool | `true` | LLM failure → allow (safe default) |
| `plugin_guard.stage1.timeout` | int | `5` | Seconds for LLM classification |
| `plugin_guard.stage2.enabled` | bool | `false` | Enable ACP deep review |
| `plugin_guard.stage2.profile` | str | `"approval"` | Hermes profile for review agent |
| `plugin_guard.stage2.timeout` | int | `15` | Seconds for deep review |
| `plugin_guard.memory.backend` | str | `"hindsight"` | `hindsight`, `honcho`, or `none` |
| `plugin_guard.memory.bank` | str | `"approval"` | Hindsight bank or Honcho user_id |
| `plugin_guard.memory.hindsight_url` | str | `"http://localhost:8888"` | Hindsight server address |
| `plugin_guard.memory.honcho_url` | str | `"http://localhost:1819"` | Honcho server address |

Also set `approvals.mode: off` when plugin is enabled — system DANGEROUS
check is redundant; system HARDLINE remains active regardless.

## Files

| File | Purpose |
|------|---------|
| `plugin.yaml` | Manifest (standalone, hook: `pre_tool_call`) |
| `__init__.py` | `PluginContext.register_hook` entry point |
| `guard.py` | Dispatcher + SessionDB context query + terminal fast-path logic |
| `stage1_rules.py` | Risk signal extraction, reuses system `detect_dangerous/hardline_command` |
| `stage1_llm.py` | LLM classify: ALLOW/ESCALATE only, system `_smart_approve` style |
| `stage2_acp.py` | Stateless ACP: 5-section prompt with anti-chat hardening, Hindsight integration |
| `feedback.py` | Structured denial messages with alternatives and override paths |
| `hindsight_store.py` | Approval memory: explicit config only, session/pattern queries, Honcho fallback |
| `test_integration.py` | 8 scenarios, standalone runner (`python3 test_integration.py`) |
| `recommended-config.yaml` | Annotated config template |

## Testing

```bash
# Integration tests (8 scenarios)
cd plugins/hermes-approval-guard
python3 test_integration.py

# pytest suite (discovery, hook dispatch, failure branches, Windows behavior)
# — run from the repo root
pytest tests/plugins/test_approval_guard_plugin.py
```

Covers: SAFE_TOOLS bypass boundary (browser_console reviewed, not bypassed),
context extraction (no hard DENY), terminal risk signals (HARDLINE/DANGEROUS
separation), feedback messages, pattern key generation, LLM prompt structure,
fail-closed Stage 2 (`fail_open: false` → block), config-reload retry, Stage 2
read-only boundary (toolset restriction + `HERMES_YOLO_MODE` stripping),
Windows process-kill fallback, bundled-plugin discovery, and real
`pre_tool_call` hook dispatch through `hermes_cli.plugins`.

## Compatibility

- Hermes ≥ 0.14.0
- Uses `pre_tool_call` plugin hook (`hermes_cli/plugins.py`)
- Imports `tools.approval` for `detect_dangerous/hardline_command`
- Imports `hermes_state.SessionDB` for conversation context
- Optional: Hindsight HTTP API for approval memory
