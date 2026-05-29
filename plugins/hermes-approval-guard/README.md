# hermes-approval-guard

Two-stage tool-call supervision plugin for Hermes Agent. Adds security
review for `write_file`, `patch`, `delegate_task`, and `execute_code`
— tools not covered by the built-in `approvals.mode`. Disabled by
default; zero measurable overhead when off.

## Quick Start

```yaml
# ~/.hermes/config.yaml
plugins:
  enabled:
    - hermes-approval-guard

plugin_guard:
  enabled: true
  provider: zjic
  fail_open: true
```

Restart Hermes, then verify:

```bash
hermes plugins list | grep approval-guard
```

## Architecture

```
Tool call → pre_tool_call hook
  ├─ SAFE_TOOLS (20+ read/query tools) → ALLOW (0ms)
  ├─ terminal / process                → delegate to approvals.mode
  ├─ HARDLINE rules                    → BLOCK (<1ms)
  ├─ Stage 1: LLM classifier           → ALLOW / DENY / ESCALATE (~500ms)
  └─ Stage 2: ACP Agent (optional)     → deep review (3-8s)
```

## Stage 1 — Rules + LLM

| Mechanism | Latency | Coverage |
|-----------|:------:|---------|
| SAFE_TOOLS bypass | 0ms | `read_file`, `search_files`, `web_search`, `vision_analyze`, `session_search`, `skill_view`, `clarify`, `hindsight_recall`, `hindsight_reflect`, and 10+ more |
| HARDLINE path block | <1ms | `/etc/`, `/boot/`, `/sys/`, `/proc/`, `/dev/`, `~/.ssh/`, `~/.gnupg/` |
| HARDLINE file block | <1ms | `.env`, `config.yaml`, `id_rsa`, `id_ed25519`, `authorized_keys` |
| Delegate danger block | <1ms | `"delete all"`, `"rm -rf /"`, `"format disk"`, `"wipe system"`, `"destroy everything"` |
| LLM classifier | ~500ms | All remaining ambiguous calls via `call_llm(task="approval")` |

## Stage 2 — ACP Agent (optional)

Launches a separate Hermes instance via `hermes chat -q --profile approval`
with `terminal,file,web` toolsets. Returns structured JSON verdict:

```json
{"verdict": "ALLOW|DENY|MODIFY", "reason": "...", "suggestion": "..."}
```

Enable with `plugin_guard.stage2.enabled: true` and create an `approval` profile.

## Relationship with approvals.mode

```
terminal commands  → approvals.mode (smart/manual) — unchanged
write_file / patch → approval-guard (newly protected)
delegate_task      → approval-guard (newly protected)
execute_code       → approval-guard (newly protected)
read/search/query  → bypass both (always safe)
```

The plugin **extends**, not replaces. Both systems coexist without conflict.

## Failure Modes

All failure paths default to **fail-open** (`plugin_guard.fail_open: true`):

| Failure | Behavior |
|---------|----------|
| Config missing/corrupt | Plugin self-disables; all tools pass |
| Stage 1 LLM unavailable | `fail_open:true` → ALLOW; `fail_open:false` → DENY |
| Stage 2 ACP crash | `fail_open:true` → ALLOW; `fail_open:false` → DENY |
| Memory backend down | Silent skip; tool execution unaffected |
| Module import failure | Caught by Hermes plugin loader; not registered |

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_guard.enabled` | bool | `false` | Master on/off switch |
| `plugin_guard.provider` | str | — | LLM provider for classification |
| `plugin_guard.model` | str | — | Model name (fast model recommended) |
| `plugin_guard.fail_open` | bool | `true` | LLM down → allow (safe default) |
| `plugin_guard.stage1.timeout` | int | `5` | Seconds for LLM classification |
| `plugin_guard.stage2.enabled` | bool | `false` | Enable ACP Agent deep review |
| `plugin_guard.stage2.profile` | str | `"approval"` | Hermes profile for review agent |
| `plugin_guard.stage2.timeout` | int | `15` | Seconds for deep review |
| `plugin_guard.memory.backend` | str | `"hindsight"` | `hindsight`, `honcho`, or `none` |
| `plugin_guard.memory.bank` | str | `"approval"` | Hindsight bank or Honcho user_id |

## Files

| File | Purpose |
|------|---------|
| `plugin.yaml` | Plugin manifest (standalone, hook: `pre_tool_call`) |
| `__init__.py` | `PluginContext.register_hook` entry |
| `guard.py` | Main dispatcher: SAFE → HARDLINE → LLM → ACP |
| `stage1_rules.py` | Pattern-based fast path |
| `stage1_llm.py` | LLM semantic classifier |
| `stage2_acp.py` | ACP Agent subprocess call |
| `feedback.py` | Structured deny messages |
| `hindsight_store.py` | Memory backend (Hindsight HTTP API / Honcho) |
| `recommended-config.yaml` | Annotated config template |

## Testing

```bash
cd plugins/hermes-approval-guard

# Unit tests (5 scenarios)
python3 test_unit.py

# Integration tests (9 scenarios, 44 cases)
python3 test_integration.py
```

## Compatibility

- Hermes ≥ 0.14.0
- Verified against `get_pre_tool_call_block_message()` at `hermes_cli/plugins.py:1666`
- Hook callback signature matches `PluginContext.register_hook`
- All three tool execution paths covered (concurrent, sequential, invoke_tool)
