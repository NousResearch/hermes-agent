# Gateway `*_runtime_service` inventory

Status relative to **production** entrypoints (`gateway/run.py` and mixins it
loads). Updated as part of the qq-napcat “清屎山” pass.

| Status | Meaning |
|--------|---------|
| **WIRED** | Imported/called from production `run.py` (or a production mixin). |
| **DEAD** | No production import; may still have unit tests or DEAD→DEAD calls. |
| **DIVERGENT** | Exists in both a production path and a dead/helper path with **different** contracts — fix before wiring. |

## WIRED (3)

| Module | Production entry |
|--------|------------------|
| `busy_followup_runtime_service` | `run.py` → `handle_gateway_busy_followup` |
| `auto_vision_runtime_service` | `run.py` → auto-vision helpers |
| `direct_shortcut_runtime_service` | `run.py` / busy path → `try_handle_direct_gateway_shortcuts` |

Related non-`*_runtime_service` production modules (do **not** treat as dead):
`empty_response_fallback`, `direct_ops_mixin`, `slash_commands`, `turn_sidecar`, etc.

## DIVERGENT (fix or align before re-use)

| Module / helper | Production | Dead / parallel |
|-----------------|------------|-----------------|
| Turn must-deliver notes | `turn_sidecar` / `_pending_turn_sidecar_notes` on user message | `message_turn_context_runtime_service` historically appended notes into `context_prompt` — **must stay sidecar-only** |
| Post-turn agent cache eviction | `GatewayRunner` post-turn path via unified `_should_evict_cached_agent_after_turn` | `agent_lifecycle_runtime_service.resolve_gateway_effective_model_state` (DEAD, depends on same helper) |

## DEAD (35)

These are **not** imported by `gateway/run.py`. Unit tests are contract-only
unless marked otherwise. Do **not** re-wire into `run.py` without replacing the
inline production path first (no parallel dead paths).

- `agent_completion_runtime_service`
- `agent_delivery_runtime_service`
- `agent_followup_runtime_service`
- `agent_lifecycle_runtime_service`
- `agent_prelude_runtime_service`
- `agent_progress_runtime_service`
- `agent_response_runtime_service`
- `agent_sync_runtime_service`
- `agent_turn_runtime_service`
- `agent_turn_start_runtime_service`
- `attachment_message_runtime_service`
- `auto_background_runtime_service`
- `command_preprocessing_runtime_service`
- `command_resolution_runtime_service`
- `context_reference_runtime_service`
- `direct_control_event_runtime_service`
- `direct_shortcut_trace_runtime_service`
- `direct_tool_result_runtime_service`
- `foreground_turn_runtime_service`
- `group_archive_runtime_service`
- `group_control_runtime_service`
- `group_moderation_runtime_service`
- `group_monitoring_runtime_service`
- `group_runtime_service`
- `group_runtime_status_runtime_service`
- `message_preprocessing_runtime_service`
- `message_turn_context_runtime_service`
- `onboarding_runtime_service`
- `qq_intel_runtime_service`
- `qq_social_runtime_service`
- `running_agent_staleness_runtime_service`
- `send_runtime_service`
- `session_hygiene_runtime_service`
- `shared_group_history_runtime_service`
- `transcript_persistence_runtime_service`

## Convention

When adding a new `*_runtime_service.py`:

1. Either wire it from production **and** delete the parallel inline path, or
2. Leave it DEAD and document it here — never ship a second production path.
