# Gateway `*_runtime_service` inventory

Status relative to the **full production import graph** (not only `run.py`
direct imports): `run.py`, mixins, routers, platforms, and their transitive
`*_runtime_service` dependencies.

| Status | Meaning |
|--------|---------|
| **WIRED** | Reachable from production gateway code |
| **removed** | Truly unwired parallel paths deleted in the 清屎山 pass |

## WIRED (21)

| Module | Notes |
|--------|--------|
| `agent_turn_start_runtime_service` | Promoted bootstrap from `_handle_message_with_agent` |
| `agent_turn_preflight_runtime_service` | Post-bootstrap pre-agent: hygiene + notes + message stage |
| `agent_turn_finish_runtime_service` | Post-`_run_agent` success: normalize, transcript, exhaust reset, voice/media |
| `agent_turn_error_runtime_service` | `_handle_message_with_agent` except: typing stop, crash persist, status copy |
| `auto_vision_runtime_service` | `run.py` |
| `busy_followup_runtime_service` | `run.py` |
| `direct_shortcut_runtime_service` | turn-start + busy path |
| `direct_shortcut_trace_runtime_service` | direct shortcuts / ops |
| `direct_control_event_runtime_service` | direct control router |
| `direct_tool_result_runtime_service` | direct control chain |
| `auto_background_runtime_service` | direct ops |
| `shared_group_history_runtime_service` | direct ops |
| `send_runtime_service` | platform specs / router |
| `group_runtime_service` | qq_napcat |
| `group_control_runtime_service` | router |
| `group_moderation_runtime_service` | router |
| `group_monitoring_runtime_service` | ops |
| `group_archive_runtime_service` | ops |
| `group_runtime_status_runtime_service` | router |
| `qq_intel_runtime_service` | router |
| `qq_social_runtime_service` | router |

## Removed (were DEAD / parallel only)

`agent_completion`, `agent_delivery`, `agent_followup`, `agent_lifecycle`,
`agent_prelude`, `agent_progress`, `agent_response`, `agent_sync`, `agent_turn`,
`attachment_message`, `command_preprocessing`, `command_resolution`,
`context_reference`, `foreground_turn`, `message_preprocessing`,
`message_turn_context`, `onboarding`, `running_agent_staleness`,
`session_hygiene`, `transcript_persistence` — each as `*_runtime_service`.

## Convention

1. Wire from production **and** delete parallel inline paths, or do not land the file.
2. Inventory must use the **transitive** production graph (mixins/routers count).
