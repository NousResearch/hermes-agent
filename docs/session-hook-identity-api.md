# Native plugin session identity

Native plugins can bind resources to trusted host-owned session identity without
reading process-global session variables or accepting model-supplied owner IDs.

## Callback contract

```python
def register(ctx):
    ctx.register_hook("on_session_identity", remember_owner)
    ctx.register_hook("pre_tool_call", before_tool)
    ctx.register_hook("on_session_finalize", release_runtime)

def remember_owner(*, runtime_session_id=None, stored_session_id=None,
                   session_id=None, task_id=None, profile=None,
                   hermes_home=None, source=None, surface=None, **kwargs):
    # Record the exact association under the owning profile/home.
    # Do not treat unknown IDs as aliases for another field.
    ...
```

The fields below are additive **keyword arguments**, not a nested object. Existing
narrow callbacks continue to receive only their declared arguments.

| Field | Meaning |
| --- | --- |
| `runtime_session_id` | Live TUI/desktop registry ID, also returned as `session_id` by session RPCs. `None` on standalone agents without a UI binding. |
| `stored_session_id` | Durable routing key of the dispatch session; initially the allocated/resumed conversation key. |
| `session_id` | Dispatching agent's current conversation ID; before lazy construction, the host-allocated/resumed conversation key. |
| `task_id` | Actual tool/turn task ID, `None` before a turn is bound. Runtime teardown carries the last bound task if any. |
| `profile` | Owning captured profile name (`default`, a named profile, or `custom`). |
| `hermes_home` | Owning captured `pathlib.Path`; stringify when writing JSON. |
| `source`, `surface` | Owning session's source, e.g. `desktop` or `tui`; standalone agents use their explicit platform, or `None`. |

Unknown identity fields are `None` on the extended agent/session dispatch paths.
Legacy direct pre-tool dispatch retains its existing empty-string defaults for
`session_id`/`task_id`. IDs are not recovered from environment variables.
Plugin discovery and invocation are scoped to the captured `hermes_home`, including
bounded hook-worker threads; the caller's previous scope is restored afterward.

## Timing

`on_session_identity` is a generic ownership observer published by the TUI/desktop
backend **before returning a created/resumed runtime ID**, even when the agent and
state.db row have not yet been created. It is also emitted on live reattachment
and when compression re-anchors the stored routing key. It is not a prompt/turn
start event. Treat it as idempotent, and keep callbacks short.

Existing observer timeout/error isolation applies. A failed/timed-out callback
is not proof of ownership: plugins authorizing REST requests must fail closed
when their trusted owner record is missing.

`on_session_start`, `pre_tool_call`, `on_session_end`, CLI session boundaries,
and TUI/desktop `on_session_reset`/`on_session_finalize` carry the same fields.
Both sequential/concurrent tool-executor policy checks and direct/inline agent
tool invocation receive the dispatching agent's identity. `post_tool_call` is
unchanged by this extension; it is not an ownership-establishment boundary.

## Compression and resume

IDs are not interchangeable. During compression:

1. `agent.session_id` may already be a child conversation while the current turn's
   `task_id` and live record's `session_key` still refer to the parent.
2. Pre-tool hooks report these **actual distinct IDs**, not synthetic equality.
3. Gateway re-anchoring emits `on_session_identity` with the updated stored key,
   retaining the same runtime ID and owning profile/home.

`stored_session_id` is therefore **not an immutable lineage root**. Plugins needing
lineage continuity should retain the associations learned from trusted events.
Resuming into a new runtime produces a new runtime ID, even for the same stored
conversation. Standalone agent session resets rebind the stored key and clear the
old task identity while preserving the captured owning profile.

## Turn end versus finalization

`on_session_end` means a **turn** ended, not that the runtime is gone. It is also
emitted for interrupted teardown for legacy compatibility. Do not release
session-scoped resources on every `on_session_end`.

`on_session_finalize` denotes the actual CLI/TUI/desktop session boundary.
TUI/desktop teardown is idempotent per live runtime, including lazy agent-less
records. A new resume runtime can still use the same durable conversation.

## Implementation seams

- `hermes_cli/session_hook_context.py`: capture/agent snapshot/profile scope.
- `agent/agent_init.py::_init_session_state`: standalone capture.
- `run_agent.py::AIAgent.reset_session_state`: explicit session rebind.
- `agent/conversation_loop.py::_restore_or_build_system_prompt`: start callback.
- `agent/tool_executor.py::_pre_tool_block` and
  `agent/agent_runtime_helpers.py::_pre_tool_block_message`: pre-tool identity.
- `agent/turn_finalizer.py::finalize_turn`: turn-end callback.
- `tui_gateway/session_lifecycle.py`: runtime publication and teardown snapshots.
- `tui_gateway/methods_session.py` and `tui_gateway/server.py`: create/resume,
  eager registration, and lazy-built-agent identity transfer.
- `tui_gateway/session_compression.py::_sync_session_key_after_compress`: re-anchor.
- `hermes_cli/cli_session_mixin.py` and `cli.py`: CLI reset/finalize/interrupt paths.

Regression receipts live in `tests/plugins/test_session_hook_identity.py`: real
plugin discovery under temporary homes, two profiles/runtimes, actual AIAgent and
SessionDB construction, lazy/cold/eager resume, pre-tool and inline tool invocation,
production compression recovery, and turn-end versus runtime-finalization events.
