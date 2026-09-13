# Worker constructor context — wave 20 (partial migration)

This extends `worker-persistence-substrate.md`; it does **not** enable the legacy
compute-host, cron-agent, or delegated-child constructors as migrated workers.
Those consumers still have unsupported operations in the Q1 inventory.

## Implemented

- `session.context`: assigned-session snapshot, resolved content-addressed system
  prompt, with the same epoch/generation/sequence/receipt fence as worker mutations.
  Read receipts are replayable snapshots, not a promise to refresh an old receipt.
- `session.prompt`: content-addressed prompt update and GC inside the receipt transaction.
- `session.sidecars`: merge only `_usage_anchor` / `_proactive_prune_rearm_tokens`;
  no arbitrary model-config, lineage, credential, or identity mutation.
- Concrete store getters restore title, compression cooldown/raw cooldown, fallback
  streak, ineffective count and recovery deadline before context-engine binding.
- A successful **self** registration/adoption through `WorkerRPC` marks that
  interpreter as a worker before AIAgent imports. The marker is process-scoped so
  fresh helper threads cannot perform owner-only delegation recovery.
- `async_delegation` leaves recovery/delivery queues with the ordinary owner and
  refuses direct worker ledger opens. Its actual dispatch/finish/delivery SQL has
  **not** been migrated to typed RPC.
- Existing child/cron opener seams refuse registered-worker fallback to writable
  `SessionDB` or persistence-disabled `None`. Ordinary unregistered callers keep
  their existing behavior. These are migration fences, not working child/cron RPC.

The process marker is cooperative ownership, not an OS sandbox. Arbitrary Python,
raw SQLite, plugins, and unrelated SessionDB constructors are not confined by it.
No environment credential scrubbing, toolset changes, prompt-prefix rewrites,
profile policy changes, new daemon, or new worker table is introduced.

## Live proof

`tests/gateway/test_worker_agent_execution.py` runs an ordinary `gateway.run`
subprocess plus a separate interpreter constructing and executing the real AIAgent
with `RuntimeSessionStore`, against an owned loopback HTTP/SSE model fixture.
It verifies one model request, actual user/assistant rows, persisted system prompt,
10 input / 5 output tokens, terminal worker receipt, released turn lease, zero
canonical SQLite connection audit events and zero writable DB/WAL/SHM descriptors.
The second case calls real child/cron openers and the delegation ledger from a fresh
helper thread, requires explicit refusal, then executes a real script-only cron
before the same model inference. This proves rejection and script preservation,
**not** child or cron agent migration or tool-enabled inference.

`tests/gateway/test_worker_persistence.py` separately retains the substrate's real
owner-kill, surviving-worker/outbox, stale epoch rejection, adoption and one-time
usage settlement proof. Lost ACK is a controlled discarded RPC result, not a
kernel packet-drop claim. It is not an AIAgent interrupted mid-inference restart.

`tests/state/test_runtime_worker_context.py` proves assigned-row isolation,
sidecar identity preservation, prompt lost-ACK replay and real ContextCompressor
binding of persisted guards. Missing methods previously silently reset guards.

## Remaining core inventory

AST method-name intersection with the Q1 index: **17 of 51 names**, not 17 complete
unrestricted SessionDB contracts. Append retains substrate option restrictions;
sidecar patches are deliberately restricted. The **34 absent names** are:

- `archive_and_compact`, `clear_compression_failure_cooldown`, `clear_session_activity_labels`
- `create_session`, `declared_scope_identity`, `end_session`, `get_active_message_watermark`
- `get_compression_lineage`, `get_compression_lock_holder`, `get_compression_tip`
- `get_conversation_root`, `get_messages_as_conversation`, `get_next_title_in_lineage`
- `get_session_title_source`, `is_explicit_fork_child`, `latest_conversation_boundary`
- `publish_compression_child`, `record_compression_failure_cooldown`, `refresh_compression_lock`
- `release_compression_lock`, `reopen_orphaned_compression_session`, `resolve_resume_session_id`
- `restore_compression_failure_cooldown_row`, `session_lifecycle_statuses`, `set_auto_title`
- `set_compression_fallback_streak`, `set_compression_ineffective_count`, `set_compression_recovery_deadline`
- `set_latest_user_api_content`, `set_session_title`, `set_session_title_source`
- `touch_session_activity`, `try_acquire_compression_lock`, `update_session_billing_route`

Producer-validated child/cron/Kanban reservation, same-PID authority adapters,
constructor-time identity backfill, compression assignment advance, raw delegation
ledger migration, recalled-tool storage reads, legacy compute control/notification
closure, and deferred finalization remain blockers. Keep all existing consumers
on their existing paths until those contracts are implemented and exercised.
