# Worker persistence substrate (first operation family)

This is an operational transport/storage slice, not Q completion or an AIAgent rollout.
The ordinary daemon exposes worker.register, worker.adopt and worker.persist.
Registration requires the existing authenticated session controller and an idle assigned
session, and currently admits only explicit local compute workers. Existing cron, child,
Kanban and compute-host consumers are NOT switched.

## Durable operations

- transcript.append: existing batch repair/serializer, tool rows, multimodal content,
  reasoning/Codex/display/API sidecars, counters, FTS, row IDs and canonical-content returns.
- turn.acquire / turn.renew / turn.release: existing logical-conversation lease key and claim
  helper, bounded TTL, holder checks, sequenced receipts; wait/abort callbacks stay local.
- usage.main: token/cost counters and per-route accounting in the receipt transaction;
  incremental and absolute semantics preserved. No token-writer enqueue-before-commit ACK.
- usage.auxiliary: per-model/task accounting only, not main session totals.
- execution.finish: terminal fencing sequenced after prior receipts, duplicate retry allowed.
  This does not end the physical session or settle an unrelated human admission.

The adapter supplies append_messages_batch, turn try/acquire/refresh/release,
queue_token_counts (synchronous durable ACK), record_auxiliary_usage, flush_token_counts
(a prior-receipt barrier), explicit retry_pending/adopt/finish and local close.
Chunked batch copy and compression-holder/nondefault append-TTL options are explicitly
unsupported; constructor-time reads and all full-consumer dependencies remain blockers.

## Authority and retry contract

No new registry or schema: reuse worker_executions, worker_receipts and runtime_epoch.
The existing adoption_digest binds a hash of the producer principal, profile, session,
execution ID, generation, PID, process birth time and private secret. Registration is
controller-authorized; adoption revalidates the live same-user process and exact durable
claim before advancing the epoch. It never reruns a tool or claims a new human input.
Persistence/reconnect sockets use the existing attenuated worker-adoption ticket purpose,
not interactive capabilities. Mutations also require the private assignment proof.
No bearer proof is put into public events, discovery, registry rows or read responses.

The private journal is atomic/fsynced, byte-bounded and exclusively file-locked, outside
age-pruned caches. Only one unacknowledged mutation is admitted: later writes fail visibly
until explicit retry. Caller payloads are frozen; sequences are never silently discarded.
Transport and disk failures are sticky. No SQLite fallback. Synchronous external calls
use the WebSocket library's frame-receiver thread and reject owner-PID/event-loop self-RPC.
The same-PID direct-authority adapter and production control/cancellation consumers are
not installed by this slice.

## Unsupported core inventory operations

- `archive_and_compact`
- `clear_compression_failure_cooldown`
- `clear_session_activity_labels`
- `create_session`
- `declared_scope_identity`
- `end_session`
- `get_active_message_watermark`
- `get_compression_failure_cooldown`
- `get_compression_failure_cooldown_row`
- `get_compression_fallback_streak`
- `get_compression_ineffective_count`
- `get_compression_lineage`
- `get_compression_lock_holder`
- `get_compression_recovery_deadline`
- `get_compression_tip`
- `get_conversation_root`
- `get_messages_as_conversation`
- `get_next_title_in_lineage`
- `get_session`
- `get_session_model_config_value`
- `get_session_title`
- `get_session_title_source`
- `is_explicit_fork_child`
- `latest_conversation_boundary`
- `patch_session_model_config`
- `publish_compression_child`
- `record_compression_failure_cooldown`
- `refresh_compression_lock`
- `release_compression_lock`
- `reopen_orphaned_compression_session`
- `resolve_resume_session_id`
- `restore_compression_failure_cooldown_row`
- `session_lifecycle_statuses`
- `set_auto_title`
- `set_compression_fallback_streak`
- `set_compression_ineffective_count`
- `set_compression_recovery_deadline`
- `set_latest_user_api_content`
- `set_session_title`
- `set_session_title_source`
- `touch_session_activity`
- `try_acquire_compression_lock`
- `update_session_billing_route`
- `update_system_prompt`

Extension inventory blockers remain: recall/search/raw storage-state access, reactions,
independent canonical opens, async-delegation dispatch/partial/completion/delivery SQL,
compute-host control helpers, producer-specific cron/child/Kanban claims and lifecycle
reservation. Do not infer those operations from absent attributes or replace all workers.

## Proof boundaries

The real-process test launches ordinary gateway.run owners and a separate worker using
production private-control tickets and WebSockets, with actual temporary SQLite rows.
It discards a committed transport result at the worker ACK boundary and retries exactly;
this is controlled lost-ACK evidence, not a kernel-packet-loss experiment. It kills the
owner, retains one pending auxiliary mutation during outage, explicitly adopts the same
live PID and retries once, preserving the inert tool marker and terminal receipt.
It records zero canonical SQLite connection audit events and zero writable canonical FDs
in the worker, and rejects wrong profile/session/generation, conflicting sequence, a
changed process birth proof, oversized outbox work and ordinary session creation from
a worker-purpose connection. This is not a full AIAgent/model/tool execution migration.
Native Windows/macOS process tests, comprehensive malformed-payload fuzzing, daemon
shutdown coordination and full frontend worker observability remain later gates.
