# Worker persistence consumer inventory (Q1)

## Scope and verdict

Source snapshot: `34407c9ec6ca0922233dc9c8f7cd4070609b0bf1`, worktree
`ugw-worker-contract-wave15`, branch `docs/ugw-worker-contract-wave15`.
This is the eval README companion for plan **Q1**, not implementation or a Q2–Q8 acceptance receipt.
The root and agent/gateway/tools/cron/tui_gateway instructions were read. No production imports,
user-state access, vendor calls, worker launch, or production edits are required by the census below.
The plan was read from the parent worktree's
`.hermes/plans/2026-09-08_031437-unified-gateway-runtime.md`; Q requires an inventory before switching a worker.
The preceding `worker-gap-wave14.md` was treated as a lead, not source truth.

**Correction to the preceding audit:** the two proposed adapter modules are absent, but
`hermes_state_runtime.py` already contains `register_worker_execution`, `adopt_worker_execution`,
`persist_worker_message`, and `finish_worker_execution`, with `worker_executions`/`worker_receipts`.
A repository search finds their callers only in `tests/state/test_runtime_worker.py` and
`tests/state/test_runtime_adoption.py`, not gateway/cron/child/compute production.
The text-only primitive is NOT a SessionDB adapter: it accepts only user/assistant/system text,
not tool rows, structured content, reasoning, sidecars, compression, or usage.
Reuse its epoch/assignment/sequence/receipt checks; do not implement a second worker registry.

**Switching any full worker is still blocked.** In particular, replacing `agent._session_db`
alone cannot satisfy zero canonical writable opens:

* `tools/async_delegation.py::_connect` directly opens **canonical `state.db`**, creates/migrates
  `async_delegations`, and performs dispatch/completion/delivery SQL independently of SessionDB.
* `tools/process_registry_results.py::_owns_result` opens writable `SessionDB()` for a lineage read.
* `tools/react_to_message_tool.py::_open_session_db` independently calls registry `acquire()`.
* `tools/session_search_tool.py::_get_message_storage_state` dereferences `_lock` and `_conn`;
  its ordinary search paths also reach cross-session/profile data, not just the execution's row.
* The legacy compute host imports the complete TUI server: `_transfer_db_to_agent` itself calls
  `_get_db()`, and `_session_db`/`_workdir_owner_db` reopen by profile even with an injected agent store.
  Its notification poller and slash/control paths enlarge the execution closure.

These are named, actionable blockers, not permission for a generic SQL or method-dispatch RPC.
The static census establishes the built-in call surface below; it does not prove arbitrary plugins,
model-chosen tools, runtime rebinding, or dynamically loaded third-party code cannot open state.db.

## 1. Construction and complete turn ordering

1. `run_agent.py::AIAgent.__init__` delegates to `agent/agent_init.py::init_agent`.
   `_init_session_state` assigns the supplied store and session/parent ids; it does not create a row.
   `_memory_provider_init_kwargs` reads `get_session_title` (errors suppressed).
   `_build_context_engine` calls the engine's `bind_session_state`, which for the built-in compressor
   immediately reads cooldown, fallback streak, ineffective count, recovery deadline and prune runway.
   Store capability and profile identity must therefore exist **before construction**, not at first append.
   The constructor also creates ordinary request-log/checkpoint resources; these are not canonical DB writes.
2. `run_agent.py::_ensure_db_session` calls `create_session` with source/model/model_config/prompt,
   gateway origin fields, parent id, cwd and explicit profile. `_session_db_created` changes only on
   non-raising return; errors log and leave it retryable. `_insert_session_row` performs a keep-existing
   UPSERT with null backfill, parent metadata inheritance and content-addressed system-prompt storage.
   `create_session` is not permission to choose a new arbitrary session/profile over the wire: the
   owner reserves the execution's row/parent and validates any allowed mutable metadata.
3. `agent/turn_facade.py` invokes `admit_durable_turn_lease` before the synchronous conversation loop.
   `get_session` missing means fresh seed (no lease/reload); a read error means **try the lease**.
   Concrete-type `acquire_session_turn_lease` availability is inspected, not just instance getattr.
   Acquire precedes history load. Only if acquisition waited, resolve resume tip and reload messages
   with `repair_alternation=True, include_row_ids=True`; immediate acquisition preserves cached history.
   Lease holder/TTL are installed only on success. Failure returns zero-call interrupted/timeout result.
4. `conversation_loop::_restore_or_build_system_prompt` reads `get_session`; an existing stored
   system prompt is reused for history. `_persist_system_prompt` writes the assembled snapshot.
   `prompt_cache_scope` reads compression lineage, fork/source identity and durable conversation boundary;
   `_conversation_root_id`/`system_prompt::_session_start_like` read the attribution lineage root.
   These are different identities: delegate children MUST NOT inherit the parent's prompt-cache scope.
5. `turn_context` ensures the row before auto-title, loads/persists usage anchors, and stamps API-only
   user content through `set_latest_user_api_content` after a preflight compaction may already have
   inserted the user row. Match is newest active user row plus encoded content, not arbitrary row id.
6. `session_persistence::_db_flush_collect` owns serialization/filtering and `_db_flush_write` calls
   `append_messages_batch` once with structured rows and compression/turn holders. Keep all existing
   reasoning/Codex/display/platform/timestamp/api_content handling and ephemeral scaffolding filtering.
   `_DB_PERSISTED_MARKER` and flush cursors are stamped **after successful durable return**, never at
   send time. `_insert_message_rows` may enrich row dictionaries with persistence identities; a remote
   adapter must return/apply those row annotations before `sync_flushed_message_markers`, not only count.
   Tool-round and tool-executor flush failures stop progress; other exit paths also flush.
7. Main response usage (`turn_usage::record_response_usage`, `codex_runtime::_queue_token_counts`)
   ensures the row and queues per-route deltas; `_persist_session` appends then calls
   `flush_token_counts` at every persist point. Auxiliary context is installed around the turn and
   copied into auxiliary/title work. `aux_accounting::record_aux_usage` and
   `background_review::_record_review_usage_to_parent` write per-model/task usage to the **parent**;
   the review fork sets `_persist_disabled=True`, `_session_db=None`, and unbinds its compressor.
   It must never regain a writable DB through `_get_session_db_for_recall`.
8. Turn finally stops/cancels renewal timers, clears lease interrupt, releases the admitted holder, then clears activity labels;
   `DurableTurnLease.release` uses the original id while refresh uses the live compression tip. Existing
   lease-key resolution keeps them in one logical conversation. Delayed finally/aux/title callbacks
   need execution-generation fencing as well as a holder check; title rank alone is not fencing.
9. `run_agent.py::_finalize_owned_session_row` ends the live session unless explicitly disabled, then
   registry-releases if `_owns_session_db`. `close` must become local adapter release, not an RPC to
   close the owner's shared connection. Token barrier, session end, worker terminal settlement, and local scope release are distinct
   operations; terminal settlement must not race outstanding durable mutations. Preserve deferred close for
   still-running child/cron threads; a caller timeout does not stop a thread or undo its side effects.

## 2. Transaction and error contracts (apply to the operation index)

The exact signatures, source definitions and every detected core consumer are in §7. Defaults below
refer to the current local SessionDB, not promised remote behavior.

| Operation family | Return/error and transaction/ordering contract |
|---|---|
| Ordinary SessionDB writes | `_execute_write` owns `BEGIN IMMEDIATE`, callback result, commit and rollback. It retries the whole callback on classified busy/locked conditions, under bounded patience. IOERR is retried only when BEGIN failed before callback execution; once callback started settlement may be unknown. Inode replacement/corruption guards remain owner-side. Do not call a public self-committing method from inside the receipt transaction: that nests BEGIN and cannot atomically bind mutation to receipt. Extract/reuse its connection-taking body. |
| `append_messages_batch` | Returns inserted count, 0 for empty. One transaction when `chunk_rows=None` (the core path). Explicit chunking is multiple transactions and partial progress is possible. Reuses `resolve_and_repair_transcript_batch`, `_insert_message_rows`, counters and FTS triggers. Turn-holder loss raises `SessionTurnLeaseLostError`; ended compression parent raises `CompressionSessionClosedError`. Ordinary appends intentionally do NOT reject a foreign compression lock: watermark preservation makes concurrent appends safe. |
| Agent append wrapper | `True` success, `False` caught persistence failure, `None` when deliberately disabled/no DB. Closed compression parent can adopt a validated live tip and retry exactly once. Replacement/corruption currently diverts JSONL; managed-worker outage must use the private bounded outbox instead, never direct SQLite/fallback transcript adoption. An outbox ACK must not be confused with owner commit when a caller needs row ids, watermarks or committed history. |
| Session reads | `get_session` returns resolved row dict/None; drains token queue first but ignores a `False` flush timeout, so “exact totals” in its docstring is conditional. Other read errors normally propagate. `get_messages_as_conversation` returns a reconstructed list; repair is in-memory only, and model reads must not include compacted display history. Most lineage methods perform multiple reads, not a snapshot transaction. Compression adoption deliberately confirms tip again after loading. |
| Turn leases | `try_acquire_session_turn_lease` is an atomic lineage-root walk + claim/reclaim. `acquire_session_turn_lease` is a polling orchestrator, bool on success/abort/deadline; only classified locked SQLite errors retry, other errors propagate. Wait/status/abort callbacks execute locally and are not serializable RPC fields. Refresh returns holder-qualified bool, release returns None/idempotent. Renewal can revive still-owned expired rows, never a successor's row. |
| Compression leases | Try-acquire and refresh return False on SQLite error (not an authoritative free lock); release logs SQLite errors and returns None. Replaced-handle RuntimeError may still propagate. Compression lock is physical session keyed; turn lease is logical-conversation keyed. `get_compression_lock_holder` is diagnostic, not write authorization. |
| Compaction | `archive_and_compact` returns new active count; archives/reinserts/foreign-tail clone/model-config merge/counters share one transaction. `lock_holder` is checked there. Watermark captured before summary, `tail_count` prevents searchable duplicate carried tail. Guard loss raises `SessionCompressionInProgressError`; config-patch missing row raises. `micro_compaction` and prune-only use this too; do not support only LLM compaction. |
| Rotation | `publish_compression_child` returns None; parent close, child row, handoff rows, bounded foreign-tail clone and `advance_local_target` commit together. Lease absence/loss raises `CompressionSessionBusyError`; missing/deliberately ended parent or empty handoff raises RuntimeError. Automatic end stamps may be reopened inside that transaction. Follow-up title/provenance/prompt/engine state writes are currently separate; worker assignment must gain the authorized child atomically with publication before any of them. |
| Cooldown and anti-thrash | get active cooldown => dict/None (expiry-filtered); raw getter => exact `{session_exists,cooldown_until,error}`. Record merge-maxes deadline, latest diagnostic wins; clear removes both. Record/clear/streak/count/deadline setters use `_write_sql_logged`, swallowing SQLite errors. `restore_compression_failure_cooldown_row` intentionally propagates write/verification errors and rejects restoring a now-existing formerly absent row. A receipt must never certify swallowed failed mutation; reuse strict connection helpers, not these best-effort outer wrappers. Numeric reads use missing/non-numeric fallback, deadline is wall-clock, local waits are monotonic. |
| Usage queue/barrier | `queue_token_counts` returns None after memory enqueue; after writer stop uses synchronous update (can raise). `_apply_token_batch` logs and discards failed deltas. `flush_token_counts` returns bool/False timeout and does not expose prior dropped-delta failures. Adjacent incremental same-route coalescing only; never merge absolute totals or reorder route switches. For worker mode, durable outbox append precedes ACK; application and receipt must atomically include both session counters and per-model usage. Existing queue return is NOT durable proof. |
| Billing/aux | `update_token_counts` creates a missing row in a separate transaction, then updates counters and incremental model-usage in one transaction. First accounted route can override requested route; absolute updates do not duplicate per-model deltas. `record_auxiliary_usage` also ensures row separately then records per-(model,provider,task) usage only, NOT session totals; despite “best-effort” docstring its DB exceptions propagate to callers which catch them. `update_session_billing_route` flushes deltas before route update and clearing prompt hash/snapshot; its ignored False barrier must not become successful worker settlement. |
| Model sidecars | `patch_session_model_config` merges JSON atomically (None deletes; missing/empty no-op). Core keys are `_usage_anchor` and `_proactive_prune_rearm_tokens`; compaction includes the latter in the transcript transaction. `get_session_model_config_value` parses tolerantly with default. RPC must restrict keys to concrete supported sidecars, not arbitrary config mutation. `set_latest_user_api_content` returns 0/1, content-fenced UPDATE. `update_system_prompt` updates content-addressed prompt ownership; keep byte stability except existing permitted boundaries. |
| Titles | Get title/source returns str/None. `set_auto_title` accepts only derived/llm source, returns False if higher authority won; explicit `set_session_title` has user rank and returns bool, ValueError on validation/conflict. Provenance setter returns whether titled row changed. `get_next_title_in_lineage` searches all titles matching base, not a child-id lineage; lookup is not atomic reservation. Preserve dedupe retry and author rank, but expose an authorized title operation, not a profile-wide arbitrary title query. |
| Activity | Touch returns None, max timestamp/normalized bounded description/provenance, short write patience; throttled by caller. Clear preserves timestamp and avoids no-op write. Errors are caught by agent callers. For workers, stale generation must not clear successor labels or advertise a revived execution. |
| End/finalize | `end_session` returns None, first end reason wins and boundary counter update shares its transaction. It has no worker-generation check today. `session_lifecycle_statuses` returns id=>status from last row; cron downgrades only known interrupted/error/empty statuses and currently fails open on probe errors. Generation validation and final mutation/receipt must precede terminal worker status; late finalizers must be refused without mutating the next turn. |

### Compression's required sequence (do not collapse into append-text)

`_try_acquire_durable_lock` => acquire compression lease => capture active watermark and authoritative
raw cooldown under lease => optional grow/adopt durable history and preflush => slow summary/pruning
outside DB transaction => validate cancellation/lease => archive OR atomic publish-child => install
returned durable transcript markers and new session id => carry title/provenance, usage-anchor/runway,
anti-thrash state and context-engine boundary => release holder. Cancellation rollback uses exact raw
cooldown snapshot, not the expiry-filtered view. `_adopt_live_compression_child` verifies child alive,
loads conversation, then confirms the tip; `_reopen_orphaned_parent` conservatively refuses live child/lease.
No async fire-and-forget replacement can preserve these dependencies.

## 3. Concrete worker producers and extra consumers

### Cron

`_open_cron_session_db` => registry acquire on a copied-context helper thread (timeout continues with None;
late result closes). `_construct_cron_agent` passes that store before constructor. Main/watchdog thread
runs `run_conversation`; `_defer_cron_worker_teardown` in `scheduler_detached_worker.py` retains store and
agent until the timed-out future actually exits. `_finalize_cron_session` resolves compression tip,
sets deduped nonblank title BEFORE end, classifies last row, ends, disarms agent's redundant end, releases.
`_BoundedCronSessionDB.__getattr__` forwards **every attribute**, runs callable cleanup under timeout,
and a timed-out cleanup may still complete later. It is a local timeout shim, never the wire API.
Concrete method set is the core index plus `session_lifecycle_statuses`, title/tip helpers already indexed.

External scheduler adoption (`_run_external_worker_payload` / `adopt_claimed_execution`) is a cron
execution-store claim, not authority adoption. Keep its durable job claim, worker PID, cancel/watchdog,
job/context_from/script/workdir/provider/memory policy. A script-only job need not invent a model session.
Cron-open failure policy must change to visible pause/failure under managed mode; None silently disables
persistence and is not a valid managed capability. Cron claim/delivery stores are not to be reimplemented
by the SessionDB adapter.

### Children / grandchildren / schema retry

`delegate_tool::_build_child_agent` calls `_open_child_session_db`, deriving `parent_store.db_path`
and registry-acquiring it; missing parent DB disables persistence. Constructor failures release the handle.
On success `_owns_session_db=True`; child-run close (including deferred late close) owns that reference.
This shares a physical connection in one PID; it is a fresh writable open in another PID. The typed
replacement needs authority-reserved child identity before construction and inherited exact profile scope.
`delegate_tool_registry::_resolve_session_lineage` uses `resolve_resume_session_id` best-effort (input
unchanged on failure). `delegate_tool_child_run` runs schema correction through another real
`run_conversation` on the same child, not an independent unregistered agent. Apply the same rules to
nested orchestrators. Async completion ledger writes below remain independently reachable even when
the child transcript handle has been replaced.

### Compute host (existing legacy route, NOT the normal canonical turn path)

`HostSupervisor.submit_turn` => `ComputeHost._run_real_turn` => `_build_server_session` => profile
registry acquire => `agent_factory::_make_agent` => `_transfer_db_to_agent` => `_init_session` =>
`_ensure_session_db_row` => `_persist_branch_seed` => `prompt_turn::_run_prompt_submit` => synchronous
AIAgent in its run thread. It joins that thread, builds metadata/session.info, then emits turn.end.
Shutdown drains in-flight turns and skips still-live session ids before `_finalize_session`; that path
persists agent snapshot, emits hooks/commits memory, reads source, conditionally ends row, stops notification
poller/active-session slot and delegates. There is no authority-issued worker assignment in these frames.

Extra **known** SessionDB operations in this route (see §8 exact call-site index):

* `_hydrate_session_cwd` reads row; `_persist_session_cwd_and_schedule_git_meta` => `update_session_cwd`
  returns cwd generation, then `_persist_session_git_meta` => `publish_session_git_metadata` checks
  cwd+generation before updating branch/root. Async metadata must carry the same capability.
* `_ensure_session_db_row` => create + deferred `set_session_hidden`; `_persist_branch_seed` => chunked
  append (partial seed can exist). Owner must reserve/seed before launching, rather than duplicate this.
* `prompt_turn::_absorb_turn_result` => `set_latest_matching_message_display_kind` after response;
  `_after_complete_turn` writes pending title (drops invalid ValueError, retains transient failures).
* Model restore/switch => `agent_factory::_persist_live_session_runtime` reads row and uses
  `update_session_meta` (fallback `update_session_model`); server `_append_model_switch_marker`
  uses structured `append_message`; `_persist_live_session_system_prompt` writes prompt.
* TUI resume helpers => `assert_resume_safe`, `get_resume_conversations`,
  `get_ancestor_display_prefix`, `reopen_session`; display reads are not model-input history.
* `methods_prompt` reaction notes => `take_unseen_reactions`; truncate => `replace_messages`;
  retry/rewind => `get_active_message_ids`, `get_messages_as_conversation`, `rewind_to_message`.

`MUTATOR_ROUTE_TABLE` includes save/compress/truncate/model/personality/prompt/reset/history-reload/retry,
not just turn submit. `_control_ack` uses server methods for save/compress and `_SLASH_MIRRORS` for the
rest. `_init_session` starts `_start_notification_poller`; this is an independent producer, not a harmless
metadata mirror. **Do not transplant all legacy TUI helpers into the scoped core facade.** Parent should
select the authority execution adapter and keep session creation/metadata/rewind/control admission and
notification ownership in authority. If full legacy control parity remains in the external host, its
expanded exact helper closure must be converted and behavior-probed before enabling that route; the
core index alone is explicitly insufficient. Plugin/mirror rebinding makes automatic reachability
classification an overapproximation; §8 includes extra helpers, not a claim all run every turn.

## 4. Raw SQL and store-property blockers

### Async delegation ledger: exact operations, existing transaction semantics

All following are `tools/async_delegation.py`; `_db_path` is `get_hermes_home()/state.db`.
`_transaction` opens via `_connect` (schema/DDL durability setup), `with conn` commits/rolls back,
then ALWAYS closes. `_DB_LOCK` serializes locally only; this is not SessionDB `_execute_write` or a
worker receipt transaction. Do not forward `_update_delivery(sql, params)` remotely.

| Consumer => concrete operation | Return/error/order contract |
|---|---|
| `_dispatch` => `_persist_dispatch(record)` | None; INSERT OR REPLACE running row, owner PID/start, routing/task JSON, pending attempts=0. Persist before submitting runner; submit failure DELETEs ledger row. DB errors can reject dispatch. Separate `_prune_durable_records` transaction follows insert. Replay must not reset completed delivery state. |
| `_prune_durable_records` | None; delete old delivered and bound terminal/pending rows, preferring delivered. Owner-only retention; must not prune Q's pending outbox/receipts. |
| `_finalize` => `_push_completion_event` => `_persist_completion(event,result)` | None; update terminal state, event/result JSON, pending delivery before live queue publication. If persistence raises, live queue publication and the subsequent in-memory terminal status assignment are not reached. No atomic ledger+queue/admission guarantee today. |
| `delegate_tool_dispatch` => `record_unit_child(id,entry)` | None; within one local transaction read running result_json, replace same task_index, write partial results. Exceptions logged/suppressed. Preserve already-finished children across recovery; do not rerun missing/unknown children. |
| `recover_abandoned_delegations` | int changed; running/finalizing rows whose PID/start witness died become unknown, retaining completed partial results. Import failure returns 0; DB/JSON errors otherwise propagate. Owner-only producer recovery, not worker self-adoption. |
| `restore_undelivered_completions(queue)` | int enqueued; invokes recovery, orders pending events by completed_at/id, drops expired events, puts restored=True events on queue inside transaction. Queue effect cannot roll back with SQL. Canonical migration must bind admitted producer identity before ACK, never treat this as atomic admission. |
| `claim_completion_delivery(id,claim)` | bool; absent legacy row returns True; otherwise pending + no claim/stale claim => claim and increment attempts. Reclaim interval in source. Managed registered events must not inherit absent-row success as proof. |
| `release_completion_delivery(id,claim)` | bool row changed; matching pending claim => capped attempts becomes dropped, otherwise clears claim for retry; charged attempt retained. |
| `defer_completion_delivery(id,claim)` | bool row changed; matching pending claim => clear + refund one attempt (floor 0), for refusal before admission. |
| `drop_completion_delivery(id,claim)` | bool row changed; matching pending claim => dropped (not delivered), target permanently gone. |
| `complete_completion_delivery(id,claim)` | bool row changed; matching pending claim => delivered and timestamps, clear claim. |
| `mark_completion_delivered(id)` | bool row changed; legacy unqualified id ACK where not already delivered. Keep owner-only migration, not a worker capability. |
| `get_durable_delegation(id)` | dict/None; decodes result JSON after read. Uses writable/schema-creating `_transaction` even for this read. |
| `claim_event_delivery` / `_event_delivery` | Local routing wrappers. Interim task-failure notices deliberately bypass final-row claiming; claim string empty=no token needed, None=refused, nonempty=token. `_event_delivery(fn,...)` is not a wire function parameter. |

Implement named producer operations for dispatch, partial child result, terminal completion and scoped
status; delivery claim/defer/drop/complete belongs to automation owner. Reuse existing SQL bodies after
extracting connection-taking helpers; fence lineage/execution plus mutation/receipt atomically. This
crosses the child and automation lanes and must be assigned once by the parent.

### Recall, reactions, completed-process lookup, and DB properties

* Inline `session_search` receives `_get_session_db_for_recall()` and calls real
  `tools/session_search_tool.py`. Typed reads used: `get_session`, `resolve_session_by_title`,
  `get_messages`, `get_anchored_view`, `fts_rebuild_status`, `search_messages`,
  `list_recent_sessions_bounded`, `get_messages_around`. `_get_message_storage_state` additionally
  executes exactly `SELECT session_id, active, compacted FROM messages WHERE id = ?` via `_lock/_conn`;
  returns dict/None and suppresses lookup error. Its visibility decision distinguishes compacted archive
  from rewind-deleted rows. Add a specific owner-side storage-state query or run authorized recall in
  owner; never give workers `_conn`. `_locate_session_db`/explicit profile reads use read_only=True,
  but cross-profile discovery is a separate capability, not implied by execution write scope.
  FTS search may initiate index-repair paths on a writable handle; “it's only a search” is not proof of
  zero writable opens. Disabled recall in a full worker is a behavior regression, not automatic parity.
* `process_registry_results::_owns_result`: owner==parent shortcut, otherwise writable SessionDB,
  `get_compression_tip(parent)==owner`, finally close. Replace the lookup with scoped lineage read.
  Completed result JSON files themselves remain private process-result artifacts, not DB receipts.
* `react_to_message_tool`: independent registry open; `latest_message_row_id` or `get_message_role`
  then `set_message_reaction` returning reaction list/None. Tool converts store failures to tool_error,
  and emits a live reaction only after write. Capability must restrict exact session/message ownership.
* `agent/system_prompt::_agent_home` derives home from `db_path` as fallback; the exact
  function is included in the census source-access list. `delegate_tool::_open_child_session_db`
  interprets that same property as permission to acquire SQLite. Expose immutable assigned profile
  identity separately; `db_path` is metadata only if retained, never a worker-side opening mechanism.
* Registry `release_or_close` first identity-refcounts, then calls `close` for unknown objects.
  A scoped proxy's local close may release transport/outbox handles but must not close shared owner DB.
* `type(db)` capability probes in leases/compression require concrete explicit adapter methods;
  instance-only `__getattr__` looks supported to some consumers and absent to others. Compressor's
  `_durable_read/_durable_write/_load_durable` dynamic method strings are fully enumerated in §7.
  Cron's bounded `__getattr__` stays a local shim. No generic remote method API is justified.
* `agent/insights.py` holds `db._conn` and issues analytics SQL, while `agent/trace_upload.py` opens/reads
  sessions for explicit trace export. These are census candidates outside the normal AIAgent turn;
  they are not to be exposed as execution-scoped operations. A plugin invoking them changes the closure.
* `agent/transcript_repair.py` receives an owner transaction connection from the append implementation;
  its SQL is already in the correct process. Move/reuse its call, not its connection, at the wire boundary.
  Append's returned row annotations include both `_row_id` and `_canonical_content` (concurrent-winner
  adoption); compaction/publication also mutate their input row dictionaries with `_row_id`. Preserve
  these post-commit effects across the RPC, not only their scalar return values.
* The remaining raw-SQL census candidates in `agent/verification_evidence.py` use the separate
  `get_hermes_home()/verification_evidence.db`, NOT canonical state.db. Keep this coding-evidence
  ledger distinct from execution persistence. The `.execute` hits in `prompt_builder::_run_backend_probe`,
  `tool_executor::_run_agent_tool_execution_middleware`, and `turn_api_call::perform_api_call`
  are environment/relay execution calls, not SQLite. These false positives are classified, not ignored.

## 5. Smallest closed typed seam and thread/RPC design

**Facade:** preserve the explicit core methods in §7 for unchanged built-in AIAgent/compressor callers;
implement `acquire_session_turn_lease` locally as a poll loop over a named try-acquire primitive.
Do not mirror all of SessionDB. Recall, delegation-ledger, reactions and legacy compute control remain
separate scoped services with concrete consumers above. `create_session`'s kwargs and queued usage kwargs
must become enumerated payload fields (see `_insert_session_row` / `update_token_counts`), not open bags.
Map core model-config access to two concrete sidecar keys; reject all other keys until a named consumer
is assigned. `set_auto_title_if_empty` is a legacy fallback in title_generator, absent on current SessionDB;
it does not need a new remote method when `set_auto_title` is present. Likewise
`is_explicit_fork_child` is the compatibility branch when `declared_scope_identity` is unavailable;
a concrete current adapter can derive the former from the latter's typed result rather than add a
second wire operation. The source census enumerates fallback consumers without mandating redundant RPCs.

**Closed wire families (each variant gets explicit fields/results, never `{method,args,kwargs}`):**

1. Assigned session create/backfill, row/history read and authorized lineage/cache-identity reads.
2. Structured transcript append returning count AND input-row persistence annotations; API-content stamp.
3. Turn try-acquire/renew/release; local wait callbacks. Compression try-acquire/renew/release/watermark.
4. In-place compact and publish compression continuation (authorized child reservation + assignment
   advancement in the same transaction). Exact cooldown snapshot/restore; explicit counters/deadline.
5. Main usage delta, auxiliary usage delta, receipt barrier, billing-route switch; typed usage-anchor
   and prune-runway read/write, system-prompt snapshot.
6. Ranked title apply/read, activity touch/clear, lifecycle classification/end, terminal execution finish.

This is the minimal closed **core** set, not a statement that six generic RPCs are sufficient. Keep
separate variants where inputs/results/authorization differ; a tagged closed union is permissible,
caller-selected SQL or Python method names are not. Snapshotting constructor reads can reduce trips,
but refreshing tip/cooldown/history after a lease wait must still cross to authority. A cached missing
row or cooldown is not authoritative after an outage. Need no new “session manager” parallel to authority.

**Same PID:** synchronous methods can call the authority-owned store's transaction path directly from
AIAgent/cron/child threads with explicit capability checks. A gateway/API event-loop thread must never
submit an RPC to itself and block on the reply. DB waits/flushes and lease polling cannot monopolize the
loop; schedule off-loop where called from async ingress. Sharing SessionDB does not automatically supply
execution authorization; use the same fenced mutation bodies for direct and remote variants.

**External process:** synchronous facade waits for a dedicated transport receiver, not the model/tool
thread and not a blocked owner loop. The receiver must process control/cancellation and ACK concurrently
with a blocked persistence call. Lease refresh scheduler, compression refresher, title thread, auxiliary
review and child threads all share the facade; serialize mutation sequence under one lock, persist the
private byte-bounded outbox before send, retain until receipt, and keep pending entries immune to cleanup.
Reads needing prior writes require an explicit receipt barrier. No secret/profile env fallback or SQLite
open on transport failure. Full/unavailable/revoked are distinguishable outcomes; caught “best-effort”
callers still require a sticky runtime pause/fail signal so they cannot continue tools after losing
persistence. Do not classify an outage as missing method, missing row or unheld compression lock.

Existing worker primitives validate epoch, assignment(session,generation), owner epoch, consecutive
sequence and payload fingerprint. A duplicate returns stored result before terminal-state refusal;
conflict rejects. Adoption validates secret and current generation; **caller must verify original
producer claim and live worker proof** (the existing docstring explicitly delegates this). New owner
epoch does not imply replay permission. The primitive currently has only text receipts: use common
private connection helpers to extend structured variants, with strict exceptions and one final commit.
Do not enqueue normal token-writer work and commit a success receipt before usage is applied.

## 6. Exclusive implementation handoff and acceptance blockers

* **Substrate lane:** `gateway/session_worker.py`, `agent/runtime_session_store.py`,
  `hermes_state_runtime.py`, `agent/session_persistence.py`, `agent/turn_facade_lease.py`,
  `tests/gateway/test_worker_persistence.py`. Request exclusive extraction seams in
  `hermes_state_messages.py`, `hermes_state_usage.py`, `hermes_state_compression.py`,
  `hermes_state_sessions.py`, `hermes_state_titles.py` for connection-taking mutation bodies.
  No maintenance/backup/recovery ownership changes. Receipt atomicity cannot be implemented correctly
  by nesting existing public methods; these extraction seams are prerequisite ownership, not optional.
* **Parent-exclusive:** authority lifecycle/epoch, authenticated worker bootstrap/dispatch, execution
  reservation and canonical compute adapter selection. Reserve a continuation/child before inference
  and atomically carry assignment over compression. Define terminal ordering relative to parent admission.
* **Cron lane:** injection before constructor, restart claim proof, no None fallback, deferred-finalize
  ordering and scoped adapter close; existing scheduler/external-worker lifecycle remains.
* **Child/automation boundary:** one owner for `tools/async_delegation.py` SQL extraction and routing;
  child lane supplies dispatch/partial/final event identity, automation lane owns claim/admission/settle.
  No full worker switch until this direct writer and process-result/reaction opens are addressed.
* **Compute lane:** replace `_get_db`/`_workdir_owner_db` bypasses and server-started notification/control
  ownership, not only `compute_host::_build_server_session`'s first acquire. Extend typed operations
  only for controls explicitly retained in external host. Profile scope must reach all helper threads.
* **Read-tool lane/parent:** authorize recall and storage-state query separately from session writes;
  preserve compacted-vs-rewind filtering, profile boundaries and result semantics.

Q2–Q8 remain unexecuted here: actual process zero writable FDs/opens, lost ACK replay/conflict, forged
profile/session/generation, outbox full/unavailable, restart adoption without rerunning tool, stale finally,
compression annotation round-trip, route/aux accounting exactly once, constructor sidecars, schema retry,
legacy compute control isolation and loop liveness. Native behavior is not proven by this source inventory.
Core consumers that currently catch/suppress errors, custom context engines/plugins, runtime tool choice,
background timeout races and SQL settlement after process death need executable behavior probes; an AST
cannot certify those. The existing broad gateway suite does not substitute for these worker cases.

## 7. Exact core consumer => operation index

Generated from the executable census in §9; consumer symbols are stable references, with diagnostic
snapshot line numbers. Each signature is the actual source signature (not a proposed open RPC schema).
Source docstrings are quoted to preserve default/return intent; §2 records important differences between
that intent and implementation (notably usage barriers and swallowed errors). Duplicate occurrences in
the same function are represented once. Helper-only `try_acquire_session_turn_lease` and
`update_token_counts` are included in §2 as transitive implementation seams, not extra direct consumers.

### `acquire_session_turn_lease`

Definition: `hermes_state_compression.py::acquire_session_turn_lease`

```python
acquire_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float=300.0, wait_seconds: float=1800.0, poll_interval_seconds: float=1.0, on_wait=None, wait_notice_interval_seconds: float=15.0, should_abort=None, acquire_patience_s: float=0.5) -> bool
```

Source contract: Wait for a cross-process turn lease without holding a SQLite lock. ``on_wait(elapsed)`` is best-effort: called when the first attempt fails and about every ``wait_notice_interval_seconds`` after. ``should_abort()`` True (e.g. ``/stop``) returns False at once.

Consumers:
- `agent/turn_facade_lease.py::admit_durable_turn_lease` (snapshot 251)
- `agent/turn_facade_lease.py::admit_durable_turn_lease` (snapshot 271)

### `append_messages_batch`

Definition: `hermes_state_messages.py::append_messages_batch`

```python
append_messages_batch(self, session_id: str, messages: List[Dict[str, Any]], compression_lock_holder: Optional[str]=None, turn_lease_holder: Optional[str]=None, chunk_rows: Optional[int]=None, turn_lease_ttl_seconds: float=300.0) -> int
```

Source contract: Append *messages* in ONE write txn (all rows land or none, guards run once); returns the inserted count. ``chunk_rows`` bounds txn size for LARGE copies (branch seeds; FTS triggers run per row).

Consumers:
- `agent/session_persistence.py::_db_flush_write` (snapshot 208)

### `archive_and_compact`

Definition: `hermes_state_messages.py::archive_and_compact`

```python
archive_and_compact(self, session_id: str, compacted_messages: List[Dict[str, Any]], model_config_patch: Optional[Dict[str, Any]]=None, watermark: Optional[int]=None, lock_holder: Optional[str]=None, tail_count: int=0) -> int
```

Source contract: Non-destructive in-place compaction under ONE session id: soft-archive the active rows (``active=0, compacted=1``: summarized away, still searchable) and insert *compacted_messages* as fresh active rows, atomically; returns the new ACTIVE count (= ``message_count``). *watermark* (compression START): rows ``id > watermark`` arrived during the slow summary and are re-sequenced after the compacted set by a pure-SQL clone (fresh ids); ``None`` archives everything. *lock_holder*: verified in-txn so a reclaimed lease fails instead of clobbering the winner. *tail_count*: the LAST N compacted rows are the verbatim carried tail; their originals and the clones' originals are superseded duplicates and get rewind flags (``active=0, compacted=0``) so search doesn't return each carried message once per compaction. ``model_config_patch`` merges in the same txn (``None`` removes a key).  Concurrent-append safety (#75316): when *watermark* is provided (the value of :meth:`get_active_message_watermark` captured at compression START), rows that arrived during the slow provider summary call (``id > watermark``) are NOT summarized away. They are re-sequenced after the compacted set by a pure-SQL column clone (every column except ``id`` — content, api_content, platform_message_id, token counts, reasoning sidecars all survive byte-exact, and the FTS triggers index the clones naturally), and the originals are archived. NOTE: re-sequencing assigns the tail rows fresh ids; consumers that reference durable row ids re-resolve by content (see 3e8ab0610).

Consumers:
- `agent/context_compressor.py::prune_tool_results_only` (snapshot 2857)
- `agent/context_compressor.py::prune_tool_results_only` (snapshot 2878)
- `agent/conversation_compression.py::_commit_compaction` (snapshot 3260)
- `agent/micro_compaction.py::_sync_micro_compact_to_db` (snapshot 352)

### `clear_compression_failure_cooldown`

Definition: `hermes_state_compression.py::clear_compression_failure_cooldown`

```python
clear_compression_failure_cooldown(self, session_id: str) -> None
```

Source contract: Clear any persisted compression-failure cooldown for a session.

Consumers:
- `agent/context_compressor.py::_clear_compression_failure_cooldown` (snapshot 2126)
- `agent/conversation_compression.py::_rollback_durable_cooldown` (snapshot 281)

### `clear_session_activity_labels`

Definition: `hermes_state_sessions.py::clear_session_activity_labels`

```python
clear_session_activity_labels(self, session_id: str) -> None
```

Source contract: Clear activity labels after a turn (``last_activity_at`` is kept so idle / watchdog clocks stay continuous). A no-op clear skips the write transaction.  Description and provenance are observation labels for *what was happening at* that timestamp during an active turn; once the turn is idle they must not keep advertising "compressing" / "executing tool" (#72039). Response-critical-path contract (#76354 review S1): runs in the turn's ``finally``; a no-op clear (labels already empty) skips the write transaction entirely, and a real clear uses the same short sub-second busy budget as :meth:`touch_session_activity` instead of the full routine write patience.

Consumers:
- `agent/activity_tracking.py::_reset_activity_labels_after_turn` (snapshot 134)
- `agent/conversation_compression.py::_finish_compaction_boundary` (snapshot 3057)

### `create_session`

Definition: `hermes_state_sessions.py::create_session`

```python
create_session(self, session_id: str, source: str, **kwargs) -> str
```

Source contract: Create (upsert) a session record. Returns the session_id.

Consumers:
- `run_agent.py::_ensure_db_session` (snapshot 337)

### `declared_scope_identity`

Definition: `hermes_state_sessions.py::declared_scope_identity`

```python
declared_scope_identity(self, session_id: str) -> Tuple[bool, str]
```

Source contract: (is_fork_child, source) in ONE read (prompt_cache_scope needs both from the same row). Missing row → (False, ""); DB errors propagate (fail closed).  ``agent/prompt_cache_scope.py`` needs both to resolve a host-declared conversation scope, and both live on the same ``sessions`` row; asking for them separately read that row twice per resolution (@teknium1 on 98811). The marker rules stay here, beside :meth:`is_explicit_fork_child`, instead of being re-implemented by the caller. See #98811.

Consumers:
- `agent/prompt_cache_scope.py::declared_conversation_scope` (snapshot 109)

### `end_session`

Definition: `hermes_state_sessions.py::end_session`

```python
end_session(self, session_id: str, end_reason: str) -> None
```

Source contract: Mark a session ended; the first end_reason wins (a compression split must keep ``'compression'`` even if a stale end_session() lands later); reopen_session() to re-end.

Consumers:
- `cron/scheduler.py::_finalize_cron_session` (snapshot 1984)
- `run_agent.py::_finalize_owned_session_row` (snapshot 985)

### `flush_token_counts`

Definition: `hermes_state_usage.py::flush_token_counts`

```python
flush_token_counts(self, timeout: float=5.0) -> bool
```

Source contract: Block until every queued token delta has been applied. False on timeout (callers then read totals stale by the queued deltas). Never raises.

Consumers:
- `agent/session_persistence.py::_persist_session` (snapshot 302)

### `get_active_message_watermark`

Definition: `hermes_state_messages.py::get_active_message_watermark`

```python
get_active_message_watermark(self, session_id: str) -> int
```

Source contract: MAX(id) of the active rows (0 if none), captured at compression START: every active row above it arrived concurrently and must survive compaction verbatim.

Consumers:
- `agent/conversation_compression.py::_publish_rotated_compaction` (snapshot 2952)
- `agent/conversation_compression.py::_try_acquire_durable_lock` (snapshot 2382)

### `get_compression_failure_cooldown`

Definition: `hermes_state_compression.py::get_compression_failure_cooldown`

```python
get_compression_failure_cooldown(self, session_id: str) -> Optional[Dict[str, Any]]
```

Source contract: Return the active (unexpired) compression-failure cooldown, or None.

Consumers:
- `agent/context_compressor.py::get_active_compression_failure_cooldown` (snapshot 2062)

### `get_compression_failure_cooldown_row`

Definition: `hermes_state_compression.py::get_compression_failure_cooldown_row`

```python
get_compression_failure_cooldown_row(self, session_id: str) -> Dict[str, Any]
```

Source contract: Exact stored cooldown columns, no expiry filtering, so compression cancellation can roll back an expired, partially-null, or absent row exactly.

Consumers:
- `agent/conversation_compression.py::_capture_authoritative_cooldown_under_lease` (snapshot 345)

### `get_compression_fallback_streak`

Definition: `hermes_state_compression.py::get_compression_fallback_streak`

```python
get_compression_fallback_streak(self, session_id: str) -> int
```

Source contract: Return the persisted deterministic-fallback streak.

Consumers:
- `agent/context_compressor.py::_load_fallback_compression_streak` (snapshot 1954)
- `agent/context_compressor.py::on_session_start` (snapshot 1896)

### `get_compression_ineffective_count`

Definition: `hermes_state_compression.py::get_compression_ineffective_count`

```python
get_compression_ineffective_count(self, session_id: str) -> int
```

Source contract: Persisted ineffective-compaction strike count — the durable half of the built-in compressor's anti-thrash guard, so a fresh compressor bound to a resumed session inherits an armed/tripped guard across restarts.

Consumers:
- `agent/context_compressor.py::_load_ineffective_compression_count` (snapshot 1972)
- `agent/context_compressor.py::on_session_start` (snapshot 1899)

### `get_compression_lineage`

Definition: `hermes_state_compression.py::get_compression_lineage`

```python
get_compression_lineage(self, session_id: str) -> List[str]
```

Source contract: Return compression ancestors through tip in chronological order.

Consumers:
- `agent/prompt_cache_scope.py::_lineage_root` (snapshot 27)

### `get_compression_lock_holder`

Definition: `hermes_state_compression.py::get_compression_lock_holder`

```python
get_compression_lock_holder(self, session_id: str) -> Optional[str]
```

Source contract: Current (non-expired) holder for ``session_id``, or None. Diagnostic only.

Consumers:
- `agent/conversation_compression.py::_sit_out_lock_contention` (snapshot 2413)
- `agent/conversation_compression.py::recover_rotated_compression_session` (snapshot 1461)

### `get_compression_recovery_deadline`

Definition: `hermes_state_compression.py::get_compression_recovery_deadline`

```python
get_compression_recovery_deadline(self, session_id: str) -> float
```

Source contract: Persisted anti-thrash recovery deadline (epoch; ``0.0`` = not armed). Durable because the gateway rebuilds the compressor every turn / cache eviction.  The deadline is the durable half of the 14694 recovery clock: the gateway rebuilds the compressor on every turn / cache eviction, so a process-local deadline restarted the wait on each rebuild and a tripped session never earned its probe (#100185).

Consumers:
- `agent/context_compressor.py::_load_anti_thrash_recovery_deadline` (snapshot 1982)

### `get_compression_tip`

Definition: `hermes_state_compression.py::get_compression_tip`

```python
get_compression_tip(self, session_id: str) -> Optional[str]
```

Source contract: Live tip of a compression chain (``get_compression_chain`` semantics); the input id when no continuation exists.

Consumers:
- `agent/conversation_compression.py::_adopt_live_compression_child` (snapshot 1389)
- `agent/session_persistence.py::_db_flush_adopt_compression_tip` (snapshot 222)
- `cron/scheduler.py::_finalize_cron_session` (snapshot 1921)

### `get_conversation_root`

Definition: `hermes_state_messages.py::get_conversation_root`

```python
get_conversation_root(self, session_id: str) -> str
```

Source contract: ROOT id of the lineage: the stable conversation id across compression segments and delegate subagents (Nous Portal usage tagging). Unchanged when there is no recorded parent.

Consumers:
- `agent/system_prompt.py::_session_start_like` (snapshot 209)
- `agent/title_generator.py::auto_title_session` (snapshot 373)
- `run_agent.py::_conversation_root_id` (snapshot 1333)

### `get_messages_as_conversation`

Definition: `hermes_state_messages.py::get_messages_as_conversation`

```python
get_messages_as_conversation(self, session_id: str, include_ancestors: bool=False, include_inactive: bool=False, repair_alternation: bool=False, include_row_ids: bool=False, include_compacted: bool=False) -> List[Dict[str, Any]]
```

Source contract: Load messages in OpenAI format. ``include_compacted`` (deduped display history) is for DISPLAY reads only: the model-fed restore must not regrow what compaction summarized away. ``repair_alternation`` repairs the loaded list for LIVE REPLAY callers (a durable ``user;user`` pair would re-trigger the per-request repair forever); the stored transcript is never mutated.

Consumers:
- `agent/conversation_compression.py::_adopt_grown_durable_parent` (snapshot 2551)
- `agent/conversation_compression.py::_adopt_live_compression_child` (snapshot 1391)
- `agent/turn_facade_lease.py::admit_durable_turn_lease` (snapshot 292)

### `get_next_title_in_lineage`

Definition: `hermes_state_titles.py::get_next_title_in_lineage`

```python
get_next_title_in_lineage(self, base_title: str) -> str
```

Source contract: Next title in a lineage ("my session" -> "my session #2"): strip any " #N" suffix, then increment the highest existing number.

Consumers:
- `agent/title_generator.py::_persist_session_title` (snapshot 329)
- `cron/scheduler.py::_finalize_cron_session` (snapshot 1951)
- `cron/scheduler.py::_set_cron_session_title` (snapshot 90)

### `get_session`

Definition: `hermes_state_sessions.py::get_session`

```python
get_session(self, session_id: str) -> Optional[Dict[str, Any]]
```

Source contract: Get a session by ID (drains queued token deltas first so cost readers see exact totals).

Consumers:
- `agent/conversation_compression.py::_adopt_live_compression_child` (snapshot 1390)
- `agent/conversation_compression.py::_parent_deliberately_ended` (snapshot 2886)
- `agent/conversation_compression.py::_session_was_rotated_by_compression` (snapshot 1178)
- `agent/conversation_loop.py::_restore_or_build_system_prompt` (snapshot 652)
- `agent/prompt_cache_scope.py::_agent_source` (snapshot 50)
- `agent/session_persistence.py::_db_flush_adopt_compression_tip` (snapshot 229)
- `agent/turn_facade_lease.py::_durable_session_exists` (snapshot 217)

### `get_session_model_config_value`

Definition: `hermes_state_sessions.py::get_session_model_config_value`

```python
get_session_model_config_value(self, session_id: str, key: str, default: Any=None) -> Any
```

Source contract: Read one key out of a session's model_config JSON (tolerant parse).

Consumers:
- `agent/context_compressor.py::_load_proactive_prune_rearm_tokens` (snapshot 1959)
- `agent/usage_anchor.py::persisted_anchor_tokens` (snapshot 157)
- `agent/usage_anchor.py::restore_usage_anchor` (snapshot 138)

### `get_session_title`

Definition: `hermes_state_titles.py::get_session_title`

```python
get_session_title(self, session_id: str) -> Optional[str]
```

Source contract: Get the title for a session, or None.

Consumers:
- `agent/agent_init.py::_memory_provider_init_kwargs` (snapshot 1217)
- `agent/conversation_compression.py::_publish_rotated_compaction` (snapshot 2964)
- `agent/conversation_loop.py::_bot_chat_prompt_stale` (snapshot 618)
- `agent/system_prompt.py::_bot_mode_parts` (snapshot 325)
- `agent/title_generator.py::_has_upgraded_title` (snapshot 296)
- `agent/title_generator.py::_session_is_untitled` (snapshot 410)

### `get_session_title_source`

Definition: `hermes_state_titles.py::get_session_title_source`

```python
get_session_title_source(self, session_id: str) -> Optional[str]
```

Source contract: Get the provenance of a session's title, or None when untitled.

Consumers:
- `agent/conversation_compression.py::_carry_session_state_to_child` (snapshot 2919)
- `agent/title_generator.py::_has_upgraded_title` (snapshot 293)

### `is_explicit_fork_child`

Definition: `hermes_state_messages.py::is_explicit_fork_child`

```python
is_explicit_fork_child(self, session_id: str) -> bool
```

Source contract: Read-only :meth:`_is_explicit_fork_child_row`; a missing row is not a fork (prompt_cache_scope keeps a declared conversation key from crossing the fork boundary).

Consumers:
- `agent/prompt_cache_scope.py::declared_conversation_scope` (snapshot 113)

### `latest_conversation_boundary`

Definition: `hermes_state_messages.py::latest_conversation_boundary`

```python
latest_conversation_boundary(self, session_key: str, source: str) -> Optional[int]
```

Source contract: Conversation boundaries (``_RESET_END_REASONS`` ends) this peer has crossed, or ``None`` if never reset. The peer is ``(session_key, source)``, never the key alone (an API caller may legally reuse a Telegram row's key). Read from ``conversation_generations`` (advanced in each boundary's txn), not an aggregate over session rows: deletes/prunes would re-emit a retired pair. Rows are never GC'd (dropping one re-issues generation 1: the ABA this prevents). Wall-clock-free, so a backwards NTP correction cannot reorder it. DBs upgraded mid-conversation take their first generation from the next boundary written (a pre-upgrade reset shares its predecessor's scope once: costs a warm prompt-cache bucket, never crosses an identity).

Consumers:
- `agent/prompt_cache_scope.py::_conversation_generation` (snapshot 79)

### `patch_session_model_config`

Definition: `hermes_state_sessions.py::patch_session_model_config`

```python
patch_session_model_config(self, session_id: str, patch: Dict[str, Any]) -> None
```

Source contract: Merge ``patch`` into model_config atomically (``None`` removes a key); no-op when the row or patch is empty.

Consumers:
- `agent/context_compressor.py::_clear_durable_proactive_prune_rearm` (snapshot 1965)
- `agent/usage_anchor.py::persist_usage_anchor` (snapshot 115)

### `publish_compression_child`

Definition: `hermes_state_compression.py::publish_compression_child`

```python
publish_compression_child(self, *, parent_session_id: str, child_session_id: str, source: str, messages: List[Dict[str, Any]], model: str=None, model_config: Dict[str, Any]=None, system_prompt: str=None, cwd: str=None, profile_name: str=None, compression_lock_holder: str=None, require_compression_lease: bool=True, require_lease_refresh: bool=False, lease_ttl_seconds: float=300.0, watermark: Optional[int]=None, watermark_ceiling: Optional[int]=None) -> None
```

Source contract: Atomically close a parent and publish its durable compression child: closure, child row, and handoff commit in one transaction, so readers see the live parent or a complete child, never an ended parent with a missing/empty child. *watermark* (parent's ``get_active_message_watermark`` at compression start): parent rows with ``id > watermark`` — appends landed during the slow summary — are column-cloned into the child AFTER the handoff. *watermark_ceiling* bounds the clone: the rotation path flushes its OWN transcript to the parent just before publishing and those rows are already in the handoff, so only ``(watermark, watermark_ceiling]`` is foreign tail (``None`` = unbounded). *require_lease_refresh* + *compression_lock_holder* refreshes the lease on the same ``conn`` before the expiry check (no TOCTOU window), so a refresher that died on transient DB errors gets one last chance.  See #75316. ``None`` = unbounded (no internal flush happened). See #47202.

Consumers:
- `agent/conversation_compression.py::_publish_rotated_compaction` (snapshot 2967)

### `queue_token_counts`

Definition: `hermes_state_usage.py::queue_token_counts`

```python
queue_token_counts(self, session_id: str, **kwargs) -> None
```

Source contract: Enqueue a token/cost delta for the background writer (same kwargs as :meth:`update_token_counts`). After close() stopped the writer, falls back to the synchronous path and may raise.

Consumers:
- `agent/codex_runtime.py::_queue_token_counts` (snapshot 80)
- `agent/turn_usage.py::record_response_usage` (snapshot 249)

### `record_auxiliary_usage`

Definition: `hermes_state_usage.py::record_auxiliary_usage`

```python
record_auxiliary_usage(self, session_id: str, task: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None, input_tokens: int=0, output_tokens: int=0, cache_read_tokens: int=0, cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None, api_call_count: int=1) -> None
```

Source contract: Record an auxiliary LLM call's usage (vision, compression, title generation, ...) as a per-(model, provider, task) delta in ``session_model_usage`` WITHOUT touching the ``sessions`` summary row (the gateway overwrites those counters with absolute main-loop totals). ``api_call_count`` may aggregate N calls. Best-effort.  See #23270. Background-review forks record an aggregate of N fork API calls in one write with ``task='background_review'`` (issue #87250).

Consumers:
- `agent/aux_accounting.py::record_aux_usage` (snapshot 84)
- `agent/background_review.py::_record_review_usage_to_parent` (snapshot 730)

### `record_compression_failure_cooldown`

Definition: `hermes_state_compression.py::record_compression_failure_cooldown`

```python
record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: Optional[str]=None) -> None
```

Source contract: Persist the active compression-failure cooldown. Merge-max with any longer live deadline so a later shorter write can't reopen the thrash window; error always takes the latest diagnostic.

Consumers:
- `agent/context_compressor.py::_record_compression_failure_cooldown` (snapshot 2110)
- `agent/conversation_compression.py::_rollback_durable_cooldown` (snapshot 277)

### `refresh_compression_lock`

Definition: `hermes_state_compression.py::refresh_compression_lock`

```python
refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float=300.0) -> bool
```

Source contract: Extend the compression lock lease if ``holder`` still owns it. Ownership is decided by ``holder`` alone, deliberately NOT ``expires_at``: a live owner whose refresher stalled past its TTL must be able to revive its still-unclaimed row, otherwise it keeps compressing with no lease — the window in which a competing path can fork the lineage. It cannot resurrect a lock someone else took: SQLite serialises writes, so the reclaim (DELETE-expired + INSERT OR IGNORE) never interleaves with this UPDATE.

Consumers:
- `agent/conversation_compression.py::_run` (snapshot 1682)

### `refresh_session_turn_lease`

Definition: `hermes_state_compression.py::refresh_session_turn_lease`

```python
refresh_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float=300.0) -> bool
```

Source contract: Extend a turn lease only while ``holder`` still owns it.

Consumers:
- `agent/turn_facade_lease.py::refresh_tick` (snapshot 184)

### `release_compression_lock`

Definition: `hermes_state_compression.py::release_compression_lock`

```python
release_compression_lock(self, session_id: str, holder: str) -> None
```

Source contract: Release the compression lock iff we own it; idempotent when gone/reclaimed.

Consumers:
- `agent/conversation_compression.py::_try_acquire_durable_lock` (snapshot 2397)
- `agent/conversation_compression.py::release_holder_only` (snapshot 2326)

### `release_session_turn_lease`

Definition: `hermes_state_compression.py::release_session_turn_lease`

```python
release_session_turn_lease(self, session_id: str, holder: str) -> None
```

Source contract: Release a turn lease iff ``holder`` still owns it; idempotent.

Consumers:
- `agent/turn_facade_lease.py::release` (snapshot 100)

### `reopen_orphaned_compression_session`

Definition: `hermes_state_compression.py::reopen_orphaned_compression_session`

```python
reopen_orphaned_compression_session(self, session_id: str) -> bool
```

Source contract: Reopen a compression parent only when no continuation was published (older builds could leave a closed parent after an interrupted handoff). Conservative: an active lease or any canonical child means another path owns the lineage.

Consumers:
- `agent/conversation_compression.py::_reopen_orphaned_parent` (snapshot 1440)

### `resolve_resume_session_id`

Definition: `hermes_state_messages.py::resolve_resume_session_id`

```python
resolve_resume_session_id(self, session_id: str) -> str
```

Source contract: Redirect a resume target to the descendant holding the messages: follow the compression chain to the live tip (lineage-aware, so delegate/branch children never hijack it), then walk ``parent_session_id`` forward to the DEEPEST node with messages (a continuation may hold newer turns), skipping branch/delegate/reset/tool children. Unchanged when nothing has messages; depth cap 32.  Context compression ends the current session and forks a new child session (linked via ``parent_session_id``). The flush cursor is reset, so the child is where new messages actually land — the parent ends up with ``message_count = 0`` rows unless messages had already been flushed to it before compression. See #15000.

Consumers:
- `agent/turn_facade_lease.py::admit_durable_turn_lease` (snapshot 288)
- `tools/delegate_tool_registry.py::_resolve_session_lineage` (snapshot 203)

### `restore_compression_failure_cooldown_row`

Definition: `hermes_state_compression.py::restore_compression_failure_cooldown_row`

```python
restore_compression_failure_cooldown_row(self, session_id: str, snapshot: Dict[str, Any]) -> None
```

Source contract: Restore and verify an exact cooldown-row snapshot. Unlike record/clear this rollback API propagates write and verification failures: cancellation must not be reported mutation-free when compensation failed.

Consumers:
- `agent/conversation_compression.py::_rollback_durable_cooldown` (snapshot 268)

### `session_lifecycle_statuses`

Definition: `hermes_state_sessions.py::session_lifecycle_statuses`

```python
session_lifecycle_statuses(self, session_ids: List[str]) -> Dict[str, str]
```

Source contract: ``{session_id: status}`` from each session's LAST message row (``'empty'`` when none); one query, MAX(id) per session joined back — never scans transcripts.

Consumers:
- `cron/scheduler.py::_finalize_cron_session` (snapshot 1973)

### `set_auto_title`

Definition: `hermes_state_titles.py::set_auto_title`

```python
set_auto_title(self, session_id: str, title: str, *, source: str) -> bool
```

Source contract: Set an automatic title; False (untouched) when a higher-authority title holds the row.

Consumers:
- `agent/title_generator.py::_persist_session_title` (snapshot 311)

### `set_compression_fallback_streak`

Definition: `hermes_state_compression.py::set_compression_fallback_streak`

```python
set_compression_fallback_streak(self, session_id: str, streak: int) -> None
```

Source contract: Persist the deterministic-fallback streak for one session.

Consumers:
- `agent/context_compressor.py::_persist_fallback_compression_streak` (snapshot 1968)

### `set_compression_ineffective_count`

Definition: `hermes_state_compression.py::set_compression_ineffective_count`

```python
set_compression_ineffective_count(self, session_id: str, count: int) -> None
```

Source contract: Persist the ineffective-compaction strike count for one session.

Consumers:
- `agent/context_compressor.py::_persist_ineffective_compression_count` (snapshot 1975)

### `set_compression_recovery_deadline`

Definition: `hermes_state_compression.py::set_compression_recovery_deadline`

```python
set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None
```

Source contract: Persist the anti-thrash recovery deadline; ``0`` / ``None`` disarms it.

Consumers:
- `agent/context_compressor.py::_set_anti_thrash_recovery_deadline` (snapshot 1989)

### `set_latest_user_api_content`

Definition: `hermes_state_messages.py::set_latest_user_api_content`

```python
set_latest_user_api_content(self, session_id: str, content: Any, api_content: str) -> int
```

Source contract: Backfill the ``api_content`` sidecar onto the newest ACTIVE user row (0/1 rows). Preflight compaction inserts that row BEFORE the sidecar exists and the later persist identity-skips compacted dicts; without this a reload reopens the prompt-cache divergence. ``content`` match guards a racing rewrite.

Consumers:
- `agent/turn_context.py::_stamp_api_content_sidecar` (snapshot 745)

### `set_session_title`

Definition: `hermes_state_titles.py::set_session_title`

```python
set_session_title(self, session_id: str, title: str) -> bool
```

Source contract: Set a title on the user's behalf (``user`` provenance). Empty clears it. Raises ValueError on conflict or validation failure.

Consumers:
- `agent/conversation_compression.py::_carry_session_state_to_child` (snapshot 2921)
- `agent/title_generator.py::_persist_session_title._set` (snapshot 322)
- `cron/scheduler.py::_set_cron_session_title` (snapshot 86)
- `cron/scheduler.py::_set_cron_session_title` (snapshot 96)

### `set_session_title_source`

Definition: `hermes_state_titles.py::set_session_title_source`

```python
set_session_title_source(self, session_id: str, source: str) -> bool
```

Source contract: Overwrite a title's provenance without touching the text (a title copied across a compression rotation keeps the original's authority).

Consumers:
- `agent/conversation_compression.py::_carry_session_state_to_child` (snapshot 2928)

### `touch_session_activity`

Definition: `hermes_state_sessions.py::touch_session_activity`

```python
touch_session_activity(self, session_id: str, ts: Optional[float]=None, *, description: Optional[str]=None, provenance: Optional[ActivityProvenance]=None) -> None
```

Source contract: Stamp durable mid-turn activity (observation-only; rate-limited by the caller) so surfaces see activity before any message row lands. Never moves ``last_activity_at`` backwards.  Called (rate-limited) from ``AIAgent._touch_activity`` so gateway/CLI surfaces and stall consumers observe API/tool/compaction activity even when no new message row has been written yet (#72016 / #72039).

Consumers:
- `agent/activity_tracking.py::_persist_session_activity_if_due` (snapshot 95)

### `try_acquire_compression_lock`

Definition: `hermes_state_compression.py::try_acquire_compression_lock`

```python
try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float=300.0) -> bool
```

Source contract: Try to atomically acquire the compression lock for ``session_id``. ``False``: another holder owns a live lock and the caller MUST NOT compress (its rotation would split the lineage). Expired locks and structured holders whose local ``pid=`` is dead are reclaimed transparently.

Consumers:
- `agent/conversation_compression.py::_lock_api_is_absent_on_session_db` (snapshot 1155)
- `agent/conversation_compression.py::_resolve_lock_api` (snapshot 2352)

### `update_session_billing_route`

Definition: `hermes_state_usage.py::update_session_billing_route`

```python
update_session_billing_route(self, session_id: str, *, provider: str, base_url: str, billing_mode: Optional[str]=None) -> None
```

Source contract: Unconditionally set the billing route (``update_token_counts`` only COALESCE-fills NULLs) so the dashboard reflects the latest /model switch; also nulls ``system_prompt`` so the cached snapshot header is rebuilt.  See #48173, #48248.

Consumers:
- `agent/agent_runtime_helpers.py::_persist_switch_billing_route` (snapshot 2096)

### `update_system_prompt`

Definition: `hermes_state_sessions.py::update_system_prompt`

```python
update_system_prompt(self, session_id: str, system_prompt: Optional[str]) -> None
```

Source contract: Store the full assembled system prompt snapshot.

Consumers:
- `agent/conversation_compression.py::_commit_compaction` (snapshot 3275)
- `agent/conversation_loop.py::_persist_system_prompt` (snapshot 633)

## 8. Extension candidates: recall/tools and legacy compute helpers

This is an overapproximate source inventory, not a claim every helper runs in every compute turn.
The routing and bypass findings in §3–4 determine which need authority-side execution instead.

### Extension `append_message`

`hermes_state_messages.py::append_message`

```python
append_message(self, session_id: str, role: str, content: str=None, tool_name: str=None, tool_calls: Any=None, tool_call_id: str=None, token_count: int=None, finish_reason: str=None, reasoning: str=None, reasoning_content: str=None, reasoning_details: Any=None, codex_reasoning_items: Any=None, codex_message_items: Any=None, platform_message_id: str=None, observed: bool=False, effect_disposition: Optional[str]=None, _compressed_summary: bool=False, timestamp: Any=None, api_content: Optional[str]=None, display_kind: Optional[str]=None, display_metadata: Optional[Dict[str, Any]]=None, compression_lock_holder: Optional[str]=None, turn_lease_holder: Optional[str]=None, turn_lease_ttl_seconds: float=300.0) -> int
```

Append one message; returns the row id and bumps the session counters. ``platform_message_id``: the platform's own id. ``api_content``: byte-fidelity sidecar, the exact string sent to the API when it differed from ``content``, stored as sent except lone surrogates.

- `tui_gateway/server.py::_append_model_switch_marker` (snapshot 1150)

### Extension `append_messages_batch`

`hermes_state_messages.py::append_messages_batch`

```python
append_messages_batch(self, session_id: str, messages: List[Dict[str, Any]], compression_lock_holder: Optional[str]=None, turn_lease_holder: Optional[str]=None, chunk_rows: Optional[int]=None, turn_lease_ttl_seconds: float=300.0) -> int
```

Append *messages* in ONE write txn (all rows land or none, guards run once); returns the inserted count. ``chunk_rows`` bounds txn size for LARGE copies (branch seeds; FTS triggers run per row).

- `tui_gateway/session_workdir.py::_persist_branch_seed` (snapshot 326)

### Extension `assert_resume_safe`

`hermes_state_messages.py::assert_resume_safe`

```python
assert_resume_safe(self, session_id: str, max_messages: Optional[int]=None, *, tip_only: bool=False) -> int
```

Resume row count, or raise ``SessionResumeTooLargeError``. ``max_messages=None`` reads config; 0 disables the guard without counting. ``tip_only`` bounds only the tip's active rows for callers that never materialize the lineage: a heavily compressed conversation is a success, not a rejection.

- `tui_gateway/session_registry.py::_load_resume_transcript` (snapshot 466)

### Extension `create_session`

`hermes_state_sessions.py::create_session`

```python
create_session(self, session_id: str, source: str, **kwargs) -> str
```

Create (upsert) a session record. Returns the session_id.

- `tui_gateway/session_workdir.py::_ensure_session_db_row` (snapshot 269)

### Extension `end_session`

`hermes_state_sessions.py::end_session`

```python
end_session(self, session_id: str, end_reason: str) -> None
```

Mark a session ended; the first end_reason wins (a compression split must keep ``'compression'`` even if a stale end_session() lands later); reopen_session() to re-end.

- `tui_gateway/session_lifecycle.py::_finalize_session` (snapshot 256)

### Extension `fts_rebuild_status`

`hermes_state_search.py::fts_rebuild_status`

```python
fts_rebuild_status(self) -> Optional[Dict[str, Any]]
```

Deferred-rebuild progress ``{"pending", "total", "indexed", "percent"}`` or None. Reads via the pooled reader (not get_meta/self._lock) so search never blocks on the writer.

- `tools/session_search_tool.py::_discover_payload` (snapshot 224)

### Extension `get_active_message_ids`

`hermes_state_messages.py::get_active_message_ids`

```python
get_active_message_ids(self, session_id: str) -> List[int]
```

Ordered physical active ids for rewind CAS checks (includes legacy harness rows projections omit).

- `tui_gateway/session_workdir.py::_rewind_active_session_history` (snapshot 410)

### Extension `get_ancestor_display_prefix`

`hermes_state_messages.py::get_ancestor_display_prefix`

```python
get_ancestor_display_prefix(self, session_id: str) -> List[Dict[str, Any]]
```

Ancestor-only display messages of a lineage (row ``session_id != tip``) that ``session.resume`` prepends. Identified by row origin, not ``display[:len(display) - len(model)]``, so alternation repair cannot overcount.

- `tui_gateway/session_registry.py::_load_resume_transcript` (snapshot 478)

### Extension `get_anchored_view`

`hermes_state_search.py::get_anchored_view`

```python
get_anchored_view(self, session_id: str, around_message_id: int, window: int=5, bookend: int=3, keep_roles: Optional[Tuple[str, ...]]=('user', 'assistant')) -> Dict[str, Any]
```

Anchored window (``get_messages_around``) plus session bookends, so one call yields the goal and the resolution of a long session. ``window`` is filtered to ``keep_roles`` (None disables) EXCEPT the anchor; ``bookend_start`` / ``bookend_end`` are the first/last ``bookend`` non-empty-content messages with ids strictly outside the window (empty when it overlaps the head/tail). Empty result when the anchor isn't in the session.

- `tools/session_search_tool.py::_hydrate_hit` (snapshot 241)
- `tools/session_search_tool.py::_title_match_result` (snapshot 205)

### Extension `get_compression_tip`

`hermes_state_compression.py::get_compression_tip`

```python
get_compression_tip(self, session_id: str) -> Optional[str]
```

Live tip of a compression chain (``get_compression_chain`` semantics); the input id when no continuation exists.

- `tools/process_registry_results.py::_owns_result` (snapshot 77)

### Extension `get_message_role`

`hermes_state_messages.py::get_message_role`

```python
get_message_role(self, session_id: str, row_id: int) -> Optional[str]
```

Role of the active message at *row_id* in *session_id*, or ``None``.

- `tools/react_to_message_tool.py::react_to_message_tool` (snapshot 43)

### Extension `get_messages`

`hermes_state_messages.py::get_messages`

```python
get_messages(self, session_id: str, include_inactive: bool=False, include_compacted: bool=False, limit: Optional[int]=None, offset: int=0, latest: bool=False, after_id: Optional[int]=None) -> List[Dict[str, Any]]
```

Load messages in insertion order (id, never timestamp: clocks regress). ``include_inactive``: rewind rows too; ``include_compacted``: compaction-archived display history (not rewind rows). ``latest`` pages back from the newest but returns chronological order; ``after_id``: keyset paging.

- `tools/session_search_tool.py::_read_session` (snapshot 373)
- `tools/session_search_tool.py::_title_match_result` (snapshot 202)

### Extension `get_messages_around`

`hermes_state_messages.py::get_messages_around`

```python
get_messages_around(self, session_id: str, around_message_id: int, window: int=5) -> Dict[str, Any]
```

Up to *window* messages either side of an anchor id (ascending). ``messages_before``/``_after`` count strictly around the anchor (fewer than *window* = session boundary). Empty for a foreign anchor.

- `tools/session_search_tool.py::_scroll` (snapshot 467)
- `tools/session_search_tool.py::_scroll` (snapshot 477)

### Extension `get_messages_as_conversation`

`hermes_state_messages.py::get_messages_as_conversation`

```python
get_messages_as_conversation(self, session_id: str, include_ancestors: bool=False, include_inactive: bool=False, repair_alternation: bool=False, include_row_ids: bool=False, include_compacted: bool=False) -> List[Dict[str, Any]]
```

Load messages in OpenAI format. ``include_compacted`` (deduped display history) is for DISPLAY reads only: the model-fed restore must not regrow what compaction summarized away. ``repair_alternation`` repairs the loaded list for LIVE REPLAY callers (a durable ``user;user`` pair would re-trigger the per-request repair forever); the stored transcript is never mutated.

- `tui_gateway/methods_prompt.py::_load_durable_truncation_history` (snapshot 76)
- `tui_gateway/methods_slash.py::_live_session_messages` (snapshot 85)
- `tui_gateway/session_registry.py::_live_visible_history` (snapshot 642)
- `tui_gateway/session_registry.py::_load_resume_transcript` (snapshot 479)
- `tui_gateway/session_workdir.py::_rewind_active_session_history` (snapshot 411)

### Extension `get_resume_conversations`

`hermes_state_messages.py::get_resume_conversations`

```python
get_resume_conversations(self, session_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
```

``(model_history, display_history)`` for a resume from ONE SELECT; byte-identical to the separate reads. model: the tip's active rows, alternation-repaired, summary marker kept for pre-compress checkpointing. display: the full lineage (``/branch`` stands alone), compaction-archived rows deduped.  The display projection also includes rows preserved by IN-PLACE compaction (``active=0, compacted=1``), deduped by :meth:`_dedupe_display_generations`. Without them a compacted conversation resumes showing only its summary plus the carried-forward tail — the user's own turns read as deleted even though every row is still on disk, and the REST transcript read (which has always included them) disagreed with this one about the same session (#92080).

- `tui_gateway/session_registry.py::_load_resume_transcript` (snapshot 477)

### Extension `get_session`

`hermes_state_sessions.py::get_session`

```python
get_session(self, session_id: str) -> Optional[Dict[str, Any]]
```

Get a session by ID (drains queued token deltas first so cost readers see exact totals).

- `tools/session_search_tool.py::_get_session_meta` (snapshot 74)
- `tools/session_search_tool.py::_title_match_result` (snapshot 198)
- `tui_gateway/agent_factory.py::_persist_live_session_runtime` (snapshot 202)
- `tui_gateway/server.py::_hydrate_session_cwd` (snapshot 1486)
- `tui_gateway/session_lifecycle.py::_finalize_session` (snapshot 253)
- `tui_gateway/session_lifecycle.py::_session_has_active_delegations` (snapshot 429)

### Extension `get_session_title`

`hermes_state_titles.py::get_session_title`

```python
get_session_title(self, session_id: str) -> Optional[str]
```

Get the title for a session, or None.

- `tui_gateway/session_registry.py::_session_live_title` (snapshot 552)

### Extension `latest_message_row_id`

`hermes_state_messages.py::latest_message_row_id`

```python
latest_message_row_id(self, session_id: str, *, role: str='user', offset: int=0, require_text: bool=True) -> Optional[int]
```

Row id of the most recent active *role* message, or ``None``. ``offset`` steps back; ``require_text`` skips rows without plain-text content so "the latest message" never resolves to an invisible bubble.

- `tools/react_to_message_tool.py::react_to_message_tool` (snapshot 39)

### Extension `list_recent_sessions_bounded`

`hermes_state_sessions.py::list_recent_sessions_bounded`

```python
list_recent_sessions_bounded(self, *, limit: int=20, exclude_sources: List[str]=None, timeout_seconds: float=3.0, candidate_limit: int=None, lineage_limit: int=None) -> List[Dict[str, Any]]
```

Latency-bounded recent-conversation browse (``session_search()``): preselect a small candidate set from the indexed durable activity timestamp (fallback ``started_at``), resolve only those across compression ancestry/chains, then hydrate activity/previews for that bounded set. Lineage traversal uses ``UNION`` plus a total-row ceiling so a corrupt cycle or a deep/branching lineage cannot defeat the bound; a lineage that hits the ceiling before a terminal root/tip is omitted, not expanded. A cooperative SQLite progress deadline interrupts sustained work past ``timeout_seconds`` and raises ``TimeoutError`` (cheap statements may finish between callbacks). Supports only the agent-tool browse filters; rich callers keep using :meth:`list_sessions_rich`.

- `tools/session_search_tool.py::_list_recent_sessions._browse` (snapshot 409)

### Extension `publish_session_git_metadata`

`hermes_state_sessions.py::publish_session_git_metadata`

```python
publish_session_git_metadata(self, session_id: str, cwd: str, generation: int, git_branch: Optional[str]=None, git_repo_root: Optional[str]=None) -> bool
```

Publish async Git enrichment only while its cwd claim is current.

- `tui_gateway/session_workdir.py::_persist_session_git_meta._run` (snapshot 491)

### Extension `reopen_session`

`hermes_state_sessions.py::reopen_session`

```python
reopen_session(self, session_id: str) -> None
```

Clear ended_at/end_reason so a session can be resumed; first stamp markerless legacy reset children that depend on the parent's mutable end_reason (WHERE shared with the listing predicate so they cannot drift).

- `tui_gateway/session_registry.py::_schedule_resume_hydration._run` (snapshot 492)

### Extension `replace_messages`

`hermes_state_messages.py::replace_messages`

```python
replace_messages(self, session_id: str, messages: List[Dict[str, Any]], active_only: bool=False, archive_dropped: bool=False, reject_active_turn_lease: bool=False) -> None
```

Atomically replace a session's messages (/retry, /undo, /compress). DESTRUCTIVE by default (rows DELETEd, leave FTS). ``active_only`` spares soft-archived rows (needed with in-place compaction). ``archive_dropped`` SOFT-archives live rows rewind-style: what rewind/edit/regenerate must use, since DELETE leaves nothing to recover. ``reject_active_turn_lease``: in-txn lease check for user rewrites.  Pass ``archive_dropped=True`` to SOFT-archive the live rows instead of DELETEing them: the replaced turns stay on disk with ``active = 0``, ``compacted = 0`` — the same "the user took it back" marking :meth:`rewind_to_message` applies — and stay readable via :meth:`get_messages` with ``include_inactive=True``. This is the mode a rewind/edit/regenerate must use: those flows overwrite a transcript the user may not have meant to drop, and a plain DELETE also evicts the rows from the FTS index, leaving nothing to recover from (#82756). It implies active-only handling — already-archived rows are never touched — so ``active_only`` is redundant with it. The rewritten set is inserted as fresh active rows exactly as in the destructive path, so the live view is identical either way; only the durability of the dropped turns differs.

- `tui_gateway/methods_prompt.py::_truncate_history_for_submit` (snapshot 413)

### Extension `resolve_session_by_title`

`hermes_state_titles.py::resolve_session_by_title`

```python
resolve_session_by_title(self, title: str) -> Optional[str]
```

Resolve a title to a session ID, preferring the latest "title #N" continuation.

- `tools/session_search_tool.py::_title_match_result` (snapshot 189)

### Extension `rewind_to_message`

`hermes_state_messages.py::rewind_to_message`

```python
rewind_to_message(self, session_id: str, target_message_id: int, *, preserve_compaction_handoff: bool=False, expected_active_ids: Optional[List[int]]=None, expected_target_content: Any=None) -> Dict[str, Any]
```

Soft-delete (``active=0``) every message with id >= *target_message_id*, target included (the caller pre-fills it as the next prompt). Returns ``{"rewound_count", "target_message", "new_head_id"}``, plus ``replacement_message_id`` with ``preserve_compaction_handoff`` (archives a composite summary carrier, inserts its hidden handoff scaffold as the new head). ``ValueError``: target missing or not ``user``. ``expected_active_ids`` / ``expected_target_content`` pin the active set and canonical live payload in-txn before any mutation (presentation-only metadata changes don't invalidate a rewind). A live turn lease refuses; expired/dead holders are reclaimed. ``rewind_count`` always increments.

- `tui_gateway/session_workdir.py::_rewind_active_session_history` (snapshot 425)

### Extension `search_messages`

`hermes_state_search.py::search_messages`

```python
search_messages(self, query: str, source_filter: List[str]=None, exclude_sources: List[str]=None, role_filter: List[str]=None, limit: int=20, offset: int=0, sort: str=None, include_inactive: bool=False, fields: Optional[Collection[str]]=None) -> List[Dict[str, Any]]
```

:meth:`_search_messages_impl` plus one log line per slow search with the routing path taken. Threshold HERMES_SEARCH_SLOW_MS (default 1000; 0 logs every call).

- `tools/session_search_tool.py::_discover` (snapshot 266)

### Extension `set_latest_matching_message_display_kind`

`hermes_state_messages.py::set_latest_matching_message_display_kind`

```python
set_latest_matching_message_display_kind(self, session_id: str, *, role: str, content: str, display_kind: str, display_metadata: Optional[Dict[str, Any]]=None) -> bool
```

Stamp presentation metadata on this turn's freshly persisted row (newest active row by content, right after the serial turn flushed); the model still sees ``role``/``content`` unchanged, so producer provenance survives without classifying by content at render time.

- `tui_gateway/prompt_turn.py::_absorb_turn_result` (snapshot 567)

### Extension `set_message_reaction`

`hermes_state_messages.py::set_message_reaction`

```python
set_message_reaction(self, session_id: str, message_row_id: int, emoji: Optional[str], *, author: str='user') -> Optional[List[Dict[str, Any]]]
```

Set (``emoji=None``: clear) *author*'s reaction. Tapback semantics: one per author per message; the same emoji again clears, a different one replaces. Returns the list after the write, or ``None`` for a foreign row.

- `tools/react_to_message_tool.py::react_to_message_tool` (snapshot 45)

### Extension `set_session_hidden`

`hermes_state_sessions.py::set_session_hidden`

```python
set_session_hidden(self, session_id: str, hidden: bool) -> bool
```

Hide/unhide a session and its compression lineage from the default listing; still resumable.

- `tui_gateway/session_workdir.py::_ensure_session_db_row` (snapshot 282)

### Extension `set_session_title`

`hermes_state_titles.py::set_session_title`

```python
set_session_title(self, session_id: str, title: str) -> bool
```

Set a title on the user's behalf (``user`` provenance). Empty clears it. Raises ValueError on conflict or validation failure.

- `tui_gateway/prompt_turn.py::_after_complete_turn` (snapshot 336)

### Extension `take_unseen_reactions`

`hermes_state_messages.py::take_unseen_reactions`

```python
take_unseen_reactions(self, session_id: str, *, author: str='user') -> List[Dict[str, Any]]
```

Return *author*'s not-yet-surfaced reactions and mark them seen. Reactions are announced on the NEXT user turn (never by rewriting the reacted message: cache-safe); ``seen`` makes it exactly once.

- `tui_gateway/methods_prompt.py::_pending_reaction_notes` (snapshot 148)

### Extension `update_session_cwd`

`hermes_state_sessions.py::update_session_cwd`

```python
update_session_cwd(self, session_id: str, cwd: str, git_branch: Optional[str]=None, git_repo_root: Optional[str]=None, replace_git_meta: bool=False) -> Optional[int]
```

Persist the authoritative cwd and claim a Git metadata generation. git fields are written only when non-empty (a probe failure never clobbers a value) except under ``replace_git_meta`` (a workspace MOVE overwrites the old repo identity). Async probes publish with the returned generation so an older worker cannot overwrite a newer claim (A -> B -> A).

- `tui_gateway/server.py::_hydrate_session_cwd` (snapshot 1491)
- `tui_gateway/session_workdir.py::_persist_session_cwd_and_schedule_git_meta` (snapshot 504)

### Extension `update_session_meta`

`hermes_state_sessions.py::update_session_meta`

```python
update_session_meta(self, session_id: str, model_config_json: str, model: Optional[str]=None) -> None
```

Update model_config and (COALESCE) optionally model.

- `tui_gateway/agent_factory.py::_persist_live_session_runtime` (snapshot 208)
- `tui_gateway/agent_factory.py::_persist_live_session_runtime` (snapshot 209)

### Extension `update_session_model`

`hermes_state_sessions.py::update_session_model`

```python
update_session_model(self, session_id: str, model: str, provider: Optional[str]=None) -> None
```

Set the model after a mid-session /model switch (unconditionally), null system_prompt so stale Model:/Provider: footers rebuild, and drop any Browser runtime lock (lineage markers survive). *provider* is merged into model_config so resume recombines model and provider.  When *provider* is given, it is merged into ``model_config`` alongside the model (``$.model`` / ``$.provider``) so a later resume recombines the persisted model with the provider that actually serves it instead of the config.yaml primary provider (#79536). Callers without provider knowledge leave any stored provider untouched.

- `tui_gateway/agent_factory.py::_persist_live_session_runtime` (snapshot 210)
- `tui_gateway/agent_factory.py::_persist_live_session_runtime` (snapshot 211)

### Extension `update_system_prompt`

`hermes_state_sessions.py::update_system_prompt`

```python
update_system_prompt(self, session_id: str, system_prompt: Optional[str]) -> None
```

Store the full assembled system prompt snapshot.

- `tui_gateway/server.py::_persist_live_session_system_prompt` (snapshot 1080)
- `tui_gateway/server.py::_persist_live_session_system_prompt` (snapshot 1092)

## 9. Executable static proof and limits

Run from this checkout root; this imports only Python stdlib, reads tracked source, and never opens
state.db or launches an agent. It is an offline inventory census, NOT a source-shape regression test.
No pytest test is added. AST handles attributes and literal indirect method names; the full report
also emits every db-like attribute/access, dynamic getter, raw SQL call, and constructor/open/close
candidate in the declared file scope. `conn.execute(sql,...)` is emitted verbatim; the async-delegation
closed wrappers in §4 resolve that local SQL parameter. Name-only matching deliberately overmatches;
the exclusions for analytics/export and common-name collisions are explicit in the executable.

```sh
python3 -c 'from pathlib import Path; p=Path("evals/shared_gateway/worker-persistence-inventory.md"); exec(p.read_text().rsplit("<!-- census:start -->",1)[1].split("```python",1)[1].split("```",1)[0])'
# Append --full to print all source hashes, consumer mappings and access/SQL/open candidates.
```

<!-- census:start -->
```python
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
tracked = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
state_files = sorted(p for p in tracked if p.startswith("hermes_state") and p.endswith(".py") and "/" not in p)
definitions = {}
for path in state_files:
    source = (root / path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append({
                "file": path, "line": node.lineno,
                "signature": ast.unparse(node.args),
                "return": ast.unparse(node.returns) if node.returns else None,
                "doc": ast.get_docstring(node),
            })
core = sorted(p for p in tracked if p.endswith(".py") and (
    p == "run_agent.py" or (p.startswith("agent/") and p.count("/") == 1)
    or (p.startswith("cron/scheduler") and p.count("/") == 1)
    or p.startswith("tools/delegate_tool") or p == "tools/async_delegation.py"
    or p.startswith("tui_gateway/compute_host") or p == "tui_gateway/host_supervisor.py"))
extensions = ['tools/session_search_tool.py', 'tools/react_to_message_tool.py', 'tools/process_registry_results.py', 'tui_gateway/server.py', 'tui_gateway/agent_factory.py', 'tui_gateway/session_workdir.py', 'tui_gateway/session_registry.py', 'tui_gateway/session_lifecycle.py', 'tui_gateway/prompt_turn.py', 'tui_gateway/methods_prompt.py', 'tui_gateway/methods_slash.py']
noise = {"add", "close", "write", "stats", "message_count", "resolve_session_id"}
nonturn = {"agent/trace_upload.py", "agent/insights.py"}
refs, accesses, indirect, sql, opens = [], [], [], [], []
class Census(ast.NodeVisitor):
    def __init__(self, path):
        self.path, self.scope = path, []
    def mark(self, node, value):
        return {"file": self.path, "consumer": ".".join(self.scope), "line": node.lineno, "value": value}
    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Attribute(self, node):
        value = ast.unparse(node)
        if "db" in value.lower():
            accesses.append(self.mark(node, value))
            if node.attr in definitions:
                refs.append({**self.mark(node, node.attr), "expression": value})
        self.generic_visit(node)
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in definitions:
            refs.append({**self.mark(node, node.value), "expression": repr(node.value)})
    def visit_Call(self, node):
        value = ast.unparse(node)
        name = ast.unparse(node.func)
        if name in {"getattr", "hasattr", "vars"} and node.args and "db" in ast.unparse(node.args[0]).lower():
            indirect.append(self.mark(node, value))
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"execute", "executemany", "executescript"}:
            sql.append(self.mark(node, value))
        if name in {"SessionDB", "acquire", "sqlite3.connect", "release_or_close"}:
            opens.append(self.mark(node, value))
        self.generic_visit(node)
for path in sorted(set(core + extensions)):
    Census(path).visit(ast.parse((root / path).read_text(encoding="utf-8"), filename=path))
core_refs = [r for r in refs if r["file"] in core and r["file"] not in nonturn and r["value"] not in noise]
methods = sorted({r["value"] for r in core_refs})
source_files = sorted(set(core + extensions + state_files))
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_files}
source_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
summary = {
    "core_files": len(core), "extension_files": len(extensions), "source_files": len(source_files),
    "core_candidate_references": len(core_refs), "core_operations": len(methods),
    "source_digest": source_digest, "raw_sql_candidates": len(sql),
    "dynamic_db_access_candidates": len(indirect), "open_close_candidates": len(opens),
    "operation_names": methods,
}
if "--full" in sys.argv:
    print(json.dumps({"summary": summary, "hashes": hashes, "core_refs": core_refs,
        "extension_refs": [r for r in refs if r["file"] in extensions],
        "db_accesses": accesses, "indirect": indirect, "sql": sql, "opens": opens,
        "definitions": {k: definitions[k] for k in methods}}, indent=2))
else:
    print(json.dumps(summary, indent=2))
```
<!-- census:end -->

Observed census output at the source snapshot:

```json
{
  "core_files": 235,
  "extension_files": 11,
  "source_files": 271,
  "core_candidate_references": 95,
  "core_operations": 51,
  "source_digest": "43b5398f1ac7f2b1811aa454b25d1040c3bd91aa20c9444ccb2173c0842bb7e9",
  "raw_sql_candidates": 45,
  "dynamic_db_access_candidates": 44,
  "open_close_candidates": 26,
  "operation_names": [
    "acquire_session_turn_lease",
    "append_messages_batch",
    "archive_and_compact",
    "clear_compression_failure_cooldown",
    "clear_session_activity_labels",
    "create_session",
    "declared_scope_identity",
    "end_session",
    "flush_token_counts",
    "get_active_message_watermark",
    "get_compression_failure_cooldown",
    "get_compression_failure_cooldown_row",
    "get_compression_fallback_streak",
    "get_compression_ineffective_count",
    "get_compression_lineage",
    "get_compression_lock_holder",
    "get_compression_recovery_deadline",
    "get_compression_tip",
    "get_conversation_root",
    "get_messages_as_conversation",
    "get_next_title_in_lineage",
    "get_session",
    "get_session_model_config_value",
    "get_session_title",
    "get_session_title_source",
    "is_explicit_fork_child",
    "latest_conversation_boundary",
    "patch_session_model_config",
    "publish_compression_child",
    "queue_token_counts",
    "record_auxiliary_usage",
    "record_compression_failure_cooldown",
    "refresh_compression_lock",
    "refresh_session_turn_lease",
    "release_compression_lock",
    "release_session_turn_lease",
    "reopen_orphaned_compression_session",
    "resolve_resume_session_id",
    "restore_compression_failure_cooldown_row",
    "session_lifecycle_statuses",
    "set_auto_title",
    "set_compression_fallback_streak",
    "set_compression_ineffective_count",
    "set_compression_recovery_deadline",
    "set_latest_user_api_content",
    "set_session_title",
    "set_session_title_source",
    "touch_session_activity",
    "try_acquire_compression_lock",
    "update_session_billing_route",
    "update_system_prompt"
  ]
}
```

Interpretation: counts are candidate references, not dynamically executed call counts. Source digest
is over exact file-name=>SHA256 pairs for the declared scope. No syntax errors were tolerated. The
51 core operations are enumerated in §7; legacy title fallback absent from current SessionDB is
classified in §5, not silently treated as a callable. Read-only source enumeration cannot prove
network ACK timing, transaction durability after crash, runtime plugin/tool activation, profile
ContextVar propagation, blocked-loop behavior or complete external-worker FD isolation. Those remain
explicit acceptance blockers in §6. No worker substrate acceptance is claimed.

### Checks actually executed for this document

* Executed the fenced stdlib census from the worktree root: 235 core files + 11 extension files;
  271 total source/definition files; 95 core references covering 51 indexed operations; source digest
  matched the recorded output. The extension index covers 34 operation names (overlaps core).
* Programmatically checked that every core operation has an index heading and all 95 candidate
  references have their consumer symbol recorded. Reviewed all raw-SQL/open/property candidate
  consumer names, classifying separate verification-evidence storage and non-SQL execute calls.
* `python3 scripts/check-windows-footguns.py --all`: passed (1534 files scanned).
* `python3 scripts/check_compat_pointers.py`: passed (2091 pointers checked); emitted existing
  Python invalid-escape SyntaxWarnings from other sources, not errors or changes in this Markdown.
* No pytest suite run: this is a source-only documentation census with no production/test edit,
  not a worker-runtime implementation. No Q2–Q8 green receipt is inferred from these checks.
