# Worker lifecycle extension — wave 22

This is a **parent-composed extension**, not activation of child/cron/legacy-compute
consumers. The shared adapter and dispatch modules are deliberately unchanged in
this lane. Tests need the hookups below; uncomposed tests reproduce absent methods.

## Parent hookups (all exercised in the disposable composition)

1. `agent/runtime_session_store.py`: import
   `RuntimeSessionLifecycleMixin` from `agent.runtime_session_lifecycle` alongside
   the module imports, and add it to `RuntimeSessionStore`'s bases.
2. `hermes_state_runtime.py::mutate_worker_execution`: late-import
   `WORKER_LIFECYCLE_HANDLERS` from `hermes_state_worker_lifecycle`, and unpack it
   into the existing `handlers` table. Keep `_epoch`, `_worker_assignment`, owner
   epoch, receipt digest/sequence, terminal status and transaction code unchanged.
3. `hermes_state_sessions.py::SessionSessionsMixin._insert_session_row`: replace
   the current function body after its docstring with:

   ```python
   from hermes_state_worker_lifecycle import insert_session_row_in_transaction
   params = dict(session_id=session_id, source=source, model=model, model_config=model_config,
                 system_prompt=system_prompt, user_id=user_id, session_key=session_key,
                 chat_id=chat_id, chat_type=chat_type, thread_id=thread_id,
                 parent_session_id=parent_session_id, cwd=cwd, profile_name=profile_name,
                 git_repo_root=git_repo_root, origin_json=origin_json, display_name=display_name)
   self._execute_write(lambda conn: insert_session_row_in_transaction(self, conn, **params),
                       patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
   ```

   The new helper contains the existing keep-existing/reset-marker upsert body,
   prompt hash/GC and metadata inheritance. This hookup removes the temporary
   duplicate body; do not retain two independent constructor implementations.

No new RPC endpoint, SQL RPC, worker registry, receipt table, or service is needed.
The ordinary `worker.persist` endpoint already forwards the closed operation.

## Contracts

- Concrete methods: `create_session`, `end_session`, `session_lifecycle_statuses`,
  `set_session_title`, `set_auto_title`, `get_session_title_source`,
  `set_session_title_source`, `get_next_title_in_lineage`, `touch_session_activity`,
  `clear_session_activity_labels`, `update_session_billing_route`,
  `set_latest_user_api_content`.
- Creation means backfilling an **already reserved** row. A worker cannot choose a
  new id, source, profile, parent, cwd, routing origin, git identity or display
  identity. Supplied identity values must match the reserved row; omitted values
  cannot replace it. Constructor config accepts only the current AIAgent's
  `max_iterations`, `max_tokens`, `reasoning_config`, `yolo_mode` keys. Arbitrary
  config/lineage mutation remains unavailable.
- Keep-existing model/config/prompt semantics reuse the constructor body. The
  existing `_end_and_bump` preserves first end reason and boundary atomicity.
  Session end and worker terminal settlement remain distinct.
- Titles reuse `_set_session_title_in_transaction`, including rank, canonical
  Bot Chat protection and compression-ancestor transfer. Expected title errors
  are receipted outcomes, converted back to `ValueError` **after** the local
  journal is acknowledged, so real title-generator dedupe can proceed.
  Error messages do not disclose the conflicting session id.
- Next-title lookup requires the assigned current title or a previous receipted
  title attempt in its active worker execution. It returns only a candidate,
  never rows/session ids. Unrelated arbitrary title queries are refused. The
  lookup is not atomic reservation; the normal setter remains authoritative.
- Activity remains monotonic and uses the existing description/provenance
  normalization; clearing labels retains the timestamp. Receipt transactions use
  the substrate's transcript-write patience, not the local observation-only short
  patience/no-op fast path. This latency distinction remains for integration.
- Billing route changes follow preceding durable usage receipts, reject sticky
  pending failures, and clear/GC the prompt snapshot just like the local method.
  Main/auxiliary usage logic is unchanged: no inferred/recomputed provider usage.
- API content stamps only the newest active user row matching encoded content.
- All operations revalidate assignment/generation/epoch **before receipt replay**.
  Old finalizers cannot close or relabel a later generation. `close()` remains
  local outbox release; it neither closes the owner's DB nor settles a worker.

## Proof

`tests/state/test_runtime_worker_lifecycle.py` (two invariants) covers constructor
backfill, identity refusal, lifecycle status, lost-ACK end replay, first-end wins,
and late/foreign end refusal. `test_runtime_worker_metadata.py` (two invariants)
covers title ranks/activity and exact two-route counters/API content. The title
retry invariant invokes the real `agent.title_generator._persist_session_title`.

`tests/gateway/test_worker_lifecycle_execution.py` starts ordinary `gateway.run`
and a separate worker interpreter using the existing loopback SSE model fixture.
The real AIAgent persists a turn; the worker applies title/activity/API content/
route/end/finish, then remains alive while a real owner `prompt.submit` executes
generation 1. Its old generation-0 end/title/clear/route requests are refused.
Foreign end/title/clear requests are refused. SQLite audit starts before AIAgent
imports; canonical worker connect events and writable DB/WAL/SHM descriptors are
both zero. This is a controlled delayed callback, not a timed-out child/cron job.

The composed broad run passed **140 files, 1511 tests, one Windows-only skip**:
all `tests/state` and `tests/cron`, selected agent title/activity/finalizer/store
files, current worker-agent integration, the new successor probe, and existing
worker owner-kill/adoption/outbox coverage. It is not the entire repository suite.

## Not completed by this lane

Producer reservation/migration for child, cron and Kanban; same-PID adapter;
compression assignment/history/locks (sibling lane); async delegation ledger
(sibling lane); recall/reaction/raw tool storage and legacy compute controls;
arbitrary legacy `update_session_meta`; coordinator tracking of outstanding
background callbacks before terminal settlement. Existing deferred-cleanup code
was not changed and its cron regression suite remains green. None of these
consumers should be enabled solely on the strength of this scoped proof.
