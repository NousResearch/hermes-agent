"""Fence an unsettled claim in a live owner without inventing a terminal result."""
import asyncio
import logging
import sqlite3

from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, get_session_admission

logger = logging.getLogger(__name__)


def _mark_unknown(authority, row):
    def write(conn):
        _epoch(conn, authority.epoch)
        current = _admission(conn, row['admission_id'])
        if current['owner_epoch'] != authority.epoch or current['generation'] != row['generation']:
            raise RuntimeStoreError('stale_generation')
        if current['status'] in {'terminal', 'unknown'}:
            return current['status']
        if current['status'] != 'started':
            raise RuntimeStoreError('stale_generation')
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return 'unknown'
    return authority.db._execute_write(write)


async def recover_failed_settlement(authority, row):
    # A full disk or held SQLite writer may also block the recovery stamp. Retain this owner
    # and its captured result until the storage fence can commit; never manufacture a reply.
    while True:
        try:
            from gateway.session_runtime_workers import track_mutation
            writer = track_mutation(authority, asyncio.to_thread(_mark_unknown, authority, row))
            return await asyncio.shield(writer)
        except (OSError, sqlite3.Error):
            await asyncio.sleep(1)


def completion_payload(row, settled, response, captured):
    # ``status`` is the message.complete contract's TurnStatus: the Desktop
    # extends a Stopped bubble to the persisted partial only on 'interrupted'.
    complete = {
        'text': response, 'content': response, 'admission_id': row['admission_id'],
        'outcome': 'cancelled' if settled['outcome'] == 'interrupted' else settled['outcome'],
        'status': {'completed': 'complete', 'interrupted': 'interrupted'}.get(
            settled['outcome'], 'error')}
    # Only the agent's reuse site sets this (never inferred from equal text): the
    # final repeats a reply the viewer already painted, so it settles in place.
    captured_result = (captured or {}).get('result') or {}
    if response and captured_result.get('response_reused'):
        complete['response_reused'] = True
    if captured_result.get('local_viewer'):
        if captured_result.get('response_previewed'):
            complete['response_previewed'] = True
        if captured_result.get('response_transformed'):
            complete['response_transformed'] = True
        if isinstance(captured_result.get('warning'), str) and captured_result['warning']:
            complete['warning'] = captured_result['warning']
        if isinstance(captured_result.get('billing_block'), dict):
            complete['billing'] = captured_result['billing_block']
        if isinstance(captured_result.get('failure_reason'), str) and captured_result['failure_reason']:
            complete['failure_reason'] = captured_result['failure_reason']
    # The committed row addresses of the turn: a viewer binds the streamed reply to
    # its stored row, so a transcript read racing this frame never paints it twice.
    # ``submission_id`` names whose turn it is: the sending viewer binds its optimistic
    # prompt (``user-<submission_id>``), which the queued admission ack could not name.
    # A local viewer's turn end recorded the session-cumulative stats (``session_usage.publish_usage``):
    # main's TUI read ``message.complete.usage`` for its status bar and ``/usage`` totals.
    if isinstance((captured or {}).get('session_usage'), dict):
        from gateway.session_usage import tui_usage
        complete['usage'] = tui_usage(captured['session_usage'])
    if isinstance(captured_result.get('persisted_turn'), dict):
        complete['persisted_turn'] = {**captured_result['persisted_turn'],
                                      'submission_id': row['request_id']}
    return complete


def _publish_completion(authority, live, ref, row, settled, response, captured):
    """Thread-safe half of terminal publication; the caller holds the stream lock. No SQLite write
    happens here: loop-side readers wait on this lock, so it never spans storage I/O."""
    try:
        live.controls.snapshot(ref.session_id, None)
    except Exception:
        logger.exception('Terminal control cleanup failed for admission %s', row['admission_id'])
    live.event_stream.publish(ref.session_id, completion_payload(row, settled, response, captured))


def _release_media(authority, row):
    """Terminal media cleanup: its own write txn after publication, never under the stream lock."""
    try:
        from gateway.session_ingress_media import release_admission_media
        release_admission_media(authority.db, row['admission_id'])
    except Exception:
        logger.exception('Terminal media cleanup failed for admission %s', row['admission_id'])


def commit_and_publish(authority, live, ref, row, response, outcome, captured, settlement):
    """Worker-thread settlement: redaction, encoding and the SQLite write never stall the owner loop.

    The write runs OUTSIDE ``event_stream.lock``: attach, replay, receipts and pending publication
    take that lock on the loop, and a writer held by another process (up to ``_WRITE_PATIENCE_S``)
    would stall every session behind it. Ordering is kept by ``live.settling``: from before the
    write until the completion frame is published, lock holders read this claim as still started,
    so none sees the terminal row without its completion. ``settlement`` receives the committed
    ``(settled, response)`` before publication, so a publish failure is not a lost commit.
    The caller publishes idle ``session.info`` on the loop afterwards (bot receipt wakeups are tasks).
    """
    from gateway.session_results import finish_result, prepare_result
    # Compaction, redaction and outcome flags are CPU-only.
    prepared = prepare_result(row, response, outcome, captured)
    admission_id = row['admission_id']
    # ``live`` is the drain's own entry: a delete committing on the loop after this write may
    # already have dropped it from ``authority.sessions``.
    with live.event_stream.lock:
        live.settling[admission_id] = row
    try:
        settlement['settled'], settlement['response'] = finish_result(
            authority.db, epoch=authority.epoch, row=row, response=response, outcome=outcome,
            result=captured, prepared=prepared)
        with live.event_stream.lock:
            live.settling.pop(admission_id, None)
            _publish_completion(authority, live, ref, row, settlement['settled'], settlement['response'], captured)
    finally:
        with live.event_stream.lock:
            live.settling.pop(admission_id, None)


# Pauses between settlement attempts after a transient storage error (held writer past its
# patience, full disk). Bounded: the attempts after the last one fall back to the unknown fence.
_SETTLE_RETRY_DELAYS_S = (0.5, 2.0)


async def settle_with_retry(authority, live, ref, row, response, outcome, captured, settlement):
    """Commit the captured result, retrying a transient storage failure a bounded number of times.

    One locked or full write must not turn a finished answer into an ``unknown`` turn whose result
    lives only in memory. Retrying is exact: a write that committed before reporting failure is
    read back by ``finish_result``, and a failure after the commit (``settled`` set) never retries.
    A fence refusal (``RuntimeStoreError``) is not transient and is raised at once."""
    from gateway.session_runtime_workers import track_mutation
    for delay in (*_SETTLE_RETRY_DELAYS_S, None):
        try:
            return await asyncio.shield(track_mutation(authority, asyncio.to_thread(
                commit_and_publish, authority, live, ref, row, response, outcome, captured, settlement)))
        except (OSError, sqlite3.Error) as exc:
            if delay is None or 'settled' in settlement:
                raise
            logger.warning('Settlement of admission %s deferred by storage (%r); retrying in %.1fs',
                           row['admission_id'], exc, delay)
            await asyncio.sleep(delay)


def publish_terminal(authority, ref, row, settled, response, captured):
    """Cleanup cannot withhold a committed outcome; publication is attempted once."""
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        _publish_completion(authority, live, ref, row, settled, response, captured)
    # Idle follows the completion so a viewer cannot mistake a settled turn for a lost frame.
    authority._publish_pending(ref)


async def settle_and_deliver(authority, live, ref, row, response, outcome):
    """Own settlement through delivery even if the drain's waiter is cancelled.

    Terminal media collection can scan a multi-GB transcript. Start it only after
    delivery (or a failed attempt), tracked separately so it cannot delay the FIFO.
    """
    admission_id = row['admission_id']
    settled = None
    try:
        settlement = {}
        try:
            # Redaction, encoding and the SQLite write run off-loop: one turn's settlement must
            # not stall every other session. Publication takes the stream lock only after
            # the write commits; this owner also outlives a cancelled drain.
            captured = authority.pending_results.get(admission_id)
            try:
                await settle_with_retry(authority, live, ref, row, response, outcome, captured, settlement)
            finally:
                if 'settled' in settlement:
                    settled, response = settlement['settled'], settlement['response']
                    authority.pending_results.pop(admission_id, None)
            # Idle follows the completion; its bot receipt wakeups are loop tasks. A delete
            # that committed while the write yielded already retired this live entry.
            if authority.sessions.get(ref.session_id) is live:
                authority._publish_pending(ref)
        except Exception:
            logger.exception(
                'Settlement or publication of admission %s failed', admission_id)
            if settled is None:
                authority.pending_results.setdefault(admission_id, {
                    'result': {'final_response': response, 'failed': outcome == 'failed'}, 'usage': {}})
                try:
                    status = await recover_failed_settlement(authority, row)
                except RuntimeStoreError:
                    # Ownership moved; this owner may only release its process-local observers.
                    authority._pause(ref, 'unknown_execution')
                    return False
                # Discard can race the off-loop recovery stamp; its terminal decision wins.
                current = get_session_admission(authority.db, admission_id=admission_id)
                status = current['status']
                if status == 'unknown':
                    from gateway.session_ingress import retain_unsettled_delivery
                    retain_unsettled_delivery(authority, admission_id)
                    authority._publish_pending(ref)
                    authority._pause(ref, 'unknown_execution')
                    return False
                # A write can commit before reporting failure. Re-read its exact result,
                # then publish the terminal outcome rather than inventing uncertainty.
                from gateway.session_results import admission_result
                with live.event_stream.lock:
                    captured = admission_result(authority.db, admission_id)
                    authority.pending_results.pop(admission_id, None)
                    settled = current
                    if captured is None:
                        # An operator discarded uncertainty while recovery yielded. There is
                        # no committed answer to replay; its queued successor may now run.
                        authority.pending_deliveries.pop(admission_id, None)
                        waiter = authority.waiters.pop(admission_id, None)
                        if waiter is not None and not waiter.done():
                            waiter.set_exception(RuntimeStoreError('unknown_execution'))
                        return True
                    response = captured['result'].get('final_response') or ''
                    publish_terminal(authority, ref, row, settled, response, captured)
        finally:
            # The stamp names a claimed, unsettled execution. Left in place, idle
            # mutations (session.updated) would carry a terminal generation and
            # a versioned viewer fence would discard them as late frames.
            with live.event_stream.lock:
                live.event_stream.execution = {}
            authority.pending_stops.pop(ref.session_id, None)
            authority.pending_steers.pop(ref.session_id, None)
            authority.adopted.pop(ref.session_id, None)
            if settled is None:
                # Uncommitted (unknown or discarded): the user is not told an answer the FIFO lost.
                authority.pending_deliveries.pop(admission_id, None)
            from gateway.session_ingress import waiter_replies
            waiter_replies(authority).pop(admission_id, None)
        from gateway.session_ingress import deliver_settled
        await deliver_settled(authority, admission_id)
        await authority._resolve_settled_waiter(admission_id, response)
        return True
    finally:
        if settled is not None:
            release_media_after_delivery(authority, row)


def release_media_after_delivery(authority, row):
    """Keep the off-loop collector owned until shutdown/profile retirement joins it."""
    from gateway.session_runtime_workers import track_mutation
    return track_mutation(authority, asyncio.to_thread(_release_media, authority, row))
