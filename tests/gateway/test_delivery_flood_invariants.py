"""Delivery retries honor the platform deadline without crossing ownership boundaries."""
import os
import sqlite3
import time

from gateway import delivery_ledger as dl

# Weixin's circuit breaker fails a send closed with this shape (gateway/platforms/weixin.py) — a bare
# error string with neither the ``flood_control:<seconds>`` prefix adapters normally use nor PTB's raw
# "Flood control exceeded... retry in N seconds" wording, but the same platform-neutral "rate_limited"
# classification (gateway.platforms.base.classify_send_error) that gates the in-attempt send backoff.
WEIXIN_RATE_LIMIT_ERROR = 'iLink sendmessage rate limited; cooldown active for 12.3s'


def record(oid, *, profile=None, platform='telegram', error='flood_control:185'):
    dl.record_obligation(obligation_id=oid, session_key='session-' + oid, platform=platform,
                         chat_id='123', thread_id='77', content='.' * 3000, adapter_profile=profile)
    dl.mark_failed(oid, error)


def read(oid):
    with sqlite3.connect(dl._db_path()) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute('SELECT * FROM delivery_obligations WHERE obligation_id=?', (oid,)).fetchone())


def test_runtime_deadline_preserves_scope_and_retry_budget():
    record('due')
    record('other', profile='other')
    record('blocked')
    dl.mark_failed('blocked', 'Forbidden: bot was blocked by the user')
    stamp = read('due')['updated_at']
    assert dl.sweep_failed_for_runtime('telegram', now=stamp + 184) == []
    assert read('due')['attempts'] == 0
    claimed = dl.sweep_failed_for_runtime('telegram', now=stamp + 186)
    assert [r['obligation_id'] for r in claimed] == ['due']
    assert claimed[0]['needs_marker'] and 'rate limit' in claimed[0]['marker']
    assert read('due')['last_error'] is None
    assert dl.sweep_failed_for_runtime('telegram', now=stamp + 187) == []
    assert dl.release_runtime_claim('due', claimed[0]['last_error'])
    assert read('due')['attempts'] == 0
    assert dl.sweep_failed_for_runtime('telegram', now=time.time()) == []
    claimed = dl.sweep_failed_for_runtime('telegram', now=time.time() + 186)
    assert len(claimed) == 1
    dl.mark_delivered('due')
    assert dl.sweep_failed_for_runtime('telegram', now=time.time() + 187) == []
    assert read('other')['attempts'] == read('blocked')['attempts'] == 0


def test_boot_adopts_waiting_rows_without_spending_or_losing_deadline():
    record('boot')
    original = read('boot')
    # No PID means a genuinely ownerless persisted row; do not patch the liveness predicate.
    with sqlite3.connect(dl._db_path()) as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=NULL, owner_started_at=NULL, adapter_profile=NULL")
    claimed = dl.sweep_recoverable(now=original['updated_at'] + 20,
                                    deliverable_targets={('telegram', None)})
    assert len(claimed) == 1 and claimed[0].get('adopted')
    adopted = read('boot')
    assert adopted['owner_pid'] == os.getpid()
    assert adopted['adapter_profile'] == 'default'
    assert adopted['attempts'] == 0 and adopted['updated_at'] == original['updated_at']
    assert dl.sweep_recoverable(now=original['updated_at'] + 30) == []
    assert dl.sweep_failed_for_runtime('telegram', now=original['updated_at'] + 184) == []
    claimed = dl.sweep_failed_for_runtime('telegram', now=original['updated_at'] + 186)
    assert len(claimed) == 1 and claimed[0]['needs_marker']
    assert read('boot')['state'] == 'attempting' and read('boot')['last_error'] is None


def test_weixin_rate_limit_text_classifies_as_flood_and_extracts_its_own_wait():
    """``is_flood_error``/``flood_wait_seconds`` must recognise Weixin's own rate-limit wording (no
    ``flood_control:`` prefix, no PTB text) via the platform-neutral classifier, and read the cooldown
    seconds Weixin itself embeds rather than falling back to the generic default."""
    assert dl.is_flood_error(WEIXIN_RATE_LIMIT_ERROR)
    assert dl.flood_wait_seconds(WEIXIN_RATE_LIMIT_ERROR) == 12.3
    # An ordinary Weixin failure (not rate-limited) must still read as an ordinary failure.
    assert not dl.is_flood_error('iLink sendmessage error: ret=1 errcode=2 errmsg=unknown error')


def test_weixin_rate_limit_arms_timed_redelivery_not_immediate_retry():
    """The ledger must treat a Weixin rate-limit failure exactly like a Telegram flood failure: adopted
    without spending an attempt while still inside the wait, then claimable once it has passed — never
    claimed as an ordinary failure that spends an attempt inside the still-active cooldown."""
    record('weixin-due', platform='weixin', error=WEIXIN_RATE_LIMIT_ERROR)
    stamp = read('weixin-due')['updated_at']
    # Still inside Weixin's own 12.3s cooldown: the runtime sweep must not claim it yet.
    assert dl.sweep_failed_for_runtime('weixin', now=stamp + 11) == []
    assert read('weixin-due')['attempts'] == 0
    # Past the cooldown: claimable, carries a marker (rate-limit wording), and clears the stale error.
    claimed = dl.sweep_failed_for_runtime('weixin', now=stamp + 13)
    assert [r['obligation_id'] for r in claimed] == ['weixin-due']
    assert claimed[0]['needs_marker'] and 'rate limit' in claimed[0]['marker']
    assert read('weixin-due')['last_error'] is None


def test_weixin_rate_limit_boot_adoption_does_not_spend_attempt():
    """A dead-owner boot sweep must adopt a still-cooling-down Weixin rate-limit row (no attempt spent,
    deadline preserved) exactly as it does for Telegram's ``flood_control:`` rows — not treat it as an
    ordinary failed row and immediately spend a redelivery attempt inside the cooldown."""
    record('weixin-boot', platform='weixin', error=WEIXIN_RATE_LIMIT_ERROR)
    original = read('weixin-boot')
    with sqlite3.connect(dl._db_path()) as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=NULL, owner_started_at=NULL, adapter_profile=NULL")
    claimed = dl.sweep_recoverable(now=original['updated_at'] + 5,
                                    deliverable_targets={('weixin', None)})
    assert len(claimed) == 1 and claimed[0].get('adopted')
    adopted = read('weixin-boot')
    assert adopted['owner_pid'] == os.getpid()
    assert adopted['adapter_profile'] == 'default'
    assert adopted['attempts'] == 0 and adopted['updated_at'] == original['updated_at']
    assert adopted['last_error'] == WEIXIN_RATE_LIMIT_ERROR
    # A second boot-sweep pass finds this process still the live owner: nothing more to adopt.
    assert dl.sweep_recoverable(now=original['updated_at'] + 6) == []
    # Still inside the cooldown: the runtime sweep (this process claiming its own adopted row) must wait.
    assert dl.sweep_failed_for_runtime('weixin', now=original['updated_at'] + 11) == []
    # Past the deadline: claimed, marker set, error cleared.
    claimed = dl.sweep_failed_for_runtime('weixin', now=original['updated_at'] + 13)
    assert len(claimed) == 1 and claimed[0]['needs_marker']
    assert read('weixin-boot')['state'] == 'attempting' and read('weixin-boot')['last_error'] is None


def test_telegram_flood_detection_unaffected_by_broader_classifier():
    """Widening ``is_flood_error`` to also consult the platform-neutral classifier must not change
    Telegram's own, already-correct behavior for its two established shapes."""
    assert dl.is_flood_error('flood_control:185')
    assert dl.is_flood_error('Flood control exceeded. Retry in 185 seconds')
    assert dl.flood_wait_seconds('flood_control:185') == 185.0
    assert dl.flood_wait_seconds('Flood control exceeded. Retry in 185 seconds') == 185.0
    # A permanent Telegram rejection must never be mistaken for a flood.
    assert not dl.is_flood_error('Forbidden: bot was blocked by the user')
