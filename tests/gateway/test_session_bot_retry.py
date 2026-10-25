"""The owner retries a transiently failed Bot DM delivery exactly once, never unknown execution.

``gateway/session_bot.py`` admits a Bot Chat delivery into the authority FIFO. When that admission
SETTLES ``failed`` with a committed result whose error classifies transient
(``tools.bot_failure_reasons.result_retry_action``), one retry is admitted under the derived
identity ``bot:<id>:retry``; the delivery's receipt follows it. A repeated ``deliver`` of the same
id or an owner restart (``recover_bot_deliveries``) re-reads that admission instead of minting
another, and an unknown/non-transient admission is never replayed.
"""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import Principal
from hermes_state_runtime import list_session_admissions

KEY = 'c' * 32
ONE_PX_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
    '0000000d49444154789c6360f8cf00000004000101f9b9e4'
    '0000000049454e44ae426082')


@pytest_asyncio.fixture
async def bot(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'bot', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Bot Chat')
    errors = []

    async def execute(authority, ref, row):
        error = errors.pop(0) if errors else None
        if error is None:
            return 'pong'
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': '', 'messages': [], 'failed': True, 'error': error}, 'usage': {}}
        return ''

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        yield SimpleNamespace(authority=authority, chat=chat.session_id, errors=errors, home=tmp_path,
                              connection=SimpleNamespace(authority=authority, actor=owner))
    finally:
        store._db.close()


def _admissions(bot):
    return list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)


def _quiet(bot):
    """Nothing left that could admit another row: no receipt task, mailbox operation or drain runs."""
    authority = bot.authority
    return (not any(not task.done() for task in getattr(authority, '_bot_receipt_tasks', ()))
            and not getattr(authority, '_bot_mailbox_operations', None)
            and all(live.task is None or live.task.done() for live in authority.sessions.values()))


async def _settle_quietly(bot):
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: _quiet(bot), timeout=10, what='receipt tasks and drains to go idle')


async def _fail_original_while_draining(bot, params):
    """Leave one real delivery failed and retry-eligible, with recovery owning the retry."""
    import gateway.session_finite as finite
    from gateway.session_bot import deliver

    started, gate = asyncio.Event(), asyncio.Event()
    execute = finite.execute_finite_admission

    async def held(authority, ref, row):
        if not row['request_id'].endswith(':retry'):
            started.set()
            await gate.wait()
        return await execute(authority, ref, row)

    finite.execute_finite_admission = held
    try:
        bot.errors[:] = ['Error code: 429 - rate limit exceeded']
        first = await deliver(bot.connection, params)
        await asyncio.wait_for(started.wait(), 5)
        bot.authority.runner._draining = True
        gate.set()
        await _settled(bot, 1)
        await _settle_quietly(bot)
        return first
    finally:
        finite.execute_finite_admission = execute


async def _recover_with_original_row_override(bot, monkeypatch, override):
    """Override only the retry's post-eligibility admission lookup, preserving real delivery flow."""
    from gateway import session_bot

    real_get = session_bot.get_session_admission
    calls = 0

    def get(db, *, admission_id):
        nonlocal calls
        calls += 1
        row = real_get(db, admission_id=admission_id)
        # Recovery reads the original result first, checks retry eligibility second, then binds the
        # committed attachment row. This is the lookup whose absence/malformed value must refuse.
        return override(row) if calls == 3 else row

    monkeypatch.setattr(session_bot, 'get_session_admission', get)
    from gateway.session_bot import recover_bot_deliveries
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    return calls


async def _settled(bot, count):
    from gateway.session_bot import _result
    from tools.bot_live_delivery import read_delivery_result
    for _ in range(200):
        await asyncio.sleep(0.02)
        rows = _admissions(bot)
        if len(rows) >= count and all(r['status'] == 'terminal' for r in rows):
            await asyncio.sleep(0.05)  # let the receipt task observe the settle
            return _result(bot.authority, (await asyncio.to_thread(read_delivery_result, bot.home, KEY)))
    raise AssertionError(f'admissions never settled: {_admissions(bot)}')


@pytest.mark.asyncio
async def test_transient_failure_retries_exactly_once_and_the_receipt_follows_the_retry(bot):
    from gateway.session_bot import deliver, recover_bot_deliveries
    params = dict(id=KEY, profile='default', message='ping')
    bot.errors[:] = ['Error code: 429 - rate limit exceeded']
    first = await deliver(bot.connection, params)
    receipt = await _settled(bot, 2)
    assert receipt['status'] == 'settled' and receipt['reply'] == 'pong', receipt
    assert receipt['admission_id'] == first['admission_id'] and receipt['retry_admission_id']
    assert [r['request_id'] for r in _admissions(bot)] == ['bot:' + KEY, 'bot:' + KEY + ':retry']
    # A repeated delivery of the same id and an owner restart re-read the one retry; never a third.
    again = await deliver(bot.connection, params)
    assert again['status'] == 'settled' and again['reply'] == 'pong'
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    assert len(_admissions(bot)) == 2


@pytest.mark.asyncio
async def test_bot_image_delivery_commits_replays_and_rejects_changed_staging_bytes(bot):
    """The Bot relay carries an image through the same local automation envelope as its text.

    Replaying the delivery is idempotent, while replacing the still-staged file at the same path
    cannot silently change the already admitted bytes.
    """
    from pathlib import Path

    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_automation import restore_local_automation
    from gateway.session_bot import deliver
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import get_session_admission, RuntimeStoreError

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'look at this',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    first = await deliver(bot.connection, params)
    row = get_session_admission(bot.authority.db, admission_id=first['admission_id'])
    assert row is not None and row['payload']['attachments_v1']['media_types'] == ['image/png']
    restored = restore_local_automation(bot.authority, SessionRef('default', bot.chat), row)
    assert restored.media_types == ['image/png'] and len(restored.media_urls) == 1
    await _settled(bot, 1)
    assert (await deliver(bot.connection, params))['admission_id'] == first['admission_id']

    Path(staged).write_bytes(ONE_PX_PNG + b'changed')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await deliver(bot.connection, params)


@pytest.mark.asyncio
async def test_bot_image_retry_recaptures_exact_bytes_after_terminal_cleanup(bot):
    """A transient Bot failure gets one derived admission with the original image identity.

    The first terminal admission may have released its native bytes before the retry runs; the
    trusted producer may recapture only the unchanged staging file and preserves the digest.
    """
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_bot import deliver

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'retry this image',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    bot.errors[:] = ['Error code: 429 - rate limit exceeded']
    first = await deliver(bot.connection, params)
    receipt = await _settled(bot, 2)
    rows = _admissions(bot)
    assert receipt['status'] == 'settled' and receipt['retry_admission_id']
    assert first['admission_id'] == rows[0]['admission_id']
    assert [row['request_id'] for row in rows] == ['bot:' + KEY, 'bot:' + KEY + ':retry']
    assert all(row['payload']['attachments_v1']['media_types'] == ['image/png'] for row in rows)


@pytest.mark.asyncio
async def test_bot_image_retry_lost_staged_and_retained_bytes_is_durably_refused(bot):
    """When neither exact source remains, recovery records one final identity conflict."""
    from pathlib import Path

    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_bot import recover_bot_deliveries
    from hermes_state_runtime import get_session_admission
    from tools.bot_live_delivery import read_delivery_result

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'lost image bytes',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    first = await _fail_original_while_draining(bot, params)
    original = get_session_admission(bot.authority.db, admission_id=first['admission_id'])
    assert original is not None
    for reference in original['payload']['attachments_v1']['media']:
        Path(reference['path']).unlink(missing_ok=True)
    Path(staged).unlink(missing_ok=True)
    bot.authority.runner._draining = False

    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    receipt = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert (receipt['status'], receipt['admission_id'], receipt['error']) == (
        'failed', first['admission_id'], 'Error code: 429 - rate limit exceeded'), receipt
    assert receipt['retry']['refused'] == 'admission_conflict', receipt
    assert not receipt.get('retry_admission_id')
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]
    refusal = receipt['retry']

    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    again = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert (again['retry'], again['admission_id'], again['error']) == (
        refusal, first['admission_id'], receipt['error']), again
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_bot_image_retry_changed_staging_refusal_is_persisted_by_recovery(bot):
    """Recovery refuses a changed staged path and a second recovery cannot admit it."""
    from pathlib import Path

    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_bot import recover_bot_deliveries
    from tools.bot_live_delivery import read_delivery_result

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'changed during recovery',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    await _fail_original_while_draining(bot, params)
    Path(staged).write_bytes(ONE_PX_PNG + b'changed-before-recovery')
    bot.authority.runner._draining = False
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    receipt = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert receipt['retry']['refused'] == 'admission_conflict', receipt
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]
    retry_marker = receipt['retry']

    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    again = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert again['retry'] == retry_marker
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_bot_image_retry_missing_admission_row_is_refused_by_recovery(bot, monkeypatch):
    """A missing original row after eligibility is a durable conflict, never ordinary recapture."""
    from gateway.platforms.base import cache_image_from_bytes
    from tools.bot_live_delivery import read_delivery_result

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'missing original row',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    first = await _fail_original_while_draining(bot, params)
    bot.authority.runner._draining = False
    calls = await _recover_with_original_row_override(bot, monkeypatch, lambda _row: None)
    receipt = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert calls >= 3 and receipt['retry']['refused'] == 'admission_conflict', receipt
    assert first['admission_id'] == _admissions(bot)[0]['admission_id']
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_bot_image_retry_malformed_manifest_is_refused_by_recovery(bot, monkeypatch):
    """A malformed committed reference is terminal evidence, not permission to recapture."""
    from gateway.platforms.base import cache_image_from_bytes
    from tools.bot_live_delivery import read_delivery_result

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'malformed original manifest',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    await _fail_original_while_draining(bot, params)
    bot.authority.runner._draining = False

    def malformed(row):
        return {**row, 'payload': {**row['payload'], 'attachments_v1': {
            'media': [{'path': '/outside/foreign.png', 'sha256': '0' * 64, 'size': 1}],
            'media_types': ['image/png'],
        }}}

    calls = await _recover_with_original_row_override(bot, monkeypatch, malformed)
    receipt = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert calls >= 3 and receipt['retry']['refused'] == 'admission_conflict', receipt
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_bot_image_retry_toctou_refusal_is_persisted_by_recovery(bot, monkeypatch):
    """A replacement after identity hashing fails the recapture and is durably refused."""
    from pathlib import Path

    from gateway import session_ingress_media
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_bot import recover_bot_deliveries
    from hermes_state_runtime import RuntimeStoreError
    from tools.bot_live_delivery import read_delivery_result

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'replace during recovery',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    await _fail_original_while_draining(bot, params)
    bot.authority.runner._draining = False
    original_hashes = session_ingress_media._attachment_hashes
    calls = 0

    def hash_then_replace(paths):
        nonlocal calls
        calls += 1
        hashes = original_hashes(paths)
        Path(staged).write_bytes(ONE_PX_PNG + b'replaced-after-identity')
        return hashes

    def committed_media_missing(_references):
        raise RuntimeStoreError('media_missing')

    monkeypatch.setattr(session_ingress_media, '_attachment_hashes', hash_then_replace)
    monkeypatch.setattr(session_ingress_media, 'restore_native_media', committed_media_missing)
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    receipt = await asyncio.to_thread(read_delivery_result, bot.home, KEY)
    assert calls == 1 and receipt['retry']['refused'] == 'admission_conflict', receipt
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_bot_image_retry_rejects_changed_staging_before_derived_capture(bot):
    """A derived retry is bound to the first admission's committed digest, not a mutable staging path."""
    from pathlib import Path

    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_bot import deliver
    from gateway.session_ingress_media import admit_attachments
    from hermes_state_runtime import RuntimeStoreError, get_session_admission

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'changed before retry',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    first = await deliver(bot.connection, params)
    await _settled(bot, 1)
    row = get_session_admission(bot.authority.db, admission_id=first['admission_id'])
    Path(staged).write_bytes(ONE_PX_PNG + b'changed-before-retry')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit_attachments(params['attachments'], admitted=lambda: row, restore_committed=True)


@pytest.mark.asyncio
async def test_bot_image_retry_rejects_replacement_between_identity_and_capture(bot, monkeypatch):
    """Recapture revalidates the bytes after the identity check, closing a pathname replacement race."""
    from pathlib import Path

    from gateway import session_ingress_media
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_bot import deliver
    from hermes_state_runtime import RuntimeStoreError, get_session_admission

    staged = cache_image_from_bytes(ONE_PX_PNG, '.png')
    params = {'id': KEY, 'profile': 'default', 'message': 'replace during capture',
              'attachments': [{'path': staged, 'mime': 'image/png'}]}
    first = await deliver(bot.connection, params)
    await _settled(bot, 1)
    row = get_session_admission(bot.authority.db, admission_id=first['admission_id'])
    original_hashes = session_ingress_media._attachment_hashes

    def hash_then_replace(paths):
        hashes = original_hashes(paths)
        Path(staged).write_bytes(ONE_PX_PNG + b'replaced-between-check-and-capture')
        return hashes

    def committed_media_missing(_references):
        raise RuntimeStoreError('media_missing')

    monkeypatch.setattr(session_ingress_media, '_attachment_hashes', hash_then_replace)
    monkeypatch.setattr(session_ingress_media, 'restore_native_media', committed_media_missing)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        session_ingress_media.admit_attachments(
            params['attachments'], admitted=lambda: row, restore_committed=True)


@pytest.mark.asyncio
async def test_legacy_migration_ignores_malformed_unversioned_attachments(tmp_path, monkeypatch):
    """Old text tickets may contain foreign extras, but migration must never route them as images."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from gateway import session_bot

    key = 'd' * 32
    path = tmp_path / f'{key}.json'
    legacy = {
        'id': key, 'delivery_id': key, 'status': 'queued', 'message': 'legacy text',
        'created_at': 1, 'sequence': 1,
        'owner': {'profile_home': str(tmp_path), 'session_id': 'sid', 'principal_id': 'uid:1000'},
        'attachments': [{'path': '/outside/foreign.bin', 'mime': 'application/octet-stream', 'extra': True}],
        'attachments_v1': {'media': [{'path': '/outside/foreign.bin'}], 'media_types': ['image/png']},
        'native_text_v1': {'media': ['/outside/foreign.bin']}, 'foreign_extra': 'drop me',
    }
    captured = {}

    async def scan(_authority, _home):
        return tmp_path, [(path, legacy)], None

    async def target(_authority, _actor):
        return SimpleNamespace(session_id='sid'), SimpleNamespace(), SimpleNamespace(session_id='sid')

    async def admit(*_args, **kwargs):
        captured.update(kwargs)

    @contextmanager
    def no_active(*_args, **_kwargs):
        yield False

    monkeypatch.setattr(session_bot, '_scan_owner_receipts', scan)
    monkeypatch.setattr(session_bot, '_target', target)
    monkeypatch.setattr(session_bot, '_admit', admit)
    monkeypatch.setattr('hermes_cli.active_sessions.active_session_liveness_guard', no_active)

    authority = SimpleNamespace(db=SimpleNamespace(get_compression_tip=lambda _sid: 'sid'))
    await session_bot._migrate(authority, SimpleNamespace(), tmp_path, tmp_path)
    assert captured['legacy']['message'] == 'legacy text'
    assert 'attachments' not in captured['legacy']
    assert 'attachments_v1' not in captured['legacy']
    assert set(captured['legacy']) <= session_bot._LEGACY_TEXT_FIELDS
    assert 'attachments' not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize('attachments', [
    [],
    [{'path': '/tmp/image.png'}],
    [{'path': '/tmp/image.png', 'mime': 'text/plain'}],
    [{'path': '/tmp/image.png', 'mime': 'image/png', 'extra': True}],
])
async def test_bot_image_delivery_rejects_malformed_attachment_shape_before_admission(bot, attachments):
    from gateway.session_bot import deliver
    from hermes_state_runtime import RuntimeStoreError

    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await deliver(bot.connection, {'id': KEY, 'profile': 'default', 'message': 'bad',
                                       'attachments': attachments})
    assert _admissions(bot) == []


@pytest.mark.asyncio
async def test_retry_that_fails_again_and_non_transient_failures_are_never_replayed(bot):
    from gateway.session_bot import deliver, recover_bot_deliveries
    # The retry itself fails transiently: the sender sees that failure, no second retry.
    bot.errors[:] = ['HTTP 503 server error', 'HTTP 503 server error']
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    receipt = await _settled(bot, 2)
    assert receipt['status'] == 'failed' and receipt['retry_admission_id']
    # The relay's Desktop forwards ``reason`` to the sender verbatim (relay.ts postReply).
    assert receipt['reason'] == 'provider_server_error' and '503' in receipt['error'], receipt
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    assert len(_admissions(bot)) == 2
    # Auth failures do not classify transient: settled failed, never retried.
    other = 'd' * 32
    bot.errors[:] = ['Error code: 401 - invalid api key']
    await deliver(bot.connection, dict(id=other, profile='default', message='auth'))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if all(r['status'] == 'terminal' for r in _admissions(bot)) and len(_admissions(bot)) == 3:
            break
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    assert [r['request_id'] for r in _admissions(bot)][2:] == ['bot:' + other]
    from gateway.session_bot import _result
    from tools.bot_live_delivery import read_delivery_result
    auth = _result(bot.authority, (await asyncio.to_thread(read_delivery_result, bot.home, other)))
    assert (auth['status'], auth['reason']) == ('failed', 'provider_auth_or_access'), auth


@pytest.mark.asyncio
async def test_context_overflow_retries_once_unless_the_failed_dm_is_still_an_open_tail(bot):
    """Main re-ran an overflowed delivery once (compress-then-resume: the re-run's own preflight
    compaction does the compress). The owner retry keeps that for an overflow that left no open DM
    row, and refuses (recorded, typed) when the DM is still the durable tail: the owner execution has
    no adoption seam, and a second copy would merge into the unanswered one."""
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import read_delivery_result
    bot.errors[:] = ["This model's maximum context length is 200000 tokens"]
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    receipt = await _settled(bot, 2)
    assert (receipt['status'], receipt['reply']) == ('settled', 'pong'), receipt

    other = 'e' * 32
    bot.errors[:] = ["This model's maximum context length is 200000 tokens"]
    bot.authority.db.append_message(bot.chat, 'user', content='overflowed dm')
    await deliver(bot.connection, dict(id=other, profile='default', message='overflowed dm'))
    for _ in range(100):
        await asyncio.sleep(0.02)
        record = (await asyncio.to_thread(read_delivery_result, bot.home, other))
        if (record.get('retry') or {}).get('refused'):
            break
    assert record['retry']['refused'] == 'open_user_tail' and len(_admissions(bot)) == 3
    assert (record['status'], record['reason']) == ('failed', 'context_overflow'), record


@pytest.mark.asyncio
async def test_peer_dm_waiter_follows_the_retry_admission_to_its_answer(bot):
    """``hermes peer dm`` waits on the delivery, not on the original admission id: once that
    fails transiently its future never fires again, and the answer is the retry's."""
    from gateway.platforms.api_server_bot_chat import await_peer_receipt
    from gateway import session_bot
    from gateway.session_bot import deliver
    from tests.waiting import async_wait_until
    gate = asyncio.Event()
    import gateway.session_finite as finite
    execute = finite.execute_finite_admission
    wait_admission, followed = session_bot.peer_wait_admission, []

    def peer_wait_admission(authority, record):
        live = wait_admission(authority, record)
        followed.append(live)
        return live

    async def held(authority, ref, row):
        if row['request_id'].endswith(':retry'):
            await gate.wait()
        return await execute(authority, ref, row)

    finite.execute_finite_admission = held
    session_bot.peer_wait_admission = peer_wait_admission
    try:
        bot.errors[:] = ['Error code: 429 - rate limit exceeded']
        first = await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
        waiting = asyncio.create_task(await_peer_receipt(bot.authority, first, 20))
        for _ in range(200):
            await asyncio.sleep(0.02)
            if len(_admissions(bot)) == 2:
                break
        retry_id = _admissions(bot)[1]['admission_id']
        # The waiter re-read the delivery after the original failed and now follows the held retry.
        await async_wait_until(lambda: (retry_id, False) in followed or waiting.done(), timeout=10,
                               what='the peer waiter to follow the retry admission')
        assert not waiting.done(), waiting.result()
        gate.set()
        receipt = await asyncio.wait_for(waiting, 5)
    finally:
        finite.execute_finite_admission = execute
        session_bot.peer_wait_admission = wait_admission
    assert (receipt['status'], receipt['reply']) == ('settled', 'pong'), receipt
    assert receipt['retry_admission_id']


@pytest.mark.asyncio
async def test_receipt_file_stays_pending_while_a_draining_owner_leaves_the_retry_to_its_successor(bot):
    """Senders wait on the published receipt file, not an authority RPC. An original that failed
    transiently while the owner drains is interim there (``claimed``), never the final ``failed``:
    the restarted owner's recovery admits the one retry and the same wait gets its answer."""
    from gateway.session_bot import deliver, recover_bot_deliveries
    from tools.bot_live_delivery import await_delivery_async
    import gateway.session_finite as finite
    started, gate = asyncio.Event(), asyncio.Event()
    execute = finite.execute_finite_admission

    async def held(authority, ref, row):
        if not row['request_id'].endswith(':retry'):
            started.set()
            await gate.wait()
        return await execute(authority, ref, row)

    finite.execute_finite_admission = held
    try:
        bot.errors[:] = ['Error code: 429 - rate limit exceeded']
        await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
        await asyncio.wait_for(started.wait(), 5)
        bot.authority.runner._draining = True  # shutdown starts while the original runs
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.02)
            if _admissions(bot)[0]['status'] == 'terminal':
                break
        await _settle_quietly(bot)  # the original's receipt task has published its interim state
        interim = await await_delivery_async(bot.home, KEY, 0.2)
        assert interim['status'] == 'claimed' and len(_admissions(bot)) == 1, interim
        waiting = asyncio.create_task(await_delivery_async(bot.home, KEY, 10))
        bot.authority.runner._draining = False  # the successor owner's startup recovery
        await recover_bot_deliveries(bot.authority)
        receipt = await asyncio.wait_for(waiting, 10)
    finally:
        finite.execute_finite_admission = execute
    assert (receipt['status'], receipt['reply']) == ('settled', 'pong'), receipt
    assert [r['request_id'] for r in _admissions(bot)] == ['bot:' + KEY, 'bot:' + KEY + ':retry']


@pytest.mark.asyncio
async def test_mailbox_file_lock_held_elsewhere_never_stalls_the_owner_loop(bot):
    """The cross-process mailbox lock guards one receipt write, taken off the event loop: a DM
    runner (another process) holding it delays only that write, never every other session, socket
    and delivery on the owner's loop, and never across an admission await."""
    import sys
    import threading
    import time
    from gateway.session_bot import deliver
    from hermes_cli.active_sessions import _FileLock
    from tools.bot_live_delivery import _locked, _root
    with _locked(bot.home):
        pass
    from tests.waiting import async_wait_until
    held, release = threading.Event(), threading.Event()

    def foreign_holder():
        with _FileLock(_root(bot.home) / '.lock'):
            held.set()
            release.wait(10)  # the test releases it as soon as the loop has been measured
    holder = threading.Thread(target=foreign_holder, daemon=True)
    holder.start()
    assert held.wait(5)

    def queued_behind_holder():
        # Another thread (the off-loop receipt write) is parked in flock behind the holder.
        for ident, frame in sys._current_frames().items():
            if ident in (holder.ident, threading.get_ident()):
                continue
            while frame is not None:
                if frame.f_code.co_name == '_flock':
                    return True
                frame = frame.f_back
        return False
    delivering = asyncio.create_task(deliver(bot.connection, dict(id=KEY, profile='default', message='ping')))
    started = time.monotonic()
    # The loop keeps polling while the write waits off it; a write on the loop would block every poll.
    await async_wait_until(queued_behind_holder, timeout=10, what='the mailbox write to wait off the loop')
    loop_stall = time.monotonic() - started
    release.set()
    receipt = await asyncio.wait_for(delivering, 5)
    assert loop_stall < 0.5, f'event loop blocked {loop_stall:.2f}s behind a foreign mailbox lock holder'
    assert receipt['status'] in {'queued', 'claimed', 'settled'}, receipt
