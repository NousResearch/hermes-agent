"""Bot receipts recover only their exact committed admission result."""
from types import SimpleNamespace

from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input,
    recover_session_inputs,
)
from gateway.session_results import retain_result
from gateway.session_bot import _result


def test_bot_reply_recovery_uses_exact_terminal_admission(tmp_path):
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('bot', source='gui')
    epoch = begin_runtime_epoch(db, instance_id='first')
    records = []
    for key, reply, settle in [('one', 'exact first reply', True),
                               ('two', 'later reply is not first', True),
                               ('three', 'uncommitted reply', False)]:
        admission = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                       request_id=key, payload={'text': key})
        row = claim_session_input(db, epoch=epoch, session_id='bot')
        # retain_result IS the settlement: the result commits with terminal status or not at all.
        if settle:
            retain_result(db, epoch=epoch, row=row, result={'result': {'final_response': reply}, 'usage': {}})
        records.append(dict(status='canonical', admission_id=admission['admission_id'],
                            delivery_id=key, profile_home=str(tmp_path), session_id='bot', message=key))
    db.close()
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='restart')
        recover_session_inputs(db, epoch=epoch)
        authority = SimpleNamespace(db=db)
        assert [_result(authority, record)['message'] for record in records] == ['one', 'two', 'three']
        assert _result(authority, records[0])['reply'] == 'exact first reply'
        assert _result(authority, records[0])['status'] == 'settled'
        assert _result(authority, records[1])['reply'] == 'later reply is not first'
        assert _result(authority, records[2])['status'] == 'ambiguous'
        assert _result(authority, records[2])['reply'] == ''
    finally:
        db.close()


def test_cancelled_queued_admission_is_a_terminal_cancelled_receipt_not_ambiguous(tmp_path):
    """A queued Bot admission cancelled before execution definitively did not run: after an
    owner restart, recovery and the exact delivery retry both report ``cancelled`` (terminal),
    never ``ambiguous``; a rejected admission without a result blob is ``failed``."""
    import asyncio
    from hermes_state_runtime import cancel_session_input, settle_session_input
    from tools.bot_live_delivery import _locked, _read, _write
    from gateway.session_bot import recover_bot_deliveries
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('bot', source='gui')
    epoch = begin_runtime_epoch(db, instance_id='first')
    running = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                  request_id='bot:' + 'a' * 32, payload={'text': 'running'})
    queued = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                 request_id='bot:' + 'b' * 32, payload={'text': 'queued'})
    row = claim_session_input(db, epoch=epoch, session_id='bot')
    assert row['admission_id'] == running['admission_id']
    cancel_session_input(db, epoch=epoch, admission_id=queued['admission_id'])
    settle_session_input(db, epoch=epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='rejected')
    records = {}
    with _locked(tmp_path) as root:
        for key, admission in (('a' * 32, running), ('b' * 32, queued)):
            records[key] = dict(status='canonical', admission_id=admission['admission_id'], delivery_id=key,
                                profile_home=str(tmp_path), session_id='bot', message=key, principal_id='owner')
            _write(root / f'{key}.json', records[key])
    db.close()
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='restart')
        recover_session_inputs(db, epoch=epoch)
        authority = SimpleNamespace(db=db, waiters={})
        with _locked(tmp_path) as root:
            (root / 'unreadable.json').write_text('{not-json', encoding='utf-8')
        asyncio.run(recover_bot_deliveries(authority))
        with _locked(tmp_path) as root:
            recovered = {key: _read(root / f'{key}.json')['status'] for key in records}
        assert recovered == {'a' * 32: 'failed', 'b' * 32: 'cancelled'}
        assert (tmp_path / 'runtime' / 'bot_live_delivery' / 'unreadable.json').exists()
        # The exact retry reads the same classification from the durable admission row.
        assert _result(authority, records['b' * 32])['status'] == 'cancelled'
        assert _result(authority, records['a' * 32])['status'] == 'failed'
    finally:
        db.close()


def test_settled_bot_receipt_relays_empty_reply_for_a_bare_silence_marker(tmp_path):
    """#110782: the authority receipt applies the gateway's silence rule — a successful bare
    marker reaches the sender as "", prose that merely mentions one is relayed verbatim."""
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    try:
        db.create_session('bot', source='gui')
        epoch = begin_runtime_epoch(db, instance_id='first')
        records = []
        for key, reply in [('quiet', ' *NO_REPLY* '), ('prose', 'The NO_REPLY marker means do not answer.')]:
            admission = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                           request_id=key, payload={'text': key})
            row = claim_session_input(db, epoch=epoch, session_id='bot')
            retain_result(db, epoch=epoch, row=row, result={'result': {'final_response': reply}, 'usage': {}})
            records.append(dict(status='canonical', admission_id=admission['admission_id'],
                                delivery_id=key, profile_home=str(tmp_path), session_id='bot', message=key))
        authority = SimpleNamespace(db=db)
        quiet, prose = (_result(authority, record) for record in records)
        assert quiet['status'] == prose['status'] == 'settled'
        assert quiet['reply'] == ''
        assert prose['reply'] == 'The NO_REPLY marker means do not answer.'
    finally:
        db.close()


def test_owner_startup_recovery_survives_a_denied_mailbox_chmod(tmp_path, monkeypatch):
    """#139485: an inherited Windows ACL denies FILE_WRITE_ATTRIBUTES on the mailbox dir, so
    ``chmod`` raises WinError 5. Owner startup recovery must still run and settle the receipts."""
    import asyncio
    from pathlib import Path
    from tools.bot_live_delivery import DELIVERY_DIR_NAME, _locked, _read, _write
    from gateway.session_bot import recover_bot_deliveries
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('bot', source='gui')
    epoch = begin_runtime_epoch(db, instance_id='first')
    settled = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                  request_id='bot:' + 'a' * 32, payload={'text': 'done'})
    row = claim_session_input(db, epoch=epoch, session_id='bot')
    retain_result(db, epoch=epoch, row=row, result={'result': {'final_response': 'hi'}, 'usage': {}})
    queued = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                 request_id='bot:' + 'b' * 32, payload={'text': 'waiting'})
    with _locked(tmp_path) as root:
        for key, admission in (('a' * 32, settled), ('b' * 32, queued)):
            _write(root / f'{key}.json', dict(status='canonical', admission_id=admission['admission_id'],
                                              delivery_id=key, profile_home=str(tmp_path.resolve()),
                                              session_id='bot', message=key, principal_id='owner'))
    db.close()
    mailbox_dir = tmp_path.resolve() / 'runtime' / DELIVERY_DIR_NAME
    real_chmod = Path.chmod

    def deny(target, mode, *, follow_symlinks=True):
        if Path(target) == mailbox_dir:
            raise PermissionError(13, '[WinError 5] Access is denied', str(target))
        return real_chmod(target, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, 'chmod', deny)
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='restart')
        recover_session_inputs(db, epoch=epoch)
        authority = SimpleNamespace(db=db, waiters={})

        async def boot():
            await recover_bot_deliveries(authority)
            return set(authority._bot_reply_watchers)

        watched = asyncio.run(boot())
        with _locked(tmp_path) as root:
            done = _read(root / ('a' * 32 + '.json'))
        assert (done['status'], done['reply']) == ('settled', 'hi')
        assert watched == {'b' * 32}
    finally:
        db.close()


def _deny_mailbox_lock(monkeypatch, home):
    """The reporters' ACL (#139485, #138872) as this token sees it: the mailbox dir exists, but
    ``.lock`` inside it cannot be opened (Windows: no ACE for the user on an Administrators-owned dir)."""
    import os
    from tools.bot_live_delivery import DELIVERY_DIR_NAME
    lock = home.resolve() / 'runtime' / DELIVERY_DIR_NAME / '.lock'
    lock.parent.mkdir(parents=True)
    real_open = os.open

    def deny(path, *args, **kwargs):
        if os.fspath(path) == str(lock):
            raise PermissionError(13, 'Permission denied', str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, 'open', deny)
    return lock.parent


def test_gateway_startup_skips_bot_recovery_when_the_mailbox_dir_is_unusable(tmp_path, monkeypatch, caplog):
    """#139485/#138872: an unusable mailbox dir pauses Bot Chat recovery instead of aborting
    ``start_gateway``; the log names the dir, its owner and the remedy."""
    import asyncio
    import logging
    from gateway import run_api, run_runtime
    root = _deny_mailbox_lock(monkeypatch, tmp_path)
    db = SessionDB(tmp_path / 'state.db')
    try:
        runner = SimpleNamespace(session_authority=SimpleNamespace(db=db, waiters={}),
                                 session_runtime_descriptor={})

        async def api(_runner):
            return SimpleNamespace(api_origin='http://127.0.0.1:1')

        monkeypatch.setattr(run_api, 'start_gateway_api', api)
        with caplog.at_level(logging.DEBUG):
            asyncio.run(run_runtime.start_gateway_runtime_api(runner))
        assert runner.session_runtime_descriptor['api_origin'] == 'http://127.0.0.1:1'
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING and 'not usable by this process' in r.getMessage()]
        assert len(warnings) == 1 and str(root) in warnings[0] and '(owner ' in warnings[0]
        assert 'To repair' in warnings[0]
    finally:
        db.close()


def test_unusable_mailbox_warns_once_per_dir_for_the_live_readers(tmp_path, monkeypatch, caplog):
    """The receipt readers re-enter ``_locked`` twice a second: one WARNING, DEBUG after, and a
    ``PermissionError`` the callers already handle (never a new exception type)."""
    import logging
    import pytest
    from tools import bot_live_delivery as mailbox
    root = _deny_mailbox_lock(monkeypatch, tmp_path)
    with caplog.at_level(logging.DEBUG, logger=mailbox.log.name):
        for _ in range(3):
            with pytest.raises(PermissionError):
                with mailbox._locked(tmp_path):
                    pass
    levels = [r.levelno for r in caplog.records if 'not usable by this process' in r.getMessage()]
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert root in mailbox._warned_unusable
