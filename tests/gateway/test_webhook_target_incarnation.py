"""Admission-time target identity and the existing native recipient door.

Real profile directories/stores and authority admission; no model/network claim.
"""
import json
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer

pytest_plugins = ['tests.gateway.test_session_bot_retry']


@pytest.mark.asyncio
async def test_source_admission_captures_target_before_final_materialization():
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        payload = json.loads(row['payload_json'])['native_text_v1']
        pin = payload.get('webhook_target')
        assert pin is not None, 'pin must exist before the final delivery intent'
        home = Path(authority.db.db_path).resolve().parent
        assert pin['home'] == str(home)
        assert pin['directory'] == [home.stat().st_dev, home.stat().st_ino]
        assert pin['store']['file'] == [(home / 'state.db').stat().st_dev, (home / 'state.db').stat().st_ino]


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True])
async def test_native_door_accepts_exact_incarnation_then_refuses_directory_replacement(bot, monkeypatch, legacy):
    from gateway.session_bot import deliver
    from tests.gateway.test_bot_receipt_refusal import _rpc_connection
    home = bot.home.resolve()
    if legacy:
        bot.authority.db._write_sql("DELETE FROM state_meta WHERE key='store_instance_id'")
    meta = bot.authority.db._read_one("SELECT value FROM state_meta WHERE key='store_instance_id'")
    pin = dict(home=str(home), directory=[home.stat().st_dev, home.stat().st_ino],
               store={'file': [(home / 'state.db').stat().st_dev, (home / 'state.db').stat().st_ino],
                      'instance': meta['value'] if meta else None})
    params = dict(id='f' * 32, profile='default', message='one exact pinned delivery', expected_incarnation=pin)
    connection = _rpc_connection(bot)
    monkeypatch.setattr(connection, '_sibling_for', lambda profile: None)
    answer = await connection.dispatch({'id': 1, 'method': 'bot_relay.deliver', 'params': params})
    assert 'result' in answer, answer
    receipt = answer['result']
    assert receipt['admission_id']
    from tests.gateway.test_session_bot_retry import _settle_quietly
    await _settle_quietly(bot)
    moved = home.with_name(home.name + '-previous')
    home.rename(moved)
    home.mkdir()
    try:
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await deliver(bot.connection, params)
        assert not (home / 'state.db').exists(), 'refusal must precede target/store creation'
    finally:
        home.rename(home.with_name(home.name + '-refused-replacement'))
        moved.rename(home)


@pytest.mark.asyncio
async def test_native_refuses_replacement_after_first_check_before_chat_creation(bot, monkeypatch):
    from gateway.session_bot import deliver
    from gateway import webhook_target, run
    monkeypatch.setattr(run, '_load_gateway_config', lambda *args: {})
    from tests.gateway.test_bot_receipt_refusal import _rpc_connection
    home = bot.home.resolve()
    meta = bot.authority.db._read_one("SELECT value FROM state_meta WHERE key='store_instance_id'")
    pin = dict(home=str(home), directory=[home.stat().st_dev, home.stat().st_ino],
        store={'file': [(home / 'state.db').stat().st_dev, (home / 'state.db').stat().st_ino],
               'instance': meta['value'] if meta else None})
    bot.authority.db.set_session_title(bot.chat, 'Previous Chat')
    assert bot.authority.db.get_session_by_title('Bot Chat') is None
    original = webhook_target.require_incarnation
    moved = home.with_name(home.name + '-before-admission')
    checks = []
    def replace_after_check(authority, expected):
        checks.append(expected)
        original(authority, expected)
        if len(checks) == 1:
            home.rename(moved)
            home.mkdir()
    monkeypatch.setattr(webhook_target, 'require_incarnation', replace_after_check)
    try:
        from hermes_state_errors import StateDbReplacedError
        with pytest.raises(StateDbReplacedError):
            await deliver(_rpc_connection(bot), dict(id='e' * 32, profile='default',
                message='refuse a same-name replacement before admission', expected_incarnation=pin))
        # The existing open-writer continuity guard refuses even before the
        # second native incarnation check, during canonical chat creation.
        assert len(checks) == 1
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(moved / 'state.db')) as db:
            assert db.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert not (home / 'state.db').exists()
    finally:
        home.rename(home.with_name(home.name + '-refused-before-admission'))
        moved.rename(home)
