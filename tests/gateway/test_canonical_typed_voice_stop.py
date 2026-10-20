"""TUI-26: a bare stop phrase TYPED into a voice chat on the shared gateway ends the voice chat
instead of reaching the agent. Ink sends prompts to the canonical ``prompt.submit``; the request
goes through the real ``AuthorityConnection.dispatch`` (contract parse, handler, result check)."""
import pytest

from gateway import run_api
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_prompt_queued_controls import _call, _connection
from tests.gateway.test_session_authority_cancel_settlement import _authority


@pytest.fixture
def voice_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gateway.config import Platform
    from tui_gateway import server
    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.LOCAL)  # TUI-26 is the local voice chat
    # The owner's voice mode is keyed by the session's source (``_voice_key_for_source``). Ending the
    # LOCAL push-to-talk capture is ``set_local_voice_mode``'s job, pinned in
    # tests/tui_gateway/test_canonical_voice_mode_writers.py.
    authority.sessions['s'].source.chat_id = 's'
    # A LOCAL session belongs to its launching user: the viewer ``_connection`` attaches as.
    from gateway.session_identity import authenticated_subject
    authority.sessions['s'].source.user_id = authenticated_subject({'user_id': 'human'})
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)  # admit only; never run a turn
    runner = authority.runner
    runner._voice_mode = {}
    runner._voice_key_for_source = lambda src: src.chat_id
    runner._save_voice_modes = lambda: None
    runner.session_authority = authority
    monkeypatch.setattr(run_api, '_ACTIVE_RUNNER', runner)
    monkeypatch.setattr(server, '_tts_stream_stop', lambda user_barge=False: None)
    return SimpleNamespace(db=db, authority=authority, runner=runner)


def _ledger(db):
    return [row['request_id'] for row in list_session_admissions(db, session_id='s', pending_only=False)]


@pytest.mark.asyncio
@pytest.mark.parametrize('typed', ['stop', '  Stop! '])
async def test_typed_stop_phrase_ends_the_voice_chat_and_admits_nothing(voice_owner, typed):
    voice_owner.runner._voice_mode['s'] = 'voice_only'
    with voice_owner.db:
        reply = await _call(_connection(voice_owner.authority), 'prompt.submit', text=typed, submission_id='r1')
    assert reply.get('result') == {'voice_stopped': True}, reply
    assert voice_owner.runner._voice_mode['s'] == 'off'
    assert _ledger(voice_owner.db) == [], 'the stop phrase must never reach the agent'


@pytest.mark.asyncio
async def test_a_retried_stop_replays_voice_stopped_and_never_starts_a_turn(voice_owner):
    """Its reply lost, Ink retries the SAME submission (Alt+K). Voice is already off by then, so
    the stop-phrase check alone would admit ``stop`` as a turn: the request id answers it, also on
    a fresh connection (a reconnect)."""
    voice_owner.runner._voice_mode['s'] = 'voice_only'
    with voice_owner.db:
        first = await _call(_connection(voice_owner.authority), 'prompt.submit', text='stop', submission_id='r1')
        retry = await _call(_connection(voice_owner.authority), 'prompt.submit', text='stop', submission_id='r1')
        fresh = await _call(_connection(voice_owner.authority), 'prompt.submit', text='stop', submission_id='r2')
    assert first.get('result') == {'voice_stopped': True}, first
    assert retry.get('result') == {'voice_stopped': True}, retry
    assert fresh['result']['status'] == 'queued', 'a new stop with voice off is an ordinary turn'
    assert _ledger(voice_owner.db) == ['r2']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode, text', [('off', 'stop'), ('voice_only', 'stop doing that')])
async def test_other_text_or_voice_off_is_admitted_as_a_turn(voice_owner, mode, text):
    voice_owner.runner._voice_mode['s'] = mode
    with voice_owner.db:
        reply = await _call(_connection(voice_owner.authority), 'prompt.submit', text=text, submission_id='r1')
    assert reply['result']['status'] == 'queued', reply
    assert voice_owner.runner._voice_mode['s'] == mode
    assert len(list_session_admissions(voice_owner.db, session_id='s', pending_only=False)) == 1


@pytest.mark.asyncio
async def test_a_stranger_cannot_end_the_voice_chat(voice_owner):
    voice_owner.runner._voice_mode['s'] = 'voice_only'
    conn = _connection(voice_owner.authority)
    conn.subscriptions.pop('s')
    with voice_owner.db:
        reply = await _call(conn, 'prompt.submit', text='stop', submission_id='r1')
    assert reply['error']['data']['reason'] == 'permission_denied', reply
    assert voice_owner.runner._voice_mode['s'] == 'voice_only'


@pytest.mark.asyncio
async def test_a_messaging_lane_viewer_typing_stop_is_admitted_untouched(voice_owner):
    """TUI-26 is the LOCAL voice chat: a Telegram lane whose chat ran ``/voice on``, viewed from
    Desktop or Ink, keeps its voice mode and gets the typed text admitted as a turn."""
    from gateway.config import Platform
    voice_owner.authority.sessions['s'].source.platform = Platform.TELEGRAM
    voice_owner.runner._voice_mode['s'] = 'voice_only'
    with voice_owner.db:
        reply = await _call(_connection(voice_owner.authority), 'prompt.submit', text='stop', submission_id='r1')
    assert reply.get('result', {}).get('status') == 'queued', reply
    assert voice_owner.runner._voice_mode['s'] == 'voice_only'
    assert _ledger(voice_owner.db) == ['r1']


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['session.redirect', 'session.steer'])
async def test_a_stop_typed_as_a_busy_correction_ends_the_voice_chat_and_reaches_no_agent(voice_owner, method):
    """Ink's ``interrupt``/``steer`` busy modes send text typed during a running turn through
    ``session.redirect``/``session.steer``, not ``prompt.submit``: the same stop decision covers them."""
    voice_owner.runner._voice_mode['s'] = 'voice_only'
    agent_calls = []
    voice_owner.authority.steer_agent = lambda *a: agent_calls.append(a) or (None, True)
    with voice_owner.db:
        reply = await _call(_connection(voice_owner.authority), method, text='stop', execution_generation=0)
    assert reply.get('result') == {'voice_stopped': True}, reply
    assert voice_owner.runner._voice_mode['s'] == 'off'
    assert agent_calls == [], 'the stop phrase must never reach the running turn'


@pytest.mark.asyncio
async def test_other_busy_corrections_still_reach_the_turn(voice_owner):
    voice_owner.runner._voice_mode['s'] = 'voice_only'
    voice_owner.authority.steer_agent = lambda *a: (None, True)
    voice_owner.authority.check_approval_generation = lambda *a: None
    with voice_owner.db:
        reply = await _call(_connection(voice_owner.authority), 'session.steer', text='stop doing that',
                            execution_generation=0)
    assert reply['result']['status'] == 'queued', reply
    assert voice_owner.runner._voice_mode['s'] == 'voice_only'


def test_the_canonical_contract_declares_the_voice_stopped_reply():
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    contract = CANONICAL_METHODS['prompt.submit']
    contract.result.model_validate({'voice_stopped': True})
    with pytest.raises(Exception):
        contract.result.model_validate({'voice_stopped': False})
