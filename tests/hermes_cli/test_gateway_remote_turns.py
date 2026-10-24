"""Ordinary native input is rendered by the existing viewer, never admitted again."""
import pytest

from tests.hermes_cli.test_cli_queue_gateway_fifo import _run_adopted
from tests.hermes_cli.test_gateway_agent import SID, frame
from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli, _play

TEXT = '/literal remote input'
ANSWER = 'remote answer'


@pytest.mark.parametrize('busy', [False, True])
def test_remote_turn_reaches_idle_or_busy_viewer_once_without_resubmission(busy):
    from hermes_cli.gateway_agent import GatewayAgent
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        cli._pending_input.put('local draft')
        start = frame('message.start', {'input': {'text': TEXT}}, admission='remote')
        _play(conn, client, [start,
            frame('message.complete', {'text': ANSWER, 'status': 'complete'}, admission='remote')])
        if busy:
            client.script = [frame('message.complete', {'text': 'local', 'status': 'complete'})]
            agent = GatewayAgent(connection=conn, session_id=SID)
            agent._pending_cli_user_message = {'role': 'user', 'content': 'local'}
            agent.run_conversation('local')
        submits = [m for m, _ in client.calls].count('prompt.submit')
        client.result = {'final_response': ANSWER}
        cli._drain_process_notifications('cli-idle')
        assert not cli._pending_input.empty()
        result = _run_adopted(cli, conn)
        assert result['final_response'] == ANSWER
        assert cli._pending_input.get_nowait() == 'local draft'
        assert [m for m, _ in client.calls].count('prompt.submit') == submits
        _play(conn, client, [start])
        cli._drain_process_notifications('cli-idle')
        assert cli._pending_input.empty(), 'replayed start painted twice'
    finally:
        conn.close()


def test_remote_input_projection_excludes_hidden_and_automation_content():
    from gateway.session_automation import admission_display
    assert admission_display({'text': TEXT}) == {'input': {'text': TEXT}}
    assert admission_display({'text': TEXT, 'display_v1': {'kind': 'hidden'}}) == {}
    assert admission_display({'text': TEXT, 'local_automation_v1': {'identity': 'private'}}) == {}


def test_completed_remote_turn_absent_from_replay_is_painted_after_reattach():
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.adopt_snapshot({'messages': [], 'admission_cursor': 5})
        state.adopt_snapshot({'messages': [], 'admission_cursor': 6})
        original_rpc = client.rpc

        async def rpc(method, **params):
            if method == 'prompt.history':
                client.calls.append((method, params))
                assert params['after_sequence'] == 5
                return {'admissions': [{'receipt': {'admission_id': 'remote', 'status': 'terminal',
                    'sequence': 6}, 'input': {'text': TEXT}}], 'last_sequence': 6, 'has_more': False}
            return await original_rpc(method, **params)

        client.rpc = rpc
        client.result = {'final_response': ANSWER}
        cli._drain_process_notifications('cli-idle')
        result = _run_adopted(cli, conn)
        assert result['final_response'] == ANSWER
        assert 'prompt.submit' not in [m for m, _ in client.calls]
        cli._drain_process_notifications('cli-idle')
        assert cli._pending_input.empty()
    finally:
        conn.close()


def test_disconnect_while_painting_keeps_the_admission_for_receipt_recovery():
    from hermes_cli.gateway_client import GatewayClientError
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        _play(conn, client, [frame('message.start', {'input': {'text': TEXT}}, admission='remote')])
        cli._drain_process_notifications('cli-idle')
        conn.events.put(GatewayClientError('fixture disconnect'))
        assert _run_adopted(cli, conn)['failed']
        assert 'remote' not in conn.state(SID).painted_admissions
        client.result = {'final_response': ANSWER}
        cli._drain_process_notifications('cli-idle')
        assert _run_adopted(cli, conn)['final_response'] == ANSWER
        assert 'prompt.submit' not in [method for method, _ in client.calls]
        assert not conn.state(SID).queued_inputs
    finally:
        conn.close()
