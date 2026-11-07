"""A refused WebSocket request leaves a trace in ``agent.log``.

Refusals (``4001 invalid_params``, ``permission_denied``, ``stale_generation``) are answered to the
client and written nowhere else, so a client whose request was rejected is indistinguishable in the
logs from a client that never sent it. That blind spot misdiagnosed the Mobile clarify hang twice —
the phone's answer was refused and ``agent.log`` stayed empty. The hook logs the method, the code and
the parameter NAMES, plus the value of a few short enum-ish fields (``source``, ``model``,
``provider``, ``operation``) whose value names the client bug; user text never appears.
"""
import logging

from tui_gateway.ws import _log_refusal


def _refusal(code, message):
    return {'jsonrpc': '2.0', 'error': {'code': code, 'message': message}, 'id': 'p1'}


def test_a_refused_request_is_logged_with_its_method_and_parameter_names(caplog):
    req = {'jsonrpc': '2.0', 'id': 'p1', 'method': 'clarify.respond',
           'params': {'session_id': 'local-s', 'response': 'синий', 'answer': 'синий'}}
    with caplog.at_level(logging.WARNING, logger='tui_gateway.ws'):
        _log_refusal('127.0.0.1:1', 'clarify.respond', req, _refusal(4001, 'invalid_params'))
    assert 'ws request refused' in caplog.text
    assert 'clarify.respond' in caplog.text and '4001' in caplog.text and 'invalid_params' in caplog.text
    # Only the key set: the answer text itself must never reach the log.
    assert 'answer' in caplog.text and 'синий' not in caplog.text


def test_a_successful_reply_logs_nothing(caplog):
    req = {'jsonrpc': '2.0', 'id': 'p1', 'method': 'session.resume', 'params': {'session_id': 's'}}
    with caplog.at_level(logging.WARNING, logger='tui_gateway.ws'):
        _log_refusal('peer', 'session.resume', req, {'jsonrpc': '2.0', 'id': 'p1', 'result': {}})
    assert caplog.text == ''


def test_a_request_without_params_still_reports_the_refusal(caplog):
    with caplog.at_level(logging.WARNING, logger='tui_gateway.ws'):
        _log_refusal('peer', 'session.list', {'id': 'p9'}, _refusal(4001, 'permission_denied'))
    assert 'permission_denied' in caplog.text and 'param_keys=[]' in caplog.text


def test_safe_routing_values_are_logged_and_user_text_never_is(caplog):
    # A refused ``source``/``model`` names the client bug outright, so those enum-ish fields are
    # logged by value; the prompt in the same request must not travel with them.
    req = {'jsonrpc': '2.0', 'id': 'p2', 'method': 'session.create',
           'params': {'source': 'desktop', 'model': 'gpt-6', 'prompt': 'мой личный текст',
                      'session_id': 'local-s'}}
    with caplog.at_level(logging.WARNING, logger='tui_gateway.ws'):
        _log_refusal('peer', 'session.create', req, _refusal(4001, 'invalid_params'))
    assert "'source': 'desktop'" in caplog.text and "'model': 'gpt-6'" in caplog.text
    # ... while the free-text parameter appears only in the key list, never with its value.
    assert 'мой личный текст' not in caplog.text


def test_a_malformed_reply_is_ignored_rather_than_breaking_the_reply(caplog):
    with caplog.at_level(logging.WARNING, logger='tui_gateway.ws'):
        _log_refusal('peer', 'session.list', None, None)
        _log_refusal('peer', 'session.list', {'id': 1}, {'error': 'not-a-dict'})
    assert caplog.text == ''
