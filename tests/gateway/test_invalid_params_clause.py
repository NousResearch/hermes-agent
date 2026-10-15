"""Clauses #139140 does not name, in that PR's ``field`` + ``detail`` shape.

``reason``, ``str(exc)`` and the JSON-RPC message stay the bare code. ``field`` is the
check, ``detail`` is why. A client-chosen value is echoed only when the log redactor
would leave it alone. Logging is the boundary's job: warning when the check is named,
debug for a bare code, and never twice.
"""
import base64
import logging

import pytest

from hermes_state_runtime import RuntimeStoreError

# The placeholder the branch stores. Defined here so this file still collects on a
# tree that does not publish the name.
REDACTED_KEY = '<redacted-key>'

_CONTRACT = 'does not match the session contract'


def test_bare_reason_code_is_unchanged():
    """Control: a refusal with no clause still stringifies as the code alone."""
    exc = RuntimeStoreError('invalid_params')
    assert exc.reason == 'invalid_params'
    assert str(exc) == 'invalid_params'


def test_gateway_log_prefix_covers_the_refusal_logger():
    """Control: the boundary logger is a gateway component, so the line lands in gateway.log."""
    from hermes_logging import COMPONENT_PREFIXES
    assert any('gateway.session_controls'.startswith(prefix) for prefix in COMPONENT_PREFIXES['gateway'])


def test_a_named_check_does_not_move_the_reason_code():
    exc = RuntimeStoreError('invalid_params', field='run_budget', detail='must be a positive number of seconds')
    assert str(exc) == 'invalid_params' and exc.reason == 'invalid_params'
    assert exc.field == 'run_budget'
    assert exc.detail == 'must be a positive number of seconds'


def test_run_budget_keeps_the_create_path_sentence(tmp_path):
    """The create-path wording stays a sentence. Sealing must not blank a static detail."""
    from gateway.session_policy import build_policy

    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', 'run_budget': 0}, {})
    assert caught.value.field == 'run_budget'
    assert caught.value.detail == 'must be a positive number of seconds'
    assert str(caught.value) == 'invalid_params'


def test_null_toolset_is_a_refusal(tmp_path):
    """``toolsets: [null]`` is the bad element. It must not reach ``resolve_toolset``."""
    from gateway.session_policy import build_policy

    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', 'toolsets': [None]}, {})
    assert caught.value.field == 'toolsets'
    assert 'None' in caught.value.detail
    assert str(caught.value) == 'invalid_params'


def test_a_safe_toolset_name_is_still_echoed(tmp_path):
    from gateway.session_policy import build_policy

    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', 'toolsets': ['gone_plugin']}, {})
    assert caught.value.field == 'toolsets'
    assert 'gone_plugin' in caught.value.detail


def test_a_token_shaped_toolset_name_is_not_echoed(tmp_path):
    from gateway.session_policy import build_policy

    secret = 'sk_live_' + 'Ab12Cd34' * 3
    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', 'toolsets': [secret]}, {})
    blob = f'{caught.value.field} {caught.value.detail}'
    assert caught.value.field == 'toolsets'
    assert secret not in blob and 'Ab12Cd34' not in blob
    assert REDACTED_KEY in caught.value.detail


def test_a_token_shaped_reasoning_value_is_not_echoed(tmp_path):
    from gateway.session_policy import build_policy

    secret = 'hf_' + 'Ab12Cd34' * 4
    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', 'reasoning': secret + '_x'}, {})
    blob = f'{caught.value.field} {caught.value.detail}'
    assert caught.value.field == 'reasoning'
    assert secret not in blob
    assert REDACTED_KEY in caught.value.detail


def test_a_token_shaped_unknown_key_is_not_a_field_name(tmp_path):
    from gateway.session_policy import build_policy

    secret = 'ghp_' + 'AbCdEf1234' * 4
    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', secret: 1}, {})
    blob = f'{caught.value.field} {caught.value.detail}'
    assert secret not in blob
    assert caught.value.field == REDACTED_KEY
    assert caught.value.detail == 'unknown create field'


def test_a_plain_unknown_key_is_still_named(tmp_path):
    from gateway.session_policy import build_policy

    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({'cwd': str(tmp_path), 'source': 'cli', 'toolsets_extra': 1}, {})
    assert caught.value.field == 'params.toolsets_extra'
    assert caught.value.detail == 'unknown create field'


def test_an_editor_key_that_is_a_token_is_not_echoed():
    from gateway.session_local_editor import validate_editor

    secret = 'npm_' + 'AbCdEf1234' * 4
    with pytest.raises(RuntimeStoreError) as caught:
        validate_editor('acp', {secret: 1})
    assert secret not in f'{caught.value.field} {caught.value.detail}'
    assert caught.value.field == REDACTED_KEY
    assert caught.value.detail == 'unknown editor key'


def test_a_short_mcp_error_is_still_named():
    """A validator message the redactor leaves alone stays in the detail."""
    from gateway.session_local_mcp import validate_servers

    spec = {'type': 'stdio', 'name': 'local', 'command': 'echo', 'args': [], 'env': []}
    with pytest.raises(RuntimeStoreError) as caught:
        validate_servers([spec, dict(spec)])
    assert caught.value.field == 'editor.mcp_servers'
    assert 'duplicate server' in caught.value.detail
    assert str(caught.value) == 'invalid_params'


def test_a_safe_editor_key_is_still_named():
    from gateway.session_local_editor import validate_editor

    with pytest.raises(RuntimeStoreError) as caught:
        validate_editor('acp', {'bogus': 1})
    assert caught.value.field == 'editor.bogus'


def test_an_mcp_spec_error_does_not_echo_a_token():
    """An unknown server type raises ``KeyError`` whose text is the type. That text is the spec."""
    from gateway.session_local_mcp import validate_servers

    secret = 'sk_live_' + 'Ab12Cd34' * 3
    with pytest.raises(RuntimeStoreError) as caught:
        validate_servers([{'type': secret}])
    blob = f'{caught.value.field} {caught.value.detail}'
    assert caught.value.field == 'editor.mcp_servers'
    assert secret not in blob and 'Ab12Cd34' not in blob


def test_missing_staging_file_names_the_staging_clause(tmp_path, monkeypatch):
    from gateway.platforms import base
    from gateway.session_ingress_media import admit_attachments

    staging = tmp_path / 'images'
    staging.mkdir()
    monkeypatch.setattr(base, 'get_image_cache_dir', lambda: staging)
    missing = staging / 'gone.png'
    with pytest.raises(RuntimeStoreError) as caught:
        admit_attachments([{'path': str(missing), 'mime': 'image/png'}])
    assert caught.value.reason == 'invalid_params'
    assert str(caught.value) == 'invalid_params'
    assert caught.value.field == 'attachments.staging'
    assert 'staging' in caught.value.detail
    assert str(missing) not in f'{caught.value.field} {caught.value.detail}'


def test_bmp_mime_names_the_mime_clause():
    from gateway.session_ingress_media import admit_attachments

    with pytest.raises(RuntimeStoreError) as caught:
        admit_attachments([{'path': '/tmp/scan.bmp', 'mime': 'image/bmp'}])
    assert caught.value.field == 'attachments.mime'
    assert 'webp' in caught.value.detail
    assert 'image/bmp' not in f'{caught.value.field} {caught.value.detail}'
    assert str(caught.value) == 'invalid_params'


def test_hardlink_staging_is_not_wrapped(tmp_path, monkeypatch, caplog):
    import os

    from gateway.platforms import base
    from gateway.session_ingress_media import admit_attachments

    staging = tmp_path / 'images'
    staging.mkdir()
    monkeypatch.setattr(base, 'get_image_cache_dir', lambda: staging)
    original = staging / 'one.png'
    alias = staging / 'two.png'
    original.write_bytes(b'\x89PNG\r\n\x1a\n' + b'0' * 16)
    os.link(original, alias)
    with caplog.at_level(logging.WARNING, logger='gateway.session_controls'):
        with pytest.raises(RuntimeStoreError) as caught:
            admit_attachments([{'path': str(alias), 'mime': 'image/png'}])
    assert caught.value.field == 'attachments.staging'
    assert not isinstance(caught.value.__cause__, RuntimeStoreError)
    assert caplog.records == []
    from gateway.session_controls import named_turn_error
    with caplog.at_level(logging.WARNING, logger='gateway.session_controls'):
        named_turn_error(caught.value, 'no details')
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert 'attachments.staging' in warnings[0].getMessage()


def test_chat_upload_rejects_bmp_outside_the_admission_set():
    from fastapi import HTTPException

    from hermes_cli.web_models import ChatImageUpload
    from hermes_cli.web_routers.files import _decode_chat_image_upload

    payload = ChatImageUpload(data_url='data:image/bmp;base64,' + base64.b64encode(b'BM' + b'\x00' * 16).decode())
    with pytest.raises(HTTPException) as caught:
        _decode_chat_image_upload(payload)
    assert caught.value.status_code == 400


def test_chat_upload_still_accepts_png():
    """Control: a mime admission accepts is still a valid clipboard upload."""
    from hermes_cli.web_models import ChatImageUpload
    from hermes_cli.web_routers.files import _decode_chat_image_upload

    png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16
    payload = ChatImageUpload(data_url='data:image/png;base64,' + base64.b64encode(png).decode())
    data, mime, ext = _decode_chat_image_upload(payload)
    assert data.startswith(b'\x89PNG') and mime == 'image/png' and ext == '.png'


def _native_event():
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', user_id='user')
    return MessageEvent(text='hello', source=source)


def _runner(*, multiplex):
    from types import SimpleNamespace

    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=multiplex))
    runner._is_user_authorized_for_source = lambda source, allow_adapter_delegation=False: True
    runner._adapter_for_source = lambda source: object()
    return runner


def test_messaging_refusal_names_the_clause_not_the_class(caplog):
    from agent.i18n import t
    from gateway.session_controls import named_turn_error

    exc = RuntimeStoreError('invalid_params', field='native.provenance_missing(multiplex)',
                            detail='a multiplex gateway message needs provenance')
    with caplog.at_level(logging.DEBUG, logger='gateway.session_controls'):
        error_type, error_detail = named_turn_error(exc, 'no details available')
    text = t('gateway.notify.turn_error_detailed', error_type=error_type, error_detail=error_detail)
    assert 'RuntimeStoreError' not in text
    assert 'native.provenance_missing(multiplex)' in text
    assert 'invalid_params' in text
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert 'field=native.provenance_missing(multiplex)' in warnings[0].getMessage()


def test_a_bare_turn_error_stays_at_debug(caplog):
    from gateway.session_controls import named_turn_error

    with caplog.at_level(logging.DEBUG, logger='gateway.session_controls'):
        error_type, _detail = named_turn_error(RuntimeStoreError('permission_denied'), 'no details')
    assert error_type == 'RuntimeStoreError'
    assert 'WARNING' not in {rec.levelname for rec in caplog.records}
    assert 'DEBUG' in {rec.levelname for rec in caplog.records}


@pytest.mark.asyncio
async def test_notify_turn_error_sends_the_named_refusal(caplog):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    class _Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
            self.sent = []

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append(content)
            from gateway.platforms.base import SendResult
            return SendResult(success=True, message_id='1')

    adapter = _Adapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='42', user_id='42')
    event = MessageEvent(text='hi', source=source)
    exc = RuntimeStoreError('invalid_params', field='native.media_urls', detail='must be a list of strings')
    with caplog.at_level(logging.WARNING, logger='gateway.session_controls'):
        await adapter._notify_turn_error(event, exc)
    assert adapter.sent
    body = adapter.sent[0]
    assert 'RuntimeStoreError' not in body
    assert 'native.media_urls' in body
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING
                and rec.name == 'gateway.session_controls']
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_role_gated_missing_provenance_names_the_clause(caplog):
    from gateway.session_envelope import prepare_native

    event = _native_event()
    event.source.role_authorized = True
    with caplog.at_level(logging.WARNING, logger='gateway.session_controls'):
        with pytest.raises(RuntimeStoreError) as caught:
            await prepare_native(_runner(multiplex=True), event)
    assert caught.value.field == 'native.provenance_missing(roles)'
    assert str(caught.value) == 'invalid_params'
    assert caplog.records == []


@pytest.mark.asyncio
async def test_role_gated_relay_without_relay_provenance_names_the_clause():
    from gateway.session_ingress_context import reauthorize_roles

    source = _native_event().source
    source.role_authorized = True
    source.delivered_via_upstream_relay = True
    with pytest.raises(RuntimeStoreError) as caught:
        await reauthorize_roles(_runner(multiplex=False), source, {'transport_home': '/tmp'})
    assert caught.value.field == 'native.relay_provenance'
    assert str(caught.value) == 'invalid_params'


@pytest.mark.asyncio
async def test_session_create_rpc_keeps_the_code_and_names_toolsets(tmp_path, monkeypatch, caplog):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {'toolsets': ['not-a-toolset']},
        params={'request_id': 'bad-toolset', 'source': 'cli', 'cwd': str(tmp_path),
                'toolsets': ['not-a-toolset'], 'model': 'm'})
    error = response['error']
    assert error['message'] == 'invalid_params'
    assert error['data']['reason'] == 'invalid_params'
    assert error['data']['field'] == 'toolsets'
    assert 'not-a-toolset' in error['data']['detail']
    assert len(logs) == 1 and 'field=toolsets' in logs[0]


@pytest.mark.asyncio
async def test_null_toolset_on_the_wire_is_a_refusal(tmp_path, monkeypatch, caplog):
    response, _logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        params={'request_id': 'null-toolset', 'source': 'cli', 'cwd': str(tmp_path),
                'toolsets': [None], 'model': 'm'})
    error = response['error']
    assert error['message'] == 'invalid_params'
    assert 'toolsets' in error['data']['field']
    assert 'AttributeError' not in error['message']


def test_cli_error_line_names_the_field(monkeypatch, capsys):
    import argparse

    from hermes_cli.cli_chat_error_copy import refusal_text
    from hermes_cli.gateway_client import GatewayRPCError
    import hermes_cli.gateway_chat as chat
    import hermes_cli.gateway_chat_startup as startup

    exc = GatewayRPCError('invalid_params', 4001, {
        'reason': 'invalid_params', 'field': 'toolsets',
        'detail': "unknown or invalid toolset name: 'gone_plugin'"})
    assert str(exc) == 'invalid_params'
    assert refusal_text(exc) == "invalid_params (toolsets: unknown or invalid toolset name: 'gone_plugin')"

    async def refused(args, emitter=None):
        raise exc

    monkeypatch.setattr(chat, 'validate_options', lambda args: None)
    monkeypatch.setattr(chat, '_register_terminal_process', lambda: None)
    monkeypatch.setattr(startup, 'ensure_launch_provider', lambda args: True)
    monkeypatch.setattr(chat, 'run_gateway_chat', refused)
    args = argparse.Namespace(
        output_format='text', list_tools=False, list_toolsets=False,
        query='hi', q=None, oneshot=None, query_file=None,
    )
    assert chat.launch_from_args(args) == 1
    assert capsys.readouterr().err.strip() == (
        "Error: invalid_params (toolsets: unknown or invalid toolset name: 'gone_plugin')")


def test_exit_code_uses_the_reason_not_the_rendered_clause(monkeypatch, capsys):
    import argparse

    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
    import hermes_cli.gateway_chat as chat
    import hermes_cli.gateway_chat_startup as startup

    exc = GatewayRPCError('invalid_params', 4001, {
        'reason': 'invalid_params', 'field': 'flags', 'detail': 'unsupported_flag'})

    async def refused(args, emitter=None):
        raise exc

    monkeypatch.setattr(chat, 'validate_options', lambda args: None)
    monkeypatch.setattr(chat, '_register_terminal_process', lambda: None)
    monkeypatch.setattr(startup, 'ensure_launch_provider', lambda args: True)
    monkeypatch.setattr(chat, 'run_gateway_chat', refused)
    args = argparse.Namespace(
        output_format='text', list_tools=False, list_toolsets=False,
        query='hi', q=None, oneshot=None, query_file=None,
    )
    assert chat.launch_from_args(args) == 1
    err = capsys.readouterr().err
    assert 'unsupported_flag' in err
    usage = GatewayClientError('Unsupported gateway CLI options: --image')
    assert chat.client_error_exit_code(usage) == 2
    assert chat.client_error_exit_code(exc) == 1


def test_the_panel_seals_a_value_the_frame_already_carried():
    from hermes_cli.cli_chat_error_copy import refusal_text
    from hermes_cli.gateway_client import GatewayRPCError

    secret = 'sk_live_' + 'Ab12Cd34' * 3
    exc = GatewayRPCError('invalid_params', 4001, {
        'reason': 'invalid_params', 'field': f'params.{secret}',
        'detail': f"not a known effort level: '{secret}'"})
    shown = refusal_text(exc)
    assert secret not in shown and 'Ab12Cd34' not in shown
    assert REDACTED_KEY in shown


def _blob(error, logs, shown):
    return ' '.join([
        str(error['data'].get('field')), str(error['data'].get('detail')),
        shown, *logs, *map(str, error['data'].get('fields') or ()),
    ])


def _shown(error):
    from hermes_cli.cli_chat_error_copy import refusal_text
    from hermes_cli.gateway_client import GatewayRPCError
    return refusal_text(GatewayRPCError(error['message'], error['code'], error['data']))


@pytest.mark.asyncio
async def test_schema_extra_key_is_named_logged_and_shown(tmp_path, monkeypatch, caplog):
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {'nope': 1})
    error = response['error']
    shown = _shown(error)
    assert error['message'] == 'invalid_params'
    assert error['data']['reason'] == 'invalid_params'
    assert error['data']['field'] == 'nope'
    assert error['data']['detail'] == _CONTRACT
    assert error['data']['fields'] == ['nope']
    assert logs == [f'session.create refused for profile=owned: invalid_params field=nope detail={_CONTRACT}']
    assert shown == f'invalid_params (nope: {_CONTRACT})'


@pytest.mark.asyncio
async def test_schema_bad_type_names_the_field_not_the_value(tmp_path, monkeypatch, caplog):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        params={'request_id': 'schema', 'cwd': str(tmp_path), 'model': 'm',
                'toolsets': 'terminal', 'source': 'cron'})
    error = response['error']
    shown = _shown(error)
    blob = _blob(error, logs, shown)
    assert error['message'] == 'invalid_params'
    assert 'toolsets' in error['data']['fields'] and 'source' in error['data']['fields']
    assert 'toolsets' in shown and 'source' in shown
    assert 'terminal' not in blob and 'cron' not in blob
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_token_shaped_key_is_not_echoed(tmp_path, monkeypatch, caplog):
    secret = 'sk_live_' + 'Ab12Cd34' * 3
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {secret: 1})
    error = response['error']
    shown = _shown(error)
    blob = _blob(error, logs, shown)
    assert error['message'] == 'invalid_params'
    assert secret not in blob and 'Ab12Cd34' not in blob
    assert error['data']['field'] == REDACTED_KEY
    assert error['data']['detail'] == _CONTRACT
    assert logs and REDACTED_KEY in logs[0] and secret not in logs[0]
    assert shown == f'invalid_params ({REDACTED_KEY}: {_CONTRACT})'


@pytest.mark.parametrize('secret, field', [
    ('ghp_AAAAAAAA.BBBBBBBB', REDACTED_KEY),
    ('sk-sp-' + 'AbCdEf12.' + 'GhIjKl34.' + 'MnOpQr56', REDACTED_KEY),
    ('ghs_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6', REDACTED_KEY),
    # xoxc- is not the Slack pattern, and it is not an identifier either.
    ('xoxc-' + '123456789012-1234567890123-' + 'AbCdEfGhIjKlMnOpQrStUv', 'create.fields'),
])
@pytest.mark.asyncio
async def test_credential_shaped_key_is_not_copied(tmp_path, monkeypatch, caplog, secret, field):
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {secret: 1})
    error = response['error']
    shown = _shown(error)
    blob = _blob(error, logs, shown)
    assert error['message'] == 'invalid_params'
    assert secret not in blob and 'AbCdEf' not in blob
    assert error['data']['field'] == field
    assert field in shown


_LEADING = [
    '__hf_' + ('Ab12Cd34' * 4) + 'Ab',
    '___hf_' + ('Ab12Cd34' * 4),
    '__sk_live_' + 'AbCdEf1234' + 'AbCdEf1234' + 'AbCd',
    'foo__hf_' + ('Ab12Cd34' * 3),
    'x__SG.' + 'AbCdEfGhIjKlMnOpQrStUv',
]


@pytest.mark.parametrize('secret', _LEADING)
@pytest.mark.asyncio
async def test_extra_leading_underscores_are_not_echoed(tmp_path, monkeypatch, caplog, secret):
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {secret: 1})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert secret not in blob and 'Ab12Cd34' not in blob and 'AbCdEf' not in blob
    assert error['data']['field'] == REDACTED_KEY


@pytest.mark.parametrize('secret', _LEADING[:3])
@pytest.mark.asyncio
async def test_nested_leading_underscores_are_not_echoed(tmp_path, monkeypatch, caplog, secret):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        method='prompt.submit',
        params={'session_id': 's', 'text': 'hi', 'turn_author': {secret: 'x'}})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert secret not in blob
    assert error['data']['field'] == REDACTED_KEY


_TRAILING = [
    ('hf_' + 'Ab12Cd34' * 4, '_x'),
    ('ghp_' + 'AbCdEf1234' * 3 + 'Ab', '_'),
    ('npm_' + 'AbCdEf1234' * 2, '_x'),
    ('am_' + 'AbCdEfGhIjKlMnOpQrSt', '_'),
    ('AKIA' + 'IOSFODNN7EXAMPLE', '_x'),
    ('npm_' + 'AbCdEf12.' + 'GhIjKl34', '_x'),
]


@pytest.mark.parametrize('token, tail', _TRAILING)
@pytest.mark.asyncio
async def test_trailing_underscore_does_not_hide_a_token(tmp_path, monkeypatch, caplog, token, tail):
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {token + tail: 1})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert token not in blob and token.replace('.', '') not in blob
    assert error['data']['field'] == REDACTED_KEY


@pytest.mark.parametrize('token, tail', _TRAILING)
@pytest.mark.asyncio
async def test_nested_trailing_underscore_does_not_hide_a_token(tmp_path, monkeypatch, caplog, token, tail):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        method='prompt.submit',
        params={'session_id': 's', 'text': 'hi', 'turn_author': {'x_' + token + tail: 'x'}})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert token not in blob
    assert error['data']['field'] == REDACTED_KEY


@pytest.mark.asyncio
async def test_overlong_key_is_not_probed_run_by_run(tmp_path, monkeypatch, caplog):
    import asyncio

    from agent import redact

    key = 'a_' * 2000
    real = redact.redact_sensitive_text
    probes = []

    def counting(text, **kwargs):
        if 'a_a_' in text:
            probes.append(len(text))
        return real(text, **kwargs)

    monkeypatch.setattr(redact, 'redact_sensitive_text', counting)
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {key: 1})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert error['message'] == 'invalid_params'
    assert error['data']['field'] == 'create.fields'
    assert key not in blob
    assert len(probes) < 20
    assert not asyncio.iscoroutine(response)


@pytest.mark.asyncio
async def test_many_extra_keys_stay_within_the_client_bound(tmp_path, monkeypatch, caplog):
    from agent import redact
    from hermes_cli.cli_chat_error_copy import refusal_text
    from hermes_cli.gateway_client import rpc_error

    keys = [f'extra{index:03d}' for index in range(200)]
    real = redact.redact_sensitive_text
    probed = set()

    def counting(text, **kwargs):
        probed.update(key for key in keys if key in text)
        return real(text, **kwargs)

    monkeypatch.setattr(redact, 'redact_sensitive_text', counting)
    response, _logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, dict.fromkeys(keys, 1))
    error = response['error']
    fields = error['data']['fields']
    assert error['message'] == 'invalid_params'
    assert fields[0] == keys[0] and fields[-1] == 'create.fields'
    received = rpc_error(error)
    assert received.data['fields'] == fields
    assert refusal_text(received).startswith(f'invalid_params ({keys[0]}')
    assert keys[-1] not in probed
    assert len(probed) < len(keys) // 4


@pytest.mark.parametrize('name', ['__private', '___private', '__am_example_identifier_123'])
@pytest.mark.asyncio
async def test_extra_underscores_on_a_plain_name_stay_echoed(tmp_path, monkeypatch, caplog, name):
    response, logs = await _dispatch_create(tmp_path, monkeypatch, caplog, {name: 1})
    error = response['error']
    shown = _shown(error)
    assert error['data']['field'] == name
    assert name in shown
    assert logs and name in logs[0]


@pytest.mark.asyncio
async def test_list_index_locs_stay_named_beside_safe_fields(tmp_path, monkeypatch, caplog):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        params={'request_id': 'schema', 'cwd': str(tmp_path), 'model': 'm',
                'toolsets': [1], 'source': 'cron'})
    error = response['error']
    shown = _shown(error)
    blob = _blob(error, logs, shown)
    assert 'toolsets.0' in error['data']['field'] and 'source' in error['data']['field']
    assert 'toolsets.0' in shown and 'source' in shown
    assert logs and 'toolsets.0' in logs[0] and 'source' in logs[0]
    assert 'create.fields' not in blob
    assert 'cron' not in blob


@pytest.mark.asyncio
async def test_secret_loc_does_not_hide_a_safe_sibling(tmp_path, monkeypatch, caplog):
    secret = 'rk_live_' + 'AbCdEf1234'
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {secret: 1},
        params={'request_id': 'schema', 'cwd': str(tmp_path), 'model': 'm',
                'source': 'cron', secret: 1})
    error = response['error']
    shown = _shown(error)
    blob = _blob(error, logs, shown)
    assert 'source' in error['data']['field'] and REDACTED_KEY in error['data']['field']
    assert 'source' in shown and REDACTED_KEY in shown
    assert secret not in blob and 'AbCdEf' not in blob
    assert 'cron' not in blob


@pytest.mark.parametrize('attachments, needle', [
    ([{'path': 1, 'mime': 'image/png'}], 'attachments.0.path'),
    ([{'path': '/tmp/a.png', 'mime': 'image/png', 'bogus': True}], 'attachments.0.bogus'),
])
@pytest.mark.asyncio
async def test_attachment_index_locs_name_the_field(tmp_path, monkeypatch, caplog, attachments, needle):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        method='prompt.submit',
        params={'session_id': 's', 'text': 'hi', 'attachments': attachments})
    error = response['error']
    shown = _shown(error)
    blob = _blob(error, logs, shown)
    assert needle in error['data']['field']
    assert needle in shown
    assert logs and needle in logs[0]
    assert 'submit.fields' not in blob


@pytest.mark.asyncio
async def test_nested_token_path_is_not_echoed(tmp_path, monkeypatch, caplog):
    secret = 'sk_live_' + ('Ab12Cd34' * 3)
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        method='prompt.submit',
        params={'session_id': 's', 'text': 'hi', 'turn_author': {secret: 'x'}})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert secret not in blob and 'Ab12Cd34' not in blob
    assert error['data']['field'] == REDACTED_KEY


@pytest.mark.asyncio
async def test_resume_fallback_is_not_create_fields(tmp_path, monkeypatch, caplog):
    response, logs = await _dispatch_create(
        tmp_path, monkeypatch, caplog, {},
        method='session.resume',
        params={'session_id': 's', '_private': 1, 'user:s3cret-pass': 1})
    error = response['error']
    blob = _blob(error, logs, _shown(error))
    assert 'create.fields' not in blob
    assert 's3cret' not in blob
    assert '_private' in error['data']['field'] or 'resume.fields' in error['data']['field']


async def _dispatch_create(tmp_path, monkeypatch, caplog, extra, *, method='session.create', params=None):
    import json
    from types import SimpleNamespace

    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'model': {'default': 'm'}}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(
        session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
        _handle_message=None, _peek_session_state=lambda key: None,
    )
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(
        runner, profile_id='owned', instance_id='owner', db=db,
        epoch=rt.begin_runtime_epoch(db, instance_id='owner'),
    )
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    try:
        body = params if params is not None else {
            'request_id': 'schema', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'm', **extra,
        }
        with caplog.at_level(logging.WARNING, logger='gateway.session_controls'):
            response = await owner.dispatch({'id': 'schema', 'method': method, 'params': body})
        logs = [rec.getMessage() for rec in caplog.records if rec.name == 'gateway.session_controls']
        return response, logs
    finally:
        await owner.close()
        store.close_all_db_handles()
