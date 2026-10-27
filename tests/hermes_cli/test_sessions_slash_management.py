"""Reachable classic /sessions routing, with isolated scripted owner transport."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_session_store import GatewaySessionStore


@pytest.fixture
def cli(monkeypatch):
    from cli import HermesCLI
    c = HermesCLI.__new__(HermesCLI)
    c.session_id = "current"
    c._session_db = GatewaySessionStore(SimpleNamespace(final=False, states={}))
    c._session_db.connection.call = Mock(return_value={})
    c._handle_resume_command = Mock()
    c._show_recent_sessions = Mock(return_value=True)
    c._app = None
    c._agent_running = False
    c.conversation_history = [{"role": "user", "content": "unchanged"}]
    c._pending_title = "unchanged"
    c._prompt_text_input_modal = Mock(return_value="cancel")
    monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("no stdin")))
    return c


@pytest.mark.parametrize("args", ["delete", "delete id extra", "delete id --yes --yes", "rename id", 'rename id "bad', 'rename id ""', "rename id x\ny", "list extra", "browse extra", "prune --days 2"])
def test_reserved_malformed_never_resumes(cli, args, capsys):
    cli._handle_sessions_command("/sessions " + args)
    cli._handle_resume_command.assert_not_called()
    cli._session_db.connection.call.assert_not_called()
    assert "Usage:" in capsys.readouterr().out


@pytest.mark.parametrize("arg", ["", "list", "ls", "browse"])
def test_existing_listing(cli, arg):
    cli._handle_sessions_command("/sessions " + arg)
    cli._show_recent_sessions.assert_called_once_with(reason="sessions")
    cli._handle_resume_command.assert_not_called()


def test_reachable_slash_dispatch_and_resume_escape_hatch(cli, monkeypatch):
    monkeypatch.setattr('hermes_cli.plugins.fire_pre_command_hook', Mock())
    cli.config, cli._pending_resume_sessions = {}, None
    calls = owner(cli)
    assert cli.process_command('/sessions delete opaque --yes') is True
    assert [m for m, _ in calls] == ['session.resume', 'session.mutate']
    calls.clear()
    for target in ['list', 'delete', 'rename']:
        assert cli.process_command('/resume ' + target) is True
        cli._handle_resume_command.assert_called_with('/resume ' + target)
    cli.process_command('/sessions new')
    cli._handle_resume_command.assert_called_with('/resume new')
    assert not calls


def test_unknown_preserves_resume(cli):
    cli._handle_sessions_command("/sessions release notes")
    cli._handle_resume_command.assert_called_once_with("/resume release notes")


def owner(cli, *, resolved="other", error=None):
    calls = []
    def call(method, **params):
        calls.append((method, params))
        if method == "session.resume":
            return {"session_id": resolved, "stored_session_id": resolved, "subscription_id": "lease", "revision": 7, "execution_generation": 3}
        if method == "session.mutate":
            if error:
                raise GatewayClientError(error)
            return {"title": "Owner title"} if params["operation"] == "rename" else {"deleted": True}
        return {}
    cli._session_db.connection.call = call
    return calls


@pytest.mark.parametrize(("text", "title"), [
    ("Sam's notes now --yes", "Sam's notes now --yes"),
    ('"Release \\"Beta\\" 雪"', 'Release "Beta" 雪'),
])
def test_rename_exact_target_and_literal_title(cli, text, title, capsys):
    calls = owner(cli)
    cli._agent_running = True
    cli._handle_sessions_command("/sessions rename 2 " + text)
    assert [m for m, _ in calls] == ["session.resume", "session.mutate", "session.detach"]
    assert calls[0][1] == {"session_id": "2"}
    mutation = calls[1][1]
    assert mutation["session_id"] == "2"
    assert mutation["payload"] == {"title": title}
    assert mutation["expected_revision"] == 7
    assert "expected_generation" not in mutation
    assert "Owner title" in capsys.readouterr().out
    assert cli.session_id == "current" and cli._pending_title == "unchanged"


def test_delete_yes_uses_owner_fences(cli):
    calls = owner(cli)
    cli._handle_sessions_command("/sessions delete opaque-ID --yes")
    assert [m for m, _ in calls] == ["session.resume", "session.mutate"]
    assert calls[1][1]["expected_generation"] == 3
    assert calls[1][1]["expected_revision"] == 7
    cli._prompt_text_input_modal.assert_not_called()


@pytest.mark.parametrize("app,answer", [(None, "once"), (SimpleNamespace(is_running=True), "cancel"), (SimpleNamespace(is_running=True), None)])
def test_no_consent_no_owner_call(cli, app, answer):
    calls = owner(cli)
    cli._app = app
    cli._prompt_text_input_modal.return_value = answer
    cli._handle_sessions_command("/sessions delete other")
    assert calls == []


def test_explicit_one_shot_modal(cli):
    calls = owner(cli)
    cli._app = SimpleNamespace(is_running=True)
    cli._prompt_text_input_modal.return_value = "once"
    cli._handle_sessions_command("/sessions delete other")
    assert [m for m, _ in calls] == ["session.resume", "session.mutate"]
    kwargs = cli._prompt_text_input_modal.call_args.kwargs
    assert {row[0] for row in kwargs["choices"]} == {"once", "cancel"}
    assert "pinned" in kwargs["detail"] and "lineage" in kwargs["detail"]


@pytest.mark.parametrize("target,resolved,held", [("current", "current", False), ("segment", "current", False), ("segment", "live", True)])
def test_current_and_live_aliases_never_delete(cli, target, resolved, held, capsys):
    calls = owner(cli, resolved=resolved)
    if held:
        cli._session_db.connection.states[resolved] = SimpleNamespace(subscription_id="held")
    cli._handle_sessions_command(f"/sessions delete {target} --yes")
    assert not any(m in {"session.mutate", "session.detach"} for m, _ in calls)
    assert "close/switch" in capsys.readouterr().out


@pytest.mark.parametrize("error", ["permission_denied", "revision_conflict", "unknown_execution", "disconnected"])
def test_owner_refusal_no_retry_and_resolved_cleanup(cli, error, capsys):
    calls = owner(cli, resolved="resolved-owner", error=error)
    cli._handle_sessions_command("/sessions delete physical --yes")
    assert [m for m, _ in calls] == ["session.resume", "session.mutate", "session.detach"]
    assert calls[-1][1] == {"session_id": "resolved-owner", "subscription_id": "lease"}
    assert error in capsys.readouterr().out


def test_prune_refuses_without_rpc(cli, capsys):
    calls = owner(cli)
    cli._handle_sessions_command("/sessions prune --older-than 2d --yes")
    assert not calls
    assert "hermes sessions prune --older-than 2d --dry-run" in capsys.readouterr().out


@pytest.mark.parametrize("title", ['"x\\n"', '"x\\r"', '"x\\u0000"', '"   "', '"ok" junk', 'x\r', 'x\n', 'x\0'])
def test_invalid_titles_before_owner(cli, title, capsys):
    calls = owner(cli)
    cli._handle_sessions_command('/sessions rename other ' + title)
    assert not calls
    assert 'Usage:' in capsys.readouterr().out


def test_stale_confirmation_and_reentrant_delete(cli):
    calls = owner(cli)
    cli._app = SimpleNamespace(is_running=True)
    def confirm(**kwargs):
        cli._handle_sessions_command('/sessions delete other --yes')
        cli.session_id = 'switched'
        return 'once'
    cli._prompt_text_input_modal.side_effect = confirm
    cli._handle_sessions_command('/sessions delete other')
    assert not calls


def test_view_switch_during_cas_read_cancels_delete(cli):
    calls = owner(cli)
    original = cli._session_db.connection.call
    def call(method, **params):
        result = original(method, **params)
        if method == 'session.resume':
            cli.session_id = 'unrelated-switch'
        return result
    cli._session_db.connection.call = call
    cli._handle_sessions_command('/sessions delete other --yes')
    assert [m for m, _ in calls] == ['session.resume', 'session.detach']


def test_non_gateway_store_never_direct_writes(cli):
    cli._session_db = Mock()
    cli._handle_sessions_command('/sessions delete other --yes')
    cli._session_db.delete_session.assert_not_called()


@pytest.mark.parametrize('error', [None, 'permission_denied'])
def test_rename_preserves_subscription_acquired_during_mutation(cli, error):
    calls = owner(cli, error=error)
    original = cli._session_db.connection.call
    def call(method, **params):
        if method == 'session.mutate':
            cli._session_db.connection.states['other'] = SimpleNamespace(subscription_id='lease')
        return original(method, **params)
    cli._session_db.connection.call = call
    cli._handle_sessions_command('/sessions rename physical title')
    assert [m for m, _ in calls] == ['session.resume', 'session.mutate']


def test_missing_owner_receipt_is_not_success(cli, capsys):
    owner(cli)
    original = cli._session_db.connection.call
    cli._session_db.connection.call = lambda method, **params: None if method == 'session.mutate' else original(method, **params)
    cli._handle_sessions_command('/sessions delete other --yes')
    out = capsys.readouterr().out
    assert 'not confirmed' in out and 'Deleted conversation' not in out


def test_cleanup_failure_preserves_owner_error(cli, capsys):
    owner(cli, error='permission_denied')
    original = cli._session_db.connection.call
    def call(method, **params):
        if method == 'session.detach':
            raise GatewayClientError('disconnected')
        return original(method, **params)
    cli._session_db.connection.call = call
    cli._handle_sessions_command('/sessions rename physical title')
    assert 'permission_denied' in capsys.readouterr().out


def test_modal_scheduling_failure_never_reads_stdin(cli):
    import threading
    from cli import HermesCLI
    cli._prompt_text_input_modal = HermesCLI._prompt_text_input_modal.__get__(cli)
    cli._prompt_text_input = Mock(side_effect=AssertionError('no raw fallback'))
    cli._app = SimpleNamespace(is_running=True, loop=SimpleNamespace(call_soon_threadsafe=Mock(side_effect=RuntimeError('closed'))))
    calls, errors = owner(cli), []
    def run():
        try:
            cli._handle_sessions_command('/sessions delete other')
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert not calls


@pytest.mark.asyncio
async def test_gateway_chat_view_explicit_refusal(cli, capsys):
    from hermes_cli.gateway_chat_commands import run_command
    view = SimpleNamespace(session_id='current', client=SimpleNamespace(rpc=Mock(side_effect=AssertionError('no slash.exec'))))
    await run_command(view, '/sessions', 'delete other --yes')
    assert 'not available in GatewayChatView' in capsys.readouterr().out


@pytest.mark.parametrize('answer', ['once', None])
def test_real_composer_modal_response(cli, answer):
    import threading
    from cli import HermesCLI
    cli._prompt_text_input_modal = HermesCLI._prompt_text_input_modal.__get__(cli)
    cli._capture_modal_input_snapshot = Mock()
    cli._restore_modal_input_snapshot = Mock()
    ready = threading.Event()
    cli._invalidate = lambda: ready.set() if getattr(cli, '_slash_confirm_state', None) else None
    cli._app = SimpleNamespace(is_running=True, loop=SimpleNamespace(call_soon_threadsafe=lambda fn: fn()))
    calls, errors = owner(cli), []
    def run():
        try:
            cli._handle_sessions_command('/sessions delete other')
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(5)
    assert not calls
    cli._submit_slash_confirm_response(answer)
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert [m for m, _ in calls] == (['session.resume', 'session.mutate'] if answer == 'once' else [])
    cli._restore_modal_input_snapshot.assert_called_once()


@pytest.mark.parametrize('operation', ['rename', 'delete', 'foreign', 'alias'])
def test_real_authority_management(cli, tmp_path, monkeypatch, operation, capsys):
    import asyncio
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from tests.hermes_cli.test_resume_compressed_gateway_id import _compressed
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    loop = asyncio.new_event_loop()
    authority = None
    try:
        authority, current = loop.run_until_complete(_compressed(tmp_path))
        actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
        target = create_local_session(authority, actor, dict(request_id='other', source='cli', cwd=str(tmp_path), model='frozen', toolsets=[])).session_id
        gateway = AuthorityConnection(authority, object(), {
            'provider': 'local', 'user_id': 'intruder' if operation == 'foreign' else 'owner',
            'profile_id': 'fixture', 'instance_id': authority.instance_id,
            'capabilities': ['session:create', 'session:read', 'session:control']})
        calls = []
        def call(method, **params):
            calls.append((method, params))
            reply = loop.run_until_complete(gateway.dispatch({'id': len(calls), 'method': method, 'params': params}))
            if 'error' in reply:
                raise GatewayClientError(reply['error']['message'])
            return reply['result']
        cli.session_id = current
        cli._session_db.connection.call = call
        if operation == 'rename':
            cli._handle_sessions_command(f'/sessions rename {target} Release 雪')
            assert authority.db.get_session_title(target) == 'Release 雪'
            assert [m for m, _ in calls] == ['session.resume', 'session.mutate', 'session.detach']
        elif operation == 'delete':
            cli._handle_sessions_command(f'/sessions delete {target} --yes')
            assert authority.db.get_session(target) is None
        elif operation == 'foreign':
            cli._handle_sessions_command(f'/sessions delete {target} --yes')
            assert authority.db.get_session(target) is not None
            assert not any(m == 'session.mutate' for m, _ in calls)
        else:
            cli._handle_sessions_command('/sessions delete seg-3 --yes')
            assert 'close/switch' in capsys.readouterr().out
            assert authority.db.get_session(current) is not None
            assert not any(m == 'session.mutate' for m, _ in calls)
    finally:
        if authority is not None:
            authority.db.close()
        loop.close()
