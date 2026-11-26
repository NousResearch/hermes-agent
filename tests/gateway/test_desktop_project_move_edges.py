"""``desktop_project``'s move for a gateway chat lands when the turn ends: the turn keeps its folder, the
boundary commits the chat's ``workspace`` like the user's Move to project, and a refused commit leaves
nothing half-moved. Also ``list`` after a move, a user's own move in between, ``create`` (which never
moves), and a delegated child (which never moves its parent)."""
import json
import shutil
from types import SimpleNamespace

import pytest

from gateway import local_desktop_ui
from tests.gateway.test_session_resume_config_refresh import (
    _Backend, _call, _connect, _home, _owner_authority, _policy)


async def _chat(tmp_path, monkeypatch, source='gui'):
    from hermes_cli import projects_db as pdb
    from tools import project_tools, terminal_tool
    built = []
    monkeypatch.setattr(terminal_tool, '_active_environments', {})
    monkeypatch.setattr(terminal_tool, '_session_cwd', {})
    monkeypatch.setattr(terminal_tool, '_container_aliases', {})
    monkeypatch.setattr(terminal_tool, '_pre_exec_block', lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, '_create_configured_env',
                        lambda config, env_type, **k: built.append(_Backend(env_type, k['cwd'])) or built[-1])
    monkeypatch.setattr(local_desktop_ui, '_turns', {})
    monkeypatch.setattr(local_desktop_ui, '_workspace_routes', {})
    monkeypatch.setattr(local_desktop_ui, '_process_routes', {})
    monkeypatch.setattr(project_tools, '_workspace_callback', None)
    db_path, connect_closing = tmp_path / 'projects.db', pdb.connect_closing
    monkeypatch.setattr(pdb, 'connect_closing', lambda: connect_closing(db_path=db_path))
    home = _home(monkeypatch)
    start, worktree = tmp_path / 'start', tmp_path / 'worktree'
    start.mkdir(), worktree.mkdir()
    store, authority = await _owner_authority(home)
    owner = _connect(authority)
    sid = (await _call(owner, 'session.create', request_id='t', source=source, cwd=str(start)))['session_id']
    return SimpleNamespace(built=built, store=store, authority=authority, owner=owner, sid=sid, start=start,
                           worktree=worktree, db_path=db_path, connect_closing=connect_closing)


def _in_turn(chat, fn):
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import policy_scope
    authority, sid = chat.authority, chat.sid
    route = authority.sessions[sid].route
    turn = SimpleNamespace(_approval_owner=(authority, sid, 1), _publish_execution=lambda *a: True,
                           _ctx=SimpleNamespace(session_id=sid, session_key=route))
    token = local_desktop_ui.wire(turn, SimpleNamespace())
    tokens = set_session_vars(platform='local', session_key=route, session_id=sid)
    try:
        with policy_scope(_policy(authority, sid), authority=authority):
            return fn()
    finally:
        clear_session_vars(tokens)
        local_desktop_ui.release(token)


async def _boundary(chat):
    from gateway.session_contract import SessionRef
    from gateway.session_mutation_policy import apply_pending_workspace
    authority = chat.authority
    await apply_pending_workspace(authority, SessionRef(authority.profile_id, chat.sid), authority.sessions[chat.sid])


def _pwd(chat):
    from tools import terminal_tool
    _in_turn(chat, lambda: terminal_tool.terminal_tool('pwd', task_id=chat.sid))
    return chat.built[-1].ran[-1]


def _create(chat, path):
    from tools import project_tools
    return _in_turn(chat, lambda: json.loads(project_tools.project_create('Worktree', path=str(path), task_id=chat.sid)))


def _move(chat, path):
    """Create the project, then ``switch`` this chat into it: the move is always the explicit verb."""
    from tools import project_tools

    def turn():
        created = json.loads(project_tools.project_create(path.name, path=str(path), task_id=chat.sid))
        return json.loads(project_tools.project_switch(created['id'], task_id=chat.sid))
    return _in_turn(chat, turn)


async def _closed(chat):
    await chat.owner.close()
    chat.store.close_all_db_handles()


@pytest.mark.asyncio
async def test_the_move_lands_when_the_turn_ends(tmp_path, monkeypatch):
    from hermes_cli import projects_db as pdb
    from tools import project_tools, terminal_tool
    chat = await _chat(tmp_path, monkeypatch)
    try:
        def turn():
            made = json.loads(project_tools.project_create('Worktree', path=str(chat.worktree), task_id=chat.sid))
            created = json.loads(project_tools.project_switch(made['id'], task_id=chat.sid))
            terminal_tool.terminal_tool('pwd', task_id=chat.sid)
            return created
        created = _in_turn(chat, turn)
        assert created['moved'] == 'pending' and str(chat.worktree) in created['note'], created
        assert chat.built[-1].ran[-1] == str(chat.start)  # the turn that asked keeps its folder
        with chat.connect_closing(db_path=chat.db_path) as conn:
            assert pdb.get_active_id(conn) is None  # a chat that moves leaves the shared pointer alone

        events = []
        stream = chat.authority.sessions[chat.sid].event_stream
        publish = stream.publish
        monkeypatch.setattr(stream, 'publish', lambda s, p, **k: events.append((k.get('event_type'), p)) or publish(s, p, **k))
        await _boundary(chat)
        assert _policy(chat.authority, chat.sid).cwd == str(chat.worktree)
        assert [p['cwd'] for kind, p in events if kind == 'session.info'] == [str(chat.worktree)]
        assert _pwd(chat) == str(chat.worktree)
    finally:
        await _closed(chat)


@pytest.mark.asyncio
@pytest.mark.parametrize('queued', [False, True], ids=['next-message', 'queued-mid-turn'])
async def test_the_drain_runs_the_next_turn_in_the_project(tmp_path, monkeypatch, queued):
    """The drain's own boundary commits the move: the turn after the switch runs in the project folder,
    whether its message arrived after the chat went idle or was queued while the switch's turn ran."""
    from gateway.session_contract import SessionRef, Submission
    from gateway.session_policy import policy_for_source
    from tests.waiting import async_wait_until
    chat = await _chat(tmp_path, monkeypatch)
    authority, seen, turns = chat.authority, [], []

    async def submit(text):
        ref = SessionRef(authority.profile_id, chat.sid)
        turns.append(await authority.submit(chat.owner.actor, Submission(text, ref, {'text': text}, 'queue')))

    async def handle(event):
        seen.append(policy_for_source(authority.runner, event.source).cwd)
        if len(seen) == 1:
            _move(chat, chat.worktree)
            if queued:
                await submit('two')
        return 'ok'

    def ran(count):
        return len(turns) == count and all(
            authority.persistence.admissions.get(admission_id=t.admission_id).status == 'terminal' for t in turns)
    authority.runner._handle_message = handle
    try:
        await submit('one')
        await async_wait_until(lambda: ran(2 if queued else 1), what='the switch turn to run')
        if not queued:
            # Nothing queued: the drain lands the move as it goes idle, before any next prompt arrives.
            await async_wait_until(authority.sessions[chat.sid].task.done, what='the drain to go idle')
            assert _policy(authority, chat.sid).cwd == str(chat.worktree)
            await submit('two')
        await async_wait_until(lambda: ran(2), what='the next turn to run')
        assert seen == [str(chat.start), str(chat.worktree)]
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_refused_move_leaves_the_chat_where_it_was(tmp_path, monkeypatch):
    from tools import terminal_tool
    chat = await _chat(tmp_path, monkeypatch)
    try:
        sub = chat.start / 'sub'
        sub.mkdir()
        _in_turn(chat, lambda: terminal_tool.terminal_tool(f'cd {sub}', task_id=chat.sid))
        assert _move(chat, chat.worktree)['moved'] == 'pending'
        shutil.rmtree(chat.worktree)  # gone before the turn ends: the owner refuses the move
        await _boundary(chat)
        assert _policy(chat.authority, chat.sid).cwd == str(chat.start)
        assert chat.authority.sessions[chat.sid].pending_workspace is None
        assert _pwd(chat) == str(sub)  # the agent's earlier cd survives
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_the_users_own_move_replaces_a_pending_one(tmp_path, monkeypatch):
    from gateway.session_persistence import authority_persistence
    chat = await _chat(tmp_path, monkeypatch)
    try:
        chosen = tmp_path / 'chosen'
        chosen.mkdir()
        _move(chat, chat.worktree)
        row = authority_persistence(chat.authority).sessions.get(chat.sid)
        await _call(chat.owner, 'session.mutate', session_id=chat.sid, request_id='user-move', operation='workspace',
                    payload={'cwd': str(chosen)}, expected_revision=row.runtime_revision,
                    expected_generation=row.runtime_generation)
        await _boundary(chat)
        assert _policy(chat.authority, chat.sid).cwd == str(chosen)
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_list_after_a_move_names_the_chats_own_project(tmp_path, monkeypatch):
    from hermes_cli import projects_db as pdb
    from tools import project_tools
    chat = await _chat(tmp_path, monkeypatch)
    try:
        other = tmp_path / 'other'
        other.mkdir()
        with chat.connect_closing(db_path=chat.db_path) as conn:
            pdb.set_active(conn, pdb.create_project(conn, name='Other', folders=[str(other)], primary_path=str(other)))
        created = _move(chat, chat.worktree)
        await _boundary(chat)
        listed = _in_turn(chat, lambda: json.loads(project_tools.project_list(task_id=chat.sid)))
        assert listed['active_id'] == created['id']
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_list_in_a_chat_without_project_tools_keeps_the_shared_pointer(tmp_path, monkeypatch):
    """Preservation: only the surfaces the gateway offers the project tools (Desktop, TUI) scope ``list``."""
    from hermes_cli import projects_db as pdb
    from tools import project_tools
    chat = await _chat(tmp_path, monkeypatch, source='cli')
    try:
        with chat.connect_closing(db_path=chat.db_path) as conn:
            pointer = pdb.create_project(conn, name='Pointer', folders=[str(tmp_path / 'p')])
            pdb.create_project(conn, name='Start', folders=[str(chat.start)])
            pdb.set_active(conn, pointer)
        listed = _in_turn(chat, lambda: json.loads(project_tools.project_list(task_id=chat.sid)))
        assert listed['active_id'] == pointer
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_move_to_the_chats_own_folder_changes_nothing(tmp_path, monkeypatch):
    from tools import terminal_tool
    chat = await _chat(tmp_path, monkeypatch)
    try:
        sub = chat.start / 'sub'
        sub.mkdir()
        _move(chat, chat.worktree)  # a pending move the second call takes back
        _in_turn(chat, lambda: terminal_tool.terminal_tool(f'cd {sub}', task_id=chat.sid))
        built = len(chat.built)
        assert _move(chat, chat.start)['moved'] is True
        assert chat.authority.sessions[chat.sid].pending_workspace is None
        await _boundary(chat)
        assert _pwd(chat) == str(sub) and len(chat.built) == built  # same cd, same terminal
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_move_to_the_chats_own_folder_by_another_path_changes_nothing(tmp_path, monkeypatch):
    from tools import terminal_tool
    chat = await _chat(tmp_path, monkeypatch)
    try:
        link = tmp_path / 'link'
        link.symlink_to(chat.start, target_is_directory=True)
        sub = chat.start / 'sub'
        sub.mkdir()
        _in_turn(chat, lambda: terminal_tool.terminal_tool(f'cd {sub}', task_id=chat.sid))
        built = len(chat.built)
        assert _move(chat, link)['moved'] is True
        await _boundary(chat)
        assert _pwd(chat) == str(sub) and len(chat.built) == built
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_missing_folder_says_so(tmp_path, monkeypatch):
    chat = await _chat(tmp_path, monkeypatch)
    try:
        result = _move(chat, tmp_path / 'gone')
        assert result['moved'] is False and 'does not exist' in result['not_moved_reason'], result
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_cli_chat_on_the_gateway_keeps_its_folder(tmp_path, monkeypatch):
    from hermes_cli import projects_db as pdb
    chat = await _chat(tmp_path, monkeypatch, source='cli')
    try:
        result = _move(chat, chat.worktree)
        assert result['moved'] is False and result['not_moved_reason']
        assert chat.authority.sessions[chat.sid].pending_workspace is None
        await _boundary(chat)
        assert _policy(chat.authority, chat.sid).cwd == str(chat.start)
        with chat.connect_closing(db_path=chat.db_path) as conn:
            assert pdb.get_active_id(conn) == result['id']
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_create_from_a_desktop_chat_leaves_the_chat_and_the_active_project_alone(tmp_path, monkeypatch):
    from hermes_cli import projects_db as pdb
    chat = await _chat(tmp_path, monkeypatch)
    try:
        with chat.connect_closing(db_path=chat.db_path) as conn:
            pointer = pdb.create_project(conn, name='Pointer', folders=[str(tmp_path / 'p')])
            pdb.set_active(conn, pointer)
        result = _create(chat, chat.worktree)
        assert result['success'] is True and 'moved' not in result, result
        assert chat.authority.sessions[chat.sid].pending_workspace is None
        await _boundary(chat)
        assert _policy(chat.authority, chat.sid).cwd == str(chat.start)
        with chat.connect_closing(db_path=chat.db_path) as conn:
            assert pdb.get_active_id(conn) == pointer
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_delegated_childs_switch_never_moves_the_parent(tmp_path, monkeypatch):
    from hermes_cli import projects_db as pdb
    from tools import project_tools
    from tools.terminal_tool import register_container_alias
    chat = await _chat(tmp_path, monkeypatch)
    try:
        with chat.connect_closing(db_path=chat.db_path) as conn:
            pdb.create_project(conn, name='Worktree', folders=[str(chat.worktree)], primary_path=str(chat.worktree))

        def child_call():
            register_container_alias('child-task', chat.sid)  # what delegate_task does at spawn
            return json.loads(project_tools.project_switch('Worktree', task_id='child-task'))
        assert _in_turn(chat, child_call)['moved'] is False
        assert chat.authority.sessions[chat.sid].pending_workspace is None
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_messaging_chat_viewed_in_desktop_is_told_it_did_not_move(tmp_path, monkeypatch):
    """Only a local chat's drain commits a move: a messaging chat gets ``moved: false``, never a
    ``pending`` that cannot land, and switches the active project as on main."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from hermes_cli import projects_db as pdb
    from tools import project_tools
    chat = await _chat(tmp_path, monkeypatch)
    authority = chat.authority
    authority.sessions['telegram-row'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='42'), 'tg:42')
    try:
        with chat.connect_closing(db_path=chat.db_path) as conn:
            pid = pdb.create_project(conn, name='Worktree', folders=[str(chat.worktree)], primary_path=str(chat.worktree))
        turn = SimpleNamespace(_approval_owner=(authority, 'telegram-row', 1), _publish_execution=lambda *a: True,
                               _ctx=SimpleNamespace(session_id='telegram-row', session_key='tg:42'))
        token = local_desktop_ui.wire(turn, SimpleNamespace())
        try:
            result = json.loads(project_tools.project_switch(pid, task_id='telegram-row'))
        finally:
            local_desktop_ui.release(token)
        assert result['moved'] is False and result['not_moved_reason'], result
        assert authority.sessions['telegram-row'].pending_workspace is None
        with chat.connect_closing(db_path=chat.db_path) as conn:
            assert pdb.get_active_id(conn) == pid
    finally:
        await _closed(chat)


@pytest.mark.asyncio
async def test_a_busy_boundary_keeps_the_move_for_the_next_one(tmp_path, monkeypatch):
    from gateway import session_mutations
    from persistence.core.errors import AuthorityRefusal
    chat = await _chat(tmp_path, monkeypatch)
    try:
        assert _move(chat, chat.worktree)['moved'] == 'pending'
        write = session_mutations.write_mutation

        async def busy(*args, **kwargs):
            monkeypatch.setattr(session_mutations, 'write_mutation', write)
            raise AuthorityRefusal('session_busy')
        monkeypatch.setattr(session_mutations, 'write_mutation', busy)
        await _boundary(chat)  # another worker still holds the chat: the move waits
        assert chat.authority.sessions[chat.sid].pending_workspace == str(chat.worktree)
        assert _policy(chat.authority, chat.sid).cwd == str(chat.start)
        await _boundary(chat)
        assert _policy(chat.authority, chat.sid).cwd == str(chat.worktree)
        assert chat.authority.sessions[chat.sid].pending_workspace is None
    finally:
        await _closed(chat)
