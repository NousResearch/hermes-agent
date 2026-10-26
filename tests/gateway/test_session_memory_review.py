"""/memory and /skills review verbs on a LOCAL session match the classic CLI (#138901).

The classic CLI hands every /memory argument (and /skills review word) to
``hermes_cli.write_approval_commands.handle_pending_subcommand``; a local session must give each
subcommand that handler serves the same reach: its listings are reads, its writes are controls."""
import asyncio
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket

_WRITES = {verb: 'control' for verb in ('approve', 'apply', 'reject', 'deny', 'drop', 'approval', 'mode')}
# Every review subcommand the classic CLI serves, by the reach it needs on a local session: a
# listing is a read; an approve/reject or the approval switch is a control.
_REVIEW_VERBS = {'memory': {'pending': 'read', **_WRITES}, 'skills': {'pending': 'read', 'diff': 'read', **_WRITES}}


@pytest.mark.parametrize('subsystem', ['memory', 'skills'])
def test_each_review_subcommand_the_cli_serves_keeps_its_reach_on_a_local_session(tmp_path, monkeypatch, subsystem):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.session_commands import command_verdict
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    for verb, reach in _REVIEW_VERBS[subsystem].items():
        # The CLI answers it (an empty queue, an unknown id, an invalid switch value): it is a review verb.
        assert handle_pending_subcommand(subsystem, [verb, 'x']) is not None, verb
        for arg in (verb, f'{verb} x', f'{verb.upper()} x'):
            assert command_verdict(subsystem, arg) == reach, (subsystem, arg)
    assert command_verdict(subsystem, '') == 'read'


def test_memory_has_no_diff_like_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.session_commands import command_verdict
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    assert handle_pending_subcommand('memory', ['diff', 'x']) is None
    assert command_verdict('memory', 'diff x') == 'refused'


@pytest.mark.parametrize(('arg', 'reach'), [
    ("pending it's", 'read'), ('pending "unclosed', 'read'), ("approve it's", 'control'),
    ("drop 'all", 'control'), ("approval on'", 'control'), ("it's pending", 'refused')])
def test_memory_reads_only_its_first_word_like_the_cli(arg, reach):
    """Main's CLI and messaging handler split /memory on whitespace: an apostrophe or an unclosed
    quote later in the line is the argument's business, never a parse failure of the command."""
    from gateway.session_commands import command_verdict
    assert command_verdict('memory', arg) == reach


@pytest.mark.parametrize(('arg', 'reach'), [
    ("pending it's", 'read'), ('diff "unclosed', 'read'), ("approve it's", 'control'), ("it's pending", 'refused')])
def test_skills_reads_only_its_first_word_like_its_handler(arg, reach):
    """The messaging /skills handler splits on whitespace too (gateway/slash_commands.py
    ``_handle_skills_command``): a quote later in the line is the argument's, not a parse failure."""
    from gateway.session_commands import command_verdict
    try:
        verdict = command_verdict('skills', arg)
    except ValueError as exc:
        verdict = f'internal error: {exc}'
    assert verdict == reach


def test_kanban_with_an_unclosed_quote_is_refused_by_name():
    """/kanban parses with shlex (hermes_cli/kanban.py ``run_slash``), so an unclosed quote can't run
    there either: the client gets ``invalid_params`` naming the argument, not an internal error."""
    from gateway.session_commands import command_verdict
    from hermes_state_runtime import RuntimeStoreError
    try:
        verdict = command_verdict('kanban', "show it's")
    except RuntimeStoreError as exc:
        verdict = (exc.reason, exc.field)
    except ValueError as exc:
        verdict = f'internal error: {exc}'
    assert verdict == ('invalid_params', 'arg')
    assert command_verdict('kanban', 'show "two words"') == 'read'

def _stage(home, pending_id, content):
    folder = home / 'pending' / 'memory'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{pending_id}.json').write_text(json.dumps({
        'id': pending_id, 'subsystem': 'memory', 'action': 'add', 'summary': f'add {content}',
        'origin': 'foreground', 'created_at': time.time(),
        'payload': {'action': 'add', 'target': 'memory', 'content': content}}))


@pytest.mark.platforms("linux")
def test_local_session_reviews_staged_memory_writes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    _stage(home, 'keepme01', 'MEMORY_REVIEW_APPROVED')
    _stage(home, 'dropme02', 'MEMORY_REVIEW_REJECTED')
    _stage(home, 'dropme03', 'MEMORY_REVIEW_REJECTED_ALL')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'review-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out, config = {}, {}
    commands = ('memory approve keepme01', 'memory reject dropme02', 'memory approval on',
                'memory mode off', 'memory apply nope', 'memory deny nope', 'memory drop all')

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='review', source='cli', toolsets=['todo'], cwd=str(work))
            sid = created['result']['session_id']
            for command in commands:
                reply = await rpc(ws, 'slash.exec', session_id=sid, command=command)
                out[command] = reply.get('result', {}).get('output') or reply.get('error', {}).get('message')
                if command.startswith(('memory approval', 'memory mode')):
                    config[command] = (home / 'config.yaml').read_text()
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
    assert all(reply != 'unsupported_command' for reply in out.values()), out
    assert 'Approved 1 memory write' in out['memory approve keepme01'], out
    assert any('MEMORY_REVIEW_APPROVED' in path.read_text() for path in home.rglob('MEMORY.md')), out
    assert "Rejected pending memory write 'dropme02'" in out['memory reject dropme02'], out
    assert 'write_approval: true' in config['memory approval on'], config
    assert 'write_approval: false' in config['memory mode off'], config
    assert "No pending memory write with id 'nope'" in out['memory deny nope'], out
    assert 'Rejected 1 pending memory write' in out['memory drop all'], out
    assert not list((home / 'pending' / 'memory').glob('*.json'))
