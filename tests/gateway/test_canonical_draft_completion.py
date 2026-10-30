"""A new workspace chat inspects its original completion base before it has a session."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def drafts(tmp_path, monkeypatch):
    from tui_gateway import server

    root = tmp_path / '.hermes'
    coder = root / 'profiles' / 'coder'
    launch, configured, explicit = (tmp_path / name for name in ('launch', 'configured', 'explicit'))
    for folder in (root, coder, launch, configured, explicit):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'base-marker.md').write_text(folder.name, encoding='utf-8')
    (root / 'config.yaml').write_text(f'terminal:\n  backend: local\n  cwd: {launch}\n', encoding='utf-8')
    (coder / 'config.yaml').write_text(f'terminal:\n  backend: local\n  cwd: {configured}\n', encoding='utf-8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('TERMINAL_CWD', str(launch))
    monkeypatch.setattr(server, '_hermes_home', root)
    monkeypatch.setattr(server, '_served_profile_homes', set())
    monkeypatch.chdir(launch)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    runner.session_authorities = SessionAuthorities(root, multiplexed=True)
    with ExitStack() as stack:
        authorities = []
        for home in (root, coder):
            db = stack.enter_context(SessionDB(db_path=home / 'state.db'))
            epoch = begin_runtime_epoch(db, instance_id='draft-completion')
            authority = SessionAuthority(runner, profile_id=str(home), instance_id='draft-completion', db=db, epoch=epoch)
            runner.session_authorities.add(home, authority, name='coder' if home == coder else None)
            authorities.append(authority)
        identity = {'user_id': 'fixture-owner', 'profile_scope': 'host'}
        connection = AuthorityConnection(authorities[0], object(), identity)
        yield connection, coder, configured, explicit, authorities


@pytest.mark.asyncio
@pytest.mark.parametrize('configured_workspace', [True, False])
@pytest.mark.parametrize('explicit_pick', [True, False])
async def test_draft_inspection_and_listing_share_the_owning_profiles_base(drafts, configured_workspace, explicit_pick):
    connection, coder, configured, explicit, _ = drafts
    if not configured_workspace:
        (coder / 'config.yaml').write_text('terminal:\n  backend: local\n', encoding='utf-8')
    default = await connection.dispatch({'id': 0, 'method': 'complete.path', 'params': {'profile': 'coder', 'word': ''}})
    assert default.get('result') == {'items': [], 'sourceCwd': str(configured if configured_workspace else coder)}, default
    params = {'profile': 'coder', 'cwd': str(explicit), 'cwd_explicit': explicit_pick}
    inspected = await connection.dispatch({'id': 1, 'method': 'complete.path', 'params': {**params, 'word': ''}})
    expected = explicit if explicit_pick else configured if configured_workspace else explicit
    assert inspected.get('result') == {'items': [], 'sourceCwd': str(expected)}, inspected
    listing = await connection.dispatch({'id': 2, 'method': 'complete.path', 'params': {**params, 'word': '@file:'}})
    assert '@file:base-marker.md' in [item['text'] for item in listing['result']['items']], listing
    default = await connection.dispatch({'id': 3, 'method': 'complete.path', 'params': {'profile': 'coder', 'word': ''}})
    assert default['result']['sourceCwd'] == str(configured if configured_workspace else coder)


@pytest.mark.asyncio
async def test_draft_completion_refuses_foreign_or_unreadable_profiles_and_remote_trees(drafts):
    connection, coder, _, _, authorities = drafts
    bound = AuthorityConnection(authorities[0], object(), {'user_id': 'fixture-owner'})
    foreign = await bound.dispatch({'id': 1, 'method': 'complete.path', 'params': {'profile': 'coder', 'word': ''}})
    assert foreign['error']['data']['reason'] == 'profile_mismatch'
    denied = AuthorityConnection(authorities[1], object(), {'user_id': 'fixture-owner', 'capabilities': []})
    for word in ('', '@file:'):
        reply = await denied.dispatch({'id': 2, 'method': 'complete.path', 'params': {'word': word}})
        assert reply['error']['data']['reason'] == 'permission_denied'
    (coder / 'config.yaml').write_text('terminal:\n  backend: ssh\n  cwd: /remote-only\n', encoding='utf-8')
    for word in ('', '@file:'):
        remote = await connection.dispatch({'id': 3, 'method': 'complete.path', 'params': {'profile': 'coder', 'word': word}})
        assert remote.get('result') == {'items': []}, remote
