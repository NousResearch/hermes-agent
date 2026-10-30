"""Slash commands and discovery on a local session resolve against THAT session: ``/diff session``
through ``slash.exec`` follows its frozen checkpoint policy (``--checkpoints`` on a profile that leaves
them off), and its repo's trusted project skills complete, list and dispatch (#134889 TUI-20, TUI-7)."""
import asyncio
import subprocess

import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import owner
from tests.gateway.test_session_tui_ops import _checkpoint, _git_rpc


def _repo_with_skill(path):
    skill = path / '.hermes' / 'skills' / 'repo-probe'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: repo-probe\ndescription: Project skill fixture\n---\n# Repo probe\n')
    (path / 'a.txt').write_text('original\n')
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    return path


@pytest.mark.platforms("linux")
def test_slash_diff_session_follows_the_sessions_checkpoint_policy(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'a.txt').write_text('original\n')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='cp', source='tui', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                plain = (await rpc(ws, 'session.create', request_id='plain', source='tui', toolsets=[],
                                   cwd=str(work)))['result']['session_id']
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                (work / 'a.txt').write_text('edited\n')
                diff = (await _git_rpc(ws, 'slash.exec', session_id=sid, command='diff session'))['result']
                assert 'a.txt' in diff['output'] and '+edited' in diff['output'], diff
                # The profile leaves checkpoints off: a session launched without the flag still says so.
                off = (await _git_rpc(ws, 'slash.exec', session_id=plain, command='diff session'))['result']
                assert 'Checkpoints are not enabled' in off['output'], off
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_project_skills_follow_the_sessions_repo_in_discovery_and_dispatch(tmp_path):
    repo = _repo_with_skill(tmp_path / 'repo')
    with owner(tmp_path, {'skills': {'trusted_project_dirs': [str(repo)]}}) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='repo', source='tui', toolsets=[],
                                 cwd=str(repo)))['result']['session_id']
                other = (await rpc(ws, 'session.create', request_id='elsewhere', source='tui', toolsets=[],
                                   cwd=str(tmp_path)))['result']['session_id']
                catalog = await rpc(ws, 'commands.catalog', session_id=sid)
                assert '/repo-probe' in catalog.get('result', {}).get('skills', {}), catalog
                items = (await rpc(ws, 'complete.slash', session_id=sid, text='/repo-pr'))['result']['items']
                assert any(item['text'].lstrip('/').startswith('repo-probe') for item in items), items
                directive = (await rpc(ws, 'slash.exec', session_id=sid, command='repo-probe hello'))
                assert directive.get('result', {}).get('type') == 'skill', directive
                # A session in another directory never sees the repo's skill.
                foreign = (await rpc(ws, 'commands.catalog', session_id=other))['result']
                assert '/repo-probe' not in foreign['skills'], foreign['skills']
                elsewhere = (await rpc(ws, 'complete.slash', session_id=other, text='/repo-pr'))['result']['items']
                assert not any(item['text'].lstrip('/').startswith('repo-probe') for item in elsewhere), elsewhere
                refused = await rpc(ws, 'slash.exec', session_id=other, command='repo-probe hello')
                assert refused['error']['message'] == 'unsupported_command', refused
        asyncio.run(probe())
