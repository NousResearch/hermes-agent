"""A held editor connection follows a committed workspace move (#134889 ACP-12).

Stdio editor servers run in the session's cwd. A viewed session keeps them connected between
turns, so the move must respawn them in the new folder rather than keep running them in the old one.
"""
from dataclasses import replace
from types import SimpleNamespace

from gateway.session_policy import policy_scope
from tests.acp_adapter.test_session_mcp_policy import owner, policy


def test_workspace_move_respawns_held_stdio_servers_in_the_new_folder(server_spec, tmp_path):
    from gateway.session_local_mcp import release_editor_mcp
    from tools.registry import registry
    script = tmp_path / 'owned_mcp.py'
    script.write_text(script.read_text().replace("os.environ['BORROWED_CREDENTIAL']", 'os.getcwd()'))
    old, new = tmp_path / 'project-a', tmp_path / 'project-b'
    old.mkdir()
    new.mkdir()
    authority = owner()
    live = SimpleNamespace(subscribers={'editor': object()})
    authority.sessions = {'moved': live}
    before = policy(authority, 'moved', server_spec, old)
    try:
        with policy_scope(before, authority=authority):
            name = next(n for n in registry.get_all_tool_names() if n.startswith('mcp__'))
            assert str(old.resolve()) in registry.dispatch(name, {})
        # What a committed ``workspace`` mutation freezes: the same editor selection, a new cwd.
        moved = replace(before, cwd=str(new))
        with policy_scope(moved, authority=authority):
            result = registry.dispatch(name, {})
        assert str(new.resolve()) in result and str(old.resolve()) not in result
    finally:
        live.subscribers.clear()
        release_editor_mcp(authority, ['moved'])
