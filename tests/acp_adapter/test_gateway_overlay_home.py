"""``hermes acp`` with a per-task overlay ``HERMES_HOME`` (#138949, #134889 ACP-18).

A Multica task home is a private real directory mirroring a served profile through symlinks, its
``gateway.lock`` among them. The symlinked lock names the owner that serves the mirrored profile:
the editor attaches to that owner (no second owner on the shared ``state.db``), over the real ACP
stdio wire and a real gateway. (A home others can write never selects an owner: covered in
tests/hermes_cli/test_gateway_runtime_discovery.py; ``hermes acp`` startup narrows its own home to
0700 first, so that case is not reachable through the entry.)
"""
import json

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer


def _overlay(daemon, tmp_path):
    home, descriptor, env, root = daemon
    overlay = tmp_path / 'task-home'
    overlay.mkdir(mode=0o700)
    for name in ('gateway.lock', 'state.db', 'config.yaml'):
        (overlay / name).symlink_to(home / name)
    return overlay, descriptor, {**env, 'HERMES_HOME': str(overlay)}, root


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_overlay_home_attaches_to_the_owner_its_lock_names(daemon, tmp_path, model_peer):
    overlay = _overlay(daemon, tmp_path)
    async with editor(overlay, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        assert 'result' in created, (created, (tmp_path / 'acp.log').read_text()[-2000:])
        sid = created['result']['sessionId']
        turn = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'WS_SHARED overlay'}])
        assert turn['result']['stopReason'] == 'end_turn', turn
        assert 'result' in await acp.rpc('session/load', cwd=str(tmp_path), sessionId=sid, mcpServers=[])
    async with viewer(daemon) as ws:  # one owner: the mirrored profile's gateway holds the session
        saved = await ws.rpc('session.resume', session_id=sid)
        assert 'WS_SHARED overlay' in json.dumps(saved['messages'])
    assert not (overlay[0] / 'gateway.sock').exists() and not (overlay[0] / 'gateway.pid').exists()


_ONE_PX_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4'
    '890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082')


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_overlay_home_image_prompt_is_staged_where_the_owner_admits_it(daemon, tmp_path, model_peer):
    """The overlay does not share ``cache/images``: an image is staged in the owner's cache, the only
    place its admission accepts (#140190 review)."""
    import base64
    overlay = _overlay(daemon, tmp_path)
    async with editor(overlay, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        turn = await acp.rpc('session/prompt', sessionId=sid, prompt=[
            {'type': 'text', 'text': 'WS_SHARED overlay image'},
            {'type': 'image', 'data': base64.b64encode(_ONE_PX_PNG).decode(), 'mimeType': 'image/png'}])
        assert 'result' in turn, (turn, (tmp_path / 'acp.log').read_text()[-2000:])
        assert turn['result']['stopReason'] == 'end_turn', turn
    assert any(daemon[0].glob('cache/images/img_*.png')), list(daemon[0].rglob('*.png'))
    assert not any(overlay[0].glob('cache/images/*')), list(overlay[0].rglob('*.png'))
