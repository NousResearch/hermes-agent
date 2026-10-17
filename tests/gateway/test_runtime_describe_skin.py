"""``runtime.describe`` carries the owning profile's resolved skin: main's TUI booted its theme from
``gateway.ready.skin`` (``tui_gateway/entry.py`` / ``ws.py`` ``resolve_skin()``), and the canonical
client publishes its own ready from this discovery reply (FINDINGS row 3: ``-p work`` with
``display.skin: ares`` must boot Ares, not the default gold theme)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_local_source_labels import _authority


@pytest.mark.asyncio
async def test_runtime_describe_carries_the_profiles_resolved_skin(tmp_path, monkeypatch):
    authority, _store = await _authority(tmp_path, monkeypatch)
    home = Path(authority.profile_id)
    (home / 'config.yaml').write_text('display:\n  skin: ares\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        described = await conn.dispatch({'id': 0, 'method': 'runtime.describe', 'params': {}})
        assert 'result' in described, described
        skin = described['result']['skin']
        assert skin['name'] == 'ares'
        assert skin['colors'] and skin['branding']
        from tui_gateway.contracts.registry import CANONICAL_METHODS
        CANONICAL_METHODS['runtime.describe'].result.model_validate(described['result'])
    finally:
        await conn.close()
        authority.db.close()
