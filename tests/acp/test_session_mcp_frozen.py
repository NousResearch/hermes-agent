"""Selected editor schemas never refresh inside an execution or escape discovery."""
import json
import sys
import time

import pytest

from gateway.session_policy import policy_scope
from hermes_state_runtime import RuntimeStoreError
from tests.acp.test_session_mcp_policy import owner, policy, server_spec  # noqa: F401


def test_editor_manifest_stays_frozen_across_live_refresh(server_spec, tmp_path):
    from tools.registry import registry
    from tools.mcp_tool import _servers
    from tools.mcp_tool_loop import _run_on_mcp_loop
    authority = owner()
    p = policy(authority, 'freeze', server_spec, tmp_path)
    with policy_scope(p, authority=authority):
        names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
        schemas = {n: registry.get_schema(n) for n in names}
        server = next(s for key, s in _servers.items() if key[-1].startswith('editor_'))
        # The real refresh re-reads the stdio peer. A changed in-memory discovery
        # response is not allowed to republish schemas even when explicitly refreshed.
        server._tools[0].description = 'changed after first discovery'
        from tools.mcp_tool_registration import _register_server_tools
        with pytest.raises(RuntimeStoreError, match='acp_mcp_schema_changed'):
            _register_server_tools(server.name, server, server._config)
        assert schemas == {n: registry.get_schema(n) for n in names}
        _run_on_mcp_loop(server._refresh_tools, timeout=10)
        assert schemas == {n: registry.get_schema(n) for n in names}
    # Changing the actual server program preserves the transport fingerprint,
    # but must not replace the session's selected schema on the next execution.
    script = tmp_path / 'owned_mcp.py'
    script.write_text(script.read_text().replace('owned echo', 'changed live schema'))
    with pytest.raises(RuntimeStoreError, match='acp_mcp_schema_changed'):
        with policy_scope(p, authority=authority):
            pass


def test_silent_editor_discovery_is_bounded_and_reaps_only_itself(server_spec, tmp_path):
    from tools.registry import registry
    authority = owner()
    healthy = policy(authority, 'healthy', server_spec, tmp_path)
    silent = {'name': 'same-editor-name', 'command': sys.executable,
              'args': ['-c', 'import time; time.sleep(120)'], 'env': []}
    with policy_scope(healthy, authority=authority):
        started = time.monotonic()
        with pytest.raises(RuntimeStoreError, match='acp_mcp_discovery_failed'):
            policy(authority, 'silent', silent, tmp_path)
        assert time.monotonic() - started < 25
        names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
        assert len(names) == 1
        assert 'PRIVATE_A' in registry.dispatch(names[0], {})
    from tools.mcp_tool import _servers
    assert not any(key[-1].startswith('editor_') for key in _servers)
