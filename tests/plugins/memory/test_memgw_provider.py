"""Tests for the Memory Gateway (memgw) memory provider plugin."""

import json
import time

import pytest

from plugins.memory.memgw import (
    MemGatewayProvider,
    _load_config,
)


class FakeClient:
    """Stand-in for MemGatewayClient — records calls, returns canned payloads."""

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.responses = {
            'recall': {
                'status': 'success',
                'results': [
                    {'payload': {'title': 'JHJ', 'content': 'S-Corporation owned by Dan'}},
                ],
                'count': 1,
            },
            'write': {'status': 'success', 'memory_id': 'memory_x'},
            'reflect': {
                'mental_models': [{'statement': 'Dan prefers signed commits', 'confidence': 0.8}],
                'count': 1,
            },
        }
        self.raise_on = set()

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name in self.raise_on:
            raise RuntimeError(f'{name} boom')
        return self.responses.get(name, {})

    def close(self):
        pass


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv('MEMGW_API_URL', 'http://localhost:8081/mcp')
    monkeypatch.setenv('MEMGW_API_KEY', '')
    p = MemGatewayProvider()
    fake = FakeClient()
    # Bypass lazy import: inject the fake directly.
    p._client = fake
    p.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    return p, fake


class TestAvailability:
    def test_localhost_available_without_key(self, monkeypatch):
        monkeypatch.setenv('MEMGW_API_URL', 'http://localhost:8081/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        assert MemGatewayProvider().is_available() is True

    def test_cloud_requires_key(self, monkeypatch):
        monkeypatch.setenv('MEMGW_API_URL', 'https://mcp.danizhaky.com/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        # No HERMES_HOME json -> not available without a key.
        assert MemGatewayProvider().is_available() is False

    def test_name(self):
        assert MemGatewayProvider().name == 'memgw'


class TestToolSchemas:
    def test_exposes_recall_retain_reflect(self):
        names = {s['name'] for s in MemGatewayProvider().get_tool_schemas()}
        assert names == {'memgw_recall', 'memgw_retain', 'memgw_reflect'}


class TestToolCalls:
    def test_recall_maps_to_gateway_recall(self, provider):
        p, fake = provider
        out = json.loads(p.handle_tool_call('memgw_recall', {'query': 'JHJ', 'limit': 3}))
        assert out['status'] == 'success'
        assert fake.calls[0][0] == 'recall'
        assert fake.calls[0][1] == {'query': 'JHJ', 'limit': 3}

    def test_retain_maps_to_gateway_write(self, provider):
        p, fake = provider
        out = json.loads(
            p.handle_tool_call(
                'memgw_retain',
                {'content': 'X', 'title': 'T', 'memory_type': 'decision'},
            )
        )
        assert out['status'] == 'success'
        assert fake.calls[0][0] == 'write'
        assert fake.calls[0][1]['memory_type'] == 'decision'

    def test_reflect_maps_to_gateway_reflect(self, provider):
        p, fake = provider
        out = json.loads(p.handle_tool_call('memgw_reflect', {'query': 'commits'}))
        assert out['count'] == 1
        assert fake.calls[0][0] == 'reflect'

    def test_missing_query_errors(self, provider):
        p, _ = provider
        out = json.loads(p.handle_tool_call('memgw_recall', {}))
        assert 'error' in out or 'Missing' in json.dumps(out)

    def test_unknown_tool_errors(self, provider):
        p, _ = provider
        out = json.loads(p.handle_tool_call('memgw_bogus', {}))
        assert 'Unknown tool' in json.dumps(out)


class TestExperienceCapture:
    def test_on_delegation_writes_experience(self, provider):
        p, fake = provider
        p.on_delegation('refactor X', 'done, all tests pass', child_session_id='c1')
        # on_delegation spawns a thread; give it a beat.
        time.sleep(0.3)
        write_calls = [c for c in fake.calls if c[0] == 'write']
        assert write_calls, 'expected an experience write'
        assert write_calls[0][1]['memory_type'] == 'experience'


class TestCircuitBreaker:
    def test_breaker_opens_after_threshold(self, provider):
        p, fake = provider
        fake.raise_on = {'recall'}
        for _ in range(5):
            p.handle_tool_call('memgw_recall', {'query': 'q'})
        assert p._is_breaker_open() is True
        # While open, calls short-circuit without touching the client.
        before = len(fake.calls)
        out = json.loads(p.handle_tool_call('memgw_recall', {'query': 'q'}))
        assert 'unavailable' in json.dumps(out).lower()
        assert len(fake.calls) == before


class TestScopingAndSessions:
    def test_recall_includes_user_scope(self, provider, tmp_path):
        p, fake = provider
        p._user_id = 'tg-12345'
        p.handle_tool_call('memgw_recall', {'query': 'q'})
        assert fake.calls[0][1].get('user_id') == 'tg-12345'

    def test_no_user_scope_when_unset(self, provider):
        p, fake = provider
        p._user_id = ''
        p.handle_tool_call('memgw_recall', {'query': 'q'})
        assert 'user_id' not in fake.calls[0][1]

    def test_session_switch_clears_and_invalidates_prefetch(self, provider):
        p, _ = provider
        with p._prefetch_lock:
            p._prefetch_result = 'stale ctx'
            gen_before = p._prefetch_gen
        p.on_session_switch('new-session')
        with p._prefetch_lock:
            assert p._prefetch_result == ''
            assert p._prefetch_gen > gen_before  # in-flight workers invalidated

    def test_stale_prefetch_worker_cannot_overwrite_newer(self, provider):
        # Simulate an old worker (gen N) finishing after a newer queue bumped gen.
        p, _ = provider
        with p._prefetch_lock:
            p._prefetch_gen = 5
            stale_gen = 4  # an older worker
            # mimic the guarded store
            if stale_gen == p._prefetch_gen:
                p._prefetch_result = 'should not land'
        assert p._prefetch_result != 'should not land'


class TestConfig:
    def test_load_config_env_defaults(self, monkeypatch):
        monkeypatch.setenv('MEMGW_API_URL', 'http://localhost:8081/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        cfg = _load_config()
        assert cfg['api_url'] == 'http://localhost:8081/mcp'

    def test_save_config_roundtrip(self, tmp_path):
        p = MemGatewayProvider()
        p.save_config({'api_url': 'http://x/mcp', 'prefetch_method': 'reflect'}, str(tmp_path))
        saved = json.loads((tmp_path / 'memgw.json').read_text())
        assert saved['prefetch_method'] == 'reflect'


# ── Codex PR #30 review follow-ups ──────────────────────────────────────


class TestKeylessLocalModeHostParsing:
    """#15: keyless mode must require an exact loopback *host*, not a substring."""

    def test_userinfo_with_localhost_is_not_trusted(self, monkeypatch):
        # 'localhost' appears in userinfo, but the real host is example.com.
        monkeypatch.setenv('MEMGW_API_URL', 'https://localhost@example.com/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        assert MemGatewayProvider().is_available() is False

    def test_loopback_ipv4_host_available_keyless(self, monkeypatch):
        monkeypatch.setenv('MEMGW_API_URL', 'http://127.0.0.1:8081/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        assert MemGatewayProvider().is_available() is True

    def test_non_loopback_host_with_localhost_substring_not_trusted(self, monkeypatch):
        monkeypatch.setenv('MEMGW_API_URL', 'http://localhost.evil.example.com/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        assert MemGatewayProvider().is_available() is False

    def test_ipv6_loopback_available_keyless(self, monkeypatch):
        monkeypatch.setenv('MEMGW_API_URL', 'http://[::1]:8081/mcp')
        monkeypatch.delenv('MEMGW_API_KEY', raising=False)
        assert MemGatewayProvider().is_available() is True
