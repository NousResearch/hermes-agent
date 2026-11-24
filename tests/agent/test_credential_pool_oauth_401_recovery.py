from __future__ import annotations

import itertools
import time

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import AUTH_TYPE_OAUTH, STATUS_EXHAUSTED, CredentialPool, PooledCredential

_API = "https://api.anthropic.com"
_EIGHT_HOURS_MS = 8 * 3600 * 1000
_ISSUE_RETRY_AFTER_S = 600.0


@pytest.fixture(autouse=True)
def _no_claude_code_credentials(monkeypatch):
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)


class _TokenEndpoint:
    def __init__(self):
        self.posts: list[str] = []
        self.live: dict[str, str] = {}
        self.ttl_ms = _EIGHT_HOURS_MS

    def refresh(self, refresh_token, *, use_json=False):
        self.posts.append(refresh_token)
        login, rotation = refresh_token.rsplit("-rt-", 1)
        rotation = int(rotation) + 1
        self.live[login] = f"{login}-at-{rotation}"
        return {
            "access_token": self.live[login],
            "refresh_token": f"{login}-rt-{rotation}",
            "expires_at_ms": int(time.time() * 1000) + self.ttl_ms,
        }

    def accepts(self, access_token):
        return access_token in self.live.values()


@pytest.fixture
def endpoint(monkeypatch):
    server = _TokenEndpoint()
    monkeypatch.setattr("agent.anthropic_credentials.refresh_anthropic_oauth_pure", server.refresh)
    return server


def _row(eid, *, access=None, refresh=None, expires_in_ms=_EIGHT_HOURS_MS, priority=0):
    return PooledCredential(
        provider="anthropic", id=eid, label=eid, auth_type=AUTH_TYPE_OAUTH, priority=priority,
        source="manual:hermes_pkce", access_token=f"{eid}-at-0" if access is None else access,
        refresh_token=refresh or f"{eid}-rt-0", expires_at_ms=int(time.time() * 1000) + expires_in_ms,
        base_url=_API,
    )


class _DelegatedChild:
    provider = "anthropic"
    model = "claude-opus-4-6"
    base_url = _API
    _fallback_activated = False

    def __init__(self, pool, entry_id):
        self._credential_pool = pool
        self._credential_pool_entry_id = entry_id
        self.api_key = next(e for e in pool.entries() if e.id == entry_id).runtime_api_key

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def _on_401(agent):
    return recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False,
        error_context={"message": "OAuth access token has expired"},
    )


def _reload(fake_pool_store):
    persisted = next(row for row in fake_pool_store["anthropic"] if row["id"] == "claude-3")
    return CredentialPool("anthropic", [PooledCredential.from_dict("anthropic", persisted)])


class TestStale401AfterRotation:
    def test_stale_hint_adopts_the_rotated_token_without_a_post(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3", access="claude-3-at-1", refresh="claude-3-rt-1")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == []
        assert (got.id, got.access_token) == ("claude-3", "claude-3-at-1")

    def test_the_failed_current_token_is_refreshed(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-0"]
        assert got.access_token == "claude-3-at-1"

    def test_a_rotated_token_that_is_expiring_is_refreshed(self, endpoint, fake_pool_store):
        pool = CredentialPool(
            "anthropic", [_row("claude-3", access="claude-3-at-1", refresh="claude-3-rt-1", expires_in_ms=30_000)],
        )

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-1"]
        assert got.access_token == "claude-3-at-2"

    def test_a_row_without_a_bearer_is_refreshed_not_adopted(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3", access="", refresh="claude-3-rt-1")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-1"]
        assert got.access_token == "claude-3-at-2"

    def test_siblings_sharing_a_pool_do_not_revoke_each_other(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3")])
        first, second = _DelegatedChild(pool, "claude-3"), _DelegatedChild(pool, "claude-3")

        assert _on_401(first) == (True, False)
        assert _on_401(second) == (True, False)

        assert endpoint.posts == ["claude-3-rt-0"]
        assert first.api_key == second.api_key == "claude-3-at-1"
        assert endpoint.accepts(first.api_key)

    def test_a_reloaded_pool_adopts_the_token_rotated_under_its_inflight_request(self, endpoint, fake_pool_store):
        first = _DelegatedChild(CredentialPool("anthropic", [_row("claude-3")]), "claude-3")
        stale_key = first.api_key
        assert _on_401(first) == (True, False)
        second = _DelegatedChild(_reload(fake_pool_store), "claude-3")
        second.api_key = stale_key

        assert _on_401(second) == (True, False)

        assert second.api_key == first.api_key == "claude-3-at-1"
        assert endpoint.accepts(second.api_key)
        assert endpoint.posts == ["claude-3-rt-0"]


@pytest.mark.parametrize(
    "topology,sent,current_expiring",
    list(itertools.product(("shared_pool", "reloaded_pool"), ("stale_gen0", "current_gen1", "no_hint"), (False, True))),
)
def test_a_second_401_refreshes_only_when_its_own_token_is_due(
    endpoint, fake_pool_store, topology, sent, current_expiring,
):
    if current_expiring:
        endpoint.ttl_ms = 30_000
    first_pool = CredentialPool("anthropic", [_row("claude-3")])
    assert _on_401(_DelegatedChild(first_pool, "claude-3")) == (True, False)
    assert endpoint.posts == ["claude-3-rt-0"]
    pool = first_pool if topology == "shared_pool" else _reload(fake_pool_store)

    if sent == "no_hint":
        got = pool.try_refresh_matching(credential_id="claude-3")
    else:
        second = _DelegatedChild(pool, "claude-3")
        second.api_key = "claude-3-at-0" if sent == "stale_gen0" else "claude-3-at-1"
        assert _on_401(second) == (True, False)
        got = next(e for e in pool.entries() if e.id == "claude-3")
        assert second.api_key == got.access_token

    must_refresh = sent != "stale_gen0" or current_expiring
    assert endpoint.posts == (["claude-3-rt-0", "claude-3-rt-1"] if must_refresh else ["claude-3-rt-0"])
    assert got.access_token == ("claude-3-at-2" if must_refresh else "claude-3-at-1")
    assert endpoint.accepts(got.access_token)


def _status(pool, eid):
    return next(e for e in pool.entries() if e.id == eid).last_status


def _two_refreshes_then_401(child):
    assert _on_401(child) == (True, False)
    assert _on_401(child) == (True, False)
    return _on_401(child)


class TestRefreshCap:
    def test_after_the_cap_the_session_rotates_to_another_healthy_row(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3"), _row("claude-4", priority=1)])
        child = _DelegatedChild(pool, "claude-3")

        recovered, _ = _two_refreshes_then_401(child)

        assert recovered is True
        assert (child._credential_pool_entry_id, child.api_key) == ("claude-4", "claude-4-at-0")
        assert _status(pool, "claude-3") == STATUS_EXHAUSTED
        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1"]

    def test_a_capped_stale_id_does_not_bench_the_row_whose_key_failed(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3"), _row("claude-4", priority=1)])
        child = _DelegatedChild(pool, "claude-3")
        assert _on_401(child) == (True, False)
        assert _on_401(child) == (True, False)
        child.api_key = "claude-4-at-0"

        assert _on_401(child) == (True, False)
        assert endpoint.posts[-1] == "claude-4-rt-0"
        assert _status(pool, "claude-3") != STATUS_EXHAUSTED
        assert _status(pool, "claude-4") != STATUS_EXHAUSTED

    def test_a_served_response_between_401s_ends_the_episode(self, endpoint, fake_pool_store):
        from agent.credential_auth_recovery import end_auth_episodes
        pool = CredentialPool("anthropic", [_row("claude-3")])
        child = _DelegatedChild(pool, "claude-3")
        for _ in range(3):
            assert _on_401(child) == (True, False)
            end_auth_episodes(child)

        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1", "claude-3-rt-2"]

    def test_at_the_cap_without_another_row_the_call_fails_without_a_post(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3")])
        child = _DelegatedChild(pool, "claude-3")

        assert _two_refreshes_then_401(child) == (False, False)
        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1"]

    def test_issue_timeline_401s_ten_minutes_apart_keep_refreshing(self, endpoint, fake_pool_store, monkeypatch):
        import types
        import agent.credential_auth_recovery as rec
        clock = [1000.0]
        monkeypatch.setattr(rec, "time", types.SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: None))
        pool = CredentialPool("anthropic", [_row("claude-3"), _row("claude-4", priority=1)])
        child = _DelegatedChild(pool, "claude-3")
        for _ in range(4):
            assert _on_401(child) == (True, False)
            clock[0] += _ISSUE_RETRY_AFTER_S

        assert child._credential_pool_entry_id == "claude-3"
        assert len(endpoint.posts) == 4


class TestCapAdoptsAPeerRotation:
    def _at_the_cap(self, pool):
        child = _DelegatedChild(pool, "claude-3")
        assert _on_401(child) == (True, False)
        assert _on_401(child) == (True, False)
        return child

    def test_a_token_a_peer_process_rotated_is_adopted_at_the_cap(self, endpoint, fake_pool_store):
        child = self._at_the_cap(CredentialPool("anthropic", [_row("claude-3")]))
        assert _reload(fake_pool_store).try_refresh_matching(credential_id="claude-3").access_token == "claude-3-at-3"

        assert _on_401(child) == (True, False)
        assert child.api_key == "claude-3-at-3"
        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1", "claude-3-rt-2"]

    def test_a_token_the_shared_pool_already_holds_is_adopted_at_the_cap(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3")])
        child = self._at_the_cap(pool)
        assert pool.try_refresh_matching(credential_id="claude-3").access_token == "claude-3-at-3"

        assert _on_401(child) == (True, False)
        assert child.api_key == "claude-3-at-3"
        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1", "claude-3-rt-2"]


def test_a_truncated_reply_ends_the_episode_before_its_continuation(endpoint, fake_pool_store, monkeypatch):
    import types
    import agent.turn_response_check as trc
    monkeypatch.setattr(trc, "stop_thinking_spinner", lambda agent, spinner: None)
    monkeypatch.setattr("agent.turn_recovery.validate_response_shape", lambda agent, response: (False, None))
    monkeypatch.setattr("agent.turn_usage.note_response_reasoning_tokens", lambda agent, response: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_harness.record_reply_finish", lambda *a: None)
    monkeypatch.setattr(trc, "_derive_finish_reason", lambda agent, response, messages: "length")
    monkeypatch.setattr(trc, "recover_from_truncation", lambda *a, **k: types.SimpleNamespace(
        action="continue", result=None, messages=[], length_continue_retries=1, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0))
    child = _DelegatedChild(CredentialPool("anthropic", [_row("claude-3")]), "claude-3")
    child.quiet_mode, child.verbose_logging = True, False
    assert _on_401(child) == (True, False)
    assert _on_401(child) == (True, False)
    unused = dict.fromkeys((
        "_retry", "thinking_spinner", "messages", "api_messages", "api_kwargs", "active_system_prompt",
        "conversation_history", "finish_reason", "retry_count", "max_retries", "compression_attempts",
        "max_compression_attempts", "length_continue_retries", "truncated_response_parts",
        "truncated_tool_call_retries", "current_turn_user_idx", "api_call_count", "api_request_id",
        "effective_task_id", "turn_id", "_preflight_compression_blocked", "_last_preflight_pressure"))

    verdict = trc.check_api_response(child, response=object(), api_start_time=time.time(), **unused)

    assert verdict.action == "continue"
    assert _on_401(child) == (True, False)
    assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1", "claude-3-rt-2"]
