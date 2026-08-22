import hashlib
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gateway import browser_handoff as handoff


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _create(store, *, token="a" * 64, ttl=1800, source=None):
    return store.create(
        token_digest=hashlib.sha256(token.encode("ascii")).hexdigest(),
        session_id="session-exact",
        browser_key="task-1",
        browser_id="browser-1",
        live_url="https://live.browser-use.com/?session=secret",
        instruction="Complete the CAPTCHA",
        source=source
        or {"platform": "discord", "chat_id": "123", "chat_type": "dm"},
        ttl_seconds=ttl,
    )


def test_store_hashes_token_and_completion_is_one_shot(tmp_path):
    path = tmp_path / "handoffs.db"
    store = handoff.BrowserHandoffStore(path)
    token = "b" * 64
    record = _create(store, token=token)

    raw = path.read_bytes()
    assert token.encode() not in raw
    assert store.lookup(token).status == "pending"
    assert store.complete(token).status == "completed"
    assert store.complete(token).status == "completed"
    assert store.claim_wake(record.id) is not None
    assert store.claim_wake(record.id) is None
    store.finish_wake(record.id, delivered=True)
    assert store.lookup(token).wake_status == "delivered"


def test_expired_and_revoked_handoffs_never_render_live_url(tmp_path):
    store = handoff.BrowserHandoffStore(tmp_path / "handoffs.db")
    expired_token = "c" * 64
    expired = _create(store, token=expired_token, ttl=-1)
    expired = store.lookup(expired_token)
    page, status = handoff.render_handoff_page(expired)
    assert status == 410
    assert "live.browser-use.com" not in page

    active_token = "d" * 64
    active = _create(store, token=active_token)
    assert store.cancel_for_browser(active.browser_id) == 1
    page, status = handoff.render_handoff_page(store.lookup(active_token))
    assert status == 410
    assert "live.browser-use.com" not in page


def test_second_pending_handoff_does_not_invalidate_first(tmp_path):
    store = handoff.BrowserHandoffStore(tmp_path / "handoffs.db")
    token = "1" * 64
    _create(store, token=token)

    with pytest.raises(handoff.BrowserHandoffError, match="already pending"):
        _create(store, token="2" * 64)

    assert store.lookup(token).status == "pending"
    assert store.retains_browser("browser-1") is True
    store.cancel_for_browser("browser-1")
    assert store.retains_browser("browser-1") is False


def test_expiry_is_wakeable_and_retains_browser_until_delivery(tmp_path):
    store = handoff.BrowserHandoffStore(tmp_path / "handoffs.db")
    record = _create(store, token="3" * 64, ttl=-1)

    assert store.pending_wake_ids() == [record.id]
    claimed = store.claim_wake(record.id)
    assert claimed is not None and claimed.status == "expired"
    assert store.retains_browser(record.browser_id) is True

    store.finish_wake(record.id, delivered=True)
    assert store.retains_browser(record.browser_id) is False


def test_completed_handoff_retains_browser_through_wake(tmp_path):
    store = handoff.BrowserHandoffStore(tmp_path / "handoffs.db")
    token = "4" * 64
    record = _create(store, token=token)
    assert store.complete(token).status == "completed"
    assert store.retains_browser(record.browser_id) is True
    assert store.claim_wake(record.id) is not None
    assert store.retains_browser(record.browser_id) is True
    store.finish_wake(record.id, delivered=True)
    assert store.retains_browser(record.browser_id) is False


def test_active_page_has_remote_browser_done_and_security_headers(tmp_path):
    store = handoff.BrowserHandoffStore(tmp_path / "handoffs.db")
    token = "e" * 64
    page, status = handoff.render_handoff_page(_create(store, token=token))
    headers = handoff.security_headers()

    assert status == 200
    assert "<iframe" in page and ">Done</button>" in page
    assert "Complete the CAPTCHA" in page
    assert headers["Cache-Control"].startswith("no-store")
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_create_handoff_dms_owner_without_persisting_bearer(
    isolated_home, monkeypatch
):
    config = handoff.BrowserHandoffConfig(
        enabled=True,
        public_base_url="https://handoff.example",
        ttl_minutes=30,
        discord_user_id="1063878950851448853",
    )
    monkeypatch.setattr(handoff, "load_browser_handoff_config", lambda: config)
    monkeypatch.setattr(
        "tools.browser_tool._get_session_info",
        lambda key: seen_keys.append(key) or {
            "bb_session_id": "browser-1",
            "live_url": "https://live.browser-use.com/?session=secret",
            "features": {"browser_use": True},
        },
    )
    monkeypatch.setenv("HERMES_SESSION_ID", "session-exact")
    sent = []
    seen_keys = []

    result = handoff.create_browser_handoff(
        instruction="Log in and solve the CAPTCHA",
        task_id="task-1",
        notifier=lambda user_id, message: sent.append((user_id, message)),
    )

    assert result["handoff_id"]
    from tools.browser_use_cli import _provider_session_cache_key

    assert seen_keys == [_provider_session_cache_key("task-1")]
    assert sent[0][0] == "1063878950851448853"
    assert "yo i need u to do this" in sent[0][1]
    assert "https://handoff.example/browser-handoff/" in sent[0][1]
    token = sent[0][1].split("/browser-handoff/", 1)[1].splitlines()[0]
    assert len(token) == 64
    db_bytes = (isolated_home / "state" / "browser-handoffs.db").read_bytes()
    assert token.encode() not in db_bytes


def test_secondary_profile_handoff_link_uses_profile_prefix(
    isolated_home, monkeypatch
):
    config = handoff.BrowserHandoffConfig(
        enabled=True,
        public_base_url="https://handoff.example",
        ttl_minutes=30,
        discord_user_id="1063878950851448853",
    )
    monkeypatch.setattr(handoff, "load_browser_handoff_config", lambda: config)
    monkeypatch.setattr(
        "tools.browser_tool._get_session_info",
        lambda key: {
            "bb_session_id": "browser-profile",
            "live_url": "https://live.browser-use.com/?session=secret",
            "features": {"browser_use": True},
        },
    )
    monkeypatch.setenv("HERMES_SESSION_ID", "session-exact")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "worker")
    sent = []

    handoff.create_browser_handoff(
        instruction="Sign in",
        task_id="task-profile",
        notifier=lambda user_id, message: sent.append(message),
    )

    assert "https://handoff.example/p/worker/browser-handoff/" in sent[0]


def test_rate_limit_state_is_hard_bounded_and_stops_token_allocation_for_peer():
    from gateway.platforms.api_server import APIServerAdapter

    adapter = object.__new__(APIServerAdapter)
    adapter._browser_handoff_attempts = {}
    request = SimpleNamespace(remote="203.0.113.1")
    for index in range(500):
        adapter._browser_handoff_rate_limited(request, f"{index:064x}")
    # One peer bucket plus token buckets admitted before the peer was blocked.
    assert len(adapter._browser_handoff_attempts) <= 61

    for index in range(2200):
        distributed = SimpleNamespace(remote=f"198.51.{index // 256}.{index % 256}")
        adapter._browser_handoff_rate_limited(distributed, f"{index + 500:064x}")
    assert len(adapter._browser_handoff_attempts) <= 2048


@pytest.mark.asyncio
async def test_public_done_handler_has_atomic_single_wake_boundary(isolated_home, monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter

    token = "f" * 64
    record = _create(handoff.BrowserHandoffStore(), token=token)
    adapter = object.__new__(APIServerAdapter)
    adapter._browser_handoff_attempts = {}
    scheduled = []
    adapter._schedule_browser_handoff_wake = scheduled.append
    monkeypatch.setattr(
        handoff,
        "load_browser_handoff_config",
        lambda: handoff.BrowserHandoffConfig(
            True, "https://handoff.example", 30, "1063878950851448853"
        ),
    )
    request = SimpleNamespace(
        match_info={"token": token},
        headers={"Origin": "https://handoff.example"},
        remote="127.0.0.1",
    )

    first = await adapter._handle_browser_handoff_complete(request)
    second = await adapter._handle_browser_handoff_complete(request)

    assert first.status == second.status == 200
    assert scheduled == [record.id, record.id]
    # Scheduling may repeat, but BrowserHandoffStore.claim_wake is the atomic
    # single-flight boundary that permits only one actual wake delivery.
    store = handoff.BrowserHandoffStore()
    assert store.claim_wake(record.id) is not None
    assert store.claim_wake(record.id) is None


@pytest.mark.asyncio
async def test_api_origin_wake_uses_raw_session_without_gateway_lookup(
    isolated_home, monkeypatch
):
    from gateway.platforms.api_server import APIServerAdapter

    token = "5" * 64
    store = handoff.BrowserHandoffStore()
    record = _create(
        store,
        token=token,
        source={
            "platform": "api_server",
            "chat_id": "raw-session",
            "chat_type": "dm",
            "profile": "worker",
        },
    )
    store.complete(token)

    class NoGatewayLookup:
        async def lookup_by_session_key(self, key):
            raise AssertionError("API sessions do not have gateway routing rows")

    class Runner:
        async_session_store = NoGatewayLookup()

        @staticmethod
        def _adapter_for_source(source):
            return adapter

        @staticmethod
        def _session_key_for_source(source):
            return "must-not-be-used"

    adapter = object.__new__(APIServerAdapter)
    adapter.gateway_runner = Runner()
    adapter._app = None
    delivered = []

    async def fake_deliver(target, **kwargs):
        delivered.append((target, kwargs))

    monkeypatch.setattr("gateway.wake.deliver_wake", fake_deliver)

    await adapter._deliver_browser_handoff_wake(record.id)

    assert delivered[0][0] is adapter
    assert delivered[0][1]["session_id"] == "session-exact"
    assert delivered[0][1]["source"] is None
    assert delivered[0][1]["profile"] == "worker"
    assert store.lookup(token).wake_status == "delivered"


def test_dm_reports_provider_capped_effective_expiry(isolated_home, monkeypatch):
    config = handoff.BrowserHandoffConfig(
        True, "https://handoff.example", 30, "1063878950851448853"
    )
    monkeypatch.setattr(handoff, "load_browser_handoff_config", lambda: config)
    monkeypatch.setattr(
        "tools.browser_tool._get_session_info",
        lambda key: {
            "bb_session_id": "browser-short",
            "live_url": "https://live.browser-use.com/?session=secret",
            "expires_at": datetime.fromtimestamp(
                time.time() + 20, tz=timezone.utc
            ).isoformat(),
            "features": {"browser_use": True},
        },
    )
    monkeypatch.setenv("HERMES_SESSION_ID", "session-exact")
    sent = []

    result = handoff.create_browser_handoff(
        instruction="Complete MFA",
        task_id="task-short",
        notifier=lambda user_id, message: sent.append(message),
    )

    assert "expires in 1 minute." in sent[0]
    assert "valid for about 1 minute." in result["message"]


def test_tool_schema_exposes_handoff_without_requiring_dummy_code():
    from tools.browser_use_cli import BROWSER_EXEC_SCHEMA

    params = BROWSER_EXEC_SCHEMA["parameters"]
    assert params["properties"]["action"]["enum"] == ["exec", "handoff"]
    assert "code" not in params["required"]


def test_browser_exec_handoff_returns_waiting_result_without_running_cli(monkeypatch):
    import tools.browser_use_cli as browser_use_cli

    monkeypatch.setattr(
        handoff,
        "create_browser_handoff",
        lambda **kwargs: {
            "handoff_id": 42,
            "expires_at": 1234.0,
            "message": "wait for Done",
        },
    )
    monkeypatch.setattr(
        browser_use_cli,
        "_find_cli",
        lambda: (_ for _ in ()).throw(AssertionError("CLI must not run")),
    )

    result = json.loads(
        browser_use_cli.browser_exec(
            action="handoff",
            instruction="Complete CAPTCHA",
            task_id="task-1",
        )
    )

    assert result["success"] is True
    assert result["output"] == "wait for Done"
    assert result["meta"]["handoff_id"] == 42


def test_public_routes_use_separate_get_and_complete_paths():
    from gateway.platforms.api_server import APIServerAdapter

    adapter = object.__new__(APIServerAdapter)
    routes = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("GET", "/browser-handoff/{token}") in routes
    assert ("POST", "/browser-handoff/{token}/complete") in routes


def test_access_log_redacts_handoff_bearer_token():
    from gateway.platforms.api_server import _redact_browser_handoff_log_path

    token = "a" * 64
    assert _redact_browser_handoff_log_path(
        f"/browser-handoff/{token}/complete"
    ) == "/browser-handoff/[REDACTED]/complete"
    assert _redact_browser_handoff_log_path(
        f"/p/work/browser-handoff/{token}"
    ) == "/p/work/browser-handoff/[REDACTED]"
    assert _redact_browser_handoff_log_path("/v1/models") == "/v1/models"


def test_successful_handoff_is_a_deterministic_turn_boundary():
    from agent.conversation_loop import _browser_handoff_wait_message

    messages = [
        {
            "role": "tool",
            "name": "browser_exec",
            "content": json.dumps(
                {
                    "success": True,
                    "output": "DM sent; waiting for Done",
                    "meta": {"browser_handoff": True},
                }
            ),
        }
    ]
    assert _browser_handoff_wait_message(messages) == "DM sent; waiting for Done"
    assert _browser_handoff_wait_message(
        [{"role": "tool", "name": "browser_exec", "content": "{}"}]
    ) == ""


def test_per_turn_cleanup_preserves_handoff_browser_once(monkeypatch):
    from agent import chat_completion_helpers as helpers
    import run_agent

    calls = []
    monkeypatch.setattr(helpers, "is_persistent_env", lambda task_id: True)
    monkeypatch.setattr(run_agent, "cleanup_browser", lambda task_id: calls.append(task_id))
    agent = SimpleNamespace(
        _preserve_browser_after_turn=True,
        verbose_logging=False,
    )

    helpers.cleanup_task_resources(agent, "task-1")
    assert calls == []
    assert agent._preserve_browser_after_turn is False

    helpers.cleanup_task_resources(agent, "task-1")
    assert calls == ["task-1"]
