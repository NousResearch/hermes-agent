"""A replaced Honcho client must not leave SDK children on its old bearer."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from types import SimpleNamespace

import httpx
import pytest

from plugins.memory.honcho import client, oauth
from plugins.memory.honcho import session as session_module
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


@pytest.fixture
def sdk_manager(tmp_path, monkeypatch):
    from honcho import Honcho

    clients = []
    requests = []
    session_context_requests = []
    live_token = ["hch-at-test-0"]
    config_path = tmp_path / "honcho.json"
    client.reset_honcho_client()
    for name in (
        "_expiry_cache",
        "_dead_grants",
        "_reauth_check_cache",
        "_refresh_failure_at",
    ):
        monkeypatch.setattr(oauth, name, {})

    def rotate(number):
        live_token[0] = f"hch-at-test-{number}"
        oauth.install_grant(
            config_path,
            "hermes",
            {
                "access_token": live_token[0],
                "refresh_token": f"hch-rt-test-{number}",
                "expires_in": 3600,
            },
            client_id="offline-test",
            token_endpoint="https://oauth.invalid/token",
        )

    rotate(0)

    def config():
        cfg = client.HonchoClientConfig.from_global_config(
            host="hermes", config_path=config_path
        )
        cfg.workspace_id = "offline-workspace"
        cfg.write_frequency = "turn"
        return cfg

    def respond(request):
        bearer = request.headers["authorization"]
        requests.append(bearer)
        if bearer != f"Bearer {live_token[0]}":
            return httpx.Response(
                401, json={"detail": "Invalid or expired access token"}
            )
        if request.url.path.endswith("/card"):
            return httpx.Response(200, json={"peer_card": ["test card"]})
        if request.url.path.endswith("/sessions/conversation/context"):
            session_context_requests.append(bearer)
            return httpx.Response(
                200,
                json={
                    "id": "conversation",
                    "messages": [],
                    "summary": {
                        "content": "test summary",
                        "message_id": "message-1",
                        "summary_type": "short",
                        "created_at": "2026-01-01T00:00:00Z",
                        "token_count": 2,
                    },
                },
            )
        if request.url.path.endswith("/context"):
            return httpx.Response(
                200,
                json={
                    "representation": "test representation",
                    "peer_card": ["test card"],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "offline-user"
                if request.url.path.endswith("/peers")
                else "conversation",
                "workspace_id": "offline-workspace",
                "metadata": {},
                "configuration": {},
                "is_active": True,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    def build(cfg):
        sdk = Honcho(
            api_key=cfg.api_key,
            workspace_id=cfg.workspace_id,
            base_url="https://honcho.invalid",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(respond), trust_env=False
            ),
        )
        clients.append(sdk)
        return sdk

    def no_network(*args, **kwargs):
        pytest.fail("real network is forbidden in this regression")

    def exchange(endpoint, form, timeout):
        # Baseline must reach a successful refresh, then fail because its SDK
        # child still owns the old client. A broken exchange would mask that bug.
        number = int(live_token[0].rsplit("-", 1)[1]) + 1
        live_token[0] = f"hch-at-test-{number}"
        return 200, {
            "access_token": live_token[0],
            "refresh_token": f"hch-rt-test-{number}",
            "expires_in": 3600,
        }

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(client, "_build_client", build)
    monkeypatch.setattr(oauth, "_http_post_form_status", exchange)
    manager = HonchoSessionManager(config=config())
    local = HonchoSession("key", "offline-user", "assistant", "conversation")
    local.add_message("user", "pending message")
    manager._cache["key"] = local
    yield SimpleNamespace(
        manager=manager,
        config=config,
        rotate=rotate,
        requests=requests,
        session_context_requests=session_context_requests,
        live_token=live_token,
        local=local,
    )
    client.reset_honcho_client()
    for sdk in clients:
        sdk._http.close()


@pytest.mark.parametrize(
    "kind,entrypoint",
    [
        ("peer", "authed"),
        ("peer", "direct"),
        ("session", "authed"),
        ("session", "direct"),
        ("get_session_context", "direct"),
        ("get_prefetch_context", "direct"),
    ],
)
@pytest.mark.parametrize("replacement", ["rotated_grant", "reset"])
def test_cached_sdk_children_follow_replaced_client(
    sdk_manager, kind, replacement, entrypoint
):
    state = sdk_manager
    manager = state.manager

    def read():
        if kind == "peer":
            return manager._get_or_create_peer("offline-user").get_card()
        if kind == "session":
            return manager._sdk_session("conversation").context().messages
        return getattr(manager, kind)("key").get("summary")

    expected = {"peer": ["test card"], "session": []}.get(kind, "test summary")
    public_read = kind in ("get_session_context", "get_prefetch_context")
    old = manager.honcho
    cached_peer = manager._get_or_create_peer("offline-user")
    cached_session = manager._sdk_session("conversation")
    assert manager._authed_call("initial read", read) == expected
    # Ordinary acquisitions must retain both caches, not rebuild on every call.
    assert manager.honcho is old
    assert manager._get_or_create_peer("offline-user") is cached_peer
    assert manager._sdk_session("conversation") is cached_session

    state.rotate(1)
    if replacement == "rotated_grant":
        # A fresh session loads the rotated refresh token, evicting the old
        # identity's slot. The older manager must invalidate its SDK children.
        assert client.get_honcho_client(state.config()) is not old
    else:
        client.reset_honcho_client()
    state.requests.clear()
    state.session_context_requests.clear()
    if public_read:
        # Observe replacement via a peer first. Public recall must rebuild its
        # own session handle, without a test-side _sdk_session() priming it.
        assert manager.get_peer_card("key") == ["test card"]

    result = (
        manager._authed_call("read after client replacement", read)
        if entrypoint == "authed"
        else read()
    )
    assert result == expected
    if public_read:
        assert read() == expected
        assert state.session_context_requests == ["Bearer hch-at-test-1"] * 2
    assert state.requests and set(state.requests) == {"Bearer hch-at-test-1"}
    assert manager._get_or_create_peer("offline-user")._honcho is manager.honcho
    assert manager._sdk_session("conversation")._honcho is manager.honcho
    assert manager._auth_failure is None
    # Invalidation is for SDK handles only, never pending conversation data.
    assert manager._cache["key"] is state.local
    assert state.local.messages[0]["content"] == "pending message"
    assert not state.local.messages[0].get("_synced")


@pytest.mark.parametrize("blocked_stage", ["acquisition", "peer", "session"])
def test_late_old_client_work_cannot_repopulate_replacement(monkeypatch, blocked_stage):
    entered, release = Event(), Event()
    old, new = SimpleNamespace(), SimpleNamespace()
    active = [old]

    def acquire(config):
        selected = active[0]
        if blocked_stage == "acquisition" and current_thread().name.startswith(
            "honcho-regression"
        ):
            if selected is old:
                entered.set()
                assert release.wait(5)
        return selected

    def child(owner, kind):
        if owner is old and kind == blocked_stage:
            entered.set()
            assert release.wait(5)
        return SimpleNamespace(_honcho=owner)

    for sdk in (old, new):
        sdk.peer = lambda key, owner=sdk: child(owner, "peer")
        sdk.session = lambda key, owner=sdk: child(owner, "session")
    monkeypatch.setattr(session_module, "get_honcho_client", acquire)
    manager = HonchoSessionManager(
        config=client.HonchoClientConfig(write_frequency="turn")
    )
    assert manager.honcho is old

    def resolve():
        if blocked_stage == "acquisition":
            return manager.honcho
        if blocked_stage == "peer":
            return manager._get_or_create_peer("peer")
        return manager._sdk_session("session")

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="honcho-regression"
    ) as pool:
        future = pool.submit(resolve)
        try:
            assert entered.wait(5)
            active[0] = new
            # This must finish while the older acquisition/construction is
            # blocked: client/network work cannot hold the manager cache lock.
            assert manager.honcho is new
        finally:
            release.set()
        result = future.result(timeout=5)

    assert manager._honcho is new
    if blocked_stage == "acquisition":
        assert result is new
    else:
        cache = (
            manager._peers_cache if blocked_stage == "peer" else manager._sessions_cache
        )
        assert result._honcho is new
        assert cache[blocked_stage] is result
