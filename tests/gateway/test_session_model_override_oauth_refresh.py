"""Session Codex overrides follow live credentials without resetting the conversation."""

import base64
import json
import time
from unittest.mock import patch

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_constants import get_hermes_home

CHATGPT = "https://chatgpt.com/backend-api/codex"
GATEWAY = "https://codex-gateway.example/backend-api/codex"
MODEL = "gpt-5.5"


def _token(marker):
    claims = json.dumps({"exp": int(time.time()) + 3600, "sub": marker}).encode()
    return (
        "eyJhbGciOiJSUzI1NiJ9."
        + base64.urlsafe_b64encode(claims).decode().rstrip("=")
        + ".sig"
    )


def _runner(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.session_store = SessionStore(
        sessions_dir=tmp_path / "sessions", config=GatewayConfig()
    )
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="user", chat_id="chat", chat_type="dm"
    )
    entry = runner.session_store.get_or_create_session(source)
    return runner, entry


def _write_pool(token, base_url, api_mode):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    config = "model:\n  provider: openai-codex\n  default: gpt-5.5\n"
    if api_mode == "codex_app_server":
        config += "  openai_runtime: codex_app_server\n"
    (home / "config.yaml").write_text(config)
    (home / "auth.json").write_text(
        json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "test-account",
                        "label": "test",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": token,
                        "refresh_token": "synthetic-refresh",
                        "base_url": base_url,
                    }
                ]
            },
        })
    )


@pytest.mark.parametrize(
    "provider,base_url,api_mode",
    [
        ("openai-codex", CHATGPT, "codex_responses"),
        ("openai-codex", GATEWAY, "codex_responses"),
        ("openai-codex", CHATGPT, "codex_app_server"),
        ("custom:static", CHATGPT, "codex_responses"),
    ],
)
def test_live_override_tracks_token_and_route_without_session_reset(
    tmp_path, monkeypatch, provider, base_url, api_mode
):
    """Real auth-store resolution must replace a cached key; static overrides retain their fast path."""
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    runner, entry = _runner(tmp_path)
    old_token, new_token = _token("old"), _token("new")
    assert entry.origin is not None
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                channel_overrides={
                    "chat": ChannelOverride(model="channel-model", provider="nous"),
                }
            )
        }
    )
    override = {
        "model": MODEL,
        "provider": provider,
        "requested_provider": None,
        "api_key": old_token,
        "api_mode": "codex_responses",
        "base_url": CHATGPT,
        "credential_pool": object(),
        "request_overrides": {"stale": True},
        "capabilities": {"stale": True},
    }
    runner._session_state(entry.session_key).conversation.model_override = dict(
        override
    )
    runner.session_store.set_model_override(entry.session_key, override)
    persisted_before = (tmp_path / "sessions" / "sessions.json").read_bytes()
    _write_pool(new_token, base_url, api_mode)

    # Any unexpected refresh/network call is a failure; the disk token is already valid.
    with patch(
        "httpx.Client.send", side_effect=AssertionError("unexpected network call")
    ):
        model, runtime = runner._resolve_session_agent_runtime(
            source=entry.origin,
            session_key=entry.session_key,
            user_config={"model": {"default": "default-model"}},
        )
        model2, runtime2 = runner._resolve_session_agent_runtime(
            source=entry.origin,
            session_key=entry.session_key,
            user_config={"model": {"default": "default-model"}},
        )
        # /model --once can restore an old snapshot; it must not resurrect that snapshot's token.
        runner._restore_session_model_override(
            entry.session_key, {"had_override": True, "override": override}
        )
        _, restored_runtime = runner._resolve_session_agent_runtime(
            session_key=entry.session_key
        )

    assert model == model2 == MODEL
    if provider == "openai-codex":
        assert runtime["api_key"] == runtime2["api_key"] == new_token
        assert runtime["base_url"] == runtime2["base_url"] == base_url
        assert runtime["credential_pool"] is not override["credential_pool"]
        assert runtime["api_mode"] == api_mode
        assert not runtime["request_overrides"].get("stale") and not runtime[
            "capabilities"
        ].get("stale")
        retained = runner._session_model_override(entry.session_key)
        assert retained is not None and retained["api_key"] == new_token
        assert restored_runtime["api_key"] == new_token
    else:
        assert runtime["api_key"] == old_token
        assert runtime["credential_pool"] is override["credential_pool"]

    def signature(rt):
        return runner._agent_config_signature(MODEL, rt, [], "frozen prompt")

    assert (signature(override) != signature(runtime)) == (provider == "openai-codex")
    assert signature(runtime) == signature(runtime2)
    assert (
        runner.session_store.get_or_create_session(entry.origin).session_id
        == entry.session_id
    )
    assert (tmp_path / "sessions" / "sessions.json").read_bytes() == persisted_before
    assert (
        old_token.encode() not in persisted_before
        and new_token.encode() not in persisted_before
    )


@pytest.mark.parametrize("default_unavailable", [False, True])
def test_unavailable_codex_override_discards_stale_key_and_retries_own_route(
    tmp_path, default_unavailable
):
    runner, entry = _runner(tmp_path)
    stale = {
        "model": MODEL,
        "provider": "openai-codex",
        "api_key": "synthetic-stale",
        "base_url": CHATGPT,
        "api_mode": "codex_responses",
        "credential_pool": object(),
    }
    runner._session_state(entry.session_key).conversation.model_override = dict(stale)
    default = {
        "provider": "nous",
        "api_key": "synthetic-default",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "api_mode": "chat_completions",
    }
    fresh = {
        "provider": "openai-codex",
        "api_key": "synthetic-fresh",
        "base_url": CHATGPT,
        "api_mode": "codex_responses",
        "credential_pool": object(),
        "capabilities": {"test": True},
    }
    default_result = (
        RuntimeError("default unavailable") if default_unavailable else dict(default)
    )
    with (
        patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            side_effect=[RuntimeError("unavailable"), fresh],
        ) as resolve,
        patch(
            "gateway.run._resolve_runtime_agent_kwargs", side_effect=[default_result]
        ) as global_resolve,
    ):
        config = {"model": {"default": "default-model"}}
        if default_unavailable:
            with pytest.raises(RuntimeError, match="default unavailable"):
                runner._resolve_session_agent_runtime(
                    session_key=entry.session_key, user_config=config
                )
        else:
            model, runtime = runner._resolve_session_agent_runtime(
                session_key=entry.session_key, user_config=config
            )
            assert model == "default-model" and runtime == default
            assert runner._pre_agent_fallback_notice
        retained = runner._session_model_override(entry.session_key)
        assert retained is not None
        assert not retained.get("api_key") and not retained.get("credential_pool")
        assert retained["model"] == MODEL
        model, runtime = runner._resolve_session_agent_runtime(
            session_key=entry.session_key, user_config=config
        )
    assert model == MODEL and runtime["api_key"] == fresh["api_key"]
    assert runtime["capabilities"] == fresh["capabilities"]
    assert runner._pre_agent_fallback_notice is None
    global_resolve.assert_called_once()
    assert resolve.call_count == 2
    for call in resolve.call_args_list:
        assert call.args == ("openai-codex",) and call.kwargs == {"target_model": MODEL}
