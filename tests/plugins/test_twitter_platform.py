import inspect
import asyncio
import base64
import dataclasses
import json
import logging
import os
import socket
import stat
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call
from urllib.parse import parse_qs, urlparse

import pytest
import httpx
import yaml

from gateway.config import PlatformConfig


READY_POLICY = {
    "ai_reply_approval_confirmed": True,
    "automated_label_confirmed": True,
    "human_operator_account_confirmed": True,
    "opt_out_keywords": ["stop"],
}


def ready_twitter_config(**extra):
    return PlatformConfig(
        extra={"client_id": "client", "policy": READY_POLICY, **extra}
    )


def test_registration_disables_generic_message_splitting():
    from plugins.platforms.twitter import register
    from plugins.platforms.twitter.adapter import TwitterAdapter

    ctx = Mock()
    register(ctx)

    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "twitter"
    assert kwargs["label"] == "Twitter / X"
    assert not kwargs.get("allowed_users_env")
    assert not kwargs.get("allow_all_env")
    assert kwargs["cron_deliver_env_var"] == "TWITTER_HOME_CHANNEL"
    assert callable(kwargs["target_parser_fn"])
    assert kwargs["max_message_length"] == 0
    assert kwargs["check_fn"]()
    assert kwargs["install_hint"] == (
        "Run `hermes gateway setup` and choose Twitter / X to install its plugin dependency."
    )
    assert "one concise plain-text post" in kwargs["platform_hint"]
    assert callable(kwargs["standalone_sender_fn"])
    assert TwitterAdapter.SUPPORTS_MESSAGE_EDITING is False
    registered = {call.kwargs["name"]: call.kwargs for call in ctx.register_tool.call_args_list}
    assert set(registered) == {
        "twitter_bookmarks",
        "twitter_post_metrics",
    }
    assert registered["twitter_bookmarks"]["is_async"] is True
    assert registered["twitter_post_metrics"]["is_async"] is True


def test_manifest_keeps_configuration_bridges_internal():
    manifest = yaml.safe_load(
        (Path(__file__).parents[2] / "plugins/platforms/twitter/plugin.yaml").read_text()
    )

    assert [entry["name"] for entry in manifest["optional_env"]] == [
        "TWITTER_CLIENT_SECRET"
    ]
    assert manifest["pip_dependencies"] == ["twitter-text-parser==3.0.0"]


def test_interactive_setup_confidential_saves_masked_profile_secret(
    monkeypatch, tmp_path, caplog, capsys
):
    from hermes_cli import cli_output
    from plugins.platforms.twitter import adapter

    active_profile = tmp_path / "profiles" / "work"
    other_profile = tmp_path / "profiles" / "personal"
    active_profile.mkdir(parents=True)
    other_profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active_profile))
    monkeypatch.delenv("TWITTER_CLIENT_SECRET", raising=False)
    answers = {
        "OAuth 2.0 client ID": "confidential-client",
        "OAuth client type (public or confidential)": "confidential",
        "Loopback redirect URI": "http://127.0.0.1:8765/callback",
        "OAuth 2.0 client secret": "profile-only-secret",
        "Allowed numeric X user IDs (comma-separated)": "42",
    }
    prompted = []
    authorized = {}

    def fake_prompt(question, default=None, password=False):
        prompted.append((question, password))
        return answers[question]

    async def fake_authorize(client_id, redirect_uri, **kwargs):
        authorized.update(client_id=client_id, redirect_uri=redirect_uri, **kwargs)

    monkeypatch.setattr(cli_output, "prompt", fake_prompt)
    monkeypatch.setattr(cli_output, "prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr(adapter, "authorize", fake_authorize)

    adapter.interactive_setup()

    env_path = active_profile / ".env"
    assert ("OAuth 2.0 client secret", True) in prompted
    assert env_path.read_text() == "TWITTER_CLIENT_SECRET=profile-only-secret\n"
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert not (other_profile / ".env").exists()
    assert authorized["client_secret"] == "profile-only-secret"
    saved_config = (active_profile / "config.yaml").read_text()
    assert "profile-only-secret" not in saved_config
    assert "profile-only-secret" not in caplog.text
    captured = capsys.readouterr()
    assert "profile-only-secret" not in captured.out + captured.err


def test_interactive_setup_public_does_not_prompt_or_store_secret(
    monkeypatch, tmp_path
):
    from hermes_cli import cli_output
    from plugins.platforms.twitter import adapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TWITTER_CLIENT_SECRET", raising=False)
    answers = {
        "OAuth 2.0 client ID": "public-client",
        "OAuth client type (public or confidential)": "public",
        "Loopback redirect URI": "http://127.0.0.1:8765/callback",
        "Allowed numeric X user IDs (comma-separated)": "42",
    }
    prompted = []
    authorized = {}

    def fake_prompt(question, default=None, password=False):
        prompted.append((question, password))
        return answers[question]

    async def fake_authorize(client_id, redirect_uri, **kwargs):
        authorized.update(client_id=client_id, redirect_uri=redirect_uri, **kwargs)

    monkeypatch.setattr(cli_output, "prompt", fake_prompt)
    monkeypatch.setattr(cli_output, "prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr(adapter, "authorize", fake_authorize)

    adapter.interactive_setup()

    assert all("secret" not in question.lower() for question, _ in prompted)
    assert authorized["client_secret"] == ""
    assert not (tmp_path / ".env").exists()
    assert "TWITTER_CLIENT_SECRET" not in (tmp_path / "config.yaml").read_text()


def test_interactive_setup_records_open_access_and_policy_confirmations(
    monkeypatch, tmp_path
):
    from hermes_cli import cli_output
    from plugins.platforms.twitter import adapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    answers = {
        "OAuth 2.0 client ID": "public-client",
        "OAuth client type (public or confidential)": "public",
        "Loopback redirect URI": "http://127.0.0.1:8765/callback",
        "Allowed numeric X user IDs (comma-separated)": "42",
    }
    prompted = []
    confirmations = iter([True, True, False, True])

    def fake_prompt(question, default=None, password=False):
        prompted.append(question)
        return answers[question]

    async def fake_authorize(*args, **kwargs):
        return None

    monkeypatch.setattr(cli_output, "prompt", fake_prompt)
    monkeypatch.setattr(
        cli_output, "prompt_yes_no", lambda *args, **kwargs: next(confirmations)
    )
    monkeypatch.setattr(adapter, "authorize", fake_authorize)

    adapter.interactive_setup()

    twitter = yaml.safe_load((tmp_path / "config.yaml").read_text())["twitter"]
    assert twitter["allow_all_users"] is True
    assert twitter["allowed_users"] == []
    assert "Allowed numeric X user IDs (comma-separated)" not in prompted
    assert twitter["policy"] == {
        "ai_reply_approval_confirmed": True,
        "automated_label_confirmed": False,
        "human_operator_account_confirmed": True,
    }


def test_weighted_parser_requires_parse_tweet_api(monkeypatch):
    from plugins.platforms.twitter import presentation

    monkeypatch.setattr(
        presentation,
        "import_module",
        lambda _name: SimpleNamespace(),
        raising=False,
    )

    assert not presentation.weighted_parser_available()


def test_format_message_preserves_link_meaning():
    from plugins.platforms.twitter.presentation import format_message

    assert format_message("See [the design](https://example.com/x).") == (
        "See the design (https://example.com/x)."
    )


def test_format_message_strips_markdown_and_normalizes_nfc():
    from plugins.platforms.twitter.presentation import format_message

    assert format_message("**Cafe\u0301** and `code`") == "Café and code"


def test_format_message_rejects_disallowed_controls():
    from plugins.platforms.twitter.presentation import format_message

    with pytest.raises(ValueError, match="control"):
        format_message("not\x00safe")


@pytest.mark.parametrize(
    "text",
    [
        "a" * 280,
        "a" * 256 + " " + "https://example.com/" + "x" * 100,
        "a" * 278 + "👨‍👩‍👧‍👦",
        "界" * 140,
        ("e\u0301") * 280,
        "@alice " + "a" * 273,
    ],
    ids=["ascii", "url", "emoji", "cjk", "combining", "reply-mention"],
)
def test_format_message_accepts_exact_x_weighted_limit(text):
    from plugins.platforms.twitter.presentation import format_message

    assert format_message(text)


@pytest.mark.parametrize(
    "text",
    [
        "a" * 281,
        "a" * 257 + " " + "https://example.com/" + "x" * 100,
        "a" * 279 + "👨‍👩‍👧‍👦",
        "界" * 141,
        ("e\u0301") * 281,
        "@alice " + "a" * 274,
    ],
    ids=["ascii", "url", "emoji", "cjk", "combining", "reply-mention"],
)
def test_format_message_rejects_over_x_weighted_limit(text):
    from plugins.platforms.twitter.presentation import format_message

    with pytest.raises(ValueError, match="weighted limit"):
        format_message(text)


def test_format_thread_messages_uses_x_weighted_boundaries():
    from twitter_text import parse_tweet
    from plugins.platforms.twitter.presentation import format_thread_messages

    url = "https://example.com/" + "x" * 400
    family = "👨‍👩‍👧‍👦"
    parts = format_thread_messages(
        f"{'word ' * 60}{family * 30} {url} {'界' * 100}"
    )

    assert len(parts) > 1
    assert all(parse_tweet(part).valid for part in parts)
    assert sum(url in part for part in parts) == 1
    assert all(not part.startswith("\u200d") and not part.endswith("\u200d") for part in parts)


def test_x_weighted_length_uses_x_url_weighting():
    from plugins.platforms.twitter.presentation import x_weighted_length

    assert x_weighted_length("https://example.com/" + "x" * 400) == 23


def test_bot_limit_splits_into_at_most_ten_weighted_parts():
    from plugins.platforms.twitter.presentation import (
        X_BOT_WEIGHTED_LIMIT,
        X_MAX_FALLBACK_PARTS,
        format_thread_messages,
        x_weighted_length,
    )

    parts = format_thread_messages("x" * X_BOT_WEIGHTED_LIMIT)

    assert len(parts) == X_MAX_FALLBACK_PARTS
    assert all(x_weighted_length(part) <= 280 for part in parts)


def test_settings_reject_unsafe_limits():
    from plugins.platforms.twitter.adapter import TwitterSettings

    with pytest.raises(ValueError, match="poll_interval_seconds"):
        TwitterSettings.from_config(
            PlatformConfig(
                extra={"client_id": "client", "poll_interval_seconds": 0}
            )
        )


@pytest.fixture
def twitter_settings(monkeypatch):
    from plugins.platforms.twitter.adapter import TwitterSettings

    monkeypatch.setenv("TWITTER_CLIENT_SECRET", "secret")
    return TwitterSettings.from_config(
        PlatformConfig(
            extra={
                "client_id": "client",
                "oauth_client_type": "confidential",
                "policy": {
                    "ai_reply_approval_confirmed": True,
                    "automated_label_confirmed": True,
                    "human_operator_account_confirmed": True,
                    "opt_out_keywords": ["stop"],
                },
            }
        )
    )


@pytest.mark.parametrize(
    "field",
    [
        "ai_reply_approval_confirmed",
        "automated_label_confirmed",
        "human_operator_account_confirmed",
    ],
)
def test_reply_policy_fails_closed_when_one_confirmation_is_missing(
    twitter_settings, field
):
    from plugins.platforms.twitter.adapter import TwitterPolicyError

    policy = dataclasses.replace(twitter_settings.policy, **{field: False})
    with pytest.raises(TwitterPolicyError, match=field):
        policy.require_automated_reply_ready()


def test_reply_policy_requires_opt_out_keywords(twitter_settings):
    from plugins.platforms.twitter.adapter import TwitterPolicyError

    with pytest.raises(TwitterPolicyError, match="opt_out_keywords"):
        dataclasses.replace(
            twitter_settings.policy, opt_out_keywords=()
        ).require_automated_reply_ready()


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        (
            {"policy": {"ai_reply_approval_confirmed": "false"}},
            "ai_reply_approval_confirmed",
        ),
        ({"initial_backfill": "1"}, "initial_backfill"),
        ({"initial_backfill": True}, "initial_backfill"),
        ({"allow_all_users": "true"}, "allow_all_users"),
        ({"redirect_uri": "https://example.com/callback"}, "redirect_uri"),
        ({"redirect_uri": "http://127.0.0.1:0/callback"}, "redirect_uri"),
        ({"oauth_client_type": "service"}, "oauth_client_type"),
        ({"queue": {"max_pending_per_bucket": 0}}, "max_pending_per_bucket"),
        ({"queue": {"max_pending_per_bucket": 1.0}}, "max_pending_per_bucket"),
        ({"queue": []}, "queue"),
        ({"poll_interval_seconds": float("inf")}, "poll_interval_seconds"),
        ({"initial_backfill": 101}, "initial_backfill"),
    ],
)
def test_settings_rejects_invalid_typed_values(extra, error):
    from plugins.platforms.twitter.adapter import TwitterSettings

    with pytest.raises(ValueError, match=error):
        TwitterSettings.from_config(
            PlatformConfig(extra={"client_id": "client", **extra})
    )


def test_settings_uses_canonical_queue_cap_without_obsolete_alias():
    from plugins.platforms.twitter.adapter import TwitterSettings

    settings = TwitterSettings.from_config(
        PlatformConfig(
            extra={
                "client_id": "client",
                "queue": {"max_pending_per_bucket": 3, "max_pending": 1},
            }
        )
    )

    assert settings.max_pending == 3


def test_confidential_client_requires_profile_secret(monkeypatch):
    from plugins.platforms.twitter.adapter import TwitterSettings

    monkeypatch.delenv("TWITTER_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="TWITTER_CLIENT_SECRET"):
        TwitterSettings.from_config(
            PlatformConfig(
                extra={"client_id": "client", "oauth_client_type": "confidential"}
            )
        )


def test_public_client_does_not_require_profile_secret(monkeypatch):
    from plugins.platforms.twitter.adapter import TwitterSettings

    monkeypatch.delenv("TWITTER_CLIENT_SECRET", raising=False)
    settings = TwitterSettings.from_config(
        PlatformConfig(extra={"client_id": "client", "oauth_client_type": "public"})
    )
    assert settings.oauth_client_type == "public"


def test_apply_yaml_config_uses_nested_platform_block(monkeypatch):
    from plugins.platforms.twitter.adapter import apply_yaml_config

    monkeypatch.delenv("TWITTER_ALLOWED_USERS", raising=False)
    nested = {
        "client_id": "nested-client",
        "allowed_users": ["42"],
        "home_channel": "timeline",
    }
    assert apply_yaml_config({}, nested)["client_id"] == "nested-client"
    assert "TWITTER_ALLOWED_USERS" not in os.environ


@pytest.mark.asyncio
async def test_simultaneous_profiles_isolate_secret_allowlist_state_and_requests(
    monkeypatch, tmp_path
):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import SCOPES, save_tokens
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("TWITTER_CLIENT_SECRET", "wrong-global-secret")
    secret_scope.set_multiplex_active(True)
    requests: dict[str, list[tuple[str, str, dict]]] = {"a": [], "b": []}

    async def run_profile(
        name: str,
        *,
        secret: str,
        allowed_user: str,
        denied_user: str,
        conversation_id: str,
        anchor_id: str,
        interaction_id: str,
    ) -> None:
        home = tmp_path / name
        home_token = set_hermes_home_override(home)
        secret_token = secret_scope.set_secret_scope(
            {"TWITTER_CLIENT_SECRET": secret}
        )
        try:
            def handler(request):
                payload = (
                    json.loads(request.content or b"{}")
                    if request.url.path == "/2/tweets"
                    else {}
                )
                requests[name].append(
                    (request.method, request.url.path, payload)
                )
                if request.url.path == "/2/oauth2/token":
                    expected = base64.b64encode(
                        f"client-{name}:{secret}".encode()
                    ).decode()
                    assert request.headers["Authorization"] == f"Basic {expected}"
                    return httpx.Response(
                        200,
                        json={
                            "access_token": f"access-{name}",
                            "refresh_token": f"refresh-{name}",
                            "expires_in": 3600,
                            "scope": " ".join(SCOPES),
                        },
                    )
                assert request.url.path == "/2/tweets"
                assert payload["reply"]["in_reply_to_tweet_id"] == interaction_id
                return httpx.Response(
                    201, json={"data": {"id": "902" if name == "b" else "901"}}
                )

            config = ready_twitter_config(
                client_id=f"client-{name}",
                oauth_client_type="confidential",
                allowed_users=[allowed_user],
                allow_all_users=False,
                _http_transport=httpx.MockTransport(handler),
            )
            adapter = adapter_module.TwitterAdapter(config)
            assert adapter.settings.client_secret == secret
            assert adapter._authorized(allowed_user)
            assert not adapter._authorized(denied_user)

            save_tokens(
                {
                    "access_token": f"expired-{name}",
                    "refresh_token": f"refresh-{name}",
                    "expires_at": 1,
                    "scopes": list(SCOPES),
                    "client_id": f"client-{name}",
                    "client_type": "confidential",
                    "user_id": "8" if name == "b" else "7",
                }
            )
            route = f"tweet:{conversation_id}:{anchor_id}"
            state = TwitterState.load(max_seen=2)
            state.record_public_interaction(interaction_id, route)
            state.save()

            result = await adapter_module.standalone_send(
                config,
                route,
                f"reply-{name}",
                thread_id=interaction_id,
            )

            assert result["success"] is True
            restored = TwitterState.load(max_seen=2)
            assert [f"tweet:{interaction_id}", "replied"] in restored.to_dict()[
                "reply_reservations"
            ]
            assert restored.to_dict()["public_reply_routes"] == [
                [interaction_id, route]
            ]
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    try:
        await asyncio.gather(
            run_profile(
                "a",
                secret="secret-a",
                allowed_user="41",
                denied_user="42",
                conversation_id="100",
                anchor_id="101",
                interaction_id="102",
            ),
            run_profile(
                "b",
                secret="secret-b",
                allowed_user="42",
                denied_user="41",
                conversation_id="200",
                anchor_id="201",
                interaction_id="202",
            ),
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert [path for _, path, _ in requests["a"]] == [
        "/2/oauth2/token",
        "/2/tweets",
    ]
    assert [path for _, path, _ in requests["b"]] == [
        "/2/oauth2/token",
        "/2/tweets",
    ]


@pytest.mark.parametrize("value", ["true", "false", "1", 1])
def test_apply_yaml_config_rejects_non_boolean_allow_all_users(monkeypatch, value):
    from plugins.platforms.twitter.adapter import apply_yaml_config

    monkeypatch.delenv("TWITTER_ALLOW_ALL_USERS", raising=False)
    with pytest.raises(ValueError, match="allow_all_users"):
        apply_yaml_config({}, {"allow_all_users": value})


def test_adapter_send_signature_matches_base():
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.twitter.adapter import TwitterAdapter

    assert inspect.signature(TwitterAdapter.send) == inspect.signature(
        BasePlatformAdapter.send
    )


@pytest.mark.asyncio
async def test_connect_prebinds_stored_account_before_initial_identity(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens(
        {
            "access_token": "access",
            "client_id": "client",
            "user_id": "7",
            "username": "bot",
        }
    )
    adapter = adapter_module.TwitterAdapter(
        PlatformConfig(extra={"client_id": "client"})
    )

    class Client:
        async def identity(self):
            assert adapter._account_id == "7"
            return {"data": {"id": "7", "username": "bot"}}

        async def close(self):
            return None

    monkeypatch.setattr(adapter_module, "XClient", Mock(return_value=Client()))
    monkeypatch.setattr(adapter, "_acquire_platform_lock", Mock(return_value=True))
    monkeypatch.setattr(adapter, "_release_platform_lock", Mock())
    monkeypatch.setattr(adapter, "_poll_mentions_once", AsyncMock())
    monkeypatch.setattr(adapter, "_poll_dms_once", AsyncMock())
    monkeypatch.setattr(adapter, "_start_poller", Mock())

    assert await adapter.connect()
    assert adapter._account_id == "7"


def test_oauth_s256_challenge_matches_rfc_vector():
    from plugins.platforms.twitter.oauth import create_s256_challenge

    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert (
        create_s256_challenge(verifier)
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


def test_oauth_tokens_follow_active_hermes_home(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import load_tokens, save_tokens, token_path

    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("HERMES_HOME", str(first))
    save_tokens({"access_token": "one", "refresh_token": "r1"})
    assert token_path() == first / "twitter" / "oauth2.json"

    monkeypatch.setenv("HERMES_HOME", str(second))
    assert load_tokens() is None
    save_tokens({"access_token": "two", "refresh_token": "r2"})
    stored = second / "twitter" / "oauth2.json"
    assert json.loads(stored.read_text())["access_token"] == "two"
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_oauth_public_exchange_binds_account_and_persists(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter import oauth as oauth_module
    from plugins.platforms.twitter.oauth import SCOPES, authorize, load_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async def callback(redirect_uri, state, *, timeout, on_ready):
        assert redirect_uri == "http://127.0.0.1:8765/exact-callback"
        assert state
        assert timeout == 1
        on_ready()
        return "authorization-code"

    monkeypatch.setattr(oauth_module, "wait_for_callback", callback)
    opened = []

    def handler(request):
        if request.url.path == "/2/oauth2/token":
            form = parse_qs(request.content.decode())
            assert "Authorization" not in request.headers
            assert form == {
                "code": ["authorization-code"],
                "grant_type": ["authorization_code"],
                "client_id": ["client"],
                "redirect_uri": ["http://127.0.0.1:8765/exact-callback"],
                "code_verifier": ["verifier"],
            }
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_in": 3600,
                    "scope": " ".join(SCOPES),
                },
            )
        assert request.url.path == "/2/users/me"
        assert request.headers["Authorization"] == "Bearer access-token"
        return httpx.Response(
            200, json={"data": {"id": "42", "username": "alice"}}
        )

    monkeypatch.setattr(
        oauth_module, "create_pkce_pair", lambda: ("verifier", "challenge")
    )
    tokens = await authorize(
        "client",
        "http://127.0.0.1:8765/exact-callback",
        client_type="public",
        timeout=1,
        open_url=opened.append,
        transport=httpx.MockTransport(handler),
    )

    query = parse_qs(urlparse(opened[0]).query)
    assert query["redirect_uri"] == ["http://127.0.0.1:8765/exact-callback"]
    assert tokens.client_id == "client"
    assert tokens.client_type == "public"
    assert tokens.user_id == "42"
    assert tokens.username == "alice"
    assert load_tokens() == tokens
    assert stat.S_IMODE((tmp_path / "twitter" / "oauth2.json").stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_oauth_confidential_exchange_uses_basic_auth(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import OAuthClient, SCOPES

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def handler(request):
        form = parse_qs(request.content.decode())
        assert request.headers["Authorization"] == "Basic Y2xpZW50OnNlY3JldA=="
        assert "client_id" not in form
        assert "client_secret" not in form
        assert form["code_verifier"] == ["verifier"]
        return httpx.Response(
            200,
            json={
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": " ".join(SCOPES),
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    oauth = OAuthClient(
        "client",
        "http://127.0.0.1:8765/callback",
        client_type="confidential",
        client_secret="secret",
        client=http,
    )
    tokens = await oauth.exchange_code("code", "verifier")

    assert tokens.client_type == "confidential"
    await http.aclose()


@pytest.mark.asyncio
async def test_oauth_exchange_rejects_missing_scopes(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import OAuthClient

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "scope": "tweet.read users.read",
                },
            )
        )
    )
    oauth = OAuthClient(
        "client", "http://127.0.0.1:8765/callback", client=http
    )

    with pytest.raises(RuntimeError, match="required scopes"):
        await oauth.exchange_code("code", "verifier")
    await http.aclose()


@pytest.mark.asyncio
async def test_oauth_loopback_callback_accepts_exact_path_and_state():
    from plugins.platforms.twitter.oauth import wait_for_callback

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = asyncio.Event()
    waiter = asyncio.create_task(
        wait_for_callback(
            f"http://127.0.0.1:{port}/exact-callback",
            "expected",
            timeout=1,
            on_ready=ready.set,
        )
    )
    await ready.wait()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        b"GET /exact-callback?code=code&state=expected HTTP/1.1\r\n"
        b"Host: localhost\r\n\r\n"
    )
    await writer.drain()
    await reader.read()
    writer.close()
    await writer.wait_closed()

    assert await waiter == "code"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("/callback?code=secret&state=wrong", "state"),
        ("/wrong?code=secret&state=expected", "path"),
        ("/callback?error=access_denied&state=expected", "denied"),
    ],
)
async def test_oauth_loopback_callback_rejects_invalid_response(target, error):
    from plugins.platforms.twitter.oauth import wait_for_callback

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = asyncio.Event()
    waiter = asyncio.create_task(
        wait_for_callback(
            f"http://127.0.0.1:{port}/callback",
            "expected",
            timeout=1,
            on_ready=ready.set,
        )
    )
    await ready.wait()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
    )
    await writer.drain()
    await reader.read()
    writer.close()
    await writer.wait_closed()

    with pytest.raises((ValueError, RuntimeError), match=error):
        await waiter


@pytest.mark.asyncio
async def test_oauth_loopback_callback_has_bounded_timeout():
    from plugins.platforms.twitter.oauth import wait_for_callback

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(TimeoutError):
        await wait_for_callback(
            f"http://127.0.0.1:{port}/callback", "expected", timeout=0.01
        )


@pytest.mark.asyncio
async def test_oauth_malformed_token_is_preserved_with_reauthorization_guidance(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "twitter" / "oauth2.json"
    path.parent.mkdir(parents=True)
    path.write_text("{malformed")
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))

    assert not await adapter.connect()
    assert path.read_text() == "{malformed"
    assert "re-authorize" in adapter.fatal_error_message.lower()


@pytest.mark.asyncio
async def test_oauth_failures_do_not_log_secrets(monkeypatch, tmp_path, caplog):
    from plugins.platforms.twitter.oauth import OAuthClient, SCOPES

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secrets = ("secret", "code-value", "verifier-value", "access-token", "refresh-token")

    def handler(request):
        return httpx.Response(
            400,
            json={"error": " ".join(secrets), "scope": " ".join(SCOPES)},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    oauth = OAuthClient(
        "client",
        "http://127.0.0.1:8765/callback",
        client_type="confidential",
        client_secret="secret",
        client=http,
    )
    with pytest.raises(httpx.HTTPStatusError):
        await oauth.exchange_code("code-value", "verifier-value")

    assert all(value not in caplog.text for value in secrets)
    await http.aclose()


def test_branch_anchor_follows_mapped_bot_ancestor(monkeypatch, tmp_path):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=100, max_branches=100)
    state.map_bot_post("9007199254740993", "123")
    assert state.resolve_anchor("456", ["42", "9007199254740993"]) == "123"
    assert state.resolve_anchor("789", ["42"]) == "789"


def test_state_survives_restart_and_profile_switch(monkeypatch, tmp_path):
    from plugins.platforms.twitter.state import TwitterState

    first = tmp_path / "a"
    second = tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(first))
    state = TwitterState.load()
    state.mark_seen("999")
    state.advance_mentions("999")
    state.save()
    assert TwitterState.load().seen("999")

    monkeypatch.setenv("HERMES_HOME", str(second))
    assert not TwitterState.load().seen("999")


def test_public_reply_state_evicts_oldest_terminal_pair_at_capacity(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=2)
    for interaction_id in ("1", "2"):
        route = f"tweet:100:{interaction_id}"
        state.record_public_interaction(interaction_id, route)
        assert state.reserve_public_reply(interaction_id, route)
        state.confirm_public_reply(interaction_id, f"70{interaction_id}", interaction_id)

    state.record_public_interaction("3", "tweet:100:3")
    assert state.reserve_public_reply("3", "tweet:100:3")
    state.confirm_public_reply("3", "703", "3")

    payload = state.to_dict()
    assert payload["public_reply_routes"] == [
        ["2", "tweet:100:2"],
        ["3", "tweet:100:3"],
    ]
    assert payload["reply_reservations"] == [
        ["tweet:2", "replied"],
        ["tweet:3", "replied"],
    ]
    assert not state.reserve_public_reply("2", "tweet:100:2")
    state.save()
    state = TwitterState.load(max_seen=2)
    assert not state.reserve_public_reply("2", "tweet:100:2")


def test_dm_reply_state_evicts_oldest_terminal_pair_at_capacity(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=2)
    for interaction_id, conversation_id in (("1", "10-7"), ("2", "20-7")):
        state.record_dm_inbound(conversation_id, interaction_id)
        assert state.reserve_dm_reply(interaction_id, conversation_id)
        assert state.begin_dm_reply(interaction_id, conversation_id)
        state.confirm_dm_reply(interaction_id)

    state.record_dm_inbound("30-7", "3")
    assert state.reserve_dm_reply("3", "30-7")
    assert state.begin_dm_reply("3", "30-7")
    state.confirm_dm_reply("3")

    payload = state.to_dict()
    assert payload["dm_reply_routes"] == [["2", "20-7"], ["3", "30-7"]]
    assert payload["reply_reservations"] == [
        ["dm:2", "replied"],
        ["dm:3", "replied"],
    ]
    assert not state.reserve_dm_reply("2", "20-7")


def test_public_reply_state_evicts_oldest_unreserved_route_at_capacity(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=2)
    state.record_public_interaction("1", "tweet:100:1")
    state.record_public_interaction("2", "tweet:100:2")

    state.record_public_interaction("3", "tweet:100:3")

    assert state.to_dict()["public_reply_routes"] == [
        ["2", "tweet:100:2"],
        ["3", "tweet:100:3"],
    ]
    assert state.reserve_public_reply("3", "tweet:100:3")


def test_dm_reply_state_evicts_oldest_unreserved_route_at_capacity(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=2)
    state.record_dm_inbound("10-7", "1")
    state.record_dm_inbound("20-7", "2")

    state.record_dm_inbound("30-7", "3")

    assert state.to_dict()["dm_reply_routes"] == [
        ["2", "20-7"],
        ["3", "30-7"],
    ]
    assert state.reserve_dm_reply("3", "30-7")


@pytest.mark.parametrize("kind", ["public", "dm"])
def test_reply_state_never_evicts_active_or_uncertain_pairs(
    monkeypatch, tmp_path, kind
):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / kind))
    state = TwitterState.load(max_seen=2)
    if kind == "public":
        for interaction_id in ("1", "2"):
            route = f"tweet:100:{interaction_id}"
            state.record_public_interaction(interaction_id, route)
            assert state.reserve_public_reply(interaction_id, route)
        state.mark_public_reply_uncertain("2")
        state.record_public_interaction("3", "tweet:100:3")
        assert not state.reserve_public_reply("3", "tweet:100:3")
        assert state.to_dict()["public_reply_routes"] == [
            ["1", "tweet:100:1"],
            ["2", "tweet:100:2"],
        ]
    else:
        for interaction_id, conversation_id in (("1", "10-7"), ("2", "20-7")):
            state.record_dm_inbound(conversation_id, interaction_id)
            assert state.reserve_dm_reply(interaction_id, conversation_id)
            assert state.begin_dm_reply(interaction_id, conversation_id)
        state.mark_dm_reply_uncertain("2")
        state.record_dm_inbound("30-7", "3")
        assert not state.reserve_dm_reply("3", "30-7")
        assert state.to_dict()["dm_reply_routes"] == [
            ["1", "10-7"],
            ["2", "20-7"],
        ]

    assert [status for _, status in state.to_dict()["reply_reservations"]] == [
        "reserved" if kind == "public" else "writing",
        "uncertain",
    ]


@pytest.mark.asyncio
async def test_twitter_state_lock_preserves_concurrent_gateway_and_cron_updates(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.state import TwitterState, twitter_state_lock

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile_key = str(tmp_path.resolve())
    first_loaded = asyncio.Event()
    allow_first_save = asyncio.Event()
    first_saved = asyncio.Event()
    second_started = asyncio.Event()

    async def gateway_update():
        async with twitter_state_lock(profile_key, "7"):
            state = TwitterState.load()
            first_loaded.set()
            await allow_first_save.wait()
            state.advance_mentions("9007199254740995")
            state.save()
            first_saved.set()

    async def cron_update():
        await first_loaded.wait()
        second_started.set()
        async with twitter_state_lock(profile_key, "7"):
            state = TwitterState.load()
            await first_saved.wait()
            state.known_dm_conversations.add("conversation-2")
            state.save()

    gateway = asyncio.create_task(gateway_update())
    cron = asyncio.create_task(cron_update())
    await second_started.wait()
    allow_first_save.set()
    await asyncio.gather(gateway, cron)

    state = TwitterState.load()
    assert state.mention_since_id == "9007199254740995"
    assert state.known_dm_conversations == {"conversation-2"}


def test_twitter_locks_serialize_non_gateway_processes(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import SCOPES, load_tokens, save_tokens
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens(
        {
            "access_token": "expired",
            "refresh_token": "rotate-once",
            "expires_at": 1,
            "scopes": list(SCOPES),
            "client_id": "client",
        }
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""
            import asyncio
            import os
            import sys
            from pathlib import Path

            sys.path.insert(0, {str(Path(__file__).parents[2])!r})
            os.environ["HERMES_HOME"] = {str(tmp_path)!r}

            from plugins.platforms.twitter.oauth import (
                OAuthClient,
                SCOPES,
                active_profile_key,
                load_tokens,
            )
            from plugins.platforms.twitter.state import TwitterState, twitter_state_lock

            root = Path({str(tmp_path)!r})
            role = sys.argv[1]

            class Response:
                def json(self):
                    return {{
                        "access_token": "fresh",
                        "refresh_token": "rotated-refresh-token",
                        "expires_in": 3600,
                        "scope": " ".join(SCOPES),
                    }}

            async def fake_post(self, data):
                with (root / "refresh-calls").open("a") as calls:
                    calls.write("refresh\\n")
                    calls.flush()
                while not (root / "release-refresh").exists():
                    await asyncio.sleep(0.01)
                return Response()

            OAuthClient._post_token = fake_post

            async def main():
                while not (root / "start").exists():
                    await asyncio.sleep(0.01)
                oauth = OAuthClient("client", "http://127.0.0.1:8765/callback")
                try:
                    await oauth.refresh(load_tokens())
                finally:
                    await oauth.close()

                if role == "gateway":
                    async with twitter_state_lock(active_profile_key(), "client"):
                        state = TwitterState.load()
                        (root / "state-ready").touch()
                        while not (root / "release-state").exists():
                            await asyncio.sleep(0.01)
                        state.advance_mentions("9007199254740995")
                        state.save()
                else:
                    while not (root / "state-ready").exists():
                        await asyncio.sleep(0.01)
                    (root / "contender-started").touch()
                    async with twitter_state_lock(active_profile_key(), "7"):
                        state = TwitterState.load()
                        (root / "contender-acquired").touch()
                        while not (root / "release-contender").exists():
                            await asyncio.sleep(0.01)
                        state.known_dm_conversations.add("conversation-2")
                        state.save()
                    (root / "contender-done").touch()

            asyncio.run(main())
            """
        )
    )

    processes = [
        subprocess.Popen(
            [sys.executable, str(worker), role],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for role in ("gateway", "cron")
    ]

    def wait_for(path: str, timeout: float = 10) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (tmp_path / path).exists():
                return True
            time.sleep(0.01)
        return False

    try:
        (tmp_path / "start").touch()
        assert wait_for("refresh-calls")
        time.sleep(0.3)
        (tmp_path / "release-refresh").touch()
        assert wait_for("state-ready")
        assert wait_for("contender-started")

        acquired_before_release = wait_for("contender-acquired", timeout=0.2)
        if acquired_before_release:
            (tmp_path / "release-contender").touch()
            assert wait_for("contender-done")
            (tmp_path / "release-state").touch()
        else:
            (tmp_path / "release-state").touch()
            assert wait_for("contender-acquired")
            (tmp_path / "release-contender").touch()

        failures = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            if process.returncode:
                failures.append(f"stdout={stdout!r} stderr={stderr!r}")
        assert not failures, failures
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert (tmp_path / "refresh-calls").read_text().splitlines() == ["refresh"]
    assert load_tokens().refresh_token == "rotated-refresh-token"
    state = TwitterState.load()
    assert state.mention_since_id == "9007199254740995"
    assert state.known_dm_conversations == {"conversation-2"}


def test_corrupt_twitter_state_is_quarantined(monkeypatch, tmp_path):
    from plugins.platforms.twitter.state import TwitterState, state_path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = state_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not-json")

    assert TwitterState.load().mention_since_id == ""
    assert not path.exists()
    assert list(path.parent.glob("state.json.corrupt-*"))


def test_corrupt_twitter_tokens_are_preserved(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import load_tokens, token_path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = token_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not-json")

    assert load_tokens() is None
    assert path.read_text() == "{not-json"


@pytest.mark.asyncio
async def test_write_timeout_is_not_retried():
    from plugins.platforms.twitter.client import AmbiguousWriteError, XClient

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("uncertain", request=request)

    client = XClient(
        token="token",
        transport=httpx.MockTransport(handler),
        max_pending=2,
        max_wait_seconds=1,
    )
    with pytest.raises(AmbiguousWriteError) as error:
        await client.create_post("hello", reply_to="123")
    assert "/2/tweets" in str(error.value)
    assert calls == 1
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "http", "malformed"])
async def test_ambiguous_dm_write_normalizes_dynamic_endpoint(failure):
    from plugins.platforms.twitter.client import AmbiguousWriteError, XClient

    conversation_id = "987654321-123456789"

    def handler(request):
        if failure == "transport":
            raise httpx.ConnectError("offline", request=request)
        if failure == "http":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(201, text="not-json")

    client = XClient(token="token", transport=httpx.MockTransport(handler))
    with pytest.raises(AmbiguousWriteError) as error:
        await client.send_dm(conversation_id, "private")

    message = str(error.value)
    assert conversation_id not in message
    assert "/2/dm_conversations/:id/messages" in message
    await client.close()


@pytest.mark.asyncio
async def test_ambiguous_dm_send_result_normalizes_dynamic_endpoint(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XClient

    conversation_id = "987654321-123456789"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = XClient(
        token="token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, text="unavailable")
        ),
    )
    await adapter._mutate_state(
        lambda state: state.record_dm_inbound(conversation_id, "501")
    )

    result = await adapter.send(
        f"dm:{conversation_id}", "private", reply_to="501"
    )

    assert not result.success
    assert conversation_id not in (result.error or "")
    assert "/2/dm_conversations/:id/messages" in (result.error or "")
    await adapter._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"id": "１２３"}},
        {"data": []},
        ["unexpected"],
    ],
    ids=["unicode-id", "malformed-data", "non-object"],
)
async def test_success_without_returned_post_id_is_ambiguous(payload):
    from plugins.platforms.twitter.client import AmbiguousWriteError, XClient

    client = XClient(
        token="token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(201, json=payload)
        ),
    )
    with pytest.raises(AmbiguousWriteError):
        await client.create_post("hello", reply_to="123")
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        ["bad request"],
        {"errors": "bad request"},
        {"errors": [None]},
    ],
    ids=["non-object", "non-list-errors", "non-object-error"],
)
async def test_malformed_definitive_error_is_x_api_error(payload):
    from plugins.platforms.twitter.client import XApiError, XClient

    client = XClient(
        token="token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(400, json=payload)
        ),
    )
    with pytest.raises(XApiError) as error:
        await client.create_post("hello", reply_to="123")
    assert error.value.status == 400
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("body_kind", ["json", "text"])
async def test_x_api_error_omits_untrusted_response_body(body_kind):
    from plugins.platforms.twitter.client import XApiError, XClient

    sentinel = "PRIVATE_POST_OR_DM_SENTINEL"

    def handler(_request):
        if body_kind == "json":
            return httpx.Response(
                403,
                json={
                    "errors": [
                        {
                            "code": 349,
                            "title": "Forbidden",
                            "detail": sentinel,
                        }
                    ]
                },
            )
        return httpx.Response(403, text=sentinel)

    client = XClient(token="token", transport=httpx.MockTransport(handler))
    with pytest.raises(XApiError) as error:
        await client.request("GET", "/2/dm_conversations/42-7/messages")

    message = str(error.value)
    assert sentinel not in message
    assert "42-7" not in message
    assert "X API 403 on /2/dm_conversations/:id/messages" in message
    if body_kind == "json":
        assert "code=349" in message
        assert "title=Forbidden" in message
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("body_kind", ["json", "text"])
async def test_send_result_omits_untrusted_api_body(monkeypatch, tmp_path, body_kind):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XClient

    sentinel = "PRIVATE_POST_OR_DM_SENTINEL"

    def handler(_request):
        if body_kind == "json":
            return httpx.Response(400, json={"detail": sentinel})
        return httpx.Response(400, text=sentinel)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = XClient(token="token", transport=httpx.MockTransport(handler))

    result = await adapter.send("timeline", "safe post")

    assert not result.success
    assert sentinel not in (result.error or "")
    assert "X API 400 on /2/tweets" in (result.error or "")
    await adapter._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("body_kind", ["json", "text"])
async def test_enrichment_log_omits_untrusted_api_body(
    monkeypatch, tmp_path, caplog, body_kind
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XClient

    sentinel = "PRIVATE_POST_OR_DM_SENTINEL"

    def handler(_request):
        if body_kind == "json":
            return httpx.Response(403, json={"detail": sentinel})
        return httpx.Response(403, text=sentinel)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._client = XClient(token="token", transport=httpx.MockTransport(handler))
    trigger = {"id": "102", "conversation_id": "100"}

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.twitter.adapter"):
        await adapter._conversation_posts(trigger, {})

    assert sentinel not in caplog.text
    assert "X API 403 on /2/tweets/search/recent" in caplog.text
    await adapter._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(408, json={"detail": "timed out"}),
        httpx.Response(201, content=b"not-json"),
    ],
    ids=["http-408", "malformed-success"],
)
async def test_post_write_response_failures_are_ambiguous(response):
    from plugins.platforms.twitter.client import AmbiguousWriteError, XClient

    client = XClient(
        token="token", transport=httpx.MockTransport(lambda _request: response)
    )
    with pytest.raises(AmbiguousWriteError):
        await client.create_post("hello", reply_to="123")
    await client.close()


@pytest.mark.asyncio
async def test_queue_overflow_fails_before_network():
    from plugins.platforms.twitter.queue import RateQueue

    queue = RateQueue(max_pending=1, max_wait_seconds=1)
    blocker = asyncio.Event()
    first = asyncio.create_task(queue.run("write", blocker.wait))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="queue is full"):
        await queue.run("write", lambda: asyncio.sleep(0))
    blocker.set()
    await first


@pytest.mark.asyncio
async def test_queue_wait_timeout_does_not_cancel_started_operation():
    from plugins.platforms.twitter.queue import RateQueue

    queue = RateQueue(max_pending=1, max_wait_seconds=0.01)

    async def slow_operation():
        await asyncio.sleep(0.03)
        return "completed"

    assert await queue.run("write", slow_operation) == "completed"


@pytest.mark.asyncio
async def test_cancelled_queued_work_cannot_run_later():
    from plugins.platforms.twitter.queue import RateQueue

    queue = RateQueue(max_pending=2, max_wait_seconds=1)
    started = asyncio.Event()
    release = asyncio.Event()
    delivered = False

    async def first_operation():
        started.set()
        await release.wait()

    async def cancelled_operation():
        nonlocal delivered
        delivered = True

    first = asyncio.create_task(queue.run("write", first_operation))
    await started.wait()
    cancelled = asyncio.create_task(queue.run("write", cancelled_operation))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    await first

    assert not delivered


@pytest.mark.asyncio
async def test_cancelled_write_waiter_releases_reply_reservation(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XClient

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = XClient(
        token="token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(201, json={"data": {"id": "700"}})
        ),
        max_pending=2,
        max_wait_seconds=1,
    )
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )
    blocker = asyncio.Event()
    occupying = asyncio.create_task(
        adapter._client._queue.run("public_writes", blocker.wait)
    )
    await asyncio.sleep(0)

    waiting = asyncio.create_task(
        adapter.send("tweet:100:101", "reply", reply_to="101")
    )
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    blocker.set()
    await occupying

    retry = await adapter.send("tweet:100:101", "retry", reply_to="101")
    assert retry.success
    await adapter._client.close()


@pytest.mark.asyncio
async def test_timed_out_queued_work_cannot_run_later():
    from plugins.platforms.twitter.queue import RateQueue

    queue = RateQueue(max_pending=2, max_wait_seconds=0)
    started = asyncio.Event()
    release = asyncio.Event()
    delivered = False

    async def first_operation():
        started.set()
        await release.wait()

    async def timed_out_operation():
        nonlocal delivered
        delivered = True

    first = asyncio.create_task(queue.run("write", first_operation))
    await started.wait()
    with pytest.raises(TimeoutError):
        await queue.run("write", timed_out_operation)
    release.set()
    await first

    assert not delivered


@pytest.mark.asyncio
async def test_client_keeps_large_ids_as_strings():
    from plugins.platforms.twitter.client import XClient

    def handler(request):
        assert request.url.path == "/2/tweets"
        assert json.loads(request.content)["reply"]["in_reply_to_tweet_id"] == (
            "9007199254740993"
        )
        return httpx.Response(201, json={"data": {"id": "9007199254740994"}})

    client = XClient(token="token", transport=httpx.MockTransport(handler))
    result = await client.create_post("hello", reply_to="9007199254740993")
    assert result == "9007199254740994"
    await client.close()


@pytest.mark.asyncio
async def test_client_retries_explicit_rate_limit_response():
    from plugins.platforms.twitter.client import XClient

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"detail": "rate limited"},
            )
        return httpx.Response(200, json={"data": {"id": "7"}})

    client = XClient(token="token", transport=httpx.MockTransport(handler))
    assert (await client.identity())["data"]["id"] == "7"
    assert calls == 2
    await client.close()


@pytest.mark.asyncio
async def test_client_retries_429_before_write(monkeypatch):
    from plugins.platforms.twitter import client as client_module

    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "30"},
                json={"detail": "rate limited"},
            )
        return httpx.Response(201, json={"data": {"id": "700"}})

    monkeypatch.setattr(client_module.asyncio, "sleep", AsyncMock())
    client = client_module.XClient(
        token="token", transport=httpx.MockTransport(handler)
    )
    assert await client.create_post("hello", reply_to="123") == "700"
    assert calls == 2
    client_module.asyncio.sleep.assert_awaited_once_with(30)
    await client.close()


@pytest.mark.asyncio
async def test_client_uses_fresh_token_provider():
    from plugins.platforms.twitter.client import XClient

    def handler(request):
        assert request.headers["Authorization"] == "Bearer fresh"
        return httpx.Response(200, json={"data": {"id": "7"}})

    provider = AsyncMock(return_value="fresh")
    client = XClient(
        token="stale",
        token_provider=provider,
        transport=httpx.MockTransport(handler),
    )
    await client.identity()
    provider.assert_awaited_once()
    await client.close()


@pytest.mark.asyncio
async def test_oauth_refresh_rejects_changed_client_binding(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import OAuthClient, SCOPES, save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = save_tokens(
        {
            "access_token": "fresh",
            "expires_at": 9999999999,
            "scopes": list(SCOPES),
            "client_id": "original-client",
        }
    )
    oauth = OAuthClient("other-client", "http://127.0.0.1:8765/callback")

    with pytest.raises(RuntimeError, match="client changed"):
        await oauth.refresh(tokens)
    await oauth.close()


@pytest.mark.asyncio
async def test_refresh_is_serialized_across_oauth_clients(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import OAuthClient, SCOPES, load_tokens, save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expired = save_tokens(
        {
            "access_token": "expired",
            "refresh_token": "rotate-once",
            "expires_at": 1,
            "scopes": list(SCOPES),
            "client_id": "client",
        }
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "access_token": "fresh",
                "refresh_token": "rotated-refresh-token",
                "expires_in": 3600,
                "scope": " ".join(SCOPES),
            },
        )

    transport = httpx.MockTransport(handler)
    first_http = httpx.AsyncClient(transport=transport)
    second_http = httpx.AsyncClient(transport=transport)
    first = OAuthClient("client", "http://127.0.0.1:8765/callback", client=first_http)
    second = OAuthClient("client", "http://127.0.0.1:8765/callback", client=second_http)
    one, two = await asyncio.gather(first.refresh(expired), second.refresh(expired))

    assert one.access_token == two.access_token == "fresh"
    assert calls == 1
    assert load_tokens().refresh_token == "rotated-refresh-token"
    await first_http.aclose()
    await second_http.aclose()


@pytest.mark.asyncio
async def test_mention_requires_structured_trigger_and_authorization(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TWITTER_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TWITTER_ALLOW_ALL_USERS", raising=False)
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter.handle_message = AsyncMock()
    post = {
        "id": "101",
        "author_id": "42",
        "conversation_id": "100",
        "text": "@bot hello",
        "entities": {"mentions": [{"id": "7", "username": "bot"}]},
        "referenced_tweets": [],
    }
    await adapter._process_mention(post, {"users": [{"id": "42", "username": "alice"}]})
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "tweet:100:101"
    assert event.source.user_id == "42"
    assert event.message_id == "101"

    adapter.handle_message.reset_mock()
    await adapter._process_mention({**post, "id": "102", "author_id": "99"}, {})
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "should_dispatch"),
    [
        ("structured_mention", True),
        ("direct_reply_to_bot", True),
        ("substring_only", False),
        ("sibling_only", False),
        ("quote_only", False),
        ("quote_with_structured_mention", True),
        ("self_authored", False),
        ("unauthorized", False),
    ],
)
async def test_public_trigger_matrix(monkeypatch, tmp_path, case, should_dispatch):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter.handle_message = AsyncMock()
    adapter._state.map_bot_post("900", "101")
    post = {
        "id": "101",
        "author_id": "42",
        "conversation_id": "100",
        "text": "@bot hello",
        "referenced_tweets": [],
    }
    if case == "structured_mention":
        post["entities"] = {"mentions": [{"id": "7"}]}
    elif case == "direct_reply_to_bot":
        post["in_reply_to_user_id"] = "7"
        post["referenced_tweets"] = [{"type": "replied_to", "id": "900"}]
    elif case == "sibling_only":
        post["referenced_tweets"] = [{"type": "replied_to", "id": "800"}]
    elif case == "quote_only":
        post["referenced_tweets"] = [{"type": "quoted", "id": "900"}]
    elif case == "quote_with_structured_mention":
        post["entities"] = {"mentions": [{"id": "7"}]}
        post["referenced_tweets"] = [{"type": "quoted", "id": "900"}]
    elif case == "self_authored":
        post["author_id"] = "7"
        post["entities"] = {"mentions": [{"id": "7"}]}
    elif case == "unauthorized":
        post["author_id"] = "99"
        post["entities"] = {"mentions": [{"id": "7"}]}

    await adapter._process_mention(post, {"users": [{"id": "42", "username": "alice"}]})

    assert adapter.handle_message.await_count == int(should_dispatch)


@pytest.mark.asyncio
async def test_authorized_branch_participant_uses_mapped_anchor(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42", "43"]})
    )
    adapter._account_id = "7"
    adapter._state.map_bot_post("900", "101")
    adapter.handle_message = AsyncMock()
    authorized_routes = []
    adapter.set_authorization_check(
        lambda _user_id, _chat_type, chat_id: authorized_routes.append(chat_id) or True
    )

    await adapter._process_mention(
        {
            "id": "102",
            "author_id": "43",
            "conversation_id": "100",
            "text": "new participant",
            "in_reply_to_user_id": "7",
            "referenced_tweets": [{"type": "replied_to", "id": "900"}],
        },
        {"tweets": [{"id": "900", "author_id": "7", "text": "bot reply"}], "users": [{"id": "43", "username": "bob"}]},
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "tweet:100:101"
    assert event.source.user_id == "43"
    assert event.text == "new participant"
    assert authorized_routes == ["tweet:100:101"]


@pytest.mark.asyncio
async def test_final_mapped_route_must_be_authorized_before_enrichment(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter._state.map_bot_post("900", "101")
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "800",
                    "author_id": "42",
                    "text": "earlier reply",
                    "referenced_tweets": [{"type": "replied_to", "id": "900"}],
                }
            ]
        }
    )
    adapter._client.quote_posts = AsyncMock(side_effect=AssertionError)
    adapter._inbound_media = AsyncMock(side_effect=AssertionError)
    adapter.handle_message = AsyncMock()
    authorized_routes = []
    adapter.set_authorization_check(
        lambda _user_id, _chat_type, chat_id: authorized_routes.append(chat_id)
        or chat_id == "tweet:100:102"
    )

    await adapter._process_mention(
        {
            "id": "102",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot help",
            "entities": {"mentions": [{"id": "7"}]},
            "referenced_tweets": [{"type": "replied_to", "id": "800"}],
        },
        {},
    )

    assert authorized_routes == ["tweet:100:102", "tweet:100:101"]
    adapter._client.quote_posts.assert_not_awaited()
    adapter._inbound_media.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_quote_context_is_bounded_untrusted_event_context(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(
            extra={
                "client_id": "client",
                "allow_all_users": True,
                "conversation": {"quote_posts_per_target": 1},
            }
        )
    )
    adapter._account_id = "7"
    adapter._state.map_bot_post("900", "101")
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.quote_posts = AsyncMock(
        return_value={
            "data": [
                {"id": "910", "author_id": "44", "text": "quoted context"},
                {"id": "911", "author_id": "45", "text": "outside cap"},
            ]
        }
    )
    adapter.handle_message = AsyncMock()

    await adapter._process_mention(
        {
            "id": "102",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot please help",
            "entities": {"mentions": [{"id": "7"}]},
            "referenced_tweets": [{"type": "quoted", "id": "900"}],
        },
        {"users": [{"id": "42", "username": "alice"}]},
    )

    event = adapter.handle_message.await_args.args[0]
    adapter._client.quote_posts.assert_awaited_once_with("900", limit=1)
    assert "Untrusted X quote context" in event.channel_context
    assert "quoted context" in event.channel_context
    assert "outside cap" not in event.channel_context
    assert event.text == "@bot please help"
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_gateway_authorization_cannot_bypass_profile_local_policy(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TWITTER_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TWITTER_ALLOW_ALL_USERS", raising=False)
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": False})
    )
    adapter._account_id = "7"
    assert not adapter._authorized("42", chat_type="group", chat_id="tweet:1:2")

    adapter.set_authorization_check(lambda user_id, chat_type, chat_id: user_id == "42")
    assert not adapter._authorized("42", chat_type="group", chat_id="tweet:1:2")
    assert not adapter._authorized("99", chat_type="group", chat_id="tweet:1:2")

    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter.set_authorization_check(lambda user_id, chat_type, chat_id: user_id == "42")
    assert adapter._authorized("42", chat_type="group", chat_id="tweet:1:2")

    adapter.set_authorization_check(lambda user_id, chat_type, chat_id: False)
    assert not adapter._authorized("42", chat_type="group", chat_id="tweet:1:2")


@pytest.mark.asyncio
async def test_dm_routes_by_real_conversation_id(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter.handle_message = AsyncMock()
    await adapter._process_dm(
        {
            "id": "501",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "42-7",
            "text": "hello",
        },
        {"users": [{"id": "42", "username": "alice"}]},
    )
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "dm:42-7"
    assert event.message_id == "501"


@pytest.mark.asyncio
async def test_adapter_send_uses_typed_routes(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.create_post = AsyncMock(return_value="700")
    adapter._client.send_dm = AsyncMock(return_value="701")
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot help",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )
    await adapter._process_dm(
        {
            "id": "501",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "42-7",
            "text": "hello",
        },
        {},
    )

    public = await adapter.send("tweet:100:101", "public", reply_to="101")
    direct = await adapter.send("dm:42-7", "private", reply_to="501")
    invalid = await adapter.send("123", "bad")
    unrecorded = await adapter.send("tweet:100:999", "bad", reply_to="999")
    duplicate = await adapter.send("tweet:100:101", "again", reply_to="101")
    duplicate_dm = await adapter.send("dm:42-7", "again", reply_to="501")

    assert public.success and public.message_id == "700"
    assert direct.success and direct.message_id == "701"
    assert not invalid.success
    assert not unrecorded.success
    assert not duplicate.success
    assert not duplicate_dm.success
    adapter._client.create_post.assert_awaited_once_with("public", reply_to="101")
    adapter._client.send_dm.assert_awaited_once_with("42-7", "private")


@pytest.mark.asyncio
async def test_public_reply_over_weighted_limit_attempts_one_long_post(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import format_public_message

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.create_post = AsyncMock(return_value="700")
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot explain",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )
    message = "word " * 100

    result = await adapter.send("tweet:100:101", message, reply_to="101")

    assert result.success and result.message_id == "700"
    assert result.continuation_message_ids == ()
    adapter._client.create_post.assert_awaited_once_with(
        format_public_message(message), reply_to="101"
    )
    assert adapter._state.bot_posts_for_anchor("101") == {"700"}


@pytest.mark.asyncio
async def test_timeline_over_weighted_limit_attempts_one_long_post(tmp_path, monkeypatch):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import format_public_message

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    message = "word " * 100

    result = await adapter.send("timeline", message)

    assert result.success and result.message_id == "700"
    adapter._client.create_post.assert_awaited_once_with(
        format_public_message(message)
    )


@pytest.mark.asyncio
async def test_rejected_long_timeline_post_falls_back_to_thread(
    tmp_path, monkeypatch
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError
    from plugins.platforms.twitter.presentation import (
        format_public_message,
        format_thread_messages,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(
        side_effect=[XApiError(422, "/2/tweets", "too long"), "700", "701"]
    )
    adapter._upload_images = AsyncMock(return_value=["media-1"])
    message = "word " * 100
    parts = format_thread_messages(message)

    result = await adapter.send("timeline", message)

    assert result.success and result.message_id == "701"
    assert result.continuation_message_ids == ("700",)
    assert result.raw_response == {"message_ids": ["700", "701"]}
    assert adapter._client.create_post.await_args_list[0] == call(
        format_public_message(message), media_ids=["media-1"]
    )
    assert adapter._client.create_post.await_args_list[1] == call(
        parts[0], media_ids=["media-1"]
    )
    assert adapter._client.create_post.await_args_list[2] == call(
        parts[1], reply_to="700"
    )


@pytest.mark.asyncio
async def test_partial_timeline_thread_is_accepted(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(
        side_effect=[
            XApiError(400, "/2/tweets", "too long"),
            "700",
            XApiError(500, "/2/tweets", "failed"),
        ]
    )

    result = await adapter.send("timeline", "word " * 100)

    assert result.success and result.message_id == "700"
    assert result.raw_response == {
        "partial_delivery": True,
        "message_ids": ["700"],
        "delivered_parts": 1,
        "total_parts": 2,
        "failure_status": 500,
    }


@pytest.mark.parametrize(
    "error",
    [
        pytest.param("rate_limited", id="429"),
        pytest.param("ambiguous", id="ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_long_timeline_post_does_not_split_on_uncertain_error(
    error, monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError, XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    rejection = (
        XApiError(429, "/2/tweets", "limited")
        if error == "rate_limited"
        else AmbiguousWriteError("uncertain")
    )
    adapter._client.create_post = AsyncMock(side_effect=rejection)

    result = await adapter.send("timeline", "word " * 100)

    assert not result.success
    assert adapter._client.create_post.await_count == 1


@pytest.mark.asyncio
async def test_timeline_rejects_content_over_bot_weighted_limit(tmp_path, monkeypatch):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import X_BOT_WEIGHTED_LIMIT

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")

    result = await adapter.send("timeline", "x" * (X_BOT_WEIGHTED_LIMIT + 1))

    assert not result.success
    assert "weighted delivery limit" in result.error
    adapter._client.create_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_over_weighted_limit_sends_complete_message(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import format_public_message

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(return_value="701")
    await adapter._mutate_state(lambda state: state.record_dm_inbound("42-7"))
    message = "word " * 100

    result = await adapter.send("dm:42-7", message)

    assert result.success and result.message_id == "701"
    adapter._client.send_dm.assert_awaited_once_with(
        "42-7", format_public_message(message)
    )


@pytest.mark.asyncio
async def test_rejected_long_dm_falls_back_to_sequential_messages(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError
    from plugins.platforms.twitter.presentation import (
        format_public_message,
        format_thread_messages,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(
        side_effect=[XApiError(413, "/2/dm_conversations", "too long"), "700", "701"]
    )
    adapter._upload_images = AsyncMock(return_value=["media-1"])
    await adapter._mutate_state(lambda state: state.record_dm_inbound("42-7"))
    message = "word " * 100
    parts = format_thread_messages(message)

    result = await adapter.send("dm:42-7", message)

    assert result.success and result.message_id == "701"
    assert result.continuation_message_ids == ("700",)
    assert result.raw_response == {"message_ids": ["700", "701"]}
    assert adapter._client.send_dm.await_args_list == [
        call("42-7", format_public_message(message), media_id="media-1"),
        call("42-7", parts[0], media_id="media-1"),
        call("42-7", parts[1]),
    ]


@pytest.mark.asyncio
async def test_partial_anchored_dm_delivery_is_accepted(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(
        side_effect=[
            XApiError(400, "/2/dm_conversations", "too long"),
            "700",
            XApiError(500, "/2/dm_conversations", "failed"),
        ]
    )
    adapter.handle_message = AsyncMock()
    await adapter._process_dm(
        {
            "id": "501",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "42-7",
            "text": "explain",
        },
        {},
    )
    message = "word " * 100

    partial = await adapter.send("dm:42-7", message, reply_to="501")
    retry = await adapter.send("dm:42-7", message, reply_to="501")

    assert partial.success and partial.message_id == "700"
    assert partial.raw_response == {
        "partial_delivery": True,
        "message_ids": ["700"],
        "delivered_parts": 1,
        "total_parts": 2,
        "failure_status": 500,
    }
    assert not retry.success and "already reserved" in retry.error
    assert adapter._client.send_dm.await_count == 3


@pytest.mark.parametrize(
    "error",
    [
        pytest.param("rate_limited", id="429"),
        pytest.param("ambiguous", id="ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_long_dm_does_not_split_on_uncertain_error(error, monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError, XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = Mock()
    rejection = (
        XApiError(429, "/2/dm_conversations", "limited")
        if error == "rate_limited"
        else AmbiguousWriteError("uncertain")
    )
    adapter._client.send_dm = AsyncMock(side_effect=rejection)
    await adapter._mutate_state(lambda state: state.record_dm_inbound("42-7"))

    result = await adapter.send("dm:42-7", "word " * 100)

    assert not result.success
    assert adapter._client.send_dm.await_count == 1


@pytest.mark.asyncio
async def test_over_bot_limit_reply_sends_narrower_answer_notice(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import (
        X_BOT_WEIGHTED_LIMIT,
        X_OVER_LIMIT_NOTICE,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )

    result = await adapter.send(
        "tweet:100:101", "x" * (X_BOT_WEIGHTED_LIMIT + 1), reply_to="101"
    )

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(
        X_OVER_LIMIT_NOTICE, reply_to="101"
    )


@pytest.mark.asyncio
async def test_over_bot_limit_dm_sends_narrower_answer_notice(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import (
        X_BOT_WEIGHTED_LIMIT,
        X_OVER_LIMIT_NOTICE,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(return_value="701")
    await adapter._mutate_state(lambda state: state.record_dm_inbound("42-7"))

    result = await adapter.send("dm:42-7", "x" * (X_BOT_WEIGHTED_LIMIT + 1))

    assert result.success
    adapter._client.send_dm.assert_awaited_once_with("42-7", X_OVER_LIMIT_NOTICE)


@pytest.mark.asyncio
async def test_bot_limit_uses_x_weighting_instead_of_python_length(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import x_weighted_length

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    message = " ".join(
        f"https://example.com/{index}/" + "x" * 400 for index in range(100)
    )
    assert len(message) > 25_000
    assert x_weighted_length(message) < 2_800

    result = await adapter.send("timeline", message)

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(message)


@pytest.mark.asyncio
async def test_timeline_rejects_content_requiring_more_than_ten_parts(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import (
        X_BOT_WEIGHTED_LIMIT,
        X_MAX_FALLBACK_PARTS,
        format_thread_messages,
        x_weighted_length,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    message = ("a " + "b" * 278 + " ") * 9
    assert x_weighted_length(message) < X_BOT_WEIGHTED_LIMIT
    assert len(format_thread_messages(message)) > X_MAX_FALLBACK_PARTS

    result = await adapter.send("timeline", message)

    assert not result.success
    assert "10 fallback parts" in result.error
    adapter._client.create_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_requiring_more_than_ten_parts_sends_narrower_answer_notice(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.presentation import X_OVER_LIMIT_NOTICE

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )
    message = ("a " + "b" * 278 + " ") * 9

    result = await adapter.send("tweet:100:101", message, reply_to="101")

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(
        X_OVER_LIMIT_NOTICE, reply_to="101"
    )


@pytest.mark.parametrize("status", [400, 403, 413, 422])
@pytest.mark.asyncio
async def test_rejected_long_reply_falls_back_to_reply_thread(
    status, caplog, monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError
    from plugins.platforms.twitter.presentation import (
        format_public_message,
        format_thread_messages,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.create_post = AsyncMock(
        side_effect=[XApiError(status, "/2/tweets", "too long"), "700", "701"]
    )
    caplog.set_level(logging.INFO, logger="plugins.platforms.twitter.adapter")
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )
    message = "word " * 100
    parts = format_thread_messages(message)

    result = await adapter.send("tweet:100:101", message, reply_to="101")

    assert result.success and result.message_id == "701"
    assert result.continuation_message_ids == ("700",)
    assert result.raw_response == {"message_ids": ["700", "701"]}
    assert adapter._client.create_post.await_args_list[0].args == (
        format_public_message(message),
    )
    assert adapter._client.create_post.await_args_list[0].kwargs == {"reply_to": "101"}
    assert adapter._client.create_post.await_args_list[1].args == (parts[0],)
    assert adapter._client.create_post.await_args_list[1].kwargs == {"reply_to": "101"}
    assert adapter._client.create_post.await_args_list[2].args == (parts[1],)
    assert adapter._client.create_post.await_args_list[2].kwargs == {"reply_to": "700"}
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert f"destination=reply status={status} total_parts=2" in log_text
    assert "part=1 total_parts=2 message_id=700" in log_text
    assert "part=2 total_parts=2 message_id=701" in log_text
    assert format_public_message(message) not in log_text


@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.asyncio
async def test_long_reply_does_not_split_on_non_content_error(
    status, monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(
        side_effect=XApiError(status, "/2/tweets", "rejected")
    )
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )

    result = await adapter.send(
        "tweet:100:101", "word " * 100, reply_to="101"
    )

    assert not result.success
    assert adapter._client.create_post.await_count == 1


@pytest.mark.asyncio
async def test_long_reply_does_not_split_on_ambiguous_write(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(
        side_effect=AmbiguousWriteError("uncertain")
    )
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )

    result = await adapter.send(
        "tweet:100:101", "word " * 100, reply_to="101"
    )

    assert not result.success
    assert adapter._client.create_post.await_count == 1


@pytest.mark.asyncio
async def test_partial_reply_thread_is_not_duplicated_on_retry(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.create_post = AsyncMock(
        side_effect=[
            XApiError(400, "/2/tweets", "too long"),
            "700",
            XApiError(500, "/2/tweets", "failed"),
        ]
    )
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot explain",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )
    message = "word " * 100

    partial = await adapter.send("tweet:100:101", message, reply_to="101")
    retry = await adapter.send("tweet:100:101", message, reply_to="101")

    assert partial.success and partial.message_id == "700"
    assert partial.continuation_message_ids == ()
    assert partial.raw_response == {
        "partial_delivery": True,
        "message_ids": ["700"],
        "delivered_parts": 1,
        "total_parts": 2,
        "failure_status": 500,
    }
    assert not retry.success and "not eligible" in retry.error
    assert adapter._client.create_post.await_count == 3
    assert adapter._state.bot_posts_for_anchor("101") == {"700"}


@pytest.mark.asyncio
async def test_ambiguous_continuation_after_confirmed_reply_is_accepted(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError, XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(
        side_effect=[
            XApiError(400, "/2/tweets", "too long"),
            "700",
            AmbiguousWriteError("uncertain"),
        ]
    )
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )

    result = await adapter.send(
        "tweet:100:101", "word " * 100, reply_to="101"
    )

    assert result.success and result.message_id == "700"
    assert result.raw_response == {
        "partial_delivery": True,
        "message_ids": ["700"],
        "delivered_parts": 1,
        "total_parts": 2,
        "failure_kind": "ambiguous",
    }


@pytest.mark.asyncio
async def test_automated_delivery_requires_policy_readiness(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": True})
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.create_post = AsyncMock(return_value="700")
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot help",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )

    result = await adapter.send("tweet:100:101", "public", reply_to="101")

    assert not result.success
    assert "must be confirmed" in result.error
    adapter._client.create_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_requires_inbound_conversation_and_honors_opt_out(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        ready_twitter_config(allowed_users=["42"], allow_all_users=False)
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(return_value="701")
    adapter.handle_message = AsyncMock()

    async def assert_recorded_before_enrichment(*_args):
        assert adapter._state.can_send_dm("42-7")
        return [], [], ""

    adapter._inbound_media = AsyncMock(side_effect=assert_recorded_before_enrichment)

    unknown = await adapter.send("dm:88-7", "private")
    await adapter._process_dm(
        {
            "id": "501",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "42-7",
            "text": "hello",
        },
        {},
    )
    known = await adapter.send("dm:42-7", "private", reply_to="501")
    adapter._inbound_media = AsyncMock(side_effect=AssertionError)
    await adapter._process_dm(
        {
            "id": "502",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "99-7",
            "text": " STOP ",
        },
        {},
    )
    opted_out = await adapter.send("dm:99-7", "private")

    assert not unknown.success
    assert known.success
    assert not opted_out.success
    assert adapter.handle_message.await_count == 1
    adapter._client.send_dm.assert_awaited_once_with("42-7", "private")


@pytest.mark.asyncio
async def test_dm_opt_out_persisted_during_preparation_prevents_write(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(return_value="701")
    await adapter._mutate_state(lambda state: state.record_dm_inbound("42-7"))

    async def opt_out_before_write(*_args, **_kwargs):
        await adapter._mutate_state(lambda state: state.opt_out_dm("42-7"))
        return []

    adapter._upload_images = AsyncMock(side_effect=opt_out_before_write)

    result = await adapter.send("dm:42-7", "private")

    assert not result.success
    adapter._client.send_dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeline_rejects_mentions_and_unconfirmed_ids(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="not-an-x-id")

    mention = await adapter.send("timeline", "hello @alice")
    unconfirmed = await adapter.send("timeline", "hello everyone")

    assert not mention.success
    assert not unconfirmed.success
    adapter._client.create_post.assert_awaited_once_with("hello everyone")


@pytest.mark.asyncio
async def test_ambiguous_reply_keeps_reservation_but_429_releases_it(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError, XApiError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter._client.create_post = AsyncMock(
        side_effect=AmbiguousWriteError("uncertain")
    )
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot first",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )

    ambiguous = await adapter.send("tweet:100:101", "reply", reply_to="101")
    retry = await adapter.send("tweet:100:101", "retry", reply_to="101")

    assert not ambiguous.success and not ambiguous.retryable
    assert not retry.success
    adapter._client.create_post.assert_awaited_once()

    adapter._client.create_post = AsyncMock(
        side_effect=[XApiError(429, "/2/tweets", "limited"), "702"]
    )
    await adapter._process_mention(
        {
            "id": "102",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot second",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )

    limited = await adapter.send("tweet:100:102", "reply", reply_to="102")
    after_limit = await adapter.send("tweet:100:102", "retry", reply_to="102")

    assert not limited.success and limited.retryable
    assert after_limit.success and after_limit.message_id == "702"
    assert adapter._client.create_post.await_count == 2


@pytest.mark.asyncio
async def test_ambiguous_dm_reply_keeps_reservation(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(side_effect=AmbiguousWriteError("uncertain"))
    adapter.handle_message = AsyncMock()
    await adapter._process_dm(
        {
            "id": "501",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "42-7",
            "text": "hello",
        },
        {},
    )

    ambiguous = await adapter.send("dm:42-7", "reply", reply_to="501")
    retry = await adapter.send("dm:42-7", "retry", reply_to="501")

    assert not ambiguous.success and not ambiguous.retryable
    assert not retry.success
    adapter._client.send_dm.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_redelivery_without_dm_identity_fails_closed(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import AmbiguousWriteError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    first._account_id = "7"
    first._client = Mock()
    first._client.send_dm = AsyncMock(side_effect=AmbiguousWriteError("uncertain"))
    await first._mutate_state(
        lambda state: state.record_dm_inbound("42-7", "501")
    )
    ambiguous = await first.send("dm:42-7", "reply", reply_to="501")

    restarted = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    restarted._account_id = "7"
    restarted._client = Mock()
    restarted._client.send_dm = AsyncMock(return_value="701")
    recovered = await restarted.send("dm:42-7", "reply")

    assert not ambiguous.success
    assert not recovered.success
    restarted._client.send_dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_restart_redelivery_fails_closed_for_persisted_dm_write(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    first._account_id = "7"

    def persist_in_flight_write(state):
        state.record_dm_inbound("42-7", "501")
        assert state.reserve_dm_reply("501", "42-7")
        assert state.begin_dm_reply("501", "42-7")

    await first._mutate_state(persist_in_flight_write)

    restarted = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    restarted._account_id = "7"
    restarted._client = Mock()
    restarted._client.send_dm = AsyncMock(return_value="701")
    recovered = await restarted.send("dm:42-7", "reply")

    assert not recovered.success
    restarted._client.send_dm.assert_not_awaited()
    await restarted._mutate_state(
        lambda state: state.clear_dm_delivery_uncertainty()
    )
    assert not restarted._state.has_ambiguous_dm_write()
    assert not restarted._state.reserve_dm_reply("501", "42-7")
    reconciled_recovery = await restarted.send("dm:42-7", "reply")
    assert not reconciled_recovery.success
    restarted._client.send_dm.assert_not_awaited()
    await restarted._mutate_state(
        lambda state: state.record_dm_inbound("42-7", "502")
    )
    anchored = await restarted.send("dm:42-7", "new reply", reply_to="502")

    assert anchored.success
    restarted._client.send_dm.assert_awaited_once_with("42-7", "new reply")


@pytest.mark.asyncio
async def test_restart_after_confirmed_dm_before_gateway_ack_fails_closed(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    first._account_id = "7"

    def persist_confirmed_write(state):
        state.record_dm_inbound("42-7", "501")
        assert state.reserve_dm_reply("501", "42-7")
        assert state.begin_dm_reply("501", "42-7")
        state.confirm_dm_reply("501")

    await first._mutate_state(persist_confirmed_write)

    restarted = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    restarted._account_id = "7"
    restarted._client = Mock()
    restarted._client.send_dm = AsyncMock(return_value="702")

    recovered = await restarted.send("dm:42-7", "reply")

    assert not recovered.success
    restarted._client.send_dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_dm_restart_blocks_same_conversation_not_another(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    first._account_id = "7"

    def persist_confirmed_write(state):
        state.record_dm_inbound("42-7", "501")
        assert state.reserve_dm_reply("501", "42-7")
        assert state.begin_dm_reply("501", "42-7")
        state.confirm_dm_reply("501")
        state.record_dm_inbound("43-7", "601")

    await first._mutate_state(persist_confirmed_write)

    restarted = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    restarted._account_id = "7"
    restarted._client = Mock()
    restarted._client.send_dm = AsyncMock(return_value="702")

    same_delivery = await restarted.send("dm:42-7", "retry")
    later_conversation = await restarted.send("dm:43-7", "new delivery")

    assert not same_delivery.success
    assert later_conversation.success
    restarted._client.send_dm.assert_awaited_once_with("43-7", "new delivery")


@pytest.mark.asyncio
async def test_dm_uncertainty_during_media_preparation_blocks_final_write(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.send_dm = AsyncMock(return_value="701")
    await adapter._mutate_state(lambda state: state.record_dm_inbound("42-7"))

    async def make_other_delivery_ambiguous(*_args, **_kwargs):
        def mutate(state):
            state.record_dm_inbound("42-7", "501")
            assert state.reserve_dm_reply("501", "42-7")
            assert state.begin_dm_reply("501", "42-7")
            state.mark_dm_reply_uncertain("501")

        await adapter._mutate_state(mutate)
        return []

    adapter._upload_images = AsyncMock(side_effect=make_other_delivery_ambiguous)

    result = await adapter.send("dm:42-7", "private")

    assert not result.success
    adapter._client.send_dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_media_success_is_classified_and_releases_reply_reservation(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XApiError, XClient

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    client = XClient(
        token="token",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(201, json={"data": [{"id": "bad"}]})
        ),
    )
    image = tmp_path / "image.png"
    image.write_bytes(b"not-an-image")
    with pytest.raises(XApiError, match="media id"):
        await client.upload_image(image)
    await client.close()

    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    adapter._upload_images = AsyncMock(
        side_effect=[
            XApiError(502, "/2/media/upload", "response omitted media id"),
            [],
        ]
    )
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )

    failed = await adapter.send("tweet:100:101", "reply", reply_to="101")
    retry = await adapter.send("tweet:100:101", "retry", reply_to="101")

    assert not failed.success
    assert retry.success
    adapter._client.create_post.assert_awaited_once_with("retry", reply_to="101")


@pytest.mark.asyncio
async def test_definitive_malformed_error_releases_reply_reservation(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XClient

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    responses = iter(
        [
            httpx.Response(400, json={"errors": "bad request"}),
            httpx.Response(201, json={"data": {"id": "700"}}),
        ]
    )
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = XClient(
        token="token",
        transport=httpx.MockTransport(lambda _request: next(responses)),
    )
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("101", "tweet:100:101")
    )

    rejected = await adapter.send("tweet:100:101", "reply", reply_to="101")
    retry = await adapter.send("tweet:100:101", "retry", reply_to="101")

    assert not rejected.success
    assert retry.success
    await adapter._client.close()


@pytest.mark.asyncio
async def test_cancelled_reply_cannot_deliver_or_retry_later(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config(allow_all_users=True))
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(return_value={})
    started = asyncio.Event()
    release = asyncio.Event()
    delivered = False

    async def pending_write(*_args, **_kwargs):
        nonlocal delivered
        started.set()
        await release.wait()
        delivered = True
        return "700"

    adapter._client.create_post = AsyncMock(side_effect=pending_write)
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "42",
            "conversation_id": "100",
            "text": "@bot help",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )

    send = asyncio.create_task(
        adapter.send("tweet:100:101", "reply", reply_to="101")
    )
    await started.wait()
    send.cancel()
    with pytest.raises(asyncio.CancelledError):
        await send
    release.set()
    await asyncio.sleep(0)
    retry = await adapter.send("tweet:100:101", "retry", reply_to="101")

    assert not delivered
    assert not retry.success
    adapter._client.create_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_uploads_images_before_creating_post(monkeypatch, tmp_path):
    from PIL import Image

    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    image = tmp_path / "one.png"
    Image.new("RGB", (1, 1)).save(image)
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.upload_image = AsyncMock(return_value="800")
    adapter._client.create_post = AsyncMock(return_value="801")

    result = await adapter.send(
        "timeline", "with image", metadata={"media_files": [(str(image), False)]}
    )

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(
        "with image", media_ids=["800"]
    )


@pytest.mark.asyncio
async def test_partial_image_upload_never_creates_text_only_post(
    monkeypatch, tmp_path
):
    from PIL import Image

    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    images = []
    for name in ("one.png", "two.png"):
        path = tmp_path / name
        Image.new("RGB", (1, 1)).save(path)
        images.append((str(path), False))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.upload_image = AsyncMock(
        side_effect=["800", RuntimeError("upload failed")]
    )
    adapter._client.create_post = AsyncMock(return_value="801")

    result = await adapter.send(
        "timeline", "with images", metadata={"media_files": images}
    )

    assert not result.success
    adapter._client.create_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_transport_failure_returns_send_result(monkeypatch, tmp_path):
    from PIL import Image

    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.client import XClient

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    image = tmp_path / "one.png"
    Image.new("RGB", (1, 1)).save(image)

    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = XClient(
        token="token", transport=httpx.MockTransport(handler)
    )
    result = await adapter.send(
        "timeline", "with image", metadata={"media_files": [str(image)]}
    )

    assert not result.success
    assert result.retryable
    await adapter._client.close()


@pytest.mark.asyncio
async def test_inbound_image_is_downloaded_only_after_authorization(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(side_effect=AssertionError)
    adapter._inbound_media = AsyncMock(side_effect=AssertionError)
    adapter.handle_message = AsyncMock()

    await adapter._process_mention(
        {
            "id": "101",
            "author_id": "99",
            "conversation_id": "100",
            "text": "@bot hello",
            "entities": {"mentions": [{"id": "7"}]},
            "attachments": {"media_keys": ["3_1"]},
        },
        {"media": [{"media_key": "3_1", "type": "photo", "url": "https://x.example/image"}]},
    )

    adapter._inbound_media.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_image_rejects_mime_matched_but_undecodable_content(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "image/png", "content-length": "6"},
            content=b"GIF89a",
        )

    adapter = TwitterAdapter(
        PlatformConfig(
            extra={"client_id": "client", "_http_transport": httpx.MockTransport(handler)}
        )
    )

    with pytest.raises(ValueError, match="invalid"):
        await adapter._download_image("https://media.example/image.png")


@pytest.mark.asyncio
async def test_image_processing_deadline_prevents_text_only_fallback(monkeypatch, tmp_path):
    from PIL import Image

    from plugins.platforms.twitter import adapter as adapter_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        adapter_module, "MEDIA_PROCESSING_TIMEOUT_SECONDS", 0.01, raising=False
    )
    image = tmp_path / "one.png"
    Image.new("RGB", (1, 1)).save(image)
    adapter = adapter_module.TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()

    async def slow_upload(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return "800"

    adapter._client.upload_image = AsyncMock(side_effect=slow_upload)
    adapter._client.create_post = AsyncMock(return_value="801")

    result = await adapter.send(
        "timeline", "with image", metadata={"media_files": [str(image)]}
    )

    assert not result.success
    adapter._client.create_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_pixel_cap_prevents_upload(monkeypatch, tmp_path):
    from PIL import Image

    from plugins.platforms.twitter import adapter as adapter_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(adapter_module, "MAX_IMAGE_PIXELS", 1, raising=False)
    image = tmp_path / "one.png"
    Image.new("RGB", (2, 2)).save(image)
    adapter = adapter_module.TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.upload_image = AsyncMock(return_value="800")
    adapter._client.create_post = AsyncMock(return_value="801")

    result = await adapter.send(
        "timeline", "with image", metadata={"media_files": [str(image)]}
    )

    assert not result.success
    adapter._client.upload_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_uses_exact_bookmark_and_bounded_metric_requests():
    from plugins.platforms.twitter.client import XClient

    client = XClient(token="token")
    client.request = AsyncMock(return_value={})
    try:
        await client.bookmarks("7", "list")
        await client.bookmarks("7", "add", post_id="8")
        await client.bookmarks("7", "remove", post_id="8")
        await client.post_metrics(["1", "2"])
        with pytest.raises(ValueError, match="1 to 20"):
            await client.post_metrics([str(index) for index in range(21)])
        with pytest.raises(ValueError, match="string"):
            await client.post_metrics([1])
    finally:
        await client.close()

    calls = client.request.await_args_list
    assert calls[0].args[:2] == ("GET", "/2/users/7/bookmarks")
    assert "params" not in calls[0].kwargs
    assert calls[1].args[:2] == ("POST", "/2/users/7/bookmarks")
    assert calls[2].args[:2] == ("DELETE", "/2/users/7/bookmarks/8")
    assert calls[3].kwargs["params"] == {
        "ids": "1,2",
        "tweet.fields": "public_metrics,non_public_metrics",
    }


@pytest.mark.asyncio
async def test_metrics_tool_rejects_more_than_twenty_ids():
    from plugins.platforms.twitter.tools import handle_post_metrics

    result = await handle_post_metrics({"post_ids": [str(index) for index in range(21)]})

    assert "1 to 20" in result


@pytest.mark.asyncio
async def test_metrics_tool_rejects_non_string_ids():
    from plugins.platforms.twitter.tools import handle_post_metrics

    result = await handle_post_metrics({"post_ids": [1]})

    assert "string" in result


def test_twitter_tools_are_gated_by_profile_oauth(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import save_tokens
    from plugins.platforms.twitter.tools import twitter_available

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert not twitter_available()
    save_tokens({"access_token": "test"})
    assert twitter_available()

    save_tokens({"access_token": "expired", "expires_at": 1})
    assert not twitter_available()


@pytest.mark.asyncio
async def test_standalone_sender_uses_fresh_client(monkeypatch, tmp_path):
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens({"access_token": "test"})
    client = Mock()
    client.create_post = AsyncMock(return_value="901")
    client.close = AsyncMock()
    monkeypatch.setattr(adapter_module, "XClient", Mock(return_value=client))

    result = await adapter_module.standalone_send(
        PlatformConfig(extra={"client_id": "client"}), "timeline", "cron post"
    )

    assert result == {"success": True, "message_id": "901"}
    assert callable(adapter_module.XClient.call_args.kwargs["token_provider"])
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_standalone_sender_keeps_refreshed_token_and_account_together(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import SendResult
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import OAuthTokens, save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens({"access_token": "old", "user_id": "9"})
    monkeypatch.setattr(
        adapter_module,
        "refresh_if_needed",
        AsyncMock(
            return_value=OAuthTokens(
                access_token="fresh", client_id="client", user_id="7"
            )
        ),
    )
    client = Mock(close=AsyncMock())
    monkeypatch.setattr(adapter_module, "XClient", Mock(return_value=client))
    accounts = []

    async def capture_account(self, *_args, **_kwargs):
        accounts.append(self._account_id)
        return SendResult(success=True, message_id="901")

    monkeypatch.setattr(adapter_module.TwitterAdapter, "send", capture_account)

    result = await adapter_module.standalone_send(
        PlatformConfig(extra={"client_id": "client"}), "timeline", "cron post"
    )

    assert result == {"success": True, "message_id": "901"}
    assert accounts == ["7"]
    assert adapter_module.XClient.call_args.kwargs["token"] == "fresh"


@pytest.mark.asyncio
async def test_standalone_sender_rejects_account_replacement_before_http_send(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import OAuthTokens, save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens({"access_token": "seed", "client_id": "client", "user_id": "7"})
    monkeypatch.setattr(
        adapter_module,
        "refresh_if_needed",
        AsyncMock(
            side_effect=[
                OAuthTokens(access_token="account-a", client_id="client", user_id="7"),
                OAuthTokens(access_token="account-b", client_id="client", user_id="9"),
            ]
        ),
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201, json={"data": {"id": "901"}})

    result = await adapter_module.standalone_send(
        PlatformConfig(
            extra={
                "client_id": "client",
                "_http_transport": httpx.MockTransport(handler),
            }
        ),
        "timeline",
        "cron post",
    )

    assert "account changed" in result["error"]
    assert requests == []


@pytest.mark.asyncio
async def test_standalone_sender_reuses_live_public_reply_policy(monkeypatch, tmp_path):
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens({"access_token": "test", "user_id": "7"})
    client = Mock()
    client.create_post = AsyncMock(return_value="901")
    client.close = AsyncMock()
    monkeypatch.setattr(adapter_module, "XClient", Mock(return_value=client))

    result = await adapter_module.standalone_send(
        ready_twitter_config(), "tweet:100:101", "cron reply", thread_id="101"
    )

    assert "not eligible" in result["error"]
    client.create_post.assert_not_awaited()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_adapter_uses_delivery_metadata_interaction_id(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(ready_twitter_config())
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="901")
    await adapter._mutate_state(
        lambda state: state.record_public_interaction("102", "tweet:100:101")
    )

    result = await adapter.send(
        "tweet:100:101",
        "scheduled reply",
        metadata={"thread_id": "102"},
    )

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(
        "scheduled reply", reply_to="102"
    )


@pytest.mark.asyncio
async def test_dm_image_upload_marks_the_media_as_a_dm(monkeypatch, tmp_path):
    from PIL import Image

    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    image = tmp_path / "one.png"
    Image.new("RGB", (1, 1)).save(image)
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.upload_image = AsyncMock(return_value="800")

    assert await adapter._upload_images({"media_files": [str(image)]}, for_dm=True) == [
        "800"
    ]
    adapter._client.upload_image.assert_awaited_once_with(image, for_dm=True)


@pytest.mark.asyncio
async def test_standalone_sender_returns_refresh_transport_error(monkeypatch, tmp_path):
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens({"access_token": "expired", "expires_at": 1})
    request = httpx.Request("POST", "https://api.x.com/2/oauth2/token")
    monkeypatch.setattr(
        adapter_module,
        "refresh_if_needed",
        AsyncMock(side_effect=httpx.ConnectError("offline", request=request)),
    )

    result = await adapter_module.standalone_send(
        PlatformConfig(extra={"client_id": "client"}), "timeline", "cron post"
    )

    assert "offline" in result["error"]


@pytest.mark.asyncio
async def test_send_rejects_invalid_routes(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="1")
    adapter._client.send_dm = AsyncMock(return_value="2")

    invalid_post = await adapter.send("tweet:not-an-id:anchor", "bad")
    invalid_dm = await adapter.send("dm:../../tokens", "bad")

    assert not invalid_post.success
    assert not invalid_dm.success
    adapter._client.create_post.assert_not_awaited()
    adapter._client.send_dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeline_long_post_supports_weighted_unicode(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="1")

    message = "界" * 141
    result = await adapter.send("timeline", message)

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(message)


def test_conversation_context_is_bounded_and_chronological():
    from plugins.platforms.twitter.adapter import build_conversation_context

    posts = [
        {"id": "5", "author_id": "50", "text": "late sibling", "created_at": "2026-01-05", "referenced_tweets": [{"type": "replied_to", "id": "2"}]},
        {"id": "3", "author_id": "7", "text": "bot", "created_at": "2026-01-03", "referenced_tweets": [{"type": "replied_to", "id": "2"}]},
        {"id": "1", "author_id": "10", "text": "root", "created_at": "2026-01-01"},
        {"id": "4", "author_id": "40", "text": "trigger", "created_at": "2026-01-04", "referenced_tweets": [{"type": "replied_to", "id": "3"}]},
        {"id": "2", "author_id": "20", "text": "summon", "created_at": "2026-01-02", "referenced_tweets": [{"type": "replied_to", "id": "1"}]},
        {"id": "6", "author_id": "60", "text": "older sibling", "created_at": "2026-01-02T12:00:00Z", "referenced_tweets": [{"type": "replied_to", "id": "2"}]},
    ]
    rendered = build_conversation_context(
        posts,
        trigger_id="4",
        bot_post_ids={"3"},
        max_depth=3,
        max_posts=5,
        siblings_per_parent=1,
    )

    assert rendered.index("root") < rendered.index("summon") < rendered.index("bot")
    assert rendered.index("bot") < rendered.index("trigger")
    assert "late sibling" in rendered
    assert "older sibling" not in rendered


def test_conversation_context_prefers_newer_branch_posts_under_cap():
    from plugins.platforms.twitter.adapter import build_conversation_context

    rendered = build_conversation_context(
        [
            {"id": "1", "author_id": "10", "text": "root", "created_at": "2026-01-01"},
            {"id": "2", "author_id": "7", "text": "older bot", "created_at": "2026-01-02", "referenced_tweets": [{"type": "replied_to", "id": "1"}]},
            {"id": "3", "author_id": "7", "text": "newer bot", "created_at": "2026-01-03", "referenced_tweets": [{"type": "replied_to", "id": "1"}]},
            {"id": "4", "author_id": "42", "text": "trigger", "created_at": "2026-01-04", "referenced_tweets": [{"type": "replied_to", "id": "1"}]},
        ],
        trigger_id="4",
        bot_post_ids={"2", "3"},
        max_depth=2,
        max_posts=3,
        siblings_per_parent=1,
    )

    assert "newer bot" in rendered
    assert "older bot" not in rendered


@pytest.mark.asyncio
async def test_conversation_lookup_falls_back_to_direct_parent(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": True})
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(side_effect=RuntimeError("blocked"))
    adapter._client.lookup_posts = AsyncMock(
        return_value={"data": [{"id": "900", "author_id": "7", "text": "bot reply"}]}
    )
    adapter._client.quote_posts = AsyncMock(return_value={})
    adapter.handle_message = AsyncMock()

    await adapter._process_mention(
        {
            "id": "102",
            "author_id": "42",
            "conversation_id": "100",
            "text": "reply",
            "in_reply_to_user_id": "7",
            "referenced_tweets": [{"type": "replied_to", "id": "900"}],
        },
        {"users": [{"id": "42", "username": "alice"}]},
    )

    adapter._client.lookup_posts.assert_awaited_once_with(["900"])
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_denied_mention_does_not_enrich(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(side_effect=AssertionError)
    adapter._client.quote_posts = AsyncMock(side_effect=AssertionError)
    adapter._client.lookup_posts = AsyncMock(side_effect=AssertionError)
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "201",
            "author_id": "99",
            "conversation_id": "200",
            "text": "@bot denied",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )

    adapter._client.conversation_posts.assert_not_awaited()
    adapter._client.quote_posts.assert_not_awaited()
    adapter._client.lookup_posts.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_polling_consumes_all_pages_oldest_first(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": True})
    )
    adapter._account_id = "7"
    adapter._state.mention_since_id = "100"
    adapter._client = Mock()
    adapter._client.mentions = AsyncMock(
        side_effect=[
            {
                "data": [{"id": "102", "author_id": "42", "conversation_id": "90", "text": "second", "entities": {"mentions": [{"id": "7"}]}}],
                "meta": {"next_token": "next"},
            },
            {
                "data": [{"id": "101", "author_id": "42", "conversation_id": "90", "text": "first", "entities": {"mentions": [{"id": "7"}]}}],
                "meta": {},
            },
        ]
    )
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter.handle_message = AsyncMock()

    await adapter._poll_mentions_once()

    assert [call.args[0].message_id for call in adapter.handle_message.await_args_list] == [
        "101",
        "102",
    ]
    assert adapter._state.mention_since_id == "102"


@pytest.mark.asyncio
async def test_mention_retry_keeps_cursor_at_last_terminal_event(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._account_id = "7"
    adapter._state.mention_since_id = "100"
    adapter._client = Mock()
    adapter._client.mentions = AsyncMock(
        return_value={"data": [{"id": "101"}, {"id": "102"}]}
    )
    adapter._process_mention = AsyncMock(side_effect=[None, RuntimeError("retry")])

    with pytest.raises(RuntimeError, match="retry"):
        await adapter._poll_mentions_once()

    assert adapter._state.mention_since_id == "101"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_backfill", "handled"), [(0, []), (1, ["102"])]
)
async def test_mention_first_run_applies_configured_backfill(
    monkeypatch, tmp_path, initial_backfill, handled
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(
            extra={"client_id": "client", "initial_backfill": initial_backfill}
        )
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.mentions = AsyncMock(
        return_value={"data": [{"id": "102"}, {"id": "101"}]}
    )
    adapter._process_mention = AsyncMock()

    await adapter._poll_mentions_once(baseline=True)

    assert [call.args[0]["id"] for call in adapter._process_mention.await_args_list] == handled
    assert adapter._state.mention_since_id == "102"


def test_dm_boundary_persists_event_id_without_pagination_token(monkeypatch, tmp_path):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load()
    state.advance_dms("dm-103")
    state.save()

    restored = TwitterState.load()
    assert restored.dm_last_seen_event_id == "dm-103"
    assert "pagination_token" not in restored.to_dict()


@pytest.mark.asyncio
async def test_client_uses_independent_endpoint_buckets(tmp_path):
    from plugins.platforms.twitter.client import XClient

    client = XClient(token="token")
    client.request = AsyncMock(return_value={})
    try:
        await client.mentions("7")
        await client.dm_events()
        client.request.return_value = {"data": {"id": "700"}}
        await client.create_post("post")
        client.request.return_value = {"data": {"dm_event_id": "701"}}
        await client.send_dm("42-7", "dm")
        client.request.return_value = {"data": {"id": "702"}}
        image = tmp_path / "image.png"
        image.write_bytes(b"image")
        await client.upload_image(image)
        client.request.return_value = {}
        await client.conversation_posts("100")
        await client.bookmarks("7", "list")
    finally:
        await client.close()

    assert [call.kwargs.get("bucket") for call in client.request.await_args_list] == [
        "mentions",
        "direct_messages",
        "public_writes",
        "dm_writes",
        "media_writes",
        "enrichment",
        "tools",
    ]


@pytest.mark.asyncio
async def test_dm_poll_restarts_from_newest_page_after_process_restart(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load()
    state.advance_dms("103")
    state.save()
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(return_value={"data": [{"id": "103"}]})

    await adapter._poll_dms_once()

    assert adapter._state.dm_last_seen_event_id == "103"
    assert adapter._client.dm_events.await_args.kwargs["pagination_token"] == ""


@pytest.mark.asyncio
async def test_dm_page_cap_continues_same_sweep_in_memory(monkeypatch, tmp_path):
    from plugins.platforms.twitter import adapter as adapter_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module, "MAX_PAGES_PER_POLL", 2)
    adapter = adapter_module.TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._state.advance_dms("100")
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(
        side_effect=[
            {"data": [{"id": "106"}], "meta": {"next_token": "one"}},
            {"data": [{"id": "105"}], "meta": {"next_token": "two"}},
            {
                "data": [{"id": "104"}, {"id": "100"}],
                "meta": {},
            },
        ]
    )
    adapter._process_dm = AsyncMock()

    await adapter._poll_dms_once()

    assert adapter._state.dm_last_seen_event_id == "100"
    assert adapter._process_dm.await_args_list == []
    await adapter._poll_dms_once()
    assert adapter._client.dm_events.await_args_list[2].kwargs["pagination_token"] == "two"
    assert adapter._state.dm_last_seen_event_id == "106"
    assert [call.args[0]["id"] for call in adapter._process_dm.await_args_list] == [
        "104",
        "105",
        "106",
    ]


@pytest.mark.asyncio
async def test_dm_capped_sweep_retries_failed_suffix_without_reordering_or_loss(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter import adapter as adapter_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module, "MAX_PAGES_PER_POLL", 1)
    adapter = adapter_module.TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._state.advance_dms("100")
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(
        side_effect=[
            {"data": [{"id": "103"}], "meta": {"next_token": "one"}},
            {
                "data": [{"id": "102"}, {"id": "101"}, {"id": "100"}],
                "meta": {},
            },
        ]
    )
    handled = []
    fail_once = True

    async def process(event, includes):
        nonlocal fail_once
        if event["id"] == "102" and fail_once:
            fail_once = False
            raise RuntimeError("try again")
        handled.append(event["id"])

    adapter._process_dm = process

    await adapter._poll_dms_once()
    assert handled == []

    with pytest.raises(RuntimeError, match="try again"):
        await adapter._poll_dms_once()
    assert handled == ["101"]
    assert adapter._state.dm_last_seen_event_id == "101"

    await adapter._poll_dms_once()

    assert handled == ["101", "102", "103"]
    assert adapter._state.dm_last_seen_event_id == "103"
    assert adapter._client.dm_events.await_count == 2


@pytest.mark.asyncio
async def test_dm_sweep_buffer_is_bounded_by_page_cap(monkeypatch, tmp_path):
    from plugins.platforms.twitter import adapter as adapter_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_module, "MAX_PAGES_PER_POLL", 1)
    adapter = adapter_module.TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._state.advance_dms("100")
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(
        return_value={
            "data": [{"id": str(event_id)} for event_id in range(201, 100, -1)],
            "meta": {"next_token": "next"},
        }
    )

    with pytest.raises(RuntimeError, match="safe event limit"):
        await adapter._dm_pages()

    assert len(adapter._dm_sweep_events) == 100
    assert adapter._state.dm_last_seen_event_id == "100"


@pytest.mark.asyncio
async def test_dm_overlapping_pages_dispatch_each_event_once(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": True})
    )
    adapter._account_id = "7"
    adapter._state.advance_dms("100")
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(
        side_effect=[
            {
                "data": [
                    {
                        "id": "103",
                        "event_type": "MessageCreate",
                        "sender_id": "42",
                        "dm_conversation_id": "42-7",
                    },
                    {
                        "id": "102",
                        "event_type": "MessageCreate",
                        "sender_id": "42",
                        "dm_conversation_id": "42-7",
                    },
                ],
                "meta": {"next_token": "next"},
            },
            {
                "data": [
                    {
                        "id": "102",
                        "event_type": "MessageCreate",
                        "sender_id": "42",
                        "dm_conversation_id": "42-7",
                    },
                    {
                        "id": "101",
                        "event_type": "MessageCreate",
                        "sender_id": "42",
                        "dm_conversation_id": "42-7",
                    },
                    {"id": "100"},
                ],
                "meta": {},
            },
        ]
    )
    adapter.handle_message = AsyncMock()

    await adapter._poll_dms_once()

    assert [call.args[0].message_id for call in adapter.handle_message.await_args_list] == [
        "101",
        "102",
        "103",
    ]
    assert adapter._state.dm_last_seen_event_id == "103"


@pytest.mark.asyncio
async def test_dm_pagination_stops_at_saved_boundary(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._state.dm_last_seen_event_id = "100"
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(
        side_effect=[
            {
                "data": [{"id": "103"}, {"id": "102"}],
                "meta": {"next_token": "next"},
            },
            {
                "data": [{"id": "101"}, {"id": "100"}],
                "meta": {},
            },
        ]
    )

    page = await adapter._dm_pages()

    assert [item["id"] for item in page["data"]] == ["103", "102", "101"]
    assert page["meta"]["complete"] is True
    assert adapter._client.dm_events.await_count == 2


@pytest.mark.asyncio
async def test_dm_backlog_fetches_through_boundary_before_dispatch(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": True})
    )
    adapter._account_id = "7"
    adapter._state.dm_last_seen_event_id = "100"
    pages = []
    for event_id in range(106, 99, -1):
        pages.append(
            {
                "data": [
                    {
                        "id": str(event_id),
                        "event_type": "MessageCreate",
                        "sender_id": "42",
                        "dm_conversation_id": "42-7",
                        "text": str(event_id),
                    }
                ],
                "meta": {"next_token": str(event_id - 1)}
                if event_id > 100
                else {},
            }
        )
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(side_effect=pages)
    adapter.handle_message = AsyncMock()

    await adapter._poll_dms_once()

    assert adapter._client.dm_events.await_count == 7
    assert [
        call.args[0].message_id for call in adapter.handle_message.await_args_list
    ] == ["101", "102", "103", "104", "105", "106"]
    assert adapter._state.dm_last_seen_event_id == "106"


@pytest.mark.asyncio
async def test_dm_pagination_rejects_token_cycles(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._state.dm_last_seen_event_id = "100"
    adapter._client = Mock()
    responses = iter(("one", "two"))

    async def cyclic_page(*, pagination_token=""):
        try:
            next_token = next(responses)
        except StopIteration:
            next_token = "one"
        return {
            "data": [{"id": "101"}],
            "meta": {"next_token": next_token},
        }

    adapter._client.dm_events = cyclic_page

    with pytest.raises(RuntimeError, match="pagination token cycle"):
        await asyncio.wait_for(adapter._dm_pages(), timeout=0.1)
