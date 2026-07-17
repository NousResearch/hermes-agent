# Twitter/X Platform Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a general-purpose Twitter/X gateway plugin to Hermes that safely handles OAuth 2.0 PKCE, public mentions, DMs, branch-aware conversations, media, cron delivery, bookmarks, and metrics.

**Architecture:** Keep every Twitter-specific production change under plugins/platforms/twitter. Reuse Hermes BasePlatformAdapter, plugin registration, scoped credential locks, profile-aware paths, atomic JSON writes, authorization, media cache, cron sender, and tool registry. Use the already-installed httpx client and Python standard library; no X SDK or new dependency.

**Tech Stack:** Python 3.11-3.13, asyncio, httpx 0.28.1, Hermes plugin/platform APIs, pytest 9, unittest.mock.

## Global Constraints

- Work from current upstream main and salvage behavior from PR #12352 without merging it.
- The upstream branch is general-purpose and contains no WebCMD, heyWebcmd, agentrhq, or product-specific policy.
- Keep Twitter-specific production code under plugins/platforms/twitter.
- Register through ctx.register_platform; do not add a core platform enum branch.
- Support OAuth 2.0 Authorization Code with S256 PKCE only.
- Do not claim app-only bearer-token or OAuth 1.0a support.
- Resolve get_hermes_home at operation time; never cache a profile path globally.
- Keep every X identifier as a string end-to-end.
- Enforce allowed_users or allow_all_users before enrichment, media download, or dispatch.
- Treat post text, DMs, profiles, metrics, and media metadata as untrusted user context.
- Do not make live X API calls or require real credentials in tests.
- Reuse httpx; add no dependency.
- Run tests only through scripts/run_tests.sh.

---

### Task 1: Register the plugin and validate configuration

**Files:**
- Create: plugins/platforms/twitter/__init__.py
- Create: plugins/platforms/twitter/plugin.yaml
- Create: plugins/platforms/twitter/adapter.py
- Create: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: PluginContext.register_platform and gateway.config.PlatformConfig.
- Produces: TwitterAdapter, TwitterSettings, check_requirements, validate_config, is_connected, apply_yaml_config, and register.

- [ ] **Step 1: Write failing registration and settings tests**

Add tests that import the plugin through its real register entrypoint and assert the complete platform contract:

~~~python
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.twitter import register
from plugins.platforms.twitter.adapter import TwitterAdapter, TwitterSettings


def test_registers_twitter_platform():
    ctx = Mock()
    register(ctx)
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "twitter"
    assert kwargs["label"] == "Twitter / X"
    assert kwargs["allowed_users_env"] == "TWITTER_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "TWITTER_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "TWITTER_HOME_CHANNEL"
    assert kwargs["max_message_length"] == 280
    assert callable(kwargs["standalone_sender_fn"])


def test_settings_reject_unsafe_limits():
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        TwitterSettings.from_config(
            PlatformConfig(extra={"client_id": "client", "poll_interval_seconds": 0})
        )


def test_adapter_send_signature_matches_base():
    import inspect
    from gateway.platforms.base import BasePlatformAdapter

    assert inspect.signature(TwitterAdapter.send) == inspect.signature(BasePlatformAdapter.send)
~~~

- [ ] **Step 2: Run the tests and verify the plugin is absent**

Run:

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q
~~~

Expected: collection fails because plugins.platforms.twitter does not exist.

- [ ] **Step 3: Add the manifest, entrypoint, and validated settings**

The manifest declares kind platform, no secret token env values, and config/setup prompts for client ID, redirect URI, allowlist, home channel, polling, branch limits, media caps, and queue caps.

The adapter module starts with:

~~~python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter

MAX_MESSAGE_LENGTH = 280


@dataclass(frozen=True)
class TwitterSettings:
    client_id: str
    redirect_uri: str = "http://127.0.0.1:8765/callback"
    poll_interval_seconds: float = 30.0
    initial_backfill: int = 0
    max_depth: int = 8
    max_posts: int = 40
    siblings_per_parent: int = 5
    max_download_bytes: int = 10_485_760
    max_upload_bytes: int = 5_242_880
    max_pending: int = 100
    max_wait_seconds: float = 900.0

    @classmethod
    def from_config(cls, config: PlatformConfig) -> "TwitterSettings":
        extra = config.extra or {}
        settings = cls(
            client_id=str(extra.get("client_id", "")).strip(),
            redirect_uri=str(extra.get("redirect_uri", cls.redirect_uri)).strip(),
            poll_interval_seconds=float(extra.get("poll_interval_seconds", 30)),
            initial_backfill=int(extra.get("initial_backfill", 0)),
            max_depth=int((extra.get("conversation") or {}).get("max_depth", 8)),
            max_posts=int((extra.get("conversation") or {}).get("max_posts", 40)),
            siblings_per_parent=int((extra.get("conversation") or {}).get("siblings_per_parent", 5)),
            max_download_bytes=int((extra.get("media") or {}).get("max_download_bytes", 10_485_760)),
            max_upload_bytes=int((extra.get("media") or {}).get("max_upload_bytes", 5_242_880)),
            max_pending=int((extra.get("queue") or {}).get("max_pending", 100)),
            max_wait_seconds=float((extra.get("queue") or {}).get("max_wait_seconds", 900)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.client_id:
            raise ValueError("twitter.client_id is required")
        if self.poll_interval_seconds <= 0:
            raise ValueError("twitter.poll_interval_seconds must be positive")
        if not 0 <= self.initial_backfill <= 100:
            raise ValueError("twitter.initial_backfill must be between 0 and 100")
        for name in ("max_depth", "max_posts", "siblings_per_parent", "max_download_bytes",
                     "max_upload_bytes", "max_pending"):
            if getattr(self, name) <= 0:
                raise ValueError(f"twitter.{name} must be positive")
        if self.max_wait_seconds <= 0:
            raise ValueError("twitter.queue.max_wait_seconds must be positive")


class TwitterAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("twitter"))
        self.settings = TwitterSettings.from_config(config)
~~~

register(ctx) must pass apply_yaml_config_fn, setup_fn, is_connected, standalone_sender_fn, authorization env names, cron env name, a 280-character limit, and a concise platform_hint.

- [ ] **Step 4: Run registration tests**

Run:

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q
~~~

Expected: registration, validation, and signature tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add plugins/platforms/twitter tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): register platform plugin"
~~~

---

### Task 2: Implement profile-safe OAuth 2.0 PKCE

**Files:**
- Create: plugins/platforms/twitter/oauth.py
- Modify: plugins/platforms/twitter/adapter.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: hermes_constants.get_hermes_home, utils.atomic_json_write, httpx.AsyncClient.
- Produces: OAuthTokens, token_path, create_pkce_pair, build_authorization_url, exchange_code, refresh_tokens, load_tokens, save_tokens, and interactive_setup.

- [ ] **Step 1: Add failing OAuth and profile-isolation tests**

~~~python
import base64
import hashlib
import json
import stat

from plugins.platforms.twitter.oauth import (
    create_s256_challenge,
    load_tokens,
    save_tokens,
    token_path,
)


def test_s256_challenge_matches_rfc_vector():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert create_s256_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_tokens_follow_active_hermes_home(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("HERMES_HOME", str(first))
    save_tokens({"access_token": "one", "refresh_token": "r1"})
    assert token_path() == first / "twitter" / "oauth2.json"
    monkeypatch.setenv("HERMES_HOME", str(second))
    assert load_tokens() is None
    save_tokens({"access_token": "two", "refresh_token": "r2"})
    assert json.loads((second / "twitter" / "oauth2.json").read_text())["access_token"] == "two"
    assert stat.S_IMODE((second / "twitter" / "oauth2.json").stat().st_mode) == 0o600
~~~

- [ ] **Step 2: Run focused OAuth tests**

Run:

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k oauth
~~~

Expected: import failure for oauth.py.

- [ ] **Step 3: Implement PKCE and token persistence**

Use secrets.token_urlsafe for verifier and state, hashlib.sha256 plus base64.urlsafe_b64encode with stripped padding for S256, urllib.parse.urlencode for the authorization URL, and atomic_json_write with mode 0o600. token_path must call get_hermes_home each time.

OAuthTokens stores access_token, refresh_token, expires_at, scopes, client_id, user_id, and username. load_tokens rejects malformed records without deleting them. save_tokens writes only this validated shape.

- [ ] **Step 4: Implement bounded callback, exchange, and refresh**

Use asyncio.start_server bound only to the configured loopback host. Validate the callback path and constant-time compare state with secrets.compare_digest. Bound acceptance with asyncio.timeout. Never log the code, verifier, access token, or refresh token.

Token exchange and refresh send form-encoded POST requests to https://api.x.com/2/oauth2/token. Serialize refresh using one asyncio.Lock per OAuthClient instance and persist rotated refresh tokens before returning.

- [ ] **Step 5: Run OAuth tests**

Run:

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "pkce or oauth or token or callback or profile"
~~~

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add plugins/platforms/twitter/oauth.py plugins/platforms/twitter/adapter.py tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): add profile-safe PKCE authentication"
~~~

---

### Task 3: Persist cursors, deduplication, and participation branches

**Files:**
- Create: plugins/platforms/twitter/state.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: get_hermes_home and atomic_json_write.
- Produces: TwitterState.load, TwitterState.save, seen, mark_seen, resolve_anchor, map_bot_post, advance_mentions, and advance_dms.

- [ ] **Step 1: Add failing state tests**

~~~python
from plugins.platforms.twitter.state import TwitterState


def test_branch_anchor_follows_mapped_bot_ancestor(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=100, max_branches=100)
    state.map_bot_post("9007199254740993", "123")
    assert state.resolve_anchor("456", ["42", "9007199254740993"]) == "123"
    assert state.resolve_anchor("789", ["42"]) == "789"


def test_state_survives_restart_and_profile_switch(monkeypatch, tmp_path):
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
~~~

- [ ] **Step 2: Run state tests and verify failure**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k state
~~~

Expected: import failure for state.py.

- [ ] **Step 3: Implement bounded JSON state**

Store state at get_hermes_home()/twitter/state.json. Use ordered dictionaries serialized as lists to bound seen event IDs and bot-post mappings. Quarantine corrupt state by renaming it with a .corrupt timestamp suffix, warn without content, and start empty. State contains version, mention_since_id, dm_since_id, seen_ids, and bot_post_anchors.

resolve_anchor(trigger_id, ancestor_ids) returns the first mapped ancestor anchor, scanning nearest parent first, or trigger_id when no mapping exists.

- [ ] **Step 4: Run state tests**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "state or branch or cursor or dedup"
~~~

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add plugins/platforms/twitter/state.py tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): persist polling and branch state"
~~~

---

### Task 4: Build the rate-aware X API client

**Files:**
- Create: plugins/platforms/twitter/queue.py
- Create: plugins/platforms/twitter/client.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: OAuth token refresh and httpx.AsyncClient.
- Produces: RateQueue, XApiError, AmbiguousWriteError, XClient.request, identity, mentions, dm_events, create_post, send_dm, conversation_posts, lookup_posts, bookmarks, post_metrics, and upload_image.

- [ ] **Step 1: Add failing request and queue tests**

~~~python
import asyncio
import httpx
import pytest

from plugins.platforms.twitter.client import AmbiguousWriteError, XClient
from plugins.platforms.twitter.queue import RateQueue


@pytest.mark.asyncio
async def test_write_timeout_is_not_retried():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("uncertain", request=request)

    client = XClient.for_test(
        token="token",
        transport=httpx.MockTransport(handler),
        max_pending=2,
        max_wait_seconds=1,
    )
    with pytest.raises(AmbiguousWriteError):
        await client.create_post("hello", reply_to="123")
    assert calls == 1


@pytest.mark.asyncio
async def test_queue_overflow_fails_before_network():
    queue = RateQueue(max_pending=1, max_wait_seconds=1)
    blocker = asyncio.Event()
    first = asyncio.create_task(queue.run("write", lambda: blocker.wait()))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="queue is full"):
        await queue.run("write", lambda: asyncio.sleep(0))
    blocker.set()
    await first
~~~

- [ ] **Step 2: Run client tests and verify failure**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "queue or client or request"
~~~

Expected: import failure for queue.py or client.py.

- [ ] **Step 3: Implement bounded per-bucket queues**

RateQueue owns one asyncio.Lock and bounded waiter count per bucket. queue.run records enqueue monotonic time, rejects max_pending overflow, applies asyncio.timeout(max_wait_seconds), and never executes a timed-out queued item later. Separate buckets are read, write_post, write_dm, media, and optional_enrichment.

Parse Retry-After and x-rate-limit-reset into a monotonic delay. Retry only requests known not to have been processed: reads and explicit HTTP 429 responses. Never retry create-post or send-DM after a transport timeout, connection loss after send, or 5xx response with an indeterminate body.

- [ ] **Step 4: Implement current X API v2 methods**

Use these exact routes:

- GET /2/users/me
- GET /2/users/{id}/mentions
- GET /2/dm_events
- GET /2/tweets/search/recent with query conversation_id:{id}
- GET /2/tweets with ids
- POST /2/tweets
- POST /2/dm_conversations/{dm_conversation_id}/messages
- GET, POST, DELETE /2/users/{authenticated_user_id}/bookmarks
- POST /2/media/upload for simple image upload

For POST /2/tweets use reply.in_reply_to_tweet_id and media.media_ids. For DMs use attachments as objects containing media_id. Sanitize API errors to status, endpoint class, X error type/title, and at most 200 characters of detail.

- [ ] **Step 5: Run client tests**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "queue or client or rate or request or media or bookmark or metrics"
~~~

Expected: all selected tests pass with no real sleeps or network.

- [ ] **Step 6: Commit**

~~~bash
git add plugins/platforms/twitter/queue.py plugins/platforms/twitter/client.py tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): add rate-aware X API client"
~~~

---

### Task 5: Implement mention and DM ingestion

**Files:**
- Modify: plugins/platforms/twitter/adapter.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: XClient, TwitterState, BasePlatformAdapter.build_source, handle_message, and common authorization.
- Produces: connect, disconnect, supervised pollers, process_mention, process_dm, and normalized MessageEvent instances.

- [ ] **Step 1: Add failing ingestion tests**

Test the approved scenario with Alice root T0, Bob mention T1, bot reply T2, Carol direct reply T3, and sibling T2b. Assert T1 anchors a branch, T3 joins it only when Carol is authorized, and T2b remains context-only unless it explicitly mentions the bot.

Add DM fixtures asserting chat_id dm:<dm_conversation_id>, message_id equal to the DM event ID, source.user_id equal to sender_id, and outbound bot-authored events ignored.

Add an early-authorization assertion by making conversation lookup and media download mocks raise if called for a denied user.

- [ ] **Step 2: Run ingestion tests and verify failure**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "mention or dm or authorization or branch"
~~~

Expected: TwitterAdapter ingestion methods are missing.

- [ ] **Step 3: Implement lifecycle and supervision**

connect loads OAuth and state, acquires _acquire_platform_lock("twitter-oauth-account", user_id, "Twitter account"), verifies /2/users/me, establishes first-run baselines, starts mention and DM tasks, and calls _mark_connected only after the initial poll succeeds.

Each task done callback ignores intentional cancellation. An unexpected exit calls _set_fatal_error with a retryable code and awaits _notify_fatal_error. disconnect cancels and awaits pollers, closes XClient, saves state, releases the platform lock, and calls _mark_disconnected.

- [ ] **Step 4: Implement trigger and authorization rules**

A public post triggers only when structured mention entities contain the authenticated account, in_reply_to_user_id equals the authenticated account, or a structured quoted-post reference explicitly summons a bot-authored post. Never use substring matching.

Parse the minimum event identity first. Call the shared authorization decision before parent lookup, conversation search, profile fetch, or media download. Build chat IDs as tweet:<conversation_id>:<anchor_id> and DMs as dm:<dm_conversation_id>.

- [ ] **Step 5: Implement cursor-safe polling**

Process pages oldest-first. Advance and persist a cursor after safe dispatch, deliberate ignore, or permanent classification. Leave the cursor before a retryable failure. On first startup with initial_backfill zero, record the newest ID without dispatch; otherwise dispatch at most the configured newest count in chronological order.

- [ ] **Step 6: Run ingestion tests**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "mention or dm or authorization or branch or cursor or lifecycle"
~~~

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

~~~bash
git add plugins/platforms/twitter/adapter.py tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): ingest mentions and direct messages"
~~~

---

### Task 6: Add bounded conversation and media enrichment

**Files:**
- Modify: plugins/platforms/twitter/adapter.py
- Modify: plugins/platforms/twitter/client.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: recent conversation search, direct post lookup, included users/media, and BasePlatformAdapter media-cache helpers.
- Produces: build_conversation_context and download_inbound_images.

- [ ] **Step 1: Add failing deterministic tree tests**

Build unordered fixture posts and assert the rendered context prioritizes the ancestor spine, mapped bot branch, direct replies to bot posts, then capped siblings. Assert max_depth, siblings_per_parent, and max_posts are all enforced and final rendering is chronological.

Add a search-403 fixture that falls back to direct parent lookup and still dispatches the trigger. Add denied-user and oversized-media fixtures proving no download occurs.

- [ ] **Step 2: Run enrichment tests and verify failure**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "enrichment or tree or context or inbound_media"
~~~

Expected: enrichment helpers are missing.

- [ ] **Step 3: Implement deterministic context selection**

Build an ID-indexed post map and parent map. Select ancestors first, active-branch bot posts and their direct replies second, and siblings last. Deduplicate IDs. Within each category prefer recent posts, cap, then render selected posts by created_at and ID.

Only the trigger text becomes MessageEvent.text. Render all enrichment into channel_context with explicit labels that profiles and posts are untrusted background.

- [ ] **Step 4: Implement bounded inbound image handling**

Accept HTTPS image media only after authorization. Enforce attachment count, Content-Length when present, streamed byte count, timeout, and supported image MIME. Reuse the existing cache_image_from_bytes path. Keep unsupported video/GIF as descriptive metadata and safe preview URLs.

- [ ] **Step 5: Run enrichment tests**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "enrichment or tree or context or inbound_media"
~~~

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add plugins/platforms/twitter/adapter.py plugins/platforms/twitter/client.py tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): enrich branch context safely"
~~~

---

### Task 7: Implement typed sending, image upload, cron, and plugin tools

**Files:**
- Create: plugins/platforms/twitter/tools.py
- Modify: plugins/platforms/twitter/__init__.py
- Modify: plugins/platforms/twitter/adapter.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: XClient create/send/upload/bookmark/metrics methods and BasePlatformAdapter media path validation.
- Produces: TwitterAdapter.send, standalone_send, twitter_bookmarks, twitter_post_metrics, and plugin tool registration.

- [ ] **Step 1: Add failing send and tool tests**

Test timeline, tweet:<conversation_id>:<anchor_id>, and dm:<conversation_id> routes. Assert bare numeric chat IDs fail. Assert IDs above 2**53 remain exact strings. Assert queued send success is returned only after the fake API returns an ID.

Test one DM image and up to four post images; reject mismatched MIME/extension, oversized files, non-files, and credential-path files before upload. Assert partial upload failure prevents text-only creation.

Test tool schemas and handlers for bookmark list/add/remove and bounded metrics IDs. Assert tool check_fn is false without a usable profile token.

- [ ] **Step 2: Run send/tool tests and verify failure**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "send or route or upload or bookmark or metrics or standalone"
~~~

Expected: send and tool handlers are absent.

- [ ] **Step 3: Implement exact send routing**

send has the exact BasePlatformAdapter signature. timeline creates a top-level post. tweet routes create a post and use reply_to exactly when supplied. dm routes send to the existing DM conversation and do not send reply_to. Return SendResult success only with confirmed post or event ID. Convert XApiError into classified SendResult failures; mark ambiguous writes as non-retryable uncertain failures.

- [ ] **Step 4: Implement outbound images**

Use BasePlatformAdapter.validate_media_delivery_path, pathlib.Path.is_file, mimetypes, Pillow verification, and byte caps. Upload images with media_category tweet_image or dm_image. Keep all IDs as strings. Do not downgrade to text-only when any requested upload fails.

- [ ] **Step 5: Implement standalone sender and tools**

standalone_send resolves the requested profile at call time, loads OAuth/settings, creates a fresh XClient, sends once, and closes in finally.

Register twitter_bookmarks and twitter_post_metrics through ctx.register_tool with check_fn that loads usable profile OAuth. The handlers validate operation names and numeric-string IDs before calling XClient. Keep the tools in a twitter toolset and do not modify core toolsets.py.

- [ ] **Step 6: Run send/tool tests**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py -q -k "send or route or upload or bookmark or metrics or standalone"
~~~

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

~~~bash
git add plugins/platforms/twitter tests/plugins/test_twitter_platform.py
git commit -m "feat(twitter): add delivery and service-gated tools"
~~~

---

### Task 8: Prove the real plugin path with a temporary Hermes home

**Files:**
- Create: tests/plugins/test_twitter_platform_integration.py
- Modify: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: real plugin loader, registered adapter factory, production OAuth/state paths, fake httpx transport, and standalone sender.
- Produces: end-to-end regression coverage without credentials or live network.

- [ ] **Step 1: Write the integration test**

The test must:

1. Set HERMES_HOME to a temporary directory.
2. Load the real bundled Twitter plugin.
3. Save fake OAuth through production persistence.
4. Create the adapter from the platform registry.
5. Connect using an injected httpx.MockTransport.
6. Poll one mention and one DM.
7. Observe normalized gateway events.
8. Send public and DM replies.
9. Disconnect and construct a second adapter under the same home.
10. Prove cursor and dedup state survives.
11. Switch HERMES_HOME and prove no token/state leakage.
12. Invoke standalone_send and assert a fresh client is closed.

- [ ] **Step 2: Run integration test and verify failure**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform_integration.py -q
~~~

Expected: the first missing test injection seam or lifecycle defect fails with a concrete assertion.

- [ ] **Step 3: Add only the minimal injection seams needed by the test**

Permit adapter_factory or config.extra to receive an httpx transport only when supplied programmatically. Do not expose this in user YAML or environment variables. Keep production defaults unchanged.

- [ ] **Step 4: Run all Twitter tests**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py tests/plugins/test_twitter_platform_integration.py -q
~~~

Expected: all Twitter tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add plugins/platforms/twitter tests/plugins/test_twitter_platform.py tests/plugins/test_twitter_platform_integration.py
git commit -m "test(twitter): cover profile-isolated gateway flow"
~~~

---

### Task 9: Document setup and verify the upstream contribution

**Files:**
- Create: website/docs/user-guide/twitter.md
- Modify: plugins/platforms/twitter/plugin.yaml
- Test: tests/plugins/test_twitter_platform.py

**Interfaces:**
- Consumes: implemented setup/config behavior.
- Produces: user-facing setup, scope, security, routing, and operational documentation.

- [ ] **Step 1: Write documentation matching implemented behavior**

Document developer app creation, loopback redirect URI, exact OAuth scopes, hermes gateway setup, numeric allowed user IDs, fail-closed default, mention triggers, public branch chat IDs, DM routes, timeline cron destination, media caps, initial backfill, rate limits, uncertain delivery, and explicit lack of OAuth 1.0a/app-only support.

- [ ] **Step 2: Run focused tests and static checks**

~~~bash
scripts/run_tests.sh tests/plugins/test_twitter_platform.py tests/plugins/test_twitter_platform_integration.py tests/hermes_cli/test_gateway_platform_gating.py tests/providers/test_plugin_discovery.py -q
.venv/bin/ruff check plugins/platforms/twitter tests/plugins/test_twitter_platform.py tests/plugins/test_twitter_platform_integration.py
git diff --check
~~~

Expected: all tests pass, ruff exits zero, and git diff --check prints nothing.

- [ ] **Step 3: Run the broader gateway and plugin regression set**

~~~bash
scripts/run_tests.sh tests/gateway/ tests/hermes_cli/test_gateway_platform_gating.py tests/hermes_cli/test_plugins.py tests/providers/test_plugin_discovery.py -q
~~~

Expected: all selected files pass. If an unrelated pre-existing failure appears, record its exact file and rerun the Twitter suite before proceeding.

- [ ] **Step 4: Commit documentation**

~~~bash
git add website/docs/user-guide/twitter.md plugins/platforms/twitter/plugin.yaml
git commit -m "docs(twitter): document platform setup and safety"
~~~

- [ ] **Step 5: Push the Step 1 branch and open the upstream PR**

~~~bash
git push -u fork feat/twitter-platform-plugin
gh pr create --repo NousResearch/hermes-agent --head ankitranjan7:feat/twitter-platform-plugin --base main --title "feat: add Twitter/X platform plugin" --body-file /tmp/twitter-pr-body.md
~~~

The PR body must explain the real problem, why PR #12352 was salvaged instead of merged, the current plugin architecture, OAuth/profile/security fixes, exact test commands and results, and that Step 2 WebCMD-specific behavior is intentionally excluded.

---

### Task 10: Start Step 2 from the immutable Step 1 boundary

**Files:**
- No Step 1 production files change in this task.
- Create later in the Step 2 branch: docs/superpowers/specs/2026-07-17-webcmd-twitter-integration-design.md

**Interfaces:**
- Consumes: the verified feat/twitter-platform-plugin tip.
- Produces: a separate fork-only branch for WebCMD-specific behavior.

- [ ] **Step 1: Record the Step 1 tip and create the Step 2 branch**

~~~bash
git status --short
git rev-parse HEAD
git switch -c feat/webcmd-twitter-agent
~~~

Expected: the worktree is clean before branch creation, and feat/webcmd-twitter-agent points at the exact verified Step 1 commit.

- [ ] **Step 2: Design Step 2 before code**

Trace the current codex/policy-only-adapter-pr implementation again, then write a separate specification for live WebCMD catalog routing, anonymous read execution, missing-adapter authoring, direct PR publication, GitHub reconciliation, durable X conversation policy, and final reply presentation. Keep those behaviors outside the upstream Step 1 PR.
