"""End-to-end broker integration on ephemeral loopback ports.

Real components under test: KeychainGrantStore over the in-memory backend,
AccountSlot with Hermes's production `refresh_codex_oauth_pure` (its token
URL monkeypatched to the fake upstream), the full server app, and real
sockets. Everything synthetic; no non-loopback traffic.
"""

from __future__ import annotations

import asyncio
import json
import time

import aiohttp
from aiohttp import web
import pytest

import hermes_cli.auth as auth_mod
from agent.oauth_broker.account_slot import AccountSlot
from agent.oauth_broker.grant_store import KeychainGrantStore
from agent.oauth_broker.models import OAUTH_GRANT_KEYCHAIN_SERVICE, OAuthGrant
from agent.oauth_broker.server import create_server_app
from tests.agent.oauth_broker.conftest import FakeKeychainBackend
from tests.agent.oauth_broker.test_proxy import SSE_CHUNKS

LOCAL_KEY = "synthetic-integration-client-key"
AUTH = {"Authorization": f"Bearer {LOCAL_KEY}"}

ACCESS = {a: f"synthetic-int-access-{a}" for a in "ABC"}
REFRESH = {a: f"synthetic-int-refresh-{a}" for a in "ABC"}
ACCOUNT_ID = {a: f"acct-int-{a.lower()}" for a in "ABC"}
ROTATED_ACCESS_B = "synthetic-int-access-B2"
ROTATED_REFRESH_B = "synthetic-int-refresh-B2"


class IntegrationUpstream:
    """Fake ChatGPT backend + OAuth token endpoint with per-token scripts."""

    def __init__(self):
        self.requests = []
        self.token_requests = []  # refresh tokens seen by /oauth/token
        self.scripts_by_bearer = {}
        self.rotations = {REFRESH["B"]: (ROTATED_ACCESS_B, ROTATED_REFRESH_B)}
        self.terminal_refresh_tokens = {REFRESH["C"]}
        self.streaming_started = asyncio.Event()
        self.client_disconnected = asyncio.Event()
        self._runner = None
        self.origin = None

    def _bearer(self, request):
        return request.headers.get("Authorization", "").removeprefix("Bearer ")

    async def _handle_responses(self, request):
        body = await request.read()
        bearer = self._bearer(request)
        self.requests.append(
            {
                "path": request.path,
                "bearer": bearer,
                "account_id": request.headers.get("ChatGPT-Account-Id"),
                "body": body,
            }
        )
        script = self.scripts_by_bearer.get(bearer)
        behavior = script.pop(0) if script else ("sse",)
        if behavior[0] == "status":
            _, status, headers, payload = behavior
            return web.Response(
                status=status,
                headers=headers,
                body=payload,
                content_type="application/json",
            )
        if behavior[0] == "sse-infinite":
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            self.streaming_started.set()
            try:
                while True:
                    await resp.write(b"data: synthetic-tick\n\n")
                    await asyncio.sleep(0.01)
            except (ConnectionResetError, asyncio.CancelledError):
                self.client_disconnected.set()
                raise
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await resp.prepare(request)
        for chunk in SSE_CHUNKS:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def _handle_usage(self, request):
        bearer = self._bearer(request)
        self.requests.append({"path": request.path, "bearer": bearer})
        return web.json_response(
            {"plan_type": "synthetic-plan", "rate_limit": {}}
        )

    async def _handle_token(self, request):
        form = await request.post()
        refresh_token = form.get("refresh_token", "")
        self.token_requests.append(refresh_token)
        if refresh_token in self.terminal_refresh_tokens:
            return web.json_response(
                {
                    "error": "invalid_grant",
                    "error_description": "synthetic terminal refresh",
                },
                status=400,
            )
        rotated = self.rotations.get(refresh_token)
        if rotated is None:
            return web.json_response(
                {"error": "invalid_request"}, status=400
            )
        access, new_refresh = rotated
        return web.json_response(
            {"access_token": access, "refresh_token": new_refresh}
        )

    async def start(self):
        app = web.Application()
        app.router.add_post("/backend-api/codex/responses", self._handle_responses)
        app.router.add_get("/backend-api/wham/usage", self._handle_usage)
        app.router.add_post("/oauth/token", self._handle_token)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        self.origin = f"http://{host}:{port}"

    async def stop(self):
        if self._runner is not None:
            await self._runner.cleanup()


def _seed_keychain():
    backend = FakeKeychainBackend()
    now = time.time()
    for alias in "ABC":
        expires = now - 10 if alias == "C" else now + 3600
        grant = OAuthGrant(
            access_token=ACCESS[alias],
            refresh_token=REFRESH[alias],
            expires_at=expires,
            account_id=ACCOUNT_ID[alias],
        )
        KeychainGrantStore(backend=backend).replace(alias, grant)
    return backend


def _build_slots(backend, tmp_path, tag):
    store = KeychainGrantStore(backend=backend)
    return {
        alias: AccountSlot(
            alias, grant_store=store, state_dir=tmp_path / f"state-{tag}"
        )
        for alias in "ABC"
    }


async def _start_broker(backend, upstream, tmp_path, tag):
    app = create_server_app(
        slots=_build_slots(backend, tmp_path, tag),
        local_key=LOCAL_KEY,
        upstream_origin=upstream.origin,
        allow_test_upstream=True,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


def test_full_broker_lifecycle_against_fake_upstream(tmp_path, monkeypatch):
    async def scenario():
        upstream = IntegrationUpstream()
        await upstream.start()
        monkeypatch.setattr(
            auth_mod, "CODEX_OAUTH_TOKEN_URL", f"{upstream.origin}/oauth/token"
        )
        backend = _seed_keychain()
        runner, base = await _start_broker(backend, upstream, tmp_path, "first")
        session = aiohttp.ClientSession()
        try:
            # Phase 1 — A/B/C stream concurrently, each with its own identity.
            async def stream(alias):
                resp = await session.post(
                    f"{base}/accounts/{alias}/backend-api/codex/responses",
                    data=json.dumps({"alias": alias}).encode(),
                    headers=AUTH,
                )
                assert resp.status == 200, alias
                return await resp.read()

            bodies = await asyncio.gather(*(stream(a) for a in "AB"))
            assert all(body == b"".join(SSE_CHUNKS) for body in bodies)
            seen = {
                r["bearer"]: r["account_id"]
                for r in upstream.requests
                if r["path"].endswith("/responses")
            }
            assert seen == {
                ACCESS["A"]: ACCOUNT_ID["A"],
                ACCESS["B"]: ACCOUNT_ID["B"],
            }
            assert upstream.token_requests == []  # fresh grants: no refresh

            # Phase 2 — usage passthrough.
            resp = await session.get(
                f"{base}/accounts/A/backend-api/wham/usage", headers=AUTH
            )
            assert resp.status == 200
            assert (await resp.json())["plan_type"] == "synthetic-plan"

            # Phase 2b — integration-boundary proof: the URL Hermes really
            # derives from a migrated broker base_url (account_usage.py
            # strips '/codex' before appending '/wham/usage') must land on
            # the broker's whitelisted usage route, and the Responses path
            # must be exactly base_url + '/responses'.
            from agent.account_usage import _resolve_codex_usage_url

            migrated_base_url = f"{base}/accounts/A/backend-api/codex"
            derived = _resolve_codex_usage_url(migrated_base_url)
            assert derived == f"{base}/accounts/A/backend-api/wham/usage"
            resp = await session.get(derived, headers=AUTH)
            assert resp.status == 200
            assert (await resp.json())["plan_type"] == "synthetic-plan"
            assert (
                f"{migrated_base_url}/responses"
                == f"{base}/accounts/A/backend-api/codex/responses"
            )

            # Phase 3 — A hits a 429: propagated verbatim, no refresh, and
            # no broker-side account switching.
            upstream.scripts_by_bearer[ACCESS["A"]] = [
                ("status", 429, {"Retry-After": "600"}, b'{"detail":"limited"}')
            ]
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 429
            assert resp.headers["Retry-After"] == "600"
            assert upstream.token_requests == []

            # Phase 4 — B gets a 401, the broker force-refreshes through the
            # REAL refresh implementation against the fake token endpoint,
            # replays once, and persists the rotated grant to the Keychain.
            upstream.scripts_by_bearer[ACCESS["B"]] = [
                ("status", 401, {}, b'{"detail":"Unauthorized"}')
            ]
            resp = await session.post(
                f"{base}/accounts/B/backend-api/codex/responses",
                data=b'{"replay":"b"}',
                headers=AUTH,
            )
            assert resp.status == 200
            assert await resp.read() == b"".join(SSE_CHUNKS)
            assert upstream.token_requests == [REFRESH["B"]]  # only B refreshed
            replayed = [
                r
                for r in upstream.requests
                if r.get("bearer") == ROTATED_ACCESS_B
            ]
            assert replayed and replayed[0]["body"] == b'{"replay":"b"}'
            stored_b = backend.items[(OAUTH_GRANT_KEYCHAIN_SERVICE, "B")]
            assert ROTATED_REFRESH_B in stored_b  # rotation persisted

            # Phase 5 — C's expired grant meets a terminal invalid_grant:
            # 503 fail-closed, exactly one token-endpoint attempt, no loop.
            for _ in range(2):
                resp = await session.post(
                    f"{base}/accounts/C/backend-api/codex/responses",
                    data=b"{}",
                    headers=AUTH,
                )
                assert resp.status == 503
                assert "invalid_grant" in await resp.text()
            assert upstream.token_requests == [REFRESH["B"], REFRESH["C"]]

            # Phase 6 — wrong client key is rejected before any upstream IO.
            upstream_count = len(upstream.requests)
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers={"Authorization": "Bearer synthetic-wrong"},
            )
            assert resp.status == 401
            assert len(upstream.requests) == upstream_count

            # Phase 7 — detailed health: A/B healthy, C terminal; no secrets.
            resp = await session.get(f"{base}/health/detailed", headers=AUTH)
            raw = await resp.text()
            health = {e["alias"]: e for e in json.loads(raw)["accounts"]}
            assert health["A"]["healthy"] is True
            assert health["B"]["healthy"] is True
            assert health["C"]["healthy"] is False
            for secret in (*ACCESS.values(), *REFRESH.values(), ROTATED_ACCESS_B):
                assert secret not in raw

            # Phase 8 — cancellation propagates through the broker.
            upstream.scripts_by_bearer[ACCESS["A"]] = [("sse-infinite",)]
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            await asyncio.wait_for(upstream.streaming_started.wait(), timeout=2)
            await resp.content.read(10)
            resp.close()
            await asyncio.wait_for(upstream.client_disconnected.wait(), timeout=2)
        finally:
            await runner.cleanup()

        # Phase 9 — broker stopped: fail closed, nothing answers.
        with pytest.raises(aiohttp.ClientConnectorError):
            await session.get(f"{base}/health")

        # Phase 10 — restart with fresh slots from the same Keychain: B's
        # rotated grant is recovered and used immediately.
        runner2, base2 = await _start_broker(backend, upstream, tmp_path, "second")
        try:
            resp = await session.post(
                f"{base2}/accounts/B/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 200
            await resp.read()
            assert upstream.requests[-1]["bearer"] == ROTATED_ACCESS_B
            assert upstream.token_requests == [REFRESH["B"], REFRESH["C"]]
        finally:
            await session.close()
            await runner2.cleanup()
            await upstream.stop()

    asyncio.run(scenario())
