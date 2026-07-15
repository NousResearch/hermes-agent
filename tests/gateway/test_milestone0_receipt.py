"""Milestone-0 end-to-end contract receipt.

Drives the real shipped operator-enrollment, scope-authorization, and scoped
profile handlers through the full lifecycle exactly as a paired Android client
would: mint a one-time enrollment -> inspect -> exchange for a scoped token ->
exercise profile CRUD + soul with If-Match optimistic concurrency -> revoke the
credential -> confirm the revoked token is refused. This is an auditable receipt
that the milestone-0 server contract works as an integrated whole, not just as
per-handler unit tests.

Run: uv run --extra dev pytest tests/gateway/test_milestone0_receipt.py -q -s
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.api_operator_auth import OperatorCredentialStore
from gateway.api_operator_enrollment import OperatorEnrollmentStore
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _receipt_app(adapter):
    app = web.Application()
    app.router.add_post(
        "/v1/operator/enrollments/inspect", adapter._handle_inspect_enrollment
    )
    app.router.add_post(
        "/v1/operator/enrollments/exchange", adapter._handle_exchange_enrollment
    )
    app.router.add_get("/v1/operator/credentials", adapter._handle_list_operator_credentials)
    app.router.add_delete(
        "/v1/operator/credentials/{credential_id}",
        adapter._handle_revoke_operator_credential,
    )
    app.router.add_get("/api/profiles", adapter._handle_list_profiles)
    app.router.add_post("/api/profiles", adapter._handle_create_profile)
    app.router.add_patch("/api/profiles/{name}", adapter._handle_patch_profile)
    app.router.add_delete("/api/profiles/{name}", adapter._handle_delete_profile)
    app.router.add_get(
        "/api/profiles/{name}/soul", adapter._handle_get_profile_soul
    )
    app.router.add_put(
        "/api/profiles/{name}/soul", adapter._handle_put_profile_soul
    )
    return app


@pytest.mark.asyncio
async def test_milestone0_full_lifecycle_receipt(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    origin = "https://hermes.receipt.example"

    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-super"}))
    adapter._operator_credentials = credentials
    adapter._operator_enrollments = enrollments

    log = []

    async with TestClient(TestServer(_receipt_app(adapter))) as client:
        # 1. Operator mints a one-time pairing grant for the phone (server-side,
        #    what POST /v1/operator/enrollments does for a settings:write caller).
        grant = enrollments.create(
            label="Galaxy S24 (receipt)",
            origin=origin,
            scopes=["profiles:read", "profiles:write"],
        )
        log.append(f"1. enrollment minted: code len={len(grant.code)}, ttl bound")

        # 2. Phone inspects the code before consenting.
        inspect = await client.post(
            "/v1/operator/enrollments/inspect",
            json={"code": grant.code, "origin": origin},
        )
        assert inspect.status == 200, await inspect.text()
        preview = await inspect.json()
        assert preview["origin"] == origin
        assert sorted(preview["scopes"]) == ["profiles:read", "profiles:write"]
        log.append(f"2. inspect ok: scopes={preview['scopes']}")

        # 3. Phone exchanges the code once for a scoped bearer token.
        exch = await client.post(
            "/v1/operator/enrollments/exchange",
            json={"code": grant.code, "origin": origin},
        )
        assert exch.status == 200, await exch.text()
        issued = await exch.json()
        token = issued["token"]
        credential_id = issued["credential"]["credential_id"]
        assert token.startswith("hop_")
        headers = {"Authorization": f"Bearer {token}"}
        log.append(f"3. exchanged for scoped token; credential={credential_id}")

        # 3a. The one-time code is now spent.
        replay = await client.post(
            "/v1/operator/enrollments/exchange",
            json={"code": grant.code, "origin": origin},
        )
        assert replay.status in (400, 404, 409), await replay.text()
        log.append("3a. code replay refused (single-use)")

        # 4. Scoped token creates a cloned profile.
        created = await client.post(
            "/api/profiles",
            headers=headers,
            json={"name": "receiptbot", "clone_from": "default"},
        )
        assert created.status == 201, await created.text()
        created_body = await created.json()
        assert created_body["name"] == "receiptbot"
        revision = created_body["revision"]
        log.append(f"4. profile created: revision={revision[:12]}...")

        # 5. It appears in the list.
        listed = await client.get("/api/profiles", headers=headers)
        listed_body = await listed.json()
        assert any(p["id"] == "receiptbot" for p in listed_body["data"])
        log.append("5. profile appears in list")

        # 6. Soul write requires the current revision (If-Match).
        no_match = await client.put(
            "/api/profiles/receiptbot/soul",
            headers=headers,
            json={"content": "You are the receipt agent."},
        )
        assert no_match.status == 428  # missing If-Match
        soul_write = await client.put(
            "/api/profiles/receiptbot/soul",
            headers={**headers, "If-Match": revision},
            json={"content": "You are the receipt agent."},
        )
        assert soul_write.status == 200, await soul_write.text()
        revision2 = (await soul_write.json())["revision"]
        assert revision2 != revision
        log.append("6. soul write: 428 without If-Match, ok with revision")

        # 7. A stale revision is rejected with no write (optimistic concurrency).
        stale = await client.patch(
            "/api/profiles/receiptbot",
            headers={**headers, "If-Match": revision},  # the OLD revision
            json={"new_name": "receiptbot2"},
        )
        assert stale.status == 412, await stale.text()
        still = await client.get("/api/profiles", headers=headers)
        assert any(p["id"] == "receiptbot" for p in (await still.json())["data"])
        log.append("7. stale-revision rename rejected 412, no write")

        # 8. Rename with the fresh revision.
        renamed = await client.patch(
            "/api/profiles/receiptbot",
            headers={**headers, "If-Match": revision2},
            json={"new_name": "receiptbot2"},
        )
        assert renamed.status == 200, await renamed.text()
        revision3 = (await renamed.json())["revision"]
        log.append("8. renamed receiptbot -> receiptbot2")

        # 9. Delete it.
        deleted = await client.delete(
            "/api/profiles/receiptbot2",
            headers={**headers, "If-Match": revision3},
        )
        assert deleted.status == 200, await deleted.text()
        gone = await client.get("/api/profiles", headers=headers)
        assert not any(p["id"] == "receiptbot2" for p in (await gone.json())["data"])
        log.append("9. profile deleted")

        # 10. Operator revokes the credential (superuser).
        super_headers = {"Authorization": "Bearer sk-super"}
        revoke = await client.delete(
            f"/v1/operator/credentials/{credential_id}", headers=super_headers
        )
        assert revoke.status in (200, 204), await revoke.text()
        log.append("10. credential revoked")

        # 11. The revoked token is now refused (fail closed).
        after = await client.get("/api/profiles", headers=headers)
        assert after.status == 401, await after.text()
        log.append("11. revoked token -> 401 (fail closed)")

    print("\n=== MILESTONE-0 RECEIPT ===")
    for line in log:
        print("  " + line)
    print("=== all steps passed ===")
