"""Dashboard ``/api/vault`` routes: metadata only, and nothing leaks on the error path.

The vault's whole contract is that the caller (agent or dashboard) never receives a stored
password, so a refusal must be scrubbed before it leaves the route, and a destructive route
must name its profile when one backend serves several. These tests exercise the handlers
directly — the routes' own behaviour, not the FastAPI wiring.
"""

from typing import cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from hermes_cli.web_routers import vault as vault_routes


def _req() -> Request:
    """Stands in for the Request; the token gate is patched out, so it is never touched."""
    return cast(Request, object())


@pytest.fixture(autouse=True)
def _no_token_gate(monkeypatch):
    monkeypatch.setattr(vault_routes, "_require_token", lambda request: None)


@pytest.fixture
def no_multiplex(monkeypatch):
    """Default single-profile host: an omitted profile means the launch home."""
    from agent import secret_scope

    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: False)


@pytest.mark.asyncio
async def test_add_item_redacts_the_secret_from_a_store_refusal(monkeypatch):
    from agent.vault_store import VaultError

    class _RefusingStore:
        def add_item(self, **kwargs):
            raise VaultError(f"cannot store password hunter2 for {kwargs['label']}")

    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: _RefusingStore())
    body = vault_routes.VaultAddBody(
        kind="login",
        label="Work email",
        origin="https://mail.example",
        secret={"identifier": "u@example.com", "password": "hunter2"},
    )

    with pytest.raises(HTTPException) as exc:
        await vault_routes.add_vault_item(body, _req())

    assert exc.value.status_code == 400
    assert "hunter2" not in exc.value.detail
    assert "[REDACTED]" in exc.value.detail


@pytest.mark.asyncio
async def test_unlock_redacts_the_master_password_from_a_backend_failure(monkeypatch):
    class _Backend:
        name = "onepassword"
        display_name = "1Password"
        needs_unlock = True

        def is_unlocked(self):
            return False

        def unlock(self, master_password):
            raise RuntimeError(f"bad master password {master_password}")

    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [_Backend()])
    body = vault_routes.VaultUnlockBody(name="onepassword", password="s3cret-master")

    with pytest.raises(HTTPException) as exc:
        await vault_routes.unlock_vault_source(body, _req())

    assert exc.value.status_code == 400
    assert "s3cret-master" not in exc.value.detail
    assert "[REDACTED]" in exc.value.detail


@pytest.mark.asyncio
async def test_add_then_list_then_remove_round_trip_stores_no_readable_secret(no_multiplex):
    """The real local store: the list route returns metadata, never the password."""
    body = vault_routes.VaultAddBody(
        kind="login",
        label="Campus portal",
        origin="https://portal.example",
        secret={
            "identifier_type": "email",
            "identifier": "student@example.com",
            "password": "correct-horse",
        },
    )
    added = await vault_routes.add_vault_item(body, _req())
    assert added["id"].startswith("vault_")

    listed = await vault_routes.list_vault_items(_req(), profile=None)
    row = next(item for item in listed["items"] if item["id"] == added["id"])
    assert row["label"] == "Campus portal"
    assert row["origin"] == "https://portal.example"
    assert row["identifier"] == "student@example.com"
    assert "correct-horse" not in str(listed)

    removed = await vault_routes.remove_vault_item(
        added["id"], _req(), profile=None
    )
    assert removed == {"removed": True}
    assert all(
        item["id"] != added["id"]
        for item in (await vault_routes.list_vault_items(_req(), profile=None))["items"]
    )


@pytest.mark.asyncio
async def test_remove_refuses_an_unnamed_profile_under_multiplex(monkeypatch):
    """One backend serves many profiles: an unnamed delete could hit the launch profile."""
    from agent import secret_scope

    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)

    with pytest.raises(HTTPException) as exc:
        await vault_routes.remove_vault_item("vault_x", _req(), profile=None)

    assert exc.value.status_code == 400
    assert "explicit profile" in exc.value.detail
