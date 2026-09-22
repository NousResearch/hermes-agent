"""Owner-mediated Bitwarden/1Password unlock for messaging sessions (#108316).

The browser-vault tools prompt through a *surface* callback (CLI panel, TUI bridge, Desktop). A
Telegram/Discord session installs none and must never accept a master password in chat, so those
sessions have one supported route: the gateway's own local control socket. The tool mints a
one-time code naming the backend, the profile home and the requesting session; the owner redeems
it from a terminal on the gateway host (``hermes vault unlock <code>``); the gateway process then
unlocks through the same ``agent.vault_backends`` API an interactive surface uses.

These tests pin that contract without any password manager installed: a Bitwarden-shaped stub
backend, the production ticket/verb encoding, the real unlock bookkeeping. ``handle_request_line``
is the production wire path minus the transport; the socket transport itself is covered by
``tests/gateway/test_control_socket.py`` (POSIX) and the named-pipe lane.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta
from gateway import control_socket, vault_unlock as vault_unlock_mod
from gateway.session_context import clear_session_vars, set_session_vars
from gateway.vault_unlock import owner_unlock_available, vault_unlock_handlers

MASTER = "correct horse battery staple"
SESSION_ID = "tg-session-108316"


class _StubBitwarden(LoginBackend):
    """Bitwarden-shaped backend: the real per-session unlock bookkeeping, no ``bw`` subprocess."""

    name, display_name, prefix, needs_unlock = "bitwarden", "Bitwarden", "bw:", True

    def __init__(self) -> None:
        self.unlock_calls: list[str] = []

    def is_unlocked(self) -> bool:
        return unlock_mod.is_unlocked(self.name)

    def unlock(self, master_password: str) -> None:
        self.unlock_calls.append(master_password)
        if master_password != MASTER:
            raise RuntimeError("Invalid master password.")
        unlock_mod.store_session_token(self.name, "SESSION-TOKEN-123",
                                      unlock_mod.begin_unlock(self.name))

    def list_items(self):
        if not self.is_unlocked():
            return []
        return [VaultItemMeta(id="bw:abc", kind="login", label="Example", origin="https://example.com",
                              created_at="2026-01-01T00:00:00Z", identifier_type="username",
                              identifier="jane@example.com")]

    def get_meta(self, handle: str):
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        return "plain sentence nobody would flag 7"


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    """A gateway process serving one Telegram session, with its control socket bound in-process."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    unlock_mod.lock()
    stub = _StubBitwarden()
    server = control_socket.GatewayControlServer(home, verb_handlers=vault_unlock_handlers())
    control_socket.set_local_control_server(server)  # what start() does once the socket is bound
    tokens = set_session_vars(platform="telegram", source="telegram", chat_id="108316",
                              user_id="42", session_key="telegram:108316", session_id=SESSION_ID)
    with patch("agent.vault_backends.base.enabled_backends", return_value=[stub]), \
         patch("agent.vault_backends.enabled_backends", return_value=[stub]):
        yield SimpleNamespace(stub=stub, server=server, home=home)
    clear_session_vars(tokens)
    control_socket.set_local_control_server(None)
    unlock_mod.lock()


def _redeem(server, code: str, password: str = MASTER) -> dict:
    """Exactly what the owner's terminal does: one control request over the socket."""
    raw = control_socket.build_control_request("vault-unlock", {"code": code, "password": password})
    return json.loads(server.handle_request_line(raw.rstrip(b"\n")).decode("utf-8"))


def test_telegram_session_unlocks_via_the_owner_terminal_without_touching_the_chat(gateway):
    from tools.browser_vault_tool import browser_vault_fill, browser_vault_list, browser_vault_unlock

    listed = json.loads(browser_vault_list())
    assert listed["items"] == []
    assert listed["locked"] == [{"backend": "bitwarden", "display_name": "Bitwarden",
                                "unlock": "owner_terminal"}]

    pending = json.loads(browser_vault_unlock("bitwarden"))
    assert pending["error_type"] == "unlock_pending"
    assert pending["success"] is False
    code = pending["code"]
    assert pending["command"] == f"hermes vault unlock {code}"
    assert MASTER not in json.dumps(pending), "the master password is never in a tool result"

    # The owner, in a terminal on the gateway host.
    reply = _redeem(gateway.server, code)
    assert reply["ok"] is True and reply["result"]["unlocked"] is True
    assert gateway.stub.unlock_calls == [MASTER]
    assert MASTER not in json.dumps(reply), "the answer never echoes the master password"

    # The code is spent: a replay cannot unlock anything after an explicit lock.
    unlock_mod.lock("bitwarden")
    replay = _redeem(gateway.server, code)
    assert replay["result"]["unlocked"] is False and replay["result"]["error_type"] == "code_invalid"
    assert not gateway.stub.is_unlocked()

    # Unlock once more and prove the asking session owns the token.
    assert json.loads(browser_vault_unlock("bitwarden"))["error_type"] == "unlock_pending"
    fresh = json.loads(browser_vault_unlock("bitwarden"))["code"]
    assert _redeem(gateway.server, fresh)["result"]["unlocked"] is True
    assert json.loads(browser_vault_unlock("bitwarden"))["already_unlocked"] is True

    listed = json.loads(browser_vault_list())
    assert listed["locked"] is None
    assert listed["items"][0]["identifier"] == "jane@example.com"
    assert MASTER not in json.dumps(listed)

    # A fill now runs against the unlocked manager (password resolved server-side).
    with patch("tools.browser_vault_tool._current_page_origin", return_value="https://example.com"), \
         patch("tools.browser_vault_tool._eval_js", return_value={"success": True, "result": json.dumps([
             {"tag": "input", "type": "password", "name": "password", "id": "pw",
              "autocomplete": "current-password", "visible": True}])}), \
         patch("tools.browser_vault_tool._eval_js_secret",
               return_value={"success": True, "result": json.dumps({"filled": 1})}):
        filled = json.loads(browser_vault_fill("bw:abc", task_id="t"))
    assert filled["success"] is True and filled["filled_fields"] == 1

    unlock_mod.release_session(SESSION_ID)  # the session that asked ends
    assert not gateway.stub.is_unlocked(), "the token belongs to the session that asked for it"


def test_headless_and_interactive_contexts_are_unaffected(gateway, monkeypatch):
    """The owner-terminal route belongs to messaging sessions only; every other posture is unchanged."""
    from tools.browser_vault_tool import browser_vault_list, browser_vault_unlock

    # An interactive surface with a wired prompt keeps its masked prompt (no command in the result).
    unlock_mod.set_unlock_prompt_callback(lambda *_: MASTER)
    try:
        assert json.loads(browser_vault_list())["locked"][0]["unlock"] == "browser_vault_unlock"
        assert _unlocked_via_prompt() is True
    finally:
        unlock_mod.set_unlock_prompt_callback(None)

    unlock_mod.lock("bitwarden")

    # Cron / API-server sessions have nobody to read the code, so they keep the honest refusal. The
    # platform is rebound on the session, not via env: the predicate reads contextvars first (that is
    # what a real api_server turn binds), so an env override would never be seen.
    for cron in ({"HERMES_CRON_SESSION": "1"}, {}):
        for key, value in cron.items():
            monkeypatch.setenv(key, value)
        tokens = set_session_vars(platform="api_server", source="api_server", session_id=SESSION_ID)
        try:
            assert owner_unlock_available() is False
            assert json.loads(browser_vault_list())["locked"][0]["unlock"] == "unavailable_in_this_session"
            refused = json.loads(browser_vault_unlock("bitwarden"))
            assert refused["error_type"] == "unlock_unavailable" and "code" not in refused
            # Only the interactive prompt above ever reached the manager; these refusals added nothing.
            assert gateway.stub.unlock_calls == [MASTER]
        finally:
            clear_session_vars(tokens)
        for key in cron:
            monkeypatch.delenv(key, raising=False)

    # No control socket in this process (single-query CLI, cron worker): same honest refusal.
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    control_socket.set_local_control_server(None)
    tokens = set_session_vars(platform="telegram", source="telegram", session_id=SESSION_ID)
    try:
        assert owner_unlock_available() is False
        assert json.loads(browser_vault_unlock("bitwarden"))["error_type"] == "unlock_unavailable"
    finally:
        clear_session_vars(tokens)


def _unlocked_via_prompt() -> bool:
    from tools.browser_vault_tool import browser_vault_unlock

    return bool(json.loads(browser_vault_unlock("bitwarden")).get("success"))


def test_unlock_code_is_reused_while_live_expires_and_never_accepts_a_wrong_password(gateway,
                                                                                    monkeypatch):
    minted = vault_unlock_mod.mint_unlock_ticket("bitwarden")
    assert minted is not None
    assert vault_unlock_mod.mint_unlock_ticket("bitwarden")["code"] == minted["code"], \
        "a retry must not churn a new code into the chat"

    wrong = _redeem(gateway.server, minted["code"], password="not the master password")
    assert wrong["result"]["unlocked"] is False and wrong["result"]["error_type"] == "unlock_failed"
    assert "not the master password" not in json.dumps(wrong), "failures never echo the attempt"
    assert gateway.stub.is_unlocked() is False

    # The code survived the failed attempt, and expires on its own TTL.
    monkeypatch.setattr(vault_unlock_mod, "TICKET_TTL_S", -1)
    stale = vault_unlock_mod.mint_unlock_ticket("bitwarden")
    assert stale is not None and stale["code"] != minted["code"]
    expired = _redeem(gateway.server, stale["code"])
    assert expired["result"]["unlocked"] is False and expired["result"]["error_type"] == "code_invalid"
    assert not gateway.stub.is_unlocked()

    # An unknown code and a code-less request are refused without touching the manager.
    assert _redeem(gateway.server, "DEADBEEF")["result"]["error_type"] == "code_invalid"
    raw = control_socket.build_control_request("vault-unlock", {"password": MASTER})
    assert json.loads(gateway.server.handle_request_line(raw.rstrip(b"\n")).decode())["result"][
        "error_type"] == "bad_request"
    # The wrong password above is the only attempt that reached the manager; the expired, unknown and
    # code-less requests never did.
    assert gateway.stub.unlock_calls == ["not the master password"]
