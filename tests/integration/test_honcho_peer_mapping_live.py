"""Live end-to-end for gateway identity mapping: simulated gateway accounts write through the real
honcho plugin into a throwaway workspace, then ``hermes honcho peers map`` re-maps one of them.

Opt-in only: ``HONCHO_E2E=1`` plus a credential. ``HONCHO_E2E_API_KEY`` wins; otherwise the access
token on the live host block in ``~/.hermes/honcho.json`` is used as a plain apiKey (no oauth block is
copied, so this process never rotates the refresh token). The workspace is deleted at the end.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

_E2E = os.environ.get("HONCHO_E2E") == "1"
_TURN_WAIT = 45.0


def _credential() -> tuple[str, str | None]:
    key = os.environ.get("HONCHO_E2E_API_KEY", "").strip()
    if key:
        return key, os.environ.get("HONCHO_E2E_BASE_URL") or None
    live = Path.home() / ".hermes" / "honcho.json"
    try:
        host = (json.loads(live.read_text()).get("hosts") or {}).get("hermes") or {}
    except Exception:
        return "", None
    token = str(host.get("apiKey") or "")
    expires = float(((host.get("oauth") or {}).get("expiresAt")) or 0)
    if expires and expires - time.time() < 180:
        pytest.skip("live access token expires in under three minutes; wait for the desktop to refresh it")
    return token, host.get("baseUrl") or None


@pytest.fixture(scope="module")
def live():
    if not _E2E:
        pytest.skip("set HONCHO_E2E=1 to run the live peer-mapping suite")
    key, base_url = _credential()
    if not key:
        pytest.skip("no honcho credential: set HONCHO_E2E_API_KEY or sign in with hermes honcho setup")
    from honcho import Honcho

    workspace = f"hermes-e2e-{uuid.uuid4().hex[:8]}"
    kwargs = {"api_key": key, "workspace_id": workspace}
    if base_url:
        kwargs["base_url"] = base_url
    client = Honcho(**kwargs)
    yield SimpleNamespace(key=key, base_url=base_url, workspace=workspace, client=client)
    try:
        client.delete_workspace(workspace)
    except Exception as exc:  # cleanup is best-effort; the workspace name says what it is
        print(f"workspace {workspace} not deleted: {exc}")


class _Account:
    def __init__(self, platform: str, user_id: str, name: str, chat_id: str | None = None,
                 user_id_alt: str | None = None):
        self.platform, self.user_id, self.name = platform, user_id, name
        self.chat_id = chat_id or user_id
        self.user_id_alt = user_id_alt

    @property
    def session_key(self) -> str:
        return f"{self.platform}:dm:{self.chat_id}"


ALICE = _Account("telegram", "111000111", "alice")
BOB = _Account("telegram", "222000222", "bob")
CAROL = _Account("discord", "333000333", "carol")
DAVE = _Account("signal", "+15550004444", "dave", user_id_alt="9f1c0d2e-dave-uuid")


def _write_sandbox(home: Path, live, host_block: dict) -> Path:
    block = {
        "enabled": True, "workspace": live.workspace, "peerName": "eri-e2e", "aiPeer": "hermes-e2e",
        "saveMessages": True, "writeFrequency": "turn", "recallMode": "context", "sessionStrategy": "per-session",
        "observationMode": "directional", "dialecticCadence": 999, "contextCadence": 999,
        **host_block,
    }
    cfg = {"apiKey": live.key, "hosts": {"hermes": block}}
    if live.base_url:
        cfg["baseUrl"] = live.base_url
    (home / "honcho.json").write_text(json.dumps(cfg, indent=2))
    return home / "honcho.json"


@pytest.fixture
def sandbox(tmp_path, monkeypatch, live):
    """Isolated HERMES_HOME whose honcho.json points at the throwaway workspace."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.memory.honcho.client import reset_honcho_client

    home = tmp_path / "hermes-home"
    home.mkdir()
    for var in ("HONCHO_API_KEY", "HONCHO_BASE_URL", "HONCHO_URL", "HONCHO_WORKSPACE", "HERMES_HONCHO_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    reset_honcho_client()
    yield SimpleNamespace(home=home, write=lambda block: _write_sandbox(home, live, block))
    reset_honcho_client()
    reset_hermes_home_override(token)


def _run_turn(sandbox, account: _Account, text: str, reply: str) -> SimpleNamespace:
    """One gateway turn for ``account`` through the real provider, the way run_turn_runner builds it."""
    from plugins.memory.honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    provider.initialize(
        session_id=f"sess-{account.platform}-{account.chat_id}", platform=account.platform,
        hermes_home=str(sandbox.home), agent_context="primary",
        user_id=account.user_id, user_id_alt=account.user_id_alt, user_name=account.name,
        chat_id=account.chat_id, chat_type="dm", gateway_session_key=account.session_key,
    )
    if provider._init_thread is not None:
        provider._init_thread.join(timeout=_TURN_WAIT)
    assert provider._ensure_session(), "provider did not reach the workspace"
    provider.on_turn_start(1, text)
    provider.sync_turn(text, reply)
    if provider._sync_thread is not None:
        provider._sync_thread.join(timeout=_TURN_WAIT)
    session = provider._manager.get_or_create(provider._session_key)
    provider._manager.flush_all()
    result = SimpleNamespace(user_peer_id=session.user_peer_id, assistant_peer_id=session.assistant_peer_id,
                             honcho_session_id=session.honcho_session_id, session_key=provider._session_key)
    provider.shutdown()
    return result


def _authors(live, honcho_session_id: str) -> dict[str, list[str]]:
    """peer id -> message contents in that honcho session, read back through the sdk."""
    out: dict[str, list[str]] = {}
    for message in live.client.session(honcho_session_id).messages(size=50):
        out.setdefault(str(message.peer_id), []).append(str(message.content))
    return out


def _peer_ids(live) -> set[str]:
    return {str(p.id) for p in live.client.peers(size=50)}


# --- identity shapes ---------------------------------------------------------------------------


def test_single_shape_collapses_every_account_onto_the_declared_peer(sandbox, live):
    sandbox.write({"pinUserPeer": True})
    a = _run_turn(sandbox, ALICE, "alice says hello", "hi alice")
    b = _run_turn(sandbox, BOB, "bob says hello", "hi bob")
    assert a.user_peer_id == b.user_peer_id == "eri-e2e"
    assert "alice says hello" in _authors(live, a.honcho_session_id)["eri-e2e"]
    assert "bob says hello" in _authors(live, b.honcho_session_id)["eri-e2e"]
    assert not {"111000111", "222000222", "tg_111000111"} & _peer_ids(live)


def test_hybrid_shape_aliases_the_operator_and_prefixes_strangers(sandbox, live):
    sandbox.write({"pinUserPeer": False, "runtimePeerPrefix": "tg_",
                   "userPeerAliases": {ALICE.user_id: "eri-e2e", CAROL.user_id: "eri-e2e"}})
    a = _run_turn(sandbox, ALICE, "alice on telegram", "hi alice")
    c = _run_turn(sandbox, CAROL, "carol on discord", "hi carol")
    b = _run_turn(sandbox, BOB, "bob is a stranger", "hi bob")
    assert a.user_peer_id == c.user_peer_id == "eri-e2e"
    assert b.user_peer_id == f"tg_{BOB.user_id}"
    assert "alice on telegram" in _authors(live, a.honcho_session_id)["eri-e2e"]
    assert "carol on discord" in _authors(live, c.honcho_session_id)["eri-e2e"]
    assert "bob is a stranger" in _authors(live, b.honcho_session_id)[f"tg_{BOB.user_id}"]
    # each gateway chat keeps its own honcho session even when the peer is shared
    assert a.honcho_session_id != c.honcho_session_id
    assert {"eri-e2e", f"tg_{BOB.user_id}", "hermes-e2e"} <= _peer_ids(live)


def test_multi_shape_gives_every_account_its_own_peer(sandbox, live):
    sandbox.write({"pinUserPeer": False})
    a = _run_turn(sandbox, ALICE, "alice alone", "hi")
    b = _run_turn(sandbox, BOB, "bob alone", "hi")
    assert a.user_peer_id == ALICE.user_id and b.user_peer_id == BOB.user_id
    assert "alice alone" in _authors(live, a.honcho_session_id)[ALICE.user_id]
    assert "bob alone" in _authors(live, b.honcho_session_id)[BOB.user_id]


def test_alias_on_the_alt_id_resolves_the_signal_account(sandbox, live):
    sandbox.write({"pinUserPeer": False, "userPeerAliases": {DAVE.user_id_alt: "dave-e2e"}})
    d = _run_turn(sandbox, DAVE, "dave from signal", "hi dave")
    assert d.user_peer_id == "dave-e2e"
    assert "dave from signal" in _authors(live, d.honcho_session_id)["dave-e2e"]


# --- hermes honcho peers map ---------------------------------------------------------------


def _seed_gateway_rows(home: Path, accounts: list[_Account]) -> None:
    """Rows the gateway stamps when it routes a message from each account."""
    from hermes_state import SessionDB

    db = SessionDB(home / "state.db")
    for i, acct in enumerate(accounts):
        origin = json.dumps({"platform": acct.platform, "chat_id": acct.chat_id, "chat_type": "dm",
                             "user_id": acct.user_id, "user_name": acct.name, "user_id_alt": acct.user_id_alt,
                             "is_bot": False})
        db.record_gateway_session_peer(
            f"sess-{acct.platform}-{acct.chat_id}", source=acct.platform, user_id=acct.user_id,
            session_key=acct.session_key, chat_id=acct.chat_id, chat_type="dm", display_name=acct.name,
            origin_json=origin)
        time.sleep(0.01 * i)


def test_peers_map_remaps_a_stranger_and_the_next_turn_lands_there(sandbox, live, monkeypatch, capsys):
    import plugins.memory.honcho.cli as honcho_cli

    cfg_path = sandbox.write({"pinUserPeer": False, "runtimePeerPrefix": "tg_",
                              "userPeerAliases": {ALICE.user_id: "eri-e2e"}})
    before = _run_turn(sandbox, BOB, "bob before mapping", "hi")
    assert before.user_peer_id == f"tg_{BOB.user_id}"
    _seed_gateway_rows(sandbox.home, [ALICE, BOB, CAROL])

    answers = iter([BOB.user_id, "bob-e2e", ""])
    monkeypatch.setattr(honcho_cli, "_prompt", lambda label, default=None, secret=False: next(answers))
    honcho_cli.honcho_command(SimpleNamespace(honcho_command="peers", peers_action="map", target_profile=None))
    out = capsys.readouterr().out

    assert "bob-e2e" in out and "eri-e2e" in out
    saved = json.loads(cfg_path.read_text())["hosts"]["hermes"]["userPeerAliases"]
    assert saved[BOB.user_id] == "bob-e2e" and saved[ALICE.user_id] == "eri-e2e"

    from plugins.memory.honcho.client import reset_honcho_client
    reset_honcho_client()
    after = _run_turn(sandbox, BOB, "bob after mapping", "hi again")
    assert after.user_peer_id == "bob-e2e"
    assert "bob after mapping" in _authors(live, after.honcho_session_id)["bob-e2e"]
    assert "bob before mapping" in _authors(live, before.honcho_session_id)[f"tg_{BOB.user_id}"]
    assert "bob-e2e" in _peer_ids(live)
