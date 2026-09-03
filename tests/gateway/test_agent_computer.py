"""BWM-796 Agent Computer acceptance gates.

Synthetic profiles only. Never copies a real browser profile or cookies.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gateway.agent_computer import resolve_runtime_name
from gateway.agent_computer.adapter import HermesChromiumRuntime, InMemoryRuntime
from gateway.agent_computer.contract import (
    FORBIDDEN_PUBLIC_KEYS,
    AgentComputerContract,
    sanitize_public,
)
from gateway.agent_computer.errors import (
    AgentComputerError,
    CheckpointRequiredError,
    ConflictError,
    ForbiddenError,
    IdentityBusyError,
    InvalidTokenError,
    ObserveRequiredError,
    RevokedError,
    StaleControllerError,
)
from gateway.agent_computer.models import OWNER_PRINCIPAL, agent_principal
from gateway.agent_computer.service import AgentComputerService
from gateway.agent_computer.store import AgentComputerStore

PERMANENT_AGENTS = [
    "majed",
    "abu-saleh",
    "agent-03",
    "agent-04",
    "agent-05",
    "agent-06",
    "agent-07",
    "agent-08",
    "agent-09",
    "agent-10",
    "agent-11",
    "agent-12",
    "agent-13",
    "agent-14",
]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 3, 1, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def _svc(tmp_path: Path, clock: Clock | None = None) -> AgentComputerService:
    runtime = InMemoryRuntime()
    store = AgentComputerStore(tmp_path / "state.db")
    return AgentComputerService(
        store,
        runtime,
        data_root=tmp_path,
        clock=clock or Clock(),
        takeover_ttl_s=60,
    )


def test_ensure_binds_permanent_profile_not_session(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    again = svc.ensure_computer("majed")
    assert computer.id == again.id
    assert computer.agent_profile_id == "majed"
    with pytest.raises(ConflictError):
        svc.ensure_computer("session:abc")
    with pytest.raises(ConflictError):
        svc.ensure_computer("run:xyz")


def test_fourteen_agents_isolated(tmp_path):
    svc = _svc(tmp_path)
    computers = {name: svc.ensure_computer(name) for name in PERMANENT_AGENTS}
    assert len({c.id for c in computers.values()}) == 14
    majed = computers["majed"]
    other = agent_principal("abu-saleh")
    with pytest.raises(ForbiddenError):
        svc.authorize_read(majed, other)
    with pytest.raises(ForbiddenError):
        svc.observe(
            majed.id,
            other,
            lease_id="x",
            fencing_epoch=1,
        )


def test_browser_identity_exclusive_lock_never_clones(tmp_path):
    svc = _svc(tmp_path)
    a = svc.ensure_computer("majed")
    b = svc.ensure_computer("abu-saleh")
    identity = svc.create_identity(ownership=["majed", "abu-saleh"], metadata={"label": "work"})
    svc.attach_identity(a.id, identity.id, OWNER_PRINCIPAL)
    with pytest.raises(IdentityBusyError) as exc:
        svc.attach_identity(b.id, identity.id, OWNER_PRINCIPAL)
    assert exc.value.code == "BROWSER_IDENTITY_BUSY"
    assert Path(identity.profile_ref).joinpath(".hermes-identity").read_text() == identity.id
    # Second attach did not create another managed dir.
    dirs = list((tmp_path / "identities").iterdir())
    assert len(dirs) == 1


def test_identity_pin_is_explicit_not_last_used(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    first = svc.create_identity(ownership=["majed"], metadata={"label": "work"})
    second = svc.create_identity(ownership=["majed"], metadata={"label": "personal"})
    attached = svc.attach_identity(computer.id, first.id, OWNER_PRINCIPAL)
    assert attached.active_browser_identity_id == first.id
    assert attached.active_browser_identity_id != second.id
    src = Path(__file__).resolve().parents[2] / "gateway" / "agent_computer"
    text = (src / "adapter.py").read_text() + (src / "service.py").read_text()
    assert "snapshot_real_profile(" not in text
    assert "profile.last_used" not in text
    assert "browser.use_real_profile" not in text


def test_same_environment_takeover_and_give_back(tmp_path):
    clock = Clock()
    svc = _svc(tmp_path, clock)
    contract = AgentComputerContract(svc)
    computer = svc.ensure_computer("majed")
    identity = svc.create_identity(ownership=["majed"])
    svc.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)
    agent = agent_principal("majed")
    computer, agent_lease = svc.wake(computer.id, agent)
    obs = svc.observe(
        computer.id, agent, lease_id=agent_lease.lease_id, fencing_epoch=agent_lease.fencing_epoch
    )
    svc.act(
        computer.id,
        agent,
        lease_id=agent_lease.lease_id,
        fencing_epoch=agent_lease.fencing_epoch,
        kind="navigate",
        target="https://erp.example.test/inbox",
    )

    takeover = contract.request_takeover(computer.id, OWNER_PRINCIPAL, reason="help")
    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            agent,
            lease_id=agent_lease.lease_id,
            fencing_epoch=agent_lease.fencing_epoch,
            kind="type",
            text="stale",
        )

    owner_lease = contract.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=takeover["takeover_token"]
    )
    owner_view = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    assert owner_view["url"] == "https://erp.example.test/inbox"
    contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="navigate",
        target="https://erp.example.test/done",
    )

    given = contract.give_back(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    new_lease = given["lease"]
    with pytest.raises(ObserveRequiredError):
        svc.act(
            computer.id,
            agent,
            lease_id=new_lease["lease_id"],
            fencing_epoch=new_lease["fencing_epoch"],
            kind="type",
            text="blind",
        )
    resumed = contract.observe(
        computer.id,
        agent,
        lease_id=new_lease["lease_id"],
        fencing_epoch=new_lease["fencing_epoch"],
    )
    assert resumed["url"] == "https://erp.example.test/done"
    assert resumed["controller"] == "AGENT_CONTROLLED"
    # Duplicate give-back is idempotent.
    again = contract.give_back(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    assert again["lease"]["lease_id"] == new_lease["lease_id"]


def test_stale_owner_lease_after_give_back(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    _, agent_lease = svc.wake(computer.id, agent)
    raw = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    owner_lease = svc.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=raw["takeover_token"]
    )
    svc.give_back(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease.lease_id,
        fencing_epoch=owner_lease.fencing_epoch,
    )
    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner_lease.lease_id,
            fencing_epoch=owner_lease.fencing_epoch,
            kind="click",
            target="x",
        )
    # Original agent lease is also dead.
    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            agent,
            lease_id=agent_lease.lease_id,
            fencing_epoch=agent_lease.fencing_epoch,
            kind="click",
            target="x",
        )


def test_unauthorized_takeover_rejected(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    with pytest.raises(ForbiddenError):
        svc.request_takeover(computer.id, agent_principal("majed"))
    with pytest.raises(ForbiddenError):
        svc.connect_takeover(computer.id, agent_principal("majed"), takeover_token="nope")


def test_takeover_token_single_use_bound_and_expiry(tmp_path):
    clock = Clock()
    svc = _svc(tmp_path, clock)
    computer = svc.ensure_computer("majed")
    raw = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    token = raw["takeover_token"]
    svc.connect_takeover(computer.id, OWNER_PRINCIPAL, takeover_token=token)
    with pytest.raises(InvalidTokenError):
        svc.connect_takeover(computer.id, OWNER_PRINCIPAL, takeover_token=token)

    other = svc.ensure_computer("abu-saleh")
    raw2 = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    with pytest.raises(ForbiddenError):
        svc.connect_takeover(other.id, OWNER_PRINCIPAL, takeover_token=raw2["takeover_token"])

    clock.advance(120)
    raw3 = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    clock.advance(1)
    # Token minted at current clock, ttl 60s — still valid. Advance past ttl.
    clock.advance(61)
    with pytest.raises(InvalidTokenError):
        svc.connect_takeover(computer.id, OWNER_PRINCIPAL, takeover_token=raw3["takeover_token"])


def test_owner_ttl_and_transport_disconnect_recover(tmp_path):
    clock = Clock()
    svc = _svc(tmp_path, clock)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    _, agent_lease = svc.wake(computer.id, agent)
    raw = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    owner_lease = svc.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=raw["takeover_token"]
    )
    clock.advance(61)
    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner_lease.lease_id,
            fencing_epoch=owner_lease.fencing_epoch,
            kind="click",
            target="late",
        )
    computer = svc.get_computer(computer.id)
    assert computer.control_authority.value == "AGENT_CONTROLLED"
    assert computer.resume_observe_required is True

    _, agent_lease = svc.wake(computer.id, agent)
    raw = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    svc.connect_takeover(computer.id, OWNER_PRINCIPAL, takeover_token=raw["takeover_token"])
    recovered = svc.owner_disconnect(computer.id, OWNER_PRINCIPAL)
    assert recovered is not None
    with pytest.raises(ObserveRequiredError):
        svc.act(
            computer.id,
            agent,
            lease_id=recovered.lease_id,
            fencing_epoch=recovered.fencing_epoch,
            kind="type",
            text="after-disconnect",
        )


def test_identity_auth_survives_sleep_wake(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    identity = svc.create_identity(ownership=["majed"])
    svc.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)
    agent = agent_principal("majed")
    _, lease = svc.wake(computer.id, agent)
    handle = svc._handles[computer.id]
    svc.runtime.act(handle, kind="set_cookie", target="sid", text="synthetic-session")
    svc.sleep(computer.id, agent)
    reopened = AgentComputerService(
        AgentComputerStore(tmp_path / "state.db"),
        svc.runtime,
        data_root=tmp_path,
    )
    persisted = reopened.store.get_computer_by_profile("majed")
    assert persisted is not None
    assert persisted.id == computer.id
    assert persisted.active_browser_identity_id == identity.id
    _, lease2 = reopened.wake(persisted.id, agent)
    reopened.observe(
        persisted.id, agent, lease_id=lease2.lease_id, fencing_epoch=lease2.fencing_epoch
    )
    assert svc.runtime.cookies_for_test(identity.id)["sid"] == "synthetic-session"


def test_checkpoint_blocks_sensitive_action(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    _, lease = svc.wake(computer.id, agent)
    svc.observe(computer.id, agent, lease_id=lease.lease_id, fencing_epoch=lease.fencing_epoch)
    with pytest.raises(CheckpointRequiredError) as exc:
        svc.act(
            computer.id,
            agent,
            lease_id=lease.lease_id,
            fencing_epoch=lease.fencing_epoch,
            kind="click",
            target="pay",
            action_class="payment",
        )
    svc.approve_checkpoint(exc.value.details["checkpoint_id"], OWNER_PRINCIPAL)
    receipt = svc.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="click",
        target="pay",
        action_class="payment",
    )
    assert receipt.accepted is True
    with pytest.raises(CheckpointRequiredError):
        svc.act(
            computer.id,
            agent,
            lease_id=lease.lease_id,
            fencing_epoch=lease.fencing_epoch,
            kind="click",
            target="pay-again",
            action_class="payment",
        )


def test_revoked_identity_cannot_attach_or_wake(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    identity = svc.create_identity(ownership=["majed"])
    svc.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)
    svc.revoke_identity(identity.id, OWNER_PRINCIPAL)
    computer = svc.get_computer(computer.id)
    assert computer.active_browser_identity_id is None
    with pytest.raises(RevokedError):
        svc.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)


def test_public_contract_strips_host_secrets(tmp_path):
    svc = _svc(tmp_path)
    contract = AgentComputerContract(svc)
    computer = svc.ensure_computer("majed")
    identity = svc.create_identity(ownership=["majed"])
    payload = contract.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)
    blob = str(payload)
    for key in FORBIDDEN_PUBLIC_KEYS:
        assert key not in payload
        assert f"/{key}" not in blob
    assert "profile_ref" not in blob
    assert "127.0.0.1" not in blob
    assert identity.profile_ref not in blob
    dirty = {
        "cdp_url": "http://127.0.0.1:9222",
        "cookies": {"sid": "secret"},
        "ok": True,
        "nested": {"user_data_dir": "/tmp/chrome", "title": "ok"},
    }
    clean = sanitize_public(dirty)
    assert "cdp_url" not in clean
    assert "cookies" not in clean
    assert clean["nested"]["title"] == "ok"
    assert "user_data_dir" not in clean["nested"]
    with pytest.raises(Exception):
        contract.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id="x",
            fencing_epoch=0,
            kind="set_cookie",
            target="sid",
            text="nope",
        )


def test_chromium_adapter_is_loopback_and_skips_real_profile_snapshot():
    source = inspect.getsource(HermesChromiumRuntime)
    assert "--remote-debugging-address=127.0.0.1" in source
    assert "--user-data-dir=" in source
    assert "snapshot_real_profile" not in source
    assert "--headless=new" in source


def test_default_toolsets_do_not_include_computer_tools():
    from toolsets import _HERMES_CORE_TOOLS, resolve_toolset

    for name in (
        "computer_ensure",
        "computer_status",
        "computer_wake",
        "computer_observe",
        "computer_act",
    ):
        assert name not in _HERMES_CORE_TOOLS
    core = resolve_toolset("hermes-cli")
    assert "computer_ensure" not in core


def test_file_safety_and_backup_exclude_agent_computers(tmp_path, monkeypatch):
    from agent.file_safety import get_read_block_error
    from hermes_cli.backup import _EXCLUDED_DIRS

    assert "agent-computers" in _EXCLUDED_DIRS
    root = tmp_path / "hermes-root"
    store = root / "agent-computers" / "identities" / "bi_test" / "Cookies"
    store.parent.mkdir(parents=True)
    store.write_text("synthetic")
    monkeypatch.setattr("agent.file_safety._hermes_root_path", lambda: root)
    monkeypatch.setattr("agent.file_safety._hermes_home_path", lambda: root)
    err = get_read_block_error(str(store))
    assert err and "Agent Computer" in err


def test_audit_has_no_tokens_or_cookies(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    events = svc.list_audit(computer.id)
    blob = str([e.detail for e in events])
    assert "takeover_token" not in blob
    assert "cookie" not in blob.lower()


def test_wake_expires_owner_and_refuses_agent_during_live_takeover(tmp_path):
    clock = Clock()
    svc = _svc(tmp_path, clock)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    svc.wake(computer.id, agent)
    raw = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    svc.connect_takeover(computer.id, OWNER_PRINCIPAL, takeover_token=raw["takeover_token"])
    with pytest.raises(ConflictError):
        svc.wake(computer.id, agent)
    clock.advance(61)
    computer2, lease = svc.wake(computer.id, agent)
    assert computer2.control_authority.value == "AGENT_CONTROLLED"
    assert computer2.resume_observe_required is True
    with pytest.raises(ObserveRequiredError):
        svc.act(
            computer.id,
            agent,
            lease_id=lease.lease_id,
            fencing_epoch=lease.fencing_epoch,
            kind="type",
            text="too-soon",
        )


def test_release_owner_for_transport(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    svc.wake(computer.id, agent)
    raw = svc.request_takeover(computer.id, OWNER_PRINCIPAL)
    svc.connect_takeover(computer.id, OWNER_PRINCIPAL, takeover_token=raw["takeover_token"])
    transport = object()
    svc.bind_owner_transport(computer.id, transport)
    assert svc.release_owner_for_transport(transport) == 1
    computer = svc.get_computer(computer.id)
    assert computer.control_authority.value == "AGENT_CONTROLLED"
    assert computer.resume_observe_required is True
    assert svc.release_owner_for_transport(transport) == 0


def test_ws_helper_ignores_client_profile_id():
    from tui_gateway.methods_agent_computer import _principal_from

    assert _principal_from(
        {"profile_id": "abu-saleh"}, None, "majed"
    ) == agent_principal("majed")
    assert _principal_from(
        {"profile_id": "majed"},
        {"user_id": "owner", "provider": "dashboard"},
        "majed",
    ) == OWNER_PRINCIPAL


def test_ws_helper_does_not_treat_server_internal_as_owner():
    from hermes_cli.dashboard_auth.ws_tickets import INTERNAL_PROVIDER, INTERNAL_USER_ID
    from tui_gateway.methods_agent_computer import _principal_from

    internal = {"user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER}
    assert _principal_from({"profile_id": "abu-saleh"}, internal, "majed") == agent_principal(
        "majed"
    )
    with pytest.raises(AgentComputerError, match="owner authentication required"):
        _principal_from({}, internal, "majed", owner_only=True)
    with pytest.raises(AgentComputerError, match="owner authentication required"):
        _principal_from({}, None, "majed", owner_only=True)


def test_tool_rejects_foreign_profile(monkeypatch):
    from tools.agent_computer_tool import _profile_id
    from gateway.agent_computer.errors import AgentComputerError

    monkeypatch.setattr("tools.agent_computer_tool._session_profile", lambda: "majed")
    assert _profile_id(None) == "majed"
    assert _profile_id("majed") == "majed"
    with pytest.raises(AgentComputerError):
        _profile_id("abu-saleh")


def test_chromium_observe_act_use_loopback_cdp(monkeypatch):
    from gateway.agent_computer.adapter import HermesChromiumRuntime, RuntimeHandle

    calls = []

    def fake_cdp(handle, method, params=None):
        calls.append((method, params))
        if method == "Runtime.evaluate":
            return {"result": {"value": {"url": "https://example.test", "title": "ok", "text": "hi"}}}
        if method == "Page.captureScreenshot":
            return {"data": "AAAA"}
        return {}

    monkeypatch.setattr("gateway.agent_computer.adapter.loopback_cdp", fake_cdp)
    handle = RuntimeHandle(
        computer_id="ac_1",
        identity_id="bi_1",
        user_data_dir="/tmp/x",
        cdp_loopback="http://127.0.0.1:9",
        backend="hermes_chromium",
    )
    runtime = HermesChromiumRuntime()
    obs = runtime.observe(handle)
    assert obs.url == "https://example.test"
    assert obs.screenshot_b64 == "AAAA"
    runtime.act(handle, kind="navigate", target="https://example.test/next")
    assert any(method == "Page.navigate" for method, _ in calls)


def test_loopback_cdp_refuses_non_loopback():
    from gateway.agent_computer.adapter import RuntimeHandle, loopback_cdp

    handle = RuntimeHandle(
        computer_id="ac_1",
        identity_id=None,
        user_data_dir="/tmp/x",
        cdp_loopback="http://10.0.0.2:9222",
    )
    with pytest.raises(RuntimeError, match="non-loopback"):
        loopback_cdp(handle, "Page.navigate", {"url": "https://example.test"})


def test_runtime_name_reads_config_yaml_env_is_override_only():
    assert resolve_runtime_name(env="", config={"agent_computer": {"runtime": "chromium"}}) == "chromium"
    assert resolve_runtime_name(env="memory", config={"agent_computer": {"runtime": "chromium"}}) == "memory"
    assert resolve_runtime_name(env="", config={}) == "memory"


def test_wake_idempotent_same_handle_and_lease(tmp_path):
    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    identity = svc.create_identity(ownership=["majed"])
    svc.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)
    agent = agent_principal("majed")
    first_c, first_lease = svc.wake(computer.id, agent)
    handle = svc._handles[computer.id]
    second_c, second_lease = svc.wake(computer.id, agent)
    third_c, third_lease = svc.wake(computer.id, agent)
    assert first_c.id == second_c.id == third_c.id
    assert first_c.active_browser_identity_id == identity.id
    assert second_lease.lease_id == first_lease.lease_id
    assert third_lease.fencing_epoch == first_lease.fencing_epoch
    assert svc._handles[computer.id] is handle
    assert len(svc._handles) == 1
    starts = [e for e in svc.list_audit(computer.id) if e.event_type == "runtime_start"]
    assert len(starts) == 1
    attaches = [e for e in svc.list_audit(computer.id) if e.event_type == "browser_identity_attach"]
    assert len(attaches) == 1


def test_wake_concurrent_retries_converge(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    svc = _svc(tmp_path)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    svc.wake(computer.id, agent)

    def _retry():
        return svc.wake(computer.id, agent)[1].lease_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: _retry(), range(8)))
    assert len(set(ids)) == 1
    assert len(svc._handles) == 1
    starts = [e for e in svc.list_audit(computer.id) if e.event_type == "runtime_start"]
    assert len(starts) == 1


def test_headed_same_host_is_false_on_public_contract(tmp_path):
    svc = _svc(tmp_path)
    contract = AgentComputerContract(svc)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    _, lease = svc.wake(computer.id, agent)
    obs = contract.observe(
        computer.id, agent, lease_id=lease.lease_id, fencing_epoch=lease.fencing_epoch
    )
    assert obs["live_view"]["headed_same_host"] is False
    assert obs["live_view"]["kind"] == "screenshot_on_demand"
    assert obs["live_view"]["remote_stream"] is False
    takeover = contract.request_takeover(computer.id, OWNER_PRINCIPAL)
    assert takeover["live_view"]["headed_same_host"] is False
    assert "headed_same_host\": true" not in str(takeover).lower().replace(" ", "")


def test_pixel_and_text_actions_are_fenced_and_not_audited(tmp_path):
    svc = _svc(tmp_path)
    contract = AgentComputerContract(svc)
    computer = svc.ensure_computer("majed")
    agent = agent_principal("majed")
    _, agent_lease = svc.wake(computer.id, agent)
    contract.act(
        computer.id,
        agent,
        lease_id=agent_lease.lease_id,
        fencing_epoch=agent_lease.fencing_epoch,
        kind="pointer_click",
        x=116,
        y=184,
    )
    takeover = contract.request_takeover(computer.id, OWNER_PRINCIPAL)
    owner = contract.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=takeover["takeover_token"]
    )
    secret = "synthetic-human-796"
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            agent,
            lease_id=agent_lease.lease_id,
            fencing_epoch=agent_lease.fencing_epoch,
            kind="pointer_click",
            x=500,
            y=184,
        )
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            agent,
            lease_id=agent_lease.lease_id,
            fencing_epoch=agent_lease.fencing_epoch,
            kind="text",
            text="STALE-SHOULD-NOT-APPEAR",
        )
    owner_click = contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner["lease_id"],
        fencing_epoch=owner["fencing_epoch"],
        kind="pointer_click",
        x=500,
        y=184,
    )
    assert "owner-pixel-clicked" in owner_click["text"]
    contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner["lease_id"],
        fencing_epoch=owner["fencing_epoch"],
        kind="text",
        text=secret,
    )
    given = contract.give_back(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner["lease_id"],
        fencing_epoch=owner["fencing_epoch"],
    )
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner["lease_id"],
            fencing_epoch=owner["fencing_epoch"],
            kind="pointer_click",
            x=500,
            y=184,
        )
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner["lease_id"],
            fencing_epoch=owner["fencing_epoch"],
            kind="text",
            text=secret,
        )
    blob = str([e.detail for e in svc.list_audit(computer.id)])
    assert secret not in blob
    assert "STALE-SHOULD-NOT-APPEAR" not in blob
    kinds = [e.detail.get("kind") for e in svc.list_audit(computer.id) if e.event_type == "input_accepted"]
    assert "pointer_click" in kinds
    assert "text" in kinds


def test_screenshot_viewport_mapping_identity_and_scale():
    from gateway.agent_computer.pointer import jpeg_dimensions, map_screenshot_to_viewport, mapping_kind

    assert mapping_kind(800, 600, 800, 600) == "1:1"
    assert map_screenshot_to_viewport(100, 50, screenshot_width=800, screenshot_height=600, viewport_width=800, viewport_height=600) == (100.0, 50.0)
    assert mapping_kind(1600, 1200, 800, 600) == "scale"
    x, y = map_screenshot_to_viewport(
        200, 100, screenshot_width=1600, screenshot_height=1200, viewport_width=800, viewport_height=600
    )
    assert (x, y) == (100.0, 50.0)
    assert jpeg_dimensions(b"not-a-jpeg") == (0, 0)


def test_did_not_build_a_new_browser_runtime():
    adapter = Path(__file__).resolve().parents[2] / "gateway" / "agent_computer" / "adapter.py"
    text = adapter.read_text()
    assert "VNC" not in text
    assert "WebRTC" not in text
    assert "Guacamole" not in text
    assert "Browserbase" not in text
    assert "_real_profile_cdp" in text or "remote-debugging-port=0" in text
