"""Real Chromium/CDP proof for BWM-796.

Synthetic local page only. Never snapshots the owner's browser profile
and never touches a credential-bearing account.
"""

from __future__ import annotations

import base64
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gateway.agent_computer import (
    release_owner_for_transport_if_active,
    reset_service_for_tests,
)
from gateway.agent_computer.adapter import HermesChromiumRuntime
from gateway.agent_computer.contract import AgentComputerContract
from gateway.agent_computer.errors import IdentityBusyError, StaleControllerError
from gateway.agent_computer.models import OWNER_PRINCIPAL, agent_principal
from gateway.agent_computer.pointer import mapping_kind
from gateway.agent_computer.service import AgentComputerService
from gateway.agent_computer.store import AgentComputerStore

pytestmark = pytest.mark.macos_only

_PAGE = """<!doctype html>
<html><head><title>BWM-796 Synthetic</title></head>
<body>
<p id="status">idle-visible</p>
<input id="box" type="text" />
<button id="go" type="button"
  onclick="document.getElementById('status').textContent='clicked-'+document.getElementById('box').value">
  Go
</button>
<button id="owner" type="button"
  onclick="document.getElementById('status').textContent='owner-took-over'">
  Owner
</button>
</body></html>
"""


def _chrome_binary() -> str | None:
    from hermes_cli.browser_connect import chromium_executable, detect_default_chromium

    browser = detect_default_chromium() or "chrome"
    return chromium_executable(browser) or chromium_executable("chrome")


requires_chrome = pytest.mark.skipif(
    _chrome_binary() is None, reason="no Chromium-family binary on this host"
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = (_HUMAN_PAGE if "human" in self.path else _PAGE).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


_HUMAN_PAGE = """<!doctype html>
<html><head><title>BWM-796 Human</title>
<style>
html, body { margin: 0; width: 800px; height: 600px; font-family: sans-serif; }
#status { position: absolute; left: 16px; top: 16px; width: 760px; height: 40px; }
#typed { position: absolute; left: 340px; top: 80px; width: 400px; height: 40px; }
#box { position: absolute; left: 16px; top: 80px; width: 300px; height: 40px; }
#agent { position: absolute; left: 16px; top: 160px; width: 200px; height: 48px; }
#owner { position: absolute; left: 400px; top: 160px; width: 200px; height: 48px; }
#scrollstate { position: absolute; left: 340px; top: 240px; }
#scrollbox { position: absolute; left: 16px; top: 240px; width: 300px; height: 120px; overflow: auto; }
#scrollinner { height: 400px; }
</style>
</head>
<body>
<p id="status">idle-visible</p>
<p id="typed"></p>
<input id="box" type="text" oninput="document.getElementById('typed').textContent=this.value" />
<button id="agent" type="button"
  onclick="document.getElementById('status').textContent='agent-ready'">Agent</button>
<button id="owner" type="button"
  onclick="document.getElementById('status').textContent='owner-pixel-clicked'">Owner</button>
<p id="scrollstate">not-scrolled</p>
<div id="scrollbox" onscroll="document.getElementById('scrollstate').textContent='scrolled'">
  <div id="scrollinner">scroll-pad</div>
</div>
</body></html>
"""


# CSS centers of the fixed layout (viewport 800x600).
_HUMAN_INPUT = (166.0, 100.0)
_HUMAN_AGENT = (116.0, 184.0)
_HUMAN_OWNER = (500.0, 184.0)
_HUMAN_SCROLL = (166.0, 300.0)


@pytest.fixture
def local_page():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/synthetic.html"
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def chromium_pair(tmp_path):
    runtime = HermesChromiumRuntime()
    svc = AgentComputerService(
        AgentComputerStore(tmp_path / "state.db"),
        runtime,
        data_root=tmp_path,
    )
    try:
        yield svc, runtime, tmp_path
    finally:
        for handle in list(svc._handles.values()):
            runtime.sleep(handle)
        reset_service_for_tests()


def _boot(svc, profile="majed", ownership=None):
    computer = svc.ensure_computer(profile)
    identity = svc.create_identity(
        ownership=list(ownership or [profile]),
        metadata={"label": "synthetic"},
    )
    svc.attach_identity(computer.id, identity.id, OWNER_PRINCIPAL)
    agent = agent_principal(profile)
    computer, lease = svc.wake(computer.id, agent)
    return computer, identity, agent, lease


@requires_chrome
def test_real_chromium_takeover_e2e(chromium_pair, local_page):
    svc, runtime, tmp_path = chromium_pair
    contract = AgentComputerContract(svc)
    computer, identity, agent, lease = _boot(svc, ownership=["majed", "abu-saleh"])
    handle = svc._handles[computer.id]

    assert handle.backend == "hermes_chromium"
    assert _pid_alive(handle.process_id)
    assert Path(handle.user_data_dir).resolve() == Path(identity.profile_ref).resolve()
    assert str(tmp_path) in handle.user_data_dir
    assert handle.cdp_loopback and handle.cdp_loopback.startswith("http://127.0.0.1:")
    assert "0.0.0.0" not in handle.cdp_loopback

    first = contract.observe(
        computer.id, agent, lease_id=lease.lease_id, fencing_epoch=lease.fencing_epoch
    )
    assert first["url"]
    assert "cdp" not in str(first).lower()
    assert handle.cdp_loopback not in str(first)

    after_nav = contract.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="navigate",
        target=local_page,
    )
    assert "127.0.0.1" in after_nav["url"]
    seen = contract.observe(
        computer.id, agent, lease_id=lease.lease_id, fencing_epoch=lease.fencing_epoch
    )
    assert "idle-visible" in seen["text"]
    assert seen["title"] == "BWM-796 Synthetic"
    assert seen["screenshot"] and seen["screenshot"]["mime"] == "image/jpeg"
    raw = base64.b64decode(seen["screenshot"]["data"])
    assert raw[:2] == b"\xff\xd8"
    assert seen["live_view"]["headed_same_host"] is False

    contract.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="type",
        target="#box",
        text="synthetic-796",
    )
    clicked = contract.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="click",
        target="#go",
    )
    assert "clicked-synthetic-796" in clicked["text"]
    agent_state = clicked["text"]

    takeover = contract.request_takeover(computer.id, OWNER_PRINCIPAL, reason="e2e")
    owner_lease = contract.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=takeover["takeover_token"]
    )
    assert takeover["status"] == "OWNER_CONTROLLED"
    assert svc.get_computer(computer.id).id == computer.id
    assert svc.get_computer(computer.id).active_browser_identity_id == identity.id
    assert svc._handles[computer.id].process_id == handle.process_id
    assert svc._handles[computer.id].cdp_loopback == handle.cdp_loopback

    owner_view = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    assert "clicked-synthetic-796" in owner_view["text"]
    assert owner_view["screenshot"] and owner_view["screenshot"]["data"]
    assert owner_view["live_view"]["same_environment"] is True

    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            agent,
            lease_id=lease.lease_id,
            fencing_epoch=lease.fencing_epoch,
            kind="type",
            target="#box",
            text="STALE-SHOULD-NOT-APPEAR",
        )
    still = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    assert "STALE-SHOULD-NOT-APPEAR" not in still["text"]
    assert "clicked-synthetic-796" in still["text"]

    owner_clicked = contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="click",
        target="#owner",
    )
    assert "owner-took-over" in owner_clicked["text"]

    given = contract.give_back(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    new_lease = given["lease"]
    assert given["control"] == "AGENT_CONTROLLED"
    assert new_lease["fencing_epoch"] != owner_lease["fencing_epoch"]
    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner_lease["lease_id"],
            fencing_epoch=owner_lease["fencing_epoch"],
            kind="click",
            target="#go",
        )
    resumed = contract.observe(
        computer.id,
        agent,
        lease_id=new_lease["lease_id"],
        fencing_epoch=new_lease["fencing_epoch"],
    )
    assert "owner-took-over" in resumed["text"]
    assert "clicked-synthetic-796" not in resumed["text"]
    assert agent_state != resumed["text"]

    other = svc.ensure_computer("abu-saleh")
    with pytest.raises(IdentityBusyError):
        svc.attach_identity(other.id, identity.id, OWNER_PRINCIPAL)
    assert len(svc._handles) == 1

    marker = Path(identity.profile_ref) / ".hermes-identity"
    assert marker.read_text() == identity.id
    old_pid = handle.process_id
    old_cdp = handle.cdp_loopback
    svc.sleep(computer.id, agent)
    assert not _cdp_up(old_cdp)
    assert not _pid_alive(old_pid)

    reopened = AgentComputerService(
        AgentComputerStore(tmp_path / "state.db"),
        HermesChromiumRuntime(),
        data_root=tmp_path,
    )
    persisted = reopened.store.get_computer_by_profile("majed")
    assert persisted is not None
    assert persisted.id == computer.id
    assert persisted.active_browser_identity_id == identity.id
    assert Path(identity.profile_ref).joinpath(".hermes-identity").exists()
    _, lease2 = reopened.wake(persisted.id, agent)
    handle2 = reopened._handles[persisted.id]
    assert handle2.process_id != old_pid
    assert handle2.cdp_loopback.startswith("http://127.0.0.1:")
    recovered = AgentComputerContract(reopened)
    rec_obs = recovered.observe(
        persisted.id, agent, lease_id=lease2.lease_id, fencing_epoch=lease2.fencing_epoch
    )
    # Process/CDP/DOM are ephemeral. Durable identity/profile dir remains.
    assert rec_obs["url"]
    assert "owner-took-over" not in rec_obs["text"]
    reopened.sleep(persisted.id, agent)


@requires_chrome
def test_real_chromium_ws_disconnect_and_remote_contract(chromium_pair, local_page):
    svc, runtime, tmp_path = chromium_pair
    import gateway.agent_computer as ac

    ac._SERVICE = svc
    contract = AgentComputerContract(svc)
    computer, identity, agent, lease = _boot(svc)
    contract.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="navigate",
        target=local_page,
    )
    takeover = contract.request_takeover(computer.id, OWNER_PRINCIPAL)
    owner_lease = contract.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=takeover["takeover_token"]
    )
    transport = object()
    svc.bind_owner_transport(computer.id, transport)
    # Same function tui_gateway/ws.py calls on WS teardown.
    assert release_owner_for_transport_if_active(transport) == 1
    computer = svc.get_computer(computer.id)
    assert computer.control_authority.value == "AGENT_CONTROLLED"
    assert computer.resume_observe_required is True
    with pytest.raises(StaleControllerError):
        svc.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner_lease["lease_id"],
            fencing_epoch=owner_lease["fencing_epoch"],
            kind="click",
            target="#go",
        )
    events = {e.event_type for e in svc.list_audit(computer.id)}
    assert "owner_disconnect" in events
    assert "fencing_recovery" in events
    reset_service_for_tests()


def _human_url(local_page: str) -> str:
    return local_page.replace("synthetic.html", "human.html")


def _click_point(obs: dict, css_x: float, css_y: float) -> tuple[float, float]:
    """Screenshot pixels of a known CSS-viewport point (what a human click sends)."""
    shot = obs["screenshot"] or {}
    view = obs["viewport"] or {}
    sw = int(shot.get("width") or 0)
    sh = int(shot.get("height") or 0)
    vw = int(view.get("width") or 0)
    vh = int(view.get("height") or 0)
    if vw <= 0 or vh <= 0 or sw <= 0 or sh <= 0:
        return css_x, css_y
    return css_x * sw / vw, css_y * sh / vh


@requires_chrome
def test_real_chromium_human_pixel_takeover(chromium_pair, local_page):
    svc, runtime, tmp_path = chromium_pair
    contract = AgentComputerContract(svc)
    computer, identity, agent, lease = _boot(svc)
    handle = svc._handles[computer.id]
    page = _human_url(local_page)
    secret = "synthetic-human-796"

    contract.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="navigate",
        target=page,
    )
    seen = contract.observe(
        computer.id, agent, lease_id=lease.lease_id, fencing_epoch=lease.fencing_epoch
    )
    assert "idle-visible" in seen["text"]
    shot = seen["screenshot"]
    assert shot and shot["mime"] == "image/jpeg"
    assert shot["width"] > 0 and shot["height"] > 0
    assert seen["viewport"]["width"] > 0 and seen["viewport"]["height"] > 0
    assert seen["live_view"]["headed_same_host"] is False
    assert seen["live_view"]["kind"] == "screenshot_on_demand"
    assert seen["live_view"]["remote_stream"] is False
    assert seen["live_view"]["mapping"] == mapping_kind(
        shot["width"], shot["height"], seen["viewport"]["width"], seen["viewport"]["height"]
    )
    assert seen["live_view"]["mapping"] in {"1:1", "scale"}
    raw = base64.b64decode(shot["data"])
    assert raw[:2] == b"\xff\xd8"
    assert "cdp" not in str(seen).lower()
    assert handle.cdp_loopback not in str(seen)
    assert "user_data_dir" not in str(seen)

    ready = contract.act(
        computer.id,
        agent,
        lease_id=lease.lease_id,
        fencing_epoch=lease.fencing_epoch,
        kind="click",
        target="#agent",
    )
    assert "agent-ready" in ready["text"]

    takeover = contract.request_takeover(computer.id, OWNER_PRINCIPAL, reason="pixel")
    assert takeover["live_view"]["headed_same_host"] is False
    owner_lease = contract.connect_takeover(
        computer.id, OWNER_PRINCIPAL, takeover_token=takeover["takeover_token"]
    )
    assert takeover["status"] == "OWNER_CONTROLLED"
    assert svc.get_computer(computer.id).id == computer.id
    assert svc.get_computer(computer.id).active_browser_identity_id == identity.id
    assert svc._handles[computer.id].process_id == handle.process_id
    assert svc._handles[computer.id].cdp_loopback == handle.cdp_loopback

    owner_view = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    assert "agent-ready" in owner_view["text"]
    ox, oy = _click_point(owner_view, *_HUMAN_OWNER)
    clicked = contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="pointer_click",
        x=ox,
        y=oy,
    )
    assert "owner-pixel-clicked" in clicked["text"]

    after_click = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    ix, iy = _click_point(after_click, *_HUMAN_INPUT)
    contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="pointer_click",
        x=ix,
        y=iy,
    )
    typed = contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="text",
        text=secret,
    )
    assert secret in typed["text"]

    after_type = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    sx, sy = _click_point(after_type, *_HUMAN_SCROLL)
    contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="pointer_click",
        x=sx,
        y=sy,
    )
    scrolled = contract.act(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
        kind="scroll",
        delta_x=0,
        delta_y=240,
    )
    assert "scrolled" in scrolled["text"]
    assert "owner-pixel-clicked" in scrolled["text"]
    assert secret in scrolled["text"]

    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            agent,
            lease_id=lease.lease_id,
            fencing_epoch=lease.fencing_epoch,
            kind="pointer_click",
            x=ox,
            y=oy,
        )
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            agent,
            lease_id=lease.lease_id,
            fencing_epoch=lease.fencing_epoch,
            kind="text",
            text="STALE-SHOULD-NOT-APPEAR",
        )
    still = contract.observe(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    assert "STALE-SHOULD-NOT-APPEAR" not in still["text"]
    assert "owner-pixel-clicked" in still["text"]

    given = contract.give_back(
        computer.id,
        OWNER_PRINCIPAL,
        lease_id=owner_lease["lease_id"],
        fencing_epoch=owner_lease["fencing_epoch"],
    )
    new_lease = given["lease"]
    assert given["control"] == "AGENT_CONTROLLED"
    assert new_lease["fencing_epoch"] != owner_lease["fencing_epoch"]
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner_lease["lease_id"],
            fencing_epoch=owner_lease["fencing_epoch"],
            kind="pointer_click",
            x=ox,
            y=oy,
        )
    with pytest.raises(StaleControllerError):
        contract.act(
            computer.id,
            OWNER_PRINCIPAL,
            lease_id=owner_lease["lease_id"],
            fencing_epoch=owner_lease["fencing_epoch"],
            kind="text",
            text=secret,
        )
    resumed = contract.observe(
        computer.id,
        agent,
        lease_id=new_lease["lease_id"],
        fencing_epoch=new_lease["fencing_epoch"],
    )
    assert "owner-pixel-clicked" in resumed["text"]
    assert secret in resumed["text"]
    assert "agent-ready" not in resumed["text"] or "owner-pixel-clicked" in resumed["text"]
    blob = str([e.detail for e in svc.list_audit(computer.id)])
    assert secret not in blob
    assert "STALE-SHOULD-NOT-APPEAR" not in blob


@requires_chrome
def test_real_chromium_wake_idempotent(chromium_pair, local_page):
    from concurrent.futures import ThreadPoolExecutor

    svc, runtime, tmp_path = chromium_pair
    computer, identity, agent, lease = _boot(svc)
    handle = svc._handles[computer.id]
    pid = handle.process_id
    cdp = handle.cdp_loopback

    again_c, again_lease = svc.wake(computer.id, agent)
    assert again_c.id == computer.id
    assert again_c.active_browser_identity_id == identity.id
    assert again_lease.lease_id == lease.lease_id
    assert again_lease.fencing_epoch == lease.fencing_epoch
    assert svc._handles[computer.id].process_id == pid
    assert svc._handles[computer.id].cdp_loopback == cdp
    assert len(svc._handles) == 1

    def _retry():
        return svc.wake(computer.id, agent)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [fut.result() for fut in [pool.submit(_retry) for _ in range(3)]]
    assert all(c.id == computer.id for c, _ in results)
    assert all(l.lease_id == lease.lease_id for _, l in results)
    assert svc._handles[computer.id].process_id == pid
    assert _pid_alive(pid)
    starts = [e for e in svc.list_audit(computer.id) if e.event_type == "runtime_start"]
    assert len(starts) == 1
    attaches = [e for e in svc.list_audit(computer.id) if e.event_type == "browser_identity_attach"]
    assert len(attaches) == 1


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _cdp_up(url: str | None) -> bool:
    if not url:
        return False
    import urllib.request

    try:
        with urllib.request.urlopen(url.rstrip("/") + "/json/version", timeout=1):
            return True
    except Exception:
        return False
