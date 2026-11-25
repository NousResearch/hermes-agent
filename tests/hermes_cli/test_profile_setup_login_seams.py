"""Provider-profile setup/sign-in seams (``providers.base`` L6 block) through the real routes.

A model-provider plugin installed in ONE profile's home declares an in-app device-code login, an
Accounts card and a ``hermes model`` step; core drives them with no provider-specific code:
  * ``/api/providers/oauth`` lists the card as ``device_code`` for that profile only;
  * ``/start`` + ``/poll`` run the profile's start/poll hooks and save inside that profile's scope;
  * a cancel before approval never saves;
  * ``hermes model`` dispatch and the effort picker consult the profile.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app
from tests.waiting import wait_until

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}
NAME = "acme-device"

_PLUGIN_SOURCE = '''
import json, os
from providers import register_provider
from providers.base import ProviderProfile

def _log(event, **kw):
    with open(os.environ["ACME_DEVICE_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": event, **kw}) + "\\n")

def start():
    _log("start")
    return {"user_code": "ACME-1234", "verification_url": "https://acme.example/device",
            "expires_in": 60, "interval": 1, "state": {"polls": 0}}

def poll(state):
    from hermes_constants import get_hermes_home
    polls = state["polls"] + 1
    _log("poll", polls=polls, home=str(get_hermes_home()))
    if polls < int(os.environ.get("ACME_APPROVE_AT", "2")):
        return {"status": "pending", "state": {"polls": polls}}
    if os.environ.get("ACME_DENY"):
        return {"status": "denied", "message": "nope"}
    return {"status": "approved", "credential": "acme-secret"}

def save(credential):
    from hermes_constants import get_hermes_home
    _log("save", home=str(get_hermes_home()), credential=credential)

def flow(config, current_model, args):
    _log("flow", current_model=current_model)

register_provider(ProviderProfile(
    name="acme-device", display_name="Acme Device", base_url="https://acme.example/v1",
    env_vars=("ACME_DEVICE_KEY",), account_card={"name": "Acme Account", "docs_url": "https://acme.example/docs"},
    start_device_login=start, poll_device_login=poll, save_device_login=save,
    model_setup_flow=flow, setup_reasoning_efforts=lambda model: () if model == "plain" else ("low", "high"),
))
'''


def _rediscover() -> None:
    import providers as _pkg
    _pkg._discovered = False
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def acme(tmp_path, monkeypatch):
    """Install the fixture plugin into profile ``coder`` only; return (coder home, log reader)."""
    root = tmp_path / "hermes"
    coder = root / "profiles" / "coder"
    plugin_dir = coder / "plugins" / "model-providers" / NAME
    plugin_dir.mkdir(parents=True)
    (coder / "config.yaml").write_text("{}\n")  # identity marker: a bare dir is not a profile
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {NAME}\nkind: model-provider\nversion: 0.0.1\ndescription: device login seam fixture\n")
    (plugin_dir / "__init__.py").write_text(_PLUGIN_SOURCE)
    log = tmp_path / "acme.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("ACME_DEVICE_LOG", str(log))
    _rediscover()

    def events() -> list[dict]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    yield coder, events
    import providers as _pkg
    _pkg._PROVIDER_LIST_CACHE = None
    _rediscover()


def _cards(profile: str | None = None) -> dict[str, dict]:
    url = "/api/providers/oauth" + (f"?profile={profile}" if profile else "")
    resp = client.get(url, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return {p["id"]: p for p in resp.json()["providers"]}


def _wait(session_id: str, profile: str) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        body = client.get(f"/api/providers/oauth/{NAME}/poll/{session_id}?profile={profile}&wait=2",
                          headers=HEADERS).json()
        if body["status"] != "pending":
            return body
    raise AssertionError("device login never settled")


def test_profile_device_login_lists_starts_and_saves_in_the_owning_profile(acme):
    coder, events = acme
    card = _cards("coder")[NAME]
    assert {k: card[k] for k in ("name", "flow", "docs_url")} == {
        "name": "Acme Account", "flow": "device_code", "docs_url": "https://acme.example/docs"}
    assert NAME not in _cards()  # the plugin lives in coder's home only

    start = client.post(f"/api/providers/oauth/{NAME}/start?profile=coder", headers=HEADERS)
    assert start.status_code == 200, start.text
    body = start.json()
    assert {k: body[k] for k in ("flow", "user_code", "verification_url", "expires_in", "poll_interval")} == {
        "flow": "device_code", "user_code": "ACME-1234", "verification_url": "https://acme.example/device",
        "expires_in": 60, "poll_interval": 1}

    assert _wait(body["session_id"], "coder")["status"] == "approved"
    saves = [e for e in events() if e["event"] == "save"]
    assert saves == [{"event": "save", "home": str(coder), "credential": "acme-secret"}]
    assert [e["polls"] for e in events() if e["event"] == "poll"] == [1, 2]


def test_profile_device_login_polls_in_the_owning_profile(acme):
    """The background poller binds the session's ``?profile=`` home around every poll, so a hook that
    reads config (or a hosted profile picking its plugin host) sees coder's home, not the launch one."""
    coder, events = acme
    body = client.post(f"/api/providers/oauth/{NAME}/start?profile=coder", headers=HEADERS).json()
    assert _wait(body["session_id"], "coder")["status"] == "approved"
    assert [e["home"] for e in events() if e["event"] == "poll"] == [str(coder), str(coder)]


def test_profile_device_login_cancel_before_approval_never_saves(acme, monkeypatch):
    _coder, events = acme
    monkeypatch.setenv("ACME_APPROVE_AT", "1")   # the very first poll would approve and save
    body = client.post(f"/api/providers/oauth/{NAME}/start?profile=coder", headers=HEADERS).json()
    resp = client.delete(f"/api/providers/oauth/sessions/{body['session_id']}?profile=coder", headers=HEADERS)
    assert resp.json()["ok"] is True
    poller = f"oauth-poll-{body['session_id'][:6]}"
    wait_until(lambda: not any(t.name == poller for t in threading.enumerate()), timeout=10, what="poller exit")
    assert [e["event"] for e in events()] == ["start"]   # cancelled inside the first wait: no poll, no save


def test_profile_device_login_denied_ends_the_session_with_the_profile_message(acme, monkeypatch):
    _coder, events = acme
    monkeypatch.setenv("ACME_APPROVE_AT", "1")
    monkeypatch.setenv("ACME_DENY", "1")
    body = client.post(f"/api/providers/oauth/{NAME}/start?profile=coder", headers=HEADERS).json()
    result = _wait(body["session_id"], "coder")
    assert (result["status"], result["error_message"]) == ("denied", "nope")
    assert not [e for e in events() if e["event"] == "save"]


def test_provider_without_device_login_cannot_start_one(acme):
    # A built-in external card is still refused; an unknown id is still unknown.
    resp = client.post("/api/providers/oauth/claude-code/start", headers=HEADERS)
    assert resp.status_code == 400 and "external CLI" in resp.json()["detail"]
    assert client.post("/api/providers/oauth/no-such-provider/start", headers=HEADERS).status_code == 400


def test_hermes_model_dispatch_and_effort_picker_consult_the_profile(acme, monkeypatch):
    coder, events = acme
    monkeypatch.setenv("HERMES_HOME", str(coder))
    _rediscover()
    from hermes_cli.main_provider_setup import _main_model_reasoning_efforts
    from hermes_cli.model_setup_flows import _profile_model_setup_flow

    flow = _profile_model_setup_flow(NAME)
    assert flow is not None
    flow({}, "acme-1", None)
    assert [e for e in events() if e["event"] == "flow"] == [{"event": "flow", "current_model": "acme-1"}]
    assert _main_model_reasoning_efforts("acme-1", NAME) == ["low", "high"]
    assert _main_model_reasoning_efforts("plain", NAME) is None
    assert _profile_model_setup_flow("openrouter") is None  # no bespoke step: the generic dispatch
