"""browser_exec ``capture_screenshot()`` on a hidden tab (tools/browser_use_shot.py).

Chrome draws no frame for a hidden tab, and Browser Use cloud never answers a png
``Page.captureScreenshot`` for one, so the harness helper hung for its 60 s IPC timeout.
``browser_exec`` below runs the real harness CLI it launches in production; only the daemon transport
is faked.
"""
import json
import os
import sys
import textwrap
from types import SimpleNamespace

import pytest
from PIL import Image

import tools.browser_use_cli as bu_cli
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud

# sitecustomize on the harness's PYTHONPATH: the real browser_harness CLI runs, only its daemon transport is
# faked. A png Page.captureScreenshot on a hidden target raises the harness's own IPC timeout.
_FAKE_TRANSPORT = textwrap.dedent("""\
    import base64, io, json, os
    from PIL import Image
    import browser_harness.helpers as h

    LOG = os.environ["FAKE_CDP_LOG"]
    state = {"hidden": True, "activate_works": os.environ["FAKE_ACTIVATE_WORKS"] == "1"}

    def log(entry):
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\\n")

    def image(fmt):
        buf = io.BytesIO()
        Image.new("RGB", (8, 6), "red").save(buf, format=fmt.upper())
        return base64.b64encode(buf.getvalue()).decode()

    def cdp(method, session_id=None, _response_timeout=5.0, **params):
        log({"method": method, "timeout": _response_timeout, **params})
        if method == "Runtime.evaluate":
            return {"result": {"value": "hidden" if state["hidden"] else "visible"}}
        if method == "Target.activateTarget":
            if not state["activate_works"]:
                raise RuntimeError("Target.activateTarget wasn't found")
            state["hidden"] = False
            return {}
        if method == "Target.createTarget":
            return {"targetId": "T2"}
        if method == "Target.attachToTarget":
            return {"sessionId": "S2"}
        if method == "Page.captureScreenshot":
            if params.get("format") == "png" and state["hidden"]:
                raise h._IPCResponseTimeout(f"{method} timed out after {_response_timeout:g}s")
            return {"data": image(params["format"])}
        raise AssertionError(method)

    h.cdp = cdp
    h._send = lambda req: {}
    h.current_tab = lambda: {"targetId": "T1", "target_id": "T1", "url": "about:blank", "title": ""}
    import browser_harness.admin as admin, browser_harness.telemetry as telemetry
    admin.ensure_daemon = admin.print_update_banner = lambda *a, **k: None
    admin.daemon_alive = lambda *a, **k: True
    admin.daemon_browser_kind = lambda *a, **k: os.environ.get("FAKE_DAEMON_KIND") or None
    telemetry.is_enabled = lambda: False
    telemetry.capture_cli_event = lambda *a, **k: None
""")


@pytest.fixture
def run_exec(monkeypatch, tmp_path):
    """browser_exec on the local or cloud-provider route; the real harness CLI with the fake transport above."""
    monkeypatch.delenv("BU_NAME", raising=False)
    monkeypatch.delenv("BU_AUTOSPAWN", raising=False)
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu_cli, "_exec_route_is_sidecar", {}, raising=False)
    monkeypatch.setattr(bu_cli, "_driven_daemons", {})
    fake_dir = tmp_path / "fake_transport"
    fake_dir.mkdir()
    (fake_dir / "sitecustomize.py").write_text(_FAKE_TRANSPORT, encoding="utf-8")
    site_dir = bu_cli._harness_site_dir()
    monkeypatch.setattr(bu_cli, "_harness_site_dir", lambda: os.pathsep.join([str(fake_dir), site_dir]))
    # tests/tools/conftest.py hides the CLI; this is _find_cli()'s production command
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, "-m", "browser_harness.run"])
    log = tmp_path / "cdp.jsonl"

    def run(activate_works: bool, cloud: bool = False, code: str = "print(capture_screenshot({shot!r}))",
            autospawn_daemon_kind: str = ""):
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", (lambda: object()) if cloud else (lambda: None))
        if autospawn_daemon_kind:  # direct-API Browser Use config: the harness picks its own browser
            monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: SimpleNamespace(name="browser-use"))
            monkeypatch.setattr(bu_cli, "_use_gateway", lambda cfg: False)
            monkeypatch.setattr(bu_cli, "_browser_use_api_key_set", lambda: True)
        monkeypatch.setenv("FAKE_DAEMON_KIND", autospawn_daemon_kind)
        monkeypatch.setattr(bs, "get_session_info",
                            lambda key: {"cdp_url": f"ws://127.0.0.1:47000/devtools/browser/{key}",
                                         "features": {} if cloud else {"local": True}})
        monkeypatch.setenv("FAKE_CDP_LOG", str(log))
        monkeypatch.setenv("FAKE_ACTIVATE_WORKS", "1" if activate_works else "0")
        shot = tmp_path / "shot.png"
        result = bu_cli.browser_exec(code.format(shot=str(shot)), task_id="t", timeout_s=60)
        if not isinstance(result, str):  # a native screenshot result (image attached)
            result = json.dumps({"success": True, "screenshot_path": str(shot)})
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return json.loads(result), shot, calls

    return run


@pytest.mark.parametrize("as_path", [False, True], ids=["str", "pathlib"])
def test_hidden_tab_screenshot_on_the_users_chrome_never_switches_their_tab(run_exec, as_path):
    # a pathlib.Path on the jpeg fallback, where the extension decides the re-encode
    code = "from pathlib import Path\nprint(capture_screenshot(Path({shot!r})))" if as_path else \
        "print(capture_screenshot({shot!r}))"
    result, shot, calls = run_exec(True, code=code)

    assert result.get("success") is True, result.get("stderr") or result
    with Image.open(shot) as img:
        assert img.format == "PNG"  # the path says .png, so a jpeg fallback is re-encoded
    # the user's visible tab stays put: one bounded png, then jpeg, no activation
    assert "Target.activateTarget" not in [c["method"] for c in calls], calls
    shots = [c for c in calls if c["method"] == "Page.captureScreenshot"]
    assert [s["format"] for s in shots] == ["png", "jpeg"] and shots[0]["timeout"] <= 10, shots


@pytest.mark.parametrize("activate_works", [True, False], ids=["activated", "activation-unsupported"])
def test_hidden_tab_screenshot_on_a_cloud_browser_is_brought_to_front(run_exec, activate_works):
    result, shot, calls = run_exec(activate_works, cloud=True)

    assert result.get("success") is True, result.get("stderr") or result
    with Image.open(shot) as img:
        assert img.format == "PNG"
    shots = [c for c in calls if c["method"] == "Page.captureScreenshot"]
    assert shots[0]["format"] == "png" and shots[0]["timeout"] <= 10, shots
    assert [c["method"] for c in calls].index("Target.activateTarget") < calls.index(shots[0])
    if activate_works:  # the tab was brought to the front, so png answered first time
        assert len(shots) == 1
    else:  # png never comes for a hidden tab: one bounded png, then jpeg
        assert [s["format"] for s in shots] == ["png", "jpeg"]


@pytest.mark.parametrize("cloud", [True, False], ids=["cloud", "local"])
def test_new_tab_is_brought_to_front_only_on_a_cloud_browser(run_exec, cloud):
    result, _, calls = run_exec(True, cloud=cloud, code="print('TAB', new_tab())")

    assert result.get("success") is True, result.get("stderr") or result
    assert "TAB T2" in result.get("output", "") + result.get("stdout", ""), result
    activations = [c for c in calls if c["method"] == "Target.activateTarget"]
    # nobody watches a cloud browser; on the user's own Chrome the visible tab stays put
    assert activations == ([{"method": "Target.activateTarget", "timeout": 5.0, "targetId": "T2"}] if cloud else [])


def test_cloud_new_tab_survives_a_provider_rejecting_activation(run_exec):
    result, _, calls = run_exec(False, cloud=True, code="print('TAB', new_tab())")

    assert result.get("success") is True, result.get("stderr") or result
    assert "TAB T2" in result.get("output", "") + result.get("stdout", ""), result
    assert [c["targetId"] for c in calls if c["method"] == "Target.activateTarget"] == ["T2"]


@pytest.mark.parametrize("daemon_kind", ["local", "cloud"])
def test_autospawn_route_activates_only_when_the_harness_actually_reached_cloud(run_exec, daemon_kind):
    # BU_AUTOSPAWN attaches to a local Chrome that is already listening instead of provisioning a cloud
    # browser: the configured cloud intent says nothing about which browser the daemon is driving.
    code = "print(capture_screenshot({shot!r})); print('TAB', new_tab())"
    result, _, calls = run_exec(True, code=code, autospawn_daemon_kind=daemon_kind)

    assert result.get("success") is True, result.get("stderr") or result
    activations = [c["targetId"] for c in calls if c["method"] == "Target.activateTarget"]
    assert activations == (["T1", "T2"] if daemon_kind == "cloud" else []), calls
