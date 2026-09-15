"""Origin-bound handles through registry dispatch and a real, dummy-only op subprocess."""
from __future__ import annotations

import json
import sys

import pytest

from agent import secret_scope
from agent.vault_backends.onepassword import OnePasswordLoginBackend
from agent.vault_store import normalize_origin


@pytest.fixture
def manager(tmp_path, monkeypatch):
    from agent import vault_login_classifier as classifier
    from agent import vault_backends
    from agent.vault_backends import base
    from tools import browser_vault_tool as tool
    from tools.registry import registry

    urls = ["not a URL", "https://first.example/login", "https://SECOND.example:443/login",
            "https://second.example/other", "https://third.example:8443/login"]
    data = tmp_path / "items.json"
    item = {"id": "dummy-item", "title": "Example", "additional_information": "dummy-user",
            "urls": [{"href": u} for u in urls]}
    data.write_text(json.dumps([item]))
    log = tmp_path / "calls.jsonl"
    exe = tmp_path / "op"
    exe.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
root = Path(__file__).parent
args = sys.argv[1:]
auth_ok = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN") == "dummy-profile-token"
assert auth_ok and not os.environ.get("OP_CONNECT_TOKEN")
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps({"args": args, "auth_ok": auth_ok}) + "\\n")
if args == ["item", "list", "--categories", "Login", "--format", "json"]:
    print((root / "items.json").read_text())
elif args == ["item", "get", "dummy-item", "--fields", "label=password", "--reveal"]:
    print("dummy-password-value")
elif args == ["item", "get", "dummy-item", "--otp"]:
    print("123789")
else:
    sys.exit(2)
''')
    exe.chmod(0o700)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "poison-launch-token")
    monkeypatch.setenv("OP_CONNECT_TOKEN", "poison-connect-token")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope({"OP_SERVICE_ACCOUNT_TOKEN": "dummy-profile-token"})
    backend = OnePasswordLoginBackend({"binary_path": str(exe)})
    monkeypatch.setattr(vault_backends, "enabled_backends", lambda: [backend])
    monkeypatch.setattr(base, "enabled_backends", lambda: [backend])
    state = {"origin": "https://second.example", "race": False}
    injections = []
    expected = []
    real_build = classifier.build_fill_js

    def build(*args, **kwargs):
        expected.append(kwargs["expected_origin"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(classifier, "build_fill_js", build)
    monkeypatch.setattr(tool, "_focus_bound_origin", lambda *args: None)
    monkeypatch.setattr(tool, "_current_page_origin", lambda *args: state["origin"])
    controls = [{"tag": "input", "type": "password", "id": "pw", "index": 0,
                 "autocomplete": "current-password", "visible": True},
                {"tag": "input", "type": "text", "id": "otp", "index": 1,
                 "autocomplete": "one-time-code", "visible": True}]
    monkeypatch.setattr(tool, "_eval_js", lambda *args: {"success": True, "result": controls})

    def inject(task, expression):
        injections.append(expression)
        return {"success": True, "result": {"refused": "origin_changed"} if state["race"] else {"filled": 1}}

    monkeypatch.setattr(tool, "_eval_js_secret", inject)

    def call(name, **args):
        result = registry.dispatch(name, args, task_id="dummy-origin-test")
        assert isinstance(result, str)
        return json.loads(result)

    def reads():
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert all(c["auth_ok"] for c in calls)
        return [c["args"] for c in calls if c["args"][:2] == ["item", "get"]]

    try:
        yield backend, call, state, injections, expected, data, item, reads
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("host", [pytest.param("linux", marks=pytest.mark.linux_only),
                                 pytest.param("macos", marks=pytest.mark.macos_only)])
def test_each_saved_origin_roundtrips_and_fills_password_and_otp(manager, host, caplog):
    backend, call, state, injections, expected, data, item, reads = manager
    listed = call("browser_vault_list")["items"]
    origins = {normalize_origin(u["href"]) for u in item["urls"] if "://" in u["href"]}
    assert {m["origin"] for m in listed} == origins
    assert len(listed) == len(origins)  # duplicates collapse by normalized origin
    assert len({m["handle"] for m in listed}) == len(origins)
    for meta in listed:
        assert backend.get_meta(meta["handle"]).origin == meta["origin"]
        state["origin"] = meta["origin"]
        for name, secret in [("browser_vault_fill", "dummy-password-value"),
                             ("browser_vault_enter_code", "123789")]:
            result = call(name, handle=meta["handle"])
            assert result["success"], result
            assert result["origin"] == expected[-1] == meta["origin"]
            assert secret in injections[-1] and secret not in json.dumps(result)
    assert {tuple(args[3:]) for args in reads()} == {
        ("--fields", "label=password", "--reveal"), ("--otp",)}
    assert "dummy-password-value" not in json.dumps(listed) + caplog.text
    assert "123789" not in json.dumps(listed) + caplog.text
    # Existing bare handles retain their original first-valid-origin meaning.
    assert backend.get_meta("op:dummy-item").origin == normalize_origin(item["urls"][1]["href"])
    state["origin"] = backend.get_meta("op:dummy-item").origin
    assert call("browser_vault_fill", handle="op:dummy-item")["success"]
    assert call("browser_vault_enter_code", handle="op:dummy-item")["success"]
    # Origin-bound handles survive URL reordering, but not removal.
    item["urls"].reverse()
    data.write_text(json.dumps([item]))
    for meta in listed:
        assert backend.get_meta(meta["handle"]).origin == meta["origin"]
    data.write_text("[]")
    assert all(backend.get_meta(meta["handle"]) is None for meta in listed)
    # An empty profile cannot inherit the poisoned process credentials.
    empty = secret_scope.set_secret_scope({})
    try:
        assert OnePasswordLoginBackend(backend.cfg).list_items() == []
    finally:
        secret_scope.reset_secret_scope(empty)


@pytest.mark.parametrize("host", [pytest.param("linux", marks=pytest.mark.linux_only),
                                 pytest.param("macos", marks=pytest.mark.macos_only)])
@pytest.mark.parametrize("name", ["browser_vault_fill", "browser_vault_enter_code"])
def test_bound_handle_refuses_wrong_origin_revocation_and_navigation(manager, host, name):
    backend, call, state, injections, expected, data, item, reads = manager
    listed = call("browser_vault_list")["items"]
    meta = next(m for m in listed if m["origin"] == "https://second.example")
    for origin in ["https://first.example", "http://second.example", "https://second.example:8443",
                   "https://sub.second.example", "https://second.example.evil.test"]:
        state["origin"] = origin
        before = reads()
        result = call(name, handle=meta["handle"])
        assert result.get("error_type") == "origin_mismatch", result
        assert reads() == before and not injections
    state["origin"] = meta["origin"]
    state["race"] = True
    result = call(name, handle=meta["handle"])
    assert result["error_type"] == "origin_changed", result
    assert expected[-1] == meta["origin"]
    before = reads()
    injections.clear()
    # Metadata is re-read on use: neither a removed URL nor a fabricated binding authorizes a read.
    item["urls"] = [{"href": "https://first.example"}]
    data.write_text(json.dumps([item]))
    assert not call(name, handle=meta["handle"])["success"]
    assert not call(name, handle="op:dummy-item#https://forged.example")["success"]
    assert reads() == before and not injections
    data.write_text("invalid JSON")
    assert not call(name, handle=meta["handle"]).get("success", False)
    assert reads() == before and not injections
