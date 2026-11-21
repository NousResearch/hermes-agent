"""Real plugin discovery keeps submission permission and ownership bound to the loading home."""
import json
import os
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


PROBE = '''
import json
import os

def register(ctx):
    def probe(raw_args):
        args = json.loads(raw_args)
        content = args.pop("content", None)
        receipt = (ctx.submission_status(**args) if content is None
                   else ctx.submit_message(content, **args))
        return json.dumps({"pid": os.getpid(), "receipt": receipt})
    ctx.register_command("receipt-probe", probe)
'''


def _install(home, *, allowed, isolation="in_process"):
    directory = home / "plugins" / "receipt-probe"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: receipt-probe\nversion: '1.0'\n", encoding="utf-8")
    (directory / "__init__.py").write_text(PROBE, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["receipt-probe"], "isolation": isolation,
        "entries": {"receipt-probe": {"allow_gateway_injection": allowed}},
    }}), encoding="utf-8")


class Gateway:
    """Only the gateway transport is replaced; discovery, contexts and host RPC are real."""

    def __init__(self):
        self.calls = []

    def _plugin_submission(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "status": "queued", "request_id": kwargs["request_id"],
                "session_id": kwargs["expected_session_id"], "admission_id": "receipt",
                "outcome": None, "result": None}


def _load(home, gateway):
    token = set_hermes_home_override(home)
    try:
        manager = PluginManager(scope_key=str(home))
        manager.discover_and_load()
        assert manager._plugins["receipt-probe"].error is None
        manager.set_gateway_message_injector(gateway, lambda **kwargs: False)
        return manager
    finally:
        reset_hermes_home_override(token)


def _invoke(manager, *, content="work"):
    args = {"session_key": "route", "expected_session_id": "conversation", "request_id": "retry"}
    if content is not None:
        args["content"] = content
    return json.loads(manager._plugin_commands["receipt-probe"]["handler"](json.dumps(args)))


def _close(manager):
    manager.unload()
    host = getattr(manager, "_plugin_host_instance", None)
    if host is not None:
        host.shutdown()


@pytest.fixture
def homes(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    empty = tmp_path / "bundled"
    empty.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    return a, b


def test_discovery_a_b_a_never_borrows_permission_or_ambient_home(homes):
    a, b = homes
    _install(a, allowed=True)
    _install(b, allowed=False)
    gateway = Gateway()
    first = _load(a, gateway)
    second = _load(b, gateway)
    try:
        for home, manager, allowed in [(a, first, True), (b, second, False), (a, first, True)]:
            token = set_hermes_home_override(home)
            try:
                result = _invoke(manager)["receipt"]
                assert result["ok"] is allowed
                if not allowed:
                    assert result["status"] == "refused" and result["error"] == "permission_denied"
            finally:
                reset_hermes_home_override(token)
        token = set_hermes_home_override(b)
        try:
            assert _invoke(first)["receipt"]["ok"]
        finally:
            reset_hermes_home_override(token)
        assert len(gateway.calls) == 3
        assert all(call["plugin_home"] == a and call["plugin_id"] == "receipt-probe"
                   for call in gateway.calls)
    finally:
        _close(second)
        _close(first)


@pytest.mark.platforms("any")
def test_hosted_plugin_submission_and_status_are_plain_json(homes):
    a, _ = homes
    _install(a, allowed=True, isolation="host")
    gateway = Gateway()
    manager = _load(a, gateway)
    try:
        submitted = _invoke(manager)
        status = _invoke(manager, content=None)
        assert submitted["pid"] != os.getpid() and status["pid"] == submitted["pid"]
        assert submitted["receipt"] == status["receipt"]
        assert submitted["receipt"]["admission_id"] == "receipt"
        assert len(gateway.calls) == 2
        assert all(call["plugin_home"] == a for call in gateway.calls)
    finally:
        _close(manager)
