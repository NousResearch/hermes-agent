"""Regression test: the cua-driver CLI-fallback transport must sanitize the
subprocess environment like every other cua-driver spawn site.

``_CuaDriverSession._call_tool_via_cli()`` (the EAGAIN/silent-empty MCP
fallback) invoked ``subprocess.run`` with no ``env=`` at all, so the
third-party ``cua-driver`` binary inherited the full, unsanitized parent
environment — including provider API keys and other Hermes-managed
secrets that ``_lifecycle_coro``'s primary MCP spawn already strips via
``_sanitize_subprocess_env(cua_driver_child_env())``.
"""

import json
from unittest.mock import MagicMock

from tools.computer_use.cua_backend import _CuaDriverSession


def _make_session() -> _CuaDriverSession:
    # _call_tool_via_cli() doesn't touch any instance state (bridge/session/
    # capabilities); bypass __init__ so the test doesn't need a real
    # _AsyncBridge.
    return object.__new__(_CuaDriverSession)


def _fake_completed_process(stdout: str) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = 0
    return proc


def test_cli_fallback_strips_provider_secret_from_subprocess_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "«redacted:sk-…»")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        "tools.computer_use.cua_backend.resolve_cua_driver_cmd",
        lambda: "/resolved/cua-driver",
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _fake_completed_process(json.dumps({"tree_markdown": "root"}))

    monkeypatch.setattr("subprocess.run", fake_run)

    session = _make_session()
    result = session._call_tool_via_cli("list_windows", {}, timeout=5.0)

    assert result["isError"] is False
    assert captured["env"] is not None, "subprocess.run must receive an explicit env="
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    # Sanitization filters secrets, not everything — an ordinary var survives.
    assert captured["env"].get("PATH") == "/usr/bin:/bin"


def test_cli_fallback_applies_telemetry_policy(monkeypatch):
    """The env should also go through cua_driver_child_env(), like every
    other cua-driver spawn site, not just _sanitize_subprocess_env alone."""
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)
    monkeypatch.setattr(
        "tools.computer_use.cua_backend.resolve_cua_driver_cmd",
        lambda: "/resolved/cua-driver",
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _fake_completed_process(json.dumps({"tree_markdown": "root"}))

    monkeypatch.setattr("subprocess.run", fake_run)

    session = _make_session()
    session._call_tool_via_cli("list_windows", {}, timeout=5.0)

    # cua_driver_child_env() injects this when telemetry is disabled
    # (the default) — confirms the fallback goes through the same helper
    # the sanctioned spawn site uses, not an ad hoc env dict.
    assert captured["env"].get("CUA_DRIVER_RS_TELEMETRY_ENABLED") == "0"
