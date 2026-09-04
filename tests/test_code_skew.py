"""Tests for gateway code-skew detection (stale-checkout guard).

Companion to ``tests/test_stale_utils_module_import.py``: that test proves the
crash; these prove the guard that turns it into a clear "restart the gateway"
message before a model switch can hit it.  The dashboard mirror (#86207) lives
here too: ``web_server._dashboard_code_skew_guard`` and the
``/api/model/options`` 503 guard, which protect the Models page from the same
stale-module ImportError after ``hermes update``.
"""

import asyncio
import contextlib

import pytest

from gateway import code_skew


@pytest.fixture(autouse=True)
def _reset_boot_fingerprint(monkeypatch):
    """Each test starts with no recorded boot fingerprint."""
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)


class TestDetectCodeSkew:
    def test_no_boot_fingerprint_means_no_skew(self, monkeypatch):
        # Nothing recorded (e.g. non-git install) -> never a false positive.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def456")
        assert code_skew.detect_code_skew() is None


    def test_drift_is_detected_with_short_revs(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        skew = code_skew.detect_code_skew()
        assert skew == ("abc1234567", "def4567890")




class TestShort:
    def test_shortens_long_sha(self):
        assert code_skew._short("git:refs/heads/main:abcdef0123456789") == "abcdef0123"

    def test_keeps_unresolved_marker(self):
        assert code_skew._short("git:refs/heads/main:unresolved") == "unresolved"

    def test_passes_short_sha_through_untruncated(self):
        assert code_skew._short("git:HEAD:abc1234") == "abc1234"


class TestTurnCodeSkewGuard:
    """Turn-path mirror of the model-switch guard.

    Regression: a serve/gateway kept alive across ``hermes update`` crashed
    with ``cannot import name 'fill_empty_non_final_wire_payload'`` on the
    first inbound message after the update, because the first lazy import on
    the new code path resolved against the stale boot-time module graph. The
    turn handler now refuses with a restart notice instead of crashing.
    """

    def test_turn_guard_returns_none_without_skew(self, monkeypatch):
        from gateway import run

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert run.GatewayRunner._turn_code_skew_guard(None) is None

    def test_turn_guard_message_names_revs_and_restart(self, monkeypatch):
        from gateway import run

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = run.GatewayRunner._turn_code_skew_guard(None)
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "restart the backend" in msg
        # Deployment-agnostic: never hardcode a Linux-only unit name.
        assert "systemctl" not in msg

    def test_handler_refuses_and_never_spawns_agent(self, monkeypatch):
        """A skewed turn delivers the restart notice and never reaches the agent."""
        import types

        from gateway import run

        monkeypatch.setattr(
            code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890")
        )

        notices: list[str] = []
        spawned: list = []

        async def _fake_deliver(self, source, content: str) -> None:
            notices.append(content)

        async def _fake_run_agent(self, *a, **k):
            spawned.append(1)
            return None

        gateway = object.__new__(run.GatewayRunner)
        gateway._deliver_platform_notice = types.MethodType(_fake_deliver, gateway)
        gateway._run_agent = types.MethodType(_fake_run_agent, gateway)

        event = types.SimpleNamespace(
            text="hello", reply_to_message_id=None, reply_to_text=None,
            message_id=42, metadata={}, channel_prompt=None, _moa_config=None,
            message_type=None, timestamp=None,
        )
        source = types.SimpleNamespace(
            platform=None, user_name="u", user_id="1", chat_id="c",
            thread_id=None, chat_type="private",
        )

        async def _call():
            await gateway._handle_message_with_agent(
                event, source, "quick", 0
            )

        asyncio.run(_call())

        assert len(notices) == 1
        assert "restart the backend" in notices[0]
        assert spawned == []


class TestModelSwitchSkewGuard:
    def test_guard_returns_none_without_skew(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert slash_commands._model_switch_skew_guard() is None

    def test_guard_message_names_revs_and_restart(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "hermes gateway restart" in msg


class TestDashboardCodeSkewGuard:
    """Dashboard mirror of the gateway's model-switch skew guard (#86207)."""

    def test_dashboard_guard_returns_none_without_skew(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert web_server._dashboard_code_skew_guard() is None

    def test_dashboard_guard_message_names_revs_and_restart(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = web_server._dashboard_code_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "restart" in msg.lower()
        # Browser-dashboard path: never hardcode a Linux-only unit (#97046).
        assert "systemctl" not in msg

    def test_serve_guard_message_points_at_desktop_backend(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = web_server._dashboard_code_skew_guard()
        assert msg is not None
        assert "Desktop-owned backend" in msg
        assert "systemctl" not in msg
        assert "hermes-dashboard" not in msg


class TestModelOptionsSkewGuard:
    """/api/model/options must refuse with a clear 503 when the dashboard is stale.

    Regression for #86207: a dashboard kept alive across ``hermes update``
    serves stale modules, so the picker's lazy import of names the update added
    (``agent.model_metadata.is_grok_46_family``) raised ImportError and the
    handler collapsed it into a generic 500.  With the guard, the stale process
    surfaces "Restart required" (503) and never reaches the payload build.
    """

    def test_stale_dashboard_returns_503_and_skips_payload_build(self, monkeypatch):
        from fastapi import HTTPException
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))

        payload_calls: list = []
        monkeypatch.setattr(
            "hermes_cli.inventory.build_model_options_payload",
            lambda *a, **k: payload_calls.append(1) or {"providers": []},
        )

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(web_server.get_model_options())

        assert excinfo.value.status_code == 503
        assert "restart" in str(excinfo.value.detail).lower()
        # The stale-import crash site (the payload build) is never reached.
        assert payload_calls == []

    def test_fresh_dashboard_builds_payload_unchanged(self, monkeypatch):
        from hermes_cli import web_server

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)

        async def _fake_run_in_threadpool(func):
            return func()

        monkeypatch.setattr(web_server, "run_in_threadpool", _fake_run_in_threadpool)
        monkeypatch.setattr(
            web_server, "_profile_scope", lambda profile: contextlib.nullcontext()
        )
        monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: {})

        payload_calls: list = []
        expected = {"providers": [], "model": {}, "provider": None}
        monkeypatch.setattr(
            "hermes_cli.inventory.build_model_options_payload",
            lambda *a, **k: payload_calls.append(1) or expected,
        )

        result = asyncio.run(web_server.get_model_options())

        assert result == expected
        assert payload_calls == [1]
