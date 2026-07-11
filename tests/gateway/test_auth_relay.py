"""Tests for the gateway auth-relay subsystem (operator WhatsApp relay).

These exercise the real relay mechanics in gateway/auth_relay.py:
  * config parsing / enable gating
  * secret capture round-trip (persisted via hermes_cli.config.save_env_value_secure)
  * sudo password round-trip
  * timeout safety (no hang, returns sentinel)
  * operator-only pending-query resolution
  * inbound resolve_* functions unblock the waiting thread

No network: the WhatsApp send is monkeypatched, so the callbacks run
synchronously and deterministically.  HERMES_HOME is redirected to a temp dir
by the conftest autouse fixture.
"""

import os
import sys
import time
import threading
import tempfile
import importlib

import pytest

# Make the repo importable regardless of CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _import_relay():
    import gateway.auth_relay as relay
    importlib.reload(relay)
    return relay


@pytest.fixture
def relay_module():
    """Fresh relay module + cleared state per test."""
    relay = _import_relay()
    relay.clear_all()
    # Default: feature off.
    relay.configure(
        relay.AuthRelayConfig(
            enabled=False, operator_chat="", timeout=10
        )
    )
    yield relay
    relay.clear_all()
    relay.configure(relay.AuthRelayConfig(enabled=False, operator_chat="", timeout=10))


class TestConfig:
    def test_disabled_by_default(self, relay_module):
        assert relay_module.is_enabled() is False

    def test_enable_requires_operator(self, relay_module):
        # enabled but no operator → still inert.
        relay_module.configure(
            relay_module.AuthRelayConfig(enabled=True, operator_chat="", timeout=10)
        )
        assert relay_module.is_enabled() is False

    def test_enable_with_operator(self, relay_module):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True, operator_chat="85251234567", timeout=10
            )
        )
        assert relay_module.is_enabled() is True
        assert relay_module.kind_enabled("secret") is True
        assert relay_module.kind_enabled("sudo") is True

    def test_per_kind_opt_out(self, relay_module):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True,
                operator_chat="85251234567",
                secret=False,
                sudo=True,
                timeout=10,
            )
        )
        assert relay_module.kind_enabled("secret") is False
        assert relay_module.kind_enabled("sudo") is True


class TestSecretRelay:
    def _drive_two_step(self, relay_module, monkeypatch, confirm_word, value):
        """fake_send that answers BOTH the confirm and value prompts."""
        relay_module._whatsapp_operator_adapter = lambda: object()
        sent = []

        def fake_send(text):
            sent.append(text)
            op = "85251234567"
            # First prompt is the confirm; second is the value.
            rid = relay_module.pending_secret_for(op)
            if rid is None:
                return True
            # Decide what to reply based on which prompt this is.
            if "Reply *yes*" in text or "reply *yes*" in text.lower():
                reply = confirm_word
            else:
                reply = value
            threading.Thread(
                target=lambda: relay_module.resolve_secret_relay(rid, reply),
                daemon=True,
            ).start()
            return True

        monkeypatch.setattr(relay_module, "_send_to_operator", fake_send)
        return sent

    def test_secret_round_trip_persists(self, relay_module, tmp_path, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True,
                operator_chat="85251234567",
                timeout=15,
                require_confirm=False,
            )
        )
        sent = []

        def fake_send(text):
            sent.append(text)
            rid = relay_module.pending_secret_for("85251234567")
            if rid:
                threading.Thread(
                    target=lambda: relay_module.resolve_secret_relay(
                        rid, "s3cr3t-value"
                    ),
                    daemon=True,
                ).start()
            return True

        monkeypatch.setattr(relay_module, "_send_to_operator", fake_send)

        result = relay_module._secret_capture_callback(
            "MY_API_KEY", "Enter your API key", None
        )
        assert result["success"] is True
        assert result["stored_as"] == "MY_API_KEY"
        assert "MY_API_KEY" in sent[0]
        dotenv = tmp_path / ".env"
        assert dotenv.exists()
        content = dotenv.read_text()
        assert "MY_API_KEY" in content
        assert "s3cr3t-value" in content

    def test_secret_confirm_flow_persists(self, relay_module, tmp_path, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True,
                operator_chat="85251234567",
                timeout=15,
                require_confirm=True,
            )
        )
        sent = self._drive_two_step(relay_module, monkeypatch, "yes", "tok-en-123")

        result = relay_module._secret_capture_callback(
            "MY_TOKEN", "Enter token", None
        )
        assert result["success"] is True
        assert result["stored_as"] == "MY_TOKEN"
        # Two messages were sent: confirm + value.
        assert len(sent) == 2
        dotenv = tmp_path / ".env"
        content = dotenv.read_text()
        assert "MY_TOKEN" in content
        assert "tok-en-123" in content

    def test_secret_skip_at_confirm(self, relay_module, monkeypatch):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True,
                operator_chat="85251234567",
                timeout=5,
                require_confirm=True,
            )
        )
        relay_module._whatsapp_operator_adapter = lambda: object()

        def fake_send(text):
            rid = relay_module.pending_secret_for("85251234567")
            if rid:
                threading.Thread(
                    target=lambda: relay_module.resolve_secret_relay(rid, "no"),
                    daemon=True,
                ).start()
            return True

        monkeypatch.setattr(relay_module, "_send_to_operator", fake_send)
        result = relay_module._secret_capture_callback("X", "p", None)
        assert result["success"] is True
        assert result["skipped"] is True


class TestSudoRelay:
    def test_sudo_round_trip(self, relay_module, monkeypatch):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True, operator_chat="85251234567", timeout=15
            )
        )
        relay_module._whatsapp_operator_adapter = lambda: object()

        def fake_send(text):
            rid = relay_module.pending_sudo_for("85251234567")
            if rid:
                threading.Thread(
                    target=lambda: relay_module.resolve_sudo_relay(rid, "p@ssw0rd"),
                    daemon=True,
                ).start()
            return True

        monkeypatch.setattr(relay_module, "_send_to_operator", fake_send)
        pw = relay_module._sudo_password_callback()
        assert pw == "p@ssw0rd"

    def test_sudo_skip_returns_empty(self, relay_module, monkeypatch):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True, operator_chat="85251234567", timeout=5
            )
        )
        relay_module._whatsapp_operator_adapter = lambda: object()

        def fake_send(text):
            rid = relay_module.pending_sudo_for("85251234567")
            if rid:
                threading.Thread(
                    target=lambda: relay_module.resolve_sudo_relay(rid, None),
                    daemon=True,
                ).start()
            return True

        monkeypatch.setattr(relay_module, "_send_to_operator", fake_send)
        assert relay_module._sudo_password_callback() == ""


class TestTimeoutSafety:
    def test_secret_timeout_no_hang(self, relay_module, monkeypatch):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True,
                operator_chat="85251234567",
                timeout=1,
                require_confirm=False,
            )
        )
        relay_module._whatsapp_operator_adapter = lambda: object()
        monkeypatch.setattr(relay_module, "_send_to_operator", lambda text: True)
        start = time.monotonic()
        result = relay_module._secret_capture_callback("X", "p", None)
        elapsed = time.monotonic() - start
        assert elapsed < 5  # didn't block on the 1s timeout for long
        assert result["skipped"] is True
        assert result["success"] is True  # skip shape, like the CLI path

    def test_sudo_timeout_no_hang(self, relay_module, monkeypatch):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True, operator_chat="85251234567", timeout=1
            )
        )
        relay_module._whatsapp_operator_adapter = lambda: object()
        monkeypatch.setattr(relay_module, "_send_to_operator", lambda text: True)
        start = time.monotonic()
        pw = relay_module._sudo_password_callback()
        assert time.monotonic() - start < 5
        assert pw == ""


class TestOperatorGating:
    def test_pending_query_ignores_wrong_operator(self, relay_module):
        relay_module.configure(
            relay_module.AuthRelayConfig(
                enabled=True, operator_chat="85251234567", timeout=10
            )
        )
        relay_module._whatsapp_operator_adapter = lambda: object()
        relay_module._register("secret", "X", "p", None)
        # A different sender must not see the pending entry.
        assert relay_module.pending_secret_for("99999999999") is None
        # The configured operator sees it.
        assert relay_module.pending_secret_for("85251234567") is not None

    def test_resolve_unknown_id_is_noop(self, relay_module):
        assert relay_module.resolve_secret_relay("nope", "v") is False
        assert relay_module.resolve_sudo_relay("nope", "v") is False


class TestConfigParse:
    def test_parse_gateway_auth_relay(self):
        from gateway.config import _parse_auth_relay

        cfg = _parse_auth_relay(
            {
                "enabled": True,
                "operator_chat": "85251234567",
                "secret": False,
                "sudo": True,
                "timeout": 45,
            }
        )
        assert cfg.enabled is True
        assert cfg.operator_chat == "85251234567"
        assert cfg.secret is False
        assert cfg.sudo is True
        assert cfg.timeout == 45

    def test_parse_missing_is_disabled(self):
        from gateway.config import _parse_auth_relay

        assert _parse_auth_relay(None).enabled is False
        assert _parse_auth_relay({}).enabled is False

    def test_parse_no_operator_warns_disabled(self, monkeypatch):
        from gateway.config import _parse_auth_relay

        # Ensure no local WhatsApp number is available to derive from.
        for v in (
            "WHATSAPP_ALLOWED_USERS",
            "WHATSAPP_CLOUD_PHONE",
            "WHATSAPP_CLOUD_BUSINESS_PHONE",
            "WHATSAPP_CLOUD_OWNER_WA_ID",
        ):
            monkeypatch.delenv(v, raising=False)
        cfg = _parse_auth_relay({"enabled": True, "operator_chat": ""})
        assert cfg.enabled is False

    def test_parse_no_operator_derives_from_local(self, monkeypatch):
        from gateway.config import _parse_auth_relay

        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "+85251234567")
        for v in (
            "WHATSAPP_CLOUD_PHONE",
            "WHATSAPP_CLOUD_BUSINESS_PHONE",
            "WHATSAPP_CLOUD_OWNER_WA_ID",
        ):
            monkeypatch.delenv(v, raising=False)
        cfg = _parse_auth_relay({"enabled": True, "operator_chat": ""})
        assert cfg.enabled is True
        # Derived + normalized (no "+").
        assert cfg.operator_chat == "85251234567"
