"""Tests for the dashboard's presence-endpoint advertisement (channels Phase 2).

When the dashboard binds beyond loopback with the explicit trusted-LAN
``--insecure`` opt-in, session-presence records should carry a dialable ws
endpoint so other devices that discover a live session know where to attach.
Loopback and OAuth-gated binds advertise nothing.
"""

from unittest.mock import patch

from hermes_cli.web_server import _presence_advertise_endpoint


TOKEN = "tok/with space"
TOKEN_QUERY = "token=tok%2Fwith+space"


def endpoint(host: str, *, auth_required: bool = False) -> str:
    return _presence_advertise_endpoint(host, 8664, auth_required=auth_required, token=TOKEN)


class TestPresenceAdvertiseEndpoint:
    def test_loopback_binds_advertise_nothing(self):
        assert endpoint("127.0.0.1") == ""
        assert endpoint("localhost") == ""
        assert endpoint("::1") == ""
        assert endpoint("") == ""

    def test_oauth_gated_binds_advertise_nothing(self):
        with patch("hermes_cli.web_server._primary_lan_ip", return_value="10.10.20.5"):
            assert endpoint("192.168.1.20", auth_required=True) == ""
            assert endpoint("0.0.0.0", auth_required=True) == ""

    def test_explicit_lan_bind_advertises_itself(self):
        assert (
            endpoint("192.168.1.20")
            == f"ws://192.168.1.20:8664/api/ws?{TOKEN_QUERY}"
        )

    def test_explicit_tailscale_style_host_advertises_itself(self):
        assert (
            endpoint("ko-mac.tailnet.ts.net")
            == f"ws://ko-mac.tailnet.ts.net:8664/api/ws?{TOKEN_QUERY}"
        )

    def test_wildcard_bind_advertises_primary_lan_ip(self):
        with patch("hermes_cli.web_server._primary_lan_ip", return_value="10.10.20.5"):
            assert (
                endpoint("0.0.0.0")
                == f"ws://10.10.20.5:8664/api/ws?{TOKEN_QUERY}"
            )

    def test_wildcard_bind_without_detectable_ip_advertises_nothing(self):
        with patch("hermes_cli.web_server._primary_lan_ip", return_value=""):
            assert endpoint("0.0.0.0") == ""
            assert endpoint("::") == ""
