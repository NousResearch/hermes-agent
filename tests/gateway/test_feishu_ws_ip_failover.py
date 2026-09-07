"""Per-IP WS connect failover for msg-frontier DNS black-holes (#89929).

msg-frontier.feishu.cn's DNS pool contains black-hole IPs: without a per-IP
deadline the whole connect attempt hangs on the first address even though a
healthy IP is one dial away. These tests pin the walk order and the skip.
"""

import asyncio
import socket as _socket
import sys
import unittest
from types import ModuleType
from typing import Any
from unittest.mock import patch

from types import SimpleNamespace


class FeishuWSIPFailoverTests(unittest.TestCase):
    def _install_fake_lark_module(self, connect_impl):
        connect_attempts: list = []

        async def fake_connect(url, **kw):
            host = kw.get("host")
            connect_attempts.append(host)
            return await connect_impl(url, host)

        fake_client_module = ModuleType("lark_oapi.ws.client")
        fake_client_module.loop = None
        fake_client_module.logger = SimpleNamespace(
            info=lambda *_a, **_k: None,
            debug=lambda *_a, **_k: None,
            error=lambda *_a, **_k: None,
            warning=lambda *_a, **_k: None,
        )
        fake_client_module.websockets = SimpleNamespace(connect=fake_connect)
        fake_ws_module = ModuleType("lark_oapi.ws")
        fake_ws_module.client = fake_client_module
        fake_root_module = ModuleType("lark_oapi")
        fake_root_module.ws = fake_ws_module

        originals = {
            name: sys.modules.get(name)
            for name in ("lark_oapi", "lark_oapi.ws", "lark_oapi.ws.client")
        }
        sys.modules["lark_oapi"] = fake_root_module
        sys.modules["lark_oapi.ws"] = fake_ws_module
        sys.modules["lark_oapi.ws.client"] = fake_client_module
        self._originals = originals
        return fake_client_module, connect_attempts

    def tearDown(self):
        for name, mod in getattr(self, "_originals", {}).items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        # _install_lark_ws_isolation flips a process-wide flag; reset so other
        # tests in the same run can install against their own fake modules.
        import plugins.platforms.feishu.adapter as _adapter_mod
        _adapter_mod._WS_ISOLATION_INSTALLED = False

    def test_blackhole_first_ip_is_skipped(self):
        """A hanging first address must be skipped within the per-IP deadline
        and the connection established via the next resolved IP."""
        import plugins.platforms.feishu.adapter as adapter_mod

        async def connect_impl(url, host):
            if host == "9.9.9.9":
                await asyncio.sleep(60)  # black hole: handshake never completes
            return SimpleNamespace(peer=host)

        fake_client_module, attempts = self._install_fake_lark_module(connect_impl)

        resolved = [
            (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("9.9.9.9", 443)),
            (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        ]
        with patch.object(_socket, "getaddrinfo", return_value=resolved), \
                patch.object(adapter_mod, "_WS_CONNECT_PER_IP_TIMEOUT", 0.2):
            # Install order mirrors production: isolation dispatcher first,
            # failover walk wraps it.
            adapter_mod._install_lark_ws_isolation(fake_client_module)
            adapter_mod._install_ws_ip_failover(fake_client_module)

            conn_url = "wss://msg-frontier.feishu.cn/ws?device_id=d1&service_id=s1"
            conn = asyncio.new_event_loop().run_until_complete(
                fake_client_module.websockets.connect(conn_url)
            )

        self.assertEqual(attempts, ["9.9.9.9", "8.8.8.8"])
        self.assertEqual(getattr(conn, "peer", None), "8.8.8.8")

    def test_all_ips_exhausted_raises_last_error(self):
        """When every resolved IP fails, the last error surfaces (never a silent None)."""
        import plugins.platforms.feishu.adapter as adapter_mod

        async def connect_impl(url, host):
            raise OSError(f"refused by {host}")

        fake_client_module, attempts = self._install_fake_lark_module(connect_impl)

        resolved = [
            (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("9.9.9.9", 443)),
            (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        ]
        with patch.object(_socket, "getaddrinfo", return_value=resolved), \
                patch.object(adapter_mod, "_WS_CONNECT_PER_IP_TIMEOUT", 0.2):
            adapter_mod._install_lark_ws_isolation(fake_client_module)
            adapter_mod._install_ws_ip_failover(fake_client_module)

            conn_url = "wss://msg-frontier.feishu.cn/ws?device_id=d1&service_id=s1"
            with self.assertRaises(OSError):
                asyncio.new_event_loop().run_until_complete(
                    fake_client_module.websockets.connect(conn_url)
                )
        self.assertEqual(attempts, ["9.9.9.9", "8.8.8.8"])

    def test_install_is_idempotent(self):
        """Double install must not double-wrap (the walk would then try each IP twice)."""
        import plugins.platforms.feishu.adapter as adapter_mod

        async def connect_impl(url, host):
            return SimpleNamespace(peer=host)

        fake_client_module, attempts = self._install_fake_lark_module(connect_impl)
        adapter_mod._install_ws_ip_failover(fake_client_module)
        once = fake_client_module.websockets.connect
        adapter_mod._install_ws_ip_failover(fake_client_module)
        self.assertIs(fake_client_module.websockets.connect, once)


if __name__ == "__main__":
    unittest.main()
