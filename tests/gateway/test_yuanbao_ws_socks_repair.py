"""test_yuanbao_ws_socks_repair.py - Verify the Yuanbao WS dial repairs an unusable system SOCKS proxy.

Regression test for #122708: ``websockets.connect()`` without an explicit ``proxy=``
picks its proxy from ``urllib.request.getproxies()``. On macOS that reads the system
proxy (SystemConfiguration), so a Clash/Surge/V2RayU SOCKS entry is selected even with
no proxy env var set — and since ``python-socks`` is not part of the dependency set,
every dial raises ``ImportError: python-socks is required to use a SOCKS proxy`` and
the platform reconnect-loops forever (while the sign-token HTTP call connects fine).

``_resolve_ws_proxy_override`` must mirror websockets' own selection and intervene only
when the selected proxy is a SOCKS proxy websockets cannot dial, falling back to the
system HTTP(S) proxy (a plain CONNECT tunnel needs no python-socks) or a direct
connection.
"""

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from websockets.uri import get_proxy, parse_uri

from gateway.platforms.yuanbao import (
    ConnectionManager,
    YuanbaoAdapter,
    _proxy_log_mode,
    _resolve_ws_proxy_override,
)

_WS_URL = "wss://bot-wss.yuanbao.tencent.com/wss/connection"


def _proxies(mapping):
    return patch("urllib.request.getproxies", return_value=dict(mapping))


def _bypass(value=False):
    return patch("urllib.request.proxy_bypass", return_value=value)


def _runtime(available):
    return patch("gateway.platforms.yuanbao._socks_runtime_available", return_value=available)


class TestResolveWsProxyOverride:
    def test_no_system_proxy_leaves_default_alone(self):
        with _proxies({}), _bypass(), _runtime(False):
            assert _resolve_ws_proxy_override(_WS_URL) == (False, None)

    def test_https_proxy_entry_leaves_default_alone(self):
        # An http(s):// proxy is a plain CONNECT tunnel for websockets — no repair needed.
        with _proxies({"https": "http://127.0.0.1:7890"}), _bypass(), _runtime(False):
            assert _resolve_ws_proxy_override(_WS_URL) == (False, None)

    def test_socks_entry_without_runtime_falls_back_to_https_proxy(self):
        # #122708 main scenario: Clash sets HTTP+SOCKS system proxies at once.
        with (
            _proxies({"socks": "socks5h://127.0.0.1:1080", "https": "http://127.0.0.1:7890"}),
            _bypass(),
            _runtime(False),
        ):
            assert _resolve_ws_proxy_override(_WS_URL) == (True, "http://127.0.0.1:7890")

    def test_socks_entry_without_runtime_and_no_http_entry_forces_direct(self):
        with _proxies({"socks": "socks5h://127.0.0.1:1080"}), _bypass(), _runtime(False):
            assert _resolve_ws_proxy_override(_WS_URL) == (True, None)

    def test_socks_entry_with_runtime_leaves_default_alone(self):
        with _proxies({"socks": "socks5h://127.0.0.1:1080"}), _bypass(), _runtime(True):
            assert _resolve_ws_proxy_override(_WS_URL) == (False, None)

    def test_bare_socks_spelling_repaired_even_with_runtime(self):
        # Windows-registry style ``socks://`` fails websockets' parse_proxy() outright.
        with (
            _proxies({"socks": "socks://127.0.0.1:1080", "https": "http://127.0.0.1:7890"}),
            _bypass(),
            _runtime(True),
        ):
            assert _resolve_ws_proxy_override(_WS_URL) == (True, "http://127.0.0.1:7890")

    def test_http_spelling_socks_entry_without_runtime_falls_back(self):
        # ``socks=http://host:port`` (env spelling) is rewritten to socks5h:// by websockets,
        # so it still needs the python-socks runtime.
        with (
            _proxies({"socks": "http://127.0.0.1:1080", "http": "http://127.0.0.1:7890"}),
            _bypass(),
            _runtime(False),
        ):
            assert _resolve_ws_proxy_override(_WS_URL) == (True, "http://127.0.0.1:7890")

    def test_proxy_bypass_entry_leaves_default_alone(self):
        with (
            _proxies({"socks": "socks5h://127.0.0.1:1080"}),
            _bypass(True),
            _runtime(False),
        ):
            assert _resolve_ws_proxy_override(_WS_URL) == (False, None)

    _PARITY_PROXY_DICTS = [
        {},  # nothing selected
        {"https": "http://127.0.0.1:7890"},  # plain https entry
        {"wss": "http://127.0.0.1:7890"},  # ws-specific entry outranks the rest
        {"http": "http://127.0.0.1:7890"},  # ignored for a secure URI
        {"https": "socks5h://127.0.0.1:1080"},  # SOCKS spelled under the https key
        {"socks": "http://127.0.0.1:1080"},  # http spelling under the socks key
        {"socks": "socks5h://127.0.0.1:1080", "https": "http://127.0.0.1:7890"},  # #122708
        {"socks": "socks://127.0.0.1:1080", "https": "http://127.0.0.1:7890"},  # bare socks://
    ]

    @pytest.mark.parametrize("proxies", _PARITY_PROXY_DICTS)
    @pytest.mark.parametrize("ws_url", [_WS_URL, "ws://bot-ws.example.test/wss/connection"])
    @pytest.mark.parametrize("runtime", [True, False])
    def test_parity_with_websockets_proxy_selection(self, ws_url, proxies, runtime):
        """The mirror must track websockets' own selection, not just today's pin of it.

        ``websockets==15.0.1`` is pinned now, but a pin bump that changes
        ``websockets.uri.get_proxy`` would otherwise drift silently. Under the same
        ``getproxies()`` / ``proxy_bypass()`` view, whatever websockets itself selects
        decides the expected outcome: a non-SOCKS (or absent) selection must leave the
        default behaviour untouched, and a SOCKS selection websockets cannot dial must
        be repaired.
        """
        with _proxies(proxies), _bypass(), _runtime(runtime):
            selected = get_proxy(parse_uri(ws_url))
            override, proxy_url = _resolve_ws_proxy_override(ws_url)
        socks_selected = selected is not None and selected.lower().startswith("socks")
        if not socks_selected:
            assert (override, proxy_url) == (False, None)
        elif runtime and not selected.startswith("socks://"):
            # socks5h/socks5/socks4a/socks4 spellings parse fine — websockets dials them itself.
            assert (override, proxy_url) == (False, None)
        else:
            # python-socks missing, or the bare socks:// spelling parse_proxy() rejects: repair.
            expected = next(
                (
                    proxies[scheme]
                    for scheme in ("https", "http")
                    if proxies.get(scheme)
                    and not proxies[scheme].lower().split("://", 1)[0].startswith("socks")
                ),
                None,
            )
            assert (override, proxy_url) == (True, expected)


class TestProxyLogMode:
    def test_direct_connection(self):
        assert _proxy_log_mode(None) == "direct connection"

    def test_scheme_only_never_credentials(self):
        assert _proxy_log_mode("http://user:secret@127.0.0.1:7890") == "http proxy"
        assert _proxy_log_mode("socks5h://127.0.0.1:1080") == "socks5h proxy"


def _make_adapter():
    adapter = MagicMock(spec=YuanbaoAdapter)
    adapter.name = "yuanbao"
    adapter._ws_url = _WS_URL
    return adapter


class TestDialWiring:
    @pytest.mark.asyncio
    async def test_dial_passes_explicit_proxy_when_repairing(self):
        cm = ConnectionManager(_make_adapter())
        mock_ws = MagicMock()
        with (
            patch("gateway.platforms.yuanbao.websockets.connect", new_callable=AsyncMock, return_value=mock_ws) as mock_connect,
            patch.object(cm, "_authenticate", new_callable=AsyncMock, return_value=True),
            patch("gateway.platforms.yuanbao._resolve_ws_proxy_override", return_value=(True, None)),
        ):
            assert await cm._dial({"bot_id": "b1", "token": "t"}) is True
        assert mock_connect.call_args.kwargs["proxy"] is None

    @pytest.mark.asyncio
    async def test_dial_omits_proxy_kwarg_when_no_repair(self):
        cm = ConnectionManager(_make_adapter())
        mock_ws = MagicMock()
        with (
            patch("gateway.platforms.yuanbao.websockets.connect", new_callable=AsyncMock, return_value=mock_ws) as mock_connect,
            patch.object(cm, "_authenticate", new_callable=AsyncMock, return_value=True),
            patch("gateway.platforms.yuanbao._resolve_ws_proxy_override", return_value=(False, None)),
        ):
            assert await cm._dial({"bot_id": "b1", "token": "t"}) is True
        assert "proxy" not in mock_connect.call_args.kwargs

    @pytest.mark.asyncio
    async def test_dial_repair_log_leaks_no_proxy_credentials(self, caplog):
        cm = ConnectionManager(_make_adapter())
        mock_ws = MagicMock()
        credentialed = "http://user:secret@127.0.0.1:7890"
        with (
            patch("gateway.platforms.yuanbao.websockets.connect", new_callable=AsyncMock, return_value=mock_ws),
            patch.object(cm, "_authenticate", new_callable=AsyncMock, return_value=True),
            patch("gateway.platforms.yuanbao._resolve_ws_proxy_override", return_value=(True, credentialed)),
            caplog.at_level("INFO", logger="gateway.platforms.yuanbao"),
        ):
            assert await cm._dial({"bot_id": "b1", "token": "t"}) is True
        repair_logs = [r.getMessage() for r in caplog.records if "SOCKS proxy" in r.getMessage()]
        assert repair_logs, "expected a repair log line"
        assert all("secret" not in m and "user" not in m for m in repair_logs)
