"""Shared fixtures for tests/browser/.

These tests lived in tests/tools/ and ran under its autouse fixtures, which keep the host's Chromium download,
browser-harness install, shared socket dirs and Bot Desktop out of every test. pytest registers the fixtures this
module imports, so both folders share one definition.
"""

from tests.tools.conftest import (
    _clear_web_result_cache,
    _materialize_mcp_sdk_symbols,
    _no_chromium_download,
    _no_host_bot_desktop_autostart,
    _no_host_browser_use_cli,
    _no_shared_temp_socket_dirs,
)
