"""Streaming drafts on the first gateway run after python-telegram-bot is installed on demand.

The registry installs a missing platform SDK when the gateway starts (#79812), and the installer rebinds
the SDK names only in the adapter copy the gateway runs. ``send_draft`` must still go out as MarkdownV2
on that run instead of raising, which turns drafts off until the gateway restarts.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("telegram")

# Linux CI slices have no python-telegram-bot (importorskip skips there); the macOS and Windows PR
# lanes install the ``telegram`` extra and import only platforms()-marked files.
pytestmark = pytest.mark.platforms("any")

_ROOT = Path(__file__).resolve().parents[4]
_FIRST_RUN_PROBE = r"""
import asyncio, importlib.abc, json, os, sys
from unittest.mock import AsyncMock, MagicMock
home = sys.argv[1]
os.environ["HERMES_HOME"] = home
with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
    fh.write("platforms:\n  telegram:\n    enabled: true\n    token: '123456:offline'\n")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "telegram" or name.startswith("telegram."):
            raise ModuleNotFoundError("No module named " + repr(name), name=name)
blocker = Block()
sys.meta_path.insert(0, blocker)
from hermes_cli.plugins import discover_plugins
discover_plugins()
from gateway.config import Platform, load_gateway_config
from gateway.platform_registry import platform_registry
platform_registry.get("telegram")
# Before the lazy install finishes, a disconnected adapter must still fail softly.
from types import SimpleNamespace
from plugins.platforms.telegram.telegram_flood_window import send_draft as send_draft_unbound
try:
    disconnected = asyncio.run(send_draft_unbound(SimpleNamespace(_bot=None), "123", 7, "preview", None))
    preinstall_disconnected = {"success": disconnected.success, "error": disconnected.error}
except Exception as exc:
    preinstall_disconnected = {"raised": f"{type(exc).__name__}: {exc}"}
sys.meta_path.remove(blocker)  # the on-demand install has landed
import pm
pm.ensure_import = lambda name: None
adapter = platform_registry.create_adapter("telegram", load_gateway_config().platforms[Platform.TELEGRAM])
adapter._bot = MagicMock()
adapter._bot.send_message_draft = AsyncMock(return_value=True)
try:
    result = asyncio.run(adapter.send_draft("123", 7, "**bold** and `code`", None))
    outcome = {"success": result.success, "error": result.error}
except Exception as exc:
    outcome = {"raised": f"{type(exc).__name__}: {exc}"}
print("RESULT " + json.dumps({
    "live_module": type(adapter).__module__,
    "preinstall_disconnected": preinstall_disconnected,
    "outcome": outcome,
    "parse_modes": [str(c.kwargs.get("parse_mode")) for c in adapter._bot.send_message_draft.await_args_list]}))
"""


def test_first_draft_after_on_demand_install_goes_out_as_markdown_v2(tmp_path):
    run = subprocess.run([sys.executable, "-c", _FIRST_RUN_PROBE, str(tmp_path)], cwd=_ROOT, capture_output=True,
                         text=True, timeout=240, stdin=subprocess.DEVNULL, check=False)
    line = next((ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, run.stdout[-2000:] + run.stderr[-6000:]
    result = json.loads(line[len("RESULT "):])

    assert result["live_module"].startswith("hermes_plugins."), "not the production plugin-loading path"
    assert result["preinstall_disconnected"] == {"success": False, "error": "not_connected"}
    assert result["outcome"] == {"success": True, "error": None}
    assert result["parse_modes"] == ["MarkdownV2"]
