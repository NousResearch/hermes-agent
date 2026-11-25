"""A Feishu plugin load must run the card helpers from the plugin directory it loads.

The directory loader imports a plugin as ``hermes_plugins.<slug>`` and evicts that namespace on
reload; a module imported by absolute path lands under ``plugins.platforms.feishu.*`` instead, so
neither a reload nor a plugin copy elsewhere ever reaches its code. Every lane that calls the
``cards`` helpers (the adapter itself, clarify/choice cards, synthetic events) is checked.
"""

import asyncio
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_loader import _evict_modules
from hermes_cli.plugins_manifest import PluginManifest

_FEISHU_DIR = Path(__file__).resolve().parents[4] / "plugins" / "platforms" / "feishu"
# Appended to the copy's cards/authz.py: later definitions win, so the reloaded helpers answer
# "edited" and record which lane reached them.
_OVERRIDES = '''

RELOADED_CALLS = []


def map_chat_type(raw_chat_type):
    return "edited"


def resolve_source_chat_type(*, chat_info, event_chat_type):
    RELOADED_CALLS.append("synthetic")
    return "edited"


def validate_card_action(adapter, *, event, state, label, ident):
    RELOADED_CALLS.append(label)
    return None
'''


@pytest.fixture
def plugin_copy(tmp_path):
    plugin_dir = tmp_path / "feishu"
    shutil.copytree(_FEISHU_DIR, plugin_dir, ignore=shutil.ignore_patterns("__pycache__"))
    manager = PluginManager(scope_key=str(tmp_path / "hermes"))
    manifest = PluginManifest(name="feishu-platform", key="feishu-platform", kind="platform",
                              source="bundled", path=str(plugin_dir))
    loaded: set[str] = set()

    def load():
        module = manager._load_directory_module(manifest)
        loaded.add(module.__name__)
        return module.__name__

    def edit():
        authz_py = plugin_dir / "cards" / "authz.py"
        authz_py.write_text(authz_py.read_text(encoding="utf-8") + _OVERRIDES, encoding="utf-8")

    yield load, edit
    for name in loaded:
        _evict_modules(name)


def _adapter(package: str):
    adapter = object.__new__(sys.modules[f"{package}.adapter"].FeishuAdapter)
    adapter._chat_info_cache = {}
    return adapter


def _chat_type_for_p2p(package: str) -> str:
    adapter = _adapter(package)
    adapter._build_get_chat_request = lambda _cid: object()
    data = SimpleNamespace(name="Chat", chat_type="private", chat_mode="p2p")
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(chat=SimpleNamespace(
        get=lambda _req: SimpleNamespace(success=lambda: True, data=data)))))

    async def _run_blocking(func, *args):
        return func(*args)

    adapter._run_blocking = _run_blocking
    return asyncio.run(adapter.get_chat_info("oc_x"))["type"]


def _click_clarify(package: str) -> None:
    adapter = _adapter(package)
    adapter._clarify_state = {"c1": {"chat_id": "oc_x"}}
    adapter._card_response = lambda *args: None
    adapter._handle_clarify_card_action(event=None, action_value={"clarify_id": "c1"}, loop=None)


def _dispatch_synthetic(package: str) -> str:
    adapter = _adapter(package)
    seen = {}

    async def _profile(_sender):
        return {"user_id": "u", "user_name": "U", "user_id_alt": None}

    async def _chat_info(_chat_id):
        return {"name": "Chat", "raw_mode": "p2p"}

    async def _guarded(event):
        seen["chat_type"] = event.source.chat_type

    adapter._resolve_sender_profile = _profile
    adapter.get_chat_info = _chat_info
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    adapter._resolve_channel_prompt = lambda _chat_id: None
    adapter._handle_message_with_guards = _guarded
    message_type = sys.modules[f"{package}.adapter"].MessageType.TEXT
    asyncio.run(adapter._dispatch_synthetic_event(
        text="x", message_type=message_type, chat_id="oc_x", sender_id=None, event_chat_type="p2p",
        raw_message=None, message_id="m1"))
    return seen["chat_type"]


def test_reload_runs_the_edited_card_helpers_on_every_lane(plugin_copy):
    load, edit = plugin_copy
    assert _chat_type_for_p2p(load()) == "dm"

    edit()
    package = load()
    calls = sys.modules[f"{package}.cards.authz"].RELOADED_CALLS
    assert _chat_type_for_p2p(package) == "edited"
    _click_clarify(package)
    assert _dispatch_synthetic(package) == "edited"
    assert calls == ["clarify", "synthetic"]
