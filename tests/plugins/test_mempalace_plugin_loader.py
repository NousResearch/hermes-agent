from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_mempalace_installed = False
try:
    import mempalace  # noqa: F401

    _mempalace_installed = True
except ImportError:
    pass

_requires_mempalace = pytest.mark.skipif(
    not _mempalace_installed,
    reason="mempalace package not installed (pip install mempalace)",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "memory" / "mempalace"
INIT_FILE = PLUGIN_DIR / "__init__.py"
MODULE_NAME = "plugins.memory.mempalace"


def _ensure_package(name: str, package_dir: Path) -> None:
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec and spec.loader and (package_dir / "__init__.py").exists():
        spec.loader.exec_module(module)


def load_plugin_module():
    _ensure_package("plugins", REPO_ROOT / "plugins")
    _ensure_package("plugins.memory", REPO_ROOT / "plugins" / "memory")
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        INIT_FILE,
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    return load_plugin_module()


def test_plugin_loads_via_importlib(plugin_module):
    assert hasattr(plugin_module, "MemPalaceMemoryProvider")
    assert hasattr(plugin_module, "register")


def test_register_works_when_loaded_like_hermes(plugin_module):
    ctx = MagicMock()
    provider_cls = plugin_module.MemPalaceMemoryProvider
    original = provider_cls.is_available
    provider_cls.is_available = lambda self: True
    try:
        plugin_module.register(ctx)
    finally:
        provider_cls.is_available = original
    ctx.register_memory_provider.assert_called_once()


def test_register_skips_when_optional_package_missing(plugin_module):
    ctx = MagicMock()
    provider_cls = plugin_module.MemPalaceMemoryProvider
    original = provider_cls.is_available
    provider_cls.is_available = lambda self: False
    try:
        plugin_module.register(ctx)
    finally:
        provider_cls.is_available = original
    ctx.register_memory_provider.assert_not_called()


def test_tool_call_returns_structured_error_when_not_initialized(plugin_module):
    provider = plugin_module.MemPalaceMemoryProvider()
    result = json.loads(provider.handle_tool_call("mempalace_search", {"query": "x"}))

    assert result["success"] is False
    assert result["error"]["message"] == "MemPalace is not initialized"
    assert result["error"]["details"]["tool_name"] == "mempalace_search"


def test_status_tool_still_works_before_initialize(plugin_module):
    provider = plugin_module.MemPalaceMemoryProvider()
    result = json.loads(provider.handle_tool_call("mempalace_status", {}))

    assert result["collection_name"] == ""


def test_write_queue_drops_after_retry_limit(plugin_module, caplog):
    from plugins.memory.mempalace import writer

    calls = []
    original_upsert = writer.upsert_memory_item
    writer.upsert_memory_item = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("wedged")
    )

    class InlineThread:
        def __init__(self, *, target, name, daemon):
            self._target = target

        def start(self):
            calls.append("start")

        def join(self, timeout=None):
            calls.append(("join", timeout))

    try:
        q = writer.WriteQueue(
            collection=object(),
            agent_id="hermes",
            thread_factory=InlineThread,
            max_retries=1,
            retry_delay=0,
        )
        q.enqueue([{"id": "m1"}])
        payload = q._q.get_nowait()
        q._flush(*payload)
        retry_payload = q._q.get_nowait()
        q._flush(*retry_payload)
    finally:
        writer.upsert_memory_item = original_upsert

    assert "MemPalace dropped batch after 1 retries" in caplog.text


def test_write_queue_drops_when_full(plugin_module, caplog):
    from plugins.memory.mempalace import writer

    class InlineThread:
        def __init__(self, *, target, name, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    q = writer.WriteQueue(
        collection=object(),
        agent_id="hermes",
        thread_factory=InlineThread,
        max_queue_size=1,
    )
    q.enqueue([{"id": "m1"}])
    q.enqueue([{"id": "m2"}])

    assert "MemPalace write queue full; dropping enqueue batch" in caplog.text


@_requires_mempalace
def test_initialize_uses_current_hermes_config_shape(plugin_module, tmp_path):
    provider = plugin_module.MemPalaceMemoryProvider()
    hermes_home = tmp_path / ".hermes"
    config = {
        "memory": {"provider": "mempalace"},
        "mempalace": {
            "palace_path": str(tmp_path / "palace"),
            "wing": "conversations",
            "n_results": 5,
            "tool_max_results": 7,
            "enable_kg": True,
            "collection_template": "hermes-{platform}-{user_id}",
            "room_strategy": "platform_session",
            "fixed_room": "memory",
        },
    }

    provider.initialize(
        session_id="sess-test",
        hermes_home=str(hermes_home),
        config=config,
        user_id="jessica",
        agent_id="hermes",
        platform="telegram",
    )

    try:
        assert provider.name == "mempalace"
        assert provider.is_available() is True
        assert provider._collection is not None
        assert provider._queue is not None
        assert provider._kg is not None
        assert provider._palace_path == str(tmp_path / "palace")
        assert provider._wing == "conversations"
        assert provider._n_results == 5
        assert provider._tool_max_results == 7
        assert provider._runtime_ctx["platform"] == "telegram"
        assert provider._collection_name == "hermes-telegram-jessica"
        result = provider.handle_tool_call("mempalace_status", {})
        payload = json.loads(result)
        assert isinstance(payload, dict)
        assert payload["collection_name"] == "hermes-telegram-jessica"
        assert payload["room_strategy"] == "platform_session"
    finally:
        provider.shutdown()


@_requires_mempalace
def test_initialize_respects_disable_kg(plugin_module, tmp_path):
    provider = plugin_module.MemPalaceMemoryProvider()
    provider.initialize(
        session_id="sess-no-kg",
        hermes_home=str(tmp_path / ".hermes"),
        config={
            "mempalace": {
                "palace_path": str(tmp_path / "palace-no-kg"),
                "enable_kg": False,
            }
        },
        user_id="u1",
        agent_id="hermes",
    )
    try:
        assert provider._kg is None
    finally:
        provider.shutdown()


def test_tool_schemas_exposed(plugin_module):
    provider = plugin_module.MemPalaceMemoryProvider()
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert names == {
        "mempalace_memorize",
        "mempalace_search",
        "mempalace_recall",
        "mempalace_forget",
        "mempalace_status",
    }
