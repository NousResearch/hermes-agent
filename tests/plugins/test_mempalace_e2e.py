from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest  # type: ignore[unresolved-import]

pytest.importorskip("mempalace")

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
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader and (package_dir / "__init__.py").exists():
        spec.loader.exec_module(module)


def load_plugin_module():
    _ensure_package("plugins", REPO_ROOT / "plugins")
    _ensure_package("plugins.memory", REPO_ROOT / "plugins" / "memory")
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        INIT_FILE,
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    return load_plugin_module()


@pytest.fixture
def provider(plugin_module, tmp_path):
    provider = plugin_module.MemPalaceMemoryProvider()
    provider.initialize(
        session_id="sess-e2e",
        hermes_home=str(tmp_path / ".hermes"),
        config={
            "memory": {"provider": "mempalace"},
            "mempalace": {
                "palace_path": str(tmp_path / "palace"),
                "wing": "conversations",
                "n_results": 5,
                "tool_max_results": 20,
                "enable_kg": True,
                "collection_template": "hermes-{platform}-{user_id}",
                "room_strategy": "platform_session",
                "fixed_room": "memory",
            },
        },
        user_id="jessica-e2e",
        agent_id="hermes",
        platform="telegram",
    )
    try:
        yield provider
    finally:
        provider.shutdown()


def _tool(provider, name: str, args: dict):
    return json.loads(provider.handle_tool_call(name, args))


def _wait_for_prefetch(provider, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        thread = provider._prefetch_thread
        if not thread or not thread.is_alive():
            return
        time.sleep(0.05)
    raise TimeoutError("prefetch thread did not finish in time")


def test_memorize_search_recall_forget_round_trip(provider):
    memorize = _tool(
        provider,
        "mempalace_memorize",
        {
            "content": "Jessica likes 深度架构分析 and publishable plugin design.",
            "memory_type": "preference",
            "importance": 0.95,
            "room": "prefs-room",
        },
    )
    assert memorize["success"] is True

    status = _tool(provider, "mempalace_status", {})
    assert status["collection_name"] == "hermes-telegram-jessica-e2e"
    assert status["room_strategy"] == "platform_session"

    search = _tool(
        provider,
        "mempalace_search",
        {"query": "publishable plugin design", "room": "prefs-room", "top_k": 5},
    )
    assert "results" in search
    assert any(
        "publishable plugin design" in item["content"] for item in search["results"]
    )

    target = next(
        item
        for item in search["results"]
        if "publishable plugin design" in item["content"]
    )
    assert target["metadata"]["room"] == "prefs-room"
    assert target["metadata"]["wing"] == provider._wing
    assert target["metadata"]["source"] == "tool"
    assert target["metadata"]["message_kind"] == "explicit_memory"
    assert target["metadata"]["memory_type"] == "preference"
    assert target["metadata"]["importance"] == 0.95
    assert target["metadata"]["platform"] == "telegram"
    assert target["metadata"]["user_id"] == "jessica-e2e"
    assert target["metadata"]["agent_id"] == "hermes"
    assert "created_at" in target["metadata"]

    recall = _tool(
        provider, "mempalace_recall", {"room": "prefs-room", "n_results": 10}
    )
    assert "results" in recall
    assert any(item["id"] == target["id"] for item in recall["results"])

    forget = _tool(provider, "mempalace_forget", {"memory_id": target["id"]})
    assert forget["success"] is True

    recall_after = _tool(
        provider, "mempalace_recall", {"room": "prefs-room", "n_results": 10}
    )
    assert all(item["id"] != target["id"] for item in recall_after["results"])


def test_sync_turn_and_prefetch_share_same_collection(provider):
    before = provider._collection.count()
    provider.sync_turn(
        "User asks about Browserbase and anti-bot verification.",
        "Assistant explains local playwright and stealth settings.",
        session_id="thread-42",
    )

    deadline = time.time() + 5
    while time.time() < deadline:
        if provider._collection.count() >= before + 2:
            break
        time.sleep(0.05)
    assert provider._collection.count() >= before + 2

    direct = provider._raw_search(
        "Browserbase anti-bot verification", n_results=5, room="telegram-thread-42"
    )
    formatted = provider._format_search_result(direct, raw=True)
    assert formatted["results"]
    assert any("Browserbase" in item["content"] for item in formatted["results"])
    matched = next(
        item for item in formatted["results"] if "Browserbase" in item["content"]
    )
    assert matched["metadata"]["room"] == "telegram-thread-42"
    assert matched["metadata"]["source"] == "sync_turn"
    assert matched["metadata"]["message_kind"] == "user_message"
    assert matched["metadata"]["session_id"] == "thread-42"

    provider.queue_prefetch("local playwright stealth", session_id="thread-42")
    _wait_for_prefetch(provider)
    prefetched = provider.prefetch("local playwright stealth", session_id="thread-42")
    assert "[MemPalace Memory]" in prefetched
    assert "local playwright" in prefetched.lower() or "stealth" in prefetched.lower()


def test_where_filter_uses_and_for_wing_plus_room(provider):
    where = provider._build_where("room-x")
    assert where == {"$and": [{"wing": provider._wing}, {"room": "room-x"}]}


def test_search_caps_top_k_by_tool_max_results(provider):
    provider._tool_max_results = 2
    _tool(
        provider,
        "mempalace_memorize",
        {"content": "cap result one", "room": "cap-room"},
    )
    _tool(
        provider,
        "mempalace_memorize",
        {"content": "cap result two", "room": "cap-room"},
    )
    _tool(
        provider,
        "mempalace_memorize",
        {"content": "cap result three", "room": "cap-room"},
    )

    search = _tool(
        provider,
        "mempalace_search",
        {"query": "cap result", "room": "cap-room", "top_k": 10},
    )
    assert len(search["results"]) <= 2


def test_handle_tool_call_returns_structured_error_payload(provider):
    payload = _tool(provider, "mempalace_search", {"room": "missing-query"})
    assert payload["success"] is False
    assert payload["error"]["code"] == "MEMPALACE_TOOL_ERROR"
    assert "query is required" in payload["error"]["message"]


def test_search_room_filter_does_not_leak_other_rooms(provider):
    _tool(
        provider,
        "mempalace_memorize",
        {"content": "alpha room memory", "room": "room-a"},
    )
    _tool(
        provider,
        "mempalace_memorize",
        {"content": "beta room memory", "room": "room-b"},
    )

    search_a = _tool(
        provider,
        "mempalace_search",
        {"query": "room memory", "room": "room-a", "top_k": 10},
    )
    assert search_a["results"]
    assert all(item["metadata"]["room"] == "room-a" for item in search_a["results"])
    assert all(
        "beta room memory" not in item["content"] for item in search_a["results"]
    )


def test_search_deduplicates_near_identical_results(provider):
    room = "dedupe-room"
    repeated = "你一开始说的记忆方案 user.md 和 memory.md 那套逻辑还在用吗"
    variants = [
        repeated,
        repeated + "？",
        repeated + "   ",
    ]
    for item in variants:
        _tool(
            provider,
            "mempalace_memorize",
            {"content": item, "room": room, "memory_type": "instruction"},
        )

    search = _tool(
        provider,
        "mempalace_search",
        {"query": "user.md memory.md 记忆方案", "room": room, "top_k": 10},
    )
    assert len(search["results"]) == 1
    assert repeated in search["results"][0]["content"]


def test_recall_deduplicates_internal_duplicates_and_backfills_unique_results(provider):
    room = "recall-dedupe-room"
    duplicates = [
        "这样子的话注入上下文时不会有重复吗",
        "这样子的话注入上下文时不会有重复吗？",
        "这样子的话注入上下文时不会有重复吗   ",
    ]
    for item in duplicates:
        _tool(
            provider,
            "mempalace_memorize",
            {"content": item, "room": room, "memory_type": "instruction"},
        )

    _tool(
        provider,
        "mempalace_memorize",
        {
            "content": "长期事实：系统现在会先做跨 provider 去重。",
            "room": room,
            "memory_type": "factual",
            "importance": 0.95,
        },
    )
    _tool(
        provider,
        "mempalace_memorize",
        {
            "content": "待办：还要继续优化 MemPalace recall 内部去重。",
            "room": room,
            "memory_type": "instruction",
            "importance": 0.9,
        },
    )

    recall = _tool(provider, "mempalace_recall", {"room": room, "n_results": 3})
    assert len(recall["results"]) == 3
    assert (
        sum(
            "这样子的话注入上下文时不会有重复吗" in item["content"]
            for item in recall["results"]
        )
        == 1
    )
    assert any("跨 provider 去重" in item["content"] for item in recall["results"])
    assert any(
        "MemPalace recall 内部去重" in item["content"] for item in recall["results"]
    )


def test_prefetch_filters_low_signal_conversation_when_memory_exists(provider):
    provider.sync_turn(
        "我之前准备为你重写一个 memorypalace plugin，现在进展如何？",
        "我们先检查 recall 注入路径，再处理重复问题。",
        session_id="thread-noise",
    )
    _tool(
        provider,
        "mempalace_memorize",
        {
            "content": "长期事实：当前系统仍同时使用 USER.md / MEMORY.md 与 MemPalace provider。",
            "room": "telegram-thread-noise",
            "memory_type": "factual",
            "importance": 0.95,
        },
    )

    provider.queue_prefetch(
        "记忆方案还在用吗 memory.md user.md", session_id="thread-noise"
    )
    _wait_for_prefetch(provider)
    prefetched = provider.prefetch(
        "记忆方案还在用吗 memory.md user.md", session_id="thread-noise"
    )
    assert "[MemPalace Memory]" in prefetched
    assert "长期事实" in prefetched
    assert "我之前准备为你重写一个 memorypalace plugin" not in prefetched


def test_on_memory_write_mirrors_builtin_memory(provider):
    provider.on_memory_write(
        "add", "memory", "Builtin memory mirror marker for hook verification."
    )

    direct = provider._raw_search(
        "Builtin memory mirror marker", n_results=5, room="memory"
    )
    formatted = provider._format_search_result(direct, raw=True)
    assert formatted["results"]
    matched = next(
        item
        for item in formatted["results"]
        if "Builtin memory mirror marker" in item["content"]
    )
    assert matched["metadata"]["room"] == "memory"
    assert matched["metadata"]["source"] == "memory"
    assert matched["metadata"]["message_kind"] == "builtin_memory_write"
    assert matched["metadata"]["source_file"] == "builtin_memory_add"
    assert matched["metadata"]["session_id"] == "sess-e2e"


def test_on_pre_compress_persists_key_facts(provider):
    messages = [
        {"role": "system", "content": "system prompt should be skipped"},
        {
            "role": "user",
            "content": "Compression hook marker alpha: we chose the modular provider layout for maintainability.",
        },
        {
            "role": "assistant",
            "content": "Compression hook marker beta: preserve this architectural rationale before compression.",
        },
    ]

    summary = provider.on_pre_compress(messages)
    assert "[MemPalace: key facts preserved from compressed context]" in summary
    assert "Compression hook marker alpha" in summary

    direct = provider._raw_search(
        "Compression hook marker modular provider layout architectural rationale",
        n_results=10,
        room="session-sess-e2e",
    )
    formatted = provider._format_search_result(direct, raw=True)
    assert (
        len(
            [
                item
                for item in formatted["results"]
                if "Compression hook marker" in item["content"]
            ]
        )
        >= 2
    )

    matched = next(
        item
        for item in formatted["results"]
        if "Compression hook marker alpha" in item["content"]
    )
    assert matched["metadata"]["room"] == "session-sess-e2e"
    assert matched["metadata"]["source"] == "compression"
    assert matched["metadata"]["message_kind"] == "compressed_context"
    assert matched["metadata"]["source_file"] == "compression"
    assert matched["metadata"]["session_id"] == "sess-e2e"


def test_on_session_end_persists_summary_room(provider):
    messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": "We chose scenario C because it is more publishable and general.",
        },
        {
            "role": "assistant",
            "content": "Understood. We will validate the publishable path end-to-end.",
        },
    ]

    provider.on_session_end(messages)
    recall = _tool(
        provider, "mempalace_recall", {"room": "session_summaries", "n_results": 10}
    )
    assert recall["results"]
    assert any(
        "scenario C" in item["content"] or "publishable path" in item["content"]
        for item in recall["results"]
    )
