"""Tests for ``/model <纯数字>`` selection against the gateway picker snapshot store.

T6 序号选择（OneBot 平台文字 picker 的网关配套）：

* 快照存网关 runner（``ModelPickerSnapshotStore``）：TTL 5 分钟 + 容量 256（LRU）
  双内存防护，TTL/容量清理均为显式可测方法。
* ``/model <纯数字>`` 仅在该 session_key 存在新鲜快照时按序号解释；歧义优先级：
  精确模型名命中 > 快照内序号（仅快照新鲜且 1≤n≤len）> 报错提示过期/越界。
* 带 --provider / --global / --session / --once 时不做序号解释；
  无快照场景零行为变化（走原有字面切换路径）。
* ``send_model_picker`` 成功发送后统一落账快照——telegram 等按钮型 picker 的
  既有分支不回归（见 tests/gateway/test_telegram_model_picker.py）。
"""

import re

import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.slash_commands_model import ModelPickerSnapshotStore, _flatten_picker_items


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
PROVIDERS = [
    {"slug": "openai", "name": "OpenAI", "models": ["gpt-4o", "o3"], "total_models": 2},
    {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-4-5"], "total_models": 1},
    # 自定义端点可能无模型行：不占序号（与 onebot 渲染约定一致）
    {"slug": "custom-endpoint", "name": "My Endpoint", "models": [],
     "total_models": 0, "is_user_defined": True, "api_url": "http://x"},
]


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text, platform=None):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=platform or Platform("onebot"),
            chat_id="group_123", chat_type="group"),
    )


@pytest.fixture
def _isolated_config(tmp_path, monkeypatch):
    """Point the handler at an empty isolated home (deterministic, no network)."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-x\n  provider: openrouter\nproviders: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


def _record_snapshot(runner, session_key, providers=PROVIDERS):
    store = ModelPickerSnapshotStore()
    runner._model_picker_snapshots = store
    store.record(session_key, providers)
    return store


# --------------------------------------------------------------------------- #
# Gateway: 序号命中 / 过期 / 越界 / 歧义优先级
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_numeric_target_resolves_fresh_snapshot(_isolated_config):
    """/model 3 在新鲜快照下按列表第 3 项切换（provider 一并改写）。"""
    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_event("/model 3").source)
    _record_snapshot(runner, session_key)
    runner._perform_model_switch = AsyncMock(return_value=(None, "switch-error"))

    reply = await runner._handle_model_command(_make_event("/model 3"))

    assert reply == "switch-error"  # mocked perform short-circuits the commit
    args = runner._perform_model_switch.await_args
    assert args.args[1] == "claude-sonnet-4-5"  # 第 3 项 = anthropic / claude-sonnet-4-5
    assert args.args[2] == "anthropic"


@pytest.mark.asyncio
async def test_numeric_target_expired_snapshot_rejected(_isolated_config):
    """快照过期：序号解释被拒绝并提示过期，不落入字面切换。"""
    runner = _make_runner()
    now = {"t": 1000.0}
    store = ModelPickerSnapshotStore(clock=lambda: now["t"])
    runner._model_picker_snapshots = store
    session_key = runner._session_key_for_source(_make_event("/model 1").source)
    store.record(session_key, PROVIDERS)
    runner._perform_model_switch = AsyncMock()

    now["t"] += 301.0  # TTL 300s 已过
    reply = await runner._handle_model_command(_make_event("/model 1"))

    assert "expired" in reply
    runner._perform_model_switch.assert_not_awaited()
    assert len(store) == 0  # 过期条目顺手剔除，不残留


@pytest.mark.asyncio
async def test_numeric_target_out_of_range_rejected(_isolated_config):
    """快照新鲜但序号越界：直接报错，不触发切换。"""
    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_event("/model 9").source)
    _record_snapshot(runner, session_key)
    runner._perform_model_switch = AsyncMock()

    reply = await runner._handle_model_command(_make_event("/model 9"))

    assert "out of range" in reply
    assert "3" in reply  # 提示合法范围上限 len(items)=3
    runner._perform_model_switch.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_model_name_beats_snapshot_index(_isolated_config):
    """歧义优先级：快照里存在与纯数字同名的精确模型名 → 名字优先，序号不解释。"""
    providers = [{"slug": "weird", "name": "Weird", "models": ["42"], "total_models": 1}]
    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_event("/model 42").source)
    _record_snapshot(runner, session_key, providers)
    runner._perform_model_switch = AsyncMock(return_value=(None, "switch-error"))

    await runner._handle_model_command(_make_event("/model 42"))

    args = runner._perform_model_switch.await_args
    assert args.args[1] == "42"        # 原样字面切换（名字优先）
    assert args.args[2] == ""          # 不带快照里的 provider 改写


@pytest.mark.asyncio
@pytest.mark.parametrize("args_line,expected_provider", [
    ("2 --session", ""),          # --session 语义不变
    ("2 --global", ""),           # --global 语义不变
    ("2 --provider anthropic", "anthropic"),  # --provider 语义不变
])
async def test_flags_skip_numeric_interpretation(_isolated_config, args_line, expected_provider):
    """带 flag 时不做序号解释：纯数字按字面模型名走原有路径。"""
    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_event("/model 2").source)
    _record_snapshot(runner, session_key)  # 第 2 项本应是 ("o3", "openai")
    runner._perform_model_switch = AsyncMock(return_value=(None, "switch-error"))

    await runner._handle_model_command(_make_event(f"/model {args_line}"))

    args = runner._perform_model_switch.await_args
    assert args.args[1] == "2"  # 字面 target，未被改写为 o3
    assert args.args[2] == expected_provider


@pytest.mark.asyncio
async def test_numeric_without_snapshot_zero_behavior_change(_isolated_config):
    """无快照（其他平台/未发过 picker）：零行为变化，纯数字按字面切换。"""
    runner = _make_runner()
    runner._perform_model_switch = AsyncMock(return_value=(None, "switch-error"))

    await runner._handle_model_command(_make_event("/model 3"))

    args = runner._perform_model_switch.await_args
    assert args.args[1] == "3"
    assert args.args[2] == ""


# --------------------------------------------------------------------------- #
# Gateway: OneBot 无参 /model 走 picker，不再文本回退
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_onebot_no_arg_model_sends_picker_not_text_fallback(_isolated_config, monkeypatch):
    """OneBot /model（无参）：走 send_model_picker，且不落文本列表回退。"""
    from plugins.platforms.onebot.adapter import OneBotAdapter

    adapter = OneBotAdapter(PlatformConfig(enabled=True))
    adapter._ws = object()  # connected
    sent = {}

    async def fake_send(chat_id, content, reply_to=None, metadata=None):
        sent.update(chat_id=chat_id, content=content)
        return SendResult(success=True, message_id="1")

    adapter.send = fake_send

    runner = _make_runner()
    runner.adapters = {Platform("onebot"): adapter}
    runner._thread_metadata_for_source = lambda *a, **k: None
    runner._reply_anchor_for_event = lambda *a, **k: None

    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers", lambda **kw: PROVIDERS)
    text_fallback_used = {"used": False}

    def _boom(**kw):
        text_fallback_used["used"] = True
        return []

    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", _boom)

    event = _make_event("/model")
    reply = await runner._handle_model_command(event)

    assert reply is None                       # picker 已发，无文本回复
    assert text_fallback_used["used"] is False  # 未走文本回退
    assert "1. gpt-4o" in sent["content"]
    # 成功发送后快照落账（网关侧），后续 /model <序号> 可解释
    state, items = runner._model_picker_store.lookup(
        runner._session_key_for_source(event.source))
    assert state == "fresh"
    assert len(items) == 3


# --------------------------------------------------------------------------- #
# Gateway: telegram 按钮型 picker 分支不回归
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_telegram_picker_flow_not_regressed(_isolated_config, monkeypatch):
    """telegram（按钮型 picker）：/model 无参仍走 picker 分支，回调链路原样传递。"""
    class _FakePickerResult:
        success = True

    class _FakePickerAdapter:
        async def send_model_picker(self, **kwargs):
            _FakePickerAdapter.kwargs = kwargs
            return _FakePickerResult()

    _FakePickerAdapter.kwargs = None

    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    runner._thread_metadata_for_source = lambda *a, **k: None
    runner._reply_anchor_for_event = lambda *a, **k: None
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers", lambda **kw: PROVIDERS)
    runner._perform_model_switch = AsyncMock()
    event = _make_event("/model", platform=Platform.TELEGRAM)
    reply = await runner._handle_model_command(event)

    assert reply is None
    kwargs = _FakePickerAdapter.kwargs
    assert kwargs["providers"] == PROVIDERS
    assert callable(kwargs["on_model_selected"])  # 回调链路原样传递
    assert kwargs["session_key"] == runner._session_key_for_source(event.source)
    runner._perform_model_switch.assert_not_awaited()


# --------------------------------------------------------------------------- #
# 快照存储：TTL / 容量（LRU）/ 空快照 —— 内存防护专项
# --------------------------------------------------------------------------- #
def test_snapshot_store_ttl_expiry_and_prune():
    """TTL 过期在 lookup 与显式 prune 时均被剔除。"""
    now = {"t": 1000.0}
    store = ModelPickerSnapshotStore(clock=lambda: now["t"])
    store.record("k1", PROVIDERS)
    store.record("k2", PROVIDERS)

    state, items = store.lookup("k1")
    assert state == "fresh" and len(items) == 3

    now["t"] += 301.0
    assert store.lookup("k1")[0] == "expired"
    assert len(store) == 1  # k1 已被顺手剔除，k2 仍在

    assert store.prune() == 1  # 显式清理 k2
    assert len(store) == 0


def test_snapshot_store_capacity_eviction_lru():
    """容量上限按最旧淘汰；fresh 命中刷新 LRU 顺序。"""
    store = ModelPickerSnapshotStore(max_entries=2)
    store.record("a", PROVIDERS)
    store.record("b", PROVIDERS)
    store.lookup("a")          # touch a → b 成为最旧
    store.record("c", PROVIDERS)  # 淘汰 b

    assert store.lookup("a")[0] == "fresh"
    assert store.lookup("b")[0] == "absent"
    assert store.lookup("c")[0] == "fresh"


def test_snapshot_store_empty_providers_not_recorded():
    """空列表（无可列项）不落账，并清掉同 key 旧快照，避免脏序号解释。"""
    store = ModelPickerSnapshotStore()
    store.record("k", PROVIDERS)
    store.record("k", [])
    assert len(store) == 0
    assert store.lookup("k")[0] == "absent"


# --------------------------------------------------------------------------- #
# 跨层一致性：网关展开顺序 ≡ OneBot 文字列表编号
# --------------------------------------------------------------------------- #
def test_picker_numbering_matches_onebot_render():
    """两侧编号一致性锁定：网关 _flatten_picker_items ≡ onebot render_model_picker_text。"""
    from plugins.platforms.onebot.onebot_utils import render_model_picker_text

    text = render_model_picker_text(PROVIDERS, "o3", "openai")
    flat = _flatten_picker_items(PROVIDERS)

    numbered = re.findall(r"^(\d+)\. (.+?)(?: ← 当前)?$", text, flags=re.M)
    assert [m for (_n, m) in numbered] == [model for (_s, _name, model) in flat]
    assert [n for (n, _m) in numbered] == [str(i + 1) for i in range(len(flat))]
    assert "2. o3 ← 当前" in text  # 当前项标记落在正确行
    # 无模型的 provider 行不占序号，但可有分组标题
    assert "【My Endpoint】" not in text
    assert "【OpenAI】" in text
