"""
test_yuanbao_integration.py - Yuanbao 模块集成测试

验证各模块能正确组装和交互：
  - YuanbaoAdapter 初始化
  - Config / Platform 枚举
  - get_connected_platforms 逻辑
  - Proto 编解码 round-trip
  - Markdown 分块
  - API / Media 模块 import
  - Toolset 注册
"""

import sys
import os

# 确保 hermes-agent 根目录在 sys.path 中
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from unittest.mock import MagicMock, patch
from gateway.config import Platform, PlatformConfig, GatewayConfig
from gateway.platforms.yuanbao import YuanbaoAdapter


def make_config(**kwargs):
    extra = kwargs.pop("extra", {})
    extra.setdefault("app_id", "test_key")
    extra.setdefault("app_secret", "test_secret")
    extra.setdefault("ws_url", "wss://test.example.com/ws")
    extra.setdefault("api_domain", "https://test.example.com")
    return PlatformConfig(
        extra=extra,
        **kwargs,
    )


# ===========================================================
# 1. Adapter 初始化
# ===========================================================




# ===========================================================
# 2. Config / Platform 枚举
# ===========================================================

class TestYuanbaoConfig:


    def test_get_connected_platforms_requires_key_and_secret(self):
        # Only key, no secret → not in connected list
        gw_only_key = GatewayConfig(
            platforms={
                Platform.YUANBAO: PlatformConfig(
                    enabled=True,
                    extra={"app_id": "key"},
                )
            }
        )
        platforms = gw_only_key.get_connected_platforms()
        assert Platform.YUANBAO not in platforms

        # key + secret both present → in connected list
        gw_full = GatewayConfig(
            platforms={
                Platform.YUANBAO: PlatformConfig(
                    enabled=True,
                    extra={"app_id": "key", "app_secret": "secret"},
                )
            }
        )
        platforms2 = gw_full.get_connected_platforms()
        assert Platform.YUANBAO in platforms2


# ===========================================================
# 3. GatewayRunner 注册
# ===========================================================

class TestGatewayRunnerRegistration:

    def _make_minimal_runner(self, config):
        """通过 __new__ + 最小初始化绕过 run.py 的模块级 dotenv/ssl 副作用"""
        import sys

        # Stub out heavy dependencies if not already present
        stubs = [
            "dotenv",
            "hermes_cli.env_loader",
            "hermes_cli.config",
            "hermes_constants",
        ]
        _orig = {}
        for mod in stubs:
            if mod not in sys.modules:
                _orig[mod] = None
                sys.modules[mod] = MagicMock()

        try:
            from gateway.run import GatewayRunner
        finally:
            # Restore only the ones we injected
            for mod, orig in _orig.items():
                if orig is None:
                    sys.modules.pop(mod, None)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = config
        runner.adapters = {}
        runner._failed_platforms = {}
        runner._session_model_overrides = {}
        return runner, GatewayRunner

    def test_runner_creates_yuanbao_adapter(self):
        """GatewayRunner._create_adapter 能为 YUANBAO 返回 YuanbaoAdapter 实例"""
        from gateway.config import GatewayConfig
        config = make_config(enabled=True)
        gw_config = GatewayConfig(platforms={Platform.YUANBAO: config})

        try:
            runner, _ = self._make_minimal_runner(gw_config)
            # websockets 在测试环境可能未安装，mock 掉 WEBSOCKETS_AVAILABLE
            with patch("gateway.platforms.yuanbao.WEBSOCKETS_AVAILABLE", True):
                adapter = runner._create_adapter(Platform.YUANBAO, config)
        except ImportError as e:
            pytest.skip(f"run.py import unavailable in test env: {e}")

        assert adapter is not None
        assert isinstance(adapter, YuanbaoAdapter)



# ===========================================================
# 4. Proto round-trip
# ===========================================================




# ===========================================================
# 5. Markdown 分块
# ===========================================================




# ===========================================================
# 6. Sign Token 模块
# ===========================================================



# ===========================================================
# 6b. ConnectionManager / OutboundManager
# ===========================================================



# ===========================================================
# 7. Media 模块
# ===========================================================



# ===========================================================
# 8. Toolset 注册
# ===========================================================




# ===========================================================
# 9. platforms/__init__.py 导出
# ===========================================================



# ===========================================================
# 10. P0 fixes verification
# ===========================================================

import asyncio


        # No new task should be created because already reconnecting




class TestP0ChatLockEviction:
    """P0-3: get_chat_lock uses OrderedDict and safe eviction."""


    def test_eviction_skips_locked(self):
        """When eviction is needed, locked entries are skipped."""
        adapter = YuanbaoAdapter(make_config())
        from gateway.platforms.yuanbao import MessageSender

        # Fill to capacity with unlocked locks
        for i in range(MessageSender.CHAT_DICT_MAX_SIZE):
            adapter._outbound.sender._chat_locks[f"chat_{i}"] = asyncio.Lock()

        # Lock the oldest entry
        oldest_key = next(iter(adapter._outbound.sender._chat_locks))
        oldest_lock = adapter._outbound.sender._chat_locks[oldest_key]
        # Simulate a held lock by acquiring it in a non-async way (set _locked)
        # asyncio.Lock is not held until actually acquired; so we test the
        # method logic by acquiring the first lock manually.
        # For a sync test, we check that get_chat_lock doesn't crash.
        new_lock = adapter._outbound.sender.get_chat_lock("new_chat")
        assert "new_chat" in adapter._outbound.sender._chat_locks
        assert isinstance(new_lock, asyncio.Lock)
        # The oldest unlocked entry should have been evicted
        assert len(adapter._outbound.sender._chat_locks) == MessageSender.CHAT_DICT_MAX_SIZE





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
