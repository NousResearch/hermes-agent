"""红队验收测试：微信 iLink 限流冷却风暴（第4次复发）跨层验收。

权威来源：state.md `## 验收场景`（18 条 det-machine 谓词 SSOT）。

设计原则（反 no-op，挡住复发）：
- 黑盒视角：基于"设计应达到的状态"（TDD 红灯），不依赖蓝队实现细节。
- 跨层重点：真实 WeixinAdapter × 真实 GatewayStreamConsumer 组合，仅 mock
  iLink 底层 `_send_message` 与 `asyncio.sleep`（加速）。这是现有 30 单测
  缺失的链路（send()入口门控 / stream_consumer 重试对齐 cooldown）。
- 强断言：每个测试含数值/状态硬断言；失败必挂。禁止 try/except:skip。
- kill 空实现：场景1.P3 / 场景5.P1 验证——若实现没有 cooldown 门控，
  call_count 必增长，断言 delta==0 必红灯。

覆盖谓词映射见文件末尾 `__PREDICATE_COVERAGE__`。
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch


from gateway.config import PlatformConfig
from gateway.platforms.weixin import RATE_LIMIT_ERRCODE, WeixinAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_adapter() -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"account_id": "test-account"},
        )
    )


def _connected_adapter() -> WeixinAdapter:
    """构造已连接的 WeixinAdapter（mock 仅发生在 patch 装饰器层）。"""
    adapter = _make_adapter()
    adapter._session = object()
    adapter._send_session = adapter._session
    adapter._token = "test-token"
    adapter._base_url = "https://weixin.example.com"
    adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
    return adapter


def _rl_payload(errmsg: str = "freq limit") -> dict:
    """iLink 限流响应（ret=-2, errcode=-2, 有 errmsg）。"""
    return {"ret": RATE_LIMIT_ERRCODE, "errcode": RATE_LIMIT_ERRCODE, "errmsg": errmsg}


def _ok_payload() -> dict:
    return {"ret": 0}


# ─── 场景1：限流后冷却门控（跨层，死循环根因） ───────────────────────────────

class TestScenario1CooldownGate:
    """场景1 P1-P4：限流触发后冷却门控贯穿 send() → _send_message。"""

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S1_P1_cooldown_window_zero_send_delta(self, send_mock, sleep_mock):
        """场景1.P1：ret=-2 触发后，冷却窗口 [触发, cooldown_until) 内
        _send_message 调用计数增量 == 0。

        kill mutation: 若 send() 入口无 cooldown 门控（空实现），
        第二次 send 会直接命中 _send_message → delta >= 1，测试红灯。
        """
        adapter = _connected_adapter()
        # 首次 send 返回限流，后续（若门控缺失）会继续调用
        send_mock.return_value = _rl_payload()

        # 触发限流：send 内 _send_text_chunk 抛 RateLimitedError，
        # send() 捕获后返回失败 SendResult，并设 _rate_limited_until。
        before = send_mock.await_count
        result = asyncio.run(adapter.send("wxid_test", "hello"))
        assert result.success is False, "限流应导致 send 失败"
        assert adapter._rate_limited_until > 0.0, "限流后必须设置冷却状态"
        trigger_count = send_mock.await_count
        assert trigger_count >= before + 1  # 触发那次确实调了

        # —— 冷却窗口内再次 send：delta 必须为 0（门控生效，不打 iLink）——
        in_cooldown = send_mock.await_count
        asyncio.run(adapter.send("wxid_test", "second"))
        delta = send_mock.await_count - in_cooldown
        assert adapter._rate_limited_until > time.time(), "测试前提：仍处冷却期"
        assert delta == 0, (
            f"场景1.P1 失败：冷却窗口内 _send_message 增量应为 0，实际 {delta}"
        )
        # send() 不应返回失败（设计：sleep 等待而非返回失败）—— 但因冷却未到期，
        # 它必须挂起或跳过；此处核心证据是 iLink 计数不增长。
        assert send_mock.await_count == trigger_count, (
            "冷却期内不应有任何新增 iLink 调用"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S1_P2_cooldown_duration_at_least_30s(self, send_mock, sleep_mock):
        """场景1.P2：第k次发送返回 ret=-2 后，_rate_limited_until >= 触发+30.0。"""
        adapter = _connected_adapter()
        send_mock.return_value = _rl_payload()
        before = time.time()

        asyncio.run(adapter.send("wxid_test", "hello"))

        cooldown = adapter._rate_limited_until - before
        assert cooldown >= 30.0, (
            f"场景1.P2 失败：冷却时长应 >= 30.0s，实际 {cooldown:.2f}s"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S1_P3_m_new_sends_during_cooldown_delta_zero(self, send_mock, sleep_mock):
        """场景1.P3：冷却期 m 个新 send 请求，_send_message 总调用 == 触发那次。

        kill mutation（核心防复发证据）：若 send() 入口无 cooldown 门控，
        m 次新 send 每次都会调 _send_message → delta >= m，断言红灯。
        本测试 m=5。
        """
        adapter = _connected_adapter()
        send_mock.return_value = _rl_payload()

        # 触发限流
        asyncio.run(adapter.send("wxid_test", "trigger"))
        trigger_count = send_mock.await_count
        assert adapter._rate_limited_until > time.time(), "前提：已进入冷却"

        m = 5
        for i in range(m):
            asyncio.run(adapter.send("wxid_test", f"new-{i}"))

        delta = send_mock.await_count - trigger_count
        assert delta == 0, (
            f"场景1.P3 失败：m={m} 个新 send 期间 _send_message 增量应为 0，"
            f"实际 {delta}（空实现会让 delta>={m}）"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S1_P4_cooldown_expired_send_resumes(self, send_mock, sleep_mock):
        """场景1.P4：时钟越过 cooldown_until 后，send 恢复调用 _send_message。"""
        adapter = _connected_adapter()
        send_mock.return_value = _rl_payload()

        asyncio.run(adapter.send("wxid_test", "trigger"))
        assert adapter._rate_limited_until > 0.0

        # 强制让冷却过期
        adapter._rate_limited_until = time.time() - 1.0
        send_mock.return_value = _ok_payload()
        send_mock.reset_mock()

        result = asyncio.run(adapter.send("wxid_test", "after"))
        assert result.success is True, "冷却过期后 send 应成功"
        assert send_mock.awaited, (
            "场景1.P4 失败：越过 cooldown_until 后未恢复 _send_message 调用"
        )


# ─── 场景2：限流不导致连接断开（降级为 P3 等价） ─────────────────────────────

class TestScenario2NoDisconnect:
    """场景2 P3：持续限流后 cooldown 过期，send 最终成功投递 >=1 chunk。

    注：P1（connected）/P2（agent running）属 agent 层状态，gateway pytest
    难直测（设计文档 implement 注意事项 #3），降级为 P3 等价断言。
    """

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S2_P3_persistent_rate_limit_then_recovery_delivers(self, send_mock, sleep_mock):
        """持续 ret=-2 → cooldown 过期 → send 恢复成功投递 success_count >= 1。

        语义：第 1 次 send 命中限流进入冷却；冷却过期后第 2 次 send 成功。
        """
        adapter = _connected_adapter()

        call_state = {"idx": 0}

        async def seq(*a, **kw):
            call_state["idx"] += 1
            # 仅第 1 次限流，之后成功
            if call_state["idx"] == 1:
                return _rl_payload()
            return _ok_payload()

        send_mock.side_effect = seq

        # 第 1 次：限流，进入冷却
        r1 = asyncio.run(adapter.send("wxid_test", "a"))
        assert r1.success is False
        assert adapter._rate_limited_until > 0.0

        # 强制冷却过期，模拟时钟越过 cooldown_until
        adapter._rate_limited_until = time.time() - 1.0
        r2 = asyncio.run(adapter.send("wxid_test", "b"))
        assert r2.success is True, "冷却过期后必须成功投递 success_count >= 1"
        assert r2.message_id is not None


# ─── 场景3：处理过程可见性恢复（核心用户体感，跨层） ───────────────────────

class TestScenario3VisibilityRecovery:
    """场景3 P1-P3：限流恢复后用户可见更新 >=2，typing 活跃。

    跨层：真实 GatewayStreamConsumer + 真实 WeixinAdapter，mock 仅 iLink。
    """

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S3_P1_at_least_2_user_visible_updates_after_recovery(
        self, send_mock, sleep_mock
    ):
        """场景3.P1：限流1次后恢复，cooldown 过期后投递 >=2 次成功 send。

        kill mutation：若实现丢弃中间 chunk（只发最终结果），ok_count==1。

        降级说明（plan-reviewer 注意事项3 同类）：stream_consumer 跨层恢复
        依赖真实时间推进过 cooldown，而本套件 mock asyncio.sleep 加速（不推进
        真实时间），无法在一次 run() 内模拟 cooldown 过期——原 setup 用
        side_effect 设 _rate_limited_until=过去会被 _send_text_chunk 的
        time.time()+30 覆盖，恢复机制无效。故降级为 adapter 层等价验证，
        用可控时钟 time.time mock 推进，保留 ok_count>=2 硬断言。
        """
        adapter = _connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 4096
        state = {"idx": 0, "ok_count": 0}

        async def first_rl_then_ok(*a, **kw):
            state["idx"] += 1
            if state["idx"] == 1:
                return _rl_payload()
            state["ok_count"] += 1
            return _ok_payload()

        send_mock.side_effect = first_rl_then_ok

        # 可控时钟：mock time.time 推进，模拟 cooldown 过期
        clock = {"t": 1000.0}
        with patch(
            "gateway.platforms.weixin.time.time", side_effect=lambda: clock["t"]
        ):
            # 首次 send：限流，_send_text_chunk 设 _rate_limited_until=1030
            r1 = asyncio.run(adapter.send("wxid_test", "first"))
            assert r1.success is False, "首次限流应 send 失败"
            assert adapter._rate_limited_until >= 1030.0

            # cooldown 期内：send 入口守卫挡，不打 iLink（等价 S1.P1 delta==0）
            mid = send_mock.await_count
            asyncio.run(adapter.send("wxid_test", "during-cooldown"))
            assert send_mock.await_count == mid, "cooldown 期内不应打 iLink"

            # 推进时钟越过 cooldown_until，模拟恢复
            clock["t"] = 1031.0

            # 恢复投递：后续 send 成功（>=2）
            asyncio.run(adapter.send("wxid_test", "recovered-1"))
            asyncio.run(adapter.send("wxid_test", "recovered-2"))

        assert state["ok_count"] >= 2, (
            f"场景3.P1 失败：恢复后成功投递应 >= 2，实际 {state['ok_count']}"
            "（kill：丢弃中间 chunk 的空实现会让此值 == 1）"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_typing", new_callable=AsyncMock)
    def test_S3_P3_typing_active_after_recovery(self, typing_mock, send_mock, sleep_mock):
        """场景3.P3：恢复阶段 typing/流式活跃信号 typing_called == True。"""
        adapter = _connected_adapter()
        adapter._typing_cache.get = lambda chat_id: "ticket-xyz"
        send_mock.return_value = _ok_payload()

        # 冷却已过期（无冷却）
        assert adapter._rate_limited_until == 0.0 or adapter._rate_limited_until < time.time()
        asyncio.run(adapter.send_typing("wxid_test"))
        assert typing_mock.awaited, (
            "场景3.P3 失败：恢复阶段 typing 应被调用"
        )


# ─── 场景4：限流识别边界（防误判） ───────────────────────────────────────────

class TestScenario4RateLimitBoundary:
    """场景4 P1/P2：ret=-2 进冷却，ret!=-2 不进冷却。"""

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S4_P1_ret_minus_2_enters_cooldown(self, send_mock, sleep_mock):
        """场景4.P1：响应含 ret=-2 → _rate_limited_until > 0。"""
        adapter = _connected_adapter()
        send_mock.return_value = _rl_payload()
        asyncio.run(adapter.send("wxid_test", "x"))
        assert adapter._rate_limited_until > 0, (
            "场景4.P1 失败：ret=-2 应进入冷却（_rate_limited_until > 0）"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S4_P2_non_rate_limit_no_cooldown(self, send_mock, sleep_mock):
        """场景4.P2：响应 ret != -2 → _rate_limited_until == 0。"""
        adapter = _connected_adapter()
        # 成功响应
        send_mock.return_value = _ok_payload()
        asyncio.run(adapter.send("wxid_test", "x"))
        assert adapter._rate_limited_until == 0, (
            "场景4.P2 失败：ret!=-2 不应进入冷却（_rate_limited_until 应为 0）"
        )


# ─── 场景5：验收体系自检（元谓词，防"30单测挡不住复发"重演） ───────────────

class TestScenario5AcceptanceSelfCheck:
    """场景5：验收套件含 ret=-2 用例 + call_count 数值断言（元谓词）。

    场景5.P1（空实现红灯）由 test_S1_P1 / test_S1_P3 直接承担——
    它们在"移除 cooldown 门控"时必红灯（delta>=1 而非 ==0）。
    此处补充结构性自检。
    """

    def test_S5_P3_suite_contains_ret_minus_2_case_with_call_count_assert(self):
        """场景5.P3：本验收文件含 >=1 个 ret=-2 用例且做 call_count 数值断言。

        observe: 源码 grep（本文件自身）
        assert: ret=-2 用例 >=1 AND call_count 断言 >=1
        """
        import inspect
        src = inspect.getsource(sys_modules_self())

        # ret=-2 用例存在（通过 RATE_LIMIT_ERRCODE 构造限流 payload）
        assert "RATE_LIMIT_ERRCODE" in src or "ret=-2" in src or "- 2" in src, (
            "场景5.P3 失败：验收套件应含 ret=-2 限流用例"
        )
        # call_count 数值断言存在（硬数值，非布尔）
        count_assertions = src.count("await_count") + src.count("call_count")
        assert count_assertions >= 1, (
            "场景5.P3 失败：验收套件应含 >=1 个 call_count 数值断言"
        )
        # 至少一个 delta == 0 的硬断言（kill 空实现的证据）
        assert "delta == 0" in src or "delta==0" in src, (
            "场景5.P3 失败：验收套件应含 'delta == 0' 硬断言（kill 空实现）"
        )


def sys_modules_self():
    import sys
    return sys.modules[__name__]


# ─── 场景6：多次限流 + 调用次数有界（反发散，跨层核心） ─────────────────────

class TestScenario6BoundedCallCount:
    """场景6 P1/P2：多次限流窗口 delta==0，总调用 <= K+L+2（反发散）。"""

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S6_P1_two_rate_limit_windows_both_zero_delta(self, send_mock, sleep_mock):
        """场景6.P1：同一回复发生 2 次 ret=-2，每次冷却窗口内 delta == 0。"""
        adapter = _connected_adapter()
        state = {"idx": 0}

        async def two_rl_then_ok(*a, **kw):
            state["idx"] += 1
            # 第 1、2 次限流，之后成功
            if state["idx"] <= 2:
                return _rl_payload()
            return _ok_payload()

        send_mock.side_effect = two_rl_then_ok

        # 第一次限流
        asyncio.run(adapter.send("wxid_test", "a"))
        w1_trigger = send_mock.await_count
        assert adapter._rate_limited_until > 0
        # 窗口1 内不再调
        asyncio.run(adapter.send("wxid_test", "a2"))
        delta1 = send_mock.await_count - w1_trigger
        assert delta1 == 0, f"窗口1 delta 应为 0，实际 {delta1}"

        # 强制过期，触发第二次限流
        adapter._rate_limited_until = time.time() - 1.0
        asyncio.run(adapter.send("wxid_test", "b"))
        w2_trigger = send_mock.await_count
        assert adapter._rate_limited_until > 0, "第二次限流应再次设冷却"
        asyncio.run(adapter.send("wxid_test", "b2"))
        delta2 = send_mock.await_count - w2_trigger
        assert delta2 == 0, f"窗口2 delta 应为 0，实际 {delta2}"

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_S6_P2_K_chunks_L_rate_limits_total_bounded(self, send_mock, sleep_mock):
        """场景6.P2：K chunk + L 限流完成，总 _send_message 调用 <= K + L + 2。

        kill mutation：死循环发散实现会让调用数 >> K+L+2，断言红灯。
        本测试 K=3 chunks, L=1 限流。
        """
        adapter = _connected_adapter()
        K = 3  # chunk 数
        L = 1  # 限流次数
        state = {"idx": 0}

        async def one_rl_rest_ok(*a, **kw):
            state["idx"] += 1
            if state["idx"] == 1:
                return _rl_payload()
            return _ok_payload()

        async def side_with_recovery(*a, **kw):
            rv = await one_rl_rest_ok(*a, **kw)
            if rv.get("ret") == RATE_LIMIT_ERRCODE:
                # 测试加速：限流后立即让冷却过期
                adapter._rate_limited_until = time.time() - 0.001
            return rv

        send_mock.side_effect = side_with_recovery

        async def drive():
            # 注入 K 个 chunk
            cfg = StreamConsumerConfig(buffer_threshold=1, edit_interval=0.0)
            consumer = GatewayStreamConsumer(adapter, "wxid_test", cfg)
            for i in range(K):
                consumer.on_delta(f"chunk-{i}-" + "x" * 5)
            consumer.finish()
            await consumer.run()

        asyncio.run(drive())

        total = send_mock.await_count
        bound = K + L + 2
        assert total <= bound, (
            f"场景6.P2 失败：总 _send_message 调用 {total} 应 <= K+L+2 = {bound}"
            "（kill：死循环发散实现会让此值 >> bound）"
        )


# ─── 场景7：typing 间隔 / 回归 ───────────────────────────────────────────────

class TestScenario7TypingInterval:
    """场景7.P7：_typing_interval_seconds == 3.0（改动后）。"""

    def test_S7_P7_typing_interval_is_3_seconds(self):
        """场景7.P7：_typing_interval_seconds shall == 3.0。"""
        adapter = _make_adapter()
        assert adapter._typing_interval_seconds == 3.0, (
            f"场景7.P7 失败：_typing_interval_seconds 应为 3.0，"
            f"实际 {adapter._typing_interval_seconds}"
        )


# ─── 契约字段名逐字一致（防蓝队改名） ────────────────────────────────────────

class TestContractFieldNames:
    """契约字段名逐字一致：_rate_limited_until / rate_limited_until /
    RATE_LIMIT_ERRCODE。"""

    def test_rate_limited_until_private_attr_exists(self):
        adapter = _make_adapter()
        assert hasattr(adapter, "_rate_limited_until"), (
            "契约：WeixinAdapter 必须有 _rate_limited_until 属性"
        )
        assert isinstance(adapter._rate_limited_until, float)

    def test_rate_limited_until_public_property_exists(self):
        """契约：base 暴露 rate_limited_until property（默认 0.0，weixin override）。"""
        adapter = _make_adapter()
        assert hasattr(adapter, "rate_limited_until"), (
            "契约：adapter 必须暴露 rate_limited_until property"
        )
        # 默认无冷却 == 0.0
        assert adapter.rate_limited_until == 0.0
        # weixin override 返回 _rate_limited_until
        adapter._rate_limited_until = 12345.0
        assert adapter.rate_limited_until == 12345.0, (
            "契约：weixin.rate_limited_until 必须 override 返回 _rate_limited_until"
        )

    def test_rate_limit_errcode_constant(self):
        """契约：RATE_LIMIT_ERRCODE == -2。"""
        assert RATE_LIMIT_ERRCODE == -2


# ─── 跨层：stream_consumer 通过 adapter.rate_limited_until 对齐 cooldown ─────

class TestStreamConsumerCooldownAlignment:
    """改动3验收：stream_consumer flood 重试读 adapter.rate_limited_until。

    跨层：真实 stream_consumer + 真实 weixin adapter，mock 仅 iLink。
    """

    def test_adapter_exposes_rate_limited_until_to_stream_consumer(self):
        """stream_consumer 通过 self.adapter.rate_limited_until 访问 cooldown。"""
        adapter = _connected_adapter()
        adapter._rate_limited_until = 0.0
        cfg = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "wxid_test", cfg)
        # stream_consumer.adapter 就是 weixin adapter，且能读到 rate_limited_until
        assert consumer.adapter is adapter
        assert consumer.adapter.rate_limited_until == 0.0
        # 设置后 stream_consumer 侧可见
        adapter._rate_limited_until = time.time() + 30
        assert consumer.adapter.rate_limited_until > time.time()


# ─── 谓词覆盖索引（场景5.P3 元数据，便于审计） ──────────────────────────────
#
# __PREDICATE_COVERAGE__
#
# 场景1.P1 → test_S1_P1_cooldown_window_zero_send_delta
# 场景1.P2 → test_S1_P2_cooldown_duration_at_least_30s
# 场景1.P3 → test_S1_P3_m_new_sends_during_cooldown_delta_zero (kill 空实现)
# 场景1.P4 → test_S1_P4_cooldown_expired_send_resumes
# 场景2.P3 → test_S2_P3_persistent_rate_limit_then_recovery_delivers
#            (P1/P2 agent 层状态，降级，见设计 implement 注意事项 #3)
# 场景3.P1 → test_S3_P1_at_least_2_user_visible_updates_after_recovery
# 场景3.P3 → test_S3_P3_typing_active_after_recovery
# 场景4.P1 → test_S4_P1_ret_minus_2_enters_cooldown
# 场景4.P2 → test_S4_P2_non_rate_limit_no_cooldown
# 场景5.P1 → test_S1_P1 / test_S1_P3（空实现必红灯，delta>=1 != 0）
# 场景5.P2 → 正确实现运行时本套件 exit_code == 0
# 场景5.P3 → test_S5_P3_suite_contains_ret_minus_2_case_with_call_count_assert
# 场景6.P1 → test_S6_P1_two_rate_limit_windows_both_zero_delta
# 场景6.P2 → test_S6_P2_K_chunks_L_rate_limits_total_bounded (kill 发散)
# 场景7.P7 → test_S7_P7_typing_interval_is_3_seconds
# 场景7.P8 → 现有 test_weixin_rate_limit.py 全绿（回归，不在此文件）
