"""ChatCardEngine — per-chat CardKit streaming-card session engine.

与注入模式的 StreamCardController（message_id 键、真 delta 追加）不同，这里会话以
**chat** 为键：GatewayStreamConsumer 的 draft 帧（send_draft(chat_id, draft_id, content)）
天然不带 message 身份，且 content 是**全量快照**（transport._draft_push 语义），所以
ANSWER segment 采用「整段置换」而非追加，避免快照重放导致文本重复。

段模型（v1 简化）：REASONING / ANSWER 各至多一个（全回合合并），TOOL 面板首个工具
事件时追加并在原位整体刷新 —— 与 fork 初始卡「工具面板在正文区」的视觉一致。
复用注入模式已验证的底层件：SegmentState / ToolUseTracker / FlushController /
builder（CardKit v2）/ FeishuClient（SDK 封装）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .builder import (
    HEARTBEAT_ELEMENT_ID,
    MAX_FAVORITE_MODELS,
    build_complete_card,
    build_cron_card,
    build_streaming_card_v2,
    cap_reasoning_text,
)
from .markdown import optimize_markdown_style
from .flush import FlushController
from .segment_helper import (
    build_add_segment_action,
    build_reasoning_finalized_action,
    build_tool_update_action,
    tool_segment_end,
)
from .segments import Segment, SegmentState, SegmentType
from .text import strip_reasoning_tags
from .tooluse import ToolUseTracker

# 本包 logger 不进 gateway.log（hermes logging 配置问题，见 AGENTS.md）——
# 诊断日志统一走 gateway.run logger。
logger = logging.getLogger(__name__)

# 心跳/interim 文本进卡前的字符上限（防异常长文本撑爆状态行）
_HEARTBEAT_MAX_LEN = 300


@dataclass
class ChatSession:
    """单 chat 的流式卡片会话（一回合一张卡）."""

    chat_id: str
    reply_to: str | None = None  # 回复锚（用户消息 id）；None = 直发 chat
    created_at: float = field(default_factory=time.time)
    state: str = "creating"  # creating → streaming → completed / failed
    card_id: str | None = None
    card_msg_id: str | None = None
    answer_seg: Segment | None = None
    thread_id: str | None = None  # 话题线程（Feishu 话题/回复链）；None = 主聊
    redirected: bool = False  # 诊断标记：本回合被 ↪ redirect 重启（seal/replace 日志用）
    model_switch: dict[str, str] | None = None  # footer 循环切换按钮数据（NOTICE 重渲透传）
    redirect_anchor: str | None = None  # redirect 新卡锚（用户纠正消息 id，来自 ack reply_to）
    # redirect 残尾拦截：旧 model 请求被取消前挤出的快照是老回合内容的超集，
    # 前缀比对命中即丢弃，防老答案闪进新卡；首个不相关内容通过后清空
    straggler_guard: str = ""
    # 本回合可接受的 draft 锚集合（followup 边界判定用）。redirect 场景同回合
    # 会出现两个锚（新指令消息 + 老回合消息），都算本回合不算新回合。
    accepted_anchors: set[str] = field(default_factory=set)
    _reasoning_logged: bool = False
    tool_seg: Segment | None = None
    heartbeat_text: str = ""
    sequence: int = 0
    card_create_task: asyncio.Task | None = None
    segment_state: SegmentState = field(default_factory=SegmentState)
    tool_tracker: ToolUseTracker = field(default_factory=ToolUseTracker)
    flush: FlushController | None = None

    completed_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in ("completed", "failed")


class ChatCardEngine:
    """chat 键流式卡片引擎 — 由 StreamingFeishuAdapter 驱动.

    生命周期：首条 draft 帧触发建卡（后台任务，不阻塞 transport）→ 后续帧整段
    置换 answer 文本 → 回合终态经 :meth:`complete` 渲染完成卡。
    """

    def __init__(self, client: Any, *, body_text_size: str = "normal_v2",
                 show_tool_use: bool = True, header_enabled: bool = False,
                 width_mode: str = "default",
                 footer_fields: list[list[str]] | None = None,
                 footer_show_label: bool = False,
                 footer_enabled: bool = True,
                 model_cycle: list[str] | None = None,
                 tool_panel_expanded: bool = False,
                 reasoning_panel_expanded: bool = False) -> None:
        self._client = client
        self._body_text_size = body_text_size
        self._show_tool_use = show_tool_use
        self._header_enabled = header_enabled
        self._width_mode = width_mode
        # 完成卡面板默认展开（streaming.tool_panel_expanded / reasoning_panel_expanded，
        # 未单独配置回退旧键 panel_expanded）——流式期恒折叠，标题行动作标签已够用
        self._tool_panel_expanded = tool_panel_expanded
        self._reasoning_panel_expanded = reasoning_panel_expanded
        # footer 形态与注入模式同源（vendor Config 读 HERMES_HOME/config.yaml
        # 的 streaming.footer 段），两形态渲染一致
        self._footer_fields = footer_fields or [["elapsed", "model", "context"]]
        self._footer_show_label = footer_show_label
        self._footer_enabled = footer_enabled
        # footer 模型切换按钮的轮换清单（streaming.footer.model_cycle，缺省
        # model.default + fallback_providers 推导）——<2 个或当前模型不在清单则不出按钮。
        # 卡片内「常用模型」管理（v0.19.0）落盘 JSON 并在启动时优先于推导清单
        self._model_cycle = [m.strip() for m in (model_cycle or []) if str(m).strip()]
        self._model_cycle_fallback = list(self._model_cycle)
        favorites = self._load_model_favorites()
        if favorites:
            self._model_cycle = favorites
        # hermes 全部可用模型（管理卡候选池）：(fetched_at, providers) 进程内缓存
        self._candidates_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._sessions: dict[str, ChatSession] = {}
        # 会话键 = (chat, thread)：Feishu 话题在 hermes 是独立会话（dm:oc_x:omt_y），
        # 卡片会话必须同粒度隔离，否则话题与主聊互相串写（2026-10-09 实测混写）
        # chat → 回合累计 usage（post_api_request 钩子按 session_id 归组，complete 消费）
        self._usage: dict[str, dict[str, Any]] = {}
        # chat → 最近一次 usage 上报的模型名（🧠⇄ 选择卡的「当前」显示用）
        self._chat_models: dict[str, str] = {}
        self._last_model = ""
        # chat → 见过的入站消息 id（on_processing_start 登记，drain 消息在 drain
        # 开始时也会触发）——followup 拆卡的门槛：新锚必须是真入站消息
        self._inbound_seen: dict[str, dict[str, float]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Any = None

    # ── usage 聚合（post_api_request 钩子 → footer tokens/t/s）──

    @staticmethod
    def _chat_of_session(session_id: str) -> str:
        """agent session key（agent:main:feishu:dm:<chat_id>）→ chat_id；解析失败返 ""."""
        m = re.search(r"oc_[0-9a-f]{16,40}", str(session_id or ""))
        return m.group(0) if m else ""

    def record_usage(self, session_id: str, usage: dict[str, Any] | None, model: str = "") -> None:
        """单次 API 调用的 usage 累加进所属 chat 的回合桶."""
        if not usage:
            return
        chat_id = self._chat_of_session(session_id)
        bucket = self._usage.setdefault(
            chat_id, {"input": 0, "output": 0, "model": "", "context_max": 0,
                      "first_started": None, "last_ended": None})
        # input 取最后值不累加：每轮 API 的 prompt 都含全量历史，累加会虚高
        # 数量级（实测 2000 字回合累出 88 万）；注入模式口径即最后值
        bucket["input"] = max(bucket["input"], int(usage.get("prompt_tokens") or 0))
        bucket["output"] += int(usage.get("completion_tokens")
                                or usage.get("output_tokens") or usage.get("total_tokens") or 0)
        if model:
            bucket["model"] = model
        if usage.get("context_length"):
            bucket["context_max"] = int(usage["context_length"])
            bucket["context_used"] = int(usage.get("prompt_tokens") or 0)
        # API 时间跨度（含排队/工具间隔）：t/s 分母用它，口径对齐注入模式的 _turn_seconds
        if usage.get("started_at"):
            if bucket["first_started"] is None:
                bucket["first_started"] = usage["started_at"]
            bucket["last_ended"] = usage.get("ended_at") or bucket["last_ended"]
        if model:
            self._chat_models[chat_id or ""] = model
            self._last_model = model

    def _pop_usage(self, chat_id: str) -> dict[str, Any] | None:
        bucket = self._usage.pop(chat_id, None) or self._usage.pop("", None)
        if not bucket or not (bucket["input"] or bucket["output"]):
            return None
        return bucket

    @staticmethod
    def _skey(chat_id: str, thread_id: str | None) -> str:
        """卡片会话键：(chat, thread) 复合；thread 为空即主聊，与旧键兼容."""
        return f"{chat_id}::{thread_id}" if thread_id else chat_id

    def on_turn_started(self, chat_id: str, thread_id: str | None = None) -> None:
        """回合开始（send_typing 时机）→ 立即建卡.

        带工具的回合 draft 要等工具全部跑完才出现，此前用户什么都看不到
        （注入模式消息进来就建卡）。typing 先行建卡（无锚直发 chat），工具
        事件与正文随后进卡；DM/普通群与注入模式体感一致。
        """
        self._capture_loop()
        key = self._skey(chat_id, thread_id)
        existing = self._sessions.get(key)
        if (existing is not None and existing.is_terminal
                and time.time() - existing.completed_at < 5.0):
            # typing 是 2s 心跳循环：回合刚完成后的尾巴调用，不是新回合——
            # 重建会得到一张永挂 loading 的空卡
            return
        existing = self._sessions.get(key)
        if existing is None or existing.is_terminal:
            self._ensure_session(chat_id, thread_id)
            logger.info("turn started: chat=%s thread=%s",
                         chat_id[:12], (thread_id or "-")[:16])
        # typing 2s 心跳循环的重复调用：会话健在时静默（此前每 2s 刷一条）

    def note_inbound(self, chat_id: str, message_id: str) -> None:
        """入站消息登记（adapter.on_processing_start 调用，drain 消息在 drain
        开始时同样触发）——followup 拆卡门槛的数据源：draft 锚只有在里面才
        允许拆卡。工具边界换 consumer 的重锚不是入站消息，不得拆卡
        （2026-10-09 实测：redirected 回合写文件后锚变新 id，误拆成
        「Done.」空卡 + 内容全落第三张卡）。"""
        if not chat_id or not message_id:
            return
        bucket = self._inbound_seen.setdefault(chat_id, {})
        bucket[message_id] = time.time()
        while len(bucket) > 50:  # 每_chat 只留最近 50 条，防无界增长
            bucket.pop(next(iter(bucket)))

    def _inbound_verified(self, chat_id: str, message_id: str | None) -> bool:
        """锚是否为引擎亲眼见过的入站消息."""
        if not message_id:
            return False
        return message_id in self._inbound_seen.get(chat_id, {})

    def mark_redirect(self, chat_id: str, anchor: str | None = None,
                      thread_id: str | None = None) -> None:
        """↪ redirect ack（用户纠正、interrupt 模式）→ **立即收旧开新**.

        hermes redirect 是同一运行回合改锚续跑（agent.redirect 取消当前 model
        请求、注入纠正、循环重试），不换消息身份——draft 锚不变，锚变化检测
        覆盖不到。此前边界挂在「首个有内容的 draft」上：思考/工具阶段全部画
        在老卡，新卡开卡即接近成品。现在 ack 一到就拆：旧卡红标 NOTICE 收尾，
        新卡立刻以纠正消息为锚建卡（loading 起步），本回合后续 reasoning/
        工具/正文从第一毫秒起全部流进新卡。

        anchor 是 ack 的 reply_to（= 用户纠正消息 id，hermes _send_busy_reply
        锚到新消息）——新卡 reply 引用它；不带时回退旧锚。旧请求取消前的
        残尾快照由 straggler_guard 在 on_draft 里前缀拦截。"""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return
        self._capture_loop()
        session.redirected = True  # 诊断：seal/replace 日志标识这是被重启的回合
        if anchor:
            session.redirect_anchor = anchor
        logger.info("redirect boundary at ack: chat=%s anchor=%s",
                     chat_id[:12], anchor or "-")
        new = ChatSession(
            chat_id=chat_id, thread_id=thread_id,
            reply_to=session.redirect_anchor or session.reply_to,
            straggler_guard=(session.answer_seg.text if session.answer_seg else ""))
        # redirected 回合的 draft 锚会中途变回老消息 id（工具边界换 consumer
        # 重新锚定回合身份）——两个锚都算本回合，防 followup 边界拦腰拆卡
        new.accepted_anchors = {a for a in (session.redirect_anchor,
                                            session.reply_to) if a}
        # 立刻置终态：seal 是异步任务（cardkit close+update 要走网络），期间旧
        # 会话若仍计为 streaming，全局 reasoning 路由会把新回合的思考误送旧卡
        session.state = "completed"
        self._sessions[self._skey(chat_id, thread_id)] = new
        self._ensure_session(chat_id, thread_id)  # 新卡立刻建，不等首条 draft
        self._seal_session(session, notice="↪ 任务已按新指令重启，结果见下方新卡片")

    # ── 会话查询 ──

    def session_for(self, chat_id: str, thread_id: str | None = None) -> ChatSession | None:
        return self._sessions.get(self._skey(chat_id, thread_id))

    def active_session(self, chat_id: str, thread_id: str | None = None) -> ChatSession | None:
        """活跃（非终态）会话 — send() 完成路由的判定入口."""
        session = self._sessions.get(self._skey(chat_id, thread_id))
        if session is not None and not session.is_terminal:
            return session
        return None

    def latest_session_for_chat(self, chat_id: str) -> ChatSession | None:
        """该 chat（任意 thread）最近创建的会话——bg 通知合并「最近一张卡」用
        （bg 交付的 thread_id 是来源标记，合并目标是聊天里最新那张卡）."""
        candidates = [s for s in self._sessions.values() if s.chat_id == chat_id]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.created_at)

    def streaming_sessions(self) -> list[ChatSession]:
        return [s for s in self._sessions.values() if s.state == "streaming"]

    def open_sessions(self) -> list[ChatSession]:
        """未终态会话（creating/streaming）——usage 路由的「回合进行中」判据
        （post_api_request 落在回合中段，首条 draft 前会话还是 creating）."""
        return [s for s in self._sessions.values() if not s.is_terminal]

    def last_card_msg_id(self, chat_id: str, thread_id: str | None = None) -> str | None:
        """最近一张卡的消息 id（文档交付 reply 锚点用，终态也算）."""
        session = self._sessions.get(self._skey(chat_id, thread_id))
        return session.card_msg_id if session else None

    # ── 流式输入 ──

    def on_draft(self, chat_id: str, content: str, reply_to: str | None = None,
                 thread_id: str | None = None) -> None:
        """draft 帧（全量快照）— 确保会话与建卡，整段置换 answer 文本.

        由 async 的 send_draft 调用：此处捕获引擎事件循环（后续 format_tool_event
        可能从 agent 工作线程同步到达，需 call_soon_threadsafe 回环）。
        reply_to 是 transport _draft_metadata 带的用户消息锚（首轮记录，后续幂等）。
        """
        self._capture_loop()
        session = self._ensure_session(chat_id, thread_id)
        guard = session.straggler_guard
        if guard and content and (content.startswith(guard) or guard.startswith(content)):
            # redirect 残尾：旧请求取消前挤出的快照（老回合内容的超集），
            # 丢弃——否则新卡开头闪现老答案，思考型模型下要挂到新请求出文本
            logger.info("redirect straggler draft dropped: len=%d",
                         len(content))
            return
        if guard:
            session.straggler_guard = ""  # 首个真实新内容已过，不再拦（防误伤后续帧）
        accepted = session.accepted_anchors or (
            {session.reply_to} if session.reply_to else set())
        if (session.state != "creating" and reply_to and session.reply_to
                and reply_to not in accepted):
            if not self._inbound_verified(chat_id, reply_to):
                # 新锚不是入站消息 = 工具边界换 consumer 的重锚（同一回合继续，
                # 2026-10-09 实测：redirected 回合写文件后锚变全新 id）——改锚
                # 不拆卡。拆卡门槛必须是「新锚 = 新入站消息」（排队 followup
                # 被 drain，processing_start 在 drain 开始时触发，先于首帧）。
                logger.info("anchor swing (not an inbound msg), "
                             "re-anchor in place: %s -> %s", session.reply_to, reply_to)
                session.reply_to = reply_to
                session.accepted_anchors.add(reply_to)
            else:
                # draft 锚变成新入站消息 = 新回合开始（排队 followup 被 drain）：
                # 旧卡按已有内容收尾（绿色完成态），新回合开新卡。
                logger.info(
                    "followup boundary: anchor %s -> %s (inbound), "
                    "sealing old card", session.reply_to, reply_to)
                old_session = session
                new = ChatSession(chat_id=chat_id, thread_id=thread_id, reply_to=reply_to)
                new.accepted_anchors = {reply_to}
                self._sessions[self._skey(chat_id, thread_id)] = new
                session = self._ensure_session(chat_id, thread_id)  # 新会话补建卡 task
                self._seal_session(old_session)
        if session.reply_to is None and reply_to:
            session.reply_to = reply_to
            session.accepted_anchors = {reply_to}
        # 话题会话：锚已到手（首帧 draft 带 reply_to_message_id）→ 此刻才建卡
        self._maybe_start_card_task(session)
        if session.answer_seg is None:
            # 空文本走 on_answer_delta：在正确位置（reasoning 之后）新建空 ANSWER 段
            session.segment_state.on_answer_delta("")
            session.answer_seg = session.segment_state.segments[-1]
        # 与注入模式 on_answer 同款防线：<thinking>/<thought> 标签形态的
        # 思考泄漏清洗（模型裸文本碎片两种模式都无法剥，此处只防标签形态）
        session.answer_seg.text = strip_reasoning_tags(content)
        session.answer_seg.dirty = True
        self._schedule(session)

    def on_tool_start(self, tool_name: str, detail: str = "",
                      anchor: tuple[str, str | None] | None = None) -> None:
        """format_tool_event(ToolCallChunk) 的落点 — 记录步骤并刷新工具面板.

        工具事件可能先于任何 draft（带工具回合的常态）：无会话时也建卡，
        让工具面板从第一步就滚动在用户眼里。``anchor``=(chat, thread) 是
        adapter 侧记录的当回合锚（format_tool_event 落在当回合的 adapter 上，
        adapter 知道自己在服务谁）——并发多会话时按锚路由，杜绝 B 会话的
        工具名/参数画进 A 会话的卡（2026-10-10 catalog review 指出的串扰）。
        """
        session = None
        if anchor is not None:
            session = self.active_session(anchor[0], anchor[1])
        if session is None:
            session = self._any_active_session()
        if session is None:
            # 工具事件先于 typing 的兜底：在事件循环线程上捕获 loop 建""占位
            # 会话（typing 到达后按 chat 归位）；跨线程无 loop 时丢弃该事件
            try:
                self._capture_loop()
            except RuntimeError:
                return
            session = self._ensure_session("")
        session.tool_tracker.record_start(tool_name, detail)
        if session.tool_seg is None:
            session.segment_state.on_tool_event(len(session.tool_tracker.build_display_steps()))
            session.tool_seg = session.segment_state.segments[-1]
            session.tool_seg.created = False  # 待 flush 建元素
        session.tool_seg.dirty = True
        self._schedule(session)

    def on_tool_end(self, tool_name: str, *, error: str = "", output: str = "") -> None:
        session = self._any_active_session()
        if session is None:
            return
        session.tool_tracker.record_end(tool_name, error=error, output=output)
        if session.tool_seg is not None:
            session.tool_seg.dirty = True
        self._schedule(session)

    def on_reasoning(self, chat_id: str, text: str, *, strict: bool = False) -> None:
        """reasoning 增量 — 多会话并发时丢弃（防串扰）.

        ``strict``（session_id 路由命中时）：只进 chat_id 自己的会话，目标
        不活跃就丢弃——绝不回落单活跃兜底（回落=把 B 的思考写进 A 的卡，
        catalog review 指出的串扰正是这一步）。
        """
        target = self.active_session(chat_id) if strict else self._resolve_reasoning_target(chat_id)
        if target is None:
            return
        if not target._reasoning_logged:
            target._reasoning_logged = True
            logger.info("reasoning streaming into card: chat=%s",
                         target.chat_id[:12])
        target.segment_state.on_reasoning_delta(text)
        self._schedule(target)

    def on_heartbeat(self, chat_id: str, text: str, thread_id: str | None = None) -> None:
        """心跳/interim 文本 → 卡片末尾状态行."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return
        session.heartbeat_text = text[:_HEARTBEAT_MAX_LEN]
        self._schedule(session)

    # ── 完成 ──

    async def complete(self, chat_id: str, final_text: str, *, is_error: bool = False,
                       duration: float | None = None, tokens: dict[str, int] | None = None,
                       model: str = "", thread_id: str | None = None) -> str | None:
        """回合终态：渲染完成卡（含 footer），返回卡片消息 id."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return None
        logger.info("complete: chat=%s state=%s segs=%d redirected=%s",
                     chat_id[:12], session.state, len(session.segment_state.segments),
                     session.redirected)
        if session.card_create_task is not None:
            await session.card_create_task
        if session.flush is not None:
            session.flush.mark_completed()
        if session.state == "failed" or session.card_id is None:
            return session.card_msg_id  # 建卡已失败：交回原生文本保底
        if final_text and session.answer_seg is not None:
            session.answer_seg.text = final_text
            session.answer_seg.dirty = False  # 完成卡整体重渲，不再单独流式
        elif final_text:
            # 短回答可能整段直发、无任何 draft 帧（仅 reasoning 段）——用终态文本
            # 补建 ANSWER 段，否则正文丢失、完成卡渲染成「Done.」占位（2026-10-09 实测）
            session.segment_state.on_answer_delta("")
            session.answer_seg = session.segment_state.segments[-1]
            session.answer_seg.text = strip_reasoning_tags(final_text)
            session.answer_seg.dirty = False
        session.segment_state.finalize_segments(
            len(session.tool_tracker.build_display_steps()))
        session.state = "failed" if is_error else "completed"
        session.completed_at = time.time()
        usage = self._pop_usage(chat_id)
        if (usage and usage.get("first_started") and usage.get("last_ended")
                and usage["last_ended"] > usage["first_started"] and duration is None):
            # 真实回合时长 = 首次 API 开始 → 最后一次 API 结束（含工具时间）；
            # 会话时长只覆盖流式尾巴，会让 t/s 虚高一个数量级
            duration = usage["last_ended"] - usage["first_started"]
        if usage:
            logger.info(
                "footer usage: chat=%s in=%d out=%d model=%s",
                chat_id[:12], usage.get("input", 0), usage.get("output", 0),
                usage.get("model") or model or "?")
        if tokens:
            usage = {"input": tokens.get("input_tokens", 0),
                     "output": tokens.get("output_tokens", 0), "model": model}
        session.model_switch = self._model_switch_data(
            (usage or {}).get("model") or model)
        card = build_complete_card(
            segments=session.segment_state.segments,
            all_tool_steps=session.tool_tracker.build_display_steps(),
            footer_data=self._footer_data(session, duration, usage, model),
            footer_fields=self._footer_fields,
            footer_show_label=self._footer_show_label,
            footer_enabled=self._footer_enabled,
            is_error=is_error,
            header_enabled=self._header_enabled,
            body_text_size=self._body_text_size,
            show_tool_use=self._show_tool_use,
            width_mode=self._width_mode,
            model_switch=session.model_switch,
            tool_panel_expanded=self._tool_panel_expanded,
            reasoning_panel_expanded=self._reasoning_panel_expanded,
        )
        try:
            session.sequence += 1
            await self._client.cardkit_close_streaming(session.card_id, sequence=session.sequence)
            session.sequence += 1
            await self._client.cardkit_update(session.card_id, card, sequence=session.sequence)
        except Exception as e:
            logger.warning("plugin complete card update failed: chat=%s err=%s", chat_id, e)
        return session.card_msg_id

    async def append_notice(self, chat_id: str, text: str,
                            thread_id: str | None = None) -> str | None:
        """把 background/系统通知追加进该 chat 最近一张卡（跨回合合并）.

        会话可能是 COMPLETED（bg 回合在主回合完成后到达）——追加 NOTICE segment
        后整体重渲完成卡；会话仍活跃则交由 flush 建 NOTICE 元素。
        返回卡片消息 id；无可用卡片返回 None（调用方走原生文本保底）。
        """
        session = self._sessions.get(self._skey(chat_id, thread_id))
        if session is None or session.card_id is None:
            return None
        session.segment_state.add_notice(text)
        if session.is_terminal:
            card = build_complete_card(
                segments=session.segment_state.segments,
                all_tool_steps=session.tool_tracker.build_display_steps(),
                footer_data=self._footer_data(session, None, None, ""),
                footer_fields=self._footer_fields,
                footer_show_label=self._footer_show_label,
                footer_enabled=self._footer_enabled,
                header_enabled=self._header_enabled,
                body_text_size=self._body_text_size,
                show_tool_use=self._show_tool_use,
                width_mode=self._width_mode,
                model_switch=getattr(session, "model_switch", None),
                tool_panel_expanded=self._tool_panel_expanded,
                reasoning_panel_expanded=self._reasoning_panel_expanded,
            )
            session.sequence += 1
            try:
                await self._client.cardkit_update(session.card_id, card, sequence=session.sequence)
            except Exception as e:
                logger.warning("append notice update failed: %s", e)
                return None
        else:
            self._schedule(session)
        logger.info("notice merged into card: chat=%s len=%d",
                     chat_id[:12], len(text))
        return session.card_msg_id

    async def send_cron_card(self, chat_id: str, content: str, *, task_name: str = "",
                             job_id: str = "", run_time: str = "",
                             template: str = "blue") -> str | None:
        """cron 结果一次性卡片（⏰ 静态卡，直发 chat，不建流式会话）.

        与 :meth:`append_notice` 同属旁路发送：无会话状态、无后续 edit，失败返回
        None 由调用方落回原生文本。失败通知 template="red"（红 header 一眼可辨）。
        """
        card = build_cron_card(content, task_name=task_name, run_time=run_time,
                               template=template)
        try:
            card_id = await self._client.cardkit_create(card)
            # cron 投递无 reply 锚，直发 chat（同 _do_create_card 无锚分支）
            msg_id: str | None = await self._client.send_card_to_chat(
                chat_id, {"type": "card", "data": {"card_id": card_id}})
            return msg_id
        except Exception as e:
            logger.warning("cron card send failed: chat=%s err=%s",
                            chat_id[:12], e)
            return None

    async def abandon(self, chat_id: str, thread_id: str | None = None) -> None:
        """流被弃（中断/异常）— 尽力按错误收尾，避免永久转圈卡."""
        session = self.active_session(chat_id, thread_id)
        if session is None:
            return
        await self.complete(chat_id, session.answer_seg.text if session.answer_seg else "",
                            is_error=True)

    # ── 内部 ──

    def _capture_loop(self) -> None:
        """捕获引擎事件循环（首个 async 入口调用；后续跨线程调度用）."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._thread = threading.current_thread()

    def _ensure_session(self, chat_id: str, thread_id: str | None = None) -> ChatSession:
        key = self._skey(chat_id, thread_id)
        session = self._sessions.get(key)
        if session is None or session.is_terminal:
            # 旧卡（终态）保留在聊天里，新回合开新会话
            if session is not None:
                logger.info(
                    "session replaced: chat=%s old_state=%s "
                    "completed_ago=%s redirected=%s",
                    chat_id[:12], session.state,
                    f"{time.time() - session.completed_at:.1f}s"
                    if session.completed_at else "never",
                    session.redirected)
            new = ChatSession(chat_id=chat_id, thread_id=thread_id)
            self._sessions[key] = new
            session = new
        self._maybe_start_card_task(session)
        return session

    def _maybe_start_card_task(self, session: ChatSession) -> None:
        """建卡任务启动；话题会话例外——建卡推迟到首个 draft（拿到 reply 锚）.

        无锚建卡直发 chat 会落在主聊顶层，话题回合的卡必须 reply 到话题内的
        消息才能进线程（2026-10-09 实测：探针期无锚建卡，卡落主聊、话题空壳）。
        """
        if session.state != "creating" or session.card_create_task is not None:
            return
        if session.thread_id and session.reply_to is None:
            return  # 话题会话等锚
        assert self._loop is not None
        if session.flush is None:
            session.flush = FlushController(loop=self._loop)
        session.card_create_task = self._loop.create_task(self._do_create_card(session))

    def _seal_session(self, session: ChatSession, notice: str | None = None) -> None:
        """旧会话收尾（followup 边界）：按已积累内容渲染完成卡，失败仅记日志."""
        assert self._loop is not None

        async def _seal() -> None:
            if session.card_create_task is not None:
                await session.card_create_task
            if session.state == "failed" or session.card_id is None:
                return
            session.completed_at = time.time()
            if notice:
                session.segment_state.add_notice(notice)
            session.segment_state.finalize_segments(
                len(session.tool_tracker.build_display_steps()))
            session.state = "completed"
            card = build_complete_card(
                segments=session.segment_state.segments,
                all_tool_steps=session.tool_tracker.build_display_steps(),
                footer_data=self._footer_data(session, None, None, ""),
                footer_fields=self._footer_fields,
                footer_show_label=self._footer_show_label,
                footer_enabled=self._footer_enabled,
                # 被打断的卡红色收尾（用户决策）：redirect 旧卡一眼可辨，
                # NOTICE 文案说明结果在新卡。注入模式原为绿色，此处按需变更。
                # 红标只活在 header 里——异常态强制显示 header，不受配置默认关闭影响
                is_aborted=bool(notice),
                header_enabled=self._header_enabled or bool(notice),
                body_text_size=self._body_text_size,
                show_tool_use=self._show_tool_use,
                width_mode=self._width_mode,
                tool_panel_expanded=self._tool_panel_expanded,
                reasoning_panel_expanded=self._reasoning_panel_expanded,
            )
            try:
                session.sequence += 1
                await self._client.cardkit_close_streaming(session.card_id, sequence=session.sequence)
                session.sequence += 1
                await self._client.cardkit_update(session.card_id, card, sequence=session.sequence)
            except Exception as e:
                logger.warning("seal old card failed: %s", e)

        self._loop.create_task(_seal())

    def _any_active_session(self) -> ChatSession | None:
        active = [s for s in self._sessions.values() if not s.is_terminal]
        return active[0] if active else None

    def _resolve_reasoning_target(self, chat_id: str) -> ChatSession | None:
        # 钩子不带 chat：显式 chat 命中优先；否则仅单会话时兜底（多会话并发丢弃防串扰）。
        # 兜底须含 creating（typing 建卡窗口内的 reasoning 也不能丢——长思考回合
        # 的 reasoning 若在建卡期被丢，卡片全程只有 loading）
        session = self.active_session(chat_id)
        if session is not None:
            return session
        active = [s for s in self._sessions.values() if not s.is_terminal]
        return active[0] if len(active) == 1 else None

    def _schedule(self, session: ChatSession) -> None:
        if session.state == "creating" or session.flush is None or self._loop is None:
            return  # 建卡完成后的首次 flush 会带上全部已暂存内容
        callback = lambda s=session: self._do_flush(s)  # noqa: E731
        if threading.current_thread() is self._thread:
            session.flush.schedule_update(callback)
        else:
            self._loop.call_soon_threadsafe(session.flush.schedule_update, callback)

    async def _do_create_card(self, session: ChatSession) -> None:
        card = build_streaming_card_v2(
            show_tool_use=False,
            show_reasoning=False,
            show_streaming_element=False,
            header_enabled=self._header_enabled,
            text_size=self._body_text_size,
            heartbeat_enabled=True,
            width_mode=self._width_mode,
        )
        try:
            card_id = await self._client.cardkit_create(card)
            if session.reply_to:
                # 有锚：卡片落在用户消息下方（话题内即同一线程）
                card_msg_id = await self._client.reply_card_by_id(session.reply_to, card_id)
            else:
                # 无锚：直发 chat（chat_id 不是合法 reply 目标，reply 会 230001）
                card_msg_id = await self._client.send_card_to_chat(
                    session.chat_id, {"type": "card", "data": {"card_id": card_id}})
        except Exception as e:
            logger.warning("plugin card create failed: chat=%s err=%s", session.chat_id, e)
            session.state = "failed"
            return
        session.card_id = card_id
        session.card_msg_id = card_msg_id
        if session.state == "creating":
            # 建 card 期间会话可能已被 redirect 边界同步置终态（旧卡 seal 流程）：
            # 不得复活为 streaming，否则全局 reasoning 路由把新回合思考当串扰丢掉
            session.state = "streaming"
        if session.flush is not None:
            session.flush.set_card_message_ready(True)
        self._schedule(session)

    async def _do_flush(self, session: ChatSession) -> None:
        """幂等 flush：新 segment 建元素 + 脏文本流式 + 工具面板/心跳更新."""
        if session.is_terminal or not session.card_id:
            return
        segments = session.segment_state.segments
        all_steps = session.tool_tracker.build_display_steps()
        actions: list[dict[str, Any]] = []
        for seg in segments:
            if not seg.created:
                actions.append(build_add_segment_action(
                    seg, all_steps, text_size=self._body_text_size))
                seg.created = True
            elif seg.type == SegmentType.TOOL and seg.dirty:
                # 按段自己的 [offset, end) 切片更新：传全量会让第二个工具面板
                # 把前面段的步骤重复渲染进去（单面板时代遗留），也与完成卡口径不一
                seg_steps = all_steps[seg.tool_offset:tool_segment_end(seg, all_steps)]
                actions.append(build_tool_update_action(
                    element_id=seg.el_id, steps=seg_steps, step_offset=seg.tool_offset))
                seg.dirty = False
            elif (seg.type == SegmentType.REASONING and seg.elapsed_ms > 0
                  and not seg.reasoning_finalized):
                actions.append(build_reasoning_finalized_action(seg))
                seg.reasoning_finalized = True
        if actions:
            session.sequence += 1
            try:
                await self._client.cardkit_batch_update(
                    session.card_id, actions, sequence=session.sequence)
            except Exception as e:
                logger.warning("plugin batch update failed: chat=%s err=%s",
                                session.chat_id, e)
                return
        # 脏文本流式（answer / reasoning）
        for seg in segments:
            if not seg.created or not seg.dirty:
                continue
            # 思考面板摘录口径与完成重渲一致（cap_reasoning_text），防长思考
            # 流式全文把卡片体积推过飞书上限；回答正文是交付物，不截断
            raw = (cap_reasoning_text(seg.text)
                   if seg.type == SegmentType.REASONING else seg.text)
            content = optimize_markdown_style(raw) or " "
            session.sequence += 1
            try:
                await self._client.cardkit_stream_element(
                    session.card_id, seg.text_el_id or seg.el_id, content,
                    sequence=session.sequence)
                seg.dirty = False
            except Exception as e:
                logger.debug("plugin stream element failed: el=%s err=%s", seg.el_id, e)
        # 心跳状态行
        if session.heartbeat_text:
            session.sequence += 1
            try:
                await self._client.cardkit_stream_element(
                    session.card_id, HEARTBEAT_ELEMENT_ID,
                    optimize_markdown_style(session.heartbeat_text) or " ",
                    sequence=session.sequence)
                session.heartbeat_text = ""
            except Exception as e:
                logger.debug("plugin heartbeat update failed: %s", e)

    def _footer_data(self, session: ChatSession, duration: float | None,
                     usage: dict[str, Any] | None, model: str) -> dict[str, Any] | None:
        data: dict[str, Any] = {
            "duration": duration if duration is not None else time.time() - session.created_at,
        }
        if usage:
            data["input_tokens"] = usage.get("input", 0)
            data["output_tokens"] = usage.get("output", 0)
            if model:
                data["model"] = model
            elif usage.get("model"):
                data["model"] = usage["model"]
            # t/s = 输出 tokens / 回合时长（注入模式同款口径）
            if data["output_tokens"] and data["duration"] > 0:
                data["tps"] = data["output_tokens"] / data["duration"]
            if usage.get("context_max"):
                data["context_max"] = usage["context_max"]
                data["context_used"] = usage.get("context_used", usage.get("input", 0))
        elif model:
            data["model"] = model
        return data

    def _model_switch_data(self, current: str) -> dict[str, str] | None:
        """footer 🧠⇄ 按钮数据：{"current"}；轮换清单 <2 或当前模型未知 → 不出按钮."""
        current = (current or "").strip()
        if len(self._model_cycle) < 2 or not current:
            return None
        return {"current": current}

    def model_picker_data(self, chat_id: str) -> dict[str, Any] | None:
        """🧠⇄ 点击后的选择卡数据：{"current", "models"}；无清单 → None."""
        if len(self._model_cycle) < 2:
            return None
        current = self._chat_models.get(chat_id) or self._chat_models.get("") \
            or self._last_model or self._model_cycle[0]
        return {"current": current, "models": list(self._model_cycle)}

    # ── 常用模型管理（v0.19.0：picker 卡 ⚙ 入口 → 管理卡 toggle，存盘跨重启）──

    _CANDIDATES_TTL = 60.0

    def _model_cycle_file(self) -> Path | None:
        """常用清单存储：<HERMES_HOME>/feishu_streaming_model_cycle.json.

        与 config.yaml 同目录（profile 作用域天然隔离；部署拷贝 plugins/ 不会
        覆盖它）。"""
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home() / "feishu_streaming_model_cycle.json"
        except Exception:
            return None

    def _load_model_favorites(self) -> list[str]:
        path = self._model_cycle_file()
        if path is None or not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            models = raw.get("models") if isinstance(raw, dict) else raw
            out: list[str] = []
            for m in models or []:
                name = str(m).strip()
                if name and name not in out:
                    out.append(name)
            return out[:MAX_FAVORITE_MODELS]
        except Exception as e:
            logger.warning("model favorites load failed: %s", e)
            return []

    def _apply_model_favorites(self, models: list[str]) -> list[str]:
        """清单落盘 + 应用到本 engine（footer 按钮与 picker 即时生效）；空清单回退推导."""
        out: list[str] = []
        for m in models:
            name = str(m).strip()
            if name and name not in out:
                out.append(name)
        out = out[:MAX_FAVORITE_MODELS]
        self._model_cycle = out or list(self._model_cycle_fallback)
        path = self._model_cycle_file()
        if path is not None:
            try:
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps({"models": out}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                tmp.replace(path)
            except Exception as e:
                logger.warning("model favorites save failed: %s", e)
        return out

    def toggle_model_favorite(self, target: str) -> tuple[list[str], str]:
        """加入/移出一个常用模型 → (新清单, notice)。上限已满拒绝加新。"""
        target = (target or "").strip()
        if not target:
            return self._load_model_favorites(), ""
        favorites = self._load_model_favorites()
        if target in favorites:
            favorites.remove(target)
        elif len(favorites) >= MAX_FAVORITE_MODELS:
            return favorites, f"常用已满 {MAX_FAVORITE_MODELS} 个，先移出一个再加「{target}」"
        else:
            favorites.append(target)
        return self._apply_model_favorites(favorites), ""

    async def model_candidates(self) -> list[dict[str, Any]]:
        """hermes 全部可用模型（provider 分组）——管理卡候选池.

        数据源 = 上游 /model picker 同款 list_picker_providers（内置 provider 目录
        + 自定义端点，non_blocking 只读磁盘缓存）。同步 IO → to_thread；60s 缓存。
        任何失败回 []（管理卡渲染空态提示，不影响其他功能）。"""
        now = time.time()
        if self._candidates_cache and now - self._candidates_cache[0] < self._CANDIDATES_TTL:
            return self._candidates_cache[1]

        def _fetch() -> list[dict[str, Any]]:
            import yaml

            from hermes_constants import get_hermes_home

            cfg = yaml.safe_load((get_hermes_home() / "config.yaml").read_text(encoding="utf-8")) or {}
            model_cfg = cfg.get("model") or {}
            try:
                from hermes_cli.config import get_compatible_custom_providers

                custom = get_compatible_custom_providers(cfg)
            except Exception:
                custom = cfg.get("custom_providers")
            excluded = (cfg.get("model_catalog") or {}).get("excluded_providers")
            from hermes_cli.model_switch_providers import list_picker_providers

            providers: list[dict[str, Any]] = list(list_picker_providers(
                current_provider=str(model_cfg.get("provider") or "openrouter"),
                current_base_url=str(model_cfg.get("base_url") or ""),
                current_model=self._last_model or str(model_cfg.get("default") or ""),
                user_providers=cfg.get("providers"),
                custom_providers=custom,
                excluded_providers=excluded if isinstance(excluded, list) else [],
                non_blocking_catalogs=True, probe_custom_providers=False,
                probe_current_custom_provider=False, max_models=50, include_moa=False))
            return providers

        try:
            providers = await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning("model candidates fetch failed: %s", e)
            return []
        slim: list[dict[str, Any]] = [
            {"slug": p.get("slug"), "name": p.get("name"),
             "models": [str(m) for m in (p.get("models") or []) if str(m).strip()],
             "total_models": int(p.get("total_models") or 0)}
            for p in providers or [] if p.get("models")]
        for item in slim:
            item["total_models"] = item["total_models"] or len(item["models"])
        self._candidates_cache = (now, slim)
        logger.info("model candidates loaded: providers=%d models=%d",
                     len(slim), sum(len(p["models"]) for p in slim))
        return slim
