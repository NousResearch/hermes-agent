"""Prompt Cache Monitor — 自动检测缓存健康状态

集成到Hermes框架:
- 每次API调用后记录缓存命中率
- 自动检测缓存失效(cache break)
- 提供 /cache 命令的诊断数据
- 基于 prompt-cache-skills 的16条陷阱做自动检查
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheRecord:
    """单次API调用的缓存记录"""
    timestamp: str
    model: str
    provider: str
    prompt_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_hit_rate: float = 0.0
    latency: float = 0.0
    api_call_index: int = 0


@dataclass
class CacheBreak:
    """缓存失效事件"""
    timestamp: str
    previous_hit_tokens: int
    current_hit_tokens: int
    drop: int
    suspected_cause: str


@dataclass
class SessionCacheStats:
    """会话级缓存统计"""
    session_id: str = ""
    start_time: str = ""
    total_api_calls: int = 0
    total_prompt_tokens: int = 0
    total_cache_read: int = 0
    total_cache_write: int = 0
    overall_hit_rate: float = 0.0
    breaks: list[CacheBreak] = field(default_factory=list)
    records: list[CacheRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CacheMonitor:
    """缓存监控器

    用法:
        monitor = CacheMonitor()
        monitor.on_api_call(model, provider, prompt_tokens, cache_read, cache_write, latency)
        stats = monitor.get_stats()
    """

    MIN_TOKEN_DROP = 2000  # 缓存命中token下降超过此值视为cache break
    HIT_RATE_WARN_THRESHOLD = 0.5  # 命中率低于50%发出警告

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.expanduser("~/.hermes/data/cache-monitor")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._records: list[CacheRecord] = []
        self._breaks: list[CacheBreak] = []
        self._warnings: list[str] = []
        self._session_id: str = ""
        self._start_time = datetime.now().isoformat()
        self._prev_cache_read = 0

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._start_time = datetime.now().isoformat()

    def on_api_call(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        latency: float = 0.0,
        api_call_index: int = 0,
    ) -> Optional[CacheBreak]:
        """记录一次API调用的缓存数据，返回CacheBreak(如果有)"""
        hit_rate = cache_read_tokens / prompt_tokens if prompt_tokens > 0 else 0.0

        record = CacheRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_hit_rate=round(hit_rate, 4),
            latency=latency,
            api_call_index=api_call_index,
        )
        self._records.append(record)

        # Detect cache break
        break_event = None
        if self._prev_cache_read > 0 and cache_read_tokens < self._prev_cache_read:
            drop = self._prev_cache_read - cache_read_tokens
            if drop >= self.MIN_TOKEN_DROP:
                cause = self._diagnose_break(record)
                break_event = CacheBreak(
                    timestamp=record.timestamp,
                    previous_hit_tokens=self._prev_cache_read,
                    current_hit_tokens=cache_read_tokens,
                    drop=drop,
                    suspected_cause=cause,
                )
                self._breaks.append(break_event)
                self._warnings.append(
                    f"⚠️ 缓存失效: {drop} tokens丢失 "
                    f"({self._prev_cache_read}→{cache_read_tokens}) "
                    f"原因: {cause}"
                )

        self._prev_cache_read = cache_read_tokens

        # Check low hit rate on warm sessions (>3 calls)
        if len(self._records) > 3 and hit_rate < self.HIT_RATE_WARN_THRESHOLD:
            avg_hit = sum(r.cache_hit_rate for r in self._records[1:]) / max(1, len(self._records) - 1)
            if avg_hit < self.HIT_RATE_WARN_THRESHOLD:
                self._warnings.append(
                    f"⚠️ 缓存命中率偏低: {avg_hit:.1%} (最近{len(self._records)}次调用)"
                )

        return break_event

    def _diagnose_break(self, record: CacheRecord) -> str:
        """诊断缓存失效原因"""
        # Simple heuristics based on prompt-cache-skills gotchas
        if record.prompt_tokens < 1024:
            return "前缀不足1024 tokens(OpenAI最低要求)"
        if self._prev_cache_read > 0 and record.cache_read_tokens == 0:
            return "缓存完全失效 — 可能system prompt变化或工具定义重排"
        return "部分缓存失效 — 可能历史消息变化或provider TTL过期"

    def get_stats(self) -> SessionCacheStats:
        """获取会话级统计"""
        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_read = sum(r.cache_read_tokens for r in self._records)
        total_write = sum(r.cache_write_tokens for r in self._records)
        overall = total_read / total_prompt if total_prompt > 0 else 0

        return SessionCacheStats(
            session_id=self._session_id,
            start_time=self._start_time,
            total_api_calls=len(self._records),
            total_prompt_tokens=total_prompt,
            total_cache_read=total_read,
            total_cache_write=total_write,
            overall_hit_rate=round(overall, 4),
            breaks=list(self._breaks),
            records=list(self._records[-20:]),  # Last 20
            warnings=list(self._warnings),
        )

    def format_report(self) -> str:
        """格式化诊断报告"""
        stats = self.get_stats()
        lines = []
        lines.append("=" * 50)
        lines.append("📊 Prompt Cache 诊断报告")
        lines.append("=" * 50)

        if stats.total_api_calls == 0:
            lines.append("\n  暂无API调用记录")
            return "\n".join(lines)

        lines.append(f"\n  会话ID:     {stats.session_id[:16]}...")
        lines.append(f"  开始时间:   {stats.start_time[:19]}")
        lines.append(f"  API调用:    {stats.total_api_calls} 次")
        lines.append(f"  总输入:     {stats.total_prompt_tokens:,} tokens")
        lines.append(f"  缓存读取:   {stats.total_cache_read:,} tokens")
        lines.append(f"  缓存写入:   {stats.total_cache_write:,} tokens")

        # Hit rate with visual bar
        rate = stats.overall_hit_rate
        bar_len = int(rate * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"  整体命中率: {bar} {rate:.1%}")

        # Per-call trend (last 10)
        recent = stats.records[-10:]
        if len(recent) > 1:
            lines.append(f"\n  ── 最近{len(recent)}次调用趋势 ──")
            for r in recent:
                pct = r.cache_hit_rate
                indicator = "🟢" if pct > 0.7 else ("🟡" if pct > 0.3 else "🔴")
                lines.append(
                    f"  {indicator} #{r.api_call_index:>3} "
                    f"命中={pct:>5.1%} "
                    f"({r.cache_read_tokens:>6}/{r.prompt_tokens:>6}) "
                    f"延迟={r.latency:.1f}s"
                )

        # Cache breaks
        if stats.breaks:
            lines.append(f"\n  ── 缓存失效事件 ({len(stats.breaks)}次) ──")
            for b in stats.breaks[-5:]:
                lines.append(
                    f"  🔴 {b.timestamp[:19]} "
                    f"丢失{b.drop} tokens "
                    f"({b.previous_hit_tokens}→{b.current_hit_tokens})"
                )
                lines.append(f"     原因: {b.suspected_cause}")

        # Warnings (deduplicated)
        unique_warnings = list(dict.fromkeys(stats.warnings))
        if unique_warnings:
            lines.append(f"\n  ── 警告 ──")
            for w in unique_warnings[-5:]:
                lines.append(f"  {w}")

        # Optimization tips
        if rate < 0.5 and stats.total_api_calls > 3:
            lines.append(f"\n  ── 优化建议 ──")
            lines.append("  1. 检查system prompt是否含动态内容(时间戳/session ID)")
            lines.append("  2. 确保工具定义JSON排序一致")
            lines.append("  3. 检查prompt_cache_key是否使用稳定hash")
            lines.append("  4. 参考 llm-prompt-cache-optimization skill 的16条陷阱")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def save_session(self) -> str:
        """保存会话缓存数据到JSON"""
        stats = self.get_stats()
        filename = f"cache_{self._session_id[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self._data_dir / filename
        filepath.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(filepath)
