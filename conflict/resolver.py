"""
Conflict Resolver — Hermes Agent's central conflict仲裁器.

依據 SOUL > RULES > CODEX > AGENTS > USER > MEMORY 優先級，
對跨模組衝突事件進行自動仲裁。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, Protocol


class Priority(IntEnum):
    """衝突優先級（數字越大優先級越高）"""
    MEMORY = 1
    USER = 2
    AGENTS = 3
    CODEX = 4
    RULES = 5
    SOUL = 6


@dataclass
class ConflictEvent:
    """衝突事件標準格式"""
    source_module: str          # 衝突來源模組（如 "delegate_tool", "budget_config"）
    conflict_type: str          # 衝突類型（如 "credential_override", "priority_conflict"）
    options: dict[str, Any]     # 可用選項（key 為選項名，value 為選項內容）
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> Priority:
        """根據 source_module 自動判定預設優先級"""
        mapping = {
            "SOUL": Priority.SOUL,
            "RULES": Priority.RULES,
            "CODEX": Priority.CODEX,
            "AGENTS": Priority.AGENTS,
            "USER": Priority.USER,
            "MEMORY": Priority.MEMORY,
        }
        return mapping.get(self.source_module, Priority.AGENTS)


@dataclass
class Resolution:
    """仲裁結果"""
    winner: str                 # 勝出選項的 key
    winner_value: Any           # 勝出選項的 value
    policy_used: str            # 使用的策略名稱
    reasoning: str             # 仲裁理由


class ResolutionPolicy(Protocol):
    """策略協議——所有仲裁策略必須實現此接口"""

    @property
    def name(self) -> str:
        """策略名稱"""
        ...

    def resolve(self, event: ConflictEvent) -> Resolution:
        """對衝突事件進行仲裁"""
        ...


class PriorityOverridePolicy:
    """優先級覆蓋策略（最高優先級：SOUL > RULES > CODEX > AGENTS > USER > MEMORY）"""

    @property
    def name(self) -> str:
        return "priority_override"

    def resolve(self, event: ConflictEvent) -> Resolution:
        """
        根據衝突來源的優先級自動選擇。

        典型場景（delegate_tool.py）：
        - config.yaml > env > default（credential override）
        - pinned > tool_overrides > registry > default（budget priority）
        """
        source_priority = event.priority

        # 按優先級排序候選項
        sorted_options = sorted(
            event.options.items(),
            key=lambda kv: self._get_option_priority(kv[0], kv[1]),
            reverse=True
        )

        winner_key, winner_val = sorted_options[0]
        return Resolution(
            winner=winner_key,
            winner_value=winner_val,
            policy_used=self.name,
            reasoning=f"Priority override: {source_priority.name} source wins"
        )

    def _get_option_priority(self, key: str, value: Any) -> int:
        """根據選項 key 判定優先級權重"""
        # 高優先級標記（pinned, forced, explicit）
        if key.startswith("pinned") or key.startswith("forced") or key.startswith("explicit"):
            return 100
        # 配置覆蓋（config_override, tool_override）
        if "override" in key.lower():
            return 50
        # 預設值（default）
        if "default" in key.lower():
            return 0
        return 25


class LastWriteWinsPolicy:
    """最後寫入勝出策略（FIFO）"""

    @property
    def name(self) -> str:
        return "last_write_wins"

    def resolve(self, event: ConflictEvent) -> Resolution:
        """按時間戳或出現順序，選擇最後一個選項"""
        # 從 metadata 中提取 timestamp 或 sequence
        seq = event.metadata.get("sequence", 0)
        options = list(event.options.items())

        # 簡化：選擇最後一個
        winner_key, winner_val = options[-1]
        return Resolution(
            winner=winner_key,
            winner_value=winner_val,
            policy_used=self.name,
            reasoning=f"Last-write-wins: sequence {seq}"
        )


class ConflictResolver:
    """
    統一仲裁器——根據事件類型自動分發到對應策略。

    使用示例（delegate_tool.py 重構後）：
        from conflict import ConflictResolver, ConflictEvent

        resolver = ConflictResolver()
        event = ConflictEvent(
            source_module="AGENTS",  # 來自 AGENTS 層
            conflict_type="credential_override",
            options={
                "pinned": creds["provider"],
                "config_override": override_provider,
                "default": parent_agent.provider,
            }
        )
        resolution = resolver.resolve(event)
        effective_provider = resolution.winner_value
    """

    def __init__(self):
        self._policies: dict[str, ResolutionPolicy] = {
            "priority_override": PriorityOverridePolicy(),
            "last_write_wins": LastWriteWinsPolicy(),
        }
        self._default_policy = "priority_override"

    def register_policy(self, name: str, policy: ResolutionPolicy) -> None:
        """註冊自定義策略"""
        self._policies[name] = policy

    def resolve(self, event: ConflictEvent, policy_name: Optional[str] = None) -> Resolution:
        """執行仲裁"""
        policy = self._policies.get(policy_name or self._default_policy)
        if not policy:
            raise ValueError(f"Unknown policy: {policy_name}")
        return policy.resolve(event)