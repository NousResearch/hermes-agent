"""
Checkpoint System — 状态持久化 + 可恢复性

设计参考：langgraph checkpoint 协议
- Checkpoint = channel_values（状态快照）+ versions_seen（版本追踪）
- PendingWrite = 中断时未完成的写入，恢复时重放
- CheckpointConfig = thread_id + checkpoint_id 定位
- 每个 channel 有单调递增的版本号，支持乐观并发控制

用法：
    store = MemoryCheckpointStore()
    store.put({"thread_id": "s1"}, checkpoint, pending_writes)
    loaded = store.get({"thread_id": "s1"})
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Checkpoint data structures
# ---------------------------------------------------------------------------


@dataclass
class ChannelVersions:
    """单个 channel 的版本号（单调递增）。"""
    version: int = 0


@dataclass
class Checkpoint:
    """
    状态快照。

    包含：
    - channel_values: 所有 channel 的当前值（dict）
    - versions_seen: 每个 channel 上次看到的版本号（用于冲突检测）
    - metadata: checkpoint 元信息（task_id, timestamp 等）
    """
    channel_values: dict[str, Any] = field(default_factory=dict)
    versions_seen: dict[str, int] = field(default_factory=dict)  # channel -> last seen version
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_values": self.channel_values,
            "versions_seen": self.versions_seen,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        return cls(
            channel_values=d.get("channel_values", {}),
            versions_seen=d.get("versions_seen", {}),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class PendingWrite:
    """
    中断时未完成的写入。

    恢复时重放这些写入，确保不丢数据。
    """
    task_id: str
    channel: str
    value: Any
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "channel": self.channel,
            "value": self.value,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# CheckpointConfig — 地址定位
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """通过 thread_id + checkpoint_id 定位一个 checkpoint。"""
    thread_id: str
    checkpoint_id: str | None = None   # None = latest
    task_id: str | None = None

    @classmethod
    def new(cls, thread_id: str | None = None, **kw) -> CheckpointConfig:
        return cls(
            thread_id=thread_id or str(uuid.uuid4()),
            **kw,
        )


# ---------------------------------------------------------------------------
# CheckpointStore — 抽象存储层
# ---------------------------------------------------------------------------


class CheckpointStore(ABC):
    """
    Checkpoint 持久化存储接口。

    实现这个接口可以有不同的存储后端：
    - MemoryCheckpointStore（内存，最快）
    - FileCheckpointStore（磁盘）
    - PostgresCheckpointStore（分布式）
    """

    @abstractmethod
    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        pending_writes: list[PendingWrite] | None = None,
    ) -> str:
        """
        保存 checkpoint。

        Returns: checkpoint_id (可以用于 load by checkpoint_id)
        """
        ...

    @abstractmethod
    def get(
        self,
        config: CheckpointConfig,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        """加载 checkpoint。"""
        ...

    @abstractmethod
    def list(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """列出某 thread_id 的所有 checkpoint（按时间倒序）。"""
        ...

    @abstractmethod
    def get_pending_writes(
        self,
        config: CheckpointConfig,
        checkpoint_id: str | None = None,
    ) -> list[PendingWrite]:
        """获取某 checkpoint 的未完成写入。"""
        ...


# ---------------------------------------------------------------------------
# Memory CheckpointStore — 进程内存储
# ---------------------------------------------------------------------------


class MemoryCheckpointStore(CheckpointStore):
    """
    内存版 CheckpointStore。

    数据结构：
        _checkpoints[thread_id][checkpoint_id] = (checkpoint, pending_writes)

    警告：进程重启后数据丢失。仅用于开发/测试或单进程场景。
    """

    __slots__ = ("_checkpoints", "_lock")

    def __init__(self):
        self._checkpoints: dict[str, dict[str, tuple[Checkpoint, list[PendingWrite]]]] = {}
        self._lock = asyncio.Lock()

    async def put_async(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        pending_writes: list[PendingWrite] | None = None,
    ) -> str:
        """Async version — uses asyncio.Lock."""
        checkpoint_id = config.checkpoint_id or str(uuid.uuid4())[:8]
        async with self._lock:
            if config.thread_id not in self._checkpoints:
                self._checkpoints[config.thread_id] = {}
            self._checkpoints[config.thread_id][checkpoint_id] = (
                checkpoint,
                list(pending_writes) if pending_writes else [],
            )
        return checkpoint_id

    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        pending_writes: list[PendingWrite] | None = None,
    ) -> str:
        """Sync version — for contexts without async."""
        checkpoint_id = config.checkpoint_id or str(uuid.uuid4())[:8]
        if config.thread_id not in self._checkpoints:
            self._checkpoints[config.thread_id] = {}
        self._checkpoints[config.thread_id][checkpoint_id] = (
            checkpoint,
            list(pending_writes) if pending_writes else [],
        )
        return checkpoint_id

    def get(
        self,
        config: CheckpointConfig,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        thread = self._checkpoints.get(config.thread_id, {})
        if not thread:
            return None
        if checkpoint_id:
            entry = thread.get(checkpoint_id)
        else:
            # latest = max by created_at
            if not thread:
                return None
            entry = max(thread.values(), key=lambda x: x[0].created_at)
        return entry[0] if entry else None

    def list(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        thread = self._checkpoints.get(thread_id, {})
        sorted_checkpoints = sorted(
            thread.items(),
            key=lambda x: x[1][0].created_at,
            reverse=True,
        )
        return [
            {
                "checkpoint_id": cid,
                "created_at": entry[0].created_at,
                "metadata": entry[0].metadata,
            }
            for cid, entry in sorted_checkpoints[:limit]
        ]

    def get_pending_writes(
        self,
        config: CheckpointConfig,
        checkpoint_id: str | None = None,
    ) -> list[PendingWrite]:
        thread = self._checkpoints.get(config.thread_id, {})
        if not thread:
            return []
        if checkpoint_id:
            entry = thread.get(checkpoint_id)
        else:
            if not thread:
                return []
            entry = max(thread.values(), key=lambda x: x[0].created_at)
        return entry[1] if entry else []


# ---------------------------------------------------------------------------
# File-based CheckpointStore — 磁盘持久化
# ---------------------------------------------------------------------------


class FileCheckpointStore(CheckpointStore):
    """
    基于磁盘文件的 CheckpointStore。

    路径：<base_dir>/<thread_id>/<checkpoint_id>.json
    """

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir or Path.home() / ".hermes" / "checkpoints")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _thread_dir(self, thread_id: str) -> Path:
        return self._base_dir / thread_id

    def _checkpoint_path(self, thread_id: str, checkpoint_id: str) -> Path:
        return self._thread_dir(thread_id) / f"{checkpoint_id}.json"

    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        pending_writes: list[PendingWrite] | None = None,
    ) -> str:
        import json

        checkpoint_id = config.checkpoint_id or str(uuid.uuid4())[:8]
        thread_dir = self._thread_dir(config.thread_id)
        thread_dir.mkdir(parents=True, exist_ok=True)

        path = self._checkpoint_path(config.thread_id, checkpoint_id)
        data = {
            "checkpoint": checkpoint.to_dict(),
            "pending_writes": [w.to_dict() for w in (pending_writes or [])],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        return checkpoint_id

    def get(
        self,
        config: CheckpointConfig,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        import json

        thread_dir = self._thread_dir(config.thread_id)
        if not thread_dir.exists():
            return None

        if checkpoint_id:
            path = self._checkpoint_path(config.thread_id, checkpoint_id)
            if not path.exists():
                return None
            data = json.loads(path.read_text())
            return Checkpoint.from_dict(data["checkpoint"])

        # latest
        checkpoints = sorted(
            thread_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not checkpoints:
            return None
        data = json.loads(checkpoints[0].read_text())
        return Checkpoint.from_dict(data["checkpoint"])

    def list(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        thread_dir = self._thread_dir(thread_id)
        if not thread_dir.exists():
            return []
        checkpoints = sorted(
            thread_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        result = []
        for p in checkpoints[:limit]:
            import json
            data = json.loads(p.read_text())
            result.append({
                "checkpoint_id": p.stem,
                "created_at": data["checkpoint"].get("created_at", p.stat().st_mtime),
                "metadata": data["checkpoint"].get("metadata", {}),
            })
        return result

    def get_pending_writes(
        self,
        config: CheckpointConfig,
        checkpoint_id: str | None = None,
    ) -> list[PendingWrite]:
        import json

        thread_dir = self._thread_dir(config.thread_id)
        if not thread_dir.exists():
            return []

        if checkpoint_id:
            path = self._checkpoint_path(config.thread_id, checkpoint_id)
            if not path.exists():
                return []
            data = json.loads(path.read_text())
        else:
            checkpoints = sorted(
                thread_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not checkpoints:
                return []
            data = json.loads(checkpoints[0].read_text())

        return [
            PendingWrite(**pw) for pw in data.get("pending_writes", [])
        ]
