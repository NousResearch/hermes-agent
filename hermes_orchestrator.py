"""
Boulder + Ultragoal — 多 Agent 任务持久化 (OMO 11.2 借鉴)

借鉴自 Sisyphus Labs OMO 工具的 ultragoal + start-work-continuation 机制。
解决的问题：delegate_task 跑完后中间结果就丢失，长任务无法断点续传。

核心设计：
  - **boulder.json** = 任务级进度（步骤、断点、状态机）
  - **ultragoal/{task_id}/** = 目标级证据（goal + 嵌入的成功标准 + 证据链）
  - **evidence/** = 每步的"为什么这么做"（来源、引用、决策依据）
  - **audit.md** = 可读审计报告（给人看的总结）

文件布局（~/.hermes/orchestrator/）：
  boulder/{task_id}.json           # 任务断点 + 步骤进度
  ultragoal/{task_id}/goal.json    # 目标 + 成功标准
  ultragoal/{task_id}/evidence/    # 证据链（每步独立 JSON）
  ultragoal/{task_id}/audit.md     # 可读审计报告

任务 ID 格式：`{YYYYMMDDHHMMSS}-{goal-slug}-{6-char-hash}`
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from hermes_constants import get_hermes_home
except ImportError:
    # standalone use
    def get_hermes_home() -> Path:
        import os
        env = os.environ.get("HERMES_HOME", "").strip()
        if env:
            return Path(env)
        return Path.home() / ".hermes"


# ---------------------------------------------------------------------------
# 路径管理
# ---------------------------------------------------------------------------

def orchestrator_root() -> Path:
    """返回 ~/.hermes/orchestrator/，自动创建。"""
    root = get_hermes_home() / "orchestrator"
    (root / "boulder").mkdir(parents=True, exist_ok=True)
    (root / "ultragoal").mkdir(parents=True, exist_ok=True)
    return root


def boulder_path(task_id: str) -> Path:
    """boulder 文件路径"""
    return orchestrator_root() / "boulder" / f"{task_id}.json"


def ultragoal_dir(task_id: str) -> Path:
    """ultragoal 目录路径"""
    d = orchestrator_root() / "ultragoal" / task_id
    (d / "evidence").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Task ID 生成
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+", re.IGNORECASE)


def make_task_id(goal: str, *, length: int = 6) -> str:
    """生成 task_id：`{YYYYMMDDHHMMSS}-{slug}-{hash}`。

    Examples
    --------
    >>> make_task_id("扫目录找 contract")
    '20260603143052-scan-contract-a1b2c3'
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    # 转小写、保留中英文数字、合并空白为单连字符
    slug = _SLUG_RE.sub("-", goal.lower()).strip("-")[:40]
    if not slug:
        slug = "task"
    # hash 用于防冲突（多个相同 goal 任务也能区分）
    h = hashlib.sha1(f"{ts}-{goal}-{time.time_ns()}".encode()).hexdigest()[:length]
    return f"{ts}-{slug}-{h}"


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """boulder 里的一步记录。"""
    step: int
    tool: str                       # 调用的工具名
    args_summary: str = ""          # 参数一句话描述（不存完整 args，避免泄密）
    result_summary: str = ""        # 结果一句话描述
    status: str = "pending"         # pending | running | success | failed | skipped
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: Optional[int] = None
    evidence_ref: Optional[str] = None  # 指向 evidence/{step_id}.json 的相对路径
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Boulder:
    """任务级进度记录（断点续传靠这个）。

    状态机：
      created → running → (paused ↔ running) → completed | failed | cancelled
    """
    task_id: str
    goal: str
    created_at: str
    updated_at: str
    status: str = "created"          # created | running | paused | completed | failed | cancelled
    parent_task_id: Optional[str] = None
    role: str = "leaf"
    model: Optional[str] = None
    toolsets: List[str] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    current_step: int = 0
    total_steps_estimate: Optional[int] = None
    result_summary: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s if isinstance(s, dict) else s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Boulder":
        d = dict(d)
        d["steps"] = [StepRecord(**s) if isinstance(s, dict) else s for s in d.get("steps", [])]
        return cls(**d)


@dataclass
class Goal:
    """ultragoal/{task_id}/goal.json — 目标定义 + 嵌入的成功标准。"""
    task_id: str
    goal: str
    success_criteria: List[str]     # 明确的可验证条件
    anti_criteria: List[str] = field(default_factory=list)  # 明确要避免的失败模式
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Goal":
        return cls(**d)


# ---------------------------------------------------------------------------
# Boulder 读写（线程安全）
# ---------------------------------------------------------------------------

_lock = threading.Lock()


def create_boulder(
    goal: str,
    *,
    role: str = "leaf",
    model: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    parent_task_id: Optional[str] = None,
    total_steps_estimate: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> Boulder:
    """创建一个新 boulder 任务。返回 Boulder 对象（也已落盘）。"""
    now = datetime.now(timezone.utc).isoformat()
    if task_id is None:
        task_id = make_task_id(goal)
    b = Boulder(
        task_id=task_id,
        goal=goal,
        created_at=now,
        updated_at=now,
        role=role,
        model=model,
        toolsets=list(toolsets or []),
        parent_task_id=parent_task_id,
        total_steps_estimate=total_steps_estimate,
        metadata=dict(metadata or {}),
    )
    save_boulder(b)
    return b


def save_boulder(b: Boulder) -> None:
    """原子写 boulder 到磁盘（用 .tmp + rename 避免半写状态）。"""
    b.updated_at = datetime.now(timezone.utc).isoformat()
    path = boulder_path(b.task_id)
    tmp = path.with_suffix(".json.tmp")
    with _lock:
        tmp.write_text(json.dumps(b.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def load_boulder(task_id: str) -> Optional[Boulder]:
    """读取 boulder，不存在返回 None。"""
    path = boulder_path(task_id)
    if not path.exists():
        return None
    with _lock:
        d = json.loads(path.read_text(encoding="utf-8"))
    return Boulder.from_dict(d)


def append_step(
    task_id: str,
    *,
    tool: str,
    args_summary: str = "",
    status: str = "running",
) -> int:
    """添加一个新 step 并返回 step index。状态默认 running（待 result 时回填）。"""
    b = load_boulder(task_id)
    if b is None:
        raise FileNotFoundError(f"boulder not found: {task_id}")
    b.current_step = len(b.steps)
    now = datetime.now(timezone.utc).isoformat()
    step = StepRecord(
        step=b.current_step,
        tool=tool,
        args_summary=args_summary,
        status=status,
        started_at=now,
    )
    b.steps.append(step)
    if b.status == "created":
        b.status = "running"
    save_boulder(b)  # save_boulder 内部会更新 updated_at
    return step.step


def complete_step(
    task_id: str,
    step_index: int,
    *,
    result_summary: str = "",
    status: str = "success",
    error: Optional[str] = None,
    evidence_ref: Optional[str] = None,
) -> None:
    """回填 step 的结果。"""
    b = load_boulder(task_id)
    if b is None or step_index >= len(b.steps):
        return
    step = b.steps[step_index]
    step.result_summary = result_summary
    step.status = status
    step.error = error
    step.evidence_ref = evidence_ref
    step.ended_at = datetime.now(timezone.utc).isoformat()
    if step.started_at:
        try:
            start = datetime.fromisoformat(step.started_at)
            end = datetime.fromisoformat(step.ended_at)
            step.duration_ms = int((end - start).total_seconds() * 1000)
        except Exception:
            pass
    save_boulder(b)


def finish_boulder(
    task_id: str,
    *,
    status: str,
    result_summary: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """收尾：标记 completed / failed / cancelled。"""
    b = load_boulder(task_id)
    if b is None:
        return
    b.status = status
    if result_summary is not None:
        b.result_summary = result_summary
    if error is not None:
        b.error = error
    save_boulder(b)


def list_boulders(status: Optional[str] = None, limit: int = 50) -> List[Boulder]:
    """列出最近的 boulder，可按 status 过滤。"""
    root = orchestrator_root() / "boulder"
    out: List[Boulder] = []
    for p in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(out) >= limit:
            break
        try:
            b = load_boulder(p.stem)
            if b and (status is None or b.status == status):
                out.append(b)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Ultragoal 读写
# ---------------------------------------------------------------------------

def set_goal(
    task_id: str,
    goal: str,
    *,
    success_criteria: List[str],
    anti_criteria: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Goal:
    """设置 ultragoal/{task_id}/goal.json —— 目标 + 成功标准。

    success_criteria 是必须满足的"可验证"条件（每条都应是布尔可判）。
    anti_criteria 是要避免的失败模式（每条都是反向条件）。
    """
    g = Goal(
        task_id=task_id,
        goal=goal,
        success_criteria=list(success_criteria),
        anti_criteria=list(anti_criteria or []),
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata=dict(metadata or {}),
    )
    d = ultragoal_dir(task_id)
    with _lock:
        (d / "goal.json").write_text(
            json.dumps(g.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return g


def load_goal(task_id: str) -> Optional[Goal]:
    p = ultragoal_dir(task_id) / "goal.json"
    if not p.exists():
        return None
    return Goal.from_dict(json.loads(p.read_text(encoding="utf-8")))


def record_evidence(
    task_id: str,
    step: int,
    *,
    source: str,                     # 一手 / 二手 / 推断 / 工具输出 / 用户输入
    content: str,                    # 证据内容（可以是引用、文件内容摘要、决策理由）
    citation: Optional[str] = None,  # 可选：URL、文件路径、章节号
    confidence: str = "high",        # high / medium / low
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """记录一步证据到 ultragoal/{task_id}/evidence/step_{NNN}.json。

    关键设计：**每个 evidence 都有 source 字段**——这是"反目标漂移"的关键。
    一手来源不被同等对待；二手 / 推断来源要在 audit.md 里被高亮。
    """
    d = ultragoal_dir(task_id)
    evidence_file = d / "evidence" / f"step_{step:03d}.json"
    record = {
        "step": step,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "confidence": confidence,
        "content": content,
        "citation": citation,
        "metadata": dict(metadata or {}),
    }
    with _lock:
        evidence_file.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return evidence_file


# ---------------------------------------------------------------------------
# Audit Report（可读输出）
# ---------------------------------------------------------------------------

def write_audit(task_id: str) -> Path:
    """生成可读审计报告 ultragoal/{task_id}/audit.md。

    包含：目标、成功标准、每步证据、反方论据触发点。
    """
    b = load_boulder(task_id)
    g = load_goal(task_id)
    d = ultragoal_dir(task_id)
    evidence_dir = d / "evidence"

    lines = [
        f"# Audit Report — {task_id}",
        "",
        f"**Generated at**: {datetime.now(timezone.utc).isoformat()}",
        f"**Boulder status**: {b.status if b else 'NOT FOUND'}",
        "",
    ]

    if g:
        lines.extend([
            "## 🎯 Goal",
            "",
            f"> {g.goal}",
            "",
            "### ✅ Success Criteria",
            "",
        ])
        for c in g.success_criteria:
            lines.append(f"- [ ] {c}")
        if g.anti_criteria:
            lines.extend(["", "### ❌ Anti-Criteria (必须避免)", ""])
            for c in g.anti_criteria:
                lines.append(f"- [ ] {c}")
        lines.append("")

    if b:
        lines.extend([
            f"## 📊 Progress",
            "",
            f"- 创建时间: {b.created_at}",
            f"- 当前状态: **{b.status}**",
            f"- 已完成步骤: {len([s for s in b.steps if s.status == 'success'])}/{len(b.steps)}",
            f"- Model: `{b.model}`",
            f"- Role: `{b.role}`",
            "",
            "### 步骤清单",
            "",
        ])
        for s in b.steps:
            icon = {
                "success": "✅",
                "failed": "❌",
                "running": "⏳",
                "skipped": "⏭️",
                "pending": "⏸️",
            }.get(s.status, "❓")
            lines.append(
                f"{icon} **Step {s.step}** [{s.tool}] {s.args_summary[:80]}"
            )
            if s.result_summary:
                lines.append(f"   - Result: {s.result_summary[:120]}")
            if s.evidence_ref:
                lines.append(f"   - Evidence: `{s.evidence_ref}`")
            if s.error:
                lines.append(f"   - Error: `{s.error[:120]}`")
            lines.append("")

    if evidence_dir.exists():
        evidence_files = sorted(evidence_dir.glob("step_*.json"))
        if evidence_files:
            lines.extend([
                f"## 📚 Evidence Chain ({len(evidence_files)} items)",
                "",
            ])
            for ef in evidence_files:
                try:
                    rec = json.loads(ef.read_text(encoding="utf-8"))
                except Exception:
                    continue
                source_emoji = {
                    "一手": "🟢", "firsthand": "🟢",
                    "二手": "🟡", "secondhand": "🟡",
                    "推断": "🟠", "inference": "🟠",
                    "工具输出": "🔵", "tool_output": "🔵",
                    "用户输入": "🟣", "user_input": "🟣",
                }.get(rec.get("source", ""), "⚪")
                conf_emoji = {
                    "high": "🟢", "medium": "🟡", "low": "🔴",
                }.get(rec.get("confidence", "high"), "⚪")
                lines.append(
                    f"### {source_emoji} {conf_emoji} Step {rec.get('step', '?')} — {rec.get('source', '?')}"
                )
                if rec.get("citation"):
                    lines.append(f"**Citation**: `{rec['citation']}`")
                lines.append(f"> {rec.get('content', '')[:300]}")
                lines.append("")

    out = d / "audit.md"
    with _lock:
        out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 高层便利 API
# ---------------------------------------------------------------------------

class OrchestratorContext:
    """delegate_task 用的上下文管理器——自动写 boulder + evidence。

    用法（in delegate_task 内部）：

        with OrchestratorContext(goal, model=...) as ctx:
            ctx.append_step(tool="terminal", args_summary="ls -la")
            # ... 执行工具 ...
            ctx.complete_step(result_summary="found 3 files")
            ctx.record_evidence(source="工具输出", content="ls 输出 3 个文件 ...")

    或者用户手动调：
        ctx = OrchestratorContext(goal="...")
        ctx.start()
        ...
        ctx.finish(status="completed", result_summary="...")
    """

    def __init__(
        self,
        goal: str,
        *,
        model: Optional[str] = None,
        role: str = "leaf",
        toolsets: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        anti_criteria: Optional[List[str]] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
    ):
        self.goal = goal
        self.model = model
        self.role = role
        self.toolsets = list(toolsets or [])
        self.success_criteria = list(success_criteria or [])
        self.anti_criteria = list(anti_criteria or [])
        self.task_id = task_id
        self.parent_task_id = parent_task_id
        self._boulder: Optional[Boulder] = None
        self._current_step_index: int = -1

    def start(self) -> str:
        """开始任务，返回 task_id。"""
        self._boulder = create_boulder(
            self.goal,
            role=self.role,
            model=self.model,
            toolsets=self.toolsets,
            parent_task_id=self.parent_task_id,
            task_id=self.task_id,
        )
        self.task_id = self._boulder.task_id
        # 写 goal
        if self.success_criteria:
            set_goal(
                self.task_id,
                self.goal,
                success_criteria=self.success_criteria,
                anti_criteria=self.anti_criteria,
            )
        return self.task_id

    def __enter__(self) -> "OrchestratorContext":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.finish(status="failed", error=f"{exc_type.__name__}: {exc_val}")
        elif self._boulder and self._boulder.status == "running":
            self.finish(status="completed")

    def append_step(
        self,
        *,
        tool: str,
        args_summary: str = "",
        status: str = "running",
    ) -> int:
        """记录一步开始，返回 step index。"""
        if self._boulder is None:
            self.start()
        assert self._boulder is not None
        step_index = append_step(
            self._boulder.task_id,
            tool=tool,
            args_summary=args_summary,
            status=status,
        )
        self._current_step_index = step_index
        # 重新加载（append_step 内部已 save，但本地 _boulder 引用没更新）
        self._boulder = load_boulder(self._boulder.task_id)
        return step_index

    def complete_step(
        self,
        *,
        result_summary: str = "",
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """回填当前 step 的结果。"""
        if self._boulder is None or self._current_step_index < 0:
            return
        complete_step(
            self._boulder.task_id,
            self._current_step_index,
            result_summary=result_summary,
            status=status,
            error=error,
        )

    def record_evidence(
        self,
        *,
        source: str,
        content: str,
        citation: Optional[str] = None,
        confidence: str = "high",
    ) -> None:
        """记录一步证据。step 默认用 current_step_index。"""
        if self._boulder is None or self._current_step_index < 0:
            return
        record_evidence(
            self._boulder.task_id,
            self._current_step_index,
            source=source,
            content=content,
            citation=citation,
            confidence=confidence,
        )

    def finish(
        self,
        *,
        status: str = "completed",
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[Path]:
        """收尾并生成 audit 报告。返回 audit.md 路径。"""
        if self._boulder is None:
            return None
        finish_boulder(
            self._boulder.task_id,
            status=status,
            result_summary=result_summary,
            error=error,
        )
        return write_audit(self._boulder.task_id)
