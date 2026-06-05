"""策略进化引擎 — 集成 hermes-agent-self-evolution (DSPy + GEPA)。

底层进化算法使用 NousResearch/hermes-agent-self-evolution 开源项目：
https://github.com/NousResearch/hermes-agent-self-evolution

本模块提供 MES 巡检场景的适配层：
- 将巡检 Skill 注册为可进化目标
- 将故障案例转化为评估数据集
- 调用 GEPA 优化器迭代改进 Skill
- 保留本地策略跟踪（使用统计、活跃/归档状态）
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


# ── 本地策略跟踪 ──────────────────────────────────────────────────────

@dataclass
class Strategy:
    """诊断策略（本地跟踪）。"""
    id: str
    name: str
    component: str
    fault_type: str
    check_order: List[str]
    thresholds: Dict[str, float]
    fix_actions: List[str]
    created_at: str = ""
    fitness: float = 0.0
    use_count: int = 0
    success_count: int = 0
    avg_time_seconds: float = 0.0
    last_used: str = ""
    status: str = "active"           # active | observing | archived

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── hermes-agent-self-evolution 集成 ───────────────────────────────────

def _find_evolution_lib() -> Optional[Path]:
    """查找 hermes-agent-self-evolution 库路径。

    搜索顺序：
    1. vendor/ 子目录（git clone）
    2. PYTHONPATH 中已安装的 evolution 包
    """
    # 1. vendor 目录
    vendor_path = Path(__file__).parent.parent / "vendor" / "hermes-agent-self-evolution"
    if vendor_path.exists() and (vendor_path / "evolution" / "__init__.py").exists():
        return vendor_path

    # 2. 已安装的包
    try:
        import evolution.core.config
        return None  # 已在 PYTHONPATH 中
    except ImportError:
        return None


def _ensure_evolution_importable():
    """确保 evolution 包可导入。"""
    lib_path = _find_evolution_lib()
    if lib_path and str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))


class StrategyEvolution:
    """策略进化引擎。

    集成 hermes-agent-self-evolution 的 DSPy + GEPA 优化器，
    用于自动进化 MES 巡检 Skills。

    进化周期：
    - 运行 evolve_skill 命令优化 SKILL.md
    - 本地跟踪策略使用统计和适应度
    - 保留 Top 80% 策略，归档 Bottom 20%
    """

    def __init__(self, config: Dict[str, Any] = None, data_dir: Optional[str] = None):
        self.config = config or {}

        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            home = os.getenv("MES_INSPECTION_HOME", "")
            if home:
                self.data_dir = Path(home) / "evolution"
            else:
                self.data_dir = Path.home() / ".mes-inspection" / "evolution"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.strategies_path = self.data_dir / "strategies.json"
        self.strategies: Dict[str, Strategy] = self._load_strategies()

        # 进化配置
        self.iterations = self.config.get("iterations", 10)
        self.eval_source = self.config.get("eval_source", "synthetic")
        self.optimizer_model = self.config.get("optimizer_model", "openai/gpt-4.1")
        self.eval_model = self.config.get("eval_model", "openai/gpt-4.1-mini")
        self.hermes_repo = self.config.get("hermes_repo", os.getenv("HERMES_AGENT_REPO", ""))
        self.skills_dir = self.config.get("skills_dir", "")

    def _load_strategies(self) -> Dict[str, Strategy]:
        if self.strategies_path.exists():
            with open(self.strategies_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {sid: Strategy(**s) for sid, s in data.items()}
        return {}

    def _save_strategies(self):
        data = {sid: s.to_dict() for sid, s in self.strategies.items()}
        with open(self.strategies_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 策略跟踪 ──────────────────────────────────────────────────────

    def record_usage(self, strategy_id: str, success: bool, elapsed_seconds: float = 0):
        """记录策略使用结果。"""
        if strategy_id not in self.strategies:
            return
        s = self.strategies[strategy_id]
        s.use_count += 1
        if success:
            s.success_count += 1
        s.avg_time_seconds = (
            (s.avg_time_seconds * (s.use_count - 1) + elapsed_seconds) / s.use_count
        )
        s.last_used = datetime.now().isoformat()
        self._save_strategies()

    def calculate_fitness(self, strategy: Strategy) -> float:
        """计算策略适应度。"""
        if strategy.use_count == 0:
            return 0.0
        success_rate = strategy.success_count / strategy.use_count
        efficiency = max(0, 1 - strategy.avg_time_seconds / 600)
        if strategy.last_used:
            try:
                days_ago = (datetime.now() - datetime.fromisoformat(strategy.last_used)).days
                recency = 1.0 if days_ago <= 30 else (0.8 if days_ago <= 60 else (0.5 if days_ago <= 90 else 0.2))
            except ValueError:
                recency = 0.5
        else:
            recency = 0.5
        return round(success_rate * efficiency * recency, 4)

    def create_strategy(
        self, name: str, component: str, fault_type: str,
        check_order: List[str], thresholds: Dict[str, float], fix_actions: List[str],
    ) -> Strategy:
        """创建新策略（先进入观察期）。"""
        sid = f"s_{component}_{fault_type}_{len(self.strategies)+1:03d}"
        strategy = Strategy(
            id=sid, name=name, component=component, fault_type=fault_type,
            check_order=check_order, thresholds=thresholds, fix_actions=fix_actions,
            created_at=datetime.now().isoformat(), status="observing",
        )
        self.strategies[sid] = strategy
        self._save_strategies()
        return strategy

    def get_active_strategies(self, component: str = None) -> List[Strategy]:
        """获取活跃策略。"""
        strategies = [s for s in self.strategies.values() if s.status == "active"]
        if component:
            strategies = [s for s in strategies if s.component == component]
        return sorted(strategies, key=lambda s: s.fitness, reverse=True)

    # ── GEPA 进化 ─────────────────────────────────────────────────────

    def evolve_skill(self, skill_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """使用 hermes-agent-self-evolution 的 GEPA 优化器进化指定 Skill。

        Args:
            skill_name: 技能名称（如 mes-nginx-check）
            dry_run: 仅验证配置，不执行优化

        Returns:
            进化结果摘要
        """
        _ensure_evolution_importable()

        cmd = [
            sys.executable, "-m", "evolution.skills.evolve_skill",
            "--skill", skill_name,
            "--iterations", str(self.iterations),
            "--eval-source", self.eval_source,
            "--optimizer-model", self.optimizer_model,
            "--eval-model", self.eval_model,
        ]

        if self.hermes_repo:
            cmd.extend(["--hermes-repo", self.hermes_repo])
        if dry_run:
            cmd.append("--dry-run")

        env = os.environ.copy()
        if self.hermes_repo:
            env["HERMES_AGENT_REPO"] = self.hermes_repo

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800, env=env,
            )
            return {
                "success": result.returncode == 0,
                "skill_name": skill_name,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "进化超时（30分钟限制）"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def evolve_all_mes_skills(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """进化所有 MES 巡检 Skills。"""
        mes_skills = [
            "mes-nginx-check", "mes-jvm-check", "mes-rabbitmq-check",
            "mes-oracle-check", "mes-elk-check", "mes-skywalking-check",
        ]
        results = []
        for skill_name in mes_skills:
            result = self.evolve_skill(skill_name, dry_run=dry_run)
            results.append(result)
        return results

    # ── 自然选择 ──────────────────────────────────────────────────────

    def evolve(self) -> Dict[str, Any]:
        """执行一轮本地策略评估和自然选择。"""
        for s in self.strategies.values():
            s.fitness = self.calculate_fitness(s)

        active = sorted(
            [s for s in self.strategies.values() if s.status == "active"],
            key=lambda s: s.fitness, reverse=True,
        )

        cutoff = max(1, int(len(active) * 0.8))
        survivors = active[:cutoff]
        archived_count = 0
        for s in active[cutoff:]:
            s.status = "archived"
            archived_count += 1

        for s in self.strategies.values():
            if s.status == "observing":
                if s.use_count >= 5 and s.fitness < 0.3:
                    s.status = "archived"
                    archived_count += 1
                elif s.use_count >= 5 and s.fitness >= 0.3:
                    s.status = "active"

        self._save_strategies()

        return {
            "timestamp": datetime.now().isoformat(),
            "total_strategies": len(self.strategies),
            "active": len([s for s in self.strategies.values() if s.status == "active"]),
            "observing": len([s for s in self.strategies.values() if s.status == "observing"]),
            "archived": archived_count,
            "top_strategies": [
                {"name": s.name, "fitness": s.fitness, "use_count": s.use_count}
                for s in survivors[:5]
            ],
        }

    def format_evolution_report(self) -> str:
        """格式化进化报告。"""
        report = self.evolve()
        lines = [
            "🧬 策略进化报告",
            "━" * 20,
            f"总策略数: {report['total_strategies']}",
            f"活跃: {report['active']}",
            f"观察中: {report['observing']}",
            f"本轮归档: {report['archived']}",
            "",
            "🏆 Top 策略:",
        ]
        for s in report.get("top_strategies", []):
            lines.append(f"  {s['name']}: 适应度={s['fitness']}, 使用={s['use_count']}次")
        return "\n".join(lines)
