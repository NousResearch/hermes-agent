"""策略进化引擎 - 遗传算法式策略优化。"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class Strategy:
    """诊断策略。"""
    id: str
    name: str
    component: str
    fault_type: str
    check_order: List[str]           # 检查步骤顺序
    thresholds: Dict[str, float]     # 阈值参数
    fix_actions: List[str]           # 修复动作
    created_at: str = ""
    fitness: float = 0.0
    use_count: int = 0
    success_count: int = 0
    avg_time_seconds: float = 0.0
    last_used: str = ""
    status: str = "active"           # active | observing | archived

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyEvolution:
    """策略进化引擎。

    进化周期：
    - 周一：种群评估（计算适应度）
    - 周三：交叉繁殖（组合策略）
    - 周五：自然选择（归档低效策略）
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            import os
            home = os.getenv("MES_INSPECTION_HOME", "")
            if home:
                self.data_dir = Path(home) / "evolution"
            else:
                self.data_dir = Path.home() / ".mes-inspection" / "evolution"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.strategies_path = self.data_dir / "strategies.json"
        self.strategies: Dict[str, Strategy] = self._load_strategies()

    def _load_strategies(self) -> Dict[str, Strategy]:
        if self.strategies_path.exists():
            with open(self.strategies_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                sid: Strategy(**s) for sid, s in data.items()
            }
        return {}

    def _save_strategies(self):
        data = {sid: s.to_dict() for sid, s in self.strategies.items()}
        with open(self.strategies_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record_usage(
        self,
        strategy_id: str,
        success: bool,
        elapsed_seconds: float = 0,
    ):
        """记录策略使用结果。"""
        if strategy_id not in self.strategies:
            return
        s = self.strategies[strategy_id]
        s.use_count += 1
        if success:
            s.success_count += 1
        # 更新平均时间
        s.avg_time_seconds = (
            (s.avg_time_seconds * (s.use_count - 1) + elapsed_seconds) / s.use_count
        )
        s.last_used = datetime.now().isoformat()
        self._save_strategies()

    def calculate_fitness(self, strategy: Strategy) -> float:
        """计算策略适应度。

        fitness = 成功率 × 效率系数 × 时间衰减
        """
        if strategy.use_count == 0:
            return 0.0

        # 成功率
        success_rate = strategy.success_count / strategy.use_count

        # 效率系数 (1 - 平均时间/最大超时)，最大超时假设 600s
        max_timeout = 600
        efficiency = max(0, 1 - strategy.avg_time_seconds / max_timeout)

        # 时间衰减
        if strategy.last_used:
            try:
                last = datetime.fromisoformat(strategy.last_used)
                days_ago = (datetime.now() - last).days
                if days_ago <= 30:
                    recency = 1.0
                elif days_ago <= 60:
                    recency = 0.8
                elif days_ago <= 90:
                    recency = 0.5
                else:
                    recency = 0.2
            except ValueError:
                recency = 0.5
        else:
            recency = 0.5

        fitness = success_rate * efficiency * recency
        return round(fitness, 4)

    def evolve(self) -> Dict[str, Any]:
        """执行一轮进化，返回进化报告。"""
        # 1. 评估所有策略
        for s in self.strategies.values():
            s.fitness = self.calculate_fitness(s)

        # 2. 分类
        active = [s for s in self.strategies.values() if s.status == "active"]
        active.sort(key=lambda s: s.fitness, reverse=True)

        # 3. 自然选择：保留 Top 80%
        cutoff = max(1, int(len(active) * 0.8))
        survivors = active[:cutoff]
        to_archive = active[cutoff:]

        archived_count = 0
        for s in to_archive:
            s.status = "archived"
            archived_count += 1

        # 4. 对观察中的策略评估
        observing = [s for s in self.strategies.values() if s.status == "observing"]
        for s in observing:
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

    def create_strategy(
        self,
        name: str,
        component: str,
        fault_type: str,
        check_order: List[str],
        thresholds: Dict[str, float],
        fix_actions: List[str],
    ) -> Strategy:
        """创建新策略。"""
        sid = f"s_{component}_{fault_type}_{len(self.strategies)+1:03d}"
        strategy = Strategy(
            id=sid,
            name=name,
            component=component,
            fault_type=fault_type,
            check_order=check_order,
            thresholds=thresholds,
            fix_actions=fix_actions,
            created_at=datetime.now().isoformat(),
            status="observing",  # 新策略先进入观察期
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
