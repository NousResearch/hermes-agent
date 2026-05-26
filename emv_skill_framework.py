"""
Emv 熵 Skill 框架 - 创新上下文学习系统

核心机制：
1. 三角色系统：Challenger（出题）、Reasoner（解题）、Judge（判题）
2. 自博弈循环：多智能体迭代优化技能
3. 跨时间重放：避免对抗性崩溃
4. 熵选择机制：基尼不纯度、信息增益、随机森林
"""

import json
import math
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """任务定义"""
    id: str
    description: str
    difficulty: float  # 0-1
    category: str
    created_by: str  # "challenger"
    created_at: str


@dataclass
class Solution:
    """解决方案"""
    id: str
    task_id: str
    approach: str
    code: str
    created_by: str  # "reasoner"
    created_at: str


@dataclass
class Judgment:
    """判题结果"""
    id: str
    solution_id: str
    correct: bool
    score: float  # 0-1
    feedback: str
    created_by: str  # "judge"
    created_at: str


@dataclass
class Skill:
    """提炼的技能"""
    id: str
    name: str
    description: str
    pattern: str  # 技能模式
    success_rate: float
    usage_count: int
    gini_impurity: float  # 基尼不纯度
    entropy: float  # 信息熵
    created_at: str


class EntropyCalculator:
    """熵计算器"""
    
    @staticmethod
    def gini_impurity(labels: List[int]) -> float:
        """
        计算基尼不纯度
        
        公式：Gini = 1 - Σ(p_k^2)
        
        Args:
            labels: 类别标签列表
            
        Returns:
            基尼不纯度（0-1）
        """
        if not labels:
            return 0.0
        
        counter = Counter(labels)
        total = len(labels)
        
        gini = 1.0
        for count in counter.values():
            p_k = count / total
            gini -= p_k ** 2
        
        return gini
    
    @staticmethod
    def entropy(labels: List[int]) -> float:
        """
        计算信息熵
        
        公式：H = -Σ(p_k * log2(p_k))
        
        Args:
            labels: 类别标签列表
            
        Returns:
            信息熵（bits）
        """
        if not labels:
            return 0.0
        
        counter = Counter(labels)
        total = len(labels)
        
        h = 0.0
        for count in counter.values():
            if count > 0:
                p_k = count / total
                h -= p_k * math.log2(p_k)
        
        return h
    
    @staticmethod
    def information_gain(
        parent_labels: List[int],
        left_labels: List[int],
        right_labels: List[int]
    ) -> float:
        """
        计算信息增益
        
        公式：IG = H_parent - (N_L/N * H_L + N_R/N * H_R)
        
        Args:
            parent_labels: 父节点标签
            left_labels: 左子节点标签
            right_labels: 右子节点标签
            
        Returns:
            信息增益
        """
        n = len(parent_labels)
        if n == 0:
            return 0.0
        
        h_parent = EntropyCalculator.entropy(parent_labels)
        
        n_l = len(left_labels)
        n_r = len(right_labels)
        
        h_left = EntropyCalculator.entropy(left_labels)
        h_right = EntropyCalculator.entropy(right_labels)
        
        ig = h_parent - (n_l / n * h_left + n_r / n * h_right)
        
        return ig


class Challenger:
    """出题者角色"""
    
    def __init__(self, difficulty_range: Tuple[float, float] = (0.3, 0.8)):
        """
        初始化出题者
        
        Args:
            difficulty_range: 难度范围（最小值，最大值）
        """
        self.difficulty_range = difficulty_range
        self.task_history = []
    
    def generate_task(self, context: str, category: str = "general") -> Task:
        """
        生成任务
        
        Args:
            context: 上下文信息
            category: 任务类别
            
        Returns:
            任务对象
        """
        # 简化实现：随机生成难度
        difficulty = random.uniform(*self.difficulty_range)
        
        task = Task(
            id=f"task_{datetime.utcnow().timestamp()}",
            description=f"Task based on context: {context[:100]}...",
            difficulty=difficulty,
            category=category,
            created_by="challenger",
            created_at=datetime.utcnow().isoformat()
        )
        
        self.task_history.append(task)
        
        logger.info(f"🎯 Challenger 生成任务: {task.id} (难度: {difficulty:.2f})")
        
        return task
    
    def adjust_difficulty(self, success_rate: float):
        """
        根据成功率调整难度
        
        Args:
            success_rate: 当前成功率
        """
        if success_rate > 0.8:
            # 太简单，提高难度
            self.difficulty_range = (
                min(self.difficulty_range[0] + 0.1, 0.9),
                min(self.difficulty_range[1] + 0.1, 1.0)
            )
            logger.info(f"📈 提高难度: {self.difficulty_range}")
        elif success_rate < 0.4:
            # 太难，降低难度
            self.difficulty_range = (
                max(self.difficulty_range[0] - 0.1, 0.1),
                max(self.difficulty_range[1] - 0.1, 0.3)
            )
            logger.info(f"📉 降低难度: {self.difficulty_range}")


class Reasoner:
    """解题者角色"""
    
    def __init__(self):
        """初始化解题者"""
        self.solution_history = []
        self.learned_patterns = []
    
    def solve_task(self, task: Task, skills: List[Skill]) -> Solution:
        """
        解决任务
        
        Args:
            task: 任务对象
            skills: 可用技能列表
            
        Returns:
            解决方案
        """
        # 选择最佳技能
        best_skill = self._select_best_skill(task, skills)
        
        # 生成解决方案
        solution = Solution(
            id=f"sol_{datetime.utcnow().timestamp()}",
            task_id=task.id,
            approach=f"Using skill: {best_skill.name if best_skill else 'default'}",
            code=f"# Solution for {task.description[:50]}...",
            created_by="reasoner",
            created_at=datetime.utcnow().isoformat()
        )
        
        self.solution_history.append(solution)
        
        logger.info(f"💡 Reasoner 生成方案: {solution.id}")
        
        return solution
    
    def _select_best_skill(self, task: Task, skills: List[Skill]) -> Optional[Skill]:
        """
        选择最佳技能（基于熵）
        
        Args:
            task: 任务对象
            skills: 技能列表
            
        Returns:
            最佳技能或 None
        """
        if not skills:
            return None
        
        # 按成功率和低熵排序
        scored_skills = [
            (skill, skill.success_rate * (1 - skill.gini_impurity))
            for skill in skills
        ]
        
        scored_skills.sort(key=lambda x: x[1], reverse=True)
        
        return scored_skills[0][0] if scored_skills else None


class Judge:
    """判题者角色"""
    
    def __init__(self):
        """初始化判题者"""
        self.judgment_history = []
    
    def evaluate_solution(self, solution: Solution, task: Task) -> Judgment:
        """
        评估解决方案
        
        Args:
            solution: 解决方案
            task: 原始任务
            
        Returns:
            判题结果
        """
        # 简化实现：基于难度随机判定
        # 实际应该调用 LLM 或执行代码验证
        success_prob = 1.0 - task.difficulty * 0.5
        correct = random.random() < success_prob
        score = random.uniform(0.7, 1.0) if correct else random.uniform(0.0, 0.4)
        
        judgment = Judgment(
            id=f"judge_{datetime.utcnow().timestamp()}",
            solution_id=solution.id,
            correct=correct,
            score=score,
            feedback=f"{'Correct' if correct else 'Incorrect'} solution",
            created_by="judge",
            created_at=datetime.utcnow().isoformat()
        )
        
        self.judgment_history.append(judgment)
        
        logger.info(f"⚖️ Judge 评估: {judgment.id} ({'✅' if correct else '❌'}, 分数: {score:.2f})")
        
        return judgment


class EmvSkillFramework:
    """Emv 熵 Skill 框架主控"""
    
    def __init__(self):
        """初始化框架"""
        self.challenger = Challenger()
        self.reasoner = Reasoner()
        self.judge = Judge()
        self.skills = []
        self.replay_buffer = []  # 跨时间重放缓冲区
    
    def run_iteration(self, context: str, category: str = "general") -> Dict[str, Any]:
        """
        运行一次迭代
        
        流程：
        1. Challenger 出题
        2. Reasoner 解题
        3. Judge 判题
        4. 提炼技能
        
        Args:
            context: 上下文信息
            category: 任务类别
            
        Returns:
            迭代结果
        """
        # 1. Challenger 出题
        task = self.challenger.generate_task(context, category)
        
        # 2. Reasoner 解题
        solution = self.reasoner.solve_task(task, self.skills)
        
        # 3. Judge 判题
        judgment = self.judge.evaluate_solution(solution, task)
        
        # 4. 提炼技能
        if judgment.correct:
            skill = self._extract_skill(task, solution, judgment)
            self.skills.append(skill)
            logger.info(f"✨ 提炼新技能: {skill.name}")
        
        # 5. 添加到重放缓冲区
        self.replay_buffer.append({
            'task': task,
            'solution': solution,
            'judgment': judgment
        })
        
        # 6. 调整难度
        recent_success_rate = self._calculate_recent_success_rate()
        self.challenger.adjust_difficulty(recent_success_rate)
        
        return {
            'task': task,
            'solution': solution,
            'judgment': judgment,
            'skills_count': len(self.skills),
            'success_rate': recent_success_rate
        }
    
    def run_multiple_iterations(self, context: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        运行多次迭代
        
        Args:
            context: 上下文信息
            n: 迭代次数
            
        Returns:
            迭代结果列表
        """
        results = []
        
        for i in range(n):
            logger.info(f"\n{'='*70}")
            logger.info(f"迭代 {i+1}/{n}")
            logger.info(f"{'='*70}")
            
            result = self.run_iteration(context)
            results.append(result)
        
        return results
    
    def _extract_skill(self, task: Task, solution: Solution, judgment: Judgment) -> Skill:
        """
        从成功案例中提炼技能
        
        Args:
            task: 任务
            solution: 解决方案
            judgment: 判题结果
            
        Returns:
            技能对象
        """
        # 计算技能的熵指标
        # 简化实现：使用随机值
        labels = [1 if j.correct else 0 for j in self.judge.judgment_history[-10:]]
        
        gini = EntropyCalculator.gini_impurity(labels)
        entropy = EntropyCalculator.entropy(labels)
        
        skill = Skill(
            id=f"skill_{datetime.utcnow().timestamp()}",
            name=f"Skill_{len(self.skills)+1}",
            description=f"Pattern from task {task.id}",
            pattern=solution.approach,
            success_rate=judgment.score,
            usage_count=1,
            gini_impurity=gini,
            entropy=entropy,
            created_at=datetime.utcnow().isoformat()
        )
        
        return skill
    
    def _calculate_recent_success_rate(self, window: int = 10) -> float:
        """
        计算最近的成功率
        
        Args:
            window: 窗口大小
            
        Returns:
            成功率（0-1）
        """
        recent_judgments = self.judge.judgment_history[-window:]
        
        if not recent_judgments:
            return 0.5
        
        success_count = sum(1 for j in recent_judgments if j.correct)
        return success_count / len(recent_judgments)
    
    def get_best_skills(self, top_k: int = 5) -> List[Skill]:
        """
        获取最佳技能
        
        Args:
            top_k: 返回数量
            
        Returns:
            技能列表
        """
        # 按成功率和低熵排序
        scored_skills = [
            (skill, skill.success_rate * (1 - skill.gini_impurity))
            for skill in self.skills
        ]
        
        scored_skills.sort(key=lambda x: x[1], reverse=True)
        
        return [skill for skill, _ in scored_skills[:top_k]]


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("Emv 熵 Skill 框架测试")
    print("=" * 70)
    print()
    
    # 创建框架
    framework = EmvSkillFramework()
    
    # 运行多次迭代
    context = "Long document about Python programming and data structures"
    results = framework.run_multiple_iterations(context, n=5)
    
    print()
    print("=" * 70)
    print("迭代结果总结")
    print("=" * 70)
    print()
    
    # 统计
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r['judgment'].correct)
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
    
    print(f"总任务数: {total_tasks}")
    print(f"成功任务: {successful_tasks}")
    print(f"成功率: {success_rate:.2%}")
    print(f"提炼技能数: {len(framework.skills)}")
    print()
    
    # 最佳技能
    best_skills = framework.get_best_skills(top_k=3)
    print(f"最佳技能 (Top 3):")
    for i, skill in enumerate(best_skills, 1):
        print(f"  {i}. {skill.name}")
        print(f"     成功率: {skill.success_rate:.2f}")
        print(f"     基尼不纯度: {skill.gini_impurity:.2f}")
        print(f"     信息熵: {skill.entropy:.2f}")
    print()
    
    print("=" * 70)
    print("✅ Emv 熵 Skill 框架测试完成")
    print("=" * 70)
