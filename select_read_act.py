"""
Select-Read-Act (SRA) 三段式闭环

核心流程：
1. Select - 根据任务选择最佳技能
2. Read - 读取技能代码和上下文
3. Act - 执行技能并记录结果
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from skill_bank import SkillBank, Skill, get_skill_bank

logger = logging.getLogger(__name__)


@dataclass
class SRAContext:
    """SRA 上下文"""
    task: str
    user_input: str
    conversation_history: List[Dict[str, str]]
    available_tools: List[str]
    metadata: Dict[str, Any] = None


@dataclass
class SRAResult:
    """SRA 执行结果"""
    selected_skill: Optional[Skill]
    execution_result: Any
    success: bool
    score: float
    feedback: str


class SelectReadAct:
    """Select-Read-Act 三段式闭环引擎"""
    
    def __init__(self, skill_bank: Optional[SkillBank] = None):
        """
        初始化 SRA 引擎
        
        Args:
            skill_bank: 技能库（可选，默认使用全局实例）
        """
        self.skill_bank = skill_bank or get_skill_bank()
    
    def execute(self, context: SRAContext) -> SRAResult:
        """
        执行完整的 SRA 流程
        
        Args:
            context: SRA 上下文
            
        Returns:
            SRA 执行结果
        """
        # 1. Select - 选择最佳技能
        selected_skill = self._select_skill(context)
        
        if not selected_skill:
            return SRAResult(
                selected_skill=None,
                execution_result=None,
                success=False,
                score=0.0,
                feedback="未找到合适的技能"
            )
        
        logger.info(f"🎯 Select: 选择技能 {selected_skill.name}")
        
        # 2. Read - 读取技能代码和依赖
        skill_context = self._read_skill(selected_skill, context)
        
        logger.info(f"📖 Read: 加载技能上下文")
        
        # 3. Act - 执行技能
        result = self._act_skill(selected_skill, skill_context, context)
        
        logger.info(f"⚡ Act: 执行完成 (成功: {result.success})")
        
        # 4. 更新技能统计
        self.skill_bank.update_skill_stats(
            selected_skill.id,
            success=result.success,
            score=result.score
        )
        
        return result
    
    def _select_skill(self, context: SRAContext) -> Optional[Skill]:
        """
        Select 阶段：选择最佳技能
        
        策略：
        1. 基于任务关键词搜索
        2. 按质量分数排序
        3. 考虑依赖可用性
        
        Args:
            context: SRA 上下文
            
        Returns:
            选中的技能或 None
        """
        # 提取任务关键词
        keywords = self._extract_keywords(context.task)
        
        # 搜索相关技能
        candidates = []
        for keyword in keywords:
            skills = self.skill_bank.search_skills(query=keyword, limit=5)
            candidates.extend(skills)
        
        # 去重
        seen = set()
        unique_candidates = []
        for skill in candidates:
            if skill.id not in seen:
                seen.add(skill.id)
                unique_candidates.append(skill)
        
        if not unique_candidates:
            logger.warning(f"⚠️ 未找到匹配技能: {context.task}")
            return None
        
        # 按质量分数排序
        unique_candidates.sort(key=lambda s: s.quality_score, reverse=True)
        
        # 选择最佳技能
        best_skill = unique_candidates[0]
        
        logger.debug(f"🎯 候选技能: {len(unique_candidates)} 个")
        logger.debug(f"🏆 最佳技能: {best_skill.name} (质量: {best_skill.quality_score:.2f})")
        
        return best_skill
    
    def _read_skill(self, skill: Skill, context: SRAContext) -> Dict[str, Any]:
        """
        Read 阶段：读取技能代码和上下文
        
        Args:
            skill: 选中的技能
            context: SRA 上下文
            
        Returns:
            技能执行上下文
        """
        skill_context = {
            'skill': skill,
            'code': skill.code,
            'dependencies': skill.dependencies,
            'task': context.task,
            'user_input': context.user_input,
            'conversation_history': context.conversation_history,
            'available_tools': context.available_tools,
        }
        
        return skill_context
    
    def _act_skill(
        self,
        skill: Skill,
        skill_context: Dict[str, Any],
        context: SRAContext
    ) -> SRAResult:
        """
        Act 阶段：执行技能
        
        Args:
            skill: 选中的技能
            skill_context: 技能执行上下文
            context: SRA 上下文
            
        Returns:
            执行结果
        """
        try:
            # TODO: 实际执行技能代码
            # 这里需要安全的代码执行环境
            # 可以使用 exec() 或调用外部工具
            
            # 模拟执行
            execution_result = {
                'skill_name': skill.name,
                'task': context.task,
                'status': 'simulated'
            }
            
            success = True
            score = 0.8
            feedback = f"技能 {skill.name} 执行成功（模拟）"
            
            return SRAResult(
                selected_skill=skill,
                execution_result=execution_result,
                success=success,
                score=score,
                feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"❌ 技能执行失败: {e}")
            return SRAResult(
                selected_skill=skill,
                execution_result=None,
                success=False,
                score=0.0,
                feedback=f"执行失败: {str(e)}"
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 简单实现：分词 + 过滤停用词
        words = text.lower().split()
        
        # 停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]  # 最多返回 5 个关键词


# 便捷函数
def execute_sra(task: str, user_input: str = "", **kwargs) -> SRAResult:
    """
    执行 SRA 流程的便捷函数
    
    Args:
        task: 任务描述
        user_input: 用户输入
        **kwargs: 其他上下文参数
        
    Returns:
        SRA 执行结果
    """
    context = SRAContext(
        task=task,
        user_input=user_input,
        conversation_history=kwargs.get('conversation_history', []),
        available_tools=kwargs.get('available_tools', []),
        metadata=kwargs.get('metadata', {})
    )
    
    sra = SelectReadAct()
    return sra.execute(context)


if __name__ == "__main__":
    # 测试
    import tempfile
    from skill_bank import SkillBank, Skill
    
    print("=" * 70)
    print("Select-Read-Act 测试")
    print("=" * 70)
    print()
    
    # 创建临时技能库
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    bank = SkillBank(db_path)
    
    # 添加测试技能
    skills = [
        Skill(
            id="search_001",
            name="web_search",
            description="网络搜索技能",
            category="search",
            code="def web_search(query): return search_results",
            dependencies=["requests"]
        ),
        Skill(
            id="file_001",
            name="read_file",
            description="读取文件技能",
            category="file",
            code="def read_file(path): return file_content",
            dependencies=[]
        ),
    ]
    
    for skill in skills:
        bank.add_skill(skill)
        # 模拟一些使用统计
        bank.update_skill_stats(skill.id, success=True, score=0.85)
        bank.update_skill_stats(skill.id, success=True, score=0.90)
    
    print(f"✅ 添加了 {len(skills)} 个技能")
    print()
    
    # 测试 SRA 流程
    sra = SelectReadAct(skill_bank=bank)
    
    context = SRAContext(
        task="search for python tutorials",
        user_input="I want to learn python",
        conversation_history=[],
        available_tools=["web_search", "read_file"]
    )
    
    print(f"📋 任务: {context.task}")
    print()
    
    result = sra.execute(context)
    
    print(f"🎯 选择技能: {result.selected_skill.name if result.selected_skill else 'None'}")
    print(f"✅ 执行成功: {result.success}")
    print(f"📊 评分: {result.score:.2f}")
    print(f"💬 反馈: {result.feedback}")
    print()
    
    # 验证统计更新
    skill = bank.get_skill("search_001")
    print(f"📈 技能统计:")
    print(f"   使用次数: {skill.usage_count}")
    print(f"   成功次数: {skill.success_count}")
    print(f"   平均分数: {skill.avg_score:.2f}")
    print(f"   质量评分: {skill.quality_score:.2f}")
    print()
    
    print("=" * 70)
    print("✅ Select-Read-Act 测试完成")
    print("=" * 70)
