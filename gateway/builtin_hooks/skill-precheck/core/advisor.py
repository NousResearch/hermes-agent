"""
SRA - Skill Runtime Advisor
让 AI Agent 知道自己有什么能力，以及什么时候该用什么能力。

主入口：SkillAdvisor 类
"""

import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from .indexer import SkillIndexer
from .matcher import SkillMatcher
from .memory import SceneMemory
from .synonyms import SYNONYMS


class SkillAdvisor:
    """技能推荐引擎主类"""

    # 推荐阈值
    THRESHOLD_STRONG = 80  # 强推荐：自动加载
    THRESHOLD_WEAK = 50    # 弱推荐：medium confidence

    def __init__(self, skills_dir: str = None, data_dir: str = None):
        """
        初始化 SRA 引擎 (轻量级 In-Process 版本)
        
        Args:
            skills_dir: 技能目录路径。默认为 ~/.hermes/skills
            data_dir: 数据持久化目录。默认为 ~/.hermes/hooks/data
        """
        from pathlib import Path
        self.skills_dir = skills_dir or os.path.expanduser("~/.hermes/skills")
        if data_dir:
            self.data_dir = data_dir
        else:
            # 默认使用 ~/.hermes/hooks/data 存储持久化数据
            base = os.path.expanduser("~/.hermes/hooks")
            self.data_dir = os.path.join(base, "data")
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化子模块
        self.indexer = SkillIndexer(self.skills_dir, self.data_dir)
        self.matcher = SkillMatcher(SYNONYMS)
        self.memory = SceneMemory(self.data_dir)
        
        # 懒加载索引
        self._index_loaded = False

    def _ensure_index(self):
        """确保索引已加载"""
        if not self._index_loaded:
            self.indexer.load_or_build()
            self._index_loaded = True

    def refresh_index(self) -> int:
        """强制刷新技能索引"""
        count = self.indexer.build()
        self._index_loaded = True
        return count

    def recommend(self, query: str, top_k: int = 3) -> Dict:
        """
        推荐匹配技能
        
        Args:
            query: 用户输入
            top_k: 返回 top-k 结果
            
        Returns:
            {
                "recommendations": [...],
                "processing_ms": float,
                "skills_scanned": int,
                "query": str
            }
        """
        self._ensure_index()
        start = time.time()
        
        skills = self.indexer.get_skills()
        stats = self.memory.load()
        
        # 提取输入关键词
        input_words = self.indexer.extract_keywords(query)
        input_expanded = self.indexer.expand_with_synonyms(input_words)
        
        if not input_expanded:
            return {"recommendations": [], "processing_ms": 0, "skills_scanned": 0, "query": query}
        
        # 对所有 skill 评分
        scored = []
        for skill in skills:
            total, details, reasons = self.matcher.score(
                input_expanded, skill, stats
            )
            
            if total >= self.THRESHOLD_WEAK:
                scored.append({
                    "skill": skill["name"],
                    "description": skill.get("description", "")[:120],
                    "category": skill.get("category", ""),
                    "score": round(total, 1),
                    "confidence": "high" if total >= self.THRESHOLD_STRONG else "medium",
                    "reasons": reasons[:3],
                    "details": details,
                })
        
        # 排序取 top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]
        
        # 更新推荐计数
        self.memory.increment_recommendations()
        
        elapsed = round((time.time() - start) * 1000, 1)
        
        return {
            "recommendations": top,
            "processing_ms": elapsed,
            "skills_scanned": len(skills),
            "query": query,
        }

    def record_usage(self, skill_name: str, user_input: str, accepted: bool = True):
        """记录技能使用场景"""
        self.memory.record_usage(skill_name, user_input, accepted)

    def show_stats(self) -> Dict:
        """获取统计信息"""
        self._ensure_index()
        stats = self.memory.load()
        skills = self.indexer.get_skills()
        
        return {
            "total_skills": len(skills),
            "total_recommendations": stats.get("total_recommendations", 0),
            "scene_patterns": len(stats.get("scene_patterns", [])),
            "skills_with_stats": len(stats.get("skills", {})),
            "memory": stats,
        }

    def analyze_coverage(self) -> Dict:
        """
        分析 SRA 对技能目录的覆盖率
        返回：每个 skill 是否能被 SRA 的 trigger 机制识别
        """
        self._ensure_index()
        skills = self.indexer.get_skills()
        stats = self.memory.load()
        
        results = []
        covered = 0
        for skill in skills:
            # 用 skill 自己的 triggers 作为测试查询
            triggers = skill.get("triggers", [])
            name = skill.get("name", "")
            
            # 构造测试查询
            test_queries = []
            if triggers:
                test_queries.extend(triggers[:3])
            # 用 name 作为后备查询
            test_queries.append(name.replace("-", " "))
            
            # 测试所有查询
            max_score = 0
            for q in test_queries:
                if not q:
                    continue
                input_words = self.indexer.extract_keywords(q)
                input_expanded = self.indexer.expand_with_synonyms(input_words)
                if input_expanded:
                    total, _, _ = self.matcher.score(input_expanded, skill, stats)
                    max_score = max(max_score, total)
            
            is_covered = max_score >= self.THRESHOLD_WEAK
            if is_covered:
                covered += 1
            
            results.append({
                "name": skill["name"],
                "category": skill.get("category", ""),
                "has_triggers": len(triggers) > 0,
                "max_score": round(max_score, 1),
                "covered": is_covered,
            })
        
        return {
            "total": len(skills),
            "covered": covered,
            "coverage_rate": round(covered / len(skills) * 100, 1) if skills else 0,
            "not_covered": [r for r in results if not r["covered"]],
            "details": results,
        }
