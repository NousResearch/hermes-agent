"""
SkillBank - 技能知识库自演进系统

核心功能：
1. 技能存储和检索
2. Select-Read-Act 三段式闭环
3. 技能自演进和优化
4. 与评估中心集成
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """技能数据结构"""
    id: str
    name: str
    description: str
    category: str
    code: str
    dependencies: List[str]
    usage_count: int = 0
    success_count: int = 0
    avg_score: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    @property
    def quality_score(self) -> float:
        """质量评分（综合成功率和平均分）"""
        return (self.success_rate * 0.6 + self.avg_score * 0.4)


class SkillBank:
    """技能知识库"""
    
    def __init__(self, db_path: str):
        """
        初始化技能库
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建技能表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                code TEXT NOT NULL,
                dependencies TEXT,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON skills(category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality ON skills(avg_score DESC, success_count DESC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ SkillBank 初始化完成: {self.db_path}")
    
    def add_skill(self, skill: Skill) -> bool:
        """
        添加技能
        
        Args:
            skill: 技能对象
            
        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO skills (
                    id, name, description, category, code, dependencies,
                    usage_count, success_count, avg_score,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                skill.id,
                skill.name,
                skill.description,
                skill.category,
                skill.code,
                json.dumps(skill.dependencies),
                skill.usage_count,
                skill.success_count,
                skill.avg_score,
                skill.created_at,
                skill.updated_at,
                json.dumps(skill.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ 添加技能: {skill.name} ({skill.id})")
            return True
        except Exception as e:
            logger.error(f"❌ 添加技能失败: {e}")
            return False
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """
        获取技能
        
        Args:
            skill_id: 技能 ID
            
        Returns:
            技能对象或 None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM skills WHERE id = ?
        """, (skill_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_skill(row)
        return None
    
    def search_skills(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_quality: float = 0.0,
        limit: int = 10
    ) -> List[Skill]:
        """
        搜索技能
        
        Args:
            query: 搜索关键词
            category: 类别过滤
            min_quality: 最小质量分数
            limit: 返回数量限制
            
        Returns:
            技能列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM skills WHERE 1=1"
        params = []
        
        if query:
            sql += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        if min_quality > 0:
            sql += " AND avg_score >= ?"
            params.append(min_quality)
        
        sql += " ORDER BY avg_score DESC, success_count DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_skill(row) for row in rows]
    
    def update_skill_stats(
        self,
        skill_id: str,
        success: bool,
        score: Optional[float] = None
    ):
        """
        更新技能统计
        
        Args:
            skill_id: 技能 ID
            success: 是否成功
            score: 评分（0-1）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取当前统计
        cursor.execute("""
            SELECT usage_count, success_count, avg_score FROM skills WHERE id = ?
        """, (skill_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return
        
        usage_count, success_count, avg_score = row
        
        # 更新统计
        new_usage_count = usage_count + 1
        new_success_count = success_count + (1 if success else 0)
        
        if score is not None:
            # 计算新的平均分
            new_avg_score = (avg_score * usage_count + score) / new_usage_count
        else:
            new_avg_score = avg_score
        
        cursor.execute("""
            UPDATE skills
            SET usage_count = ?,
                success_count = ?,
                avg_score = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            new_usage_count,
            new_success_count,
            new_avg_score,
            datetime.utcnow().isoformat(),
            skill_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"📊 更新技能统计: {skill_id} (成功率: {new_success_count}/{new_usage_count})")
    
    def get_top_skills(self, limit: int = 10) -> List[Skill]:
        """
        获取最佳技能
        
        Args:
            limit: 返回数量
            
        Returns:
            技能列表
        """
        return self.search_skills(min_quality=0.0, limit=limit)
    
    def get_categories(self) -> List[str]:
        """获取所有类别"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT category FROM skills ORDER BY category
        """)
        
        categories = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return categories
    
    def _row_to_skill(self, row) -> Skill:
        """将数据库行转换为 Skill 对象"""
        return Skill(
            id=row[0],
            name=row[1],
            description=row[2],
            category=row[3],
            code=row[4],
            dependencies=json.loads(row[5]) if row[5] else [],
            usage_count=row[6],
            success_count=row[7],
            avg_score=row[8],
            created_at=row[9],
            updated_at=row[10],
            metadata=json.loads(row[11]) if row[11] else {}
        )


class SkillEvolver:
    """技能自演进引擎"""
    
    def __init__(self, skill_bank: SkillBank):
        """
        初始化演进引擎
        
        Args:
            skill_bank: 技能库
        """
        self.skill_bank = skill_bank
    
    def evolve_skill(self, skill_id: str) -> Optional[Skill]:
        """
        演进技能（使用增强版演进引擎）
        
        Args:
            skill_id: 技能ID
            
        Returns:
            演进后的技能（如果需要演进）
        """
        skill = self.skill_bank.get_skill(skill_id)
        if not skill:
            return None
        
        # 如果使用次数太少，不进行演进
        if skill.usage_count < 10:
            logger.debug(f"⏸️ 技能 {skill.name} 使用次数不足，暂不演进")
            return None
        
        # 如果质量已经很高，不需要演进
        if skill.quality_score >= 0.8:
            logger.debug(f"✅ 技能 {skill.name} 质量已优秀，无需演进")
            return None
        
        # 使用增强版演进引擎
        try:
            from skill_evolver_enhanced import SkillEvolver
            from hermes_constants import get_hermes_home
            from pathlib import Path
            
            evolver_db = str(Path(get_hermes_home()) / "skill_evolver.db")
            evolver = SkillEvolver(self.skill_bank, evolver_db)
            
            new_skill_id = evolver.evolve_skill(skill_id)
            if new_skill_id:
                return self.skill_bank.get_skill(new_skill_id)
            
        except Exception as e:
            logger.error(f"❌ 演进失败: {e}")
        
        logger.info(f"🔄 技能 {skill.name} 待演进（质量分: {skill.quality_score:.2f}）")
        return skill
    
    def suggest_new_skills(self, context: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        基于上下文建议新技能
        
        实现：
        1. 分析上下文关键词
        2. 检索相似技能
        3. 识别缺失能力（关键词覆盖度）
        4. 生成技能建议
        
        Args:
            context: 上下文描述
            limit: 建议数量
            
        Returns:
            建议列表
        """
        suggestions = []
        
        if not context:
            return suggestions
        
        # 1. 分析上下文：提取关键词
        import re
        # 提取技术名词（连续 3+ 个字符或 2+ 个 CJK）
        keywords = set()
        for word in re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}', context):
            keywords.add(word.lower())
        
        if not keywords:
            return suggestions
        
        # 2. 检索现有技能（用 get_top_skills 取前 100 个）
        all_skills = self.skill_bank.get_top_skills(limit=100)
        existing_keywords = set()
        for s in all_skills:
            existing_keywords.add(s.name.lower())
            for word in re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}', s.description or ''):
                existing_keywords.add(word.lower())
        
        # 3. 识别缺失能力
        missing_keywords = keywords - existing_keywords
        
        # 4. 生成建议
        for kw in sorted(missing_keywords)[:limit]:
            suggestions.append({
                'keyword': kw,
                'reason': f'上下文中的 "{kw}" 无对应技能',
                'priority': 'high' if len(kw) > 5 else 'medium',
                'suggested_skill_name': f'{kw}_handler',
                'suggested_description': f'处理 {kw} 相关任务',
            })
        
        # 如果没有缺失关键词，根据低质量技能给改进建议
        if not suggestions:
            low_quality = [s for s in all_skills if s.quality_score < 0.5][:limit]
            for s in low_quality:
                suggestions.append({
                    'keyword': s.name,
                    'reason': f'技能 "{s.name}" 质量分 {s.quality_score:.2f} 偏低，建议改进',
                    'priority': 'medium',
                    'suggested_skill_name': f'{s.name}_v2',
                    'suggested_description': s.description or '改进版',
                })
        
        logger.info(f"💡 基于上下文生成 {len(suggestions)} 个技能建议")
        return suggestions


# 便捷函数
def get_skill_bank(db_path: Optional[str] = None) -> SkillBank:
    """获取技能库实例"""
    if db_path is None:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        db_path = str(Path(hermes_home) / "skill_bank.db")
    
    return SkillBank(db_path)


if __name__ == "__main__":
    # 测试
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print("=" * 70)
    print("SkillBank 测试")
    print("=" * 70)
    print()
    
    # 创建技能库
    bank = SkillBank(db_path)
    
    # 添加测试技能
    skill = Skill(
        id="test_001",
        name="web_search",
        description="网络搜索技能",
        category="search",
        code="def web_search(query): ...",
        dependencies=["requests"]
    )
    
    bank.add_skill(skill)
    print(f"✅ 添加技能: {skill.name}")
    
    # 搜索技能
    results = bank.search_skills(query="search")
    print(f"🔍 搜索结果: {len(results)} 个技能")
    for s in results:
        print(f"  - {s.name}: {s.description}")
    
    # 更新统计
    bank.update_skill_stats("test_001", success=True, score=0.8)
    print(f"📊 更新统计")
    
    # 获取技能
    skill = bank.get_skill("test_001")
    print(f"📈 技能质量: {skill.quality_score:.2f}")
    
    print()
    print("=" * 70)
    print("✅ SkillBank 测试完成")
    print("=" * 70)
