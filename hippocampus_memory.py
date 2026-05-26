"""
海马体记忆系统 - 类比 SWRs 机制

核心机制：
1. 临时记忆（海马体）→ 长期记忆（新皮层）
2. 重要性评分（SWRs 选择）
3. 记忆巩固和回放
4. 记忆检索和复用
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    importance: float  # 0-1, 类比 SWRs 选择强度
    access_count: int
    last_accessed: str
    created_at: str
    tags: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def fingerprint(self) -> str:
        """记忆指纹"""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return content_hash[:16]


class HippocampusMemorySystem:
    """海马体记忆系统"""
    
    def __init__(self, db_path: str):
        """
        初始化记忆系统
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self.short_term_buffer = []  # 短期记忆缓冲区（海马体）
        self.consolidation_threshold = 0.6  # 巩固阈值
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 长期记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id TEXT PRIMARY KEY,
                fingerprint TEXT UNIQUE,
                content TEXT,
                memory_type TEXT,
                importance REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # 索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON long_term_memory(importance DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint ON long_term_memory(fingerprint)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON long_term_memory(memory_type)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ 海马体记忆系统初始化: {self.db_path}")
    
    def encode(
        self,
        content: str,
        memory_type: str = "episodic",
        tags: Optional[List[str]] = None
    ) -> Memory:
        """
        编码新记忆（海马体编码）
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            tags: 标签
            
        Returns:
            记忆对象
        """
        # 计算重要性（类比 SWRs 选择）
        importance = self._calculate_importance(content, memory_type)
        
        memory = Memory(
            id=f"mem_{datetime.utcnow().timestamp()}",
            content=content,
            memory_type=memory_type,
            importance=importance,
            access_count=0,
            last_accessed=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            tags=tags or []
        )
        
        # 添加到短期记忆缓冲区
        self.short_term_buffer.append(memory)
        
        logger.info(f"🧠 编码记忆: {memory.id} (重要性: {importance:.2f})")
        
        # 如果重要性足够高，立即巩固
        if importance >= self.consolidation_threshold:
            self._consolidate_memory(memory)
        
        return memory
    
    def consolidate_batch(self, batch_size: int = 10):
        """
        批量巩固记忆（类比睡眠回放）
        
        Args:
            batch_size: 批量大小
        """
        if not self.short_term_buffer:
            return
        
        # 按重要性排序
        self.short_term_buffer.sort(key=lambda m: m.importance, reverse=True)
        
        # 巩固最重要的记忆
        to_consolidate = self.short_term_buffer[:batch_size]
        
        for memory in to_consolidate:
            self._consolidate_memory(memory)
        
        # 从缓冲区移除
        self.short_term_buffer = self.short_term_buffer[batch_size:]
        
        logger.info(f"💤 批量巩固: {len(to_consolidate)} 条记忆")
    
    def _consolidate_memory(self, memory: Memory) -> bool:
        """
        巩固单条记忆到长期存储（海马体 → 新皮层）
        
        Args:
            memory: 记忆对象
            
        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查指纹是否已存在
            cursor.execute("""
                SELECT id FROM long_term_memory WHERE fingerprint = ?
            """, (memory.fingerprint,))
            
            if cursor.fetchone():
                conn.close()
                logger.debug(f"⏭️ 记忆已存在: {memory.fingerprint}")
                return False
            
            # 插入长期记忆
            cursor.execute("""
                INSERT INTO long_term_memory (
                    id, fingerprint, content, memory_type, importance,
                    access_count, last_accessed, created_at, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.fingerprint,
                memory.content,
                memory.memory_type,
                memory.importance,
                memory.access_count,
                memory.last_accessed,
                memory.created_at,
                json.dumps(memory.tags),
                json.dumps(memory.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 巩固记忆: {memory.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 巩固失败: {e}")
            return False
    
    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Memory]:
        """
        回忆记忆（检索）
        
        Args:
            query: 查询关键词
            memory_type: 记忆类型过滤
            limit: 返回数量
            
        Returns:
            记忆列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = """
            SELECT * FROM long_term_memory
            WHERE content LIKE ?
        """
        params = [f"%{query}%"]
        
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        
        sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        # 更新访问计数
        for row in rows:
            cursor.execute("""
                UPDATE long_term_memory
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), row[0]))
        
        conn.commit()
        conn.close()
        
        memories = [self._row_to_memory(row) for row in rows]
        
        logger.info(f"🔍 回忆: 找到 {len(memories)} 条记忆")
        
        return memories
    
    def get_important_memories(self, limit: int = 10) -> List[Memory]:
        """
        获取最重要的记忆
        
        Args:
            limit: 返回数量
            
        Returns:
            记忆列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM long_term_memory
            ORDER BY importance DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def forget_old_memories(self, days: int = 30, min_importance: float = 0.3):
        """
        遗忘旧记忆（类比记忆衰减）
        
        Args:
            days: 天数阈值
            min_importance: 最小重要性（低于此值才遗忘）
        """
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM long_term_memory
            WHERE last_accessed < ?
            AND importance < ?
        """, (threshold_date, min_importance))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"🗑️ 遗忘: {deleted} 条旧记忆")
    
    def _calculate_importance(self, content: str, memory_type: str) -> float:
        """
        计算记忆重要性（类比 SWRs 选择强度）
        
        考虑因素：
        1. 内容长度
        2. 记忆类型
        3. 关键词密度
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            
        Returns:
            重要性分数（0-1）
        """
        importance = 0.5
        
        # 内容长度加分
        if len(content) > 100:
            importance += 0.2
        
        # 记忆类型加分
        if memory_type == "semantic":  # 语义记忆更重要
            importance += 0.2
        elif memory_type == "procedural":  # 程序记忆也重要
            importance += 0.15
        
        # 关键词加分
        keywords = ["important", "remember", "key", "critical", "重要", "关键"]
        if any(kw in content.lower() for kw in keywords):
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _row_to_memory(self, row) -> Memory:
        """将数据库行转换为 Memory 对象"""
        return Memory(
            id=row[0],
            content=row[2],
            memory_type=row[3],
            importance=row[4],
            access_count=row[5],
            last_accessed=row[6],
            created_at=row[7],
            tags=json.loads(row[8]) if row[8] else [],
            metadata=json.loads(row[9]) if row[9] else {}
        )


# 便捷函数
def get_memory_system(db_path: Optional[str] = None) -> HippocampusMemorySystem:
    """获取记忆系统实例"""
    if db_path is None:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        db_path = str(Path(hermes_home) / "hippocampus_memory.db")
    
    return HippocampusMemorySystem(db_path)


if __name__ == "__main__":
    # 测试
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print("=" * 70)
    print("海马体记忆系统测试")
    print("=" * 70)
    print()
    
    # 创建记忆系统
    memory_system = HippocampusMemorySystem(db_path)
    
    # 编码记忆
    print("🧠 编码记忆...")
    memories = [
        ("Python is a programming language", "semantic", ["python", "programming"]),
        ("I learned about neural networks today", "episodic", ["learning", "ai"]),
        ("Remember to use git commit before push", "procedural", ["git", "workflow"]),
        ("The weather is nice", "episodic", ["weather"]),
    ]
    
    for content, mem_type, tags in memories:
        mem = memory_system.encode(content, mem_type, tags)
        print(f"  • {content[:50]}... (重要性: {mem.importance:.2f})")
    print()
    
    # 批量巩固
    print("💤 批量巩固...")
    memory_system.consolidate_batch(batch_size=10)
    print()
    
    # 回忆记忆
    print("🔍 回忆记忆...")
    query = "python"
    recalled = memory_system.recall(query, limit=3)
    print(f"  查询: '{query}'")
    for mem in recalled:
        print(f"  • {mem.content[:50]}... (重要性: {mem.importance:.2f}, 访问: {mem.access_count})")
    print()
    
    # 获取重要记忆
    print("⭐ 最重要的记忆:")
    important = memory_system.get_important_memories(limit=3)
    for i, mem in enumerate(important, 1):
        print(f"  {i}. {mem.content[:50]}... (重要性: {mem.importance:.2f})")
    print()
    
    print("=" * 70)
    print("✅ 海马体记忆系统测试完成")
    print("=" * 70)
