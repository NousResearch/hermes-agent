"""
技能索引构建模块 — 扫描技能目录，构建完整索引
"""

import os
import re
import json
import glob
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional

from .synonyms import SYNONYMS


class SkillIndexer:
    """技能索引构建器"""
    
    def __init__(self, skills_dir: str, data_dir: str):
        self.skills_dir = skills_dir
        self.index_file = os.path.join(data_dir, "skill_full_index.json")
        self._skills: List[Dict] = []
        
        # 缓存
        self._synonyms = SYNONYMS

    def extract_keywords(self, text: str, min_len: int = 2) -> Set[str]:
        """提取中英文关键词 — N-gram + jieba 分词双引擎"""
        words = set()
        
        # 英文单词
        eng = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{1,}', text.lower())
        words.update(eng)
        
        # 中文词组（2-4 字）—— N-gram 引擎（保持原有逻辑）
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        for ch in chinese:
            if len(ch) >= min_len:
                words.add(ch.lower())
            for i in range(len(ch)):
                for j in range(2, min(5, len(ch) - i + 1)):
                    sub = ch[i:i+j]
                    if i == 0 or i + j == len(ch):
                        words.add(sub.lower())
        
        # ── jieba 分词引擎（追加，不替换 N-gram）──
        try:
            import jieba
            jieba_words = set()
            for ch in chinese:
                for w in jieba.cut(ch):
                    w = w.strip()
                    if len(w) >= min_len:
                        jieba_words.add(w.lower())
            words.update(jieba_words)
        except ImportError:
            pass  # jieba 未安装时静默降级
        
        return words

    def expand_with_synonyms(self, words: Set[str]) -> Set[str]:
        """同义词扩展"""
        expanded = set(words)
        for word in list(words):
            if word in self._synonyms:
                expanded.update(s.strip().lower() for s in self._synonyms[word])
            # 反向查找
            for key, syns in self._synonyms.items():
                if word in [s.lower() for s in syns]:
                    expanded.add(key.lower())
                    expanded.update(s.lower() for s in syns)
        return expanded

    def _parse_frontmatter(self, content: str) -> dict:
        """解析 YAML frontmatter"""
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 2:
                try:
                    return yaml.safe_load(parts[1]) or {}
                except:
                    pass
        return {}

    def _extract_triggers(self, frontmatter: dict) -> List[str]:
        """提取 triggers，支持多种格式"""
        triggers = []
        raw = frontmatter.get("triggers", [])
        if isinstance(raw, list):
            triggers = raw
        elif isinstance(raw, str):
            triggers = [raw]
        
        # 也从 metadata.hermes.tags 中提取
        meta = frontmatter.get("metadata", {})
        if isinstance(meta, dict):
            hermes = meta.get("hermes", {})
            if isinstance(hermes, dict):
                tags = hermes.get("tags", [])
                if isinstance(tags, list):
                    triggers.extend(tags)
        
        return [str(t).lower() for t in triggers if t]

    def build(self) -> int:
        """构建完整技能索引"""
        if not os.path.exists(self.skills_dir):
            print(f"⚠️  技能目录不存在: {self.skills_dir}")
            return 0
        
        sk_files = glob.glob(os.path.join(self.skills_dir, '**/SKILL.md'), recursive=True)
        
        index = []
        for f in sk_files:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                
                frontmatter = self._parse_frontmatter(content)
                body = content
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        body = parts[2]
                
                name = frontmatter.get('name', os.path.basename(os.path.dirname(f)))
                description = frontmatter.get('description', '')
                triggers = self._extract_triggers(frontmatter)
                version = frontmatter.get('version', '0.0.0')
                
                rel_path = os.path.relpath(f, self.skills_dir)
                parts_path = rel_path.split(os.sep)
                category = parts_path[0] if len(parts_path) >= 2 else "general"
                
                # tags
                meta = frontmatter.get("metadata", {})
                hermes_meta = meta.get("hermes", {}) if isinstance(meta, dict) else {}
                tags = hermes_meta.get("tags", []) if isinstance(hermes_meta, dict) else []
                related = hermes_meta.get("related_skills", []) if isinstance(hermes_meta, dict) else []
                
                # 正文关键词
                body_text = body[:800].lower()
                body_keywords = list(self.extract_keywords(body_text))
                
                # 构建匹配文本
                match_text = f"{name} {description} {' '.join(triggers)} {' '.join(tags)}".lower()
                
                index.append({
                    "name": name,
                    "description": description[:500],
                    "triggers": triggers,
                    "tags": tags,
                    "related_skills": related,
                    "category": category,
                    "version": version,
                    "path": f,
                    "match_text": match_text,
                    "body_keywords": body_keywords,
                    "full_description": description[:1000],
                })
            except Exception as e:
                print(f"⚠️  跳过 {f}: {e}")
        
        self._skills = index
        
        # 持久化
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, 'w') as f:
            json.dump({
                "built_at": datetime.now().isoformat(),
                "count": len(index),
                "skills": index,
            }, f, indent=2, ensure_ascii=False)
        
        return len(index)

    def _skills_dir_mtime(self) -> float:
        """获取技能目录的最新修改时间（递归扫描）"""
        latest = 0.0
        try:
            for root, dirs, files in os.walk(self.skills_dir):
                for f in files:
                    if f == "SKILL.md":
                        m = os.path.getmtime(os.path.join(root, f))
                        if m > latest:
                            latest = m
                # 也检查目录本身的修改时间（新增/删除子目录）
                for d in dirs:
                    m = os.path.getmtime(os.path.join(root, d))
                    if m > latest:
                        latest = m
        except OSError:
            pass
        return latest

    def load_or_build(self) -> List[Dict]:
        """加载或构建索引 — 支持自动检测目录变更"""
        if self._skills:
            return self._skills
        
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file) as f:
                    data = json.load(f)
                self._skills = data.get("skills", [])
                if self._skills:
                    # 检查索引文件的构建时间
                    built_at_str = data.get("built_at", "")
                    if built_at_str:
                        try:
                            from datetime import datetime
                            built_at = datetime.fromisoformat(built_at_str).timestamp()
                            # 如果目录在索引构建之后有变更 → 强制重建
                            dir_mtime = self._skills_dir_mtime()
                            if dir_mtime > built_at:
                                print(f"[indexer] Skills directory changed after index build, rebuilding...")
                                self._skills = []  # clear cached skills
                                self.build()
                                return self._skills
                        except (ValueError, TypeError):
                            pass
                    return self._skills
            except:
                pass
        
        self.build()
        return self._skills

    def get_skills(self) -> List[Dict]:
        """获取技能列表"""
        return self._skills
