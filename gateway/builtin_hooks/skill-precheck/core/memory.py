"""
场景记忆持久化模块 — 记录使用历史，优化匹配
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Set


class SceneMemory:
    """场景记忆管理器"""

    def __init__(self, data_dir: str):
        self.stats_file = os.path.join(data_dir, "skill_usage_stats.json")
        self._cache = None

    def load(self) -> Dict:
        """加载场景记忆"""
        if self._cache is not None:
            return self._cache
        
        if not os.path.exists(self.stats_file):
            self._cache = {
                "skills": {},
                "scene_patterns": [],
                "total_recommendations": 0,
            }
            return self._cache
        
        try:
            with open(self.stats_file) as f:
                self._cache = json.load(f)
        except:
            self._cache = {
                "skills": {},
                "scene_patterns": [],
                "total_recommendations": 0,
            }
        
        return self._cache

    def save(self):
        """保存场景记忆"""
        if self._cache is None:
            return
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def increment_recommendations(self):
        """增加推荐计数"""
        stats = self.load()
        stats["total_recommendations"] = stats.get("total_recommendations", 0) + 1
        self.save()

    def record_usage(self, skill_name: str, user_input: str, accepted: bool = True):
        """记录技能使用场景"""
        stats = self.load()
        
        if skill_name not in stats["skills"]:
            stats["skills"][skill_name] = {
                "total_uses": 0,
                "last_used": None,
                "trigger_phrases": [],
                "accepted_count": 0,
                "acceptance_rate": 1.0,
            }
        
        s = stats["skills"][skill_name]
        s["total_uses"] += 1
        s["last_used"] = datetime.now().isoformat()
        
        # 记录触发短语
        input_lower = user_input.lower().strip()
        if input_lower and input_lower not in [p.lower() for p in s["trigger_phrases"]]:
            s["trigger_phrases"].append(input_lower[:100])
            if len(s["trigger_phrases"]) > 20:
                s["trigger_phrases"] = s["trigger_phrases"][-20:]
        
        # 更新接受率
        total = s["total_uses"]
        accepted_count = s.get("accepted_count", 0) + (1 if accepted else 0)
        s["accepted_count"] = accepted_count
        s["acceptance_rate"] = round(accepted_count / total, 2)
        
        # 更新场景模式
        self._update_scene_patterns(stats, skill_name, user_input)
        
        self.save()

    def _update_scene_patterns(self, stats: Dict, skill_name: str, user_input: str):
        """更新场景模式"""
        # 简单的关键词提取
        import re
        keywords = set()
        chinese = re.findall(r'[\u4e00-\u9fff]+', user_input)
        for ch in chinese:
            if len(ch) >= 2:
                keywords.add(ch)
        
        for kw in list(keywords)[:5]:
            found = False
            for pattern in stats["scene_patterns"]:
                if kw in pattern["pattern"] or pattern["pattern"] in kw:
                    if skill_name not in pattern["recommended_skills"]:
                        pattern["recommended_skills"].append(skill_name)
                    pattern["hit_count"] += 1
                    found = True
                    break
            
            if not found:
                stats["scene_patterns"].append({
                    "pattern": kw,
                    "recommended_skills": [skill_name],
                    "hit_count": 1,
                })

    def get_skill_stats(self, skill_name: str) -> Dict:
        """获取某个技能的使用统计"""
        stats = self.load()
        return stats.get("skills", {}).get(skill_name, {})
