"""
四维匹配引擎 — 词法 + 语义 + 场景 + 类别
"""

import re
from typing import List, Dict, Set, Tuple


class SkillMatcher:
    """四维技能匹配引擎"""
    
    # 匹配权重
    WEIGHT_LEXICAL = 0.40
    WEIGHT_SEMANTIC = 0.25
    WEIGHT_SCENE = 0.20
    WEIGHT_CATEGORY = 0.15

    # 短查询自动提升因子（1-2个词 → 减轻权重稀释）
    SHORT_QUERY_BOOST = 1.6

    def __init__(self, synonyms: Dict[str, List[str]]):
        self.synonyms = synonyms

    def score(
        self, 
        input_words: Set[str], 
        skill: Dict,
        stats: Dict,
    ) -> Tuple[float, Dict, List[str]]:
        """
        对单个 skill 进行四维评分
        
        Returns:
            (total_score, details_dict, reasons_list)
        """
        lex_score, lex_reasons = self._match_lexical(input_words, skill)
        sem_score = self._match_semantic(input_words, skill)
        sce_score = self._match_scene(input_words, skill["name"], stats)
        cat_score = self._match_category(input_words, skill)
        
        total = (
            lex_score * self.WEIGHT_LEXICAL +
            sem_score * self.WEIGHT_SEMANTIC +
            sce_score * self.WEIGHT_SCENE +
            cat_score * self.WEIGHT_CATEGORY
        )
        
        # ═══ 短查询自动提升 ═══
        # 1-2个词的查询容易被权重稀释（如 "生图" trigger匹配25→加权后仅10分）
        # 此时保留更多原始分，避免低于40分阈值
        raw_word_count = len([w for w in input_words if len(w) >= 2])
        if raw_word_count <= 2 and lex_score >= 20:
            total = total * self.SHORT_QUERY_BOOST
        
        details = {
            "lexical": round(lex_score, 1),
            "semantic": round(sem_score, 1),
            "scene": round(sce_score, 1),
            "category": round(cat_score, 1),
        }
        
        return total, details, lex_reasons[:5]

    def _match_lexical(self, input_words: Set[str], skill: Dict) -> Tuple[float, List[str]]:
        """词法匹配：triggers + name + description + tags + 同义词反向匹配"""
        score = 0
        reasons = []
        
        skill_name = skill["name"].lower()
        triggers = [t.lower() for t in skill.get("triggers", [])]
        tags = [t.lower() for t in skill.get("tags", [])]
        match_text = skill.get("match_text", "").lower()
        desc = skill.get("full_description", "").lower()
        
        # 构建词列表 + 中文组合词拆解
        word_list = list(input_words)
        extra_words = set()
        multi_word_syns = set()
        
        for w in word_list:
            # 中文组合词拆解
            if len(w) >= 3 and all('\u4e00' <= c <= '\u9fff' for c in w):
                for i in range(len(w)):
                    for j in range(2, min(4, len(w) - i + 1)):
                        sub = w[i:i+j]
                        if len(sub) >= 2:
                            extra_words.add(sub)
            # 多词同义词值拆解
            if ' ' in w:
                for part in w.split():
                    if len(part) >= 2:
                        multi_word_syns.add(part.lower())
        
        word_list.extend(list(extra_words))
        word_list.extend(list(multi_word_syns))
        
        # 同义词反向匹配（改进：区分精确匹配和宽泛匹配）
        for word in list(input_words):
            if word in self.synonyms:
                for syn in self.synonyms[word]:
                    syn_lower = syn.lower()
                    if len(syn_lower) < 2:
                        continue
                    # 精确匹配：在 name/trigger/tags 中
                    if syn_lower in skill_name or syn_lower in str(triggers) or syn_lower in str(tags):
                        score += 25
                        if f"同义词'{word}'→'{syn_lower}'" not in str(reasons):
                            reasons.append(f"同义词'{word}'→'{syn_lower}'")
                    # 宽泛匹配：只在 description/match_text 中（风险更低）
                    elif syn_lower in desc or syn_lower in match_text:
                        score += 12
                        if f"同义词'{word}'→'{syn_lower}'" not in str(reasons):
                            reasons.append(f"同义词(描述)'{word}'→'{syn_lower}'")
        
        # 逐词遍历
        for word in word_list:
            w = word.lower()
            if len(w) < 2:
                continue
            
            if w == skill_name or skill_name in w:
                score += 30
                if "name匹配" not in str(reasons):
                    reasons.append(f"name匹配'{w}'")
            
            if len(w) >= 3 and w in skill_name:
                score += 20
                if f"name部分'{w}'" not in reasons:
                    reasons.append(f"name部分'{w}'")
            
            if w in triggers:
                score += 25
                if f"trigger'{w}'" not in reasons:
                    reasons.append(f"trigger'{w}'")
            
            if w in tags:
                score += 15
                if f"tag'{w}'" not in reasons:
                    reasons.append(f"tag'{w}'")
            
            if len(w) >= 2 and w in desc:
                score += 8
                if f"描述'{w}'" not in reasons:
                    reasons.append(f"描述'{w}'")
            
            if len(w) >= 2 and w in match_text:
                score += 3
        
        return min(score, 100), reasons

    def _match_semantic(self, input_words: Set[str], skill: Dict) -> float:
        """语义匹配：基于 description + body_keywords"""
        desc = skill.get("full_description", "").lower()
        body_kws = set(skill.get("body_keywords", []))
        
        score = 0
        overlap = 0
        
        for word in input_words:
            w = word.lower()
            if len(w) < 2:
                continue
            
            if w in desc:
                score += 10
                overlap += 1
            if w in body_kws:
                score += 5
                overlap += 1
        
        total = len(input_words)
        if total > 0:
            ratio = overlap / total
            score = min(int(score * (0.5 + ratio * 0.5)), 100)
        
        return score

    def _match_scene(self, input_words: Set[str], skill_name: str, stats: Dict) -> float:
        """场景匹配：基于历史使用模式"""
        scene_patterns = stats.get("scene_patterns", [])
        
        score = 0
        for pattern in scene_patterns:
            pat = pattern["pattern"].lower()
            for word in input_words:
                if len(word) < 2:
                    continue
                if word.lower() in pat or pat in word.lower():
                    if skill_name in pattern.get("recommended_skills", []):
                        score += min(pattern.get("hit_count", 1) * 3, 30)
        
        # 使用频率加分
        skill_stats = stats.get("skills", {}).get(skill_name, {})
        total_uses = skill_stats.get("total_uses", 0)
        score += min(total_uses * 2, 20)
        
        return min(score, 100)

    def _match_category(self, input_words: Set[str], skill: Dict) -> float:
        """类别匹配：基于 category + tags"""
        score = 0
        category = skill.get("category", "").lower()
        tags = [t.lower() for t in skill.get("tags", [])]
        
        for word in input_words:
            w = word.lower()
            if len(w) < 2:
                continue
            if w in category:
                score += 20
            for tag in tags:
                if w in tag or tag in w:
                    score += 15
        
        return min(score, 100)
