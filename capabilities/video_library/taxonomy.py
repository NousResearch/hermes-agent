"""Controlled vocabularies for semantic video-shot indexing."""

from __future__ import annotations

from collections.abc import Iterable


BEEF_NOODLE_V1: dict[str, set[str]] = {
    "主体": {"厨师", "顾客", "员工", "牛肉", "面条", "汤锅", "成品面", "门店", "餐桌", "配菜"},
    "场景": {"后厨", "前厅", "门头", "餐桌", "街景", "收银台", "备餐区"},
    "动作": {"和面", "醒面", "拉面", "切肉", "下锅", "煮面", "浇汤", "撒香菜", "装碗", "端碗", "吃面", "出餐"},
    "工序": {"和面", "醒面", "拉面", "煮面", "切肉", "备菜", "装碗", "出餐"},
    "景别": {"特写", "近景", "中景", "全景", "环境远景"},
    "机位": {"平视", "俯拍", "仰拍", "第一视角"},
    "运镜": {"固定", "推进", "拉远", "横移", "跟拍", "环绕", "手持"},
    "画面特点": {"热气", "汤汁", "油亮", "火焰", "慢动作", "烟火气", "人流", "招牌"},
    "情绪": {"食欲", "温暖", "忙碌", "专业", "热闹", "治愈", "真实", "亲切"},
    "用途": {"开头钩子", "产品证明", "制作过程", "品牌故事", "门店环境", "顾客体验", "结尾召唤"},
    "音频": {"有效对白", "环境声", "噪声", "静音", "适合保留原声"},
    "门店信息": {"门头", "Logo", "价格", "员工", "顾客", "地理标识"},
}

ALIASES: dict[str, dict[str, str]] = {
    "动作": {"抻面": "拉面", "下面": "下锅", "放面": "下锅", "盛面": "装碗"},
    "工序": {"抻面": "拉面", "下面": "煮面", "盛面": "装碗"},
    "用途": {"品质展示": "产品证明", "过程展示": "制作过程", "转化": "结尾召唤"},
}

TAXONOMIES = {"beef-noodle-v1": BEEF_NOODLE_V1}


def normalize_controlled(dimension: str, values: Iterable[str], taxonomy: str) -> list[str]:
    try:
        vocabulary = TAXONOMIES[taxonomy][dimension]
    except KeyError as exc:
        raise ValueError(f"unknown taxonomy dimension: {taxonomy}/{dimension}") from exc
    aliases = ALIASES.get(dimension, {})
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = aliases.get(str(value).strip(), str(value).strip())
        if normalized in vocabulary:
            tag = f"{dimension}/{normalized}"
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
    return result


def taxonomy_prompt(taxonomy: str) -> str:
    try:
        dimensions = TAXONOMIES[taxonomy]
    except KeyError as exc:
        raise ValueError(f"unknown video taxonomy: {taxonomy}") from exc
    return "\n".join(f"- {name}: {', '.join(sorted(values))}" for name, values in dimensions.items())


__all__ = ["BEEF_NOODLE_V1", "normalize_controlled", "taxonomy_prompt"]
