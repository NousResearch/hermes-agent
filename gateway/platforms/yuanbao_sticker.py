"""
Yuanbao sticker (TIMFaceElem) support. Ported from yuanbao-openclaw-plugin/src/sticker/.

TIMFaceElem wire format: {"msg_type": "TIMFaceElem", "msg_content": {"index": 0, "data": "<json>"}}
index is always 0 per Yuanbao convention; data is serialised sticker metadata so the receiver
can look up the asset in the emoji pack.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from typing import Optional

# Sticker catalogue – ported from builtin-stickers.json. Every builtin sticker is in
# package 1003, 128x128 png, with name == key; only (name, sticker_id, description) vary.
_STICKERS: list[tuple[str, str, str]] = [
    ("六六六", "278", "666 厉害 牛 棒 绝了 好强 awesome"),
    ("我想开了", "262", "想开 佛系 释怀 顿悟 看淡了 无所谓"),
    ("害羞", "130", "腼腆 不好意思 脸红 娇羞 羞涩 捂脸"),
    ("比心", "252", "笔芯 爱你 爱心手势 love heart 喜欢你"),
    ("委屈", "125", "难过 想哭 可怜巴巴 瘪嘴 受伤 被欺负"),
    ("亲亲", "146", "么么 mua 亲一下 kiss 飞吻 啵"),
    ("酷", "131", "帅 墨镜 cool 高冷 有型 swagger"),
    ("睡", "145", "睡觉 困 zzZ 打盹 躺平 休眠 sleepy"),
    ("发呆", "152", "懵 愣住 放空 呆滞 出神 脑子空白"),
    ("可怜", "157", "卖萌 求饶 委屈巴巴 弱小 拜托 眼巴巴"),
    ("摊手", "200", "无奈 没办法 耸肩 随便 那咋整 whatever"),
    ("头大", "213", "头疼 烦恼 郁闷 难搞 崩溃 一团乱"),
    ("吓", "256", "害怕 惊恐 震惊 吓一跳 恐怖 怂"),
    ("吐血", "203", "无语 崩溃 被雷 内伤 一口老血 屮"),
    ("哼", "185", "傲娇 生气 不满 撇嘴 不理 赌气"),
    ("嘿嘿", "220", "坏笑 猥琐笑 偷笑 憨笑 得意 你懂的"),
    ("头秃", "218", "程序员 加班 焦虑 没头发 秃了 肝爆"),
    ("暗中观察", "221", "窥屏 潜水 偷偷看 角落 围观 屏住呼吸"),
    ("我酸了", "224", "嫉妒 柠檬精 羡慕 吃柠檬 眼红 恰柠檬"),
    ("打call", "246", "应援 加油 支持 喝彩 助威 call"),
    ("庆祝", "251", "祝贺 开心 耶 party 胜利 干杯"),
    ("奋斗", "151", "努力 加油 拼搏 冲 干劲 卷起来"),
    ("惊讶", "143", "震惊 哇 不敢相信 OMG 居然 这么离谱"),
    ("疑问", "144", "问号 不懂 啥 为什么 啥情况 懵逼问"),
    ("仔细分析", "248", "思考 推敲 认真 研究 琢磨 让我想想"),
    ("撅嘴", "184", "嘟嘴 卖萌 不高兴 撒娇 嘴翘"),
    ("泪奔", "199", "大哭 伤心 破防 感动哭 泪流满面 呜呜"),
    ("尊嘟假嘟", "276", "真的假的 真假 可爱问 你骗我 是不是"),
    ("略略略", "113", "调皮 吐舌 不服 略 气死你 鬼脸"),
    ("困", "180", "想睡 倦 打哈欠 睁不开眼 好困啊 sleepy"),
    ("折磨", "181", "难受 痛苦 煎熬 蚌埠住了 受不了 要命"),
    ("抠鼻", "182", "不屑 无聊 淡定 无所谓 鄙视 挖鼻"),
    ("鼓掌", "183", "拍手 叫好 赞同 666 喝彩 掌声"),
    ("斜眼笑", "204", "滑稽 坏笑 doge 意味深长 阴阳怪气 嘿嘿嘿"),
    ("辣眼睛", "216", "看不下去 cringe 毁三观 太丑了 瞎了"),
    ("哦哟", "217", "惊讶 起哄 哇哦 有戏 不简单 哟"),
    ("吃瓜", "222", "围观 看戏 八卦 路人 看热闹 板凳"),
    ("狗头", "225", "doge 保命 开玩笑 滑稽 反讽 懂的都懂"),
    ("敬礼", "227", "salute 尊重 收到 遵命 致敬 报告"),
    ("哦", "231", "知道了 明白 敷衍 嗯 这样啊 收到"),
    ("拿到红包", "236", "红包 谢谢老板 发财 开心 抢到了 欧气"),
    ("牛吖", "239", "牛 厉害 强 666 佩服 大佬"),
    ("贴贴", "272", "抱抱 亲昵 蹭蹭 亲密 靠靠 撒娇贴"),
    ("爱心", "138", "心 love 喜欢你 红心 示爱 么么哒"),
    ("晚安", "170", "好梦 睡了 night 早点休息 安啦 moon"),
    ("太阳", "176", "晴天 早上好 阳光 morning 好天气 日"),
    ("柠檬", "266", "酸 嫉妒 柠檬精 羡慕 我酸 恰柠檬"),
    ("大冤种", "267", "倒霉 吃亏 自嘲 好心没好报 背锅 工具人"),
    ("吐了", "132", "恶心 yue 受不了 嫌弃 想吐 生理不适"),
    ("怒", "134", "生气 愤怒 火大 暴躁 气炸 怼"),
    ("玫瑰", "165", "花 示爱 表白 浪漫 送你花 情人节"),
    ("凋谢", "119", "花谢 失恋 难过 枯萎 心碎 凉了"),
    ("点赞", "159", "赞 认同 好棒 good like 大拇指 顶"),
    ("握手", "164", "合作 你好 商务 hello deal 成交 友好"),
    ("抱拳", "163", "谢谢 失敬 江湖 承让 拜托 有礼"),
    ("ok", "169", "好的 收到 没问题 okay 行 可以 懂了"),
    ("拳头", "174", "加油 干 冲 fight 力量 击拳 硬气"),
    ("鞭炮", "191", "过年 喜庆 爆竹 春节 噼里啪啦 红"),
    ("烟花", "258", "庆典 漂亮 新年 嘭 绽放 节日快乐"),
]

STICKER_MAP: dict[str, dict] = {
    name: {
        "sticker_id": sid, "package_id": "1003", "name": name, "description": desc,
        "width": 128, "height": 128, "formats": "png",
    }
    for name, sid, desc in _STICKERS
}


def get_sticker_by_name(name: str) -> Optional[dict]:
    """完全相等 → name 与查询词互为子串 → description 包含查询词 → search_stickers 最高分；找不到返回 None。"""
    if not name:
        return None
    query = name.strip()
    if query in STICKER_MAP:
        return STICKER_MAP[query]
    for key, sticker in STICKER_MAP.items():
        if query in key or key in query:
            return sticker
    for sticker in STICKER_MAP.values():
        if query in sticker["description"]:
            return sticker
    matches = search_stickers(query, limit=1)
    return matches[0] if matches else None


def get_random_sticker(category: str = None) -> dict:
    """随机贴纸；指定 category 时优先在 description/name 含该词的贴纸中选取。"""
    candidates = [s for s in STICKER_MAP.values() if category in s["description"] or category in s["name"]] if category else []
    return random.choice(candidates or list(STICKER_MAP.values()))


def get_sticker_by_id(sticker_id: str) -> Optional[dict]:
    if not sticker_id:
        return None
    sid = str(sticker_id).strip()
    return next((s for s in STICKER_MAP.values() if s["sticker_id"] == sid), None)


# 模糊搜索（对齐 chatbot-web yuanbao-openclaw-plugin/sticker-cache.ts.searchStickers）

_PUNCT_RE = re.compile(r"[\s\u3000\-_·.,，。!！?？\"“”'‘’、/\\]+")


def _normalize_text(raw: str) -> str:
    return unicodedata.normalize("NFKC", str(raw or "")).strip().lower()


def _compact_text(raw: str) -> str:
    return _PUNCT_RE.sub("", _normalize_text(raw))


def _multiset_char_hit_ratio(needle: str, haystack: str) -> float:
    return sum((Counter(needle) & Counter(haystack)).values()) / len(needle) if needle else 0.0


def _bigram_jaccard(a: str, b: str) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    A, B = ({x[i:i + 2] for i in range(len(x) - 1)} for x in (a, b))
    return len(A & B) / len(A | B)


def _longest_subsequence_ratio(needle: str, haystack: str) -> float:
    if not needle:
        return 0.0
    j = 0
    for ch in haystack:
        if j >= len(needle):
            break
        j += ch == needle[j]
    return j / len(needle)


def _score_field(haystack: str, query: str) -> float:
    hay = _normalize_text(haystack)
    q = _normalize_text(query)
    if not hay or not q:
        return 0.0
    hay_c = _compact_text(haystack)
    q_c = _compact_text(query)
    return max(
        100.0 if hay == q else 0.0,
        92 + min(6, len(q)) if q in hay else 0.0,
        88.0 if len(q) >= 2 and hay.startswith(q) else 0.0,
        86.0 if q_c and q_c in hay_c else 0.0,
        _multiset_char_hit_ratio(q_c, hay_c) * 62,
        _bigram_jaccard(q_c, hay_c) * 58,
        _longest_subsequence_ratio(q_c, hay_c) * 52,
        68.0 if len(q) == 1 and q in hay else 0.0,
    )


def search_stickers(query: str, limit: int = 10) -> list[dict]:
    """内置贴纸表按模糊匹配排序返回前 N 条。

    评分综合 name/description 的子串、字符多重集覆盖、bigram Jaccard、子序列比例，
    name 权重高于 description（×0.88）；sticker_id 精确/子串命中另计。空 query 按表顺序返回。
    """
    safe_limit = max(1, min(500, int(limit) if limit else 10))
    q_norm = _normalize_text(query)
    if not query or not q_norm:
        return list(STICKER_MAP.values())[:safe_limit]
    scored: list[tuple[float, dict]] = []
    for sticker in STICKER_MAP.values():
        sid_norm = _normalize_text(sticker["sticker_id"].strip())
        id_s = 100.0 if sid_norm == q_norm else 84.0 if sid_norm and q_norm in sid_norm else 0.0
        scored.append((max(_score_field(sticker["name"], query), _score_field(sticker["description"], query) * 0.88, id_s), sticker))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][0] if scored else 0
    if top <= 0:
        return [s for _, s in scored[:safe_limit]]
    floor = 18.0 if top >= 22 else max(10.0, top * 0.5) if top >= 12 else max(6.0, top * 0.35)
    filtered = [pair for pair in scored if pair[0] >= floor]
    return [s for _, s in (filtered or scored)[:safe_limit]]


def build_face_msg_body(face_index: int, face_type: int = 1, data: Optional[str] = None) -> list:
    """TIMFaceElem 消息体。Yuanbao 约定 index 固定 0（服务端通过 data 字段识别表情）；face_index > 0 视为旧版 QQ
    表情 ID。face_type 为兼容旧接口保留，不影响 wire format。data 为 None 时仅传 index。"""
    msg_content: dict = {"index": face_index}
    if data is not None:
        msg_content["data"] = data
    return [{"msg_type": "TIMFaceElem", "msg_content": msg_content}]


def build_sticker_msg_body(sticker: dict) -> list:
    """从 STICKER_MAP 的 sticker dict 构造 TIMFaceElem 消息体（data 字段与原始 JS 插件一致）。"""
    data_payload = json.dumps(
        {
            "sticker_id": sticker["sticker_id"], "package_id": sticker["package_id"],
            "width": sticker.get("width", 128), "height": sticker.get("height", 128),
            "formats": sticker.get("formats", "png"), "name": sticker["name"],
        },
        ensure_ascii=False, separators=(",", ":"),
    )
    return build_face_msg_body(face_index=0, data=data_payload)
