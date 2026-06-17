"""
Field extraction for pharma compliance visit descriptions.

Extracts 17 structured fields from free-text visit descriptions and
auto-detects the task type (药店拜访 / 医疗机构拜访 / 学术推广).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The 17 extractable fields for pharma compliance
FIELD_NAMES = [
    "task_type",          # 任务类型：药店拜访/医疗机构拜访/学术推广
    "org_name",           # 药店名/医院名/机构名
    "org_address",        # 机构地址
    "contact_person",     # 拜访对象（店长/医生/主任等）
    "contact_title",      # 拜访对象职务
    "visit_date",         # 拜访日期
    "visit_time",         # 拜访时间
    "products",           # 涉及产品（逗号分隔）
    "topic",              # 拜访主题/会议主题
    "content_summary",    # 拜访内容摘要
    "competitor_info",    # 竞品信息
    "feedback",           # 客户反馈
    "next_steps",         # 下一步计划
    "attendee_count",     # 参会人数
    "meeting_duration",   # 会议时长（分钟）
    "materials_used",     # 使用资料
    "notes",              # 备注
]

# Task type detection rules — keyword-based fallback when DeepSeek unavailable
TASK_TYPE_RULES: List[Tuple[str, List[str]]] = [
    (
        "学术推广",
        ["科室会", "学术", "推广会", "讲座", "培训", "沙龙",
         "研讨会", "城市会", "区域会", "院内会"],
    ),
    (
        "医疗机构拜访",
        ["医院", "科室", "主任", "医生", "教授", "门诊", "病房",
         "住院部", "手术室", "协和", "人民医", "附属医", "中医", "西京"],
    ),
    (
        "药店拜访",
        ["药店", "药房", "店长", "陈列", "药店拜访", "铺货",
         "大药房", "连锁", "百姓", "同仁堂", "零售"],
    ),
]


def detect_task_type(text: str) -> str:
    """Auto-detect task type from visit description using keyword matching.

    Tries DeepSeek via Hermes auxiliary client if available, falls back
    to keyword matching.
    """
    # Try DeepSeek-based classification first
    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        async def _ask_deepseek():
            prompt = (
                "你是一个药企合规分类器。请根据以下代表描述，判断任务类型。\n"
                "类型选项：药店拜访、医疗机构拜访、学术推广\n\n"
                "代表描述：{text}\n\n"
                "请只返回类型名称，不要加任何其他文字。"
            ).format(text=text)
            result = await async_call_llm(
                prompt=prompt,
                model="deepseek-chat",
                temperature=0.0,
                max_tokens=32,
            )
            return extract_content_or_reasoning(result)

        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # Running in async context, can't block — fall through to keyword
            pass
        except RuntimeError:
            # No running loop — safe to run
            raw = asyncio.run(_ask_deepseek())
            for task_type, _ in TASK_TYPE_RULES:
                if task_type in raw:
                    return task_type
            if raw.strip():
                logger.info("DeepSeek task type: %s", raw.strip())
                return raw.strip()

    except Exception as e:
        logger.warning("DeepSeek task type detection failed: %s, falling back to keywords", e)

    # Keyword-based fallback
    for task_type, keywords in TASK_TYPE_RULES:
        for kw in keywords:
            if kw in text:
                logger.debug("Keyword match '%s' → '%s'", kw, task_type)
                return task_type

    # Default for ambiguous input
    logger.debug("No keyword match, defaulting to 药店拜访")
    return "药店拜访"


# ── Date/time extraction helpers ────────────────────────────────────────────

_DATE_PATTERNS = [
    (r"(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})[日号]?", "%Y-%m-%d"),
    (r"(\d{1,2})月(\d{1,2})[日号]", "month_day"),
]

# Relative date keyword → day offset from today
_RELATIVE_DATE_MAP = {
    "今天": 0, "今日": 0,
    "昨天": -1, "昨日": -1,
    "前天": -2, "前日": -2,
    "大前天": -3,
    "明天": 1, "明日": 1,
    "后天": 2, "后日": 2,
}

# Weekday name → 0-indexed weekday (Mon=0 ... Sun=6)
_WEEKDAY_MAP = {
    "周一": 0, "星期一": 0,
    "周二": 1, "星期二": 1,
    "周三": 2, "星期三": 2,
    "周四": 3, "星期四": 3,
    "周五": 4, "星期五": 4,
    "周六": 5, "星期六": 5,
    "周日": 6, "星期日": 6, "星期天": 6,
}


def _extract_date(text: str) -> str:
    import datetime
    today = datetime.date.today()

    # 1. Explicit date patterns (2024-01-15, 1月15日)
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            if fmt == "%Y-%m-%d":
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return f"{y:04d}-{mo:02d}-{d:02d}"
            elif fmt == "month_day":
                mo, d = int(m.group(1)), int(m.group(2))
                return f"{today.year:04d}-{mo:02d}-{d:02d}"

    # 2. Relative date keywords (今天/昨天/前天/明天 etc.)
    for keyword, offset in _RELATIVE_DATE_MAP.items():
        if keyword in text:
            target = today + datetime.timedelta(days=offset)
            return target.strftime("%Y-%m-%d")

    # 3. Week-based dates (上周三/这周一/下周天 etc.)
    m = re.search(r"(上|本|这|下)?\s*周\s*([一二三四五六日天])", text)
    if m:
        prefix = m.group(1) or "本"
        day_char = m.group(2)
        if day_char == "天":
            day_char = "日"
        weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
        if day_char in weekday_names:
            target_weekday = weekday_names.index(day_char)  # 0=Mon, 6=Sun
            current_weekday = today.weekday()  # 0=Mon, 6=Sun
            if prefix in ("本", "这"):
                delta = target_weekday - current_weekday
            elif prefix == "下":
                delta = target_weekday - current_weekday + 7
            else:  # 上
                delta = target_weekday - current_weekday - 7
            target = today + datetime.timedelta(days=delta)
            return target.strftime("%Y-%m-%d")

    # 4. "X天前" / "X天之前"
    m = re.search(r"(\d+)\s*天[之以]?前", text)
    if m:
        days = int(m.group(1))
        target = today - datetime.timedelta(days=days)
        return target.strftime("%Y-%m-%d")

    # 5. Implicit "today" from time-of-day words without any date indicator
    time_of_day_words = ["早上", "早晨", "上午", "中午", "下午", "傍晚", "晚上", "夜里", "凌晨"]
    if any(w in text for w in time_of_day_words):
        return today.strftime("%Y-%m-%d")

    return ""


_TIME_PERIOD_MAP = {
    "凌晨": "05:00",
    "早上": "08:00",
    "早晨": "08:00",
    "上午": "10:00",
    "中午": "12:00",
    "午后": "13:00",
    "下午": "15:00",
    "傍晚": "17:00",
    "晚上": "19:00",
    "夜里": "21:00",
    "夜间": "22:00",
}


def _extract_time(text: str) -> str:
    # 1. Explicit time patterns (HH:MM)
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"

    # 2. "X点半" (must check before "X点" to avoid short-circuit)
    m = re.search(r"上午(\d{1,2})点半", text)
    if m:
        return f"{int(m.group(1)):02d}:30"
    m = re.search(r"下午(\d{1,2})点半", text)
    if m:
        h = int(m.group(1))
        if h < 12:
            h += 12
        return f"{h:02d}:30"

    # 3. "上午X点" / "下午X点" (whole-hour)
    m = re.search(r"上午(\d{1,2})点", text)
    if m:
        return f"{int(m.group(1)):02d}:00"
    m = re.search(r"下午(\d{1,2})点", text)
    if m:
        h = int(m.group(1))
        if h < 12:
            h += 12
        return f"{h:02d}:00"

    # 4. Natural language time periods (早上/上午/中午/下午/晚上 etc.)
    for keyword, time_str in _TIME_PERIOD_MAP.items():
        if keyword in text:
            return time_str

    return ""


# ── Address extraction ──────────────────────────────────────────────────────

def _extract_address(text: str) -> str:
    patterns = [
        # OCR-style: store/org name followed by address
        # "百康药房 天赐街1号院8号楼一层8-204室 好邻居"
        r"(?:药房|药店|医院|诊所|卫生院|中心|连锁|药业|医药)[\s，,]*"
        r"([^\s,，。；;\n]{2,}(?:路|街|巷|大道|道|里|胡同)"
        r"(?:[^\s,，。；;\n]{2,40}))",

        # Full structured address with 号院/号楼/层/室/单元
        r"([^\s,，。；;\n]{2,}(?:路|街|巷|大道|道|里|胡同)"
        r"\s*\d+[号院楼单元层室栋幢\-—\d一二两三四五六七八九十a-zA-Z]+"
        r"(?:[^\s,，。；;\n]{0,20})?)",

        # 地址/位置/在/位于 prefix
        r"(?:地址|位置|在|位于)[：:]?\s*"
        r"([^\s,，。；;]+[路街巷大道][^\s,，。；;]*)",

        # Generic road-based address
        r"([^\s,，。；;]{2,}(?:[路街巷大道])[\d号]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            result = m.group(1).strip()
            # Reject matches that are clearly not addresses (too short, or just a store name)
            if len(result) >= 4 and re.search(r"[\d号院楼单元层室]", result):
                return result

    # Fallback: look for any address-like pattern without store prefix
    fallback_pat = (
        r"([^\s,，。；;\n]{2,}(?:路|街|巷|大道|道|里|胡同)"
        r"[^\s,，。；;\n]{2,30})"
    )
    m = re.search(fallback_pat, text)
    if m:
        return m.group(1).strip()

    return ""


def _extract_person(text: str) -> Tuple[str, str]:
    person_patterns = [
        (r"见了?\s*([^\s,，。；;]{1,6}(?:店长|主任|医生|教授|经理|院长|药师))", r"见了?\s*"),
        (r"拜访了?\s*([^\s,，。；;]{1,6}(?:店长|主任|医生|教授|经理|院长|药师))", r"拜访了?\s*"),
        (r"([^\s,，。；;]{1,6})(?:店长|主任|医生|教授|经理|院长|药师)", r""),
    ]
    for pat, _ in person_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1), ""
    return "", ""


def _extract_products(text: str) -> str:
    common_products = [
        "小儿宝泰康", "小儿肺热咳喘", "小儿柴桂", "肠炎宁", "健胃消食",
        "金水宝", "复方丹参", "板蓝根", "阿莫西林", "头孢",
        "布洛芬", "对乙酰氨基酚", "奥司他韦", "连花清瘟",
    ]
    found = []
    for p in common_products:
        if p in text:
            found.append(p)
    return ", ".join(found) if found else ""


def _extract_attendee_count(text: str) -> str:
    # 1. Explicit count: N + 人/位/名 + (参会/到场/参加/出席/到来/在座/与会/在场)
    m = re.search(r"(\d+)\s*[人位名]?\s*(?:参会|到场|参加|出席|到来|在座|与会|在场)", text)
    if m:
        return m.group(1)

    # 2. "见了/拜访了/找了/约了/联系了 N位/个/名"
    m = re.search(r"(?:见了|拜访了|找了|约了|联系了|见了有)\s*(\d+)\s*[位个名]", text)
    if m:
        return m.group(1)

    # 3. "N位医生/药师/店长/主任/客户/代表/护士" etc. (title-qualified count)
    m = re.search(
        r"(\d+)\s*位\s*"
        r"(?:医生|药师|店长|主任|教授|经理|院长|客户|代表|护士|专家|老师|领导)",
        text,
    )
    if m:
        return m.group(1)

    # 4. "N个人" / "N人" (general)
    m = re.search(r"(\d+)\s*个?\s*人", text)
    if m:
        return m.group(1)

    # 5. Infer from contact person listing: count unique title-bearing contacts
    # "和王店长、杜药师聊了" = 2 contacts
    # "跟李主任、张医生讨论了" = 2 contacts
    # Strategy: find all title positions, then look back 1-3 Chinese chars for the name
    contact_titles = [
        "店长", "药师", "医生", "主任", "教授", "经理", "院长",
        "护士", "护士长", "科长", "处长", "老师", "代表",
    ]
    # Characters that are verbs/particles, not part of a name
    _verb_chars = set("去了到见找拜访跟和与请叫让给对向了过的")
    found_contacts = set()
    for title in contact_titles:
        for m in re.finditer(re.escape(title), text):
            pos = m.start()
            name_candidate = ""
            # Walk back 1-3 chars to find the name
            for name_len in (1, 2, 3):
                if pos >= name_len:
                    candidate = text[pos - name_len : pos]
                    if (
                        re.match(r"^[\u4e00-\u9fff]+$", candidate)
                        and candidate[0] not in _verb_chars
                    ):
                        name_candidate = candidate
                        break
            if name_candidate:
                found_contacts.add(name_candidate + title)
            elif pos == 0 or (pos >= 1 and text[pos - 1] in "，,。；;、\s跟和与"):
                # Title appears standalone (e.g., "跟店长聊了") — count as 1
                found_contacts.add(title)

    if found_contacts:
        return str(len(found_contacts))

    return ""


# ── Main extraction ─────────────────────────────────────────────────────────

def extract_fields(text: str) -> Dict[str, Any]:
    """Extract 17 compliance fields from visit description text.

    Args:
        text: Free-text visit description from rep (could be typed, STT, or merged).

    Returns:
        dict with all 17 fields populated.
    """
    person, title = _extract_person(text)

    result: Dict[str, Any] = {
        "task_type": detect_task_type(text),
        "org_name": _extract_org_name(text),
        "org_address": _extract_address(text),
        "contact_person": person,
        "contact_title": title or _infer_title(person),
        "visit_date": _extract_date(text),
        "visit_time": _extract_time(text),
        "products": _extract_products(text),
        "topic": _extract_topic(text),
        "content_summary": text.strip(),
        "competitor_info": _extract_competitor(text),
        "feedback": _extract_feedback(text),
        "next_steps": _extract_next_steps(text),
        "attendee_count": _extract_attendee_count(text),
        "meeting_duration": "",
        "materials_used": _extract_materials_used(text),
        "notes": "",
    }
    logger.info("Extraction result: %s", json.dumps(result, ensure_ascii=False))
    return result


def _extract_org_name(text: str) -> str:
    patterns = [
        r"([^\s,，。；;]{2,10}(?:医院|诊所|卫生院|社区卫生|疾控))",
        r"([^\s,，。；;]{2,10}(?:大药房|药房|药店|药业|医药|连锁))",
        r"去了\s*([^\s,，。；;]{2,10}(?:药房|药店|医院|诊所)?)",
        r"到了\s*([^\s,，。；;]{2,10}(?:药房|药店|医院|诊所)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return ""


def _infer_title(person: str) -> str:
    for suffix in ["店长", "主任", "医生", "教授", "经理", "院长", "药师"]:
        if suffix in person:
            return suffix
    return ""


def _extract_topic(text: str) -> str:
    patterns = [
        r"(?:主题|议题|讲了|介绍了|聊了|讨论了)\s*[：:]?\s*([^\s,，。；;]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return ""


def _extract_competitor(text: str) -> str:
    # 1. Explicit competitor keywords followed by description
    patterns = [
        r"竞品[：:]*\s*([^。；;，,\n]{2,60})",
        r"竞争[对手品牌产品]*[：:]*\s*([^。；;，,\n]{2,60})",
        # Competitor brand mentioned with context (对比/比起 + brand-like word)
        r"(?:对比|比起)\s*([^，,。；;\s]{2,12}(?:品牌|产品|药|颗粒|胶囊|片|剂))",
        # "XX也在做/推/卖" pattern
        r"([^\s，,。；;]{2,10}(?:也在做|也在推|也在卖|也有[在卖推]))",
        # Competitor mention in reported speech
        r"(?:他们说|客户说|店长说|医生[说]?)[^。;；\n]*?(?:别的?|其他|人家)([^。;；，,\n]{2,50})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            result = m.group(1).strip()
            if len(result) >= 2:
                return result

    # 2. Fallback: detect known competitor brands in text with surrounding context
    competitor_brands = [
        "葵花", "三九", "白云山", "同仁堂", "修正", "仁和", "太极",
        "江中", "哈药", "天士力", "以岭", "扬子江", "恒瑞", "正大天晴",
        "步长", "华润", "丽珠", "人福", "科伦", "诺华", "辉瑞", "拜耳",
        "强生", "默沙东", "GSK", "赛诺菲", "阿斯利康", "罗氏",
        "百特", "武田", "第一三共", "卫材", "安斯泰来",
        "灵北", "费森尤斯", "礼来", "诺和诺德", "雅培",
        "东阿阿胶", "片仔癀", "云南白药", "康恩贝", "济川",
    ]
    found = []
    for brand in competitor_brands:
        if brand in text:
            idx = text.find(brand)
            # Skip if the brand appears to be the store being visited (preceded by 去了/到了/在)
            prefix_ctx = text[max(0, idx - 3):idx]
            if re.search(r"(?:去了|到了|在|拜访)\s*$", prefix_ctx):
                continue
            start = max(0, idx)
            end = min(len(text), idx + len(brand) + 10)
            snippet = text[start:end].strip()
            found.append(snippet)
    if found:
        return "; ".join(found[:3])

    return ""


def _extract_feedback(text: str) -> str:
    # 1. Direct feedback/sentiment markers
    patterns = [
        # Explicit feedback lead-ins
        r"(?:反馈|觉得|认为|表示|说|提到|反映|指出)[：:]?\s*([^。；;，,\n]{3,80})",
        # Customer satisfaction/opinion expressions
        r"(?:销量|效果|价格|质量|反应|口感|包装|动销|走量|患者|顾客|病人)"
        r"[^。；;\n]{0,20}?(?:不错|很好|还行|一般|不好|差|可以|满意|不满意"
        r"|有改善|有提升|下降|增长|减少|变差|变好)"
        r"([^。；;，,\n]{0,50})",
        # Patient/customer relayed feedback
        r"(?:病人|患者|顾客|老百姓)[^。;；\n]*?(?:说|反馈|觉得|认为|反映)"
        r"([^。;；，,\n]{3,60})",
        # Store staff recommendation/suggestion
        r"(?:建议|希望|要求|提出|提议)([^。;；，,\n]{3,60})",
        # Opinion about product positioning
        r"觉得([^。;；，,\n]{3,60})",
        # "XX说YY" pattern
        r"说[：:]?\s*([^。;；，,\n]{4,80})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            result = m.group(1).strip()
            if len(result) >= 2:
                return result

    # 2. Fallback: look for sentiment/opinion keywords and capture surrounding sentence
    sentiment_keywords = ["销量好", "卖得好", "走得好", "反馈好", "效果好",
                          "不好卖", "卖不动", "滞销", "退货", "投诉"]
    for kw in sentiment_keywords:
        if kw in text:
            idx = text.find(kw)
            start = max(0, idx - 15)
            end = min(len(text), idx + len(kw) + 20)
            return text[start:end].strip()

    return ""


def _extract_next_steps(text: str) -> str:
    # 1. Direct next-step / follow-up markers
    patterns = [
        # Explicit next-step markers
        r"(?:下次|下一步|接下来|后续|之后|往后|以后|回头|改天)[：:]?\s*"
        r"([^。；;，,\n]{3,80})",

        # Planning markers
        r"(?:计划|准备|打算|安排|预计|将[要会]|会再|约好|约定|定于|说好)"
        r"[：:]?\s*([^。；;，,\n]{3,60})",

        # Promise/commitment markers
        r"(?:答应|承诺|保证|到时候|到时候再|过[几两]\s*[天周月]|下次来)"
        r"[：:]?\s*([^。；;，,\n]{3,60})",

        # Future action verbs
        r"(?:再[来过次趟]|还会|还要|需要[再还])[：:]?\s*([^。；;，,\n]{3,60})",

        # "等XX时候" pattern
        r"(?:等|等到)[^。;；\n]{1,15}?(?:时候|时|后|以后|之后)"
        r"[：:]?\s*([^。；;，,\n]{3,60})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            result = m.group(1).strip()
            if len(result) >= 2:
                return result

    # 2. Fallback: look for any sentence that strongly indicates future action
    future_indicators = [
        (r"下次.{0,20}?带.{0,20}?(?:资料|样品|DA|彩页|单页)", "下次带资料/样品"),
        (r"下次.{0,20}?再聊", "下次再聊"),
        (r"再[来过]", "再来拜访"),
        (r"会再联系", "会再联系"),
    ]
    for pat, label in future_indicators:
        if re.search(pat, text):
            return label

    return ""


def _extract_materials_used(text: str) -> str:
    """Extract materials used during the visit (DA, brochures, samples, etc.)."""
    materials_keywords = [
        "DA", "da",
        "宣传单", "宣传册", "宣传资料", "宣传页",
        "彩页", "单页", "折页", "手册",
        "资料", "文献", "论文", "指南",
        "样品", "试用装", "试用", "样本",
        "礼品", "台历", "日历", "小礼品",
        "PPT", "幻灯片", "视频", "iPad", "平板", "电脑",
        "海报", "展架", "易拉宝", "X展架", "pop", "POP",
        "拜访夹", "拜访手册", "产品手册", "说明书",
        "白大褂", "工作服", "笔", "笔记本",
        "问卷", "调研表", "知情同意",
    ]
    found = []
    lower_text = text.lower()
    for kw in materials_keywords:
        if kw.lower() in lower_text:
            found.append(kw)
    return ", ".join(found) if found else ""


# ── Field completeness checking ────────────────────────────────────────────

# P0 required fields (7 core fields)
CORE_REQUIRED_FIELDS = [
    "task_type",
    "org_name",
    "contact_person",
    "products",
    "topic",
    "content_summary",
    "next_steps",
]

# P1 recommended fields (3 important supplemental fields)
CORE_RECOMMENDED_FIELDS = [
    "visit_date",
    "feedback",
    "competitor_info",
]

# All core fields in priority order (P0 first, then P1)
CORE_FIELDS_PRIORITY = CORE_REQUIRED_FIELDS + CORE_RECOMMENDED_FIELDS

# Field display names for user-facing messages
CORE_FIELD_LABELS: Dict[str, str] = {
    "task_type": "任务类型",
    "org_name": "机构名称",
    "contact_person": "拜访对象",
    "products": "涉及产品",
    "topic": "拜访主题",
    "content_summary": "内容摘要",
    "next_steps": "下一步计划",
    "visit_date": "拜访日期",
    "feedback": "客户反馈",
    "competitor_info": "竞品信息",
}


def get_missing_core_fields(fields: Dict[str, Any]) -> List[str]:
    """Return the top 2–3 missing core fields (P0 then P1).

    Checks P0 required fields first, then P1 recommended fields.
    Returns at most 3 field names, ordered by priority.
    Returns empty list if all core fields are present.
    """
    missing: List[str] = []
    for field in CORE_FIELDS_PRIORITY:
        value = fields.get(field, "")
        if not value or (isinstance(value, str) and not value.strip()):
            missing.append(field)
            if len(missing) >= 3:
                break
    return missing


def fields_to_summary(fields: Dict[str, Any]) -> str:
    """Convert extracted fields to a human-readable summary."""
    lines = []
    label_map = {
        "task_type": "任务类型",
        "org_name": "机构名称",
        "org_address": "机构地址",
        "contact_person": "拜访对象",
        "contact_title": "对象职务",
        "visit_date": "拜访日期",
        "visit_time": "拜访时间",
        "products": "涉及产品",
        "topic": "拜访主题",
        "content_summary": "内容摘要",
        "competitor_info": "竞品信息",
        "feedback": "客户反馈",
        "next_steps": "下一步计划",
        "attendee_count": "参会人数",
        "meeting_duration": "会议时长",
        "materials_used": "使用资料",
        "notes": "备注",
    }
    for field in FIELD_NAMES:
        value = fields.get(field, "")
        if value:
            lines.append(f"{label_map.get(field, field)}: {value}")
    return "\n".join(lines)
