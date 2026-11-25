"""Robust Streaming Message Boundary Parser for Hermes Agent (Phase 2.3 Hardened + Paragraph Fallback).

Features:
- Strict Delimiter Extraction: Detects autonomous message boundaries with balanced nesting depth.
- Alternative Delimiter Support: <<<bubble>>> and <bubble>...</bubble> with correct token precedence.
- Code Fence State Machine: Cross-bubble code fence tracking, balancing, and language preservation.
- Markdown/HTML Protection: Code blocks containing <bubble> example tags are preserved verbatim.
- Robust Incomplete Tag Handling: Incomplete control tags at stream end are stripped cleanly without leaking.
- Natural Paragraph Fallback: When no bubble tags are used, breaks on clean paragraph boundaries (\\n\\n)
  outside code blocks for smooth streaming on plain text outputs.
- Comprehensive Multimodal Tags: Tapbacks, interactive polls, calendar cards (.ics), Apple Maps cards,
  and native iOS animations / invisible ink effects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

from gateway.bubble_actions import BubbleAction, EffectAction, PollAction, TapbackAction, TextBubble

logger = logging.getLogger(__name__)


def _get_base_scratch_dir() -> Path:
    try:
        from hermes_constants import get_scratch_dir
        return get_scratch_dir(prune=False)
    except Exception:  # health: allow BLE001 -- fallback when hermes_constants is uninitialized
        import tempfile
        return Path(tempfile.gettempdir()) / "hermes_scratch"


class BubbleParser:
    """Parses incremental LLM streaming deltas into discrete, Markdown-balanced bubbles."""

    # Explicit trusted extension points for optional local customizations
    _custom_action_handlers: dict[str, tuple[re.Pattern, Callable]] = {}
    _custom_transforms: list[Callable[[str], list[Any]]] = []
    _custom_finish_hooks: list[Callable[[Any], list[BubbleAction]]] = []

    @classmethod
    def register_custom_action(cls, name: str, pattern: re.Pattern, handler: Callable) -> None:
        """Explicit registration point for trusted local customizations."""
        if name in cls._custom_action_handlers:
            existing_pat, existing_fn = cls._custom_action_handlers[name]
            if existing_fn is handler and existing_pat.pattern == pattern.pattern:
                return  # Idempotent re-registration
            logger.warning(
                "[BubbleParser] Refusing silent overwrite of existing custom action '%s' with a different handler",
                name,
            )
            return
        cls._custom_action_handlers[name] = (pattern, handler)

    @classmethod
    def register_custom_transform(cls, transform_fn: Callable[[str], list[Any]]) -> None:
        """Explicit registration point for trusted local text transforms."""
        if transform_fn not in cls._custom_transforms:
            cls._custom_transforms.append(transform_fn)

    @classmethod
    def register_custom_finish_hook(cls, hook_fn: Callable[[Any], list[BubbleAction]]) -> None:
        """Explicit registration point for trusted local finish hooks."""
        if hook_fn not in cls._custom_finish_hooks:
            cls._custom_finish_hooks.append(hook_fn)

    @classmethod
    def clear_custom_extensions(cls) -> None:
        """Clear all registered local extensions."""
        cls._custom_action_handlers.clear()
        cls._custom_transforms.clear()
        cls._custom_finish_hooks.clear()


    _OPEN_TAG = "<bubble>"
    _CLOSE_TAG = "</bubble>"
    _SEP_MARKER = "<<<bubble>>>"
    _MD_LINK_RE = re.compile(r'\[(?P<text>[^\]\r\n]+)\]\((?P<url>(?:https?|shortcuts|maps|mobilenotes)://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+)\)')
    _BARE_URL_RE = re.compile(r'(?:https?|shortcuts|maps|mobilenotes)://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+')
    _TRAILING_PUNCT = ".,;:!?()[]{}<>\"'“”‘’，。！？；：、（）《》【】"

    _TAPBACK_RE = re.compile(
        r'<(?:tapback|reaction)(?:\s+emoji=["\']([^"\']+)["\'])?\s*>(.*?)</(?:tapback|reaction)>|<(?:tapback|reaction)\s+emoji=["\']([^"\']+)["\']\s*/?>',
        re.IGNORECASE | re.DOTALL,
    )
    _POLL_RE = re.compile(
        r'<poll(?:\s+title=["\'](?P<attr_title>[^"\']+)["\'])?\s*>(?P<body>.*?)</poll>',
        re.IGNORECASE | re.DOTALL,
    )
    _EFFECT_RE = re.compile(
        r'<(?:effect:(?P<colon_eff>[a-zA-Z0-9_]+)|effect(?:\s+(?P<attrs>[^>]*))?)\s*>(?P<text>.*?)</(?:effect(?::[a-zA-Z0-9_]+)?)>|'
        r'<(?:effect:(?P<prefix_colon>[a-zA-Z0-9_]+)|effect\s+name=["\'](?P<prefix_name>[^"\']+)["\']|effect\s+type=["\'](?P<prefix_type>[^"\']+)["\']|effect\s+(?P<prefix_bare>[a-zA-Z0-9_]+))\s*/?>',
        re.IGNORECASE | re.DOTALL,
    )
    _CALENDAR_RE = re.compile(
        r'<(?:calendar|calendar_event|event|schedule)(?:\s+(?P<attrs>[^\s>][^>]*?))?\s*>(?P<body>.*?)</(?:calendar|calendar_event|event|schedule)>|'
        r'<(?:calendar|calendar_event|event|schedule)(?:\s+(?P<self_attrs>[^\s>][^>]*?))?\s*/>',
        re.IGNORECASE | re.DOTALL,
    )
    _MAP_RE = re.compile(
        r'<(?:map|location)(?:\s+(?P<attrs>[^\s>][^>]*?))?\s*>(?P<body>.*?)</(?:map|location)>|'
        r'<(?:map|location)(?:\s+(?P<self_attrs>[^\s>][^>]*?))?\s*/>',
        re.IGNORECASE | re.DOTALL,
    )
    _APPLE_MAPS_URL_RE = re.compile(
        r'https?://(?:maps\.apple\.com|maps\.google\.com)[^\s<>"\'\)]+',
        re.IGNORECASE,
    )
    _SHORTCUTS_URL_RE = re.compile(
        r'shortcuts://[^\r\n<>"\'\)]+',
        re.IGNORECASE,
    )
    _APPLE_MUSIC_URL_RE = re.compile(
        r'https?://(?:music|podcasts)\.apple\.com/[^\s<>"\'\)]+',
        re.IGNORECASE,
    )
    _SPOILER_RE = re.compile(
        r'<spoiler>(?P<text>.*?)</spoiler>|(?<!\|)\|\|(?!\|)(?P<pipe_text>.+?)\|\|',
        re.IGNORECASE | re.DOTALL,
    )
    _BOILERPLATE_PATTERNS = [
        re.compile(r'(?:[🌦️🎬📖]?\s*)?(?:天气胶囊|天气卡片|电影卡片|影视卡片|电影档案|单词卡片|词典卡片|词汇卡片)(?:已|给你)?(?:生成好?|备好|整理好|做好)啦?[～！。:，,\s]*', re.IGNORECASE),
        re.compile(r'已为你?生成(?:天气|电影|影视|美剧|单词|词典).*?卡片[～！。:，,\s]*', re.IGNORECASE),
        re.compile(r'名片卡片(?:给你)?生成好啦[，,、\s]*点击[^，。！\n]*?(?:拨打|存入)[^，。！\n]*[～！。]?', re.IGNORECASE),
        re.compile(r'投票卡片(?:给你)?做好了[，,、\s]*戳选项[^，。！:\n]*[～！。:]?', re.IGNORECASE),
        re.compile(r'原生日程卡片已生成[，,、\s]*点(?:一下|击)[^，。！\n]*?存入[^，。！\n]*[～！。]?', re.IGNORECASE),
        re.compile(r'[📅📇📍]?\s*已为你?生成.*?卡片[，,、\s]*点击即可[^，。！:\n]*[～！。:]?', re.IGNORECASE),
        re.compile(r'(?:给你)?做成\s*(?:Apple\s*HIG\s*)?(?:视网膜|高清|对比)?(?:长图|表格)啦?[:：]?', re.IGNORECASE),
        re.compile(r'原生地图卡片给你放这[，,、\s]*点开能直接[^，。！:\n]*[～！。:]?', re.IGNORECASE),
        re.compile(r'(?:想去哪[种个]|需要的话)[，,、\s]*我?直接把地图(?:导航)?卡片给你甩过来[～！。]?', re.IGNORECASE),
        re.compile(r'答案给你用隐形墨水遮好啦[，,、\s]*猜完刮开[^，。！:\n]*[～！。:👀\s]*', re.IGNORECASE),
        re.compile(r'(?:睡前压轴)?全屏大烟花给你放上啦[！!～]?', re.IGNORECASE),
        re.compile(r'点[击一下]卡片(?:即可|可以|直接).*?[存入保存添加查看导航拨打][^，。！\n]*[～！。]?', re.IGNORECASE),
        re.compile(r'点击(?:上面|下方|这里)的?卡片(?:即可|可以|直接)?[^，。！\n]*[～！。]?', re.IGNORECASE),
        re.compile(r'(?:和|的)?(?:具体)?(?:路线|地图)?卡片[？?～！。]?', re.IGNORECASE),
        re.compile(r'.*?(?:土区|国区|Apple\s*Music|播客|音乐|歌单)?(?:搜索直达|歌单|播放器)?(?:给你)?(?:备好|生成好|整理好)啦?[，,、\s]*点(?:开|击)?(?:就能|即可|直接)?(?:直接)?(?:拉起|打开)?(?:大图|原生)?(?:播放器|卡片)?(?:试听|收听|播放)?[:：]?', re.IGNORECASE),
        re.compile(r'.*?(?:搜索直达|大图播放器).*?[:：]?', re.IGNORECASE),
    ]

    _CITY_COORDINATES = {
        # Hong Kong / Macao
        "旺角": "22.3193,114.1694",
        "尖沙咀": "22.2988,114.1722",
        "中环": "22.2819,114.1582",
        "铜锣湾": "22.2800,114.1850",
        "湾仔": "22.2770,114.1751",
        "油麻地": "22.3129,114.1708",
        "九龙": "22.3204,114.1698",
        "香港": "22.3193,114.1694",
        "澳门": "22.1987,113.5439",
        "氹仔": "22.1568,113.5587",
        # Guangdong
        "越秀": "23.1415,113.2625",
        "天河": "23.1246,113.3611",
        "海珠": "23.0833,113.3172",
        "荔湾": "23.1259,113.2430",
        "白云": "23.1579,113.2731",
        "番禺": "22.9388,113.3843",
        "黄埔": "23.1064,113.4589",
        "花都": "23.4036,113.2201",
        "南沙": "22.8028,113.5248",
        "广州": "23.1291,113.2644",
        "南山": "22.5329,113.9303",
        "福田": "22.5215,114.0551",
        "罗湖": "22.5482,114.1315",
        "宝安": "22.5533,113.8831",
        "龙岗": "22.7215,114.2477",
        "龙华": "22.6913,114.0298",
        "深圳": "22.5431,114.0579",
        "斗门": "22.2082,113.2965",
        "香洲": "22.2667,113.5433",
        "金湾": "22.0636,113.3444",
        "横琴": "22.1408,113.5358",
        "珠海": "22.2707,113.5767",
        "佛山": "23.0215,113.1214",
        "东莞": "23.0205,113.7518",
        "中山": "22.5176,113.3928",
        "惠州": "23.1118,114.4162",
        "汕头": "23.3541,116.6820",
        # Top Metros
        "海淀": "39.9599,116.2981",
        "朝阳": "39.9215,116.4431",
        "东城": "39.9175,116.4188",
        "西城": "39.9123,116.3659",
        "丰台": "39.8585,116.2862",
        "北京": "39.9042,116.4074",
        "黄浦": "31.2317,121.4844",
        "徐汇": "31.1883,121.4365",
        "静安": "31.2288,121.4533",
        "浦东": "31.2211,121.5447",
        "杨浦": "31.2595,121.5260",
        "虹口": "31.2647,121.5052",
        "上海": "31.2304,121.4737",
        "西湖": "30.2431,120.1344",
        "滨江": "30.2084,120.2119",
        "余杭": "30.4199,120.3015",
        "杭州": "30.2741,120.1551",
        "成都": "30.5728,104.0668",
        "重庆": "29.5630,106.5516",
        "武汉": "30.5928,114.3055",
        "南京": "32.0603,118.7969",
        "西安": "34.3416,108.9398",
        "长沙": "28.2282,112.9388",
        "苏州": "31.2989,120.5853",
        "天津": "39.0842,117.2009",
        "青岛": "36.0671,120.3826",
        "厦门": "24.4798,118.0894",
        "福州": "26.0745,119.2965",
        "济南": "36.6512,117.1201",
        "郑州": "34.7579,113.6654",
        "合肥": "31.8206,117.2272",
        "南昌": "28.6829,115.8582",
        "昆明": "25.0406,102.7123",
        "贵阳": "26.6470,106.6302",
        "南宁": "22.8170,108.3665",
        "海口": "20.0440,110.1999",
        "三亚": "18.2528,109.5120",
        "台北": "25.0330,121.5654",
        "高雄": "22.6273,120.3014",
        # International
        "东京": "35.6762,139.6503",
        "新宿": "35.6938,139.7034",
        "涩谷": "35.6580,139.7016",
        "大阪": "34.6937,135.5023",
        "京都": "35.0116,135.7681",
        "首尔": "37.5665,126.9780",
        "新加坡": "1.3521,103.8198",
        "曼谷": "13.7563,100.5018",
        "伦敦": "51.5074,-0.1278",
        "巴黎": "48.8566,2.3522",
        "纽约": "40.7128,-74.0060",
        "旧金山": "37.7749,-122.4194",
        "洛杉矶": "34.0522,-118.2437",
        "西雅图": "47.6062,-122.3321",
        "悉尼": "-33.8688,151.2093",
    }

    @classmethod
    def _prune_robotic_boilerplate(cls, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        res = text
        matched_any = False
        for p in cls._BOILERPLATE_PATTERNS:
            if p.search(res):
                res = p.sub("", res)
                matched_any = True
        if matched_any:
            res = re.sub(r'[\s:：，,、-]+$', '', res)
        return res.strip()
    _CONTACT_RE = re.compile(
        r'<(?:contact|vcard|card)(?:\s+(?P<attrs>[^\s>][^>]*?))?\s*>(?P<body>.*?)</(?:contact|vcard|card)>|'
        r'<(?:contact|vcard|card)(?:\s+(?P<self_attrs>[^\s>][^>]*?))?\s*/>',
        re.IGNORECASE | re.DOTALL,
    )
    _IMAGE_TAG_RE = re.compile(
        r'<(?:image|img|photo|media|meme)(?:\s+(?P<attrs>[^\s>][^>]*?))?\s*>(?P<body>.*?)</(?:image|img|photo|media|meme)>|'
        r'<(?:image|img|photo|media|meme)(?:\s+(?P<self_attrs>[^\s>][^>]*?))?\s*/>',
        re.IGNORECASE | re.DOTALL,
    )
    _MEDIA_DIRECTIVE_RE = re.compile(
        r'(?P<pre>[\r\n\s>]|^)(?P<tag>(?:[`*_]{0,3})(?:MEDIA|IMAGE|FILE):[ \t]*'
        r'(?:"(?P<qpath>[^\r\n"]+)"|\'(?P<sqpath>[^\r\n\']+)\'|`(?P<btpath>[^\r\n`]+)`|(?P<path>[^\r\n<>]+?))'
        r'[`"\'*_]*[ \t]*)(?=(?:[\r\n]|$|<))',
        re.IGNORECASE,
    )
    _GENERIC_UNSUPPORTED_TAG_RE = re.compile(
        r'<(?:weather|forecast|movie|film|tv|drama|word|dict|dictionary|lexicon|cheat_sheet|visual_table|sheet)'
        r'(?:\s+(?:[^"\'/>]|"[^"]*"|\'[^\']*\')*?)?>'
        r'(?P<body>.*?)'
        r'</(?:weather|forecast|movie|film|tv|drama|word|dict|dictionary|lexicon|cheat_sheet|visual_table|sheet)>|'
        r'<(?:weather|forecast|movie|film|tv|drama|word|dict|dictionary|lexicon|cheat_sheet|visual_table|sheet)'
        r'(?:\s+(?P<attrs>(?:[^"\'/>]|"[^"]*"|\'[^\']*\')*?))?\s*/>',
        re.IGNORECASE | re.DOTALL,
    )
    _ACTION_RE = re.compile(
        r'<(?:action|shortcut|deeplink)(?:\s+(?P<attrs>[^>]*?))?\s*>(?P<body>.*?)</(?:action|shortcut|deeplink)>|'
        r'<(?:action|shortcut|deeplink)(?:\s+(?P<self_attrs>(?:[^"\'/>]|"[^"]*"|\'[^\']*\')*?))?\s*/>',
        re.IGNORECASE | re.DOTALL,
    )
    _EFFECT_ALIAS_MAP = {
        "invisible_ink": "invisible", "ink": "invisible", "hidden": "invisible", "secret": "invisible",
        "firework": "fireworks", "laser": "lasers", "balloon": "balloons",
        "sparkle": "sparkles", "hearts": "heart", "love": "heart",
        "impact": "slam", "shout": "loud", "whisper": "gentle",
    }
    _CODE_BLOCK_RE = re.compile(
        r'```(?P<lang>[a-zA-Z0-9_\-\+]*)\r?\n(?P<code>.*?)```',
        re.DOTALL,
    )
    _EXT_MAP = {
        "python": "py", "py": "py", "javascript": "js", "js": "js", "typescript": "ts", "ts": "ts",
        "bash": "sh", "sh": "sh", "shell": "sh", "zsh": "sh", "json": "json", "html": "html",
        "css": "css", "sql": "sql", "yaml": "yaml", "yml": "yml", "rust": "rs", "cpp": "cpp",
        "c": "c", "go": "go", "java": "java", "ruby": "rb", "php": "php", "swift": "swift",
    }
    _EMOJI_NAME_MAP = {
        "love": "❤️", "heart": "❤️", "thanks": "❤️", "thank": "❤️",
        "like": "👍", "thumbsup": "👍", "thumbs_up": "👍", "ok": "👍",
        "dislike": "👎", "thumbsdown": "👎", "thumbs_down": "👎",
        "haha": "😂", "laugh": "😂", "lol": "😂", "joy": "😂",
        "exclamation": "‼️", "shock": "‼️", "emphasize": "‼️", "wow": "‼️",
        "question": "❓", "confused": "❓", "what": "❓",
    }

    def __init__(
        self,
        paragraph_fallback: bool = True,
        isolate_urls: bool = False,
        config: Optional[Any] = None,
        chat_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> None:
        self._buffer: str = ""
        self._in_code_block: bool = False
        self._active_code_lang: str = ""
        self._pending_effect: Optional[str] = None
        self._has_emitted_map: bool = False
        self._has_emitted_poll: bool = False
        self.paragraph_fallback: bool = paragraph_fallback
        self.isolate_urls: bool = isolate_urls
        self.chat_id: str = chat_id or f"chat_{uuid.uuid4().hex[:8]}"
        self.turn_id: Optional[str] = turn_id or f"turn_{uuid.uuid4().hex[:8]}"
        if config is None:
            from gateway.airhermes_config import AirHermesConfig
            self._config = AirHermesConfig()
        else:
            self._config = config

    @property
    def buffer(self) -> str:
        return self._buffer

    @property
    def in_code_block(self) -> bool:
        return self._in_code_block

    @property
    def active_code_lang(self) -> str:
        return self._active_code_lang

    @classmethod
    def _normalize_tapback_emoji(cls, raw: str) -> Optional[str]:
        raw = raw.strip()
        if not raw:
            return None
        lower = raw.lower()
        return cls._EMOJI_NAME_MAP.get(lower, raw)

    @staticmethod
    def _action_to_legacy_str(action: Any) -> str:
        if isinstance(action, str):
            return action
        from gateway.bubble_actions import (
            EffectAction,
            FileAttachmentAction,
            ImageAttachmentAction,
            PollAction,
            TapbackAction,
            TextBubble,
        )
        if isinstance(action, TextBubble):
            return action.content
        if isinstance(action, PollAction):
            return f"POLL:{json.dumps({'title': action.title, 'options': list(action.options)}, ensure_ascii=False)}"
        if isinstance(action, TapbackAction):
            return f"TAPBACK:{action.reaction}"
        if isinstance(action, EffectAction):
            return f"EFFECT:{json.dumps({'effect': action.effect_type, 'text': action.content}, ensure_ascii=False)}"
        if isinstance(action, FileAttachmentAction):
            return f"FILE:{action.staged_path}"
        if isinstance(action, ImageAttachmentAction):
            return f"IMAGE:{json.dumps({'path': action.staged_path, 'caption': action.caption or '', 'delay': 1.5}, ensure_ascii=False)}"
        return str(action)

    def feed(self, delta: str) -> list[str]:
        """Feed a streaming chunk/delta and return list of fully ready bubbles (legacy string API)."""
        actions = self.feed_actions(delta)
        return [self._action_to_legacy_str(a) for a in actions]

    def finish(self, exception_occurred: bool = False) -> list[str]:
        """Finalize parsing at the end of streaming or upon exception (legacy string API)."""
        actions = self.finish_actions(exception_occurred=exception_occurred)
        return [self._action_to_legacy_str(a) for a in actions]

    def feed_actions(self, delta: str) -> list[BubbleAction]:
        """Feed a streaming chunk/delta and return list of ready typed BubbleAction objects."""
        self._buffer += delta
        return self._extract_ready_actions(is_finish=False)

    def finish_actions(self, exception_occurred: bool = False) -> list[BubbleAction]:
        """Finalize parsing at stream end and return list of remaining ready typed BubbleActions."""
        ready = self._extract_ready_actions(is_finish=True)
        if exception_occurred:
            self._buffer = ""
            self._in_code_block = False
            self._active_code_lang = ""
            self._pending_effect = None
            return ready

        # Check for unclosed <tapback> tag in remaining buffer
        unclosed_tap = re.search(
            r'<(?:tapback|reaction)(?:\s+emoji=["\']([^"\']+)["\'])?\s*>([^<]*)$',
            self._buffer,
            re.IGNORECASE,
        )
        if unclosed_tap and not self._is_inside_code_block(self._buffer[:unclosed_tap.start()]):
            raw_emoji = unclosed_tap.group(1) or unclosed_tap.group(2) or ""
            clean_emoji = self._normalize_tapback_emoji(raw_emoji)
            pre = self._buffer[:unclosed_tap.start()].strip()
            self._buffer = ""
            if pre:
                ready.extend(self._emit_actions(pre))
            if clean_emoji and getattr(self._config, "enable_tapbacks", True):
                ready.append(TapbackAction(reaction=clean_emoji))

        rem = self._buffer.strip()
        self._buffer = ""

        if rem:
            rem = self._strip_dangling_control_tags(rem)
            if rem:
                if self.paragraph_fallback and "\n\n" in rem and not self._is_inside_code_block(rem):
                    paras = [p.strip() for p in re.split(r'\n\s*\n', rem) if p.strip()]
                    for p in paras:
                        ready.extend(self._emit_actions(p))
                else:
                    ready.extend(self._emit_actions(rem))

        for hook in self._custom_finish_hooks:
            ready.extend(hook(self))

        # Reset cross-turn parser state
        self._in_code_block = False
        self._active_code_lang = ""
        self._pending_effect = None
        self._has_emitted_poll = False
        return ready

    def _find_matching_bubble_close(self, buffer: str, open_pos: int) -> int:
        """Find the matching </bubble> for <bubble> at open_pos, respecting tag nesting."""
        idx = open_pos + len(self._OPEN_TAG)
        depth = 1
        while idx < len(buffer):
            next_open = buffer.find(self._OPEN_TAG, idx)
            next_close = buffer.find(self._CLOSE_TAG, idx)
            if next_close == -1:
                return -1
            if next_open != -1 and next_open < next_close:
                depth += 1
                idx = next_open + len(self._OPEN_TAG)
            else:
                depth -= 1
                if depth == 0:
                    return next_close
                idx = next_close + len(self._CLOSE_TAG)
        return -1

    def _balance_and_track_fences(self, content: str) -> str:
        """Auto-balance code fences for a single bubble while updating cross-bubble state."""
        text = content
        if self._in_code_block:
            text = f"```{self._active_code_lang}\n" + text

        lines = text.splitlines()
        in_code = False
        current_lang = ""
        for l in lines:
            idx = 0
            while idx < len(l):
                fence_idx = l.find("```", idx)
                if fence_idx == -1:
                    break
                if not in_code:
                    in_code = True
                    rest = l[fence_idx + 3:].strip()
                    current_lang = rest.split()[0] if rest else ""
                    idx = fence_idx + 3
                    if current_lang:
                        idx += len(current_lang)
                else:
                    in_code = False
                    current_lang = ""
                    idx = fence_idx + 3

        if in_code:
            text = text.rstrip() + "\n```"
            self._in_code_block = True
            self._active_code_lang = current_lang
        else:
            self._in_code_block = False
            self._active_code_lang = ""

        return text

    def _strip_dangling_control_tags(self, text: str) -> str:
        """Strip incomplete dangling tags at stream end while protecting code blocks."""
        cleaned = re.sub(r'^\s*<{1,3}bubble>{1,3}', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?b(u(b(b(l(e)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'<{1,3}b(u(b(b(l(e)?)?)?)?)?>{0,3}$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?t(a(p(b(a(c(k)?)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?r(e(a(c(t(i(o(n)?)?)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?p(o(l(l)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?e(f(f(e(c(t)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?c(a(l(e(n(d(a(r)?)?)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?(?:calendar_event|event|schedule)[^>]*$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'<effect:[^>]*$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?m(a(p)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?l(o(c(a(t(i(o(n)?)?)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?s(p(o(i(l(e(r)?)?)?)?)?)?$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?(?:contact|vcard|card)[^>]*$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?(?:image|img|photo|media|meme)[^>]*$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?:^|[\r\n\s>])(?:[`*_]{0,3})(?:MEDIA|IMAGE|FILE):[ \t]*[`"\'*_]{0,3}$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?(?:cheat_sheet|visual_table|sheet|weather|forecast|movie|film|tv|drama|word|dict|dictionary|lexicon)[^>]*$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?(?:action|shortcut|deeplink)[^>]*$', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _parse_tag_attrs(attr_str: str) -> dict[str, str]:
        if not attr_str:
            return {}
        pattern = re.compile(
            r'([a-zA-Z0-9_\-]+)\s*=\s*(?:"((?:[^"\\\n]|\\.)*?)"|\'((?:[^\'\\\n]|\\.)*?)\')',
            re.DOTALL
        )
        result = {}
        for m in pattern.finditer(attr_str):
            key = m.group(1).lower()
            val = m.group(2) if m.group(2) is not None else m.group(3)
            val = val.replace(r'\"', '"').replace(r"\'", "'")
            result[key] = val
        return result

    @classmethod
    def _resolve_media_directive(cls, raw_path: str, caption: str = "", delay: float = 1.5) -> Optional[str]:
        if not raw_path or not isinstance(raw_path, str):
            return None
        p = raw_path.strip(" \t\r\n`\"'").strip()
        if not p:
            return None

        md_match = re.match(r'!?\[.*?\]\((.+?)\)', p)
        if md_match:
            p = md_match.group(1).strip(" \t\r\n`\"'").strip()

        if p.lower().startswith("file://"):
            from urllib.parse import unquote
            p = unquote(p[7:].strip())

        cand_dirs = []
        try:
            from hermes_constants import get_hermes_home
            cand_dirs.append(str(get_hermes_home() / "cache"))
            cand_dirs.append(str(get_hermes_home() / "memes"))
        except (ImportError, AttributeError):  # health: allow S110 -- optional hermes home fallback
            pass

        if not os.path.isabs(p) and not p.startswith(('http://', 'https://')):
            for c_dir in cand_dirs:
                cand = os.path.join(c_dir, p)
                if os.path.exists(cand):
                    p = cand
                    break
                elif os.path.exists(cand + ".gif"):
                    p = cand + ".gif"
                    break
                elif os.path.exists(cand + ".png"):
                    p = cand + ".png"
                    break

        if not p.startswith(('http://', 'https://')):
            p = os.path.normpath(p)
            if not os.path.isfile(p):
                return None

        ext = os.path.splitext(p)[1].lower()
        image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.heic', '.tiff')
        if ext in image_exts or p.startswith(('http://', 'https://')):
            payload = json.dumps({"path": p, "caption": caption, "delay": delay}, ensure_ascii=False)
            return f"IMAGE:{payload}"
        else:
            return f"FILE:{p}"

    @staticmethod
    def _is_valid_map_url(url_str: str) -> bool:
        """Validate whether a URL string is a legitimate Apple Maps or Google Maps URL.

        Rejects deceptive domains, embedded credentials, non-standard ports, and unsupported schemes.
        """
        if not url_str or not isinstance(url_str, str):
            return False
        raw = url_str.strip()
        if not (raw.startswith("http://") or raw.startswith("https://") or raw.startswith("maps://")):
            return False
        try:
            parsed = urlsplit(raw)
            scheme = (parsed.scheme or "").lower()
            if scheme not in ("http", "https", "maps"):
                return False
            # Reject embedded credentials (user:pass@)
            if parsed.username or parsed.password:
                return False
            host = (parsed.hostname or "").lower()
            if scheme == "maps":
                # maps://?q=... or maps://maps.apple.com/?q=...
                return (not host or host == "maps.apple.com") and parsed.port is None
            if host in ("maps.apple.com", "maps.google.com"):
                return parsed.port in (None, 80, 443)
            return False
        except Exception:  # health: allow S110, BLE001 -- url split error fallback
            return False

    @staticmethod
    def _escape_vcf_text(val: str, multiline: bool = False) -> str:
        """Escape text for vCard 3.0 per RFC 2426 §2.4.2."""
        if not val:
            return ""
        # 1. Normalize line breaks
        v = val.replace("\r\n", "\n").replace("\r", "\n")
        # 2. Strip non-printable ASCII control characters (keep \n if multiline)
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        # 3. Escape backslashes, semicolons, and commas
        v = v.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
        # 4. Handle line breaks
        if multiline:
            v = v.replace("\n", "\\n")
        else:
            v = re.sub(r"\n+", " ", v).strip()
        return v

    @staticmethod
    def _fold_ics_line(line: str, max_octets: int = 75) -> str:
        """Fold an iCalendar content line to at most 75 octets per RFC 5545 §3.1.

        Inserts CRLF followed by a single space for continuation lines, while strictly
        preserving multi-byte UTF-8 character boundaries.
        """
        line_bytes = line.encode("utf-8")
        if len(line_bytes) <= max_octets:
            return line

        chunks: list[str] = []
        current_line = line
        is_first = True
        while current_line:
            limit = max_octets if is_first else (max_octets - 1)
            cur_bytes = current_line.encode("utf-8")
            if len(cur_bytes) <= limit:
                chunks.append(current_line)
                break

            slice_len = limit
            while slice_len > 0 and (cur_bytes[slice_len] & 0xC0) == 0x80:
                slice_len -= 1

            chunk_str = cur_bytes[:slice_len].decode("utf-8")
            chunks.append(chunk_str)
            current_line = cur_bytes[slice_len:].decode("utf-8")
            is_first = False

        return "\r\n ".join(chunks)

    @staticmethod
    def _escape_ics_text(val: str, multiline: bool = False) -> str:
        """Escape text for iCalendar 2.0 per RFC 5545 §3.3.11."""
        if not val:
            return ""
        # 1. Normalize line breaks
        v = val.replace("\r\n", "\n").replace("\r", "\n")
        # 2. Strip non-printable ASCII control characters (keep \n if multiline)
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        # 3. Escape backslashes, semicolons, and commas
        v = v.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
        # 4. Handle line breaks
        if multiline:
            v = v.replace("\n", "\\n")
        else:
            v = re.sub(r"\n+", " ", v).strip()
        return v

    @staticmethod
    def _generate_vcf(
        name: str,
        phone: str = "",
        email: str = "",
        org: str = "",
        title: str = "",
        address: str = "",
        note: str = "",
    ) -> Optional[str]:
        clean_name = BubbleParser._escape_vcf_text(name, multiline=False) or "新联系人"
        lines = [
            "BEGIN:VCARD",
            "VERSION:3.0",
            "PRODID:-//Hermes Agent//iMessage Contact Card//CN",
            f"FN;CHARSET=UTF-8:{clean_name}",
            f"N;CHARSET=UTF-8:;{clean_name};;;",
        ]
        if org:
            clean_org = BubbleParser._escape_vcf_text(org, multiline=False)
            if clean_org:
                lines.append(f"ORG;CHARSET=UTF-8:{clean_org}")
        if title:
            clean_title = BubbleParser._escape_vcf_text(title, multiline=False)
            if clean_title:
                lines.append(f"TITLE;CHARSET=UTF-8:{clean_title}")
        if phone:
            first_phone = phone.splitlines()[0].strip() if phone else ""
            clean_phone = re.sub(r"[^\d\+\-\s]", "", first_phone).strip()
            if clean_phone:
                lines.append(f"TEL;TYPE=CELL,VOICE:{clean_phone}")
        if email:
            first_email = email.splitlines()[0].strip() if email else ""
            clean_email = re.sub(r"[\x00-\x1f\x7f\s;,:\\]", "", first_email)
            if clean_email:
                lines.append(f"EMAIL;TYPE=INTERNET,WORK:{clean_email}")
        if address:
            clean_addr = BubbleParser._escape_vcf_text(address, multiline=True)
            if clean_addr:
                lines.append(f"ADR;TYPE=WORK;CHARSET=UTF-8:;;{clean_addr};;;;")
        if note:
            clean_note = BubbleParser._escape_vcf_text(note, multiline=True)
            if clean_note:
                lines.append(f"NOTE;CHARSET=UTF-8:{clean_note}")
        lines.append("END:VCARD")
        lines.append("")

        vcf_content = "\r\n".join(lines)
        h = hashlib.sha256(vcf_content.encode("utf-8")).hexdigest()[:8]
        safe_name = re.sub(r'[\\/:*?"<>|\r\n\t]+', '', clean_name).strip()
        safe_name = re.sub(r'\s+', ' ', safe_name)
        safe_name = safe_name[:40].strip(' ._') or "联系人"
        scratch_dir = os.path.join(str(_get_base_scratch_dir()), "contacts")
        target_dir = os.path.join(scratch_dir, h)
        try:
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, f"{safe_name}.vcf")
            with open(file_path, "w", encoding="utf-8", newline="") as fp:
                fp.write(vcf_content)
            return file_path
        except (OSError, ValueError):
            return None

    @classmethod
    def _generate_location_vcf(
        cls,
        title: str,
        query: str = "",
        address: str = "",
        ll: str = "",
    ) -> Optional[str]:
        clean_title = cls._escape_vcf_text(title or query or "地图定位", multiline=False)
        raw_map = (query or "").strip()
        if cls._is_valid_map_url(raw_map):
            maps_url = raw_map
        else:
            encoded_query = quote_plus(raw_map or clean_title, safe=",+")
            maps_url = f"https://maps.apple.com/?q={encoded_query}"
            if address:
                clean_raw_addr = re.sub(r"[\r\n\x00-\x1f\x7f]", "", address.strip())
                maps_url += f"&address={quote_plus(clean_raw_addr, safe=',+')}"
            if ll:
                clean_raw_ll = re.sub(r"[^\d\.\,\-\+]", "", ll.strip())
                if clean_raw_ll:
                    maps_url += f"&ll={clean_raw_ll}"

        lines = [
            "BEGIN:VCARD",
            "VERSION:3.0",
            "PRODID:-//Apple Inc.//macOS 14.5//EN",
            f"N;CHARSET=UTF-8:;{clean_title};;;",
            f"FN;CHARSET=UTF-8:{clean_title}",
            f"item1.URL;type=pref:{maps_url}",
            "item1.X-ABLabel:map url",
        ]
        if ll:
            clean_ll = re.sub(r"[^\d\.\,\-\+]", "", ll.strip())
            if "," in clean_ll:
                lines.append(f"geo:{clean_ll}")
        if address:
            clean_addr = cls._escape_vcf_text(address, multiline=False)
            lines.append(f"item2.ADR;type=pref;CHARSET=UTF-8:;;{clean_addr};;;;")
            lines.append("item2.X-ABLabel:")
        lines.append("END:VCARD")
        lines.append("")

        vcf_content = "\r\n".join(lines)
        h = hashlib.sha256(vcf_content.encode("utf-8")).hexdigest()[:8]
        safe_name = re.sub(r'[\\/:*?"<>|\r\n\t]+', '', clean_title).strip()
        safe_name = re.sub(r'\s+', ' ', safe_name)
        safe_name = safe_name[:40].strip(' ._') or "地图定位"
        scratch_dir = os.path.join(str(_get_base_scratch_dir()), "locations")
        target_dir = os.path.join(scratch_dir, h)
        try:
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, f"{safe_name}.loc.vcf")
            with open(file_path, "w", encoding="utf-8", newline="") as fp:
                fp.write(vcf_content)
            return file_path
        except (OSError, ValueError):
            return None

    @staticmethod
    def _generate_ics(title: str, start_str: str, end_str: str = "", location: str = "", description: str = "") -> Optional[str]:
        def parse_dt(s: str) -> Optional[datetime]:
            s = s.strip()
            try:
                return datetime.fromisoformat(s)
            except (ValueError, TypeError):  # health: allow S110 -- ISO-8601 fallback to format table
                pass
            formats = [
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
                "%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    pass
            return None

        dt_start = parse_dt(start_str) if start_str else None
        if not dt_start:
            now = datetime.now()
            dt_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        dt_end = parse_dt(end_str) if end_str else None
        if not dt_end or dt_end <= dt_start:
            dt_end = dt_start + timedelta(hours=1)

        fmt_ics = "%Y%m%dT%H%M%S"
        if dt_start.tzinfo is not None:
            start_ics = dt_start.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        else:
            start_ics = dt_start.strftime(fmt_ics)

        if dt_end.tzinfo is not None:
            end_ics = dt_end.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        else:
            end_ics = dt_end.strftime(fmt_ics)

        now_ics = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        uid = f"{uuid.uuid4().hex[:12]}@hermes-calendar"

        clean_title = BubbleParser._escape_ics_text(title, multiline=False) or "新日程"
        clean_desc = BubbleParser._escape_ics_text(description, multiline=True) if description else ""
        clean_loc = BubbleParser._escape_ics_text(location, multiline=False) if location else ""

        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Hermes Agent//iMessage Calendar//CN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now_ics}",
            f"DTSTART:{start_ics}",
            f"DTEND:{end_ics}",
            f"SUMMARY:{clean_title}",
        ]
        if clean_desc:
            lines.append(f"DESCRIPTION:{clean_desc}")
        if clean_loc:
            lines.append(f"LOCATION:{clean_loc}")

        lines.extend([
            "STATUS:CONFIRMED",
            "BEGIN:VALARM",
            "TRIGGER:-PT15M",
            "ACTION:DISPLAY",
            f"DESCRIPTION:提醒: {clean_title}",
            "END:VALARM",
            "END:VEVENT",
            "END:VCALENDAR",
        ])

        folded_lines = [BubbleParser._fold_ics_line(ln) for ln in lines]
        ics_content = "\r\n".join(folded_lines) + "\r\n"

        h = hashlib.sha256(ics_content.encode("utf-8")).hexdigest()[:8]
        safe_name = re.sub(r'[\\/:*?"<>|\r\n\t]+', '', clean_title).strip()
        safe_name = re.sub(r'\s+', ' ', safe_name)
        safe_name = safe_name[:40].strip(' ._') or "日程"
        scratch_dir = os.path.join(str(_get_base_scratch_dir()), "events")
        target_dir = os.path.join(scratch_dir, h)
        try:
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, f"{safe_name}.ics")
            with open(file_path, "w", encoding="utf-8", newline="") as fp:
                fp.write(ics_content)
            return file_path
        except (OSError, ValueError):
            return None

    def _is_inside_code_block(self, text: str) -> bool:
        """Check if *text* currently ends inside an open markdown code block."""
        in_code = self._in_code_block
        for line in text.splitlines():
            if "```" in line:
                in_code = not in_code
        return in_code

    @staticmethod
    def _encode_custom_schemes(text: str) -> str:
        """Ensure custom URL schemes (shortcuts://, maps://) have properly percent-encoded query parameters for iOS."""
        def _sub(m: re.Match) -> str:
            raw = m.group(0).strip()
            if not any(ord(c) > 127 for c in raw) and " " not in raw:
                return raw
            from urllib.parse import parse_qsl
            try:
                parsed = urlsplit(raw)
                if not parsed.query:
                    return raw
                pairs = parse_qsl(parsed.query, keep_blank_values=True)
                encoded_query = "&".join(f"{quote(k)}={quote(v, safe=',+')}" for k, v in pairs)
                return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, encoded_query, parsed.fragment))
            except (ValueError, KeyError, AttributeError):
                return raw
        return re.sub(r'(?:(?:shortcuts|maps|mobilenotes)://|https?://(?:maps\.apple\.com|maps\.google\.com|(?:music|podcasts)\.apple\.com)/)[^\r\n<>\'\"()]+', _sub, text)

    @classmethod
    def _normalize_apple_media_url(cls, url: str) -> str:
        """Ensure Apple Music and Apple Podcasts URLs target the Turkey storefront (/tr/) and have percent-encoded query params."""
        if not url:
            return url
        clean = url.strip()
        m = re.match(r'^(https?://(?:music|podcasts)\.apple\.com)(?:/([a-zA-Z]{2}))?(/.*)?$', clean, re.IGNORECASE)
        if m:
            base = m.group(1)
            rest = m.group(3) or ""
            clean = f"{base}/tr{rest}"
        return cls._encode_custom_schemes(clean).strip()

    def _split_standalone_urls(self, text: str) -> list[str]:
        """Split text so bare URLs (or markdown links) become their own discrete bubbles for iMessage Rich Link Previews."""
        if not self.isolate_urls or "```" in text or self._in_code_block:
            return [text]

        text = self._encode_custom_schemes(text)

        # Expand markdown links [text](url) -> "text\nurl"
        def _expand_md(match):
            title = match.group("text").strip()
            url = match.group("url").strip().rstrip(self._TRAILING_PUNCT)
            if title:
                return f"{title}\n{url}\n"
            return f"\n{url}\n"

        expanded = self._MD_LINK_RE.sub(_expand_md, text)

        parts: list[str] = []
        last_idx = 0

        for m in self._BARE_URL_RE.finditer(expanded):
            start, end = m.span()
            raw_url = m.group(0)
            clean_url = raw_url.rstrip(self._TRAILING_PUNCT)
            trailing_punct = raw_url[len(clean_url):]

            pre = expanded[last_idx:start].strip()
            if pre.endswith(("（", "(", "【", "[")):
                pre = pre[:-1].strip()
            if pre:
                parts.append(pre)

            if clean_url:
                if self._has_emitted_map and ("maps.apple.com" in clean_url or "maps.google.com" in clean_url):
                    continue
                if "music.apple.com" in clean_url or "podcasts.apple.com" in clean_url:
                    if "/search?" in clean_url or "/search/" in clean_url:
                        continue
                    clean_url = self._normalize_apple_media_url(clean_url)
                parts.append(clean_url)

            last_idx = end - len(trailing_punct)

        tail = expanded[last_idx:].strip()
        if tail.startswith(("）", ")", "】", "]")):
            tail = tail[1:].strip()
        tail = tail.lstrip(".,;:!?，。！？；：、 ")
        if tail:
            parts.append(tail)

        return parts or [text]

    def _offload_code_blocks(self, text: str) -> list[str]:
        """If a bubble contains a large code block (>15 lines), offload it to a scratch file attachment."""
        matches = list(self._CODE_BLOCK_RE.finditer(text))
        if not matches:
            return [text]

        file_items: list[str] = []
        replaced = text
        for m in reversed(matches):
            lang = (m.group("lang") or "txt").strip()
            code = m.group("code")
            lines = code.splitlines()
            if len(lines) > 15:
                h = hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]
                ext = self._EXT_MAP.get(lang.lower(), "txt")
                scratch_dir = os.path.join(str(_get_base_scratch_dir()), "scratch")
                try:
                    os.makedirs(scratch_dir, exist_ok=True)
                    filename = f"snippet_{h}.{ext}"
                    file_path = os.path.join(scratch_dir, filename)
                    with open(file_path, "w", encoding="utf-8", newline="") as fp:
                        fp.write(code)

                    preview = "\n".join(lines[:5])
                    omitted = len(lines) - 5
                    sub = f"```{lang}\n{preview}\n# ... (已省略其余 {omitted} 行代码，完整脚本见下方附件)\n```"
                    start, end = m.span()
                    replaced = replaced[:start] + sub + replaced[end:]
                    file_items.append(f"FILE:{file_path}")
                except (OSError, ValueError) as exc:
                    logger.debug("Code offload failed: %s", exc)

        return [replaced] + file_items

    def _process_embedded_spoilers(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._SPOILER_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            spoiler_content = (m.group("text") or m.group("pipe_text") or "").strip()
            if spoiler_content:
                if getattr(self._config, "enable_effects", True):
                    results.append(EffectAction(effect_type="invisible", content=spoiler_content))
                else:
                    results.append(spoiler_content)
            last_idx = end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_embedded_effects(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._EFFECT_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            colon_eff = m.group("colon_eff") if "colon_eff" in m.groupdict() else None
            prefix_colon = m.group("prefix_colon") if "prefix_colon" in m.groupdict() else None
            prefix_name = m.group("prefix_name") if "prefix_name" in m.groupdict() else None
            prefix_type = m.group("prefix_type") if "prefix_type" in m.groupdict() else None
            prefix_bare = m.group("prefix_bare") if "prefix_bare" in m.groupdict() else None
            attrs_str = m.group("attrs") if "attrs" in m.groupdict() else None

            eff_name = colon_eff or prefix_colon or prefix_name or prefix_type or prefix_bare
            if not eff_name:
                attrs = self._parse_tag_attrs(attrs_str or "")
                eff_name = attrs.get("name") or attrs.get("type") or "confetti"
            eff_name = self._EFFECT_ALIAS_MAP.get(eff_name.lower(), eff_name.lower())
            eff_text = (m.group("text") or "").strip() if "text" in m.groupdict() and m.group("text") else ""

            if not eff_text:
                tail = text[end:].strip()
                if tail:
                    if getattr(self._config, "enable_effects", True):
                        results.append(EffectAction(effect_type=eff_name, content=tail))
                    else:
                        results.append(tail)
                    last_idx = len(text)
                    break
            else:
                if getattr(self._config, "enable_effects", True):
                    results.append(EffectAction(effect_type=eff_name, content=eff_text))
                else:
                    results.append(eff_text)
            last_idx = end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_embedded_calendars(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._CALENDAR_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            attrs_str = m.group("attrs") or m.group("self_attrs") or ""
            attrs = self._parse_tag_attrs(attrs_str)
            title = attrs.get("title") or attrs.get("summary") or attrs.get("name") or "新日程"
            start_str = attrs.get("start") or attrs.get("time") or attrs.get("date") or attrs.get("start_time") or ""
            end_str = attrs.get("end") or attrs.get("end_time") or ""
            location = attrs.get("location") or attrs.get("loc") or attrs.get("place") or attrs.get("address") or ""
            description = attrs.get("description") or attrs.get("desc") or attrs.get("notes") or attrs.get("note") or ""
            body = (m.group("body") or "").strip() if "body" in m.groupdict() and m.group("body") else ""

            if body:
                results.append(body)
            ics_path = self._generate_ics(title, start_str, end_str, location, description)
            if ics_path:
                results.append(f"FILE:{ics_path}")
            last_idx = end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    @classmethod
    def _build_apple_maps_url(
        cls,
        name: str = "",
        coordinate: str = "",
        address: str = "",
        query: str = "",
    ) -> str:
        raw = (query or name or "").strip()
        if cls._is_valid_map_url(raw):
            return raw

        clean_name = (name or query or "地图定位").strip()
        clean_coord = (coordinate or "").strip()
        clean_addr = (address or "").strip()

        # If raw starts with a URL scheme but failed validation, safely extract readable location query
        if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("maps://"):
            try:
                parsed = urlsplit(raw)
                if parsed.query:
                    from urllib.parse import parse_qs
                    qs_dict = parse_qs(parsed.query)
                    for k in ("q", "query", "address", "name"):
                        if qs_dict.get(k):
                            extracted = qs_dict[k][0].strip()
                            if extracted:
                                clean_name = extracted
                                break
                if clean_name == raw:
                    path_part = parsed.path.strip("/")
                    clean_name = path_part.split("/")[-1] if path_part else "地图定位"
            except Exception:  # health: allow S110, BLE001 -- url parse fallback
                clean_name = "地图定位"

        if not clean_coord:
            haystack = f"{clean_addr} {clean_name} {raw}"
            m = re.search(r'(-?\d{1,3}\.\d{3,8})\s*,\s*(-?\d{1,3}\.\d{3,8})', haystack)
            if m:
                clean_coord = f"{m.group(1)},{m.group(2)}"
            else:
                for k, v in cls._CITY_COORDINATES.items():
                    if k in haystack:
                        clean_coord = v
                        break

        params = []
        if clean_coord:
            params.append(("coordinate", clean_coord))
        if clean_name:
            params.append(("name", clean_name))
        if clean_addr:
            params.append(("address", clean_addr))
        params.append(("map", "explore"))

        qs = "&".join(f"{k}={quote(v, safe=',+')}" for k, v in params)
        return f"https://maps.apple.com/place?{qs}"

    def _process_embedded_maps(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._MAP_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            attrs_str = m.group("attrs") or m.group("self_attrs") or ""
            attrs = self._parse_tag_attrs(attrs_str)
            body = (m.group("body") or "").strip() if "body" in m.groupdict() and m.group("body") else ""
            coord = attrs.get("coordinate") or attrs.get("coords") or attrs.get("ll") or attrs.get("coord") or attrs.get("location") or ""
            if not coord and attrs.get("lat") and (attrs.get("lon") or attrs.get("lng") or attrs.get("long")):
                coord = f"{attrs['lat']},{attrs.get('lon') or attrs.get('lng') or attrs.get('long')}"
            name = attrs.get("name") or attrs.get("title") or attrs.get("query") or attrs.get("q") or body or "地图定位"
            address = attrs.get("address") or attrs.get("addr") or ""
            query = attrs.get("query") or attrs.get("q") or ""

            if not getattr(self._config, "enable_maps", True):
                parts = [f"📍 地图: {name}"]
                if address:
                    parts.append(f"地址: {address}")
                if coord:
                    parts.append(f"坐标: {coord}")
                if body and body != name:
                    parts.append(body)
                fallback = "\n".join(parts)
                results.append(fallback)
            else:
                maps_url = self._build_apple_maps_url(name=name, coordinate=coord, address=address, query=query)

                if body and body != name:
                    results.append(body)
                results.append(maps_url)
                self._has_emitted_map = True
            last_idx = end
            trailing_match = re.match(r'^\s*(?:https?://(?:maps\.apple\.com|maps\.google\.com)[^\s<>"\'\)]*|maps://[^\s<>"\'\)]*)', text[last_idx:])
            if trailing_match:
                last_idx += trailing_match.end()
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_embedded_contacts(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._CONTACT_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            attrs_str = m.group("attrs") or m.group("self_attrs") or ""
            attrs = self._parse_tag_attrs(attrs_str)
            body = (m.group("body") or "").strip() if "body" in m.groupdict() and m.group("body") else ""
            name = attrs.get("name") or attrs.get("fn") or attrs.get("title") or body or "联系人"
            phone = attrs.get("phone") or attrs.get("tel") or attrs.get("mobile") or ""
            email = attrs.get("email") or attrs.get("mail") or ""
            org = attrs.get("org") or attrs.get("company") or attrs.get("organization") or ""
            title = attrs.get("title") or attrs.get("role") or attrs.get("position") or ""
            address = attrs.get("address") or attrs.get("addr") or attrs.get("location") or ""
            note = attrs.get("note") or attrs.get("desc") or attrs.get("description") or ""

            if body:
                results.append(body)
            vcf_path = self._generate_vcf(name, phone, email, org, title, address, note)
            if vcf_path:
                results.append(f"FILE:{vcf_path}")
            last_idx = end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_unsupported_personal_tags(self, text: str) -> list[str]:
        """Safely degrade unsupported personal tags to readable text without errors or Playwright/Pillow."""
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        if not self._GENERIC_UNSUPPORTED_TAG_RE.search(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._GENERIC_UNSUPPORTED_TAG_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            body = (m.group("body") or "").strip() if "body" in m.groupdict() and m.group("body") else ""
            if body:
                results.append(body)
            else:
                attrs_str = m.group("attrs") or ""
                attrs = self._parse_tag_attrs(attrs_str)
                label = (
                    attrs.get("title")
                    or attrs.get("name")
                    or attrs.get("city")
                    or attrs.get("word")
                    or ""
                )
                if label:
                    results.append(label)
            last_idx = end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_embedded_images(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._IMAGE_TAG_RE.finditer(text):
            start, end = m.span()
            pre = text[last_idx:start].strip()
            if pre:
                results.append(pre)
            attrs_str = m.group("attrs") or m.group("self_attrs") or ""
            attrs = self._parse_tag_attrs(attrs_str)
            body = (m.group("body") or "").strip() if "body" in m.groupdict() and m.group("body") else ""
            caption = attrs.get("caption") or attrs.get("title") or body or ""
            if caption:
                results.append(caption)
            last_idx = end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_embedded_media_directives(self, text: str) -> list[str]:
        if "```" in text or self._is_inside_code_block(text):
            return [text]

        if not self._MEDIA_DIRECTIVE_RE.search(text):
            return [text]

        results: list[str] = []
        last_idx = 0
        for m in self._MEDIA_DIRECTIVE_RE.finditer(text):
            pre_str = m.group("pre") or ""
            tag_start = m.start() + len(pre_str)
            tag_end = m.end()
            pre = text[last_idx:tag_start].strip()
            if pre:
                results.append(pre)
            raw_path = m.group("qpath") or m.group("sqpath") or m.group("btpath") or m.group("path") or ""
            norm_item = self._resolve_media_directive(raw_path)
            if norm_item:
                results.append(norm_item)
            last_idx = tag_end
        post = text[last_idx:].strip()
        if post:
            results.append(post)
        return results or [text]

    def _process_embedded_actions(self, text: str) -> str:
        if "```" in text or self._is_inside_code_block(text):
            return text

        def _sub_action(m: re.Match) -> str:
            attrs_str = m.group("attrs") or m.group("self_attrs") or ""
            attrs = self._parse_tag_attrs(attrs_str)
            body = (m.group("body") or "").strip() if "body" in m.groupdict() and m.group("body") else ""
            title = attrs.get("title") or attrs.get("name") or body or "快捷操作"
            url = attrs.get("url") or attrs.get("link") or ""

            shortcut = attrs.get("shortcut") or attrs.get("shortcuts")
            shortcut_input = attrs.get("input") or attrs.get("text") or attrs.get("param") or ""
            from urllib.parse import quote
            if shortcut and not url:
                url = f"shortcuts://run-shortcut?name={quote(shortcut)}"
                if shortcut_input:
                    url += f"&input=text&text={quote(shortcut_input)}"

            map_dest = attrs.get("map") or attrs.get("navigate") or attrs.get("destination") or attrs.get("daddr")
            if map_dest and not url:
                url = f"maps://?daddr={quote(map_dest)}&dirflg=d"

            if attrs.get("reminder") or attrs.get("todo"):
                if not url:
                    url = "x-apple-reminderkit://"

            if attrs.get("note") or attrs.get("notes") or attrs.get("mobilenotes"):
                if not url:
                    url = "mobilenotes://"

            clean_url = url.strip()
            if clean_url:
                return f"\n\n👉 {title}：\n{clean_url}"
            return body or title

        return self._ACTION_RE.sub(_sub_action, text)

    def _apply_pipeline(self, items: list[Any], transform_fn) -> list[Any]:
        out: list[Any] = []
        for it in items:
            if isinstance(it, (EffectAction, PollAction, TapbackAction, TextBubble)) or isinstance(it, str) and it.startswith(("FILE:", "TAPBACK:", "POLL:", "EFFECT:", "IMAGE:")):
                out.append(it)
            else:
                out.extend(transform_fn(str(it)))
        return out

    def _emit_bubble(self, raw_text: str) -> list[Any]:
        """Balance fences, offload large code blocks, unpack embedded effects/spoilers/cards, and isolate URLs."""
        balanced = self._balance_and_track_fences(raw_text)
        balanced = self._encode_custom_schemes(balanced)
        current: list[Any] = self._offload_code_blocks(balanced)

        transforms = [
            self._process_embedded_effects,
            self._process_embedded_spoilers,
            self._process_embedded_calendars,
            self._process_embedded_maps,
            self._process_embedded_contacts,
            self._process_embedded_images,
        ]
        if self._custom_transforms:
            transforms.extend(self._custom_transforms)
        else:
            transforms.append(self._process_unsupported_personal_tags)
        for fn in transforms:
            current = self._apply_pipeline(current, fn)

        results: list[Any] = []
        for item in current:
            if isinstance(item, (EffectAction, PollAction, TapbackAction, TextBubble)):
                results.append(item)
                continue
            if isinstance(item, str) and item.startswith(("FILE:", "TAPBACK:", "POLL:", "EFFECT:", "IMAGE:")):
                results.append(item)
                continue
            act_processed = self._process_embedded_actions(str(item))
            for sub in self._split_standalone_urls(act_processed):
                if sub.startswith(("http://", "https://", "maps://", "shortcuts://")):
                    results.append(sub)
                else:
                    cleaned = self._prune_robotic_boilerplate(sub)
                    if cleaned:
                        results.append(cleaned)

        if self._pending_effect and getattr(self._config, "enable_effects", True) and results:
            for i, res in enumerate(results):
                if not isinstance(res, (PollAction, TapbackAction, EffectAction)):
                    results[i] = EffectAction(effect_type=self._pending_effect, content=str(res))
                    self._pending_effect = None
                    break

        return results

    def _handle_tapback(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_tapback_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_poll(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_poll_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_calendar(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_calendar_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_map(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        _start, end = c_obj.span()
        attrs_str = c_obj.group("attrs") or c_obj.group("self_attrs") or ""
        attrs = self._parse_tag_attrs(attrs_str)
        body = (c_obj.group("body") or "").strip() if "body" in c_obj.groupdict() and c_obj.group("body") else ""
        coord = attrs.get("coordinate") or attrs.get("coords") or attrs.get("ll") or attrs.get("coord") or attrs.get("location") or ""
        if not coord and attrs.get("lat") and (attrs.get("lon") or attrs.get("lng") or attrs.get("long")):
            coord = f"{attrs['lat']},{attrs.get('lon') or attrs.get('lng') or attrs.get('long')}"
        name = attrs.get("name") or attrs.get("title") or attrs.get("query") or attrs.get("q") or body or "地图定位"
        address = attrs.get("address") or attrs.get("addr") or ""
        query = attrs.get("query") or attrs.get("q") or ""
        self._buffer = self._buffer[end:]
        bubbles = []
        if pre_clean:
            bubbles.extend(self._emit_bubble(pre_clean))

        if not getattr(self._config, "enable_maps", True):
            parts = [f"📍 地图: {name}"]
            if address:
                parts.append(f"地址: {address}")
            if coord:
                parts.append(f"坐标: {coord}")
            if body and body != name:
                parts.append(body)
            fallback_text = "\n".join(parts)
            bubbles.extend(self._emit_bubble(fallback_text))
            return bubbles

        maps_url = self._build_apple_maps_url(name=name, coordinate=coord, address=address, query=query)
        if body and body != name:
            bubbles.extend(self._emit_bubble(body))
        bubbles.append(maps_url)
        self._has_emitted_map = True
        return bubbles

    def _handle_effect(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_effect_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_spoiler(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_spoiler_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_apple_maps_url(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        _start, end = c_obj.span()
        raw_url = c_obj.group(0).rstrip(self._TRAILING_PUNCT)
        self._buffer = self._buffer[end:]
        bubbles = []
        if pre_clean and not has_bubble_tags:
            bubbles.extend(self._emit_bubble(pre_clean))
        if not (self._has_emitted_map or "<map" in self._buffer) and raw_url:
            clean_maps = self._encode_custom_schemes(raw_url).strip()
            bubbles.append(clean_maps)
        return bubbles

    def _handle_shortcuts_url(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        _start, end = c_obj.span()
        raw_url = c_obj.group(0).rstrip(self._TRAILING_PUNCT)
        self._buffer = self._buffer[end:]
        bubbles = []
        if pre_clean and not has_bubble_tags:
            bubbles.extend(self._emit_bubble(pre_clean))
        if raw_url:
            clean_sc = self._encode_custom_schemes(raw_url).strip()
            bubbles.append(clean_sc)
        return bubbles

    def _handle_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        _start, end = c_obj.span()
        action_text = c_obj.group(0)
        self._buffer = self._buffer[end:]
        bubbles = []
        if pre_clean and not has_bubble_tags:
            bubbles.extend(self._emit_bubble(pre_clean))
        processed = self._process_embedded_actions(action_text)
        if processed:
            bubbles.extend(self._emit_bubble(processed))
        return bubbles

    def _handle_apple_music_url(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        _start, end = c_obj.span()
        raw_url = c_obj.group(0).rstrip(self._TRAILING_PUNCT)
        self._buffer = self._buffer[end:]
        bubbles = []
        if pre_clean and not has_bubble_tags:
            bubbles.extend(self._emit_bubble(pre_clean))
        if not ("/search?" in raw_url or "/search/" in raw_url) and raw_url:
            clean_music = self._normalize_apple_media_url(raw_url).strip()
            bubbles.append(clean_music)
        return bubbles

    def _handle_contact(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_contact_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_media_directive(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        m_match, _tag_start, tag_end = c_obj
        raw_path = m_match.group("qpath") or m_match.group("sqpath") or m_match.group("btpath") or m_match.group("path") or ""
        self._buffer = self._buffer[tag_end:]
        bubbles = []
        if pre_clean:
            bubbles.extend(self._emit_bubble(pre_clean))
        norm_item = self._resolve_media_directive(raw_path)
        if norm_item:
            bubbles.append(norm_item)
        return bubbles

    def _handle_image(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._handle_image_action(c_obj, pre_clean, has_bubble_tags)]

    def _handle_sep_marker(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[str]:
        self._buffer = self._buffer[c_obj + len(self._SEP_MARKER):]
        bubbles = []
        if pre_clean:
            bubbles.extend(self._emit_bubble(pre_clean))
        return bubbles

    def _handle_bubble_open(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> Optional[list[str]]:
        close_idx = self._find_matching_bubble_close(self._buffer, c_obj)
        if close_idx != -1:
            bubble_raw = self._buffer[c_obj + len(self._OPEN_TAG):close_idx].strip()
            self._buffer = self._buffer[close_idx + len(self._CLOSE_TAG):]
            is_uri_or_action = bool(re.search(r'[a-zA-Z0-9_\-\.]+:[^\s]+', pre_clean))
            bubbles = []
            if pre_clean and (not has_bubble_tags or is_uri_or_action):
                bubbles.extend(self._emit_bubble(pre_clean))
            if bubble_raw:
                bubbles.extend(self._emit_bubble(bubble_raw))
            return bubbles
        return None

    def _find_candidate_tags(self, is_finish: bool) -> list[tuple[int, str, Any]]:
        candidates = []
        checks = [
            (self._TAPBACK_RE, "tapback"),
            (self._POLL_RE, "poll"),
            (self._CALENDAR_RE, "calendar"),
            (self._MAP_RE, "map"),
            (self._EFFECT_RE, "effect"),
            (self._SPOILER_RE, "spoiler"),
            (self._ACTION_RE, "action"),
            (self._CONTACT_RE, "contact"),
            (self._IMAGE_TAG_RE, "image"),
        ]
        for name, (pat, _handler) in self._custom_action_handlers.items():
            checks.append((pat, name))
        for regex, tag_name in checks:
            m = regex.search(self._buffer)
            if m and not self._is_inside_code_block(self._buffer[:m.start()]):
                candidates.append((m.start(), tag_name, m))

        for regex, tag_name in [
            (self._APPLE_MAPS_URL_RE, "apple_maps_url"),
            (self._SHORTCUTS_URL_RE, "shortcuts_url"),
            (self._APPLE_MUSIC_URL_RE, "apple_music_url"),
        ]:
            m = regex.search(self._buffer)
            if m and not self._is_inside_code_block(self._buffer[:m.start()]):
                if m.end() < len(self._buffer) or is_finish:
                    candidates.append((m.start(), tag_name, m))

        open_idx = self._buffer.find(self._OPEN_TAG)
        if open_idx != -1 and open_idx >= 2 and self._buffer[open_idx-2:open_idx+len(self._OPEN_TAG)+2] == self._SEP_MARKER:
            open_idx = -1
        if open_idx != -1 and not self._is_inside_code_block(self._buffer[:open_idx]):
            candidates.append((open_idx, "bubble_open", open_idx))

        sep_idx = self._buffer.find(self._SEP_MARKER)
        if sep_idx != -1 and not self._is_inside_code_block(self._buffer[:sep_idx]):
            candidates.append((sep_idx, "sep_marker", sep_idx))

        return candidates

    def _extract_ready_bubbles(self, is_finish: bool = False) -> list[str]:
        return [self._action_to_legacy_str(a) for a in self._extract_ready_actions(is_finish=is_finish)]

    def _emit_actions(self, raw_text: str) -> list[BubbleAction]:
        """Convert a bubble or paragraph text into typed BubbleAction objects."""
        raw_bubbles = self._emit_bubble(raw_text)
        actions: list[BubbleAction] = []
        for b in raw_bubbles:
            if isinstance(b, (TextBubble, PollAction, TapbackAction, EffectAction)):
                actions.append(b)
            else:
                # All ordinary text (including any text with 'FILE:', 'IMAGE:', 'POLL:', 'TAPBACK:', 'EFFECT:', etc.)
                # is strictly an unprivileged TextBubble!
                actions.append(TextBubble(content=str(b)))
        return actions

    def _handle_tapback_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        _start, end = c_obj.span()
        attr1 = c_obj.group(1)
        body = c_obj.group(2)
        attr2 = c_obj.group(3)
        clean_emoji = self._normalize_tapback_emoji(attr1 or body or attr2 or "")
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if clean_emoji and getattr(self._config, "enable_tapbacks", True):
            actions.append(TapbackAction(reaction=clean_emoji))
        return actions

    def _handle_poll_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        self._has_emitted_poll = True
        _start, end = c_obj.span()
        attr_title = c_obj.group("attr_title")
        body = c_obj.group("body") or ""
        opt_matches = re.findall(r'<option>(.*?)</option>', body, re.IGNORECASE | re.DOTALL)
        if opt_matches:
            options = [o.strip() for o in opt_matches if o.strip()]
        elif "|" in body:
            options = [o.strip() for o in body.split("|") if o.strip()]
        else:
            options = [line.strip().lstrip("1234567890.-*• ") for line in body.splitlines() if line.strip()]
        title = (attr_title or "").strip() or "请选择"
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if not getattr(self._config, "enable_polls", True):
            if options:
                opt_lines = "\n".join(f"{idx+1}. {opt}" for idx, opt in enumerate(options))
                actions.append(TextBubble(content=f"📊 投票: {title}\n{opt_lines}"))
            return actions
        if options:
            actions.append(PollAction(title=title, options=tuple(options)))
        return actions

    def _handle_effect_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        _start, end = c_obj.span()
        gdict = c_obj.groupdict()
        colon_eff = gdict.get("colon_eff")
        prefix_colon = gdict.get("prefix_colon")
        prefix_name = gdict.get("prefix_name")
        prefix_type = gdict.get("prefix_type")
        prefix_bare = gdict.get("prefix_bare")
        attrs_str = gdict.get("attrs") or gdict.get("self_attrs")
        eff_name = colon_eff or prefix_colon or prefix_name or prefix_type or prefix_bare
        if not eff_name:
            attrs = self._parse_tag_attrs(attrs_str or "")
            eff_name = attrs.get("name") or attrs.get("type") or ""
            if not eff_name and attrs_str:
                eff_name = attrs_str.strip().split()[0] if attrs_str.strip() else ""
            if not eff_name:
                eff_name = "confetti"
        eff_name = self._EFFECT_ALIAS_MAP.get(eff_name.lower(), eff_name.lower())
        eff_text = (gdict.get("text") or "").strip()
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if not getattr(self._config, "enable_effects", True):
            if eff_text:
                actions.extend(self._emit_actions(eff_text))
            self._pending_effect = None
            return actions
        if eff_text:
            actions.append(EffectAction(effect_type=eff_name, content=eff_text))
        else:
            self._pending_effect = eff_name
        return actions

    def _handle_spoiler_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        _start, end = c_obj.span()
        spoiler_content = (c_obj.group("text") or c_obj.group("pipe_text") or "").strip()
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if spoiler_content:
            if getattr(self._config, "enable_effects", True):
                actions.append(EffectAction(effect_type="invisible", content=spoiler_content))
            else:
                actions.append(TextBubble(content=spoiler_content))
        return actions

    def _handle_bubble_open_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> Optional[list[BubbleAction]]:
        close_idx = self._find_matching_bubble_close(self._buffer, c_obj)
        if close_idx != -1:
            bubble_raw = self._buffer[c_obj + len(self._OPEN_TAG):close_idx].strip()
            self._buffer = self._buffer[close_idx + len(self._CLOSE_TAG):]
            is_uri_or_action = bool(re.search(r'[a-zA-Z0-9_\-\.]+:[^\s]+', pre_clean))
            actions: list[BubbleAction] = []
            if pre_clean and (not has_bubble_tags or is_uri_or_action):
                actions.extend(self._emit_actions(pre_clean))
            if bubble_raw:
                actions.extend(self._emit_actions(bubble_raw))
            return actions
        return None

    def _handle_sep_marker_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        self._buffer = self._buffer[c_obj + len(self._SEP_MARKER):]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        return actions

    def _handle_calendar_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        _start, end = c_obj.span()
        attrs_str = c_obj.group("attrs") or c_obj.group("self_attrs") or ""
        attrs = self._parse_tag_attrs(attrs_str)
        title = attrs.get("title") or attrs.get("summary") or attrs.get("name") or "新日程"
        start_str = attrs.get("start") or attrs.get("time") or attrs.get("date") or attrs.get("start_time") or ""
        end_str = attrs.get("end") or attrs.get("end_time") or ""
        location = attrs.get("location") or attrs.get("loc") or attrs.get("place") or attrs.get("address") or ""
        description = attrs.get("description") or attrs.get("desc") or attrs.get("notes") or attrs.get("note") or ""
        body = (c_obj.group("body") or "").strip() if "body" in c_obj.groupdict() and c_obj.group("body") else ""
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if not getattr(self._config, "enable_calendar", True):
            parts = [f"📅 日程: {title}"]
            if start_str:
                parts.append(f"时间: {start_str}")
            if location:
                parts.append(f"地点: {location}")
            if body and body != title:
                parts.append(body)
            actions.append(TextBubble(content="\n".join(parts)))
            return actions
        if body:
            actions.extend(self._emit_actions(body))
        ics_path = self._generate_ics(title, start_str, end_str, location, description)
        if ics_path:
            try:
                from gateway.attachment_registry import registry
                from gateway.bubble_actions import FileAttachmentAction
                record, _err = registry.stage_and_authorize(
                    chat_id=self.chat_id,
                    source_path=ics_path,
                    producer="calendar_generator",
                    file_name=f"{title}.ics",
                    turn_id=self.turn_id,
                )
                if record:
                    actions.append(FileAttachmentAction(
                        staged_path=record.staged_path,
                        file_name=record.file_name,
                        token=record.token,
                    ))
            except Exception as e:  # health: allow BLE001 -- calendar action staging warning
                logger.warning("[BubbleParser] Exception staging calendar action: %s", e)
        return actions

    def _handle_map_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        raw_bubbles = self._handle_map(c_obj, pre_clean, has_bubble_tags)
        return [TextBubble(content=b) for b in raw_bubbles]

    def _handle_apple_maps_url_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        raw_bubbles = self._handle_apple_maps_url(c_obj, pre_clean, has_bubble_tags)
        return [TextBubble(content=b) for b in raw_bubbles]

    def _handle_shortcuts_url_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        raw_bubbles = self._handle_shortcuts_url(c_obj, pre_clean, has_bubble_tags)
        return [TextBubble(content=b) for b in raw_bubbles]

    def _handle_action_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        raw_bubbles = self._handle_action(c_obj, pre_clean, has_bubble_tags)
        return [TextBubble(content=b) for b in raw_bubbles]

    def _handle_apple_music_url_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        raw_bubbles = self._handle_apple_music_url(c_obj, pre_clean, has_bubble_tags)
        return [TextBubble(content=b) for b in raw_bubbles]

    def _handle_contact_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        _start, end = c_obj.span()
        attrs_str = c_obj.group("attrs") or c_obj.group("self_attrs") or ""
        attrs = self._parse_tag_attrs(attrs_str)
        body = (c_obj.group("body") or "").strip() if "body" in c_obj.groupdict() and c_obj.group("body") else ""
        name = attrs.get("name") or attrs.get("fn") or attrs.get("title") or body or "联系人"
        phone = attrs.get("phone") or attrs.get("tel") or attrs.get("mobile") or ""
        email = attrs.get("email") or attrs.get("mail") or ""
        org = attrs.get("org") or attrs.get("company") or attrs.get("organization") or ""
        title = attrs.get("title") or attrs.get("role") or attrs.get("position") or ""
        address = attrs.get("address") or attrs.get("addr") or attrs.get("location") or ""
        note = attrs.get("note") or attrs.get("desc") or attrs.get("description") or ""
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if body:
            actions.extend(self._emit_actions(body))
        vcf_path = self._generate_vcf(name, phone, email, org, title, address, note)
        if vcf_path:
            try:
                from gateway.attachment_registry import registry
                from gateway.bubble_actions import FileAttachmentAction
                record, _err = registry.stage_and_authorize(
                    chat_id=self.chat_id,
                    source_path=vcf_path,
                    producer="contact_generator",
                    file_name=f"{name}.vcf",
                    turn_id=self.turn_id,
                )
                if record:
                    actions.append(FileAttachmentAction(
                        staged_path=record.staged_path,
                        file_name=record.file_name,
                        token=record.token,
                    ))
            except Exception as e:  # health: allow BLE001 -- contact action staging warning
                logger.warning("[BubbleParser] Exception staging contact action: %s", e)
        return actions

    def _handle_image_action(self, c_obj: Any, pre_clean: str, has_bubble_tags: bool) -> list[BubbleAction]:
        _start, end = c_obj.span()
        attrs_str = c_obj.group("attrs") or c_obj.group("self_attrs") or ""
        attrs = self._parse_tag_attrs(attrs_str)
        body = (c_obj.group("body") or "").strip() if "body" in c_obj.groupdict() and c_obj.group("body") else ""
        caption = attrs.get("caption") or attrs.get("title") or body or ""
        self._buffer = self._buffer[end:]
        actions: list[BubbleAction] = []
        if pre_clean:
            actions.extend(self._emit_actions(pre_clean))
        if caption:
            actions.append(TextBubble(content=caption))
        return actions

    def _extract_ready_actions(self, is_finish: bool = False) -> list[BubbleAction]:
        actions: list[BubbleAction] = []
        handlers = {
            "tapback": self._handle_tapback_action,
            "poll": self._handle_poll_action,
            "calendar": self._handle_calendar_action,
            "map": self._handle_map_action,
            "effect": self._handle_effect_action,
            "spoiler": self._handle_spoiler_action,
            "apple_maps_url": self._handle_apple_maps_url_action,
            "shortcuts_url": self._handle_shortcuts_url_action,
            "action": self._handle_action_action,
            "apple_music_url": self._handle_apple_music_url_action,
            "contact": self._handle_contact_action,
            "image": self._handle_image_action,
            "sep_marker": self._handle_sep_marker_action,
            "bubble_open": self._handle_bubble_open_action,
        }
        for name, (_pat, handler) in self._custom_action_handlers.items():
            handlers[name] = lambda m, pre, has_bt, h=handler: h(self, m, pre, has_bt)

        while True:
            candidates = self._find_candidate_tags(is_finish)
            if not candidates:
                if self.paragraph_fallback:
                    if re.search(r'</?[a-zA-Z]{0,10}$', self._buffer) or re.search(r'<{1,3}[a-zA-Z]{0,10}$', self._buffer):
                        break
                    stripped = self._buffer.lstrip("\r\n")
                    if len(stripped) < len(self._buffer):
                        self._buffer = stripped
                    nl_idx = self._buffer.find("\n\n")
                    if nl_idx != -1:
                        pre_nl = self._buffer[:nl_idx]
                        if not self._is_inside_code_block(pre_nl):
                            candidate_text = pre_nl.strip()
                            self._buffer = self._buffer[nl_idx + 2:].lstrip("\r\n")
                            if candidate_text:
                                actions.extend(self._emit_actions(candidate_text))
                                continue
                break

            candidates.sort(key=lambda c: c[0])
            start_pos, c_type, c_obj = candidates[0]
            pre = self._buffer[:start_pos]
            has_bubble_tags = (self._OPEN_TAG in self._buffer) or (self._SEP_MARKER in self._buffer)
            pre_clean = pre.strip()

            handler = handlers.get(c_type)
            if handler:
                res = handler(c_obj, pre_clean, has_bubble_tags)
                if res is None:  # open tag waiting for close
                    break
                actions.extend(res)
            else:
                break

        return actions
