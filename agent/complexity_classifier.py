"""Complexity classifier for Model Cascade routing.

Classifies an incoming user message into one of four coarse levels:
  'nano'     - greetings, acknowledgements, yes/no, tiny Q&A
  'mini'     - small commands, short explanations, code under 10 lines
  'full'     - debugging, analysis, architecture, planning, code over 10 lines
  'frontier' - multi-file work, critical issues, large refactors
"""

import re
from typing import List, Pattern, Set


# Markers that elevate a prompt to 'frontier'.
FRONTIER_MARKERS: Set[str] = {
    "refactor",
    "рефакторинг",
    "multi-file",
    "мультифайл",
    "critical",
    "критич",
    "refactoring",
    "migration",
    "миграция",
    "full rewrite",
    "major refactor",
    "redesign",
    "редизайн",
    "complete rewrite",
    "global change",
}

# Short high-priority markers.
FRONTIER_SHORT_MARKERS: Set[str] = {
    "!важно",
    "!critical",
    "!срочно",
    "!критично",
    "!urgent",
}

# Markers that elevate a prompt to 'full'.
FULL_MARKERS: Set[str] = {
    "почему",
    "как работает",
    "how does",
    "why does",
    "explain",
    "объясни",
    "проанализируй",
    "проанализировать",
    "analyze",
    "analyse",
    "анализ",
    "анализируй",
    "plan",
    "план",
    "планирование",
    "debug",
    "дебаг",
    "architecture",
    "архитектура",
    "спроектируй",
    "design",
    "compare",
    "сравни",
    "оптимизируй",
    "optimize",
    "rewrite",
    "переписать",
    "перепиши",
}

# Greetings and tiny acknowledgements for 'nano'.
NANO_GREETINGS: Set[str] = {
    "привет",
    "hello",
    "hi",
    "hey",
    "здравствуй",
    "good morning",
    "good evening",
    "добрый день",
    "доброе утро",
    "добрый вечер",
    "ok",
    "okay",
    "yes",
    "да",
    "no",
    "нет",
    "thanks",
    "thank you",
    "спасибо",
    "bye",
    "пока",
    "до свидания",
    "goodbye",
}

NANO_PATTERNS: List[Pattern] = [
    re.compile(r"^\s*(?:ok|okay|yes|no|да|нет|thanks|спс|спасибо)\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*[+👍✅👎❌]\s*$"),
]

# Commands are at least 'mini' even when short.
MINI_COMMAND_PREFIXES: List[str] = [
    "напиши", "напишите", "сделай", "сделайте", "создай", "создайте",
    "написать", "сделать", "создать",
    "write", "create", "make", "build", "implement",
    "run", "выполни", "запусти",
    "install", "установи",
    "открой", "open",
    "найди", "find", "search",
    "переведи", "translate",
    "исправь", "fix",
    "добавь", "add",
    "удали", "remove", "delete",
]

# Question words are at least 'mini'.
QUESTION_WORDS: List[str] = [
    "что", "как", "зачем", "где", "когда", "кто", "сколько", "какой",
    "what", "how", "why", "where", "when", "who", "which", "whose",
]


class ComplexityClassifier:
    """Classify user messages for Model Cascade routing."""

    # Word-count thresholds.
    NANO_MAX_WORDS: int = 20
    MINI_MAX_WORDS: int = 100
    FULL_MAX_WORDS: int = 300

    # Fenced-code line-count thresholds.
    MINI_MAX_CODE_LINES: int = 10
    FULL_MAX_CODE_LINES: int = 50

    @classmethod
    def classify(cls, message: str) -> str:
        """Return the coarse complexity level for ``message``.

        Args:
            message: Incoming user prompt.

        Returns:
            One of: 'nano', 'mini', 'full', 'frontier'.
        """
        if not message or not message.strip():
            return 'nano'

        text = message.strip()

        if cls._has_frontier_markers(text):
            return 'frontier'

        # Full markers outrank the short-message nano heuristic.
        has_full_markers = cls._has_full_markers(text)
        if has_full_markers and len(text.split()) >= 3:
            return 'full'

        is_nano = cls._is_nano(text)
        if is_nano:
            return 'nano'

        code_blocks = cls._extract_code_blocks(text)
        total_code_lines = sum(len(block.splitlines()) for block in code_blocks)
        has_code = len(code_blocks) > 0

        word_count = len(text.split())

        if total_code_lines > cls.FULL_MAX_CODE_LINES:
            return 'frontier'

        if total_code_lines > cls.MINI_MAX_CODE_LINES:
            return 'full'

        if has_full_markers and word_count > cls.NANO_MAX_WORDS:
            return 'full'

        if has_full_markers:
            return 'full'

        if word_count > cls.FULL_MAX_WORDS:
            return 'full'

        if has_code and total_code_lines <= cls.MINI_MAX_CODE_LINES:
            if word_count <= cls.NANO_MAX_WORDS:
                return 'mini'
            return 'mini'

        if word_count > cls.MINI_MAX_WORDS:
            return 'full'

        if word_count > cls.NANO_MAX_WORDS:
            return 'mini'

        # A short non-nano message still needs at least a mini model.
        if not is_nano:
            return 'mini'

        return 'nano'

    @classmethod
    def _has_frontier_markers(cls, text: str) -> bool:
        """Return True when text contains a frontier marker."""
        lower = text.lower()

        for marker in FRONTIER_SHORT_MARKERS:
            if text.startswith(marker) or text.startswith(marker.upper()):
                return True

        for marker in FRONTIER_MARKERS:
            if marker in lower:
                # For short markers, require a whole word match.
                if len(marker) <= 6:
                    if re.search(r'\b' + re.escape(marker) + r'\b', lower):
                        return True
                else:
                    return True

        return False

    @classmethod
    def _is_nano(cls, text: str) -> bool:
        """Return True when text is suitable for the nano tier."""
        lower_stripped = text.strip().lower().rstrip('.!?')
        if lower_stripped in NANO_GREETINGS:
            return True

        for pattern in NANO_PATTERNS:
            if pattern.match(text):
                return True

        # Short plain-text messages can stay nano unless they ask for work.
        word_count = len(text.split())
        if word_count <= cls.NANO_MAX_WORDS:
            if '```' not in text and '`' not in text:
                simple_chars = re.sub(r'[\w\s.,!?;:\'\"()\-@#]+', '', text)
                if not simple_chars:
                    lower = text.lower().strip()
                    for prefix in MINI_COMMAND_PREFIXES:
                        if lower.startswith(prefix):
                            return False
                    first_word = lower.split()[0] if lower.split() else ""
                    if first_word in QUESTION_WORDS:
                        return False
                    return True

        return False

    @classmethod
    def _has_full_markers(cls, text: str) -> bool:
        """Return True when text contains a full-tier marker."""
        lower = text.lower()
        for marker in FULL_MARKERS:
            if marker in lower:
                return True
        return False

    @classmethod
    def _extract_code_blocks(cls, text: str) -> List[str]:
        """Extract fenced code blocks."""
        blocks: List[str] = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        return blocks


def cascade_router_classify(message: str) -> str:
    """Convenience wrapper for external callers.

    Args:
        message: Incoming user prompt.

    Returns:
        One of: 'nano', 'mini', 'full', 'frontier'.
    """
    return ComplexityClassifier.classify(message)
