"""
Intent Parser for Natural Language Commands.

Parses user commands to extract structured intents.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Wake words that trigger Cosmos
WAKE_WORDS = [
    "hey farn",
    "hey cosmos",
    "cosmos",
    "farn",
    "professor",
    "hey professor",
]


@dataclass
class Intent:
    """Parsed intent from natural language."""
    action: str  # Primary action verb
    target: str  # What the action applies to
    parameters: dict = field(default_factory=dict)
    original_text: str = ""
    confidence: float = 1.0

    def to_prompt(self) -> str:
        """Convert intent back to natural language prompt."""
        return self.original_text or f"{self.action} {self.target}"


class IntentParser:
    """
    Parse natural language into structured intents.

    Uses pattern matching for common commands and falls back to
    LLM-based parsing for complex queries.
    """

    # Common action patterns
    ACTION_PATTERNS = {
        # Trading
        r"(?:buy|purchase|get) \$?(\d+(?:\.\d+)?)(?: of)? (\w+)": ("buy", {"amount_usd": 1, "token": 2}),
        r"(?:sell|dump) (\d+(?:\.\d+)?)(?: of)? (\w+)": ("sell", {"amount": 1, "token": 2}),
        r"swap (\d+(?:\.\d+)?)(?: of)? (\w+) (?:to|for) (\w+)": ("swap", {"amount": 1, "from_token": 2, "to_token": 3}),
        r"bridge (\d+(?:\.\d+)?)(?: of)? (\w+) (?:from )?(\w+) to (\w+)": ("bridge", {"amount": 1, "token": 2, "from_chain": 3, "to_chain": 4}),

        # Price queries
        r"(?:what(?:'s| is) the )?price (?:of )?(\w+)": ("get_price", {"token": 1}),
        r"how much is (\w+)": ("get_price", {"token": 1}),

        # Balance/portfolio
        r"(?:show|get|what(?:'s| is)) (?:my )?(?:balance|wallet)": ("get_balance", {}),
        r"(?:show|get|what(?:'s| is)) (?:my )?portfolio": ("get_portfolio", {}),

        # Polymarket
        r"(?:what are the )?odds (?:for|on) (.+)": ("get_odds", {"market": 1}),
        r"(?:bet|wager|place) \$?(\d+(?:\.\d+)?) on (.+?) (?:for|in) (.+)": ("place_bet", {"amount_usd": 1, "outcome": 2, "market": 3}),

        # Code/build
        r"(?:build|create|implement|write|code|make) (.+)": ("build", {"target": 1}),
        r"(?:fix|repair|debug) (.+)": ("fix", {"target": 1}),
        r"(?:add|implement) (.+) to (.+)": ("add_feature", {"feature": 1, "target": 2}),

        # Research
        r"(?:find|search|look up|research) (.+)": ("research", {"query": 1}),
        r"what is (.+)": ("explain", {"topic": 1}),
        r"how (?:do|does|to) (.+)": ("explain", {"topic": 1}),

        # Automation
        r"(?:go to|navigate to|open) (.+)": ("navigate", {"url": 1}),
        r"(?:scrape|extract|get data from) (.+)": ("scrape", {"url": 1}),

        # Communication
        r"(?:send|post|tweet|message) (.+)": ("communicate", {"content": 1}),
        r"(?:email|mail) (.+) to (.+)": ("email", {"content": 1, "recipient": 2}),
    }

    def __init__(self, llm_fallback: bool = True):
        self.llm_fallback = llm_fallback
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), action, params)
            for pattern, (action, params) in self.ACTION_PATTERNS.items()
        ]

    def strip_wake_word(self, text: str) -> str:
        """Remove wake words from the beginning of text."""
        text_lower = text.lower().strip()

        for wake_word in sorted(WAKE_WORDS, key=len, reverse=True):
            if text_lower.startswith(wake_word):
                # Remove wake word and dict following punctuation/whitespace
                text = text[len(wake_word):].lstrip(" ,:")
                break

        return text.strip()

    def parse(self, text: str) -> Intent:
        """
        Parse natural language text into an Intent.

        Args:
            text: User's natural language command

        Returns:
            Parsed Intent with action, target, and parameters
        """
        # Strip wake word
        clean_text = self.strip_wake_word(text)

        # Try pattern matching first
        for pattern, action, param_map in self._compiled_patterns:
            match = pattern.search(clean_text)
            if match:
                groups = match.groups()
                parameters = {}

                for param_name, group_idx in param_map.items():
                    if isinstance(group_idx, int) and group_idx <= len(groups):
                        value = groups[group_idx - 1]
                        # Try to convert to number if applicable
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass
                        parameters[param_name] = value

                return Intent(
                    action=action,
                    target=parameters.get("target", parameters.get("token", "")),
                    parameters=parameters,
                    original_text=clean_text,
                    confidence=0.9,
                )

        # No pattern match - create generic intent
        # First word is likely the action
        words = clean_text.split()
        action = words[0].lower() if words else "unknown"
        target = " ".join(words[1:]) if len(words) > 1 else ""

        return Intent(
            action=action,
            target=target,
            parameters={},
            original_text=clean_text,
            confidence=0.5,
        )

    def has_wake_word(self, text: str) -> bool:
        """Check if text starts with a wake word."""
        text_lower = text.lower().strip()
        return any(ww in WAKE_WORDS)

    async def parse_with_llm(self, text: str) -> Intent:
        """
        Use LLM for complex intent parsing.

        Falls back to this when pattern matching fails or confidence is low.
        """
        # Try to import local LLM
        try:
            from Cosmos.core.local_llm import get_local_llm
            llm = get_local_llm()

            prompt = f"""Extract the intent from this command:
            "{text}"

            Return JSON with:
            - action: the main verb/action
            - target: what the action applies to
            - parameters: relevant extracted values

            JSON only, no explanation:"""

            response = await llm.generate(prompt)

            # Parse JSON response
            import json
            data = json.loads(response)

            return Intent(
                action=data.get("action", "unknown"),
                target=data.get("target", ""),
                parameters=data.get("parameters", {}),
                original_text=text,
                confidence=0.8,
            )

        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}")
            # Fall back to basic parsing
            return self.parse(text)
