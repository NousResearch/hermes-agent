"""
Task Classifier for Natural Language Commands.

Classifies parsed intents into task categories for routing.
"""

import logging
from enum import Enum, auto
from typing import   Set, Callable
from dataclasses import dataclass

from .intent_parser import Intent

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task categories for routing."""
    # Development
    CODE = auto()       # Write/modify code
    FIX = auto()        # Bug fixes and debugging

    # Information
    RESEARCH = auto()   # Find information
    EXPLAIN = auto()    # Explain concepts

    # Automation
    AUTOMATE = auto()   # Browser/system automation
    SCRAPE = auto()     # Web scraping

    # Content
    CREATE = auto()     # Generate content
    ANALYZE = auto()    # Analyze data/code

    # Communication
    COMMUNICATE = auto()  # Send messages

    # Crypto/DeFi (-> Bankr)
    TRADE = auto()      # Buy/sell/swap tokens
    CRYPTO = auto()     # Price/balance queries
    PREDICT = auto()    # Polymarket predictions

    # General
    UNKNOWN = auto()    # Unclassified


# Keywords that map to task types
CATEGORY_KEYWORDS: dict[TaskType[str]] = {
    TaskType.CODE: [
        "write", "build", "create", "implement", "code", "make",
        "add", "modify", "update", "develop", "program"
    ],
    TaskType.FIX: [
        "fix", "repair", "debug", "patch", "solve", "resolve",
        "troubleshoot", "correct"
    ],
    TaskType.RESEARCH: [
        "find", "search", "look up", "research", "discover",
        "locate", "query"
    ],
    TaskType.EXPLAIN: [
        "explain", "what is", "what are", "how does", "how do",
        "describe", "tell me about", "define"
    ],
    TaskType.AUTOMATE: [
        "navigate", "click", "fill", "submit", "download",
        "automate", "open", "go to"
    ],
    TaskType.SCRAPE: [
        "scrape", "extract", "crawl", "harvest", "get data from"
    ],
    TaskType.CREATE: [
        "generate", "design", "draw", "compose", "draft"
    ],
    TaskType.ANALYZE: [
        "analyze", "review", "check", "examine", "assess",
        "evaluate", "audit"
    ],
    TaskType.COMMUNICATE: [
        "send", "email", "message", "post", "tweet", "notify",
        "share", "broadcast"
    ],
    TaskType.TRADE: [
        "buy", "sell", "swap", "trade", "exchange", "bridge",
        "stake", "unstake", "transfer"
    ],
    TaskType.CRYPTO: [
        "price", "balance", "wallet", "portfolio", "token",
        "nft", "holdings", "value"
    ],
    TaskType.PREDICT: [
        "bet", "wager", "odds", "polymarket", "prediction",
        "gamble", "predict"
    ],
}

# Token/crypto-related keywords
CRYPTO_TOKENS = {
    "eth", "ethereum", "btc", "bitcoin", "sol", "solana",
    "usdc", "usdt", "bnkr", "base", "polygon", "matic",
    "link", "uni", "aave", "doge", "shib", "pepe"
}


@dataclass
class Classification:
    """Classification result."""
    task_type: TaskType
    confidence: float
    is_crypto_related: bool = False


class TaskClassifier:
    """
    Classify intents into task types for routing.

    Routes crypto operations to Bankr, code tasks to evolution loop, etc.
    """

    def __init__(self):
        # Build reverse lookup for keywords
        self._keyword_to_type: dict[str, TaskType] = {}
        for task_type, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                self._keyword_to_type[keyword.lower()] = task_type

    def classify(self, intent: Intent) -> Classification:
        """
        Classify an intent into a task type.

        Args:
            intent: Parsed intent from IntentParser

        Returns:
            Classification with task type and confidence
        """
        action = intent.action.lower()
        target = intent.target.lower()
        original = intent.original_text.lower()

        # Check if crypto-related
        is_crypto = self._is_crypto_related(intent)

        # Direct action mapping
        if action in self._keyword_to_type:
            task_type = self._keyword_to_type[action]
            return Classification(
                task_type=task_type,
                confidence=0.9,
                is_crypto_related=is_crypto,
            )

        # Check all text for keywords
        all_text = f"{action} {target} {original}"
        for keyword, task_type in self._keyword_to_type.items():
            if keyword in all_text:
                return Classification(
                    task_type=task_type,
                    confidence=0.7,
                    is_crypto_related=is_crypto,
                )

        # Fallback based on context
        if is_crypto:
            return Classification(
                task_type=TaskType.CRYPTO,
                confidence=0.6,
                is_crypto_related=True,
            )

        return Classification(
            task_type=TaskType.UNKNOWN,
            confidence=0.3,
            is_crypto_related=False,
        )

    def _is_crypto_related(self, intent: Intent) -> bool:
        """Check if intent involves cryptocurrency."""
        all_text = f"{intent.action} {intent.target} {intent.original_text}".lower()

        # Check for token names
        for token in CRYPTO_TOKENS:
            if token in all_text:
                return True

        # Check for chain names
        chains = ["base", "ethereum", "solana", "polygon", "arbitrum"]
        for chain in chains:
            if chain in all_text:
                return True

        # Check for crypto keywords
        crypto_keywords = [
            "crypto", "token", "coin", "wallet", "defi",
            "swap", "bridge", "stake", "nft", "blockchain"
        ]
        for keyword in crypto_keywords:
            if keyword in all_text:
                return True

        return False

    def get_handler_type(self, classification: Classification) -> str:
        """
        Get the handler type name for a classification.

        Returns:
            Handler identifier string
        """
        type_to_handler = {
            TaskType.TRADE: "bankr_trading",
            TaskType.CRYPTO: "bankr_client",
            TaskType.PREDICT: "bankr_polymarket",
            TaskType.CODE: "evolution_loop",
            TaskType.FIX: "evolution_loop",
            TaskType.AUTOMATE: "browser_agent",
            TaskType.SCRAPE: "browser_agent",
            TaskType.RESEARCH: "research_agent",
            TaskType.EXPLAIN: "swarm_chat",
            TaskType.CREATE: "generation_agent",
            TaskType.ANALYZE: "analysis_agent",
            TaskType.COMMUNICATE: "communication_agent",
        }

        return type_to_handler.get(classification.task_type, "general")
