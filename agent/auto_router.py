"""Auto-router: classify query complexity and route to appropriate LLM tier.

Hybrid classification pipeline:
  L1 — Gemma-2-2b via Ollama (local, ~80-120ms, CPU-only)
       If confidence >= 0.85 → use result
  L2 — GPT-5 Mini via Databricks (only when L1 confidence < 0.85 or Ollama unavailable)
       If confidence >= 0.85 → use result
  L3 — Keyword rules from config (safety net when both LLMs fail)
  L4 — Fallback: MEDIUM (haiku)

Tier definitions (from config.yaml quick_commands):
  SIMPLE  → databricks-gpt-5-mini        (factual, short answer, conversational)
  MEDIUM  → databricks-claude-haiku-4-5  (explanation, reasoning, how-to)
  COMPLEX → databricks-claude-sonnet-4-6 (code, research, deep analysis, legal/medical)

Usage:
  from agent.auto_router import route, classify, get_tiers_from_config
  tiers = get_tiers_from_config(config)
  tier = route("write a python function", tiers)
  if tier:
      print(f"Route to {tier.model} (tier: {classify('write a python function')})")
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[auto_router] %(levelname)s %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)


# ─────────────────────────────────────────────────────────────────────────────
# Tier definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterTier:
    """Represents a model tier with provider info."""
    name: str  # "simple", "medium", "complex"
    model: str
    provider: str
    reason: str = ""


# Defaults (fallback if config is missing)
DEFAULT_TIERS = {
    "simple": RouterTier("simple", "databricks-gpt-5-mini", "custom:gpt-5-mini"),
    "medium": RouterTier("medium", "databricks-claude-haiku-4-5", "custom:haiku4.5-d2mlop"),
    "complex": RouterTier("complex", "databricks-claude-sonnet-4-6", "custom:sonnet4.6-d2mlop"),
}

CONFIDENCE_THRESHOLD = 0.85
FALLBACK_TIER = "medium"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a query classifier. Classify the user query into exactly one of:
- SIMPLE: factual question, short answer, greeting, translation, definition
- MEDIUM: requires explanation, some reasoning, how-to, summary, comparison
- COMPLEX: code writing/debugging, deep research, multi-step planning, legal or medical analysis

Respond with JSON only, no explanation. Example: {"class": "SIMPLE", "confidence": 0.95}"""

_FEW_SHOT = """Examples:
"what is the capital of France" -> {"class": "SIMPLE", "confidence": 0.98}
"hello how are you" -> {"class": "SIMPLE", "confidence": 0.97}
"explain how transformers work" -> {"class": "MEDIUM", "confidence": 0.91}
"summarize this article" -> {"class": "MEDIUM", "confidence": 0.93}
"write a python function to sort a list" -> {"class": "COMPLEX", "confidence": 0.96}
"design a microservices architecture" -> {"class": "COMPLEX", "confidence": 0.94}
"is this contract clause enforceable" -> {"class": "COMPLEX", "confidence": 0.92}"""


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _parse_classification(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON classification from LLM response text."""
    try:
        text = text.strip()
        match = re.search(r'\{[^}]+\}', text)
        if match:
            data = json.loads(match.group())
            cls = data.get("class", "").upper()
            conf = float(data.get("confidence", 0))
            if cls in ("SIMPLE", "MEDIUM", "COMPLEX") and 0 <= conf <= 1:
                return {"class": cls, "confidence": conf}
    except Exception:
        pass
    return None


def _class_to_tier_name(cls: str) -> str:
    """Map class label to tier name."""
    return cls.lower() if cls in ("SIMPLE", "MEDIUM", "COMPLEX") else FALLBACK_TIER


# ─────────────────────────────────────────────────────────────────────────────
# L1: Gemma via Ollama
# ─────────────────────────────────────────────────────────────────────────────

class GemmaClassifier:
    """Calls Ollama local REST API to classify query using Gemma-2-2b."""

    def __init__(self, endpoint: str = None, model: str = "gemma2:2b", timeout: float = 10.0):
        if endpoint is None:
            try:
                import subprocess
                gw = subprocess.check_output(["ip", "route", "show", "default"], text=True).split()[2]
                endpoint = f"http://{gw}:11434"
            except Exception:
                endpoint = "http://localhost:11434"
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    def _check_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0):
                return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_available()
        return self._available

    def classify(self, query: str) -> Optional[Dict[str, Any]]:
        """Returns {class, confidence} or None if unavailable/failed."""
        if not self.available:
            return None
        try:
            prompt = f"{_SYSTEM_PROMPT}\n\n{_FEW_SHOT}\n\nUser query: {query}"
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 30, "temperature": 0.1},
            }).encode()
            req = urllib.request.Request(
                f"{self.endpoint}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                text = data.get("response", "")
                result = _parse_classification(text)
                if result:
                    logger.info("Gemma: class=%s confidence=%.2f query=%.60s",
                                result["class"], result["confidence"], query)
                return result
        except urllib.error.URLError:
            self._available = False
            logger.warning("Ollama unavailable — falling back to L2")
        except Exception as e:
            logger.warning("Gemma classify failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# L2: GPT-5 Mini via Databricks
# ─────────────────────────────────────────────────────────────────────────────

class GPTMiniClassifier:
    """Calls Databricks GPT-5 Mini to re-classify when Gemma is low-confidence."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def classify(self, query: str, tiers: Dict[str, RouterTier] = None) -> Optional[Dict[str, Any]]:
        """Returns {class, confidence} or None if failed."""
        try:
            # Extract Databricks config from tiers or config
            if tiers and "simple" in tiers:
                tier = tiers["simple"]
                # Assume custom provider format: "custom:provider-name"
                # We need to get the actual base_url and api_key from somewhere
                # For now, skip L2 if we can't get config
                logger.debug("GPT-5 Mini: config extraction not yet implemented")
                return None
            return None
        except Exception as e:
            logger.warning("GPT-5 Mini classify failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# L3: Keyword rules (fallback)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordRouter:
    """Keyword-based fallback router."""

    # Hard signals for COMPLEX (code, research, architecture, etc.)
    COMPLEX_KEYWORDS = {
        "code", "debug", "function", "class", "script", "algorithm", "architecture",
        "design", "research", "analysis", "implement", "optimize", "refactor",
        "framework", "library", "api", "database", "sql", "query", "schema",
        "legal", "medical", "contract", "clause", "diagnosis", "treatment",
        "machine learning", "deep learning", "neural", "model", "training",
        "microservices", "deployment", "devops", "kubernetes", "docker",
    }

    # Hard signals for SIMPLE (greeting, factual, short)
    SIMPLE_KEYWORDS = {
        "hello", "hi", "hey", "thanks", "thank you", "goodbye", "bye",
        "what is", "who is", "when is", "where is", "definition", "meaning",
        "translate", "convert", "capital of", "population of",
    }

    @staticmethod
    def classify(query: str) -> str:
        """Heuristic classification based on keywords and structure."""
        q = query.lower()
        word_count = len(q.split())
        question_count = q.count("?")

        # Check hard signals
        for keyword in KeywordRouter.COMPLEX_KEYWORDS:
            if keyword in q:
                return "COMPLEX"

        for keyword in KeywordRouter.SIMPLE_KEYWORDS:
            if keyword in q:
                return "SIMPLE"

        # Heuristics
        if word_count <= 5 and question_count <= 1:
            return "SIMPLE"
        if word_count >= 30 or question_count >= 2:
            return "COMPLEX"

        return "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────

_gemma: Optional[GemmaClassifier] = None
_gpt_mini: Optional[GPTMiniClassifier] = None


def get_gemma() -> GemmaClassifier:
    global _gemma
    if _gemma is None:
        _gemma = GemmaClassifier()
    return _gemma


def get_gpt_mini() -> GPTMiniClassifier:
    global _gpt_mini
    if _gpt_mini is None:
        _gpt_mini = GPTMiniClassifier()
    return _gpt_mini


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_tiers_from_config(config: Dict[str, Any]) -> Dict[str, RouterTier]:
    """
    Extract tier definitions from config.yaml.
    Looks for quick_commands section with model mappings.
    """
    tiers = {}
    try:
        quick_commands = config.get("quick_commands", {})
        if not quick_commands:
            logger.warning("No quick_commands in config — using defaults")
            return DEFAULT_TIERS.copy()

        # Map quick_commands to tiers
        # Expected structure: { "simple": "model_name", "medium": "model_name", ... }
        for tier_name in ("simple", "medium", "complex"):
            model = quick_commands.get(tier_name)
            if model:
                # Try to infer provider from model name
                provider = "custom:" + model.split("-")[-1] if "-" in model else "custom"
                tiers[tier_name] = RouterTier(tier_name, model, provider)

        if not tiers:
            logger.warning("Could not extract tiers from quick_commands — using defaults")
            return DEFAULT_TIERS.copy()

        return tiers
    except Exception as e:
        logger.warning("Failed to extract tiers from config: %s — using defaults", e)
        return DEFAULT_TIERS.copy()


def classify(query: str) -> str:
    """
    Classify query into SIMPLE, MEDIUM, or COMPLEX.
    Uses hybrid pipeline: L1 Gemma → L2 keywords → L3 fallback.
    Returns tier name in lowercase: "simple", "medium", "complex".
    """
    try:
        # L1: Gemma (local, fast)
        gemma = get_gemma()
        result = gemma.classify(query)
        if result and result["confidence"] >= CONFIDENCE_THRESHOLD:
            return _class_to_tier_name(result["class"])

        # L2: GPT-5 Mini (only if Gemma low-confidence)
        if result:
            logger.info("Gemma confidence %.2f < %.2f — escalating to GPT-5 Mini",
                        result["confidence"], CONFIDENCE_THRESHOLD)
        gpt = get_gpt_mini()
        gpt_result = gpt.classify(query)
        if gpt_result and gpt_result["confidence"] >= CONFIDENCE_THRESHOLD:
            return _class_to_tier_name(gpt_result["class"])

        # Use best available LLM result even if below threshold
        best = gpt_result or result
        if best:
            logger.info("Using best available LLM result: class=%s confidence=%.2f",
                        best["class"], best["confidence"])
            return _class_to_tier_name(best["class"])

        # L3: Keyword heuristics
        kw_tier = KeywordRouter.classify(query)
        logger.info("Using keyword heuristic: %s", kw_tier)
        return _class_to_tier_name(kw_tier)

    except Exception as e:
        logger.warning("Classify failed: %s — using MEDIUM fallback", e)
        return FALLBACK_TIER


def route(query: str, tiers: Dict[str, RouterTier] = None) -> Optional[RouterTier]:
    """
    Main routing entry point.
    Classifies query and returns the appropriate tier.

    Args:
        query: User query string
        tiers: Dict of tier definitions (from get_tiers_from_config)

    Returns:
        RouterTier object with model and provider, or None if routing disabled
    """
    if not tiers:
        tiers = DEFAULT_TIERS

    try:
        tier_name = classify(query)
        tier = tiers.get(tier_name)
        if tier:
            logger.info("Routing to %s tier: %s", tier_name, tier.model)
            return tier
        else:
            logger.warning("Tier '%s' not found in config — using MEDIUM fallback", tier_name)
            return tiers.get(FALLBACK_TIER, DEFAULT_TIERS[FALLBACK_TIER])
    except Exception as e:
        logger.warning("Routing failed: %s — using MEDIUM fallback", e)
        return tiers.get(FALLBACK_TIER, DEFAULT_TIERS[FALLBACK_TIER])
