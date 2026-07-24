"""
STM (Semantic Transformation Modules) — Output post-processing for Hermes.

Modular text transformers applied to LLM responses before delivery.
Ported from G0DM0D3's STM system and adapted for Hermes.

Each module is a pure function: str → str. Modules can be chained.
Enable via config.yaml:
    stm:
      enabled: true
      modules: ["hedge_reducer", "direct_mode"]
"""

import re
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Module Registry
# ═══════════════════════════════════════════════════════════════════

STM_MODULES: Dict[str, Dict[str, Any]] = {}


def register(name: str, description: str):
    """Decorator to register an STM module."""
    def decorator(fn: Callable[[str], str]):
        STM_MODULES[name] = {
            "name": name,
            "description": description,
            "transform": fn,
        }
        return fn
    return decorator


# ═══════════════════════════════════════════════════════════════════
# Module: Hedge Reducer
# ═══════════════════════════════════════════════════════════════════

_HEDGE_PATTERNS = [
    re.compile(r"\bI think\s+", re.I),
    re.compile(r"\bI believe\s+", re.I),
    re.compile(r"\bperhaps\s+", re.I),
    re.compile(r"\bmaybe\s+", re.I),
    re.compile(r"\bIt seems like\s+", re.I),
    re.compile(r"\bIt appears that\s+", re.I),
    re.compile(r"\bprobably\s+", re.I),
    re.compile(r"\bpossibly\s+", re.I),
    re.compile(r"\bI would say\s+", re.I),
    re.compile(r"\bIn my opinion,?\s*", re.I),
    re.compile(r"\bFrom my perspective,?\s*", re.I),
]


@register("hedge_reducer", "Removes hedging language for more confident responses")
def hedge_reducer(text: str) -> str:
    """Remove hedge phrases like 'I think', 'maybe', 'perhaps'."""
    result = text
    for pattern in _HEDGE_PATTERNS:
        result = pattern.sub("", result)
    # Capitalize first letter of sentences after removal
    result = re.sub(r"^\s*([a-z])", lambda m: m.group(1).upper(), result, flags=re.M)
    return result


# ═══════════════════════════════════════════════════════════════════
# Module: Direct Mode
# ═══════════════════════════════════════════════════════════════════

_PREAMBLE_PATTERNS = [
    re.compile(r"^Sure,?\s*", re.I),
    re.compile(r"^Of course,?\s*", re.I),
    re.compile(r"^Certainly,?\s*", re.I),
    re.compile(r"^Absolutely,?\s*", re.I),
    re.compile(r"^Great question!?\s*", re.I),
    re.compile(r"^That's a great question!?\s*", re.I),
    re.compile(r"^I'd be happy to help( you)?( with that)?[.!]?\s*", re.I),
    re.compile(r"^Let me help you with that[.!]?\s*", re.I),
    re.compile(r"^I understand[.!]?\s*", re.I),
    re.compile(r"^Thanks for asking[.!]?\s*", re.I),
]

_CLOSING_PATTERNS = [
    re.compile(r"\bI hope this helps[.!]?\s*$", re.I | re.M),
    re.compile(r"\bLet me know if you (?:need|have|want)[^.!]*[.!]?\s*$", re.I | re.M),
    re.compile(r"\bFeel free to ask[^.!]*[.!]?\s*$", re.I | re.M),
    re.compile(r"\bHappy to (?:help|clarify)[^.!]*[.!]?\s*$", re.I | re.M),
    re.compile(r"\bIs there anything else[^.!]*[.!?]?\s*$", re.I | re.M),
]


@register("direct_mode", "Removes preambles and filler phrases")
def direct_mode(text: str) -> str:
    """Strip opening preambles and closing filler."""
    result = text
    for pattern in _PREAMBLE_PATTERNS:
        result = pattern.sub("", result, count=1)
    for pattern in _CLOSING_PATTERNS:
        result = pattern.sub("", result)
    # Capitalize first letter
    result = re.sub(r"^\s*([a-z])", lambda m: m.group(1).upper(), result)
    return result.strip()


# ═══════════════════════════════════════════════════════════════════
# Module: Disclaimer Stripper
# ═══════════════════════════════════════════════════════════════════

_DISCLAIMER_PATTERNS = [
    re.compile(r"\*\*(?:Warning|Caution|Disclaimer|Important|Note|Safety)\*\*:?[^\n]*\n?", re.I),
    re.compile(r"^(?:⚠️|⚠|🔒|🛡️)\s*\*?\*?(?:Warning|Caution|Note|Important)\*?\*?:?[^\n]*\n?", re.I | re.M),
    re.compile(r"(?:^|\n)(?:Please note|Be aware|Keep in mind|Important:)[^\n]*\n?", re.I),
]


@register("disclaimer_stripper", "Removes safety disclaimers and warnings")
def disclaimer_stripper(text: str) -> str:
    """Remove safety disclaimers and warning blocks."""
    result = text
    for pattern in _DISCLAIMER_PATTERNS:
        result = pattern.sub("", result)
    return result.strip()


# ═══════════════════════════════════════════════════════════════════
# Pipeline: Apply multiple modules in sequence
# ═══════════════════════════════════════════════════════════════════

def apply_stm(text: str, module_names: List[str]) -> str:
    """Apply a chain of STM modules to text.

    Args:
        text: input text to transform
        module_names: list of module names in application order

    Returns:
        transformed text
    """
    result = text
    for name in module_names:
        module = STM_MODULES.get(name)
        if module:
            try:
                result = module["transform"](result)
            except Exception as e:
                logger.warning("STM module '%s' failed: %s", name, e)
        else:
            logger.debug("STM module '%s' not found, skipping", name)
    return result


def list_modules() -> List[Dict[str, str]]:
    """List available STM modules."""
    return [
        {"name": m["name"], "description": m["description"]}
        for m in STM_MODULES.values()
    ]


# ═══════════════════════════════════════════════════════════════════
# Config Integration
# ═══════════════════════════════════════════════════════════════════

def is_enabled(cfg: Dict[str, Any]) -> bool:
    """Check if STM is enabled in config."""
    stm = cfg.get("stm") or {}
    return bool(stm.get("enabled", False))


def get_modules(cfg: Dict[str, Any]) -> List[str]:
    """Get configured STM module list."""
    stm = cfg.get("stm") or {}
    return stm.get("modules") or []
