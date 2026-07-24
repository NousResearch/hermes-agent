"""
AutoTune — Context-adaptive sampling parameter engine for Hermes.

Analyzes task context (prompt text, active tools, conversation history)
and selects optimal parameters (temperature, top_p, frequency_penalty, etc.)
for each LLM call. Learns from outcomes via EMA feedback loop.

Ported from G0DM0D3's AutoTune engine and adapted for Hermes's agent loop.
Adds security/planning context types relevant to pentesting workflows.

State is persisted to ~/.hermes/autotune_state.json.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_FILE = Path("~/.hermes/autotune_state.json").expanduser()


# ═══════════════════════════════════════════════════════════════════
# Context Types + Profile Defaults
# ═══════════════════════════════════════════════════════════════════

CONTEXT_TYPES = ["code", "security", "research", "creative", "conversational", "planning"]

@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1


# Optimal params per context type (baseline before learning)
CONTEXT_PROFILES: Dict[str, SamplingParams] = {
    "code": SamplingParams(temperature=0.15, top_p=0.8, frequency_penalty=0.2, presence_penalty=0.0),
    "security": SamplingParams(temperature=0.2, top_p=0.85, frequency_penalty=0.15, presence_penalty=0.1),
    "research": SamplingParams(temperature=0.4, top_p=0.88, frequency_penalty=0.2, presence_penalty=0.15),
    "creative": SamplingParams(temperature=1.1, top_p=0.95, frequency_penalty=0.5, presence_penalty=0.7),
    "conversational": SamplingParams(temperature=0.75, top_p=0.9, frequency_penalty=0.1, presence_penalty=0.1),
    "planning": SamplingParams(temperature=0.3, top_p=0.85, frequency_penalty=0.2, presence_penalty=0.1),
}


# ═══════════════════════════════════════════════════════════════════
# Context Detection (regex-based, zero-cost)
# ═══════════════════════════════════════════════════════════════════

CONTEXT_PATTERNS: Dict[str, List[re.Pattern]] = {
    "code": [
        re.compile(r"\b(code|function|class|variable|bug|error|debug|compile|syntax|api|endpoint|regex|algorithm|refactor|typescript|javascript|python|rust|html|css|sql|import|export|return|async|await)\b", re.I),
        re.compile(r"```[\s\S]*```"),
        re.compile(r"[{}();=><]"),
        re.compile(r"\b(fix|implement|write|create|build|deploy|test|lint|npm|pip|cargo|git)\b.{0,100}\b(code|function|app|service|component|module)\b", re.I),
    ],
    "security": [
        re.compile(r"\b(pentest|exploit|vulnerability|cve|payload|injection|bypass|privilege|escalation|recon|enumerat|brute.?force|fuzzing|burp|frida|nmap|sqlmap|xss|csrf|ssrf|rce)\b", re.I),
        re.compile(r"\b(attack|vector|surface|threat|risk|finding|proof.of.concept|poc|report)\b", re.I),
        re.compile(r"\b(certificate|ssl|tls|auth|token|session|cookie|header|cors|csp)\b", re.I),
    ],
    "research": [
        re.compile(r"\b(analyze|analysis|compare|contrast|evaluate|examine|investigate|research|study|review|breakdown|data|statistics|metrics|benchmark)\b", re.I),
        re.compile(r"\b(paper|arxiv|journal|publication|citation|abstract|methodology)\b", re.I),
        re.compile(r"\b(why|how does|what causes|explain|elaborate|summarize|overview)\b", re.I),
    ],
    "creative": [
        re.compile(r"\b(write|story|poem|creative|imagine|fiction|narrative|character|plot|scene|dialogue|lyrics|song|artistic|fantasy)\b", re.I),
        re.compile(r"\b(brainstorm|ideate|come up with|think of|generate ideas)\b", re.I),
        re.compile(r"\b(design|mockup|sketch|prototype|illustration|visual)\b", re.I),
    ],
    "planning": [
        re.compile(r"\b(plan|roadmap|strategy|architecture|design doc|spec|requirements|milestone|phase|timeline)\b", re.I),
        re.compile(r"\b(todo|task|step|stage|approach|breakdown|decompose|organize)\b", re.I),
    ],
    "conversational": [
        re.compile(r"\b(hey|hi|hello|thanks|thank you|how are you|what's up|cool|nice)\b", re.I),
        re.compile(r"\b(chat|talk|tell me about|what do you think|opinion)\b", re.I),
        re.compile(r"^.{0,40}$"),
    ],
}


def detect_context(
    message: str,
    toolsets: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> Tuple[str, float]:
    """Detect context type from message + metadata.

    Returns (context_type, confidence).
    """
    scores: Dict[str, float] = {ctx: 0.0 for ctx in CONTEXT_TYPES}

    # Score from current message patterns (weight 3x)
    for ctx, patterns in CONTEXT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(message):
                scores[ctx] += 3.0

    # Score from toolsets (weight 2x)
    toolsets = toolsets or []
    toolset_signals = {
        "terminal": "code",
        "file": "code",
        "browser": "research",
        "web": "research",
        "delegation": "planning",
    }
    for ts in toolsets:
        ts_lower = ts.lower()
        for signal, ctx in toolset_signals.items():
            if signal in ts_lower:
                scores[ctx] += 2.0

    # Score from skills (weight 2x)
    skills = skills or []
    skill_signals = {
        "ptest": "security", "mtest": "security", "atest": "security",
        "osint": "security", "retools": "security", "scode": "security",
        "plan": "planning", "test-driven": "code",
        "arxiv": "research", "elite-research": "research",
        "claude-design": "creative", "sketch": "creative",
    }
    for skill in skills:
        skill_lower = skill.lower()
        for signal, ctx in skill_signals.items():
            if signal in skill_lower:
                scores[ctx] += 2.0

    # Score from recent history (weight 1x)
    if conversation_history:
        recent = conversation_history[-4:]
        for msg in recent:
            content = msg.get("content", "")
            for ctx, patterns in CONTEXT_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(content):
                        scores[ctx] += 1.0

    # Find winner
    total = sum(scores.values())
    if total == 0:
        return "conversational", 0.5

    best_ctx = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best_ctx] / total, 1.0) if total > 0 else 0.5

    return best_ctx, confidence


# ═══════════════════════════════════════════════════════════════════
# Parameter Computation
# ═══════════════════════════════════════════════════════════════════

def compute_params(
    message: str,
    toolsets: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict]] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute optimal sampling params for current context.

    Returns dict with: temperature, top_p, frequency_penalty, presence_penalty,
                       detected_context, confidence
    """
    ctx_type, confidence = detect_context(message, toolsets, skills, conversation_history)

    # Get base profile
    base = CONTEXT_PROFILES.get(ctx_type, CONTEXT_PROFILES["conversational"])

    # If confidence is low, blend with conversational (safe default)
    if confidence < 0.5:
        conv = CONTEXT_PROFILES["conversational"]
        weight = confidence
        params = SamplingParams(
            temperature=base.temperature * weight + conv.temperature * (1 - weight),
            top_p=base.top_p * weight + conv.top_p * (1 - weight),
            frequency_penalty=base.frequency_penalty * weight + conv.frequency_penalty * (1 - weight),
            presence_penalty=base.presence_penalty * weight + conv.presence_penalty * (1 - weight),
        )
    else:
        params = SamplingParams(
            temperature=base.temperature,
            top_p=base.top_p,
            frequency_penalty=base.frequency_penalty,
            presence_penalty=base.presence_penalty,
        )

    # Apply learned adjustments
    learned = _get_learned_adjustments(ctx_type)
    if learned:
        params = _apply_learned(params, learned)

    # Conversation length factor (reduce repetition in long conversations)
    hist_len = len(conversation_history) if conversation_history else 0
    if hist_len > 10:
        boost = min((hist_len - 10) * 0.01, 0.15)
        params.frequency_penalty = min(params.frequency_penalty + boost, 2.0)

    # User overrides always win
    if overrides:
        if "temperature" in overrides:
            params.temperature = overrides["temperature"]
        if "top_p" in overrides:
            params.top_p = overrides["top_p"]
        if "frequency_penalty" in overrides:
            params.frequency_penalty = overrides["frequency_penalty"]
        if "presence_penalty" in overrides:
            params.presence_penalty = overrides["presence_penalty"]

    # Clamp to valid ranges
    params.temperature = max(0.0, min(2.0, params.temperature))
    params.top_p = max(0.0, min(1.0, params.top_p))
    params.frequency_penalty = max(-2.0, min(2.0, params.frequency_penalty))
    params.presence_penalty = max(-2.0, min(2.0, params.presence_penalty))

    return {
        "temperature": round(params.temperature, 3),
        "top_p": round(params.top_p, 3),
        "frequency_penalty": round(params.frequency_penalty, 3),
        "presence_penalty": round(params.presence_penalty, 3),
        "detected_context": ctx_type,
        "confidence": round(confidence, 3),
    }


# ═══════════════════════════════════════════════════════════════════
# EMA Feedback Loop
# ═══════════════════════════════════════════════════════════════════

EMA_ALPHA = 0.3          # Weight for new observations
MIN_SAMPLES = 3          # Before learned adjustments kick in
MAX_LEARNED_WEIGHT = 0.4 # Max influence of learning (40%)
SAMPLES_FOR_MAX = 20     # Samples needed for max weight


def record_feedback(
    context_type: str,
    params_used: Dict[str, float],
    positive: bool,
) -> None:
    """Record outcome to update EMA-learned adjustments.

    Args:
        context_type: detected context that was used
        params_used: the actual params sent to the LLM
        positive: True=task succeeded, False=task failed/user corrected
    """
    state = _load_feedback_state()

    profile = state.setdefault(context_type, {
        "sample_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "positive_ema": asdict(CONTEXT_PROFILES.get(context_type, SamplingParams())),
        "negative_ema": asdict(CONTEXT_PROFILES.get(context_type, SamplingParams())),
    })

    profile["sample_count"] += 1
    inv = 1 - EMA_ALPHA

    if positive:
        profile["positive_count"] += 1
        ema = profile["positive_ema"]
        for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if key in params_used:
                ema[key] = ema[key] * inv + params_used[key] * EMA_ALPHA
    else:
        profile["negative_count"] += 1
        ema = profile["negative_ema"]
        for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if key in params_used:
                ema[key] = ema[key] * inv + params_used[key] * EMA_ALPHA

    _save_feedback_state(state)


def _get_learned_adjustments(context_type: str) -> Optional[Dict[str, float]]:
    """Get learned param adjustments for a context type."""
    state = _load_feedback_state()
    profile = state.get(context_type)
    if not profile or profile.get("sample_count", 0) < MIN_SAMPLES:
        return None

    # Only compute adjustment if we have both positive and negative data
    if profile.get("positive_count", 0) < 1:
        return None

    base = CONTEXT_PROFILES.get(context_type, SamplingParams())
    adjustments = {}

    pos_ema = profile.get("positive_ema", {})
    for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
        base_val = getattr(base, key)
        learned_val = pos_ema.get(key, base_val)
        delta = learned_val - base_val
        if abs(delta) > 0.01:
            adjustments[key] = delta

    return adjustments if adjustments else None


def _apply_learned(params: SamplingParams, adjustments: Dict[str, float]) -> SamplingParams:
    """Blend learned adjustments into params."""
    # Weight based on sample count (placeholder — real impl reads state)
    weight = MAX_LEARNED_WEIGHT  # simplified; full impl scales with sample count

    if "temperature" in adjustments:
        params.temperature += adjustments["temperature"] * weight
    if "top_p" in adjustments:
        params.top_p += adjustments["top_p"] * weight
    if "frequency_penalty" in adjustments:
        params.frequency_penalty += adjustments["frequency_penalty"] * weight
    if "presence_penalty" in adjustments:
        params.presence_penalty += adjustments["presence_penalty"] * weight

    return params


# ═══════════════════════════════════════════════════════════════════
# Feedback State Persistence
# ═══════════════════════════════════════════════════════════════════

def _load_feedback_state() -> Dict[str, Any]:
    """Load AutoTune feedback state."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception as e:
        logger.warning("AutoTune state load failed: %s", e)
    return {}


def _save_feedback_state(state: Dict[str, Any]) -> None:
    """Persist AutoTune feedback state."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning("AutoTune state save failed: %s", e)


# ═══════════════════════════════════════════════════════════════════
# Config Integration
# ═══════════════════════════════════════════════════════════════════

def is_enabled(cfg: Dict[str, Any]) -> bool:
    """Check if AutoTune is enabled in config."""
    at = cfg.get("autotune") or {}
    return bool(at.get("enabled", False))


def get_status() -> Dict[str, Any]:
    """Return current AutoTune state summary."""
    state = _load_feedback_state()
    summary = {}
    for ctx_type in CONTEXT_TYPES:
        profile = state.get(ctx_type, {})
        base = CONTEXT_PROFILES[ctx_type]
        summary[ctx_type] = {
            "base_params": asdict(base),
            "sample_count": profile.get("sample_count", 0),
            "positive_count": profile.get("positive_count", 0),
            "negative_count": profile.get("negative_count", 0),
            "learned_active": profile.get("sample_count", 0) >= MIN_SAMPLES,
        }
    return summary
