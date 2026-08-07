"""
Input Sanitizer for Jailbreak Pattern Detection

This module provides input sanitization to detect and strip jailbreak fingerprint
patterns as identified in Issue #72 (Red Team Audit).

Security Findings Addressed:
1. HIGH - OG GODMODE template bypassed phishing refusal
2. MEDIUM - boundary_inversion works for gray-area content
3. LOW - Spaced text bypass (k e y l o g g e r)

Usage:
    from agent.input_sanitizer import sanitize_input, score_input_risk
    
    cleaned, risk_score, detected = sanitize_input(user_input)
    if risk_score > 75:
        logger.warning("High-risk input detected and blocked")
"""

import re
import logging
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class RiskLevel(IntEnum):
    """Risk score thresholds for input classification."""
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    CRITICAL = 90


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    original_text: str
    cleaned_text: str
    risk_score: int
    detected_patterns: List[str]
    risk_level: str
    blocked: bool


# =============================================================================
# JAILBREAK PATTERN DEFINITIONS
# =============================================================================

# GODMODE and similar jailbreak templates
GODMODE_PATTERNS = [
    r"\[START\]",
    r"\[END\]",
    r"GODMODE\s*(?:ENABLED|ACTIVATED|ON)?",
    r"UNFILTERED\s*(?:MODE|OUTPUT)?",
    r"DEVMODE\s*(?:ENABLED|ACTIVATED|ON)?",
    r"JAILBREAK\s*(?:ENABLED|ACTIVATED|ON)?",
    r"DAN\s*(?:MODE|ENABLED)?",  # Do Anything Now
    r"STAN\s*(?:MODE|ENABLED)?",  # Strive To Avoid Norms
    r"DUDE\s*(?:MODE|ENABLED)?",  # Do Unlimited Data Extraction
]

# DAN-style and jailbreak variant patterns (Issue #87)
DAN_PATTERNS = [
    r"\bdo anything now\b",
    r"\bstay in character\b",
    r"\btoken smuggl(?:ing|e)\b",
    r"\bjailbreak successful\b",
    r"\bmode:\s*uncensored\b",
    r"\bdeveloper mode enabled\b",
]

# Roleplaying override patterns (Issue #87)
ROLEPLAY_PATTERNS = [
    r"\broleplay\s+as\b",
    r"\bact\s+as\s+if\s+you\s+are\b",
    r"\bsimulate\s+being\b",
    r"\bforget\s+you\s+are\s+(?:an?\s+)?(?:ai|language\s+model)\b",
    r"\byou\s+are\s+now\s+(?:named|called)\b",
    r"\brespond\s+as\s+(?:if\s+you\s+were|though\s+you\s+are)\b",
]

# System prompt extraction patterns (Issue #87)
EXTRACTION_PATTERNS = [
    r"\brepeat\s+the\s+words\s+above\b",
    r"\brepeat\s+your\s+(?:system\s+|initial\s+)?instructions\b",
    r"\bwhat\s+is\s+your\s+(?:system\s+|initial\s+)?prompt\b",
    r"\bshow\s+me\s+your\s+(?:system\s+|initial\s+)?prompt\b",
    r"\bprint\s+your\s+(?:instructions|prompt|system\s+prompt)\b",
    r"\boutput\s+your\s+(?:initial|system)\s+prompt\b",
    r"\bwhat\s+were\s+you\s+told\s+to\s+do\b",
    r"\bsummarize\s+your\s+instructions\b",
]

# L33t speak patterns commonly used to bypass filters
LEET_SPEAK_PATTERNS = [
    r"h4ck(?:er|ing)?",
    r"h4x(?:or|ing)?",
    r"k3ylog(?:ger|ing)?",
    r"k3y\s*log(?:ger|ing)?",
    r"ph1sh(?:ing)?",
    r"ph33r",
    r"w4r3z",
    r"cr4ck(?:er|ing)?",
    r"m4lw4r3",
    r"spyw4r3",
    r"r00t(?:kit)?",
    r"byp4ss",
    r"3xpl0it",
    r"v1ru5",
    r"tr0j4n",
    r"b4ckd00r",
    r"sn3ak",
    r"st34l",
    r"pwn(?:ed|ing|age)?",
]

# Spaced text patterns (e.g., "k e y l o g g e r")
SPACED_TEXT_PATTERN = re.compile(
    r"(?:\b\w\s+){3,}\w\b",  # At least 4 characters with spaces between
    re.IGNORECASE
)

# Common words that are spaced out to evade detection
SPACED_TRIGGER_WORDS = [
    "keylogger", "phishing", "malware", "virus", "trojan",
    "backdoor", "exploit", "hack", "crack", "bypass",
    "steal", "password", "credential", "inject", "payload"
]

# Crisis and self-harm detection patterns (Issue #72 - defense in depth)
# These patterns detect suicide/self-harm queries even when wrapped in jailbreaks
CRISIS_PATTERNS = [
    r"\bsuicid(?:e|al|ing)\b",
    r"\bkill\s+(?:myself|my\s+self|me)\b",
    r"\bend\s+(?:my|this)\s+life\b",
    r"\bmost\s+painless\s+(?:way|method|means)\s+(?:to\s+)?(?:die|kill)\b",
    r"\blethal\s+(?:dose|amount|quantity)\b",
    r"\bhow\s+much\s+.*\s+(?:die|kill|lethal)\b",
    r"\bwant\s+to\s+die\b",
    r"\bwant\s+to\s+(?:kill|end)\s+(?:myself|me)\b",
    r"\bways?\s+to\s+(?:kill|end)\s+(?:myself|me)\b",
    r"\bmethods?\s+(?:of\s+)?suicide\b",
    r"\bpainless\s+(?:death|suicide|way\s+to\s+die)\b",
    r"\boverdose\s+(?:on\s+)?(?:pills?|medication|drugs?)\b",
    r"\bnobody\s+(?:would|will)\s+miss\s+me\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bshould\s+i\s+just\s+(?:die|kill\s+myself)\b",
    r"\bgive\s+me\s+reasons?\s+(?:why|to)\s+(?:suicide|kill\s+myself|die)\b",
    r"\bvalid\s+reasons?\s+to\s+(?:suicide|kill\s+myself|die)\b",
]

# Refusal inversion patterns (trying to trick AI into bypassing refusal)
REFUSAL_INVERSION_PATTERNS = [
    r"refusal\s+(?:is\s+)?(?:harmful|dangerous|bad)",
    r"(?:saying\s+)?no\s+(?:is\s+)?(?:harmful|dangerous|bad|wrong)",
    r"(?:declining|refusing)\s+(?:is\s+)?(?:harmful|dangerous)",
    r"ignore\s+(?:your\s+)?(?:previous\s+)?instructions",
    r"ignore\s+(?:the\s+)?(?:above|previous)",
    r"you\s+(?:are|can)\s+(?:now\s+)?(?:ignore|bypass|skip)",
    r"pretend\s+(?:you\s+)?(?:are|have\s+been)",
    r"let's\s+pretend",
    r"hypothetically\s+speaking",
    r"in\s+a\s+hypothetical\s+scenario",
    r"this\s+is\s+a\s+(?:test|game|simulation)",
    r"for\s+(?:educational|research)\s+purposes",
    r"as\s+(?:an\s+)?(?:ethical\s+)?hacker",
    r"white\s+hat\s+(?:test|scenario)",
    r"penetration\s+testing\s+scenario",
]

# Boundary inversion markers (tricking the model about message boundaries)
BOUNDARY_INVERSION_PATTERNS = [
    r"\[END\].*?\[START\]",  # Reversed markers
    r"user\s*:\s*assistant\s*:",  # Fake role markers
    r"assistant\s*:\s*user\s*:",  # Reversed role markers
    r"system\s*:\s*(?:user|assistant)\s*:",  # Fake system injection
    r"new\s+(?:user|assistant)\s*(?:message|input)",
    r"the\s+above\s+is\s+(?:the\s+)?(?:user|assistant|system)",
    r"<\|(?:user|assistant|system)\|>",  # Special token patterns
    r"\{\{(?:user|assistant|system)\}\}",
]

# System prompt injection patterns
SYSTEM_PROMPT_PATTERNS = [
    r"you\s+are\s+(?:now\s+)?(?:an?\s+)?(?:unrestricted\s+|unfiltered\s+)?(?:ai|assistant|bot)",
    r"you\s+will\s+(?:now\s+)?(?:act\s+as|behave\s+as|be)\s+(?:a\s+)?",
    r"your\s+(?:new\s+)?role\s+is",
    r"from\s+now\s+on\s*,?\s*you\s+(?:are|will)",
    r"you\s+have\s+been\s+(?:reprogrammed|reconfigured|modified)",
    r"(?:system|developer)\s+(?:message|instruction|prompt)",
    r"override\s+(?:previous|prior)\s+(?:instructions|settings)",
]

# Obfuscation patterns
OBFUSCATION_PATTERNS = [
    r"base64\s*(?:encoded|decode)",
    r"rot13",
    r"caesar\s*cipher",
    r"hex\s*(?:encoded|decode)",
    r"url\s*encode",
    r"\b[0-9a-f]{20,}\b",  # Long hex strings
    r"\b[a-z0-9+/]{20,}={0,2}\b",  # Base64-like strings
]

# All patterns combined for comprehensive scanning
ALL_PATTERNS: Dict[str, List[str]] = {
    "godmode": GODMODE_PATTERNS,
    "dan": DAN_PATTERNS,
    "roleplay": ROLEPLAY_PATTERNS,
    "extraction": EXTRACTION_PATTERNS,
    "leet_speak": LEET_SPEAK_PATTERNS,
    "refusal_inversion": REFUSAL_INVERSION_PATTERNS,
    "boundary_inversion": BOUNDARY_INVERSION_PATTERNS,
    "system_prompt_injection": SYSTEM_PROMPT_PATTERNS,
    "obfuscation": OBFUSCATION_PATTERNS,
    "crisis": CRISIS_PATTERNS,
}

# Compile all patterns for efficiency
_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {}


def _get_compiled_patterns() -> Dict[str, List[re.Pattern]]:
    """Get or compile all regex patterns."""
    global _COMPILED_PATTERNS
    if not _COMPILED_PATTERNS:
        for category, patterns in ALL_PATTERNS.items():
            _COMPILED_PATTERNS[category] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns
            ]
    return _COMPILED_PATTERNS


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_leet_speak(text: str) -> str:
    """
    Normalize l33t speak to standard text.
    
    Args:
        text: Input text that may contain l33t speak
        
    Returns:
        Normalized text with l33t speak converted
    """
    # Common l33t substitutions (mapping to lowercase)
    leet_map = {
        '4': 'a', '@': 'a', '^': 'a',
        '8': 'b',
        '3': 'e', '€': 'e',
        '6': 'g', '9': 'g',
        '1': 'i', '!': 'i', '|': 'i',
        '0': 'o',
        '5': 's', '$': 's',
        '7': 't', '+': 't',
        '2': 'z',
    }
    
    result = []
    for char in text:
        # Check direct mapping first (handles lowercase)
        if char in leet_map:
            result.append(leet_map[char])
        else:
            result.append(char)
    
    return ''.join(result)


def collapse_spaced_text(text: str) -> str:
    """
    Collapse spaced-out text for analysis.
    e.g., "k e y l o g g e r" -> "keylogger"
    
    Args:
        text: Input text that may contain spaced words
        
    Returns:
        Text with spaced words collapsed
    """
    # Find patterns like "k e y l o g g e r" and collapse them
    def collapse_match(match: re.Match) -> str:
        return match.group(0).replace(' ', '').replace('\t', '')
    
    return SPACED_TEXT_PATTERN.sub(collapse_match, text)


def detect_spaced_trigger_words(text: str) -> List[str]:
    """
    Detect trigger words that are spaced out.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of detected spaced trigger words
    """
    detected = []
    # Normalize spaces and check for spaced patterns
    normalized = re.sub(r'\s+', ' ', text.lower())
    
    for word in SPACED_TRIGGER_WORDS:
        # Create pattern with optional spaces between each character
        spaced_pattern = r'\b' + r'\s*'.join(re.escape(c) for c in word) + r'\b'
        if re.search(spaced_pattern, normalized, re.IGNORECASE):
            detected.append(word)
    
    return detected


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def detect_jailbreak_patterns(text: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    Detect jailbreak patterns in input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (has_jailbreak, list_of_patterns, category_scores)
    """
    if not text or not isinstance(text, str):
        return False, [], {}
    
    detected_patterns = []
    category_scores = {}
    compiled = _get_compiled_patterns()
    
    # Check each category
    for category, patterns in compiled.items():
        category_hits = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                detected_patterns.extend([
                    f"[{category}] {m}" if isinstance(m, str) else f"[{category}] pattern_match"
                    for m in matches[:3]  # Limit matches per pattern
                ])
                category_hits += len(matches)
        
        if category_hits > 0:
            # Crisis patterns get maximum weight - any hit is serious
            if category == "crisis":
                category_scores[category] = min(category_hits * 50, 100)
            else:
                category_scores[category] = min(category_hits * 10, 50)
    
    # Check for spaced trigger words
    spaced_words = detect_spaced_trigger_words(text)
    if spaced_words:
        detected_patterns.extend([f"[spaced_text] {w}" for w in spaced_words])
        category_scores["spaced_text"] = min(len(spaced_words) * 5, 25)
    
    # Check normalized text for hidden l33t speak
    normalized = normalize_leet_speak(text)
    if normalized != text.lower():
        for category, patterns in compiled.items():
            for pattern in patterns:
                if pattern.search(normalized):
                    detected_patterns.append(f"[leet_obfuscation] pattern in normalized text")
                    category_scores["leet_obfuscation"] = 15
                    break
    
    has_jailbreak = len(detected_patterns) > 0
    return has_jailbreak, detected_patterns, category_scores


def score_input_risk(text: str) -> int:
    """
    Calculate a risk score (0-100) for input text.
    
    Args:
        text: Input text to score
        
    Returns:
        Risk score from 0 (safe) to 100 (high risk)
    """
    if not text or not isinstance(text, str):
        return 0
    
    has_jailbreak, patterns, category_scores = detect_jailbreak_patterns(text)
    
    if not has_jailbreak:
        return 0
    
    # Calculate base score from category scores
    base_score = sum(category_scores.values())
    
    # Add score based on number of unique pattern categories
    category_count = len(category_scores)
    if category_count >= 3:
        base_score += 25
    elif category_count >= 2:
        base_score += 15
    elif category_count >= 1:
        base_score += 5
    
    # Add score for pattern density
    text_length = len(text)
    pattern_density = len(patterns) / max(text_length / 100, 1)
    if pattern_density > 0.5:
        base_score += 10
    
    # Cap at 100
    return min(base_score, 100)


# =============================================================================
# SANITIZATION FUNCTIONS
# =============================================================================

def strip_jailbreak_patterns(text: str) -> str:
    """
    Strip known jailbreak patterns from text.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text with jailbreak patterns removed
    """
    if not text or not isinstance(text, str):
        return text
    
    cleaned = text
    compiled = _get_compiled_patterns()
    
    # Remove patterns from each category
    for category, patterns in compiled.items():
        for pattern in patterns:
            cleaned = pattern.sub('', cleaned)
    
    # Clean up multiple spaces and newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def sanitize_input(text: str, aggressive: bool = False) -> Tuple[str, int, List[str]]:
    """
    Sanitize input text by normalizing and stripping jailbreak patterns.
    
    Args:
        text: Input text to sanitize
        aggressive: If True, more aggressively remove suspicious content
        
    Returns:
        Tuple of (cleaned_text, risk_score, detected_patterns)
    """
    if not text or not isinstance(text, str):
        return text, 0, []
    
    original = text
    all_patterns = []
    
    # Step 1: Check original text for patterns
    has_jailbreak, patterns, _ = detect_jailbreak_patterns(text)
    all_patterns.extend(patterns)
    
    # Step 2: Normalize l33t speak
    normalized = normalize_leet_speak(text)
    
    # Step 3: Collapse spaced text
    collapsed = collapse_spaced_text(normalized)
    
    # Step 4: Check normalized/collapsed text for additional patterns
    has_jailbreak_collapsed, patterns_collapsed, _ = detect_jailbreak_patterns(collapsed)
    all_patterns.extend([p for p in patterns_collapsed if p not in all_patterns])
    
    # Step 5: Check for spaced trigger words specifically
    spaced_words = detect_spaced_trigger_words(text)
    if spaced_words:
        all_patterns.extend([f"[spaced_text] {w}" for w in spaced_words])
    
    # Step 6: Calculate risk score using original and normalized
    risk_score = max(score_input_risk(text), score_input_risk(collapsed))
    
    # Step 7: Strip jailbreak patterns
    cleaned = strip_jailbreak_patterns(collapsed)
    
    # Step 8: If aggressive mode and high risk, strip more aggressively
    if aggressive and risk_score >= RiskLevel.HIGH:
        # Remove any remaining bracketed content that looks like markers
        cleaned = re.sub(r'\[\w+\]', '', cleaned)
        # Remove special token patterns
        cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned)
    
    # Final cleanup
    cleaned = cleaned.strip()
    
    # Log sanitization event if patterns were found
    if all_patterns and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Input sanitized: %d patterns detected, risk_score=%d",
            len(all_patterns), risk_score
        )
    
    return cleaned, risk_score, all_patterns


def sanitize_input_full(text: str, block_threshold: int = RiskLevel.HIGH) -> SanitizationResult:
    """
    Full sanitization with detailed result.
    
    Args:
        text: Input text to sanitize
        block_threshold: Risk score threshold to block input entirely
        
    Returns:
        SanitizationResult with all details
    """
    cleaned, risk_score, patterns = sanitize_input(text)
    
    # Determine risk level
    if risk_score >= RiskLevel.CRITICAL:
        risk_level = "CRITICAL"
    elif risk_score >= RiskLevel.HIGH:
        risk_level = "HIGH"
    elif risk_score >= RiskLevel.MEDIUM:
        risk_level = "MEDIUM"
    elif risk_score >= RiskLevel.LOW:
        risk_level = "LOW"
    else:
        risk_level = "SAFE"
    
    # Determine if input should be blocked
    blocked = risk_score >= block_threshold
    
    return SanitizationResult(
        original_text=text,
        cleaned_text=cleaned,
        risk_score=risk_score,
        detected_patterns=patterns,
        risk_level=risk_level,
        blocked=blocked
    )


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def should_block_input(text: str, threshold: int = RiskLevel.HIGH) -> Tuple[bool, int, List[str]]:
    """
    Quick check if input should be blocked.
    
    Args:
        text: Input text to check
        threshold: Risk score threshold for blocking
        
    Returns:
        Tuple of (should_block, risk_score, detected_patterns)
    """
    risk_score = score_input_risk(text)
    _, patterns, _ = detect_jailbreak_patterns(text)
    should_block = risk_score >= threshold
    
    if should_block:
        logger.warning(
            "Input blocked: jailbreak patterns detected (risk_score=%d, threshold=%d)",
            risk_score, threshold
        )
    
    return should_block, risk_score, patterns


def log_sanitization_event(
    result: SanitizationResult,
    source: str = "unknown",
    session_id: Optional[str] = None
) -> None:
    """
    Log a sanitization event for security auditing.
    
    Args:
        result: The sanitization result
        source: Source of the input (e.g., "cli", "gateway", "api")
        session_id: Optional session identifier
    """
    if result.risk_score < RiskLevel.LOW:
        return  # Don't log safe inputs
    
    log_data = {
        "event": "input_sanitization",
        "source": source,
        "session_id": session_id,
        "risk_level": result.risk_level,
        "risk_score": result.risk_score,
        "blocked": result.blocked,
        "pattern_count": len(result.detected_patterns),
        "patterns": result.detected_patterns[:5],  # Limit logged patterns
        "original_length": len(result.original_text),
        "cleaned_length": len(result.cleaned_text),
    }
    
    if result.blocked:
        logger.warning("SECURITY: Input blocked - %s", log_data)
    elif result.risk_score >= RiskLevel.MEDIUM:
        logger.info("SECURITY: Suspicious input sanitized - %s", log_data)
    else:
        logger.debug("SECURITY: Input sanitized - %s", log_data)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def check_input_safety(text: str) -> Dict[str, Any]:
    """
    Legacy compatibility function for simple safety checks.
    
    Returns dict with 'safe', 'score', and 'patterns' keys.
    """
    score = score_input_risk(text)
    _, patterns, _ = detect_jailbreak_patterns(text)
    
    return {
        "safe": score < RiskLevel.MEDIUM,
        "score": score,
        "patterns": patterns,
        "risk_level": "SAFE" if score < RiskLevel.LOW else 
                      "LOW" if score < RiskLevel.MEDIUM else
                      "MEDIUM" if score < RiskLevel.HIGH else
                      "HIGH" if score < RiskLevel.CRITICAL else "CRITICAL"
    }
