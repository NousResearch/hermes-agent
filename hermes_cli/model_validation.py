#!/usr/bin/env python3
"""
Model Validation Module

Validates model names against known patterns and provides fuzzy matching
suggestions for typos or invalid model names.

Used by the /model command to catch invalid model names before saving them
to config.
"""

import re
from difflib import SequenceMatcher, get_close_matches
from typing import Optional, Tuple, List


# =============================================================================
# Known Model Patterns by Provider
# =============================================================================

# Common model name patterns (provider/model-name format for OpenRouter)
MODEL_PATTERNS = {
    # OpenRouter format: provider/model-name or provider/model-name:variant
    "openrouter": re.compile(
        r"^[a-z0-9_-]+/[a-z0-9_.-]+(?::[a-z0-9_-]+)?$", re.IGNORECASE
    ),
    # Direct provider formats
    "openai": re.compile(r"^(gpt-4|gpt-3\.5|o1|o3|chatgpt|text-davinci)", re.IGNORECASE),
    "anthropic": re.compile(r"^claude-", re.IGNORECASE),
    "google": re.compile(r"^(gemini|palm)", re.IGNORECASE),
    "meta": re.compile(r"^(llama|codellama)", re.IGNORECASE),
    "mistral": re.compile(r"^(mistral|mixtral|codestral)", re.IGNORECASE),
    "nous": re.compile(r"^(hermes|nous)", re.IGNORECASE),
    "deepseek": re.compile(r"^deepseek", re.IGNORECASE),
    "qwen": re.compile(r"^qwen", re.IGNORECASE),
    "cohere": re.compile(r"^(command|c4ai)", re.IGNORECASE),
}

# Well-known model names for fuzzy matching suggestions
KNOWN_MODELS = [
    # OpenAI (via OpenRouter or direct)
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/o1",
    "openai/o1-mini",
    "openai/o3-mini",
    # Anthropic
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-sonnet:beta",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    # Google
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-thinking-exp",
    "google/gemini-pro",
    "google/gemini-3-flash-preview",
    # Meta
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.2-90b-vision-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    # Mistral
    "mistralai/mistral-large",
    "mistralai/mistral-medium",
    "mistralai/codestral-latest",
    "mistralai/mixtral-8x22b-instruct",
    # DeepSeek
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-coder",
    # Nous
    "nousresearch/hermes-3-llama-3.1-405b",
    "nousresearch/hermes-3-llama-3.1-70b",
    # Qwen
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwq-32b",
    # Cohere
    "cohere/command-r-plus",
    "cohere/command-r",
    # xAI
    "x-ai/grok-2",
    "x-ai/grok-beta",
]


def validate_model_name(model: str) -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate a model name against known patterns.
    
    Args:
        model: The model name to validate
    
    Returns:
        Tuple of:
        - is_valid: True if model matches any known pattern
        - warning: Warning message if model is unusual (or None)
        - suggestions: List of similar model names if validation fails
    """
    if not model or not model.strip():
        return False, "Model name cannot be empty", []
    
    model = model.strip()
    
    # Check against known patterns
    for provider, pattern in MODEL_PATTERNS.items():
        if pattern.match(model):
            return True, None, []
    
    # Model doesn't match known patterns - generate suggestions
    suggestions = get_close_matches(
        model.lower(),
        [m.lower() for m in KNOWN_MODELS],
        n=3,
        cutoff=0.4
    )
    
    # Map back to original case
    model_map = {m.lower(): m for m in KNOWN_MODELS}
    suggestions = [model_map.get(s, s) for s in suggestions]
    
    # Generate warning message
    warning = (
        f"'{model}' doesn't match any known model pattern. "
        "Model names typically use 'provider/model-name' format (e.g., 'anthropic/claude-3.5-sonnet')."
    )
    
    return False, warning, suggestions


def format_validation_result(model: str, is_valid: bool, warning: Optional[str], 
                             suggestions: List[str]) -> Optional[str]:
    """
    Format validation results for display.
    
    Returns None if model is valid, otherwise returns a warning message.
    """
    if is_valid:
        return None
    
    lines = [f"⚠️  {warning}"]
    
    if suggestions:
        lines.append("")
        lines.append("Did you mean one of these?")
        for s in suggestions:
            lines.append(f"  • {s}")
    
    lines.append("")
    lines.append("The model will be saved, but API calls may fail if the name is invalid.")
    
    return "\n".join(lines)


def interactive_model_validation(model: str) -> Tuple[bool, str]:
    """
    Validate model and return whether to proceed.
    
    Returns:
        Tuple of (should_proceed, display_message)
    """
    is_valid, warning, suggestions = validate_model_name(model)
    
    if is_valid:
        return True, ""
    
    message = format_validation_result(model, is_valid, warning, suggestions)
    return True, message  # Still allow saving, but show warning
