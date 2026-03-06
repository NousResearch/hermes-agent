"""Detect model capabilities — structured tool calling vs text-based injection."""

from dataclasses import dataclass, field
import os


@dataclass
class ModelCapabilities:
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    tool_call_format: str = "native"  # "native" (OpenAI-style) or "xml" (<tool_call> tags)
    max_tools: int = 0  # 0 = unlimited


# Model prefix -> capability overrides
_MODEL_CAPS: dict[str, dict] = {
    "local/": {
        "supports_tools": False,
        "supports_vision": True,
        "tool_call_format": "xml",
    },
    "qwen": {
        "supports_tools": False,
        "tool_call_format": "xml",
    },
    "gpt-4": {
        "supports_tools": True,
        "supports_vision": True,
        "tool_call_format": "native",
    },
    "gpt-3.5": {
        "supports_tools": True,
        "tool_call_format": "native",
    },
    "claude-": {
        "supports_tools": True,
        "supports_vision": True,
        "tool_call_format": "native",
    },
    "gemini": {
        "supports_tools": True,
        "supports_vision": True,
        "tool_call_format": "native",
    },
}

_NATIVE_PROVIDERS = ("openrouter.ai", "anthropic", "openai", "api.groq.com")


def detect_capabilities(model: str, base_url: str = "") -> ModelCapabilities:
    """Detect capabilities based on model name and provider URL."""
    caps = ModelCapabilities()
    model_lower = model.lower()
    base_lower = base_url.lower()

    # Apply registry overrides (longest prefix match first)
    matched_prefix = ""
    for prefix, overrides in _MODEL_CAPS.items():
        if model_lower.startswith(prefix) and len(prefix) > len(matched_prefix):
            matched_prefix = prefix

    if matched_prefix:
        overrides = _MODEL_CAPS[matched_prefix]
        for k, v in overrides.items():
            setattr(caps, k, v)

    # Cloud providers always support native tool calling
    if any(p in base_lower for p in _NATIVE_PROVIDERS):
        caps.supports_tools = True
        caps.tool_call_format = "native"

    # Qwen models on openrouter get native support
    if model_lower.startswith("qwen") and "openrouter" in base_lower:
        caps.supports_tools = True
        caps.tool_call_format = "native"

    # Env var override: force text-based injection
    if os.environ.get("HERMES_FORCE_TOOL_INJECTION", "").lower() in ("1", "true", "yes"):
        caps.supports_tools = False
        caps.tool_call_format = "xml"

    return caps


def needs_tool_adapter(model: str, base_url: str = "") -> bool:
    """Return True if the model needs the text-based tool call adapter."""
    return not detect_capabilities(model, base_url).supports_tools
