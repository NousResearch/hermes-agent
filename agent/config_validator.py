"""Configuration validation utilities.

Validates config.yaml and environment setup before running the agent.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration is invalid."""
    pass


def validate_api_keys() -> List[Tuple[str, bool, str]]:
    """Check required and optional API keys.
    
    Returns:
        List of (key_name, is_set, message) tuples
    """
    results = []
    
    # Required keys - need at least one inference provider
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openrouter_key:
        results.append(("OPENROUTER_API_KEY", True, "OpenRouter configured"))
    else:
        results.append(("OPENROUTER_API_KEY", False, "Not set"))
    
    if openai_key:
        results.append(("OPENAI_API_KEY", True, "OpenAI configured"))
    else:
        results.append(("OPENAI_API_KEY", False, "Not set (optional)"))
    
    if anthropic_key:
        results.append(("ANTHROPIC_API_KEY", True, "Anthropic configured"))
    else:
        results.append(("ANTHROPIC_API_KEY", False, "Not set (optional)"))
    
    # Check that at least one provider is available
    has_provider = openrouter_key or openai_key or anthropic_key
    if not has_provider:
        results.append(("INFERENCE_PROVIDER", False, 
            "No inference provider configured. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"))
    
    # Optional feature keys
    optional_keys = [
        ("ELEVENLABS_API_KEY", "TTS with ElevenLabs"),
        ("FIRECRAWL_API_KEY", "Web scraping with Firecrawl"),
        ("VOICE_TOOLS_OPENAI_KEY", "OpenAI TTS"),
    ]
    
    for key, feature in optional_keys:
        if os.environ.get(key):
            results.append((key, True, f"{feature} enabled"))
        else:
            results.append((key, False, f"{feature} disabled (optional)"))
    
    return results


def validate_hermes_home() -> Tuple[bool, str]:
    """Validate HERMES_HOME directory structure.
    
    Returns:
        (is_valid, message) tuple
    """
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    
    if not hermes_home.exists():
        return (False, f"{hermes_home} does not exist")
    
    # Check for config file
    config_file = hermes_home / "config.yaml"
    if not config_file.exists():
        return (False, f"No config.yaml found at {config_file}")
    
    # Check for skills directory (optional but recommended)
    skills_dir = hermes_home / "skills"
    if not skills_dir.exists():
        return (True, f"Valid but no skills directory at {skills_dir}")
    
    skill_count = len(list(skills_dir.rglob("SKILL.md")))
    return (True, f"Valid with {skill_count} skills")


def validate_model_config(model: str) -> Tuple[bool, str]:
    """Validate that a model string is usable.
    
    Returns:
        (is_valid, message) tuple
    """
    if not model:
        return (False, "No model specified")
    
    # Check for common model formats
    if "/" in model:
        # OpenRouter/provider format: "anthropic/claude-sonnet-4"
        provider, model_name = model.split("/", 1)
        return (True, f"Provider: {provider}, Model: {model_name}")
    
    # Plain model name
    return (True, f"Model: {model}")


def run_validation() -> Dict[str, Any]:
    """Run all validation checks.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "api_keys": validate_api_keys(),
        "hermes_home": validate_hermes_home(),
        "errors": [],
        "warnings": [],
    }
    
    # Check for critical errors
    api_results = results["api_keys"]
    has_provider = any(
        name in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"] 
        and is_set 
        for name, is_set, _ in api_results
    )
    
    if not has_provider:
        results["errors"].append("No inference provider API key configured")
    
    home_valid, home_msg = results["hermes_home"]
    if not home_valid:
        results["warnings"].append(f"HERMES_HOME issue: {home_msg}")
    
    results["is_valid"] = len(results["errors"]) == 0
    
    return results


if __name__ == "__main__":
    # Allow running as standalone script
    import json
    results = run_validation()
    print(json.dumps(results, indent=2))
