#!/usr/bin/env python3
"""
Daily script to evaluate free LLM providers for MOA aggregator selection.
Evaluates freebuff (deepseek/deepseek-v4-flash), opencode (opencode-zen), 
nvidia (nvidia/auto), nous (nous/auto-free) based on web research about
tool calling ability, reasoning, and coding capability.
Updates the active MOA preset in ~/.hermes/config.yaml to use the best provider.
If evaluation fails, falls back to a preset centered on gpt-5.5 and grok-4.5.
"""

import json
import subprocess
import sys
import yaml
from pathlib import Path

# Path to Hermes config
CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"

# Providers to evaluate: (provider_name, model_alias, description)
PROVIDERS = [
    ("freebuff", "deepseek/deepseek-v4-flash", "FreeBuff DeepSeek V4 Flash"),
    ("opencode", "auto-free", "OpenCode Zen free tier"),
    ("nvidia", "auto", "NVIDIA API free tier"),
    ("nous", "auto-free", "Nous Research free tier"),
]

# Fallback preset name
FALLBACK_PRESET = "gpt55-grok45-fallback"

def hermes_search(query):
    """Use hermes -z to perform a web search and return parsed results."""
    # Ask the agent to return JSON
    prompt = f"Search the web for: {query}. Return the results as a JSON list of objects with keys 'title' and 'description'. If you cannot return JSON, return a text summary."
    cmd = ["hermes", "-z", prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"hermes -z failed: {result.stderr}")
            return None
        output = result.stdout.strip()
        # Try to parse as JSON
        try:
            data = json.loads(output)
            # Ensure it's a list
            if isinstance(data, list):
                return data
            else:
                # If it's a dict, wrap in a list
                return [data]
        except json.JSONDecodeError:
            # If not JSON, return the text as a single item for simplicity
            return [{"title": "Search result", "description": output}]
    except subprocess.TimeoutExpired:
        print("hermes -z timed out")
        return None
    except Exception as e:
        print(f"Error running hermes -z: {e}")
        return None

def evaluate_provider(provider, model, desc):
    """Use web research to score a provider on tool calling, reasoning, coding."""
    queries = [
        f"{provider} tool calling ability",
        f"{provider} reasoning benchmark",
        f"{provider} coding performance",
        f"{provider} vs other LLMs agent use",
    ]
    score = 0
    details = []
    for q in queries:
        print(f"Searching: {q}")
        res = hermes_search(q)
        if not res:
            continue
        # Extract text from results
        text = ""
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    text += " " + item.get("title", "") + " " + item.get("description", "")
                else:
                    text += " " + str(item)
        # Simple keyword scoring
        keywords = ["tool use", "agent", "function call", "reasoning", "math", "code", "programming", "benchmark", "score"]
        for kw in keywords:
            if kw in text.lower():
                score += 1
        details.append(f"Q: {q} -> score inc (found {sum(1 for kw in keywords if kw in text.lower())} matches)")
    return score, details

def update_config(provider, model):
    """Update the active MOA preset's aggregator to use the given provider/model."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to read config: {e}")
        return False

    # Ensure moa section exists
    if "moa" not in config:
        config["moa"] = {}
    if "presets" not in config["moa"]:
        config["moa"]["presets"] = {}

    # Determine which preset is active
    active_preset = config["moa"].get("active_preset") or config["moa"].get("default_preset") or "free-orchestrator-gpt55-last"
    # Ensure the preset exists
    if active_preset not in config["moa"]["presets"]:
        # Create a default preset if missing
        config["moa"]["presets"][active_preset] = {
            "reference_models": [],
            "aggregator": {"provider": provider, "model": model},
            "reference_temperature": 0.25,
            "aggregator_temperature": 0.2,
            "max_tokens": 8192,
            "enabled": True,
        }
    else:
        # Update aggregator
        config["moa"]["presets"][active_preset]["aggregator"] = {"provider": provider, "model": model}

    # Write back
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"Updated config: set {active_preset}.aggregator to {provider}:{model}")
        return True
    except Exception as e:
        print(f"Failed to write config: {e}")
        return False

def ensure_fallback_preset():
    """Ensure a fallback preset exists that uses gpt-5.5 and grok-4.5 as reference/aggregator."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to read config for fallback: {e}")
        return False

    if "moa" not in config:
        config["moa"] = {}
    if "presets" not in config["moa"]:
        config["moa"]["presets"] = {}

    if FALLBACK_PRESET not in config["moa"]["presets"]:
        config["moa"]["presets"][FALLBACK_PRESET] = {
            "reference_models": [
                {"provider": "openai-codex", "model": "gpt-5.5"},
                {"provider": "xai", "model": "grok-4.5"},  # assuming xai provider for grok
            ],
            "aggregator": {"provider": "openai-codex", "model": "gpt-5.5"},
            "reference_temperature": 0.35,
            "aggregator_temperature": 0.25,
            "max_tokens": 8192,
            "enabled": True,
            "description": "Fallback preset using GPT-5.5 and Grok-4.5",
        }
        # Write back
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"Ensured fallback preset {FALLBACK_PRESET} exists.")
        except Exception as e:
            print(f"Failed to write fallback preset: {e}")
            return False
    else:
        print(f"Fallback preset {FALLBACK_PRESET} already exists.")
    return True

def main():
    print("Starting daily MOA provider evaluation...")
    # Ensure fallback preset exists
    ensure_fallback_preset()

    # Build provider -> model mapping
    provider_to_model = {p: m for p, m, _ in PROVIDERS}

    scores = {}
    details = {}
    for provider, model, desc in PROVIDERS:
        print(f"\nEvaluating {provider} ({model}) - {desc}")
        score, detail = evaluate_provider(provider, model, desc)
        scores[provider] = score
        details[provider] = detail
        print(f"  Score: {score}")

    if not scores:
        print("No scores obtained; using fallback.")
        selected_provider, selected_model = "openai-codex", "gpt-5.5"  # fallback aggregator
    else:
        # Select provider with highest score
        selected_provider = max(scores, key=scores.get)
        selected_model = provider_to_model[selected_provider]
        print(f"\nSelected provider: {selected_provider} ({selected_model}) with score {scores[selected_provider]}")

    # Update config
    success = update_config(selected_provider, selected_model)
    if not success:
        print("Failed to update config; leaving as is.")
    else:
        print("Configuration updated successfully.")

if __name__ == "__main__":
    main()