#!/usr/bin/env python3
"""
Hermes Configuration Migration Tool

Migrate from config.yaml to the new config.json format.

Usage:
    python migrate_config.py              # Migrate with preview
    python migrate_config.py --apply      # Apply migration
    python migrate_config.py --compare    # Compare old and new formats
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from hermes_cli.config import get_config_path, get_env_path, get_hermes_home


def load_yaml_config(path: Path) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_env_file(path: Path) -> dict:
    """Load .env file."""
    env_vars = {}
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    env_vars[key.strip()] = value.strip().strip('"\'')
    return env_vars


def mask_sensitive(value: str, show_chars: int = 4) -> str:
    """Mask sensitive values for display."""
    if not value or len(value) <= show_chars * 2:
        return "***"
    return f"{value[:show_chars]}...{value[-show_chars:]}"


def extract_providers(yaml_config: dict, env_vars: dict) -> dict:
    """Extract provider configurations from YAML config."""
    providers = {}
    
    # Map common API key env vars to providers
    provider_env_map = {
        "bailian": ["BAILIAN_API_KEY", "DASHSCOPE_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "zai": ["ZAI_API_KEY", "GLM_API_KEY"],
        "kimi": ["KIMI_API_KEY"],
        "minimax": ["MINIMAX_API_KEY"],
    }
    
    # Extract from custom_providers
    for provider in yaml_config.get("custom_providers", []):
        name = provider.get("name", "unknown")
        
        # Find matching env var
        api_key_env = None
        if name in provider_env_map:
            for env_key in provider_env_map[name]:
                if env_key in env_vars and env_vars[env_key]:
                    api_key_env = env_key
                    break
        
        providers[name] = {
            "base_url": provider.get("base_url", ""),
            "api_key": f"${{{api_key_env}}}" if api_key_env else "${API_KEY}",
            "models": []
        }
    
    # Add common providers if they have env vars
    common_providers = {
        "bailian": ["BAILIAN_API_KEY", "DASHSCOPE_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }
    
    for provider_name, env_keys in common_providers.items():
        if provider_name not in providers:
            for env_key in env_keys:
                if env_key in env_vars and env_vars[env_key]:
                    providers[provider_name] = {
                        "base_url": get_default_base_url(provider_name),
                        "api_key": f"${{{env_key}}}",
                        "models": []
                    }
                    break
    
    return providers


def get_default_base_url(provider: str) -> str:
    """Get default base URL for common providers."""
    urls = {
        "bailian": "https://coding.dashscope.aliyuncs.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "zai": "https://api.z.ai/v1",
        "kimi": "https://api.moonshot.cn/v1",
    }
    return urls.get(provider, "")


def convert_to_json(yaml_config: dict, env_vars: dict) -> dict:
    """Convert YAML config to JSON format."""
    json_config = {
        "$schema": "https://hermes-agent.dev/schemas/config.v1.json",
        "_version": 1,
        "_comment": f"Migrated from config.yaml on {datetime.now().isoformat()}",
        "_migration_notes": [
            "API keys are referenced via environment variables (${VAR} syntax)",
            "Providers are centralized - one API key serves multiple models",
            "See config.json.example for full documentation"
        ]
    }
    
    # 1. Providers section (centralized)
    providers = extract_providers(yaml_config, env_vars)
    if providers:
        json_config["providers"] = providers
    
    # 2. Defaults section
    model_config = yaml_config.get("model", {})
    agent_config = yaml_config.get("agent", {})
    display_config = yaml_config.get("display", {})
    
    json_config["defaults"] = {
        "primary_model": model_config.get("default", "bailian/qwen3.5-plus"),
        "fallback_model": "",  # Can be configured from fallback_providers
        "max_turns": agent_config.get("max_turns", 90),
        "personality": display_config.get("personality", "kawaii")
    }
    
    # 3. Features section (auxiliary models)
    auxiliary = yaml_config.get("auxiliary", {})
    features = {}
    for feature_name, feature_config in auxiliary.items():
        if feature_config.get("provider") or feature_config.get("model"):
            features[feature_name] = {
                "provider": feature_config.get("provider", "auto"),
                "model": feature_config.get("model", ""),
                "timeout": feature_config.get("timeout")
            }
    if features:
        json_config["features"] = features
    
    # 4. Core settings
    json_config["toolsets"] = yaml_config.get("toolsets", ["hermes-cli"])
    
    # 5. Terminal config
    terminal = yaml_config.get("terminal", {})
    json_config["terminal"] = {
        "backend": terminal.get("backend", "local"),
        "timeout": terminal.get("timeout", 180),
        "docker_image": terminal.get("docker_image", ""),
        "persistent_shell": terminal.get("persistent_shell", True)
    }
    
    # 6. Browser config
    browser = yaml_config.get("browser", {})
    json_config["browser"] = {
        "inactivity_timeout": browser.get("inactivity_timeout", 120),
        "command_timeout": browser.get("command_timeout", 30)
    }
    
    # 7. Display config
    json_config["display"] = {
        "compact": display_config.get("compact", False),
        "streaming": display_config.get("streaming", True),
        "show_cost": display_config.get("show_cost", False),
        "skin": display_config.get("skin", "default"),
        "personality": display_config.get("personality", "kawaii")
    }
    
    # 8. Memory config
    memory = yaml_config.get("memory", {})
    json_config["memory"] = {
        "enabled": memory.get("memory_enabled", True),
        "char_limit": memory.get("memory_char_limit", 2200)
    }
    
    # 9. Security config
    security = yaml_config.get("security", {})
    json_config["security"] = {
        "redact_secrets": security.get("redact_secrets", True),
        "tirith_enabled": security.get("tirith_enabled", True)
    }
    
    # 10. Platform configs
    platforms = {}
    for platform in ["feishu", "telegram", "discord", "slack", "whatsapp"]:
        if platform in yaml_config and yaml_config[platform]:
            platforms[platform] = yaml_config[platform]
    if platforms:
        json_config["platforms"] = platforms
    
    return json_config


def print_comparison(yaml_config: dict, json_config: dict):
    """Print comparison between old and new formats."""
    print("\n" + "="*70)
    print("CONFIGURATION FORMAT COMPARISON")
    print("="*70)
    
    print("\n📄 YAML Format (Legacy):")
    print("-" * 70)
    print("  • Scattered provider configurations")
    print("  • API keys repeated in multiple sections")
    print("  • Deep nesting (4-5 levels)")
    print(f"  • {len(yaml.safe_dump(yaml_config).splitlines())} lines of config")
    
    print("\n📋 JSON Format (New):")
    print("-" * 70)
    print("  ✓ Centralized provider management")
    print("  ✓ API keys referenced via environment variables")
    print("  ✓ Flatter structure (2-3 levels)")
    print(f"  ✓ {len(json.dumps(json_config, indent=2).splitlines())} lines of config")
    
    print("\n📊 Key Improvements:")
    print("-" * 70)
    
    # Count provider references in YAML
    yaml_providers = yaml_config.get("custom_providers", [])
    aux_count = len(yaml_config.get("auxiliary", {}))
    
    print(f"  • Provider configs: {len(yaml_providers)} → centralized in one place")
    print(f"  • Auxiliary model configs: {aux_count} → unified in 'features' section")
    print(f"  • Environment variable references: automatic (${{VAR}} syntax)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Hermes config from YAML to JSON format"
    )
    parser.add_argument(
        "--apply", 
        action="store_true",
        help="Apply the migration (write config.json)"
    )
    parser.add_argument(
        "--compare",
        action="store_true", 
        help="Show comparison between formats"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: ~/.hermes/config.json)"
    )
    
    args = parser.parse_args()
    
    # Paths
    yaml_path = get_config_path()
    env_path = get_env_path()
    json_path = Path(args.output) if args.output else get_hermes_home() / "config.json"
    
    # Check source files
    if not yaml_path.exists():
        print(f"❌ Error: config.yaml not found at {yaml_path}")
        sys.exit(1)
    
    print("🔍 Loading configuration files...")
    yaml_config = load_yaml_config(yaml_path)
    env_vars = load_env_file(env_path)
    
    print(f"  ✓ Loaded config.yaml ({len(yaml_config)} top-level keys)")
    print(f"  ✓ Loaded .env ({len(env_vars)} variables)")
    
    # Convert
    print("\n🔄 Converting to JSON format...")
    json_config = convert_to_json(yaml_config, env_vars)
    
    # Show comparison if requested
    if args.compare:
        print_comparison(yaml_config, json_config)
    
    # Preview or apply
    if args.apply:
        print(f"\n💾 Writing to {json_path}...")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Migration complete!")
        print(f"\n📝 Next steps:")
        print(f"  1. Review {json_path}")
        print(f"  2. Update API key references in .env if needed")
        print(f"  3. Restart Hermes to use new config")
    else:
        print("\n📋 Preview (use --apply to write):")
        print("-" * 70)
        print(json.dumps(json_config, indent=2, ensure_ascii=False)[:2000])
        print("...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
