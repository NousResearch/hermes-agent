#!/usr/bin/env python3
"""
Provider Auto-Detector & Routing Table Synchronizer

Detects new providers in ~/.hermes/config.yaml and auto-syncs them
into the smart model routing table.

Usage:
    python3 detect-providers.py              # List all configured providers
    python3 detect-providers.py --check      # Show providers not in routing table
    python3 detect-providers.py --sync       # Sync routing table
    python3 detect-providers.py --dry-run    # Preview without writing
    python3 detect-providers.py --watch      # Continuously watch config for changes
    python3 detect-providers.py --json       # JSON output
    python3 detect-providers.py --patch      # Output config snippet for new providers
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# Provider Catalog
PROVIDER_CATALOG = {
    "anthropic": {
        "display_name": "Anthropic (Claude)",
        "auth_env_vars": ["ANTHROPIC_API_KEY"],
        "auth_type": "oauth_or_key",
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "claude-3-5-haiku-20241022", "medium": "claude-sonnet-4-20250514", "heavy": "claude-opus-4-20250514"},
        "cost": "subscription", "notes": "Claude Pro $20/mo. Haiku high rate limit, Opus lowest.",
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "auth_env_vars": ["OPENROUTER_API_KEY"],
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "owl-alpha", "medium": "owl-alpha", "heavy": "openrouter/anthropic/claude-sonnet-4"},
        "cost": "freemium", "notes": "Free tier + paid premium",
    },
    "opencode-zen": {
        "display_name": "OpenCode Zen",
        "auth_env_vars": ["OPENCODE_ZEN_API_KEY"],
        "tiers": ["light", "medium"],
        "models": {"light": "glm-4.7-free", "medium": "glm-4.7-free"},
        "cost": "freemium", "notes": "GLM free tier via OpenCode",
    },
    "opencode-go": {
        "display_name": "OpenCode Go",
        "auth_env_vars": ["OPENCODE_GO_API_KEY"],
        "tiers": ["light", "medium"],
        "models": {"light": "default", "medium": "default"},
        "cost": "freemium", "notes": "OpenCode Go tier",
    },
    "lmstudio": {
        "display_name": "LM Studio (local)",
        "auth_env_vars": [], "check_func": "check_lmstudio",
        "tiers": ["light", "medium"],
        "models": {"light": "qwen2.5-3b-instruct", "medium": "qwen2.5-3b-instruct"},
        "cost": "free", "notes": "Local inference, zero cost, no rate limits",
    },
    "groq": {
        "display_name": "Groq",
        "auth_env_vars": ["GROQ_API_KEY"],
        "tiers": ["light", "medium"],
        "models": {"light": "llama-3.1-8b-instant", "medium": "llama-3.1-70b-versatile"},
        "cost": "freemium", "notes": "Fast free tier",
    },
    "nous": {
        "display_name": "Nous Portal",
        "auth_type": "oauth",
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "hermes-3-llama-3.1-8b", "medium": "hermes-3-llama-3.1-70b", "heavy": "hermes-4-70b"},
        "cost": "subscription", "notes": "Nous Research portal",
    },
    "openai-codex": {
        "display_name": "OpenAI (Codex/GPT)",
        "auth_type": "oauth",
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "gpt-4o-mini", "medium": "gpt-4o", "heavy": "gpt-4-turbo"},
        "cost": "subscription", "notes": "OpenAI OAuth",
    },
    "google": {
        "display_name": "Google Gemini",
        "auth_env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "gemini-2.0-flash", "medium": "gemini-2.0-pro", "heavy": "gemini-2.5-pro"},
        "cost": "freemium", "notes": "Google AI / Vertex AI",
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "auth_env_vars": ["DEEPSEEK_API_KEY"],
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "deepseek-v3", "medium": "deepseek-r1", "heavy": "deepseek-r1"},
        "cost": "paid", "notes": "Very cheap, high quality",
    },
    "xai": {
        "display_name": "xAI (Grok)",
        "auth_env_vars": ["XAI_API_KEY"],
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "grok-3-mini", "medium": "grok-3", "heavy": "grok-4"},
        "cost": "paid", "notes": "xAI Grok models",
    },
    "kimi-coding": {
        "display_name": "Kimi / Moonshot",
        "auth_env_vars": ["KIMI_API_KEY"],
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "moonshot-v1-8k", "medium": "moonshot-v1-32k", "heavy": "kimi-k2"},
        "cost": "paid", "notes": "Moonshot AI / Kimi",
    },
    "minimax": {
        "display_name": "MiniMax",
        "auth_env_vars": ["MINIMAX_API_KEY"],
        "tiers": ["light", "medium"],
        "models": {"light": "abab-6.5s", "medium": "abab-7"},
        "cost": "freemium", "notes": "MiniMax models",
    },
    "bedrock": {
        "display_name": "AWS Bedrock",
        "auth_type": "aws",
        "tiers": ["light", "medium", "heavy"],
        "models": {"light": "anthropic.claude-3-haiku", "medium": "anthropic.claude-sonnet-4", "heavy": "anthropic.claude-opus-4"},
        "cost": "paid", "notes": "AWS Bedrock",
    },
    "nvidia-nim": {
        "display_name": "NVIDIA NIM",
        "auth_env_vars": ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"],
        "tiers": ["light", "medium"],
        "models": {"light": "meta/llama-3.1-8b", "medium": "meta/llama-3.1-70b"},
        "cost": "freemium", "notes": "NVIDIA inference microservices",
    },
}

def check_lmstudio():
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:1234/v1/models", timeout=2)
        return True
    except Exception:
        return False

def check_oauth_provider(name):
    """Check if an OAuth provider has tokens configured.

    Reads auth.json to check for provider key names only.
    Never reads or stores actual token values from the file.
    """
    auth_path = Path.home() / ".hermes" / "auth.json"
    if not auth_path.exists():
        return False
    try:
        with open(auth_path) as f:
            return name in json.load(f)
    except Exception:
        return False

def check_custom_provider(entry):
    base_url = entry.get("base_url", "").strip().rstrip("/")
    if not base_url:
        return False, None
    try:
        import urllib.request
        for path in ["/v1/models", "/health", "/"]:
            try:
                urllib.request.urlopen(base_url + path, timeout=3)
                return True, base_url
            except Exception:
                continue
    except Exception:
        pass
    return False, base_url

def load_hermes_config():
    p = Path.home() / ".hermes" / "config.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None

def detect_configured_providers():
    """Detect all providers configured in Hermes config.yaml.

    Detection is config-driven: if a provider appears in model.provider,
    fallback_providers, or custom_providers, it is considered configured.
    Live reachability (LM Studio HTTP check, custom provider /health) is
    used as a secondary signal but does not gate inclusion.

    Does NOT read .env or check API keys — auth configuration is out of scope.
    If credentials are missing, Hermes failover handles 401s automatically.
    """
    config = load_hermes_config()
    if not config:
        return {}
    found = {}

    # Build a set of configured provider names from config.yaml
    configured_names = set()

    # Primary model provider
    primary_provider = config.get("model", {}).get("provider", "")
    if primary_provider:
        configured_names.add(primary_provider)

    # Fallback providers
    for entry in config.get("fallback_providers", []) or []:
        if isinstance(entry, dict):
            prov = entry.get("provider", "")
            if prov:
                configured_names.add(prov)

    # Custom providers
    for i, entry in enumerate(config.get("custom_providers", []) or []):
        if isinstance(entry, dict):
            cname = entry.get("name", f"custom-{i}")
            configured_names.add(f"custom:{cname}")

    # Credential pool providers (if configured)
    for key in (config.get("credential_pool_strategies") or {}):
        configured_names.add(key)

    # Catalog-based detection for known providers
    for name, meta in PROVIDER_CATALOG.items():
        detected = False
        # Check if provider name appears in config
        if name in configured_names:
            detected = True
        # Check for local services by reachability
        elif meta.get("check_func") == "check_lmstudio":
            detected = check_lmstudio()
        # Check for OAuth providers via auth.json key presence
        elif meta.get("auth_type") == "oauth":
            detected = check_oauth_provider(name)

        if detected:
            found[name] = {**meta, "provider_key": name,
                          "source": "config" if name in configured_names else "auto"}

    # Always add custom providers (from config)
    for i, entry in enumerate(config.get("custom_providers", []) or []):
        if not isinstance(entry, dict):
            continue
        cname = entry.get("name", f"custom-{i}")
        is_running, base_url = check_custom_provider(entry)
        key = f"custom:{cname}"
        found[key] = {
            "display_name": f"Custom: {cname}", "provider_key": key,
            "source": "custom_providers", "base_url": base_url,
            "is_running": is_running,
            "tiers": ["light", "medium"],
            "models": {"light": f"{cname}/auto", "medium": f"{cname}/auto"},
            "cost": "unknown", "notes": f"Custom at {base_url}",
            "auto_detected": True,
        }

    return found


def get_routing_table_path():
    # Works for both in-repo and user-local installs
    candidates = [
        Path.home() / ".hermes/skills/devops/smart-model-router/references/routing-table.yaml",
        Path("skills/devops/smart-model-router/references/routing-table.yaml").resolve(),
    ]
    for p in candidates:
        if p.parent.exists():
            return p
    return candidates[0]


def load_routing_table():
    p = get_routing_table_path()
    if p.exists():
        t = yaml.safe_load(p.read_text())
        if t and "providers" in t:
            return t
    return {"providers": {}, "chains": {"light": [], "medium": [], "heavy": []}}

def save_routing_table(table):
    p = get_routing_table_path()
    p.write_text(yaml.dump(table, default_flow_style=False, sort_keys=False, allow_unicode=True))
    return p

def update_routing_table(dry_run=False):
    configured = detect_configured_providers()
    current = load_routing_table()
    existing = set(current.get("providers", {}).keys())
    detected = set(configured.keys())
    added, removed = detected - existing, existing - detected
    updated = copy.deepcopy(current)
    updated.setdefault("providers", {})
    updated.setdefault("chains", {"light": [], "medium": [], "heavy": []})
    for pk in added:
        info = configured[pk]
        updated["providers"][pk] = {
            "display_name": info["display_name"], "tiers": info.get("tiers", ["light", "medium"]),
            "models": info.get("models", {}), "cost": info.get("cost", "unknown"),
            "notes": info.get("notes", ""), "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "auto_detected": True,
        }
    for pk in removed:
        prov = updated["providers"].get(pk, {})
        if prov.get("auto_detected") and not prov.get("pinned"):
            del updated["providers"][pk]
    free, paid_l, paid_m, paid_h = [], [], [], []
    for pk, info in updated["providers"].items():
        cost = info.get("cost", "unknown")
        is_free = cost in ("free", "freemium")
        for tier in ["light", "medium", "heavy"]:
            if tier in info.get("tiers", []) and info.get("models", {}).get(tier):
                e = {"provider": pk, "model": info["models"][tier], "cost": cost}
                if is_free:
                    free.append(e)
                elif tier == "light":
                    paid_l.append(e)
                elif tier == "medium":
                    paid_m.append(e)
                else:
                    paid_h.append(e)
    sk = lambda x: (0 if x["cost"] in ("free", "freemium") else 1, x["provider"])
    updated["chains"]["light"] = sorted(dict((f"{e['provider']}/{e['model']}", e) for e in (free + paid_l[:2])).values(), key=sk)
    updated["chains"]["medium"] = sorted(dict((f"{e['provider']}/{e['model']}", e) for e in (paid_m[:3] + free)).values(), key=sk)
    updated["chains"]["heavy"] = sorted(dict((f"{e['provider']}/{e['model']}", e) for e in (paid_h[:2] + paid_m[:2])).values(), key=sk)
    if not dry_run:
        save_routing_table(updated)
    return sorted(added), sorted(removed), updated

def get_config_patch(added):
    conf = detect_configured_providers()
    entries = []
    for pk in added:
        info = conf.get(pk, {})
        models = info.get("models", {})
        med = models.get("medium", models.get("light", "default"))
        prov = pk.split(":")[-1] if pk.startswith("custom:") else pk
        entries.append({"model": med, "provider": prov})
    return entries

def print_results(configured, added=None, removed=None, chains=None):
    if not configured:
        print("No providers detected.")
        return
    free = [(k, v) for k, v in sorted(configured.items()) if v.get("cost") in ("free", "freemium")]
    paid = [(k, v) for k, v in sorted(configured.items()) if v.get("cost") in ("paid", "subscription")]
    unk = [(k, v) for k, v in sorted(configured.items()) if v.get("cost") not in ("free", "freemium", "paid", "subscription")]
    print(f"\n{'='*60}\n  DETECTED PROVIDERS: {len(configured)} total\n{'='*60}")
    if free:
        print("\n  Free / Freemium:")
        for pk, info in free:
            run = " YES" if info.get("is_running", True) else " NO (unreachable)"
            ms = " ".join(f"{t}={info['models'].get(t,'?')}" for t in info.get("tiers",[]) if t in info.get("models",{}))
            print(f"    * {info['display_name']:<30} ({pk})  running:{run}")
            print(f"      {ms}")
    if paid:
        print("\n  Paid / Subscription:")
        for pk, info in paid:
            ms = " ".join(f"{t}={info['models'].get(t,'?')}" for t in info.get("tiers",[]) if t in info.get("models",{}))
            print(f"    * {info['display_name']:<30} ({pk})")
            print(f"      {ms}")
    if unk:
        print("\n  Unknown cost:")
        for pk, info in unk:
            print(f"    * {info['display_name']:<30} ({pk})")
    if added is not None:
        if added:
            print(f"\n  [NEW] Adding to routing: {', '.join(added)}")
        if removed:
            print(f"\n  [GONE] Removing from routing: {', '.join(removed)}")
    if chains:
        for tier_label in ["light", "medium", "heavy"]:
            chain = chains.get(tier_label, []) 
            print(f"\n  {tier_label.upper()} chain:")
            if not chain:
                print("    (empty)")
            for i, e in enumerate(chain):
                icon = ">>" if i == 0 else "  "
                cost = "free" if e.get("cost") in ("free","freemium") else "paid"
                print(f"    {icon} [{cost}] {e['provider']}/{e['model']}")

def main():
    parser = argparse.ArgumentParser(description="Provider Auto-Detector & Routing Syncer")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()

    if args.watch:
        cp = Path.home() / ".hermes" / "config.yaml"
        last = 0
        print("Watching ~/.hermes/config.yaml ... (Ctrl+C to stop)")
        try:
            while True:
                try:
                    mtime = cp.stat().st_mtime
                except FileNotFoundError:
                    time.sleep(2); continue
                if mtime != last:
                    last = mtime
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{ts}] Config changed!")
                    added, removed, table = update_routing_table(dry_run=False)
                    if added:
                        print(f"  [NEW] {', '.join(added)}")
                        for e in get_config_patch(added):
                            print(f"    fallback: - model: {e['model']}")
                            print(f"               provider: {e['provider']}")
                    if removed:
                        print(f"  [GONE] {', '.join(removed)}")
                    if not added and not removed:
                        print("  No provider changes")
                    for tl in ["light","medium","heavy"]:
                        ch = table.get("chains",{}).get(tl,[])
                        chain_str = " -> ".join(e['provider'] + "/" + e['model'] for e in ch) or "(empty)"
                        print(f"  {tl}: {chain_str}")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    configured = detect_configured_providers()

    if args.json:
        out = {"detected": {k: {kk: vv for kk, vv in v.items() if kk != "config_entry"} for k, v in configured.items()}}
        if args.sync or args.check or args.dry_run:
            a, r, t = update_routing_table(dry_run=True)
            out["added"] = a
            out["removed"] = r
            out["chains"] = t.get("chains", {})
        print(json.dumps(out, indent=2))
        return

    if args.sync:
        added, removed, table = update_routing_table(dry_run=args.dry_run)
        print_results(configured, added, removed, table.get("chains"))
        if args.dry_run:
            print("\n  (dry run — no files written)")
        return

    if args.check:
        current = load_routing_table()
        existing = set(current.get("providers", {}).keys())
        new_p = set(configured.keys()) - existing
        gone = existing - set(configured.keys())
        if new_p:
            print(f"\n  [NEW] Providers to add: {', '.join(sorted(new_p))}")
        if gone:
            print(f"\n  [GONE] Providers no longer configured: {', '.join(sorted(gone))}")
        if not new_p and not gone:
            print("\n  Routing table is up to date. All configured providers are tracked.")
        print_results(configured)
        if args.patch and new_p:
            print("\n  Config fallback_providers additions:")
            for e in get_config_patch(sorted(new_p)):
                print(f"    - model: {e['model']}")
                print(f"      provider: {e['provider']}")
        return

    if args.patch:
        current = load_routing_table()
        existing = set(current.get("providers", {}).keys())
        new_p = sorted(set(configured.keys()) - existing)
        if new_p:
            print("\n  New providers and suggested fallback_providers config:")
            for e in get_config_patch(new_p):
                print(f"\n    - model: {e['model']}")
                print(f"      provider: {e['provider']}")
        else:
            print("  No new providers to patch.")
        return

    # Default: just list providers
    print_results(configured)

if __name__ == "__main__":
    main()
