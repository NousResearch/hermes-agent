#!/usr/bin/env python3
"""
Capability Map — Auto-Discovery of Available Research Tools

Automatically discovers what tools, skills, and capabilities are available
in the current Hermes environment. This makes deep-research work with
whatever tools are installed, not hardcoded assumptions.

Usage:
  python3 capability_map.py discover              # Discover all capabilities
  python3 capability_map.py check --tool ddgs     # Check if specific tool available
  python3 capability_map.py recommend --type forecast  # Recommend tools for task type
  python3 capability_map.py export                # Export capability map as JSON
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
SKILLS_DIR = os.path.join(HERMES_HOME, "skills")


# Known tool patterns and their research utility
TOOL_SIGNATURES = {
    # Search tools
    "ddgs": {
        "name": "DuckDuckGo Search",
        "type": "web_search",
        "tier": "primary",
        "install": "pip install duckduckgo-search",
        "use_for": ["general_search", "news_search", "image_search"],
    },
    "scrapling": {
        "name": "Scrapling Web Fetcher",
        "type": "web_fetch",
        "tier": "primary",
        "install": "pip install scrapling",
        "use_for": ["article_extraction", "stealth_fetch", "cloudflare_bypass"],
    },
    "curl": {
        "name": "curl HTTP Client",
        "type": "http_client",
        "tier": "fallback",
        "install": "apt-get install curl",
        "use_for": ["http_requests", "api_calls"],
    },
    "wget": {
        "name": "wget Downloader",
        "type": "http_client",
        "tier": "fallback",
        "install": "apt-get install wget",
        "use_for": ["file_download", "http_requests"],
    },
    
    # Social media
    "twitter": {
        "name": "Twitter CLI",
        "type": "social_media",
        "tier": "primary",
        "install": "pip install twitter-api",
        "use_for": ["twitter_search", "twitter_read", "sentiment"],
    },
    
    # Academic
    "arxiv": {
        "name": "arXiv API",
        "type": "academic",
        "tier": "primary",
        "install": "pip install arxiv",
        "use_for": ["paper_search", "academic_research"],
    },
    
    # Prediction markets
    "polymarket": {
        "name": "Polymarket CLI",
        "type": "prediction_market",
        "tier": "primary",
        "install": "pip install polymarket",
        "use_for": ["probability_estimates", "crowd_forecasts", "market_sentiment"],
    },
    
    # Python packages
    "python3": {
        "name": "Python 3",
        "type": "runtime",
        "tier": "essential",
        "install": "apt-get install python3",
        "use_for": ["scripting", "data_processing"],
    },
    
    # Browser
    "chrome": {
        "name": "Chrome Browser",
        "type": "browser",
        "tier": "interactive",
        "install": "system dependent",
        "use_for": ["interactive_browsing", "js_rendering", "paywall_access"],
    },
    
    # Financial data (via Python packages)
    "yfinance": {
        "name": "Yahoo Finance",
        "type": "financial_data",
        "tier": "primary",
        "install": "pip install yfinance",
        "use_for": ["stock_data", "market_data", "historical_prices"],
    },
    
    # News
    "newsapi": {
        "name": "News API",
        "type": "news",
        "tier": "primary",
        "install": "pip install newsapi-python",
        "use_for": ["news_search", "headlines"],
    },
}

# Skill patterns to look for
SKILL_SIGNATURES = {
    "deep-research": {
        "name": "Deep Research Methodology",
        "provides": ["research_methodology", "quality_scoring", "evidence_cards"],
    },
    "duckduckgo-search": {
        "name": "DuckDuckGo Search Skill",
        "provides": ["web_search", "news_search"],
    },
    "arxiv": {
        "name": "arXiv Search Skill",
        "provides": ["academic_search", "paper_fetch"],
    },
    "polymarket": {
        "name": "Polymarket Skill",
        "provides": ["prediction_market", "probability_estimates"],
    },
    "scrapling": {
        "name": "Scrapling Skill",
        "provides": ["web_fetch", "article_extraction"],
    },
    "google-news": {
        "name": "Google News Skill",
        "provides": ["news_search", "current_events"],
    },
    "x-twitter-cli": {
        "name": "X/Twitter CLI Skill",
        "provides": ["twitter_search", "twitter_read", "social_sentiment"],
    },
    "blogwatcher": {
        "name": "Blog Watcher Skill",
        "provides": ["rss_monitoring", "blog_tracking"],
    },
}

# Built-in Hermes tools
BUILTIN_TOOLS = {
    "browser_navigate": {"type": "browser", "use_for": ["interactive_browsing"]},
    "browser_snapshot": {"type": "browser", "use_for": ["page_capture"]},
    "browser_click": {"type": "browser", "use_for": ["interaction"]},
    "browser_vision": {"type": "browser", "use_for": ["visual_analysis"]},
    "web_search": {"type": "web_search", "use_for": ["general_search"]},
    "web_extract": {"type": "web_fetch", "use_for": ["article_extraction"]},
    "terminal": {"type": "runtime", "use_for": ["command_execution"]},
    "execute_code": {"type": "runtime", "use_for": ["scripting"]},
}


def check_command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    try:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def check_python_package(package: str) -> bool:
    """Check if a Python package is installed."""
    try:
        result = subprocess.run(
            ["python3", "-c", f"import {package}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def discover_skills() -> Dict[str, dict]:
    """Discover installed Hermes skills."""
    skills = {}
    
    if not os.path.exists(SKILLS_DIR):
        return skills
    
    # Walk through skills directory
    for root, dirs, files in os.walk(SKILLS_DIR):
        if "SKILL.md" in files:
            skill_path = os.path.join(root, "SKILL.md")
            skill_name = os.path.basename(root)
            
            # Try to read skill metadata
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    content = f.read(2000)  # Read first 2KB
                
                # Extract name and description from YAML frontmatter
                name = skill_name
                description = ""
                provides = []
                
                if "---" in content:
                    parts = content.split("---", 2)
                    if len(parts) >= 2:
                        yaml_part = parts[1]
                        for line in yaml_part.split("\n"):
                            if line.startswith("name:"):
                                name = line.split(":", 1)[1].strip().strip('"')
                            if line.startswith("description:"):
                                description = line.split(":", 1)[1].strip().strip('"')
                            if line.startswith("  - ") and "provides" in yaml_part:
                                provides.append(line.strip("- ").strip())
                
                # Check against known signatures
                for sig_name, sig_data in SKILL_SIGNATURES.items():
                    if sig_name.lower() in skill_name.lower():
                        provides = sig_data.get("provides", provides)
                        break
                
                skills[skill_name] = {
                    "name": name,
                    "path": root,
                    "description": description[:100],
                    "provides": provides,
                }
                
            except Exception as e:
                skills[skill_name] = {
                    "name": skill_name,
                    "path": root,
                    "error": str(e),
                }
    
    return skills


def discover_tools() -> Dict[str, dict]:
    """Discover available CLI tools and Python packages."""
    tools = {}
    
    for cmd, signature in TOOL_SIGNATURES.items():
        if cmd in ["python3", "curl", "wget"]:
            # CLI tool
            available = check_command_exists(cmd)
        else:
            # Python package or CLI
            available = check_python_package(cmd) or check_command_exists(cmd)
        
        if available:
            tools[cmd] = {
                **signature,
                "available": True,
            }
    
    return tools


def discover_builtin_tools() -> Dict[str, dict]:
    """Return built-in Hermes tools (always available)."""
    return {
        name: {**data, "available": True}
        for name, data in BUILTIN_TOOLS.items()
    }


def build_capability_map() -> dict:
    """Build complete capability map."""
    skills = discover_skills()
    tools = discover_tools()
    builtins = discover_builtin_tools()
    
    # Build capability index (what can we do?)
    capabilities = {}
    
    # From skills
    for skill_name, skill_data in skills.items():
        for cap in skill_data.get("provides", []):
            if cap not in capabilities:
                capabilities[cap] = []
            capabilities[cap].append({
                "source": "skill",
                "name": skill_name,
                "priority": 1,
            })
    
    # From tools
    for tool_name, tool_data in tools.items():
        for use in tool_data.get("use_for", []):
            if use not in capabilities:
                capabilities[use] = []
            capabilities[use].append({
                "source": "tool",
                "name": tool_name,
                "priority": 2,
            })
    
    # From builtins
    for tool_name, tool_data in builtins.items():
        for use in tool_data.get("use_for", []):
            if use not in capabilities:
                capabilities[use] = []
            capabilities[use].append({
                "source": "builtin",
                "name": tool_name,
                "priority": 3,
            })
    
    # Sort by priority
    for cap in capabilities:
        capabilities[cap].sort(key=lambda x: x["priority"])
    
    return {
        "skills": skills,
        "tools": tools,
        "builtins": builtins,
        "capabilities": capabilities,
        "summary": {
            "skills_count": len(skills),
            "tools_count": len(tools),
            "builtins_count": len(builtins),
            "capabilities_count": len(capabilities),
        },
    }


def cmd_discover(args):
    """Discover all capabilities."""
    cap_map = build_capability_map()
    
    if args.quiet:
        print(json.dumps(cap_map, indent=2))
    else:
        print("=" * 60)
        print("  CAPABILITY MAP")
        print("=" * 60)
        print()
        
        print(f"  SKILLS: {cap_map['summary']['skills_count']}")
        for name, data in list(cap_map['skills'].items())[:10]:
            provides = ", ".join(data.get("provides", [])[:3])
            print(f"    - {name}: {provides}")
        
        print()
        print(f"  TOOLS: {cap_map['summary']['tools_count']}")
        for name, data in cap_map['tools'].items():
            uses = ", ".join(data.get("use_for", [])[:3])
            print(f"    - {name}: {uses}")
        
        print()
        print(f"  BUILT-INS: {cap_map['summary']['builtins_count']}")
        
        print()
        print(f"  CAPABILITIES: {cap_map['summary']['capabilities_count']}")
        for cap, sources in list(cap_map['capabilities'].items())[:15]:
            primary = sources[0] if sources else None
            if primary:
                print(f"    - {cap}: {primary['name']} ({primary['source']})")
        
        print()
        print("=" * 60)
    
    return 0


def cmd_check(args):
    """Check if a specific tool is available."""
    cap_map = build_capability_map()
    
    tool_name = args.tool.lower()
    
    # Check in tools
    if tool_name in cap_map["tools"]:
        print(json.dumps({
            "available": True,
            "tool": tool_name,
            "details": cap_map["tools"][tool_name],
        }, indent=2))
        return 0
    
    # Check in skills
    for skill_name, skill_data in cap_map["skills"].items():
        if tool_name in skill_name.lower():
            print(json.dumps({
                "available": True,
                "type": "skill",
                "name": skill_name,
                "details": skill_data,
            }, indent=2))
            return 0
    
    # Check in builtins
    if tool_name in cap_map["builtins"]:
        print(json.dumps({
            "available": True,
            "type": "builtin",
            "name": tool_name,
            "details": cap_map["builtins"][tool_name],
        }, indent=2))
        return 0
    
    # Not found
    signature = TOOL_SIGNATURES.get(tool_name, {})
    print(json.dumps({
        "available": False,
        "tool": tool_name,
        "install_hint": signature.get("install", "Unknown tool"),
    }, indent=2))
    return 1


def cmd_recommend(args):
    """Recommend tools for a specific task type."""
    cap_map = build_capability_map()
    
    task_type = args.type.lower()
    
    # Map task types to capabilities
    task_capability_map = {
        "forecast": ["prediction_market", "probability_estimates", "market_sentiment"],
        "search": ["web_search", "general_search", "news_search"],
        "academic": ["academic_search", "paper_search", "paper_fetch"],
        "news": ["news_search", "current_events", "headlines"],
        "social": ["twitter_search", "social_sentiment", "sentiment"],
        "financial": ["stock_data", "market_data", "historical_prices"],
        "extraction": ["article_extraction", "web_fetch", "stealth_fetch"],
        "interactive": ["interactive_browsing", "js_rendering", "paywall_access"],
    }
    
    needed_caps = task_capability_map.get(task_type, [task_type])
    
    recommendations = []
    for cap in needed_caps:
        if cap in cap_map["capabilities"]:
            for source in cap_map["capabilities"][cap]:
                recommendations.append({
                    "capability": cap,
                    "tool": source["name"],
                    "source": source["source"],
                    "priority": source["priority"],
                })
    
    if not recommendations:
        print(f"No tools found for task type: {task_type}")
        print(f"Available capabilities: {list(cap_map['capabilities'].keys())[:20]}")
        return 1
    
    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    
    print(f"RECOMMENDED TOOLS FOR: {task_type}")
    print("=" * 50)
    
    for rec in recommendations[:10]:
        print(f"  [{rec['source']}] {rec['tool']} → {rec['capability']}")
    
    return 0


def cmd_export(args):
    """Export capability map as JSON."""
    cap_map = build_capability_map()
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(cap_map, f, indent=2)
        print(f"Exported to: {args.output}")
    else:
        print(json.dumps(cap_map, indent=2))
    
    return 0


def cmd_missing(args):
    """Show missing critical tools."""
    cap_map = build_capability_map()
    
    critical_tools = ["ddgs", "scrapling", "polymarket"]
    critical_caps = ["web_search", "prediction_market", "article_extraction"]
    
    missing_tools = []
    missing_caps = []
    
    for tool in critical_tools:
        if tool not in cap_map["tools"]:
            missing_tools.append({
                "tool": tool,
                "install": TOOL_SIGNATURES.get(tool, {}).get("install", "unknown"),
            })
    
    for cap in critical_caps:
        if cap not in cap_map["capabilities"] or not cap_map["capabilities"][cap]:
            missing_caps.append(cap)
    
    result = {
        "missing_tools": missing_tools,
        "missing_capabilities": missing_caps,
        "recommendations": [],
    }
    
    if missing_tools:
        result["recommendations"].append("Install missing tools with pip:")
        for t in missing_tools:
            result["recommendations"].append(f"  pip install {t['tool']}")
    
    print(json.dumps(result, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Capability Map — Auto-Discovery of Research Tools")
    sub = parser.add_subparsers(dest="command")
    
    # discover
    p_discover = sub.add_parser("discover", help="Discover all capabilities")
    p_discover.add_argument("--quiet", action="store_true", help="Output JSON only")
    
    # check
    p_check = sub.add_parser("check", help="Check if specific tool available")
    p_check.add_argument("--tool", required=True, help="Tool name to check")
    
    # recommend
    p_recommend = sub.add_parser("recommend", help="Recommend tools for task type")
    p_recommend.add_argument("--type", required=True, 
                            help="Task type (forecast, search, academic, news, social, financial, extraction, interactive)")
    
    # export
    p_export = sub.add_parser("export", help="Export capability map")
    p_export.add_argument("--output", help="Output file path")
    
    # missing
    sub.add_parser("missing", help="Show missing critical tools")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "discover": cmd_discover,
        "check": cmd_check,
        "recommend": cmd_recommend,
        "export": cmd_export,
        "missing": cmd_missing,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())