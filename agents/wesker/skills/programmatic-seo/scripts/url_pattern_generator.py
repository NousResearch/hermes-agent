#!/usr/bin/env python3
"""
URL Pattern Generator for Programmatic SEO

Generates URL patterns and page templates from a data source.
Helps plan template-based page generation at scale.

Usage:
  python3 url_pattern_generator.py                    # Demo mode
  python3 url_pattern_generator.py data.json          # From data file
  python3 url_pattern_generator.py data.json --json   # JSON output

Input format (JSON):
{
  "template": "{tool}-vs-{competitor}-comparison",
  "variables": {
    "tool": ["slack", "teams", "discord"],
    "competitor": ["zoom", "webex"]
  },
  "base_url": "https://example.com/compare"
}
"""

import json
import sys
import os
from itertools import product as cartesian_product


def generate_urls(config):
    """Generate all URL combinations from template and variables."""
    template = config["template"]
    variables = config["variables"]
    base_url = config.get("base_url", "https://example.com")

    var_names = list(variables.keys())
    var_values = [variables[name] for name in var_names]

    urls = []
    for combo in cartesian_product(*var_values):
        mapping = dict(zip(var_names, combo))

        # Skip self-comparisons
        values = list(mapping.values())
        if len(values) != len(set(values)):
            continue

        slug = template
        for key, val in mapping.items():
            slug = slug.replace("{" + key + "}", str(val).lower().replace(" ", "-"))

        url = f"{base_url}/{slug}"
        urls.append({
            "url": url,
            "slug": slug,
            "variables": mapping
        })

    return urls


def analyze_patterns(urls, config):
    """Analyze generated URL patterns for SEO concerns."""
    issues = []
    warnings = []

    # Check total page count
    total = len(urls)
    if total > 10000:
        issues.append(f"Generating {total:,} pages — risk of thin content penalty. Consider narrowing variables.")
    elif total > 1000:
        warnings.append(f"Generating {total:,} pages — ensure each has unique, substantial content.")

    # Check URL length
    long_urls = [u for u in urls if len(u["url"]) > 75]
    if long_urls:
        warnings.append(f"{len(long_urls)} URLs exceed 75 chars — may truncate in SERPs.")

    # Check for potential duplicate intent
    template = config["template"]
    var_names = list(config["variables"].keys())
    if len(var_names) >= 2:
        # Check if swapped variables create duplicate intent
        # e.g., "slack-vs-zoom" and "zoom-vs-slack"
        seen_pairs = set()
        dupes = 0
        for u in urls:
            vals = tuple(sorted(u["variables"].values()))
            if vals in seen_pairs:
                dupes += 1
            seen_pairs.add(vals)
        if dupes > 0:
            warnings.append(f"{dupes} URL pairs may have duplicate search intent (e.g., 'A vs B' and 'B vs A'). Consider canonicalizing.")

    # Score
    score = 100
    score -= len(issues) * 20
    score -= len(warnings) * 5
    score = max(0, min(100, score))

    return {
        "total_pages": total,
        "avg_url_length": sum(len(u["url"]) for u in urls) // max(len(urls), 1),
        "long_urls": len(long_urls),
        "issues": issues,
        "warnings": warnings,
        "score": score
    }


def format_report(urls, analysis, config):
    """Format human-readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  PROGRAMMATIC SEO — URL PATTERN REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Template:    {config['template']}")
    lines.append(f"  Base URL:    {config.get('base_url', 'https://example.com')}")
    lines.append(f"  Variables:   {len(config['variables'])} ({', '.join(config['variables'].keys())})")
    lines.append(f"  Total Pages: {analysis['total_pages']:,}")
    lines.append(f"  Avg URL Len: {analysis['avg_url_length']} chars")
    lines.append("")

    # Score
    score = analysis["score"]
    bar_filled = score // 5
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    lines.append(f"  PATTERN SCORE: {score}/100")
    lines.append(f"  [{bar}]")
    lines.append("")

    # Issues
    if analysis["issues"]:
        lines.append("  🔴 ISSUES:")
        for issue in analysis["issues"]:
            lines.append(f"     • {issue}")
        lines.append("")

    if analysis["warnings"]:
        lines.append("  🟡 WARNINGS:")
        for warn in analysis["warnings"]:
            lines.append(f"     • {warn}")
        lines.append("")

    # Sample URLs
    lines.append("  📋 SAMPLE URLS (first 10):")
    for u in urls[:10]:
        lines.append(f"     {u['url']}")
    if len(urls) > 10:
        lines.append(f"     ... and {len(urls) - 10} more")
    lines.append("")

    return "\n".join(lines)


SAMPLE_CONFIG = {
    "template": "{tool}-vs-{competitor}-comparison",
    "variables": {
        "tool": ["slack", "microsoft-teams", "discord", "zoom"],
        "competitor": ["slack", "microsoft-teams", "discord", "zoom", "webex", "google-meet"]
    },
    "base_url": "https://example.com/compare"
}


def main():
    use_json = "--json" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--json"]

    if args and os.path.isfile(args[0]):
        with open(args[0]) as f:
            config = json.load(f)
    else:
        if not args:
            print("[Demo mode — using sample comparison page config]")
        config = SAMPLE_CONFIG

    urls = generate_urls(config)
    analysis = analyze_patterns(urls, config)

    if use_json:
        print(json.dumps({
            "config": config,
            "urls": urls,
            "analysis": analysis
        }, indent=2))
    else:
        print(format_report(urls, analysis, config))


if __name__ == "__main__":
    main()
