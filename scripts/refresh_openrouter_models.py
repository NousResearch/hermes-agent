#!/usr/bin/env python3
"""
refresh_openrouter_models.py

Weekly script for keeping the subagent-model-routing skill current.

Hits the OpenRouter /api/v1/models endpoint, compares the live catalog against
your configured whitelists, and produces a read-only advisory digest for review.

USAGE — run as a Hermes cron job script:

    cronjob(
        action="create",
        name="OpenRouter Model Refresh",
        script="refresh_openrouter_models.py",       # path relative to ~/.hermes/scripts/
        schedule="0 11 * * 0",                        # every Sunday at 11 AM
        model={"model": "openrouter/auto", "provider": "openrouter"},
        deliver="origin",
        prompt=CRON_PROMPT_TEMPLATE,                  # see bottom of this file
    )

The script's stdout is injected into the cron prompt as context. The cron agent
reads the output and produces a digest for the operator. It does NOT write any
files — all changes require explicit operator approval.

SETUP:
    1. Copy this file to ~/.hermes/scripts/refresh_openrouter_models.py
    2. Customize the WHITELISTS dict below for your use case
    3. Create the cron job (see usage above)
    4. Set OPENROUTER_API_KEY in your environment

REQUIREMENTS:
    - Python 3.10+ (stdlib only, no dependencies)
    - OPENROUTER_API_KEY environment variable

MAINTENANCE — HOW TO KEEP EVERYTHING IN SYNC:

    There are THREE locations that must stay consistent:

    1. ~/.hermes/scripts/refresh_openrouter_models.py         ← THIS FILE (cron runs this)
    2. ~/.hermes/hermes-agent-feat/scripts/refresh_openrouter_models.py  ← feat branch copy
    3. ~/.hermes/skills/autonomous-ai-agents/subagent-model-routing/SKILL.md  ← human-readable mirror

    CANONICAL SOURCE OF TRUTH: This file's WHITELISTS dict.

    UPDATE FLOW (approved changes only — never edit whitelists speculatively):
    1. Cron delivers a read-only digest to Jordan on Sundays
    2. Jordan approves specific changes in a follow-up session
    3. Agent patches WHITELISTS in THIS file first
    4. Agent patches the skill's tier tables and WHITELISTS section to match (same session)
    5. Agent copies this file AND the skill to the feat branch:
           cp ~/.hermes/scripts/refresh_openrouter_models.py \
              ~/.hermes/hermes-agent-feat/scripts/refresh_openrouter_models.py
           cp ~/.hermes/skills/autonomous-ai-agents/subagent-model-routing/SKILL.md \
              ~/.hermes/hermes-agent-feat/skills/autonomous-ai-agents/subagent-model-routing/SKILL.md
    6. Steps 3–5 must happen atomically in one session. A partial update
       (script only, or skill only, or feat branch not synced) leaves the
       system inconsistent.

    WHY TWO SCRIPT LOCATIONS:
    - The Hermes cron sandbox only executes scripts inside ~/.hermes/scripts/
      (symlinks that resolve outside are blocked by path traversal guard)
    - The feat branch (hermes-agent-feat) is a separate git repo tracking
      the PR. It must stay in sync so the PR reflects current whitelist state.
    - These two files are NOT linked — copies must be explicit after every edit.

    POST-MERGE (when PR #12794 lands in upstream main):
    - Update the copy step above to use ~/.hermes/hermes-agent/scripts/ instead
    - The feat worktree can then be deleted
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PRICE_CACHE_PATH = Path.home() / ".hermes/caches/openrouter_prices_last.json"
PRICE_CHANGE_THRESHOLD = 0.20  # 20% delta triggers a flag

# ---------------------------------------------------------------------------
# WHITELISTS — CUSTOMIZE THESE FOR YOUR USE CASE
#
# Define which models are allowed in each routing tier. The tier names and
# model selections here are examples — rename, add, or remove tiers to match
# your operational needs.
#
# Keep these in sync with:
#   - Your subagent-model-routing skill's whitelist/tier tables
#   - Any account-level OpenRouter whitelist settings you've configured
#
# When the cron reports changes and you approve them, update BOTH this dict
# AND the skill's tier tables atomically in the same session.
# ---------------------------------------------------------------------------

WHITELISTS: dict[str, list[str]] = {
    # PREMIUM: Orchestrator-level, synthesis, judgment, reviews
    "premium": [
        "x-ai/grok-4.20",
        "x-ai/grok-4.20-multi-agent",
        "x-ai/grok-4.3",
        "anthropic/claude-opus-4.7",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-2.5-pro",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
    ],
    # STANDARD: Regular delegation — business ops, analysis, general tasks
    "standard": [
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.4-mini",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "minimax/minimax-m2.7",
        "z-ai/glm-5.1",
    ],
    # CODING: Code writing, review, and modification tasks only (may overlap other tiers)
    "coding": [
        "x-ai/grok-code-fast-1",
        "qwen/qwen3-coder-flash",
        "openai/gpt-5.1-codex-mini",
        "openai/gpt-5.1-codex",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "minimax/minimax-m2.7",
        "z-ai/glm-5.1",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.7",
    ],
    # BUDGET: Cron jobs, automated extraction, simple parsing
    "budget": [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "x-ai/grok-4.1-fast",
        "openai/gpt-5-nano",
        "deepseek/deepseek-v4-flash",
    ],
}

# ---------------------------------------------------------------------------
# TRACKED PROVIDERS — new model discovery scope
# Only models from these providers will appear in the "new models" section.
# Add or remove providers to match what you care about.
# ---------------------------------------------------------------------------

TRACKED_PROVIDERS: set[str] = {
    "anthropic",
    "google",
    "openai",
    "x-ai",
    "mistralai",
    "deepseek",
    "qwen",
    "meta-llama",
}

# PRICE_CHANGE_THRESHOLD: reserved for a future pricing-delta feature.
# The script currently reports a live pricing snapshot but does not compare
# against a baseline (previous run or skill file). Implementing delta tracking
# would require persisting previous prices to disk between runs. Not yet
# implemented — when added, use this threshold (20%) to filter noise.

# ---------------------------------------------------------------------------
# Implementation — no customization needed below this line
# ---------------------------------------------------------------------------

# Validate whitelist exclusivity:
# budget / standard / premium must be mutually exclusive.
# coding is allowed to overlap with any tier.
_EXCLUSIVE_TIERS = {"budget", "standard", "premium"}
_exclusive_seen: dict[str, str] = {}
_overlap_errors: list[str] = []
for _tier in _EXCLUSIVE_TIERS:
    for _mid in WHITELISTS.get(_tier, []):
        if _mid in _exclusive_seen:
            _overlap_errors.append(
                f"  '{_mid}' appears in both '{_exclusive_seen[_mid]}' and '{_tier}'"
            )
        else:
            _exclusive_seen[_mid] = _tier
if _overlap_errors:
    raise ValueError(
        "WHITELIST OVERLAP DETECTED — budget/standard/premium must be mutually exclusive:\n"
        + "\n".join(_overlap_errors)
        + "\n(coding is exempt — it may overlap any tier)"
    )

# Flatten all unique models across all whitelists
ALL_WHITELISTED: set[str] = set()
for _models in WHITELISTS.values():
    ALL_WHITELISTED.update(_models)


def fetch_models(api_key: str, order: str | None = None) -> list[dict]:
    """Fetch model list from OpenRouter /api/v1/models."""
    url = "https://openrouter.ai/api/v1/models"
    if order:
        url += "?" + urllib.parse.urlencode({"order": order})
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://hermes-agent.local",
            "X-Title": "Hermes Model Refresh",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter API returned {e.code}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to fetch models: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"OpenRouter returned malformed JSON (offset {e.pos}): {e.msg}\n"
            f"Response snippet: {raw[:200]!r}"
        ) from e
    return data.get("data", [])


def parse_price(pricing: dict, key: str) -> float | None:
    """Parse a pricing field to float (price per token)."""
    val = pricing.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def price_per_million(price_per_token: float | None) -> str:
    if price_per_token is None:
        return "unknown"
    return f"${price_per_token * 1_000_000:.2f}"


def load_price_cache() -> dict:
    """Load previous run's price snapshot. Returns empty dict if none exists."""
    try:
        return json.loads(PRICE_CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_price_cache(prices: dict) -> None:
    """Persist current run's prices for next week's delta comparison."""
    PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRICE_CACHE_PATH.write_text(json.dumps(prices, indent=2))


def fmt_ctx(ctx) -> str:
    c = int(ctx) if ctx else 0
    if not c:
        return "unknown"
    return f"{c // 1000}K" if c >= 1000 else "< 1K"


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"OpenRouter Model Refresh — {now}")
    print("=" * 60)

    # Fetch with top-weekly ordering: returns full catalog ranked by weekly usage.
    try:
        models = fetch_models(api_key, order="top-weekly")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Sanity checks — fewer than 50 almost certainly means truncation.
    if len(models) <= 50:
        print(
            f"WARNING: Only {len(models)} models returned — expected >50; "
            "aborting to avoid false MISSING positives.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(models) < 300:
        print(
            f"WARNING: Only {len(models)} models returned — OpenRouter usually has 300+; "
            "possible truncation.",
            file=sys.stderr,
        )

    live_ids = {m["id"] for m in models if "id" in m}
    live_by_id = {m["id"]: m for m in models if "id" in m}

    # Load previous price snapshot for delta comparison
    prev_prices = load_price_cache()
    if prev_prices:
        print(f"📅 Price baseline: {prev_prices.get('_run_date', 'unknown')}")
    else:
        print("📅 Price baseline: none (first run — delta tracking starts next week)")
    print()

    # Precompute per-model whitelist membership for O(1) lookup.
    whitelist_by_model: dict[str, list[str]] = {}
    for wl_name, wl_models in WHITELISTS.items():
        for mid in wl_models:
            whitelist_by_model.setdefault(mid, []).append(wl_name)

    print(f"Total models available on OpenRouter: {len(models)}")
    print()

    # ------------------------------------------------------------------
    # 1. Missing — whitelisted models no longer in live catalog
    # ------------------------------------------------------------------
    missing = [m for m in ALL_WHITELISTED if m not in live_ids]
    if missing:
        print("⚠️  MISSING — Whitelisted models NOT found in live catalog:")
        for m in sorted(missing):
            in_lists = [k for k, v in WHITELISTS.items() if m in v]
            print(f"  - {m}  [in: {', '.join(in_lists)}]")
    else:
        print("✅ All whitelisted models found in live catalog.")
    print()

    # ------------------------------------------------------------------
    # 2. Pricing snapshot for all whitelisted models
    # ------------------------------------------------------------------
    print("💰 PRICING SNAPSHOT — Current prices for whitelisted models:")
    print(f"  {'Model':<40} {'In ($/Mi)':>12} {'Out ($/Mi)':>12}  {'Context':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*12}  {'-'*10}")
    for model_id in sorted(ALL_WHITELISTED):
        if model_id not in live_by_id:
            print(f"  {model_id:<40} {'MISSING':>12} {'MISSING':>12}  {'':>10}")
            continue
        m = live_by_id[model_id]
        pricing = m.get("pricing", {})
        p_in = parse_price(pricing, "prompt")
        p_out = parse_price(pricing, "completion")
        ctx = m.get("context_length", 0)
        print(
            f"  {model_id:<40} {price_per_million(p_in):>12} "
            f"{price_per_million(p_out):>12}  {fmt_ctx(ctx):>10}"
        )
    print()

    # ------------------------------------------------------------------
    # 2b. Pricing deltas vs previous run
    # ------------------------------------------------------------------
    current_prices: dict = {"_run_date": now}
    price_changes = []
    for model_id in sorted(ALL_WHITELISTED):
        m = live_by_id.get(model_id)
        if not m:
            continue
        pricing = m.get("pricing", {})
        p_in = parse_price(pricing, "prompt")
        p_out = parse_price(pricing, "completion")
        current_prices[model_id] = {"in": p_in, "out": p_out}

        if prev_prices and model_id in prev_prices:
            old = prev_prices[model_id]
            for label, old_val, new_val in [
                ("in", old.get("in"), p_in),
                ("out", old.get("out"), p_out),
            ]:
                if old_val and new_val and abs(new_val - old_val) / old_val >= PRICE_CHANGE_THRESHOLD:
                    direction = "▲" if new_val > old_val else "▼"
                    pct = abs(new_val - old_val) / old_val * 100
                    price_changes.append(
                        f"  {direction} {model_id} [{label}]: "
                        f"{price_per_million(old_val)} → {price_per_million(new_val)} "
                        f"({pct:.0f}%)"
                    )

    if prev_prices:
        if price_changes:
            print(f"💸 PRICING CHANGES vs {prev_prices.get('_run_date', 'last run')} (>{PRICE_CHANGE_THRESHOLD*100:.0f}% delta):")
            for line in price_changes:
                print(line)
        else:
            print("✅ No significant pricing changes vs last run.")
        print()

    # ------------------------------------------------------------------
    # 3. New models from tracked providers not yet in any whitelist
    # ------------------------------------------------------------------
    new_models = []
    for m in models:
        mid = m["id"]
        if mid in ALL_WHITELISTED:
            continue
        provider = mid.split("/")[0] if "/" in mid else ""
        if provider not in TRACKED_PROVIDERS:
            continue
        pricing = m.get("pricing", {})
        p_in = parse_price(pricing, "prompt")
        p_out = parse_price(pricing, "completion")
        ctx = m.get("context_length", 0)
        new_models.append((mid, p_in, p_out, ctx))

    if new_models:
        print(f"🆕 NEW MODELS from tracked providers not in any whitelist ({len(new_models)}):")
        print(f"  {'Model':<45} {'In ($/Mi)':>12} {'Out ($/Mi)':>12}  {'Context':>10}")
        print(f"  {'-'*45} {'-'*12} {'-'*12}  {'-'*10}")
        for mid, p_in, p_out, ctx in sorted(new_models, key=lambda x: x[0]):
            print(
                f"  {mid:<45} {price_per_million(p_in):>12} "
                f"{price_per_million(p_out):>12}  {fmt_ctx(ctx):>10}"
            )
    else:
        print("✅ No new models from tracked providers outside whitelists.")
    print()

    # ------------------------------------------------------------------
    # 4. Whitelist membership summary
    # ------------------------------------------------------------------
    print("📋 WHITELIST SUMMARY:")
    for name, models_list in WHITELISTS.items():
        live_count = sum(1 for m in models_list if m in live_ids)
        print(f"  {name}: {live_count}/{len(models_list)} models live")
    print()

    # ------------------------------------------------------------------
    # 5. Top 15 trending models this week
    # ------------------------------------------------------------------
    print("🔥 TOP 15 TRENDING MODELS — Most used on OpenRouter this week:")
    print(
        f"  {'Rank':<5} {'Model':<45} {'In ($/Mi)':>10} {'Out ($/Mi)':>11}"
        f"  {'Context':>8}  In Whitelist?"
    )
    print(f"  {'-'*5} {'-'*45} {'-'*10} {'-'*11}  {'-'*8}  {'-'*13}")
    for rank, m in enumerate(models[:15], 1):
        mid = m["id"]
        pricing = m.get("pricing", {})
        p_in = parse_price(pricing, "prompt")
        p_out = parse_price(pricing, "completion")
        ctx = m.get("context_length", 0)
        in_wl = whitelist_by_model.get(mid, [])
        wl_str = ", ".join(in_wl) if in_wl else "—"
        print(
            f"  #{rank:<4} {mid:<45} {price_per_million(p_in):>10} "
            f"{price_per_million(p_out):>11}  {fmt_ctx(ctx):>8}  {wl_str}"
        )
    print()

    print("=" * 60)
    print("Agent instructions:")
    print("1. Report any MISSING models — they may need replacing.")
    print("2. Highlight compelling NEW models worth adding to a whitelist.")
    print("3. Note significant pricing changes vs. skill documentation.")
    print("4. If whitelist changes are warranted, propose specific edits.")
    print("5. Flag trending models NOT in whitelists that appear cost-competitive.")
    print()
    print("⛔ DO NOT write any files. Present findings only.")
    print("   All changes require operator approval in a subsequent session.")
    print("   When approved: patch BOTH this script's WHITELISTS dict AND")
    print("   the skill's tier tables atomically in the same session.")

    # Persist current prices for next run's delta comparison
    save_price_cache(current_prices)


if __name__ == "__main__":
    main()
