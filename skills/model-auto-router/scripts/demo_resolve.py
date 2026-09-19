#!/usr/bin/env python3
"""Live verification of the `/model auto` router against the user's REAL config.

Usage:
    python3 scripts/model_auto_router/demo_resolve.py [args]

Args (any combination, in addition to a positional override alias):
    --auto          -> is_auto path (default tier)
    --deep          -> deep_intent=True (highest reasoning tier)
    --image         -> has_vision=True (prefer multimodal-capable alias)
    --context 128000 -> large input signal (>64KB budget)

Positional override:
    auto-researcher / researcher   -> prefer_alias manual override path

Exit code is 0 when at least one case produced a pick; non-zero otherwise.
This mirrors what cli_model_switch_mixin._resolve_auto_model does in the handler,
so a failure here means the wiring has not taken effect (not that the router is bad).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli.config import load_config  # noqa: E402
from hermes_cli.models_router import build_aliases_from_config, resolve  # noqa: E402


def _load_aliases() -> dict:
    """Read model_aliases from the live config.yaml (same loader Stage 1 used)."""
    cfg = load_config()
    aliases = cfg.get("model_aliases") or {}
    if not aliases:
        # Fallback: read raw so a missing-config demo still shows something.
        cfg_path = Path(cfg.get("_path", "")) if isinstance(cfg, dict) else None
        if cfg_path and cfg_path.exists():
            aliases = build_aliases_from_config(str(cfg_path))
    return aliases


def run(label: str, *, has_vision: bool, context_bytes: int, deep_intent: bool, prefer_alias=None):
    """One routing case; returns the pick dict (or None)."""
    aliases = _load_aliases()
    if not aliases:
        print(f"[{label}] SKIP — no model_aliases configured")
        return None
    decision = resolve(
        aliases,
        has_vision=has_vision,
        context_bytes=context_bytes,
        deep_intent=deep_intent,
        prefer_alias=prefer_alias,
    )
    print(f"[{label}] picked: {decision.alias or '(none)'}"
          f"  provider_model={decision.provider_model}"
          f"  tier={decision.tier}")
    print(f"            reason: {decision.reason}")
    return {"alias": decision.alias, "reason": decision.reason}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live /model auto router verification")
    parser.add_argument("override", nargs="?", default=None,
                        help="manual override alias e.g. 'auto-researcher' or 'researcher'")
    parser.add_argument("--deep", action="store_true", help="deep_intent=True")
    parser.add_argument("--image", action="store_true", help="has_vision=True")
    parser.add_argument("--context", type=int, default=0,
                        help="context_bytes signal (default 8000 = small input)")
    args = parser.parse_args(argv)

    context_bytes = args.context or 8000
    prefer_alias = None
    if args.override:
        # 'auto-researcher' -> researcher; plain 'researcher' -> researcher too.
        prefer_alias = args.override.removeprefix("auto-")

    cases = [
        ("auto",       dict(has_vision=False, context_bytes=context_bytes, deep_intent=False)),
        ("auto+deep",  dict(has_vision=False, context_bytes=context_bytes, deep_intent=True)),
        ("auto+image", dict(has_vision=True,  context_bytes=context_bytes, deep_intent=False)),
    ]
    picks = []
    for label, kw in cases:
        result = run(label, **kw)
        if result and result.get("alias"):
            picks.append(result)

    if prefer_alias:
        result = run(f"override:{prefer_alias}", has_vision=False,
                     context_bytes=args.context or 8000, deep_intent=False,
                     prefer_alias=prefer_alias)
        if result and result.get("alias"):
            picks.append(result)

    print()
    if picks:
        print(f"PASS — {len(picks)} routing case(s) produced a pick")
        return 0
    print("FAIL — no routing case produced a pick; check _resolve_auto_model wiring")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
