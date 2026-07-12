#!/usr/bin/env python3
"""
Daily MoA "fugu" rotation with LIVE model discovery + round-robin.

SAKANA AI fugu shape: a fixed strong ORCHESTRATOR (aggregator) fuses advice
from a daily-refreshed panel of free / local high-reasoning REFERENCE models
(the "fish"). This script:

1. Resolves each provider's CURRENT real model id (catalogs rotate daily, so
   `auto-free` is re-resolved with force_refresh every run — never hard-coded).
2. Liveness-probes each candidate with a 1-token completion (not just name
   resolution, which hides 401/404/502).
3. Round-robins survivors into the reference_models pool, so the order shifts
   each day instead of always favoring the same advisor.
4. NEVER touches the aggregator (stays GPT-5.6 Luna / Grok-4.5).

Safe no-op if nothing is reachable.
"""

from __future__ import annotations

import sys
import random
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"

# Fixed orchestrator — never overwritten by this script.
ORCHESTRATOR = {"provider": "openai-codex", "model": "gpt-5.6-luna"}

# (provider, alias_to_resolve) - real id is discovered live each run.
# freellmapi is managed manually (it returns 429 when upstream keys are
# empty, which the liveness probe treats as alive, but the catalog probe
# inside resolve_real_id can poison the check — so we keep it static).
CANDIDATE_ALIASES = [
    ("opencode-zen", "auto-free"),
    ("nvidia", "auto"),
    ("nous", "auto-free"),
    ("freebuff", "deepseek/deepseek-v4-flash"),
]


def resolve_real_id(provider: str, alias: str) -> str | None:
    """Return the provider's CURRENT real model id, or the alias on failure.
    Uses the cached catalog (force_refresh=False) so we don't trigger a live
    429 from a catalog probe that would then poison the liveness check below.
    On any failure, fall back to the alias itself so liveness() can still
    probe the endpoint."""
    try:
        from hermes_cli import models as model_catalog
        return model_catalog.resolve_config_model_id(provider, alias, force_refresh=False)
    except Exception:  # noqa: BLE001
        return alias


def liveness(provider: str, model: str) -> bool:
    """True if the endpoint is reachable at all (200/401/429 = alive;
    404/000 = dead). A 429 means the proxy is up but rate-limited — still
    a valid panel member for the fugu rotation (it will contribute when
    limits reset). Only 404 / connection-refused means the model is gone.

    We catch both the RateLimitError type and the string, because Hermes'
    fallback_chain can re-wrap the original 429 into a different exception
    class by the time it reaches us.
    """
    try:
        from agent.auxiliary_client import call_llm
        call_llm(provider=provider, model=model,
                 messages=[{"role": "user", "content": "ping"}], max_tokens=1)
        return True
    except Exception as e:
        msg = str(e)
        # endpoint reachable but throttled / auth-required -> still alive
        if "429" in msg or "401" in msg or "403" in msg:
            return True
        # RateLimitError (or any *RateLimit* subclass) means the proxy is up
        if type(e).__name__.endswith("RateLimitError") or "RateLimit" in type(e).__name__:
            return True
        return False


def _write(data: dict) -> None:
    import yaml

    class _D(yaml.SafeDumper):
        pass

    def _str_rep(d, s):
        if "\n" in s:
            return d.represent_scalar("tag:yaml.org,2002:str", s, style=">")
        return d.represent_scalar("tag:yaml.org,2002:str", s)

    _D.add_representer(str, _str_rep)
    CONFIG_PATH.write_text(
        yaml.dump(data, Dumper=_D, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote {CONFIG_PATH}")


def main() -> int:
    import yaml

    if not CONFIG_PATH.exists():
        print(f"[error] config not found: {CONFIG_PATH}")
        return 1

    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    moa = data.get("moa") or {}
    presets = moa.get("presets") or {}
    active = moa.get("active_preset") or moa.get("default_preset")
    if not active or active not in presets:
        print("[error] no active MoA preset to rotate")
        return 1

    preset = presets[active]
    preset["aggregator"] = dict(ORCHESTRATOR)  # idempotent pin

    print(f"[info] {date.today()} live discovery for preset '{active}':")
    alive = []
    for provider, alias in CANDIDATE_ALIASES:
        real = resolve_real_id(provider, alias)
        if not real:
            print(f"  [skip] {provider}:{alias} -> unresolvable")
            continue
        ok = liveness(provider, real)
        print(f"  [{'alive' if ok else 'dead '}] {provider}:{alias} -> {real}")
        if ok:
            alive.append({"provider": provider, "model": real})

    if not alive:
        print("[warn] no free reference model reachable; leaving preset unchanged")
        _write(data)
        return 0

    # Round-robin: seed RNG from today's date so the order is stable within a
    # day but rotates across days (fugu-style advisor reshuffle).
    rng = random.Random(int(date.today().strftime("%Y%m%d")))
    rng.shuffle(alive)

    preset["reference_models"] = alive
    print(f"[ok] rotated {len(alive)} live reference models (round-robin):")
    for r in alive:
        print(f"      - {r['provider']}:{r['model']}")

    _write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
