#!/usr/bin/env python3
"""Phase 0.2 baseline freezer for the unify-cost-engine parity oracle.

Captures a stratified 7d session sample into spend_frozen_sessions.json and the
CURRENT spend_core.session_cost output for each into spend_core_golden.json.

MUST be run BEFORE any pricing-code edit, with the pinned tier, so the parity
oracle reflects upstream/today's behavior:

    SPEND_CODEX_TIER=priority venv/bin/python \
        tests/fixtures/_freeze_spend_baseline.py

It imports the live ~/.hermes/scripts/spend_core.py (outside the repo) which is
the reference flow Phase 3 must reproduce.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

HERMES_HOME = os.path.expanduser("~/.hermes")
DB = os.path.join(HERMES_HOME, "state.db")
SCRIPTS = os.path.join(HERMES_HOME, "scripts")
FIX_DIR = os.path.dirname(os.path.abspath(__file__))

# Pin the codex tier so the golden is reproducible.
os.environ.setdefault("SPEND_CODEX_TIER", "priority")

if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import spend_core  # noqa: E402  (the live reference flow)

ROW_COLS = (
    "id",
    "parent_session_id",
    "model",
    "billing_provider",
    "billing_base_url",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def _clean(row: dict) -> dict:
    out = {}
    for k in ROW_COLS:
        v = row.get(k)
        if k.endswith("_tokens"):
            v = int(v or 0)
        out[k] = v
    return out


def sample_real_rows() -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cut = time.time() - 7 * 86400
    cols = ", ".join(ROW_COLS)
    picks: list[dict] = []
    seen: set[str] = set()

    # Stratified queries: each grabs a few representative real rows.
    strata = [
        # codex gpt-* flagships (C1: company-OAuth $0 bug)
        f"SELECT {cols} FROM sessions WHERE billing_provider='openai-codex' "
        "AND model='gpt-5.5' AND input_tokens>0 AND started_at>=? "
        "ORDER BY input_tokens DESC LIMIT 3",
        f"SELECT {cols} FROM sessions WHERE billing_provider='openai-codex' "
        "AND model='gpt-5.4' AND (input_tokens>0 OR output_tokens>0) AND started_at>=? "
        "ORDER BY input_tokens DESC LIMIT 2",
        f"SELECT {cols} FROM sessions WHERE billing_provider='openai-codex' "
        "AND model='gpt-5.4-mini' AND (input_tokens>0 OR output_tokens>0) AND started_at>=? "
        "ORDER BY input_tokens DESC LIMIT 2",
        # mislabeled: anthropic model stamped openai-codex (C2)
        f"SELECT {cols} FROM sessions WHERE billing_provider='openai-codex' "
        "AND model LIKE 'us.anthropic.%' AND input_tokens>0 AND started_at>=? "
        "ORDER BY input_tokens DESC LIMIT 2",
        # bedrock cross-region geo (C3 territory) + bare
        f"SELECT {cols} FROM sessions WHERE billing_provider='bedrock' "
        "AND model LIKE 'us.anthropic.%' AND input_tokens>0 AND started_at>=? "
        "ORDER BY input_tokens DESC LIMIT 3",
        f"SELECT {cols} FROM sessions WHERE billing_provider='bedrock' "
        "AND model NOT LIKE 'us.%' AND model NOT LIKE 'eu.%' AND model LIKE '%claude%' "
        "AND input_tokens>0 AND started_at>=? ORDER BY input_tokens DESC LIMIT 2",
        # cache-bearing bedrock row
        f"SELECT {cols} FROM sessions WHERE billing_provider='bedrock' "
        "AND cache_read_tokens>0 AND started_at>=? ORDER BY cache_read_tokens DESC LIMIT 2",
    ]
    for q in strata:
        for r in con.execute(q, (cut,)):
            d = _clean(dict(r))
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            picks.append(d)
    con.close()
    return picks


def synthetic_rows() -> list[dict]:
    """Edge strata that may be sparse/absent in 7d live data but must be
    covered by the oracle: direct anthropic, openai gpt-4o, eu/apac geo,
    mislabeled gpt stamped bedrock, and a genuinely unknown model."""
    def row(rid, model, provider, base_url=None, inp=120_000, outp=30_000, cr=0, cw=0):
        return {
            "id": rid,
            "parent_session_id": None,
            "model": model,
            "billing_provider": provider,
            "billing_base_url": base_url,
            "input_tokens": inp,
            "output_tokens": outp,
            "cache_read_tokens": cr,
            "cache_write_tokens": cw,
        }

    return [
        row("syn_anthropic_opus", "claude-opus-4-8", "anthropic"),
        row("syn_anthropic_sonnet", "claude-sonnet-4-5", "anthropic", cr=50_000, cw=10_000),
        row("syn_openai_gpt4o", "gpt-4o", "openai"),
        row("syn_bedrock_eu", "eu.anthropic.claude-sonnet-4-5", "bedrock"),
        row("syn_bedrock_apac_opus", "apac.anthropic.claude-opus-4-8", "bedrock", cr=200_000),
        # mislabeled: gpt-5.5 stamped bedrock (C2 family fallback -> codex price)
        row("syn_mislabel_gpt_bedrock", "gpt-5.5", "bedrock"),
        # mislabeled: us.anthropic stamped openai-codex (C2 -> anthropic price)
        row("syn_mislabel_claude_codex", "us.anthropic.claude-opus-4-8", "openai-codex"),
        # codex gpt model not in table -> must fall through, NOT $0 included
        row("syn_codex_unknown_gpt", "gpt-9.9-imaginary", "openai-codex"),
        # genuinely unknown model/provider
        row("syn_unknown_model", "totally-made-up-model", "mysteryprov"),
        # zero-token row
        row("syn_zero", "claude-opus-4-8", "anthropic", inp=0, outp=0),
    ]


def main() -> int:
    rows = sample_real_rows() + synthetic_rows()
    # Capture the golden: CURRENT spend_core.session_cost for each row.
    golden = []
    for r in rows:
        usd, status = spend_core.session_cost(dict(r))
        golden.append({"id": r["id"], "usd": usd, "status": status})

    meta = {
        "captured_at": time.time(),
        "codex_tier": os.environ.get("SPEND_CODEX_TIER"),
        "codex_pricing_version": spend_core.CODEX_PRICING_VERSION,
        "pricing_ok": spend_core.PRICING_OK,
        "row_count": len(rows),
    }
    with open(os.path.join(FIX_DIR, "spend_frozen_sessions.json"), "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=2, sort_keys=True)
    with open(os.path.join(FIX_DIR, "spend_core_golden.json"), "w") as fh:
        json.dump({"meta": meta, "golden": golden}, fh, indent=2, sort_keys=True)

    print(f"froze {len(rows)} rows; tier={meta['codex_tier']} pricing_ok={meta['pricing_ok']}")
    print(f"codex_pricing_version={meta['codex_pricing_version']}")
    nonzero = sum(1 for g in golden if (g['usd'] or 0) > 0.01)
    print(f"golden rows >$0.01: {nonzero}/{len(golden)}")
    for r, g in zip(rows, golden):
        print(f"  {r['id'][:34]:34} {str(r['billing_provider']):13} "
              f"{str(r['model'])[:30]:30} -> ${g['usd']:.6f} [{g['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
