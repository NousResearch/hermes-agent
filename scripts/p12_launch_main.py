#!/usr/bin/env python3
"""P12 production launch seam — the single entry point for the main-lane spawn.

Implements the Phase 5 launch-seam contract for
``docs/adr/0012-p12-256k-memory-policy.md`` and
``docs/design/p12-long-context-mode-switch-contract.md``:

  weights GPU0-only (fail-fast)  ->  KV tier decision  ->  final argv

This wrapper is what ``scripts/systemd/turbohaul-main.service`` (and the
Turbohaul manifest-to-argv path) invokes instead of calling ``llama-server``
directly. It is deliberately dependency-free (stdlib only) so it can run in
any launch context, including under systemd with a minimal environment.

Two modes, both provable WITHOUT live activation:

- Fast path (default, ctx <= 128K): ``enforce_gpu0_weights()`` appends the
  canonical ``--split-mode none --main-gpu 0`` pins; ``decide_offload()``
  neutralizes any offload flags so the server stays GPU-KV resident. The
  emitted argv is exactly what a pre-P12 launch passed, plus the two
  behaviour-neutral pins (mirrors ``p12_fast_path_verify.py``'s "enforced"
  argv).
- Long-context route (explicit ``--model-tag darwin-28b-256k``): the
  wrapper computes the mode-switch argv (host-RAM KV via
  ``--no-kv-offload``) only when the opt-in flag is set. Without the flag
  it fails closed: the offload request is rejected and the fast-path argv
  is emitted instead — never a silent partial offload.

Output contract:
- ``--json``: emit ``{"mode": "fast"|"long-context", "allowed": bool,
  "argv": [...]}`` on stdout and exit 0; exit 1 when a weight-placement
  violation or a disallowed offload is detected (fail closed).
- default: print the final argv one token per line (the systemd
  ``ExecStart`` form is ``$(p12_launch_main.py ...)`` style consumers).

Never spawns, never touches the live service, never imports torch — it only
produces the argv a spawner would use. Tests in
``tests/scripts/test_p12_launch_main.py`` prove both paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.p12_offload_gate import decide_offload, ENV_OFFLOAD_FLAG  # noqa: E402
try:  # repo layout: scripts/benchmarks/...; deployed layout: flat scripts/...
    from scripts.benchmarks.p12_weight_placement import enforce_gpu0_weights  # noqa: E402
except ModuleNotFoundError:
    from scripts.p12_weight_placement import enforce_gpu0_weights  # noqa: E402

# The reserved explicit long-context route tag (design doc §3.1).
LONG_CONTEXT_MODEL_TAG = "darwin-28b-256k"

# Default fast-path context ceiling (tokens) — mirrors kv_cache_policy's
# KV_CACHE_OFFLOAD_THRESHOLD default of 131072 (128K).
FAST_PATH_MAX_CONTEXT = 131072


def _strip_leading_sep(argv: list[str]) -> list[str]:
    """Drop a leading ``--`` separator a caller may have passed."""
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def build_launch_argv(
    server_argv: list[str],
    *,
    ctx_size: int,
    model_tag: str | None = None,
    env: dict | None = None,
) -> dict:
    """Return the launch decision dict for the production seam.

    Order matters: weight enforcement is the non-negotiable fail-fast gate
    (there is no safe fallback for weights), THEN the KV tier decision.

    Args:
        server_argv: the llama-server argv (without the binary).
        ctx_size: requested context length in tokens.
        model_tag: optional reserved route tag; ``darwin-28b-256k`` requests
            the explicit long-context path.
        env: environment mapping (defaults to ``os.environ``); used by the
            opt-in flag check so tests can isolate.

    Returns:
        {"mode": "fast"|"long-context", "allowed": bool, "reason": str,
         "argv": [...], "kv_offload": bool}
    """
    env = dict(env) if env is not None else dict(os.environ)
    argv = _strip_leading_sep(list(server_argv))

    # 1. Weights: GPU0-only, fail-fast. Raises WeightPlacementError on any
    #    configuration that would place weights on CPU/system RAM/GPU1.
    enforced = enforce_gpu0_weights(argv, env=env)

    # 2. KV tier: the explicit long-context route is the ONLY way to enter
    #    host-RAM KV mode, and it requires the opt-in flag.
    long_context_requested = model_tag == LONG_CONTEXT_MODEL_TAG
    if long_context_requested:
        decision = decide_offload(enforced, ctx_size=ctx_size, env=env)
        if not decision.allowed:
            # Hard policy violation (weights pinned off GPU0): fail closed
            # with the neutralized argv and a rejection.
            return {
                "mode": "fast",
                "allowed": False,
                "reason": decision.reason,
                "argv": list(decision.argv),
                "kv_offload": False,
            }
        if not decision.enabled:
            # The route was explicitly requested but the policy will not
            # grant it (opt-in flag unset, or ctx below the offload
            # threshold). This is a denial of the long-context request —
            # fail closed to the fast-path argv.
            return {
                "mode": "fast",
                "allowed": False,
                "reason": decision.reason,
                "argv": list(decision.argv),
                "kv_offload": False,
            }
        # Opted-in huge-context job: sanitize_argv keeps existing offload
        # flags but does NOT invent one — the long-context route is defined
        # by host-RAM KV, so ensure --no-kv-offload is present.
        final_argv = list(decision.argv)
        if not any(tok in final_argv for tok in ("--no-kv-offload", "-nokv")):
            final_argv.append("--no-kv-offload")
        return {
            "mode": "long-context",
            "allowed": True,
            "reason": decision.reason,
            "argv": final_argv,
            "kv_offload": True,
        }

    # Default path: no long-context route requested. Neutralize any stray
    # offload flags so the server can never silently enter RAM-KV mode.
    decision = decide_offload(enforced, ctx_size=ctx_size, env=env)
    return {
        "mode": "fast",
        "allowed": True,
        "reason": decision.reason,
        "argv": list(decision.argv),
        "kv_offload": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="P12 production launch seam (weights + KV tier argv builder)"
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        required=True,
        help="requested context length in tokens",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help=f"reserved route tag; '{LONG_CONTEXT_MODEL_TAG}' requests the "
        "explicit long-context path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full decision as JSON on stdout",
    )
    parser.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="llama-server argv (after the binary); optional leading '--' is stripped",
    )
    args, _ = parser.parse_known_args(argv)

    try:
        result = build_launch_argv(
            list(args.argv or []),
            ctx_size=args.ctx_size,
            model_tag=args.model_tag,
        )
    except Exception as exc:  # WeightPlacementError and friends -> fail closed
        payload = {
            "mode": "error",
            "allowed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "argv": [],
            "kv_offload": False,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"p12-launch-error: {payload['reason']}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for token in result["argv"]:
            print(token)
    return 0 if result["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
