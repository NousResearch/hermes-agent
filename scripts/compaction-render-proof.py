#!/usr/bin/env python3
"""compaction-render-proof — drive a REAL LCM compaction and print the granular
in-turn announce, as the fidelity backstop for the committed stub test (AC-5).

By default uses NO model (the L3 deterministic-truncation summarizer — offline,
the same path the committed test exercises). With ``--real-model <name>`` it runs
the real cheap-model summarizer end-to-end, so the rendered announce reflects the
actual summarizer's output shape (the sole defense against stub-vs-real drift on a
path with a 3-of-5 production-degrade history).

ISOLATION (INV-4): always uses a throwaway temp HERMES_HOME + temp LCM store; it
never touches the live ~/.hermes/lcm.db and never restarts a gateway.

Usage:
    python scripts/compaction-render-proof.py                 # offline (L3 truncation)
    python scripts/compaction-render-proof.py --real-model claude-app/claude-haiku
    python scripts/compaction-render-proof.py --real-model openai-codex/gpt-5.5
    python scripts/compaction-render-proof.py --pairs 60      # bigger turn
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile


def _tool_heavy_turn(n_pairs: int):
    msgs = [{"role": "system", "content": "You are Apollo."}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"do task {i} " + ("w" * 30)})
        msgs.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"calling tool {i}"}],
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "run", "arguments": f'{{"i": {i}}}'}}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": [{"type": "tool_result", "content": f"result {i} " + ("z" * 60)}],
        })
    return msgs


def _register_runtime_for(model_slug: str) -> None:
    """Register ``model_slug`` as the live 'main runtime' so the LCM summarizer's
    ``call_llm(task="compression")`` → ``auto`` resolution picks it — exactly what
    ``turn_context.py:95`` does on every real turn. Resolves the provider's
    base_url + api_key from the built-in PROVIDER_REGISTRY (so claude-app → the
    relay's /anthropic endpoint + CLAUDE_POOL_KEY, codex → its own, etc.).

    Slug forms accepted: ``provider/model`` or a bare ``model`` (defaults to the
    config's main provider).
    """
    import os as _os

    from agent.auxiliary_client import set_runtime_main

    if "/" in model_slug:
        provider, model = model_slug.split("/", 1)
    else:
        provider, model = "claude-app", model_slug

    base_url = ""
    api_key = ""
    api_mode = ""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        entry = PROVIDER_REGISTRY.get(provider)
        if entry is not None:
            base_url = getattr(entry, "inference_base_url", "") or ""
            for k in getattr(entry, "api_key_env_vars", ()) or ():
                if _os.environ.get(k):
                    api_key = _os.environ[k]
                    break
    except Exception as exc:
        print(f"[warn] provider registry lookup failed for {provider}: {exc}")

    set_runtime_main(
        provider=provider, model=model,
        base_url=base_url, api_key=api_key, api_mode=api_mode,
    )
    print(f"[real-model] runtime registered: provider={provider} model={model} "
          f"base_url={base_url or '(default)'} key={'set' if api_key else 'MISSING'}")
    # Fail LOUDLY rather than silently degrading to the off-gateway auto-chain:
    # if we couldn't resolve a base_url+key for this provider, the summarizer would
    # fall through to openrouter/nous (unhealthy off-gateway) and 400 — exactly the
    # confusing failure this driver exists to avoid. Surface it (Greptile #113).
    if not base_url or not api_key:
        raise SystemExit(
            f"[real-model] cannot resolve credentials for provider={provider!r} "
            f"(base_url={'set' if base_url else 'EMPTY'}, key={'set' if api_key else 'EMPTY'}). "
            f"Pass an in-registry provider/model (e.g. claude-app/claude-haiku-4-5) and ensure "
            f"its key env var is loaded (this driver loads ~/.hermes/.env for --real-model)."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-model", default=None,
                    help="provider/model for a real summarizer (e.g. claude-app/claude-haiku). "
                         "Omit for the offline L3-truncation path.")
    ap.add_argument("--pairs", type=int, default=40, help="user/assistant/tool triples to generate")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="render-proof-")

    # Record the live db mtime up front so we can prove we didn't touch it.
    live_db = os.path.join(os.path.expanduser("~"), ".hermes", "lcm.db")
    live_before = os.path.getmtime(live_db) if os.path.exists(live_db) else None

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # For the real-model run we need provider credentials. Load them from the REAL
    # ~/.hermes/.env BEFORE pointing HERMES_HOME at the throwaway dir (the LCM DB +
    # sample sink stay isolated in tmp; only the credentials come from the real home).
    if args.real_model:
        try:
            from pathlib import Path
            from hermes_cli.env_loader import load_hermes_dotenv
            load_hermes_dotenv(hermes_home=str(Path.home() / ".hermes"))
        except Exception as exc:
            print(f"[warn] could not load .env for real-model run: {exc}")
    os.environ["HERMES_HOME"] = tmp  # isolate the sink + LCM store

    from agent.compaction_stats import build_inturn_stats
    from agent.model_metadata import estimate_messages_tokens_rough as _est
    from plugins.context_engine.lcm.config import LCMConfig
    from plugins.context_engine.lcm.engine import LCMEngine

    cfg_kwargs = dict(
        database_path=os.path.join(tmp, "proof.db"),
        leaf_chunk_tokens=120,
        condensation_fanin=2,
        fresh_tail_count=4,
        context_threshold=0.05,
        incremental_max_depth=3,
    )
    if args.real_model:
        # Point the LCM summarizer at the real cheap model.
        cfg_kwargs["summary_model"] = args.real_model
        print(f"[real-model] summarizer = {args.real_model}")
    else:
        print("[offline] summarizer = L3 deterministic truncation (no model)")

    cfg = LCMConfig(**cfg_kwargs)
    eng = LCMEngine(config=cfg, hermes_home=tmp)
    eng.on_session_start("proof-sess")

    if args.real_model:
        # Mimic what a live turn does: the LCM summarizer calls
        # call_llm(task="compression") which resolves "auto" → the agent's CURRENT
        # main runtime via set_runtime_main() (turn_context.py:95). A bare script
        # has no AIAgent, so nothing populates that — without this the summarizer
        # falls through to the openrouter/nous auto-detect chain (unhealthy
        # off-gateway) and 400s. Register the requested model's provider/base_url/
        # key as the "main runtime" so the real-model path resolves exactly as it
        # would inside the gateway.
        _register_runtime_for(args.real_model)

    pre = _tool_heavy_turn(args.pairs)
    print(f"input: {len(pre)} messages")
    compressed = eng.compress(list(pre), current_tokens=10**9)
    print(f"status: {eng._last_compression_status}  |  compressed: {len(compressed)} messages")

    stats = build_inturn_stats(messages=pre, compressed=compressed, estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    print(f"validate: {ok}  {why}")
    print(f"approx_attribution (False = exact Option-B split): {stats.approx_attribution}")
    print(f"folded={stats.folded_count}  kept_pre={stats._kept_pre_messages}  pre={stats.pre_messages}")
    print("--- rendered granular announce ---")
    try:
        from agent.conversation_compression import _format_granular_announce
        head = f"🗜️ Context compacted ({eng._last_compression_status})"
        print(_format_granular_announce(
            head, stats, model_part="lcm", after_fallback=False,
            window_from=None, window_to=None,
        ))
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        print(f"(formatter call failed: {exc})")
        print(f"Context compacted: {stats.pre_messages}→{len(compressed)} messages "
              f"({stats.folded_count} folded, {stats._kept_pre_messages} kept)")

    fn = getattr(eng, "shutdown", None)
    if callable(fn):
        fn()

    live_after = os.path.getmtime(live_db) if os.path.exists(live_db) else None
    untouched = live_before == live_after
    print(f"\nlive ~/.hermes/lcm.db untouched: {untouched}")
    return 0 if (ok and untouched) else 1


if __name__ == "__main__":
    raise SystemExit(main())
