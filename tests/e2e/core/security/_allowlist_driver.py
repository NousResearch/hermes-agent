"""Child-process driver for the model-allowlist E2E: one real AIAgent, one real delegated turn.

Run as ``python tests/e2e/core/security/_allowlist_driver.py <spec.json>`` with the
hermetic env the parent builds. The agent is built the way the CLI builds it —
``config.yaml`` under HERMES_HOME supplies the provider, the retry knobs and the
``security.model_allowlist`` under test — so the fallback chain, the auxiliary
routing and ``delegate_task`` all resolve exactly as they do for a user.

Protocol: one JSON object per stdout line prefixed ``ALLOW `` (everything else on
stdout/stderr is the agent's own output and is ignored by the parent)::

    {"ev": "ready"}
    {"ev": "turn_end", "failed": bool, "final": str, "exit_reason": str}
    {"ev": "closed"}

``failed`` is the turn's own verdict, not the test's: the blocked legs are EXPECTED
to end without an answer, and what the parent judges is which host got billed.
"""

from __future__ import annotations

import json
import sys


def _emit(**payload: object) -> None:
    sys.__stdout__.write("ALLOW " + json.dumps(payload, default=str) + "\n")
    sys.__stdout__.flush()


def main(spec_path: str) -> None:
    spec = json.loads(open(spec_path, encoding="utf-8").read())

    from hermes_cli.config import load_config
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_state import SessionDB
    from run_agent import AIAgent

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    agent = AIAgent(
        provider=model_cfg.get("provider"),
        base_url=model_cfg.get("base_url"),
        api_key=model_cfg.get("api_key"),
        model=model_cfg.get("default"),
        # Exactly what cli_init_mixin / oneshot / the gateway hand AIAgent: the configured chain,
        # read through the same shared reader. Without it the driver would run with no fallback at
        # all and could not tell a filtered chain from an absent one.
        fallback_model=get_fallback_chain(cfg) or None,
        max_iterations=int((cfg.get("agent") or {}).get("max_turns") or 6),
        session_db=SessionDB(),
        session_id=spec["session_id"],
        quiet_mode=True,
        platform="cli",
    )
    _emit(ev="ready")
    try:
        result = agent.run_conversation("Read the attached scan and report the totals.")
        _emit(
            ev="turn_end",
            failed=not result.get("final_response"),
            final=str(result.get("final_response") or "")[:400],
            exit_reason=str(result.get("exit_reason") or ""),
        )
    except Exception as exc:  # a surfaced error is a valid outcome for a blocked leg
        _emit(ev="turn_end", failed=True, final="", exit_reason=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        try:
            agent.close()
        except Exception:
            pass
        _emit(ev="closed")


if __name__ == "__main__":
    main(sys.argv[1])
