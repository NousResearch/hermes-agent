"""Child-process driver: one real AIAgent + real SessionDB, several turns, stdin control.

Run as ``python -m tests.e2e.core.chaos._agent_driver <spec.json>`` with a hermetic
env (see ``_helpers.hermetic_env``). The agent is built the way the CLI builds it:
config.yaml under HERMES_HOME supplies the provider, retry and timeout knobs, and
``agent.max_turns`` becomes the iteration budget. Each turn feeds the previous
turn's messages back as ``conversation_history`` (the CLI's multi-turn contract).

Protocol (one JSON object per stdout line, prefixed ``CHAOS ``; everything else on
stdout/stderr is the agent's own output and is ignored by the parent):
  {"ev": "ready"}                                    agent built
  {"ev": "turn_start", "i": n}
  {"ev": "turn_end", "i": n, "failed": .., "interrupted": .., "completed": ..,
   "final": "...", "exit_reason": "..."}
  {"ev": "closed"}                                   agent.close() returned
stdin lines: ``interrupt`` -> ``agent.interrupt()`` (the CLI's Ctrl+C / Esc path).
"""

from __future__ import annotations

import json
import sys
import threading


def _emit(**payload: object) -> None:
    sys.__stdout__.write("CHAOS " + json.dumps(payload, default=str) + "\n")
    sys.__stdout__.flush()


def main(spec_path: str) -> None:
    spec = json.loads(open(spec_path, encoding="utf-8").read())

    from hermes_cli.config import load_config
    from hermes_state import SessionDB
    from run_agent import AIAgent

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    agent = AIAgent(
        provider=model_cfg.get("provider"),
        base_url=model_cfg.get("base_url"),
        api_key="sk-fake-chaos",
        model=model_cfg.get("default"),
        max_iterations=int((cfg.get("agent") or {}).get("max_turns") or 90),
        session_db=SessionDB(),
        session_id=spec["session_id"],
        quiet_mode=True,
        platform="cli",
    )

    def _control() -> None:
        for line in sys.stdin:
            if line.strip() == "interrupt":
                agent.interrupt("chaos: user interrupt")

    threading.Thread(target=_control, name="chaos-control", daemon=True).start()
    _emit(ev="ready")

    history: list = []
    for i, message in enumerate(spec["turns"]):
        _emit(ev="turn_start", i=i)
        result = agent.run_conversation(message, conversation_history=history)
        history = result.get("messages") or history
        _emit(
            ev="turn_end", i=i,
            failed=bool(result.get("failed")),
            interrupted=bool(result.get("interrupted")),
            completed=bool(result.get("completed")),
            final=(result.get("final_response") or "")[:2000],
            exit_reason=result.get("turn_exit_reason"),
        )
    agent.close()
    _emit(ev="closed")


if __name__ == "__main__":
    main(sys.argv[1])
