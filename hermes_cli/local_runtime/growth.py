"""In-session context growth for the managed llama.cpp runtime.

The live half of the window ladder (context_policy.growth_decision): when a
session reaches the edge of its granted window, Hermes grows the window
toward the model's native max INSTEAD of compressing. Compression becomes
what the design says it is — the move of last resort, once the window is at
native (or the speed floor / physics say stop).

Mechanism: growth is re-prefill. A per-model window
override is persisted, presets regenerate with the bigger window, the
supervised server bounces, and the next request autoloads the model at the
new window and re-prefills the conversation. Nothing about the Hermes
conversation mutates — no prompt-cache or role-alternation risk; the whole
operation is server-side.

Scope guard: only a server THIS process supervises grows. Detected external
servers and other-process supervisors keep their own policies.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def window_overrides_path():
    from hermes_cli.local_runtime.binaries import runtimes_root

    return runtimes_root() / "window_overrides.json"


def load_window_overrides() -> dict:
    """model_id -> granted window (int). Empty on any read problem."""
    try:
        with open(window_overrides_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): int(v) for k, v in data.items()}
    except Exception:  # noqa: BLE001
        return {}


def save_window_override(model_id: str, window: int) -> None:
    overrides = load_window_overrides()
    overrides[model_id] = int(window)
    path = window_overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=1), encoding="utf-8")


def clear_window_override(model_id: str) -> None:
    """Drop a model's growth state (delete/re-download paths)."""
    overrides = load_window_overrides()
    if model_id in overrides:
        del overrides[model_id]
        window_overrides_path().write_text(
            json.dumps(overrides, indent=1), encoding="utf-8")


def is_managed_endpoint(base_url: str) -> bool:
    """True when base_url is the server this process's state file points at."""
    try:
        from hermes_cli.local_runtime.endpoint import _state_endpoint

        state = _state_endpoint()
        if state is None:
            return False
        return (base_url or "").rstrip("/") == str(
            state.get("base_url", "")).rstrip("/")
    except Exception:  # noqa: BLE001
        return False


def maybe_grow_window(model_id: str, *, base_url: str, session_tokens: int,
                      current_window: int,
                      measured_decode_tok_s: float | None = None) -> int | None:
    """One growth evaluation + execution. Returns the NEW window when the
    ladder granted a bigger one, else None (hold / compress / not ours).

    The caller sits at a request boundary by construction (the pre-API
    compression gate), so re-prefill growth is safe at any call: the next
    request rebuilds server state from scratch in the larger window —
    nothing rewinds.
    """
    from hermes_cli.local_runtime.bootstrap import (
        get_supervisor,
        refresh_local_runtime,
        staged_models,
    )
    from hermes_cli.local_runtime.context_policy import growth_decision
    from hermes_cli.local_runtime.estimator import profile_from_gguf
    from hermes_cli.local_runtime.gguf import read_gguf_header
    from hermes_cli.local_runtime.hardware import probe_budget

    sup = get_supervisor()
    if sup is None or not is_managed_endpoint(base_url):
        return None

    gguf = next((p for p in staged_models()
                 if p.stem.startswith(model_id) or model_id in p.stem), None)
    if gguf is None:
        return None

    try:
        profile = profile_from_gguf(read_gguf_header(gguf))
    except (ValueError, OSError) as exc:
        logger.debug("growth skip %s: unreadable gguf (%s)", model_id, exc)
        return None

    try:
        server_idle = sup.is_idle(model_id)
    except Exception:  # noqa: BLE001
        server_idle = False

    decision = growth_decision(
        # Capacity budget, not live-free: growth executes via a server
        # bounce, so the grown instance loads onto a freed card. Live-free
        # here is distorted by the very model being grown — it reads its
        # own residency as unavailable and vetoes rungs that fit.
        profile, probe_budget(planning=True),
        current_window=current_window,
        session_tokens=session_tokens,
        measured_decode_tok_s=measured_decode_tok_s,
        server_idle=server_idle,
        # The caller IS the occupancy signal: this runs from the agent's
        # compression gate, which fired on its own threshold. Two
        # separately-derived edges must not deadlock into
        # compress-before-grow.
        occupancy_confirmed=True,
    )
    if decision.action != "grow" or not decision.next_window:
        logger.debug("growth %s: %s (%s)", model_id, decision.action, decision.reason)
        return None

    logger.info("context growth %s: %s", model_id, decision.reason)
    save_window_override(model_id, decision.next_window)
    if not refresh_local_runtime():
        # The override still lands at the next boot; report no growth NOW
        # so the caller compresses instead of overflowing a stale window.
        logger.warning("growth %s: server refresh failed; compression proceeds", model_id)
        return None
    return decision.next_window
