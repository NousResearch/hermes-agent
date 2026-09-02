"""Model-load progress: SSE events -> composite percent -> wait notices.

The 40-second problem: a cold local model streams 16-21 GB of weights
before the first token, and the chat rendered that as the generic
"provider may be slow or overloaded" stall warning. llama-server's child
emits real per-tensor progress which the router relays over /models/sse
ONLY — these tests pin the consumer that turns that stream into the
status route's `loading` field and the chat's load notice."""

from __future__ import annotations

import json
import time

import hermes_cli.local_runtime.load_progress as lp


def setup_function(_fn):
    with lp._lock:
        lp._snapshot.clear()


# ── composite percent ────────────────────────────────────────


def test_composite_percent_text_stage_dominates():
    stages = ["text_model", "spec_model", "mmproj_model"]
    # Text model owns [0, 85): halfway through it reads ~42%.
    assert lp._composite_percent(stages, "text_model", 0.5) == 42  # 0.5*85
    # Extras start where text ends and never regress below it.
    assert lp._composite_percent(stages, "spec_model", 0.0) == 85
    assert lp._composite_percent(stages, "mmproj_model", 1.0) == 100


def test_composite_percent_monotone_across_stage_walk():
    """Walking the stages in llama-server's real order never moves the
    bar backwards — the property that makes the bar trustworthy."""
    stages = ["text_model", "spec_model", "mmproj_model"]
    walk = [("text_model", v / 10) for v in range(11)] + \
           [("spec_model", v / 10) for v in range(11)] + \
           [("mmproj_model", v / 10) for v in range(11)]
    seen = [lp._composite_percent(stages, s, v) for s, v in walk]
    assert seen == sorted(seen)
    assert seen[0] == 0 and seen[-1] == 100


def test_composite_percent_single_stage_is_plain():
    assert lp._composite_percent(["text_model"], "text_model", 0.4) == 40


# ── event application ────────────────────────────────────────


def _loading_event(value: float, current: str = "text_model") -> dict:
    return {"status": "loading",
            "progress": {"stages": ["text_model", "mmproj_model"],
                         "current": current, "value": value}}


def test_loading_events_build_snapshot_and_terminal_clears():
    lp._apply_event("m1", "status_change", _loading_event(0.5))
    snap = lp.get_loading_progress()
    assert "m1" in snap
    assert snap["m1"]["percent"] == 42  # 0.5 * 85 within text stage
    assert snap["m1"]["stage"] == "text_model"

    lp._apply_event("m1", "status_change", {"status": "loaded", "info": {}})
    assert lp.get_loading_progress() == {}


def test_unload_and_failure_clear_too():
    lp._apply_event("m1", "status_change", _loading_event(0.2))
    lp._apply_event("m1", "status_change", {"status": "unloaded", "exit_code": 1})
    assert lp.get_loading_progress() == {}

    lp._apply_event("m2", "status_change", _loading_event(0.9))
    lp._apply_event("m2", "model_remove", {})
    assert lp.get_loading_progress() == {}


def test_progressless_loading_event_keeps_entry_alive():
    """The router's first model_status event says just {status: loading} —
    it must register the load (indeterminate) without inventing a percent."""
    lp._apply_event("m1", "model_status", {"status": "loading"})
    snap = lp.get_loading_progress()
    assert snap["m1"]["percent"] == 0


def test_stale_entries_expire():
    lp._apply_event("m1", "status_change", _loading_event(0.5))
    with lp._lock:
        lp._snapshot["m1"]["ts"] -= lp._STALE_ENTRY_TTL_S + 1
    assert lp.get_loading_progress() == {}


# ── chat wait-notice ─────────────────────────────────────────


def test_load_notice_for_managed_model(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import _managed_local_load_notice

    state = tmp_path / "server.json"
    state.write_text(json.dumps({"base_url": "http://127.0.0.1:18434/v1",
                                 "api_key": "k"}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.state_path",
                        lambda: state)
    lp._apply_event("Qwen-Test", "status_change", _loading_event(0.5))
    monkeypatch.setattr(lp, "_ensure_watcher", lambda: None)

    class _Agent:
        base_url = "http://127.0.0.1:18434/v1"

    notice = _managed_local_load_notice(_Agent(), {"model": "Qwen-Test"})
    assert notice is not None
    assert notice.startswith("⏳ loading Qwen-Test into memory — 42%")

    # Different endpoint (user's own server): never claim its loads.
    class _Other:
        base_url = "http://127.0.0.1:9999/v1"

    assert _managed_local_load_notice(_Other(), {"model": "Qwen-Test"}) is None
    # Managed endpoint but a model that isn't loading: no notice.
    assert _managed_local_load_notice(_Agent(), {"model": "Elsewhere"}) is None


def test_load_notice_matches_desktop_wait_filter():
    """The notices must pass the desktop's providerWaitText regex and parse
    under parseModelLoadWait's shapes — pinned here as plain string
    contracts so the two sides can't drift silently."""
    import re

    accept = r"^(?:⏳|⚠|↻|⚙)\s*(?:waiting on|loading|processing prompt|no (?:output|response)|model returned)"

    load = "⏳ loading Qwen3.6-35B-A3B-UD-Q4_K_M into memory — 43% (responses start once the model is loaded)"
    assert re.match(accept, load)
    m = re.match(r"^⏳\s*loading\s+(.+?)\s+into memory\s+—\s+(\d{1,3})%", load)
    assert m and m.group(1) == "Qwen3.6-35B-A3B-UD-Q4_K_M" and m.group(2) == "43"

    prefill = "⚙ processing prompt — 31%"
    assert re.match(accept, prefill)
    p = re.match(r"^⚙\s*processing prompt(?:\s+—\s+(\d{1,3})%)?", prefill)
    assert p and p.group(1) == "31"

    bare = "⚙ processing prompt"
    assert re.match(accept, bare)
    b = re.match(r"^⚙\s*processing prompt(?:\s+—\s+(\d{1,3})%)?", bare)
    assert b and b.group(1) is None


# ── prefill progress ─────────────────────────────────────────


def test_prefill_notice_for_managed_model(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import _managed_local_load_notice

    state = tmp_path / "server.json"
    state.write_text(json.dumps({"base_url": "http://127.0.0.1:18434/v1",
                                 "api_key": "k"}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.state_path",
                        lambda: state)
    monkeypatch.setattr(lp, "_ensure_watcher", lambda: None)
    # No load in flight; a prefill counter is live.
    monkeypatch.setattr(lp, "get_prefill_progress",
                        lambda model: {"processed": 12288})
    import agent.chat_completion_helpers as cch

    monkeypatch.setattr(cch, "estimate_request_context_tokens",
                        lambda kw: 39551)

    class _Agent:
        base_url = "http://127.0.0.1:18434/v1"

    notice = _managed_local_load_notice(_Agent(), {"model": "Qwen-Test"})
    assert notice == "⚙ processing prompt — 31%"

    # Counter past the estimate (estimator undercounted): no honest
    # denominator, so no percent — never >100%.
    monkeypatch.setattr(cch, "estimate_request_context_tokens", lambda kw: 100)
    notice = _managed_local_load_notice(_Agent(), {"model": "Qwen-Test"})
    assert notice == "⚙ processing prompt"


def test_load_notice_outranks_prefill(tmp_path, monkeypatch):
    """While a load entry exists the load notice wins — prefill can't start
    before the model is resident, so a simultaneous claim means the load
    snapshot is authoritative."""
    from agent.chat_completion_helpers import _managed_local_load_notice

    state = tmp_path / "server.json"
    state.write_text(json.dumps({"base_url": "http://127.0.0.1:18434/v1",
                                 "api_key": "k"}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.state_path",
                        lambda: state)
    monkeypatch.setattr(lp, "_ensure_watcher", lambda: None)
    lp._apply_event("Qwen-Test", "status_change", _loading_event(0.5))
    monkeypatch.setattr(lp, "get_prefill_progress",
                        lambda model: {"processed": 999})

    class _Agent:
        base_url = "http://127.0.0.1:18434/v1"

    notice = _managed_local_load_notice(_Agent(), {"model": "Qwen-Test"})
    assert notice is not None and notice.startswith("⏳ loading")


def test_prefill_progress_reads_busiest_processing_slot(monkeypatch):
    monkeypatch.setattr(lp, "_endpoint", lambda: ("http://127.0.0.1:1", "k"))

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    slots = [
        {"id": 0, "is_processing": False, "n_prompt_tokens_processed": 500},
        {"id": 1, "is_processing": True, "n_prompt_tokens_processed": 42},
        {"id": 2, "is_processing": True, "n_prompt_tokens_processed": 32768},
    ]
    monkeypatch.setattr(lp.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(slots))
    assert lp.get_prefill_progress("m") == {"processed": 32768}

    # Nothing processing -> None (idle slots' counters are leftovers).
    idle = [{"id": 0, "is_processing": False, "n_prompt_tokens_processed": 500}]
    monkeypatch.setattr(lp.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(idle))
    assert lp.get_prefill_progress("m") is None

    # Unreachable server -> None, never an exception.
    def _boom(req, timeout=0):
        raise OSError("refused")

    monkeypatch.setattr(lp.urllib.request, "urlopen", _boom)
    assert lp.get_prefill_progress("m") is None


def test_endpoint_respects_ownership_guard(monkeypatch):
    """The watcher's endpoint MUST come from the ownership-guarded reader.
    Regression: a raw state-file read attached the SSE watcher to a
    foreign install's server on the shared stable port (health answers
    for anyone; only the dead-pid check proves ownership)."""
    import hermes_cli.local_runtime.load_progress as lp

    # Guard says "not ours": no endpoint, regardless of state on disk.
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint._state_endpoint",
                        lambda: None)
    assert lp._endpoint() is None

    monkeypatch.setattr(
        "hermes_cli.local_runtime.endpoint._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:18434/v1", "api_key": "k"})
    assert lp._endpoint() == ("http://127.0.0.1:18434", "k")
