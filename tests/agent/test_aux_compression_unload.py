"""Aux compression idle-unload: config parsing, probe verdicts, timer semantics.

Contracts (agent/aux_compression_unload.py):
  * Off unless auxiliary.compression.unload_after_seconds > 0 (the TTL is the switch).
    The unload target resolves per endpoint: literal unload_cmd override, else the
    provider entry's unload_cmd, else LM Studio auto-detection; unresolved = off.
    The pre-summary probe records loaded-state, and scheduling arms an unload ONLY when
    the model had to load for the summary (was_offline True) or a timer is already armed
    (idle window counts from the LAST compression), and never for the main conversation
    route.
  * A newer compression resets the idle timer instead of stacking one per run.
  * Firing re-probes: a model that is verifiably not loaded is left alone; the
    configured target fires with {model}/{instance_id} substituted.
  * A summary in flight on that (endpoint, model) blocks eviction: the timer re-arms
    once, and a second in-flight hit skips (the next compression re-arms anyway).
"""

import threading
import time

import pytest

import agent.aux_compression_unload as aux

KEY = ("http://localhost:8642/v1", "intermediate")


# ---- probe verdicts -------------------------------------------------------

LLAMA_SWAP = {"models": [
    {"id": "intermediate", "status": {"value": "loaded"}},
    {"id": "advanced", "status": {"value": "not loaded"}},
]}
LM_STUDIO_NATIVE = {"models": [
    {"key": "m-loaded", "loaded_instances": [{"id": 7}]},
    {"key": "m-off", "loaded_instances": []},
]}
LM_STUDIO_V1 = {"data": [{"id": "m-loaded", "object": "model"}, {"id": "m-off", "object": "model"}]}


def test_probe_verdicts_loaded_offline_absent_unknown():
    assert aux._verdict_from(LLAMA_SWAP, "intermediate") == (True, None)
    assert aux._verdict_from(LLAMA_SWAP, "advanced") == (False, None)
    assert aux._verdict_from(LLAMA_SWAP, "nope") == (False, None)  # absent from served list
    assert aux._verdict_from(LM_STUDIO_NATIVE, "m-loaded") == (True, "7")
    assert aux._verdict_from(LM_STUDIO_NATIVE, "m-off") == (False, None)
    assert aux._verdict_from(LM_STUDIO_V1, "m-loaded") == (None, None)  # no state field
    assert aux._verdict_from({}, "x") == (None, None)  # undecidable


def test_state_falls_back_to_native_when_v1_undecidable(monkeypatch):
    monkeypatch.setattr(aux, "_fetch_models", lambda base, key="": LM_STUDIO_V1)
    monkeypatch.setattr(aux, "_fetch_native_models", lambda base, key="": LM_STUDIO_NATIVE)
    assert aux.probe_aux_state("http://localhost:1234/v1", "m-loaded") == (True, "7")
    # native also undecidable -> unknown, never a guess
    monkeypatch.setattr(aux, "_fetch_native_models", lambda base, key="": LM_STUDIO_V1)
    assert aux.probe_aux_state("http://localhost:1234/v1", "m-loaded") == (None, None)


def test_served_root_strips_v1_suffix():
    assert aux._served_root("http://localhost:8642/v1") == "http://localhost:8642"
    assert aux._served_root("http://127.0.0.1:1234/") == "http://127.0.0.1:1234"


# ---- config parsing -------------------------------------------------------

def _cfg(cmd="", delay=300.0):  # noqa: ANN001
    return {"auxiliary.compression": {"unload_cmd": cmd, "unload_after_seconds": delay}}  # delay: float|str


def test_config_off_without_ttl():
    # the TTL is the switch: unset/0/negative = off regardless of unload_cmd
    assert aux._unload_config(_cfg(cmd="curl x {model}", delay=0))[1] == 0.0
    assert aux._unload_config({"auxiliary.compression": {"unload_cmd": "curl x {model}"}})[1] == aux._DEFAULT_UNLOAD_AFTER_SECONDS
    assert aux._unload_config({}) == ("", 0.0)


def test_config_delay_defaults_and_clamps():
    _, delay = aux._unload_config(_cfg(cmd="curl x", delay="bogus"))
    assert delay == aux._DEFAULT_UNLOAD_AFTER_SECONDS
    assert aux._unload_config(_cfg(cmd="curl x", delay=-5))[1] == 0.0
    # empty-string unload_cmd (the shipped default) means "not set", not "run empty"
    assert aux._unload_config(_cfg())[0] == ""


def test_template_fill_recurses():
    filled = aux._fill_model({"instance_id": "{instance_id}", "list": ["{model}"]}, "intermediate", "7")
    assert filled == {"instance_id": "7", "list": ["intermediate"]}
    # no probe verdict: {instance_id} falls back to the model key (LM Studio accepts it)
    assert aux._fill_model('{"instance_id": "{instance_id}"}', "m", None) == '{"instance_id": "m"}'


# ---- target resolution cascade --------------------------------------------

def test_auto_detects_lmstudio_only(monkeypatch):
    monkeypatch.setattr(aux, "_fetch_native_models", lambda base, key="": LM_STUDIO_NATIVE)
    target = aux._detect_endpoint_unload("http://localhost:1234/v1")
    assert target == {"url": "http://localhost:1234/api/v1/models/unload",
                      "body": {"instance_id": "{instance_id}"},
                      "method": "POST", "base_url": "http://localhost:1234/v1"}
    # an endpoint whose native list lacks loaded_instances is NOT a known server: stays off
    monkeypatch.setattr(aux, "_fetch_native_models", lambda base, key="": {"models": [{"id": "x"}]})
    assert aux._detect_endpoint_unload("http://localhost:9999/v1") is None


def test_resolve_cascade_override_then_entry_then_auto(monkeypatch):
    # only :1234 answers the LM Studio native probe; everything else is unknown
    monkeypatch.setattr(aux, "_fetch_native_models",
                        lambda base, key="": LM_STUDIO_NATIVE if "1234" in base else None)
    entry_cfg = {"custom_providers": [
        {"name": "swap", "base_url": "http://localhost:8642/v1",
         "unload_cmd": "curl -s -X POST http://localhost:8642/api/models/unload/{model}"},
        {"name": "other", "base_url": "http://localhost:7000/v1"},
    ]}
    # 1. literal override wins, no probing
    assert aux.resolve_unload_target("curl x {model}", "http://h/v1", cfg=entry_cfg) == {"cmd": "curl x {model}"}
    # 2. provider entry for THIS endpoint (matched on /v1-stripped root)
    assert aux.resolve_unload_target("", "http://localhost:8642/v1", cfg=entry_cfg) == {
        "cmd": "curl -s -X POST http://localhost:8642/api/models/unload/{model}"}
    # entry exists but for another endpoint: no match -> falls to auto
    assert aux.resolve_unload_target("", "http://localhost:1234/v1", cfg=entry_cfg) == {
        "url": "http://localhost:1234/api/v1/models/unload", "body": {"instance_id": "{instance_id}"},
        "method": "POST", "base_url": "http://localhost:1234/v1"}
    # 3. no override, no entry, not LM Studio -> off
    assert aux.resolve_unload_target("", "http://localhost:9999/v1", cfg=entry_cfg) is None


def test_mapping_shape_direct_http_target(monkeypatch):
    """{path, method, body} on the provider entry -> direct HTTP target, no shell."""
    entry_cfg = {"custom_providers": [
        {"name": "swap", "base_url": "http://localhost:8642/v1",
         "unload_cmd": {"path": "/api/models/unload/{model}", "method": "delete"}},
    ]}
    assert aux.resolve_unload_target("", "http://localhost:8642/v1", cfg=entry_cfg) == {
        "url": "http://localhost:8642/api/models/unload/{model}", "method": "DELETE",
        "body": None, "base_url": "http://localhost:8642/v1"}
    # method defaults to POST; body passes through; mapping override works too
    override = {"path": "/admin/eject", "body": {"model": "{model}", "force": True}}
    assert aux.resolve_unload_target(override, "http://h:1/v1") == {
        "url": "http://h:1/admin/eject", "method": "POST",
        "body": {"model": "{model}", "force": True}, "base_url": "http://h:1/v1"}
    # a mapping without path is unusable -> falls through to auto-detect (off here)
    bad_cfg = {"custom_providers": [{"name": "x", "base_url": "http://localhost:9999/v1",
                                     "unload_cmd": {"method": "DELETE"}}]}
    assert aux.resolve_unload_target("", "http://localhost:9999/v1", cfg=bad_cfg) is None


def test_unload_headers_carry_provider_auth_and_extra(monkeypatch):
    entry_cfg = {"custom_providers": [
        {"name": "swap", "base_url": "http://localhost:8642/v1",
         "unload_cmd": {"path": "/u/{model}"}, "extra_headers": {"X-Key": "abc"}},
    ]}
    monkeypatch.setattr(aux, "_load_config", lambda: entry_cfg)
    assert aux._unload_headers("http://localhost:8642/v1", "tok") == {
        "Authorization": "Bearer tok", "X-Key": "abc"}
    assert aux._unload_headers("", "tok") == {"Authorization": "Bearer tok"}
    assert aux._unload_headers("", "") == {}


def test_ttl_without_target_disables_scheduling(monkeypatch, fired):
    """TTL on, but endpoint resolves to nothing: schedule nothing (pre-summary guard)."""
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(delay=0.05))
    _patch_route(monkeypatch)
    monkeypatch.setattr(aux, "_detect_endpoint_unload", lambda base, key="": None)
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (False, None))
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    assert agent.context_compressor._aux_compression_ctx is None  # unknown endpoint: never probed/armed
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager._timers == {}


def test_ttl_zero_disables_everything(monkeypatch, fired):
    """The TTL is the switch: delay 0 must not probe or schedule even with a valid cmd."""
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=0))
    _patch_route(monkeypatch)
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (False, None))
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    assert agent.context_compressor._aux_compression_ctx is None
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager._timers == {}


# ---- timer manager semantics ---------------------------------------------

@pytest.fixture
def fired(monkeypatch):
    calls = []
    monkeypatch.setattr(aux._AuxUnloadTimerManager, "_fire_unload",
                        staticmethod(lambda target, model, iid, key: calls.append((target, model, iid))))
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (False, None))  # False->skip
    yield calls
    for key in list(aux._manager._timers):
        timer = aux._manager._timers.pop(key)
        aux._manager._tokens.pop(key, None)
        timer.cancel()
    with aux._in_flight_lock:
        aux._in_flight.clear()


def _fire_now(key, token, target, iid=None):
    aux._manager._fire(key, token, 0.0, target, "")


def test_reprobe_skips_when_already_unloaded(fired):
    aux._manager._tokens[KEY] = (token := object())
    _fire_now(KEY, token, {"cmd": "curl x {model}"})
    assert fired == []  # probe said not loaded -> nothing fired


def test_reprobe_true_fires_target(fired, monkeypatch):
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (True, "7"))
    aux._manager._tokens[KEY] = (token := object())
    _fire_now(KEY, token, {"cmd": "curl x {model}"})
    # _fire_unload (stubbed here) receives the raw template and fills placeholders itself;
    # template filling is covered by test_template_fill_recurses.
    assert fired == [({"cmd": "curl x {model}"}, "intermediate", "7")]


def test_in_flight_summary_rearms_instead_of_evicting(fired):
    with aux._in_flight_lock:
        aux._in_flight[KEY] = aux._in_flight.get(KEY, 0) + 1
    aux._manager._tokens[KEY] = (token := object())
    _fire_now(KEY, token, {"cmd": "curl x {model}"})
    assert fired == []  # mid-summary: never evict
    assert aux._manager.is_armed(*KEY)  # re-armed once
    # second hit while still in flight: skip (the next compression re-arms anyway)
    _fire_now(KEY, aux._manager._tokens[KEY], {"cmd": "curl x {model}"})
    assert fired == []


def test_reset_cancels_previous_timer(fired):
    aux._manager.reset("http://h/v1", "m", 60, {"cmd": "curl x"})
    first = aux._manager._timers[("http://h/v1", "m")]
    aux._manager.reset("http://h/v1", "m", 0.05, {"cmd": "curl y {model}"})
    second = aux._manager._timers[("http://h/v1", "m")]
    first.join(2)  # cancel() is async: the cancelled thread exits within moments
    assert first.is_alive() is False and second is not first
    second.join(2)
    assert fired == []  # probe False path skips the fire; timer ran clean
    assert not aux._manager.is_armed("http://h/v1", "m")


def test_stale_fire_does_not_clear_new_owner(fired):
    aux._manager.reset("http://h/v1", "m", 60, {"cmd": "curl x"})
    stale_token = object()
    aux._manager._fire(("http://h/v1", "m"), stale_token, 60, {"cmd": "curl x"}, "")
    # token mismatch returns early; the live timer's bookkeeping survives
    assert aux._manager.is_armed("http://h/v1", "m")


def test_run_unload_cmd_fills_template(tmp_path, monkeypatch):
    out = tmp_path / "fired.txt"
    aux._run_unload_cmd(f"echo model={{model}} iid={{instance_id}} > {out}", "intermediate", "7")
    assert out.read_text().strip() == "model=intermediate iid=7"


# ---- scheduling guards ----------------------------------------------------

class FakeComp:
    """Stands in for ContextCompressor: the probe hook reads these fields."""
    base_url = "http://localhost:9999/v1"
    model = "main-model"
    provider = "custom"
    api_key = ""
    api_mode = ""
    _aux_compression_ctx = None
    _aux_compression_unload_key = None


class FakeAgent:
    base_url = "http://localhost:9999/v1"
    model = "main-model"
    session_id = "test"

    def __init__(self):
        self.context_compressor = FakeComp()


def _patch_route(monkeypatch, base_url="http://localhost:8642/v1", model="intermediate"):
    monkeypatch.setattr(aux, "_resolve_aux_route",
                        lambda agent: (base_url, model, ""))


def test_nothing_scheduled_when_feature_off(monkeypatch, fired):
    monkeypatch.setattr(aux, "_load_config", lambda: {})
    _patch_route(monkeypatch)
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    assert agent.context_compressor._aux_compression_ctx is None
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager._timers == {}


def test_schedules_only_after_a_cold_load(monkeypatch, fired):
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=0.05))
    _patch_route(monkeypatch)
    # offline before the summary, loaded at schedule time (the summary cold-loaded it)
    state = {"loaded": False}
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (state["loaded"], None))
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    assert agent.context_compressor._aux_compression_ctx[3] is True  # was not loaded before the call
    state["loaded"] = True
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager.is_armed(*KEY)
    # the summary finished; timer fires after the idle delay
    aux.clear_aux_compression_in_flight(agent.context_compressor)
    deadline = time.monotonic() + 2
    while aux._manager.is_armed(*KEY) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not aux._manager.is_armed(*KEY)


def test_skipped_summary_never_arms(monkeypatch, fired):
    """A compression that never loaded the aux model (low-context skip / fallback lane)
    must not arm the idle unload — there is no idle load of ours to evict."""
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=0.05))
    _patch_route(monkeypatch)
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (False, None))
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    assert agent.context_compressor._aux_compression_ctx[3] is True
    aux.schedule_aux_unload_after_compression(agent)  # still offline at schedule time
    assert aux._manager._timers == {}


def test_unknown_probe_never_arms(monkeypatch, fired):
    """Probe can't decide at note time: conservative — never arm (only a known
    cold-load arms; the timer's own pre-fire probe is the second backstop)."""
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=60))
    _patch_route(monkeypatch)
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (None, None))
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    assert agent.context_compressor._aux_compression_ctx[3] is False  # unknown probe: not a known cold-load
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager._timers == {}


def test_warm_model_never_arms_and_main_route_guarded(monkeypatch, fired):
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=0.05))
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (True, None))
    _patch_route(monkeypatch)
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager._timers == {}  # was already loaded -> not ours to evict

    # aux on the SAME route as the main model: never armed, even when "offline"
    _patch_route(monkeypatch, base_url=FakeAgent.base_url, model="main-model")
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (False, None))
    agent2 = FakeAgent()
    aux.note_aux_state_before_summary_call(agent2.context_compressor)
    aux.schedule_aux_unload_after_compression(agent2)
    assert aux._manager._timers == {}
    assert getattr(agent2.context_compressor, "_aux_compression_ctx", None) in (None, False) or agent2.context_compressor._aux_compression_ctx[3] is False


def test_warm_compression_extends_existing_timer(monkeypatch, fired):
    """Idle window counts from the LAST compression, not from the cold-load one."""
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=60))
    _patch_route(monkeypatch)
    state = {"loaded": False}
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (state["loaded"], None))
    agent = FakeAgent()
    aux.note_aux_state_before_summary_call(agent.context_compressor)
    state["loaded"] = True  # the cold summary loaded it
    aux.schedule_aux_unload_after_compression(agent)
    assert aux._manager.is_armed(*KEY)
    aux.clear_aux_compression_in_flight(agent.context_compressor)
    # second compression: model already warm (probe True), timer still re-armed
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": (True, None))
    agent2 = FakeAgent()
    aux.note_aux_state_before_summary_call(agent2.context_compressor)
    aux.schedule_aux_unload_after_compression(agent2)
    assert aux._manager.is_armed(*KEY)
    aux.clear_aux_compression_in_flight(agent2.context_compressor)


# ---- probe placement (the optimization): probe at the call site, not before compress() ----

def _real_compressor():
    from unittest.mock import patch
    from agent.context_compressor import ContextCompressor
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="main-model", summary_model_override="intermediate", quiet_mode=True)


def test_skipped_summarisation_pays_no_probe(monkeypatch, fired):
    """A compression that never reaches the summary LLM must not probe loaded-state at all."""
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=60))
    probes = []
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": probes.append(1) or (False, None))
    comp = _real_compressor()
    # structural no-op: nothing eligible to compress (tiny transcript, forced)
    comp.compress([{"role": "user", "content": "hi"}], current_tokens=10, force=True)
    assert probes == []
    assert comp._aux_compression_ctx is None


def test_summary_call_probes_once_across_retry(monkeypatch, fired):
    """The probe fires at the single aux-call seam; a main-model retry reuses the verdict."""
    from unittest.mock import MagicMock, patch
    monkeypatch.setattr(aux, "_load_config", lambda: _cfg(cmd="curl x {model}", delay=60))
    _patch_route(monkeypatch)
    probes = []
    monkeypatch.setattr(aux, "probe_aux_state", lambda base, model, key="": probes.append(1) or (False, None))
    comp = _real_compressor()

    def _truncated(*a, **k):
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "partial"
        choice.finish_reason = "length"
        resp.choices = [choice]
        return resp

    with patch("agent.context_compressor.call_llm", side_effect=_truncated):
        comp._generate_summary([{"role": "user", "content": "x " * 50}])
    assert comp._aux_compression_ctx == ("http://localhost:8642/v1", "intermediate", "", True)
    assert len(probes) == 1  # the retry on the main model must not re-probe
