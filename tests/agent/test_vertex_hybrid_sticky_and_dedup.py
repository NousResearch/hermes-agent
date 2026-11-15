import json
import os
from agent import vertex_adapter
from tools import file_tools_read_tracking as ftrt


def test_canonical_affinity_session_id_strips_bg_review():
    assert (
        vertex_adapter._canonical_affinity_session_id("20261114_010000_abc123:bg-review:2")
        == "20261114_010000_abc123"
    )
    assert (
        vertex_adapter._canonical_affinity_session_id("20261114_010000_abc123")
        == "20261114_010000_abc123"
    )


def test_turn1_sub_45k_global_records_sticky_affinity(monkeypatch):
    vertex_adapter.reset_vertex_hybrid_router_state()
    monkeypatch.setattr(vertex_adapter, "vertex_regional_cooldown_remaining", lambda: 10.0)
    region, slot = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=True,
        session_id="sess_turn1_39k",
        last_prompt_tokens=39_000,
    )
    assert region == "global"
    assert slot is False
    assert vertex_adapter.get_session_vertex_region_affinity("sess_turn1_39k") == "global"

    # Even after regional cooldown expires on Turn 2 (44k tokens), session stays pinned to global
    monkeypatch.setattr(vertex_adapter, "vertex_regional_cooldown_remaining", lambda: 0.0)
    region2, slot2 = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=True,
        session_id="sess_turn1_39k",
        last_prompt_tokens=44_000,
    )
    assert region2 == "global"
    assert slot2 is False


def test_bg_review_inherits_parent_europe_west2_affinity():
    vertex_adapter.reset_vertex_hybrid_router_state()
    parent_sid = "sess_parent_ew2"
    bg_sid = f"{parent_sid}:bg-review:1"
    vertex_adapter._record_session_vertex_region_affinity(parent_sid, "europe-west2")
    region, slot = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=False,
        session_id=bg_sid,
        last_prompt_tokens=120_000,
    )
    assert region == "europe-west2"
    assert slot is True
    vertex_adapter._EUROPE_WEST2_ARGON_SEMAPHORE.release()

    # Real build_cache_parity_fork shares parent.session_id directly (without :bg-review:)
    region_direct, slot_direct = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=False,
        session_id=parent_sid,
        last_prompt_tokens=0,
    )
    assert region_direct == "europe-west2"
    assert slot_direct is True
    vertex_adapter._EUROPE_WEST2_ARGON_SEMAPHORE.release()


def test_seed_file_dedup_from_messages_seeds_surviving_reads(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n" * 60, encoding="utf-8")
    resolved = os.path.realpath(str(f))
    tid = "sess_resume_1"
    dedup_key = (resolved, 1, 2000)

    with ftrt._read_tracker_lock:
        td = ftrt._task_data(tid)
        td["dedup"][dedup_key] = ftrt._file_metadata(resolved)
        td["dedup_generation_reads"].add(dedup_key)

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": resolved, "offset": 1, "limit": 2000}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"content": "1|x = 1\n" * 60, "total_lines": 60}),
        },
    ]
    ftrt.reset_file_dedup(tid, surviving_messages=messages)
    _, _, stub_eligible = ftrt._dedup_lookup(tid, dedup_key)
    assert stub_eligible is True


def test_post_compaction_sentinel_preserves_europe_west2_affinity_and_clears_global():
    vertex_adapter.reset_vertex_hybrid_router_state()
    vertex_adapter._record_session_vertex_region_affinity("child_ew2", "europe-west2")
    region, slot = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=True,
        session_id="child_ew2",
        last_prompt_tokens=-1,
    )
    assert region == "europe-west2"
    assert slot is True
    vertex_adapter._EUROPE_WEST2_ARGON_SEMAPHORE.release()

    vertex_adapter._record_session_vertex_region_affinity("stuck_global", "global")
    region_g, slot_g = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=True,
        session_id="stuck_global",
        last_prompt_tokens=-1,
    )
    assert region_g == "europe-west2"
    assert slot_g is True
    vertex_adapter._EUROPE_WEST2_ARGON_SEMAPHORE.release()


def test_non_interactive_with_europe_west2_configured_region_routes_to_global_when_parent_not_ew2():
    vertex_adapter.reset_vertex_hybrid_router_state()
    vertex_adapter._record_session_vertex_region_affinity("parent_global", "global")
    region_bg, slot_bg = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="europe-west2",
        interactive=False,
        session_id="parent_global:bg-review:1",
        last_prompt_tokens=90_000,
    )
    assert (region_bg, slot_bg) == ("global", False)

    region_cron, slot_cron = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="europe-west2",
        interactive=False,
        session_id="cron_job_1",
        last_prompt_tokens=15_000,
    )
    assert (region_cron, slot_cron) == ("global", False)


def test_record_vertex_regional_429_pins_session_to_global_without_clobbering_from_bg_review(monkeypatch):
    vertex_adapter.reset_vertex_hybrid_router_state()
    vertex_adapter._record_session_vertex_region_affinity("sess_ew2", "europe-west2")
    vertex_adapter.record_vertex_regional_429(cooldown_seconds=5.0, session_id="sess_ew2")
    assert vertex_adapter.vertex_regional_cooldown_remaining() > 0.0
    assert vertex_adapter.get_session_vertex_region_affinity("sess_ew2") == "global"

    # Even after regional cooldown expires, sess_ew2 remains pinned to global for the epoch
    monkeypatch.setattr(vertex_adapter, "vertex_regional_cooldown_remaining", lambda: 0.0)
    region, slot = vertex_adapter.acquire_vertex_hybrid_region(
        "google/gemini-4-argon",
        configured_region="global",
        interactive=True,
        session_id="sess_ew2",
        last_prompt_tokens=20_000,
    )
    assert (region, slot) == ("global", False)

    # A :bg-review: session_id trips cooldown without overwriting parent's europe-west2 affinity
    vertex_adapter._record_session_vertex_region_affinity("sess_parent_keep", "europe-west2")
    vertex_adapter.record_vertex_regional_429(
        cooldown_seconds=5.0, session_id="sess_parent_keep:bg-review:1"
    )
    assert vertex_adapter.get_session_vertex_region_affinity("sess_parent_keep") == "europe-west2"


def test_route_classified_error_trips_vertex_regional_cooldown_on_empty_fallback_chain_and_timeout():
    from types import SimpleNamespace
    from agent.error_classifier import FailoverReason
    from agent.turn_recovery import TurnRetryState, route_classified_error

    # 1. Subagent with empty _fallback_chain hitting 429 on europe-west2
    vertex_adapter.reset_vertex_hybrid_router_state()
    vertex_adapter._record_session_vertex_region_affinity("subagent_ew2", "europe-west2")
    subagent = SimpleNamespace(
        provider="vertex",
        session_id="subagent_ew2",
        _fallback_index=0,
        _fallback_chain=[],
        _fallback_activated=False,
        _credential_pool=None,
        _write_origin=None,
        _memory_write_origin=None,
        _turn_origin=None,
        _buffer_diagnostic_status=lambda *_a, **_k: None,
    )
    classified_429 = SimpleNamespace(
        reason=FailoverReason.rate_limit,
        error_context={},
        compaction_helped=False,
        is_auth=False,
        should_fallback=False,
    )
    route_classified_error(
        subagent,
        RuntimeError("HTTP 429: TPM quota exhausted"),
        classified_429,
        TurnRetryState(),
        error_msg="HTTP 429: TPM quota exhausted",
        error_context={},
        recovered_with_pool=False,
        base_url="https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/p/locations/europe-west2/endpoints/openapi",
        model="google/gemini-4-argon",
        messages=[],
        api_messages=[],
        system_message=None,
        active_system_prompt="",
        conversation_history=[],
        retry_count=1,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=2,
        api_call_count=1,
        effective_task_id="subagent_ew2",
    )
    assert vertex_adapter.vertex_regional_cooldown_remaining() > 0.0
    assert vertex_adapter.get_session_vertex_region_affinity("subagent_ew2") == "global"

    # 2. Interactive session hitting StreamStaleTimeout (FailoverReason.timeout) on retry #1
    vertex_adapter.reset_vertex_hybrid_router_state()
    vertex_adapter._record_session_vertex_region_affinity("sess_timeout_ew2", "europe-west2")
    subagent.session_id = "sess_timeout_ew2"
    classified_timeout = SimpleNamespace(
        reason=FailoverReason.timeout,
        error_context={},
        compaction_helped=False,
        is_auth=False,
        should_fallback=False,
    )
    route_classified_error(
        subagent,
        TimeoutError("Stream stale for 600s"),
        classified_timeout,
        TurnRetryState(),
        error_msg="Stream stale for 600s",
        error_context={},
        recovered_with_pool=False,
        base_url="https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/p/locations/europe-west2/endpoints/openapi",
        model="google/gemini-4-argon",
        messages=[],
        api_messages=[],
        system_message=None,
        active_system_prompt="",
        conversation_history=[],
        retry_count=1,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=2,
        api_call_count=1,
        effective_task_id="sess_timeout_ew2",
    )
    assert vertex_adapter.vertex_regional_cooldown_remaining() > 0.0
    assert vertex_adapter.get_session_vertex_region_affinity("sess_timeout_ew2") == "global"


def test_cron_split_modules_concurrent_import_no_py314_deadlock():
    import importlib
    import sys
    import threading

    targets = (
        "cron",
        "cron.scheduler",
        "cron.scheduler_delivery",
        "cron.scheduler_delivery_standalone",
        "cron.scheduler_script",
        "cron.scheduler_preflight",
        "cron.scheduler_prompt",
    )
    for m in targets:
        importlib.import_module(m)
    saved = {k: v for k, v in sys.modules.items() if k == "cron" or k.startswith("cron.")}
    try:
        for _ in range(8):
            for k in list(sys.modules):
                if k == "cron" or k.startswith("cron."):
                    sys.modules.pop(k, None)
            barrier = threading.Barrier(len(targets))
            errors: list = []

            def worker(mod_name: str):
                try:
                    barrier.wait(timeout=5.0)
                    mod = importlib.import_module(mod_name)
                    if mod_name == "cron":
                        assert callable(mod.tick)
                    elif mod_name == "cron.scheduler_delivery":
                        assert callable(mod._sched.tick)
                        assert callable(mod._standalone_send)
                except BaseException as exc:
                    errors.append((mod_name, repr(exc)))

            threads = [threading.Thread(target=worker, args=(m,)) for m in targets]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            assert not errors, f"Concurrent cron import failed: {errors}"
    finally:
        for k in list(sys.modules):
            if k == "cron" or k.startswith("cron."):
                sys.modules.pop(k, None)
        sys.modules.update(saved)
        cron_pkg = saved.get("cron")
        if cron_pkg is not None:
            for k, v in saved.items():
                if k.startswith("cron.") and "." not in k[5:]:
                    setattr(cron_pkg, k[5:], v)

