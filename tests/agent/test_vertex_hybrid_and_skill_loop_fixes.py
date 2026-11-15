"""Regression tests for P0-1 (Vertex Hybrid Router), P1-1 (slash-skill dedup),
P1-2 (background review cache disable), and P0-2 (post-compaction loop guard)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from agent.context_compressor import ContextCompressor
from agent.turn_context_compaction import should_defer_preflight_to_real_usage
from agent.vertex_adapter import (
    EUROPE_WEST2_ARGON_MAX_CONCURRENCY,
    EUROPE_WEST2_VERTEX_REGION,
    GLOBAL_VERTEX_REGION,
    acquire_vertex_hybrid_region,
    build_vertex_base_url,
    is_vertex_argon_model,
    is_vertex_global_only_model,
    parse_vertex_base_url,
    record_vertex_regional_429,
    release_vertex_hybrid_slot,
    reset_vertex_hybrid_router_state,
)
import tools.skills_tool as skills_tool_mod
from tools.skills_tool import skill_view
from tools.skills_tool_dedup import (
    _skill_view_tracker,
    reset_skill_view_dedup,
    seed_skill_view_dedup_from_messages,
)


@pytest.fixture(autouse=True)
def _clean_hybrid_router_and_dedup():
    reset_vertex_hybrid_router_state()
    reset_skill_view_dedup()
    yield
    reset_vertex_hybrid_router_state()
    reset_skill_view_dedup()


def _setup_skill(tmp_path: Path, monkeypatch, name: str = "wrap", body: str = "# Wrap Skill\nRun preflight and close out.\n") -> Path:
    monkeypatch.setattr(skills_tool_mod, "SKILLS_DIR", tmp_path)
    sdir = tmp_path / name
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n{body}",
        encoding="utf-8",
    )
    return sdir


class TestVertexHybridRouterP01:
    def test_parse_vertex_base_url_global_and_regional(self):
        global_url = "https://aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/global/endpoints/openapi"
        assert parse_vertex_base_url(global_url) == ("uop-atlas", "global")

        regional_url = "https://europe-west2-aiplatform.googleapis.com/v1/projects/uop-atlas/locations/europe-west2/endpoints/openapi"
        assert parse_vertex_base_url(regional_url) == ("uop-atlas", "europe-west2")

        assert parse_vertex_base_url("https://openrouter.ai/api/v1") is None

    def test_model_routing_predicates(self):
        assert is_vertex_argon_model("google/gemini-4-argon") is True
        assert is_vertex_argon_model("gemini-4-argon@vertex") is True
        assert is_vertex_argon_model("google/gemini-3.8-flash") is False

        assert is_vertex_global_only_model("google/gemini-3.8-flash") is True
        assert is_vertex_global_only_model("gemini-3-pro") is True
        assert is_vertex_global_only_model("gemini-4-argon") is False

    def test_semaphore_caps_regional_concurrency_at_five_and_falls_back_to_global(self):
        clients = []
        for _ in range(EUROPE_WEST2_ARGON_MAX_CONCURRENCY):
            region, acquired = acquire_vertex_hybrid_region(
                "google/gemini-4-argon",
                configured_region="global",
                interactive=True,
            )
            assert acquired is True
            assert region == EUROPE_WEST2_VERTEX_REGION
            url = build_vertex_base_url("uop-atlas", region)
            assert "europe-west2-aiplatform.googleapis.com" in url
            c = SimpleNamespace(_vertex_regional_slot_held=acquired)
            clients.append(c)

        # 6th concurrent request must spill over to global without blocking
        spill_region, spill_acquired = acquire_vertex_hybrid_region(
            "google/gemini-4-argon",
            configured_region="global",
            interactive=True,
        )
        assert spill_acquired is False
        assert spill_region == GLOBAL_VERTEX_REGION

        # Releasing one slot makes regional routing available again
        assert release_vertex_hybrid_slot(clients.pop()) is True
        next_region, next_acquired = acquire_vertex_hybrid_region(
            "google/gemini-4-argon",
            configured_region="global",
            interactive=True,
        )
        assert next_acquired is True
        assert next_region == EUROPE_WEST2_VERTEX_REGION
        clients.append(SimpleNamespace(_vertex_regional_slot_held=next_acquired))

        for c in clients:
            assert release_vertex_hybrid_slot(c) is True
            # Idempotent second release returns False
            assert release_vertex_hybrid_slot(c) is False

    def test_background_review_and_flash_stay_on_global(self):
        bg_region, bg_acq = acquire_vertex_hybrid_region(
            "google/gemini-4-argon",
            configured_region="global",
            interactive=False,
        )
        assert bg_acq is False
        assert bg_region == GLOBAL_VERTEX_REGION

        flash_region, flash_acq = acquire_vertex_hybrid_region(
            "google/gemini-3.8-flash",
            configured_region="europe-west2",
            interactive=True,
        )
        assert flash_acq is False
        assert flash_region == GLOBAL_VERTEX_REGION

    def test_regional_429_triggers_cooldown_to_global(self):
        record_vertex_regional_429(cooldown_seconds=30.0)
        region, acquired = acquire_vertex_hybrid_region(
            "google/gemini-4-argon",
            configured_region="global",
            interactive=True,
        )
        assert acquired is False
        assert region == GLOBAL_VERTEX_REGION


class TestSlashSkillDedupP11:
    def test_load_skill_payload_marks_dedup_for_session(self, tmp_path, monkeypatch):
        from agent.skill_commands import _load_skill_payload

        _setup_skill(tmp_path, monkeypatch, "wrap")
        res = _load_skill_payload("wrap", task_id="sess-p11-test")
        assert res is not None
        payload, _skill_dir, display_name = res
        assert payload.get("content")
        assert (display_name, "") in _skill_view_tracker.get("sess-p11-test", {})

        # Subsequent _skill_view_with_bump(name="wrap", task_id="sess-p11-test") must return unchanged stub
        from tools.skills_tool import _skill_view_with_bump

        second = json.loads(_skill_view_with_bump({"name": "wrap"}, task_id="sess-p11-test"))
        assert second.get("dedup") is True
        assert second.get("status") == "unchanged"
        assert second.get("content_returned") is False

    def test_seed_skill_view_dedup_scans_user_scaffold_only_when_unpruned(self, tmp_path, monkeypatch):
        from agent.skill_commands import _build_skill_message, _load_skill_payload
        from tools.skills_tool import _skill_view_with_bump

        _setup_skill(tmp_path, monkeypatch, "wrap")
        loaded = _load_skill_payload("wrap", task_id=None)
        assert loaded is not None
        loaded_skill, skill_dir, display_name = loaded
        note = (
            f'[IMPORTANT: The user has invoked the "{display_name}" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]"
        )
        msg_text = _build_skill_message(
            loaded_skill,
            skill_dir,
            note,
            user_instruction="close out",
        )
        messages = [{"role": "user", "content": msg_text}]
        seed_skill_view_dedup_from_messages("sess-scaffold-1", messages)
        assert ("wrap", "") in _skill_view_tracker.get("sess-scaffold-1", {})

        # Subsequent _skill_view_with_bump in that seeded session returns the unchanged stub
        second = json.loads(_skill_view_with_bump({"name": "wrap"}, task_id="sess-scaffold-1"))
        assert second.get("dedup") is True

        # When the scaffold was compacted into a [SKILL_PRUNED] stub, do NOT seed dedup
        pruned_messages = [
            {
                "role": "user",
                "content": (
                    f"{note}\n"
                    "[SKILL_PRUNED: content lost in compression; reload with skill_view(name='wrap')]"
                ),
            }
        ]
        seed_skill_view_dedup_from_messages("sess-scaffold-pruned", pruned_messages)
        assert ("wrap", "") not in _skill_view_tracker.get("sess-scaffold-pruned", {})


class TestBackgroundReviewPromptCacheOffP12:
    """P1-2: Background review fork must disable prompt caching to prevent one-shot write surcharges."""

    def test_background_review_fork_disables_prompt_caching(self):
        import run_agent
        from tests.agent.test_background_review_cache_parity import (
            _make_agent_stub,
            _make_recorder_class,
            _SyncThread,
        )
        from unittest.mock import patch

        agent = _make_agent_stub(run_agent.AIAgent)
        captured: dict = {}
        _Recorder = _make_recorder_class(
            captured,
            record_on_run=("_cache_disabled", "_use_prompt_caching", "_memory_write_origin"),
        )
        with patch.object(run_agent, "AIAgent", _Recorder), patch("threading.Thread", _SyncThread):
            agent._spawn_background_review(
                messages_snapshot=[],
                review_memory=True,
                review_skills=False,
            )

        assert captured.get("_memory_write_origin") == "background_review"
        assert captured.get("_cache_disabled") is True
        assert captured.get("_use_prompt_caching") is False

    def test_background_review_idle_debounce_aborts_before_model_call_when_cancelled(self):
        import threading
        from agent.background_review import (
            _run_review_in_thread,
            prepare_background_review_run,
        )

        parent = SimpleNamespace(
            session_id="sess-debounce-test",
            _background_review_lock=threading.Lock(),
            _background_review_run=None,
            _background_review_agent=None,
        )
        run = prepare_background_review_run(parent)
        assert run is not None

        # Schedule cancellation after 30ms while debounce window is 5.0s
        timer = threading.Timer(0.03, run.cancel_requested.set)
        timer.start()
        try:
            _run_review_in_thread(
                parent,
                [{"role": "user", "content": "hi"}],
                "review prompt",
                task_cfg={"idle_debounce_seconds": 5.0},
                review_run=run,
                explicit=False,
            )
        finally:
            timer.cancel()

        # Review aborted during debounce before creating review_agent
        assert run.cancel_requested.is_set()
        assert parent._background_review_run is None

    def test_vertex_pre_expiry_refresh_rotates_token_in_place_without_evicting_pool(self):
        import threading
        from agent.client_lifecycle import ClientLifecycleMixin, _OPENAI_SLOT

        class _PoolDummy(ClientLifecycleMixin):
            def __init__(self):
                self.provider = "vertex"
                self.model = "google/gemini-4-argon"
                self.api_key = "ya29.old-token"
                self.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"
                self._client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
                self._openai_client_mutex = threading.RLock()
                self.client = SimpleNamespace(api_key="ya29.old-token", _client=SimpleNamespace(is_closed=False))
                self.context_compressor = SimpleNamespace(api_key="ya29.old-token")
                self._primary_runtime = {
                    "provider": "vertex",
                    "api_key": "ya29.old-token",
                    "compressor_api_key": "ya29.old-token",
                    "base_url": self.base_url,
                    "client_kwargs": {"api_key": "ya29.old-token", "base_url": self.base_url},
                }
                self.warm_slot_client = SimpleNamespace(
                    api_key="ya29.old-token",
                    _client=SimpleNamespace(is_closed=False),
                )
                slot = self._request_slot(_OPENAI_SLOT)
                slot["client"] = self.warm_slot_client
                slot["key"] = {
                    "api_key": "ya29.old-token",
                    "base_url": "https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/p/locations/europe-west2/endpoints/openapi",
                }

        ag = _PoolDummy()
        orig_primary = ag.client
        assert ag._adopt_openai_credentials(
            "ya29.new-token",
            ag.base_url,
            reason="vertex_pre_expiry_refresh",
        ) is True
        # Primary client and _OPENAI_SLOT client kept the same object identity and updated api_key in-place
        assert ag.client is orig_primary
        assert ag.client.api_key == "ya29.new-token"
        assert ag.warm_slot_client.api_key == "ya29.new-token"
        reused, stale = ag._checkout_request_slot(
            _OPENAI_SLOT,
            {
                "api_key": "ya29.new-token",
                "base_url": "https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/p/locations/europe-west2/endpoints/openapi",
            },
        )
        assert reused is ag.warm_slot_client
        assert stale is None


class TestPostCompactionLoopGuardP02:
    def test_defers_180k_unanchored_rough_estimate_immediately_after_compaction(self):
        cc = SimpleNamespace(
            context_length=1_048_576,
            threshold_tokens=157_286,
            awaiting_real_usage_after_compression=True,
            last_compression_rough_tokens=95_000,
            should_defer_preflight_to_real_usage=lambda est: True,
        )
        agent = SimpleNamespace(
            context_compressor=cc,
            _request_pressure_anchored=False,
        )
        # Before P0-2 fix, last_compression_rough_tokens > 0 short-circuited deferral at >=180k
        assert should_defer_preflight_to_real_usage(agent, 210_000) is True

        # When NOT awaiting post-compression real usage, the 180k pricing-cliff cap still fires
        cc.awaiting_real_usage_after_compression = False
        assert should_defer_preflight_to_real_usage(agent, 210_000) is False

    def test_compressor_enforces_post_compaction_step_floor_and_shallow_strike(self):
        cc = ContextCompressor(
            model="google/gemini-4-argon",
            threshold_percent=0.15,
            quiet_mode=True,
        )
        cc.context_length = 1_048_576
        cc.threshold_tokens = 157_286

        # Compaction #1 lands at 152,000 real tokens (barely below 157,286 due to high system+tools floor)
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": 152_000})
        assert cc._last_post_compaction_prompt_tokens == 152_000
        assert cc._ineffective_compression_count == 0

        # Effective threshold is now 152,000 + 18,874 = 170,874 (capped at 180,000)
        assert cc._effective_compression_threshold() == 170_874
        # A 7,000-token tool call (159,000 real tokens) must NOT immediately re-trigger compaction
        assert cc.should_compress(159_000) is False
        assert cc.should_compress(170_000) is False
        # Once growth crosses the step floor (171,000), compaction is allowed again
        assert cc.should_compress(171_000) is True

        # Compaction #2 is another shallow compaction landing at 153,000 (<5% headroom below 157,286)
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": 153_000})
        assert cc._ineffective_compression_count == 1

        # Compaction #3 is also shallow (154,000) -> 2nd strike trips anti-thrash breaker
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": 154_000})
        assert cc._ineffective_compression_count == 2
        should_now, reason = cc.should_compress_info(178_000)
        assert should_now is False
        assert reason == "ineffective"

    def test_warm_session_class_rebind_has_class_level_compaction_defaults(self, monkeypatch):
        assert hasattr(ContextCompressor, "_last_post_compaction_prompt_tokens")
        assert ContextCompressor._last_post_compaction_prompt_tokens == 0

        cc = ContextCompressor(
            model="google/gemini-4-argon",
            threshold_percent=0.15,
            quiet_mode=True,
        )
        cc.context_length = 1_048_576
        cc.threshold_tokens = 157_286

        # Simulate a warm session constructed before P0-2 whose __class__ was rebound across importlib.reload
        inst_dict: dict = vars(cc)
        inst_dict.pop("_last_post_compaction_prompt_tokens", None)
        inst_dict.pop("_verify_compaction_cleared_threshold", None)
        assert "_last_post_compaction_prompt_tokens" not in inst_dict

        assert cc._effective_compression_threshold() == 157_286
        assert cc.should_compress(120_000) is False
        cc.update_from_response({"prompt_tokens": 120_000})

        # Negative polarity: without the class-level default, a warm rebound instance raises AttributeError
        inst_dict.pop("_last_post_compaction_prompt_tokens", None)
        monkeypatch.delattr(ContextCompressor, "_last_post_compaction_prompt_tokens")
        with pytest.raises(AttributeError, match="_last_post_compaction_prompt_tokens"):
            cc._effective_compression_threshold()


class TestWave5LongTermCostArchitectureFixes:
    """Tests for Wave 5 fixes:
    - Fix 1: Unrecoverable auth-wall halt on failure-tolerant tools & progress-sensing IterationBudget extension
    - Fix 2: System-prompt byte-identity preservation across _assemble_head & 12 KB execute_code stdout spill
    - Fix 3: Session-sticky europe-west2 affinity with bounded wait & non-evicting _OPENAI_GLOBAL_FALLBACK_SLOT
    - Fix 4: Cross-epoch skill_view structural digest on 3rd+ full load across compactions
    """

    def test_fix1_auth_wall_halts_terminal_on_third_failure_and_progress_extends_budget(self):
        from agent.iteration_budget import IterationBudget
        from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

        ctrl = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
        ctrl.reset_for_turn()

        # Distinct terminal commands failing with RAPT/invalid_grant auth walls:
        d1 = ctrl.after_call("terminal", {"command": "bq query --q1"}, '{"exit_code": 1, "error": "invalid_grant: reauth_required"}')
        assert d1.action == "allow"
        assert ctrl.is_making_productive_progress() is False

        d2 = ctrl.after_call("terminal", {"command": "bq query --q2"}, '{"exit_code": 1, "error": "invalid_grant: rapt_required"}')
        assert d2.action == "warn"
        assert d2.code == "auth_wall_failure_warning"

        d3 = ctrl.after_call("terminal", {"command": "bq query --q3"}, '{"exit_code": 1, "error": "HTTP 401: Request had invalid authentication credentials"}')
        assert d3.action == "halt"
        assert d3.code == "auth_wall_failure_halt"

        # Productive progress on a normal turn extends a low-remaining real budget (initial >= 20),
        # while a small test budget (initial < 20) or non-progress turn does NOT extend.
        ctrl.reset_for_turn()
        ok = ctrl.after_call("read_file", {"path": "/tmp/a.py"}, '{"content": "x = 1\\n"}')
        assert ok.action == "allow"
        assert ctrl.is_making_productive_progress() is True

        # A one-off non-repeated tool failure (e.g. a probe with a typo or red pytest run)
        # marks that step non-productive, but once the agent runs a succeeding tool call
        # with fixed args (without re-running the broken command string), progress recovers
        # instead of staying permanently poisoned by _exact_failure_counts.
        one_off_fail = ctrl.after_call(
            "terminal", {"command": "sqlite3 db 'SELECT * FROM wrong_table'"}, '{"exit_code": 1, "error": "no such table"}'
        )
        assert one_off_fail.action == "allow"
        assert ctrl.is_making_productive_progress() is False
        fixed_ok = ctrl.after_call(
            "terminal", {"command": "sqlite3 db 'SELECT * FROM active_leases'"}, '{"exit_code": 0, "output": "ok"}', failed=False
        )
        assert fixed_ok.action == "allow"
        assert ctrl.is_making_productive_progress() is True

        # Conversely, an unrecovered repeated exact failure (count >= exact_failure_warn_after)
        # blocks progress until either that tool succeeds or a file mutation lands.
        ctrl.after_call("terminal", {"command": "bad_cmd"}, '{"exit_code": 1, "error": "fail"}')
        rep_warn = ctrl.after_call("terminal", {"command": "bad_cmd"}, '{"exit_code": 1, "error": "fail"}')
        assert rep_warn.code == "repeated_exact_failure_warning"
        assert ctrl.is_making_productive_progress() is False
        ctrl.after_call("read_file", {"path": "/tmp/other.py"}, '{"content": "y = 2\\n"}')
        assert ctrl.is_making_productive_progress() is False
        ctrl.after_call("patch", {"path": "/tmp/other.py"}, '{"success": true}', failed=False)
        assert ctrl.is_making_productive_progress() is True

        budget = IterationBudget(35)
        for _ in range(31):
            assert budget.consume() is True
        assert budget.remaining == 4
        assert budget.extend_on_progress(1) is True
        assert budget.max_total == 36
        assert budget.remaining == 5

        small_budget = IterationBudget(5)
        for _ in range(4):
            assert small_budget.consume() is True
        assert small_budget.extend_on_progress(1) is False
        assert small_budget.max_total == 5

    def test_fix2_assemble_head_preserves_system_prompt_byte_identity_and_12kb_spill(self, monkeypatch):
        from tools.code_execution_tool import _effective_max_stdout_bytes, _truncate_stdout_text

        cc = ContextCompressor(model="google/gemini-4-argon", quiet_mode=True)
        sys_text = "You are Hermes Agent.\n" + ("A" * 4096)
        msgs = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": "hello"},
        ]
        head = cc._assemble_head(msgs, 1)
        assert len(head) == 1
        assert head[0]["content"] == sys_text
        assert cc._COMPRESSION_NOTE not in head[0]["content"]

        # When code_execution.max_stdout_bytes is configured to 12288, _truncate_stdout_text caps at 12288 bytes
        monkeypatch.setattr("tools.code_execution_tool._load_config", lambda: {"max_stdout_bytes": 12288})
        assert _effective_max_stdout_bytes() == 12288
        big_out = "0123456789abcdef" * 2000  # 32,000 bytes
        truncated, meta = _truncate_stdout_text(big_out)
        assert len(truncated) > 12288
        assert meta["stdout_truncated"] is True
        assert meta["stdout_bytes_captured"] == 12288
        assert meta["stdout_bytes_total"] == 32000

    def test_fix3_sticky_europe_west2_waits_for_slot_and_global_fallback_preserves_regional_client(self):
        import threading
        from agent.client_lifecycle import ClientLifecycleMixin, _OPENAI_GLOBAL_FALLBACK_SLOT, _OPENAI_SLOT
        from agent.vertex_adapter import get_session_vertex_region_affinity

        # Session 'sess-sticky-1' acquires slot #1 on europe-west2
        r1, acq1 = acquire_vertex_hybrid_region(
            "google/gemini-4-argon",
            configured_region="global",
            interactive=True,
            session_id="sess-sticky-1",
        )
        assert (r1, acq1) == (EUROPE_WEST2_VERTEX_REGION, True)
        assert get_session_vertex_region_affinity("sess-sticky-1") == EUROPE_WEST2_VERTEX_REGION
        c1 = SimpleNamespace(_vertex_regional_slot_held=True)

        # Fill the remaining 4 slots
        other_clients = []
        for _ in range(EUROPE_WEST2_ARGON_MAX_CONCURRENCY - 1):
            r, acq = acquire_vertex_hybrid_region("google/gemini-4-argon", configured_region="global", interactive=True)
            assert (r, acq) == (EUROPE_WEST2_VERTEX_REGION, True)
            other_clients.append(SimpleNamespace(_vertex_regional_slot_held=True))

        # Release c1's slot from Turn 1, then saturate all 5 slots briefly for 40ms
        assert release_vertex_hybrid_slot(c1) is True
        temp_slot_r, temp_slot_acq = acquire_vertex_hybrid_region("google/gemini-4-argon", configured_region="global", interactive=True)
        assert (temp_slot_r, temp_slot_acq) == (EUROPE_WEST2_VERTEX_REGION, True)
        temp_client = SimpleNamespace(_vertex_regional_slot_held=True)

        timer = threading.Timer(0.04, lambda: release_vertex_hybrid_slot(temp_client))
        timer.start()
        try:
            # Turn 2 of 'sess-sticky-1' waits up to sticky_wait_seconds and stays on europe-west2!
            r2, acq2 = acquire_vertex_hybrid_region(
                "google/gemini-4-argon",
                configured_region="global",
                interactive=True,
                session_id="sess-sticky-1",
                sticky_wait_seconds=0.5,
            )
            assert (r2, acq2) == (EUROPE_WEST2_VERTEX_REGION, True)
            c2 = SimpleNamespace(_vertex_regional_slot_held=True)
        finally:
            timer.cancel()

        assert release_vertex_hybrid_slot(c2) is True
        for oc in other_clients:
            assert release_vertex_hybrid_slot(oc) is True

        # Verify _OPENAI_GLOBAL_FALLBACK_SLOT prevents evicting a warm europe-west2 client in _OPENAI_SLOT
        class _FallbackDummy(ClientLifecycleMixin):
            def __init__(self):
                self.provider = "vertex"
                self.model = "google/gemini-3.8-flash"
                self.api_key = "ya29.tok"
                self.base_url = "https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/europe-west2/endpoints/openapi"
                self._client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
                self._openai_client_mutex = threading.RLock()
                self.client = SimpleNamespace(
                    api_key="ya29.tok",
                    _client=SimpleNamespace(is_closed=False),
                )
                self.regional_client = SimpleNamespace(
                    api_key="ya29.tok",
                    _client=SimpleNamespace(is_closed=False),
                )
                slot = self._request_slot(_OPENAI_SLOT)
                slot["client"] = self.regional_client
                slot["key"] = {"api_key": "ya29.tok", "base_url": self.base_url, "max_retries": 0}

            @staticmethod
            def _api_kwargs_have_image_parts(api_kwargs):
                return False

            def _create_openai_client(self, kw, *, reason, shared=False):
                return SimpleNamespace(api_key=kw.get("api_key"), base_url=kw.get("base_url"), _client=SimpleNamespace(is_closed=False))

            def _close_openai_client(self, client, *, reason, shared=False):
                if client is not None:
                    client._closed_reason = reason

        dummy = _FallbackDummy()
        global_client = dummy._create_request_openai_client(api_kwargs={"model": "google/gemini-3.8-flash"}, reason="test_flash")
        assert getattr(global_client, "_request_slot_name", None) == _OPENAI_GLOBAL_FALLBACK_SLOT
        # Regional client in _OPENAI_SLOT was NOT evicted or closed!
        assert dummy._request_slot(_OPENAI_SLOT)["client"] is dummy.regional_client
        assert not hasattr(dummy.regional_client, "_closed_reason")
        dummy._close_request_openai_client(global_client, reason="request_complete")
        assert dummy._request_slot(_OPENAI_GLOBAL_FALLBACK_SLOT)["client"] is global_client

    def test_fix4_cross_epoch_skill_digest_serves_compact_summary_on_third_epoch_load(self, tmp_path, monkeypatch):
        from tools.skills_tool import _skill_view_with_bump

        body = (
            "# Cost Audit Skill\n"
            "## Phase 1: Query BigQuery\n"
            "Run `bq query --use_legacy_sql=false` to inspect SKUs.\n"
            + ("Detailed documentation line.\n" * 600)
        )
        _setup_skill(tmp_path, monkeypatch, "cost-audit-test", body=body)
        tid = "sess-epoch-digest-test"

        # Epoch 1: Full load #1
        r1 = json.loads(_skill_view_with_bump({"name": "cost-audit-test"}, task_id=tid))
        assert r1["success"] is True
        assert "dedup" not in r1
        assert len(r1["content"]) > 10_000

        # Compaction #1 prunes the skill; Epoch 2 reload #2 still returns full content (parity with rc01/rc02)
        reset_skill_view_dedup(tid, surviving_messages=[])
        r2 = json.loads(_skill_view_with_bump({"name": "cost-audit-test"}, task_id=tid))
        assert r2["success"] is True
        assert "dedup" not in r2
        assert len(r2["content"]) > 10_000

        # Compaction #2 prunes the skill again; Epoch 3 reload #3 returns compact cross_epoch_digest (<2.5 KB)
        reset_skill_view_dedup(tid, surviving_messages=[])
        r3_raw = _skill_view_with_bump({"name": "cost-audit-test"}, task_id=tid)
        r3 = json.loads(r3_raw)
        assert r3["success"] is True
        assert r3["dedup"] is True
        assert r3["status"] == "cross_epoch_digest"
        assert r3["epoch_load_count"] == 3
        assert "Headings: # Cost Audit Skill | ## Phase 1: Query BigQuery" in r3["content_digest"]
        assert len(r3_raw) < 2500

    def test_fix5_pre_reload_guardrail_and_budget_instances_survive_mid_turn_hot_reload(self):
        from agent.iteration_budget import IterationBudget
        from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

        # Simulate a pre-reload ToolCallGuardrailController instance whose reset_for_turn()
        # ran before _turn_productive_calls / _auth_wall_failure_count existed and whose
        # self.config was created from an older ToolCallGuardrailConfig class without
        # auth_wall_warn_after / auth_wall_halt_after.
        ctrl = ToolCallGuardrailController()
        ctrl.__dict__.pop("_turn_productive_calls", None)
        ctrl.__dict__.pop("_auth_wall_failure_count", None)
        base_cfg = ToolCallGuardrailConfig(hard_stop_enabled=True)
        legacy_cfg_dict = {
            k: getattr(base_cfg, k)
            for k in base_cfg.__dataclass_fields__
            if k not in ("auth_wall_warn_after", "auth_wall_halt_after")
        }
        setattr(ctrl, "config", SimpleNamespace(**legacy_cfg_dict))  # noqa: B010

        assert ctrl.is_making_productive_progress() is False
        d_ok = ctrl.after_call("terminal", {"command": "echo ok"}, "ok", failed=False)
        assert d_ok.action == "allow"
        assert ctrl.is_making_productive_progress() is True

        # Auth-wall failures on the legacy config stub must fall back cleanly to 2 / 3 without AttributeError
        auth_err = "unauth" + "enticated"
        d_auth1 = ctrl.after_call("terminal", {"command": "gcloud1"}, auth_err, failed=True)
        assert d_auth1.action == "allow"
        assert ctrl.is_making_productive_progress() is False
        d_auth2 = ctrl.after_call("terminal", {"command": "gcloud2"}, auth_err, failed=True)
        assert d_auth2.action == "warn"
        assert d_auth2.code == "auth_wall_failure_warning"
        d_auth3 = ctrl.after_call("terminal", {"command": "gcloud3"}, auth_err, failed=True)
        assert d_auth3.action == "halt"
        assert d_auth3.code == "auth_wall_failure_halt"

        # Simulate a pre-reload IterationBudget instance that only has _initial_max_total
        budget = IterationBudget(25)
        budget.__dict__.pop("initial_max_total", None)
        budget.__dict__.pop("_extensions_granted", None)
        budget._initial_max_total = 25
        budget._used = 21  # within low_watermark=5

        assert budget.extend_on_progress(1) is True
        assert budget.max_total == 26
        assert budget.initial_max_total == 25
        assert budget._extensions_granted == 1

    def test_append_guardrail_observation_isolates_after_call_exceptions(self):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._stall_guards_enabled = lambda: False

        class _BrokenGuardrails:
            halt_decision = None

            def after_call(self, *args, **kwargs):
                raise AttributeError("simulated hot-reload attribute gap")

        setattr(agent, "_tool_guardrails", _BrokenGuardrails())  # noqa: B010
        out = agent._append_guardrail_observation(
            "write_file",
            {"path": "/tmp/example.txt"},
            '{"verified": true}',
            failed=False,
            tool_call_id="call_30",
        )
        assert out == '{"verified": true}'




