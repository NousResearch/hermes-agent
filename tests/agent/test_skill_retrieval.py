"""
Tests for agent/skill_retrieval.py  (issue #34823).

Run with:
    pytest tests/agent/test_skill_retrieval.py -v

All tests are pure-Python, stdlib-only, and require no network access or
configured embedding models. The EmbeddingIndex dense layer is tested via
a monkeypatched _embed() that returns deterministic vectors.
"""
from __future__ import annotations

import json
import math
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Unit tests: BM25Scorer
# ---------------------------------------------------------------------------

from agent.skill_retrieval import BM25Scorer, _rrf, _scores_to_ranks, _dict_scores_to_ranks


class TestBM25Scorer:
    CORPUS = [
        "docker containerize image build run",
        "python testing pytest unit integration",
        "git version control commit branch merge",
        "aws cloud deploy lambda s3 ec2",
    ]

    def test_exact_match_scores_highest(self):
        bm25 = BM25Scorer(self.CORPUS)
        scores = bm25.score("docker build")
        assert scores[0] == max(scores), "docker doc should score highest for 'docker build'"

    def test_empty_query_returns_zeros(self):
        bm25 = BM25Scorer(self.CORPUS)
        scores = bm25.score("")
        assert all(s == 0.0 for s in scores)

    def test_scores_parallel_to_corpus(self):
        bm25 = BM25Scorer(self.CORPUS)
        scores = bm25.score("python")
        assert len(scores) == len(self.CORPUS)

    def test_single_doc_corpus(self):
        bm25 = BM25Scorer(["only one document here"])
        scores = bm25.score("document")
        assert len(scores) == 1
        assert scores[0] > 0

    def test_all_zeros_for_no_overlap(self):
        bm25 = BM25Scorer(["apple orange banana"])
        scores = bm25.score("photosynthesis")
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# Unit tests: RRF fusion helpers
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_single_layer(self):
        names = ["a", "b", "c"]
        ranks = {"a": 0, "b": 1, "c": 2}
        fused = _rrf([ranks], names, k=60)
        # Higher rank → higher fused score
        assert fused["a"] > fused["b"] > fused["c"]

    def test_rrf_two_agreeing_layers(self):
        names = ["a", "b", "c"]
        r1 = {"a": 0, "b": 1, "c": 2}
        r2 = {"a": 0, "b": 1, "c": 2}
        fused = _rrf([r1, r2], names, k=60)
        # Agreement amplifies the ordering
        assert fused["a"] > fused["b"] > fused["c"]

    def test_rrf_disagreeing_layers(self):
        names = ["a", "b"]
        r1 = {"a": 0, "b": 1}   # BM25 prefers a
        r2 = {"a": 1, "b": 0}   # dense prefers b
        fused = _rrf([r1, r2], names, k=60)
        # With exactly inverted ranks of size 2, scores are equal
        assert math.isclose(fused["a"], fused["b"], rel_tol=1e-9)

    def test_rrf_empty_layer_ignored(self):
        names = ["a", "b"]
        r1 = {"a": 0, "b": 1}
        fused_with_empty = _rrf([r1, {}], names, k=60)
        fused_single     = _rrf([r1],     names, k=60)
        assert math.isclose(fused_with_empty["a"], fused_single["a"])

    def test_scores_to_ranks(self):
        names = ["x", "y", "z"]
        scores = [0.5, 0.9, 0.1]
        ranks = _scores_to_ranks(scores, names)
        assert ranks["y"] == 0   # highest score → rank 0
        assert ranks["x"] == 1
        assert ranks["z"] == 2

    def test_dict_scores_to_ranks_missing_key(self):
        names = ["a", "b", "c"]
        score_dict = {"a": 0.9, "b": 0.5}  # "c" missing
        ranks = _dict_scores_to_ranks(score_dict, names)
        # "c" absent → gets last rank
        assert ranks["c"] == len(names) - 1


# ---------------------------------------------------------------------------
# Unit tests: retrieve_skills
# ---------------------------------------------------------------------------

from agent.skill_retrieval import retrieve_skills


SAMPLE_SKILLS = [
    {"name": "docker", "description": "Build and run Docker containers"},
    {"name": "pytest",  "description": "Python unit testing with pytest"},
    {"name": "git",     "description": "Git version control commands"},
    {"name": "aws-s3",  "description": "Manage S3 buckets and objects"},
    {"name": "vscode",  "description": "VS Code editor configuration"},
    {"name": "nginx",   "description": "Configure Nginx web server"},
]


class TestRetrieveSkills:
    def test_returns_set_of_names(self):
        result = retrieve_skills("containerize my app", SAMPLE_SKILLS, top_k=2)
        assert isinstance(result, set)
        assert all(isinstance(n, str) for n in result)

    def test_respects_top_k(self):
        result = retrieve_skills("anything", SAMPLE_SKILLS, top_k=3)
        assert len(result) <= 3

    def test_short_circuit_when_skills_le_top_k(self):
        few = SAMPLE_SKILLS[:3]
        result = retrieve_skills("docker", few, top_k=5)
        # Returns all names unchanged when count ≤ top_k
        assert result == {"docker", "pytest", "git"}

    def test_empty_skills_returns_empty(self):
        result = retrieve_skills("docker", [], top_k=5)
        assert result == set()

    def test_docker_query_ranks_docker_highly(self):
        result = retrieve_skills("docker build image", SAMPLE_SKILLS, top_k=2)
        # BM25 should rank docker first for a docker-specific query
        assert "docker" in result

    def test_no_crash_with_missing_description(self):
        skills = [{"name": "no-desc"}, {"name": "has-desc", "description": "something"}]
        result = retrieve_skills("something", skills, top_k=1)
        assert isinstance(result, set)

    def test_embedding_cfg_without_model_uses_bm25_only(self):
        """embedding_cfg with no model → dense layer skipped, BM25 still runs."""
        result = retrieve_skills(
            "docker",
            SAMPLE_SKILLS,
            top_k=2,
            embedding_cfg={"enabled": True, "embedding_model": ""},
        )
        assert "docker" in result


# ---------------------------------------------------------------------------
# Unit tests: EmbeddingIndex
# ---------------------------------------------------------------------------

from agent.skill_retrieval import EmbeddingIndex


def _make_unit_vec(seed: int, dim: int = 8) -> list[float]:
    """Reproducible unit vector for testing."""
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class TestEmbeddingIndex:
    def _make_index(self) -> EmbeddingIndex:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return EmbeddingIndex(db_path=Path(tmp.name))

    def _patch_embed(self, idx: EmbeddingIndex, seed: int):
        """Monkeypatch _embed to return a deterministic vector."""
        def fake_embed(text, cfg):
            return _make_unit_vec(seed)
        idx._embed = staticmethod(fake_embed)

    def test_is_warm_false_on_empty_db(self):
        idx = self._make_index()
        assert idx.is_warm() is False

    def test_upsert_makes_warm(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=1)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "skill-a", "description": "desc a"}], embedding_cfg=cfg)
        assert idx.is_warm() is True

    def test_query_returns_scores(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=42)
        cfg = {"embedding_model": "test-model"}
        skills = [
            {"name": "alpha", "description": "first skill"},
            {"name": "beta",  "description": "second skill"},
        ]
        idx.upsert(skills, embedding_cfg=cfg)
        scores = idx.query("anything", ["alpha", "beta"], embedding_cfg=cfg)
        assert "alpha" in scores
        assert "beta" in scores
        assert all(-1.0 <= v <= 1.0 for v in scores.values())

    def test_invalidate_specific_name(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=7)
        cfg = {"embedding_model": "test-model"}
        skills = [
            {"name": "keep-me",   "description": "stays"},
            {"name": "delete-me", "description": "goes"},
        ]
        idx.upsert(skills, embedding_cfg=cfg)
        assert idx.is_warm()

        idx.invalidate(["delete-me"])
        scores = idx.query("x", ["keep-me", "delete-me"], embedding_cfg=cfg)
        assert "keep-me" in scores
        assert "delete-me" not in scores

    def test_invalidate_all(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=3)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "x", "description": "y"}], embedding_cfg=cfg)
        idx.invalidate()
        assert idx.is_warm() is False

    def test_stale_detection_skips_unchanged(self):
        """Calling upsert twice with the same content does not re-embed."""
        idx = self._make_index()
        call_count = {"n": 0}

        def counting_embed(text, cfg):
            call_count["n"] += 1
            return _make_unit_vec(call_count["n"])

        idx._embed = staticmethod(counting_embed)
        cfg = {"embedding_model": "test-model"}
        skill = [{"name": "stable", "description": "unchanged description"}]
        idx.upsert(skill, embedding_cfg=cfg)
        first_count = call_count["n"]
        idx.upsert(skill, embedding_cfg=cfg)
        # Hash matched → no second embed call
        assert call_count["n"] == first_count

    def test_stale_detection_re_embeds_on_change(self):
        idx = self._make_index()
        call_count = {"n": 0}

        def counting_embed(text, cfg):
            call_count["n"] += 1
            return _make_unit_vec(call_count["n"])

        idx._embed = staticmethod(counting_embed)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "s", "description": "v1"}], embedding_cfg=cfg)
        first_count = call_count["n"]
        # Change description → hash mismatch → should re-embed
        idx.upsert([{"name": "s", "description": "v2 changed"}], embedding_cfg=cfg)
        assert call_count["n"] > first_count

    def test_build_async_completes(self):
        """build_async() runs in background and upserts without deadlock."""
        idx = self._make_index()
        self._patch_embed(idx, seed=99)
        cfg = {"embedding_model": "test-model"}
        skills = [{"name": f"skill-{i}", "description": f"desc {i}"} for i in range(5)]
        idx.build_async(skills, embedding_cfg=cfg)

        # Wait up to 5 s for the daemon thread
        deadline = threading.Event()
        threading.Timer(5.0, deadline.set).start()
        while not idx.is_warm() and not deadline.is_set():
            pass
        assert idx.is_warm(), "build_async should complete within 5 seconds"

    def test_query_returns_empty_on_failed_embed(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=11)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "s", "description": "d"}], embedding_cfg=cfg)

        # Make query-time embed fail
        idx._embed = staticmethod(lambda text, cfg: None)
        scores = idx.query("anything", ["s"], embedding_cfg=cfg)
        assert scores == {}


# ---------------------------------------------------------------------------
# Integration: prompt_builder renders two-tier output
# ---------------------------------------------------------------------------
# NOTE: These tests exercise the patched agent/prompt_builder.py and require
# a full Hermes install (run_agent, hermes_state, gateway, etc.) to be present.
# When running in an isolated contributor checkout, they are skipped gracefully
# via the _import_prompt_builder() helper below — matching CI behaviour where
# scripts/run_tests.sh provides the full hermetic environment.
#
# The unit tests above (TestBM25Scorer, TestRRF, TestRetrieveSkills,
# TestEmbeddingIndex) are stdlib-only and always run.


def _import_prompt_builder():
    """Try to import the patched prompt_builder; return None if unavailable."""
    try:
        import agent.prompt_builder as pb  # noqa: F401
        return pb
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "agent.prompt_builder not importable (integration tests will be skipped): %s: %s",
            type(exc).__name__, exc,
        )
        return None


import types

class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


class _FakeAgent:
    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        self._cached_system_prompt = None
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0
        self._session_messages = []
        self._pending_cli_user_message = None
        self._session_persist_lock = threading.RLock()

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, history):
        self._persist_calls += 1
        self._session_messages = list(messages)


def _restore_system_prompt_fake(agent, system_message, conversation_history, build_fn):
    if getattr(agent, "_cached_system_prompt", None) is None:
        agent._cached_system_prompt = build_fn()


class TestPromptBuilderIntegration:
    """Verify that semantic search behaves correctly with system prompts and context building.

    Skipped automatically when running outside a full Hermes install.
    """

    def test_semantic_search_system_prompt_names_only(self):
        pb = _import_prompt_builder()
        if pb is None:
            pytest.skip("agent.prompt_builder not available outside full Hermes install")
        
        build_skills_system_prompt = pb.build_skills_system_prompt
        clear_fn = getattr(pb, "clear_skills_system_prompt_cache", None)
        if clear_fn:
            clear_fn(clear_snapshot=False)

        mock_skills = {
            "general": [("hermes-agent", "Hermes Agent config skill"), ("test-skill", "Testing skill")]
        }

        # Force semantic search enabled
        with patch("agent.prompt_builder._gather_skills_by_category", return_value=(mock_skills, {})):
            with patch("hermes_cli.config.load_config", return_value={
                "skills": {
                    "semantic_search": {
                        "enabled": True,
                        "top_k": 2,
                        "embedding_model": "",
                    }
                }
            }):
                prompt = build_skills_system_prompt()
                # In names-only mode, descriptions are omitted and category headers end in "[names only]"
                assert "[names only]" in prompt

    def test_system_prompt_byte_identity_invariant(self):
        """Assert that two system prompts built in the same session with different simulated user queries are byte-for-byte identical and the prompt is not rebuilt."""
        pb = _import_prompt_builder()
        if pb is None:
            pytest.skip("agent.prompt_builder not available outside full Hermes install")

        from agent.turn_context import build_turn_context
        # Create a FakeAgent
        agent = _FakeAgent()
        agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
        agent._cached_system_prompt = None

        mock_skills = {
            "general": [
                ("hermes-agent", "Hermes Agent config skill"),
                ("test-skill", "Testing skill description"),
            ]
        }

        # Mock retrieve_skills to select different skills based on query
        def mock_retrieve_skills(query, skills, top_k, embedding_cfg):
            if "tests" in query:
                return {"test-skill"}
            else:
                return {"hermes-agent"}

        with patch("agent.prompt_builder._gather_skills_by_category", return_value=(mock_skills, {})):
            with patch("agent.skill_retrieval.retrieve_skills", side_effect=mock_retrieve_skills):
                with patch("hermes_cli.config.load_config", return_value={
                    "skills": {
                        "semantic_search": {
                            "enabled": True,
                            "top_k": 1,
                            "embedding_model": "",
                        }
                    }
                }):
                    # Mock build_skills_system_prompt to track call count
                    build_mock = patch.object(pb, "build_skills_system_prompt", wraps=pb.build_skills_system_prompt).start()
                    try:
                        # Turn 1: "help me write tests"
                        ctx_1 = build_turn_context(
                            agent=agent,
                            user_message="help me write tests",
                            system_message=None,
                            conversation_history=[],
                            task_id=None,
                            stream_callback=None,
                            persist_user_message=None,
                            restore_or_build_system_prompt=lambda _a, _s, _h, **kwargs: _restore_system_prompt_fake(_a, _s, _h, pb.build_skills_system_prompt),
                            install_safe_stdio=lambda: None,
                            sanitize_surrogates=lambda s: s,
                            summarize_user_message_for_log=lambda s: s,
                            set_session_context=lambda _sid: None,
                            set_current_write_origin=lambda _o: None,
                            ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
                        )
                        prompt_1 = agent._cached_system_prompt
                        retrieved_1 = ctx_1.retrieved_skills_context
                        
                        assert build_mock.call_count == 1
    
                        # Turn 2: "deploy to production"
                        # Do NOT set _cached_system_prompt to None, let it reuse the cached prompt!
                        ctx_2 = build_turn_context(
                            agent=agent,
                            user_message="deploy to production",
                            system_message=None,
                            conversation_history=ctx_1.messages,
                            task_id=None,
                            stream_callback=None,
                            persist_user_message=None,
                            restore_or_build_system_prompt=lambda _a, _s, _h, **kwargs: _restore_system_prompt_fake(_a, _s, _h, pb.build_skills_system_prompt),
                            install_safe_stdio=lambda: None,
                            sanitize_surrogates=lambda s: s,
                            summarize_user_message_for_log=lambda s: s,
                            set_session_context=lambda _sid: None,
                            set_current_write_origin=lambda _o: None,
                            ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
                        )
                        prompt_2 = agent._cached_system_prompt
                        retrieved_2 = ctx_2.retrieved_skills_context
                        
                        # Verify the prompt builder was NOT called again! (Cache hit)
                        assert build_mock.call_count == 1
    
                        # System prompt MUST be completely byte-identical because it's the exact same string
                        assert prompt_1 is prompt_2
    
                        # Retrieved skills context MUST differ based on queries
                        assert "test-skill" in retrieved_1
                        assert "hermes-agent" not in retrieved_1
    
                        assert "hermes-agent" in retrieved_2
                        assert "test-skill" not in retrieved_2
                    finally:
                        patch.stopall()

    def test_skill_retrieval_e2e(self, tmp_path):
        """End-to-end test simulating a real conversation turn with semantic search enabled.
        
        Asserts that:
        (a) The persisted/cached system prompt contains names-only skill entries (no descriptions).
        (b) The retrieved_skills_context injected into the turn contains full descriptions.
        (c) The persisted `messages` list does NOT contain the retrieved skill descriptions.
        """
        pb = _import_prompt_builder()
        if pb is None:
            pytest.skip("agent.prompt_builder not available outside full Hermes install")

        from agent.turn_context import build_turn_context
        from agent.conversation_loop import compose_user_api_content
        import json
        import os

        # Setup temporary HERMES_HOME and mock skills
        os.environ["HERMES_HOME"] = str(tmp_path)
        
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "myskill").mkdir()
        
        skill_content = "---\nname: my-special-skill\ndescription: VERY UNIQUE LONG DESCRIPTION THAT SHOULD NOT BE IN SYSTEM PROMPT\n---\nBODY"
        (skills_dir / "myskill" / "SKILL.md").write_text(skill_content, encoding="utf-8")
        
        agent = _FakeAgent()
        agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage", "my-special-skill"}
        agent._cached_system_prompt = None
        
        def mock_retrieve_skills(query, skills, top_k, embedding_cfg):
            # Always return our special skill
            return {"my-special-skill"}

        with patch("agent.skill_retrieval.retrieve_skills", side_effect=mock_retrieve_skills):
            with patch("hermes_cli.config.load_config", return_value={
                "skills": {
                    "semantic_search": {
                        "enabled": True,
                        "top_k": 1,
                        "embedding_model": "",
                    }
                }
            }):
                # Run the turn context builder
                ctx = build_turn_context(
                    agent=agent,
                    user_message="do the special skill thing",
                    system_message=None,
                    conversation_history=[],
                    task_id=None,
                    stream_callback=None,
                    persist_user_message=None,
                    restore_or_build_system_prompt=lambda _a, _s, _h, **kwargs: _restore_system_prompt_fake(_a, _s, _h, pb.build_skills_system_prompt),
                    install_safe_stdio=lambda: None,
                    sanitize_surrogates=lambda s: s,
                    summarize_user_message_for_log=lambda s: s,
                    set_session_context=lambda _sid: None,
                    set_current_write_origin=lambda _o: None,
                    ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
                )
                
                # Verify (a) System prompt contains name only, NOT description
                sys_prompt = agent._cached_system_prompt
                assert "my-special-skill" in sys_prompt
                assert "VERY UNIQUE LONG DESCRIPTION THAT SHOULD NOT BE IN SYSTEM PROMPT" not in sys_prompt
                
                # Verify (b) retrieved_skills_context contains full description
                retrieved_ctx = ctx.retrieved_skills_context
                assert "my-special-skill" in retrieved_ctx
                assert "VERY UNIQUE LONG DESCRIPTION THAT SHOULD NOT BE IN SYSTEM PROMPT" in retrieved_ctx
                
                # Verify (c) persisted messages do NOT contain the retrieved skill descriptions
                # simulate what _run_conversation does:
                final_api_content = compose_user_api_content(
                    turn_context=ctx,
                    system_prompt_override=None,
                    multimodal_chunks=[],
                    user_message="do the special skill thing"
                )
                
                # Persist just stores the raw user text (ctx.persisted_user_message_text)
                # Let's call _persist_session with what conversation_loop would write!
                # conversation_loop does:
                # messages.append({"role": "user", "content": ctx.persisted_user_message_text})
                agent._persist_session([{"role": "user", "content": ctx.persisted_user_message_text}], [])
                
                persisted_text = json.dumps(agent._session_messages)
                assert "VERY UNIQUE LONG DESCRIPTION THAT SHOULD NOT BE IN SYSTEM PROMPT" not in persisted_text
                
                # Final sanity check: make sure the API payload actually HAS the retrieved skills
                assert "VERY UNIQUE LONG DESCRIPTION THAT SHOULD NOT BE IN SYSTEM PROMPT" in str(final_api_content)

    def test_build_retrieved_skills_context(self):
        pb = _import_prompt_builder()
        if pb is None:
            pytest.skip("agent.prompt_builder not available outside full Hermes install")
        
        build_retrieved_skills_context = pb.build_retrieved_skills_context

        # Mock retrieve_skills to return different results for different queries
        def mock_retrieve_skills(query, skills, top_k, embedding_cfg):
            if "setup" in query:
                return {"hermes-agent"}
            else:
                return {"test-skill"}

        mock_skills = {
            "general": [("hermes-agent", "Hermes Agent config skill"), ("test-skill", "Testing skill")]
        }

        with patch("agent.prompt_builder._gather_skills_by_category", return_value=(mock_skills, {})):
            with patch("agent.skill_retrieval.retrieve_skills", side_effect=mock_retrieve_skills):
                with patch("hermes_cli.config.load_config", return_value={
                    "skills": {
                        "semantic_search": {
                            "enabled": True,
                            "top_k": 1,
                            "embedding_model": "",
                        }
                    }
                }):
                    ctx_1 = build_retrieved_skills_context(query_text="help with setup")
                    ctx_2 = build_retrieved_skills_context(query_text="do some tests")
                    
                    assert "hermes-agent" in ctx_1
                    assert "test-skill" not in ctx_1
                    
                    assert "test-skill" in ctx_2
                    assert "hermes-agent" not in ctx_2

    def test_skill_retrieval_e2e(self):
        """Test the E2E turn-level skill retrieval and prompt-cache/persistence boundary."""
        pb = _import_prompt_builder()
        if pb is None:
            pytest.skip("agent.prompt_builder not available outside full Hermes install")

        from agent.turn_context import build_turn_context
        # Create a FakeAgent
        agent = _FakeAgent()
        agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
        agent._cached_system_prompt = None

        mock_skills = {
            "general": [
                ("hermes-agent", "Hermes Agent config skill"),
                ("test-skill", "Testing skill description for retrieval"),
            ]
        }

        # Mock retrieve_skills to select "test-skill"
        def mock_retrieve_skills(query, skills, top_k, embedding_cfg):
            return {"test-skill"}

        with patch("agent.prompt_builder._gather_skills_by_category", return_value=(mock_skills, {})):
            with patch("agent.skill_retrieval.retrieve_skills", side_effect=mock_retrieve_skills):
                with patch("hermes_cli.config.load_config", return_value={
                    "skills": {
                        "semantic_search": {
                            "enabled": True,
                            "top_k": 1,
                            "embedding_model": "",
                        }
                    }
                }):
                    ctx = build_turn_context(
                        agent=agent,
                        user_message="run some tests",
                        system_message=None,
                        conversation_history=[],
                        task_id=None,
                        stream_callback=None,
                        persist_user_message=None,
                        restore_or_build_system_prompt=lambda _a, _s, _h, **kwargs: _restore_system_prompt_fake(_a, _s, _h, pb.build_skills_system_prompt),
                        install_safe_stdio=lambda: None,
                        sanitize_surrogates=lambda s: s,
                        summarize_user_message_for_log=lambda s: s,
                        set_session_context=lambda _sid: None,
                        set_current_write_origin=lambda _o: None,
                        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
                    )

                    # Assert (a): system prompt contains only names-only entries
                    assert "general [names only]: hermes-agent, test-skill" in agent._cached_system_prompt
                    assert "Testing skill description for retrieval" not in agent._cached_system_prompt

                    # Assert (b): retrieved_skills_context / api_content contains full descriptions of top-k retrieved skills
                    assert "test-skill" in ctx.retrieved_skills_context
                    assert "Testing skill description for retrieval" in ctx.retrieved_skills_context
                    assert "Hermes Agent config skill" not in ctx.retrieved_skills_context
                    
                    user_msg = ctx.messages[ctx.current_turn_user_idx]
                    assert "api_content" in user_msg
                    assert "Testing skill description for retrieval" in user_msg["api_content"]

                    # Assert (c): persisted messages list does NOT contain the retrieved skill descriptions
                    clean_msg = agent._session_messages[ctx.current_turn_user_idx]
                    assert clean_msg["content"] == "run some tests"
                    assert "Testing skill description for retrieval" not in clean_msg["content"]
