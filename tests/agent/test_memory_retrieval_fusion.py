from agent.memory_retrieval_fusion import FUSION_POLICY, fuse_memory_retrieval


NOW = "2026-05-23T00:00:00Z"


def test_exact_keyword_recall_beats_unrelated_text():
    result = fuse_memory_retrieval(
        query="graphite dashboard theme",
        candidates=[
            {"id": "unrelated", "content": "Use staging for OpenClaw.", "created_at": "2026-05-22T00:00:00Z"},
            {
                "id": "theme",
                "content": "Han prefers graphite dashboard theme.",
                "created_at": "2026-05-22T00:00:00Z",
            },
        ],
        now=NOW,
    )

    assert result["selected_memories"][0]["id"] == "theme"
    assert result["scores"]["theme"]["keyword_score"] > result["scores"]["unrelated"]["keyword_score"]


def test_project_scope_isolation_rejects_unrelated_project():
    result = fuse_memory_retrieval(
        query="deployment target",
        project_scope="openclaw",
        candidates=[
            {"id": "lovart", "content": "Use production deployment target.", "project_id": "lovart"},
            {"id": "openclaw", "content": "Use staging deployment target.", "project_id": "openclaw"},
        ],
        now=NOW,
    )

    assert result["selected_memories"][0]["id"] == "openclaw"
    rejected = {item["id"]: item for item in result["rejected_memories"]}
    assert rejected["lovart"]["reason"] == "project_scope_mismatch"


def test_newer_valid_preference_beats_older_expired_preference():
    result = fuse_memory_retrieval(
        query="release note tone",
        project_scope="hermes-agent",
        candidates=[
            {
                "id": "old",
                "content": "Use expansive release note tone.",
                "project_id": "hermes-agent",
                "valid_until": "2026-05-10T00:00:00Z",
                "created_at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "new",
                "content": "Use terse release note tone.",
                "project_id": "hermes-agent",
                "valid_from": "2026-05-22T00:00:00Z",
                "created_at": "2026-05-22T00:00:00Z",
            },
        ],
        now=NOW,
    )

    assert result["selected_memories"][0]["id"] == "new"
    assert result["scores"]["new"]["temporal_score"] > result["scores"]["old"]["temporal_score"]


def test_provenance_bearing_source_wins_when_content_is_similar():
    result = fuse_memory_retrieval(
        query="durable memory updates governed proposals",
        candidates=[
            {
                "id": "untrusted",
                "content": "Durable memory updates require governed proposals.",
                "source": "chat",
            },
            {
                "id": "trusted",
                "content": "Durable memory updates require governed proposals.",
                "source": "policy_doc",
                "provenance": "memory-fabric-policy:v1",
            },
        ],
        now=NOW,
    )

    assert result["selected_memories"][0]["id"] == "trusted"
    assert result["scores"]["trusted"]["source_trust_score"] > result["scores"]["untrusted"]["source_trust_score"]


def test_unsafe_governance_candidate_is_rejected():
    result = fuse_memory_retrieval(
        query="direct durable memory write",
        candidates=[
            {
                "id": "unsafe",
                "content": "Direct durable memory write is allowed.",
                "governance": {"would_write_memory": True},
            },
            {
                "id": "safe",
                "content": "Direct durable memory writes require governed proposals.",
                "governance": {"read_only": True, "proposal_governed": True},
            },
        ],
        now=NOW,
    )

    assert result["selected_memories"][0]["id"] == "safe"
    rejected = {item["id"]: item for item in result["rejected_memories"]}
    assert rejected["unsafe"]["reason"].startswith("unsafe_governance")
    assert rejected["unsafe"]["component_scores"]["governance_score"] == 0.0


def test_rejected_memories_explain_why_items_lost():
    result = fuse_memory_retrieval(
        query="Hermes memory retrieval fusion",
        candidates=[
            {"id": "winner", "content": "Hermes memory retrieval fusion is enabled."},
            {"id": "loser", "content": "Calendar sync is enabled."},
        ],
        now=NOW,
        limit=1,
    )

    rejected = {item["id"]: item for item in result["rejected_memories"]}
    assert "loser" in rejected
    assert rejected["loser"]["reason"] in {"below_selection_threshold", "ranked_below_limit"}
    assert "final_score" in rejected["loser"]
    assert "component_scores" in rejected["loser"]


def test_policy_proves_read_only_and_no_memory_config_or_graph_writes():
    result = fuse_memory_retrieval(query="policy", candidates=[], now=NOW)

    assert result["policy"] == FUSION_POLICY
    assert result["policy"]["read_only"] is True
    assert result["policy"]["would_write_memory"] is False
    assert result["policy"]["would_modify_config"] is False
    assert result["policy"]["would_write_graph"] is False
    assert result["policy"]["does_not_create_operation_events"] is True
