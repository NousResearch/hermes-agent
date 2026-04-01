from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_agent
from run_agent import AIAgent



def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]



def _base_config(auto_learning_config: dict) -> dict:
    return {
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "skills": {"creation_nudge_interval": 10},
        "auto_learning": auto_learning_config,
    }



def _make_agent(auto_learning_config: dict):
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("memory", "skill_manage"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=_base_config(auto_learning_config)),
    ):
        agent = AIAgent(
            api_key="***",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="test/model",
            usage=None,
        )
        return agent



def test_agent_initializes_auto_learning_store_when_enabled(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 2,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 5,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    assert agent._auto_learning_enabled is True
    assert agent._auto_learning_store is not None
    assert agent._auto_learning_config["candidate_max_entries"] == 5



def test_agent_keeps_auto_learning_inert_when_disabled():
    agent = _make_agent(
        {
            "enabled": False,
            "review_interval": 1,
            "min_tool_iterations": 2,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 5,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": "",
            "debug": False,
        }
    )

    assert agent._auto_learning_enabled is False
    assert agent._auto_learning_store is None



def test_auto_learning_review_stages_low_confidence_candidate(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    review_text = (
        '{"candidates": ['
        '{"category": "memory", "summary": "User prefers concise responses", '
        '"confidence": 0.5, "reason": "single weak signal", "target": "user", '
        '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
    )

    result = agent._process_auto_learning_review_result(review_text)
    items = agent._auto_learning_store.list_candidates()

    assert result["staged"] == 1
    assert result["promoted"] == 0
    assert len(items) == 1
    assert items[0]["status"] == "candidate"



def test_auto_learning_review_stages_structured_audit_evidence(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": False,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    review_text = (
        '{"candidates": ['
        '{"category": "memory", "summary": "User prefers concise responses", '
        '"confidence": 0.5, "reason": "single weak signal", "target": "user", '
        '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
    )

    result = agent._process_auto_learning_review_result(
        review_text,
        review_context={
            "hook_reason": "failure_recovery",
            "hook_signals": ["failure_recovery", "delegated_completion"],
            "source": {
                "trigger": "post_response_review",
                "actor": "reviewer",
                "model": "anthropic/claude-opus-4.6",
            },
            "metrics": {
                "iteration_count": 3,
                "tool_call_count": 2,
                "failed_tool_call_count": 1,
                "delegated_task_count": 1,
            },
            "transcript_refs": [
                {"message_index": 0, "role": "user"},
                {"message_index": 1, "role": "assistant"},
            ],
            "transcript_excerpt": "User asked for concise responses after a failed tool run that recovered via delegation.",
        },
    )

    items = agent._auto_learning_store.list_candidates()

    assert result["staged"] == 1
    assert len(items) == 1
    evidence = items[0]["evidence"]
    assert evidence["candidate_reason"] == "single weak signal"
    assert evidence["hook_reason"] == "failure_recovery"
    assert evidence["hook_signals"] == ["failure_recovery", "delegated_completion"]
    assert evidence["source"]["trigger"] == "post_response_review"
    assert evidence["metrics"] == {
        "iteration_count": 3,
        "tool_call_count": 2,
        "failed_tool_call_count": 1,
        "delegated_task_count": 1,
    }
    assert evidence["transcript_refs"] == [
        {"message_index": 0, "role": "user"},
        {"message_index": 1, "role": "assistant"},
    ]
    assert evidence["transcript_excerpt"] == "User asked for concise responses after a failed tool run that recovered via delegation."



def test_auto_learning_role_records_critic_audit_when_rejecting(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
            "verifier": {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-latest",
            },
            "critic": {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-latest",
            },
        }
    )
    agent._memory_store = MagicMock()

    review_text = (
        '{"candidates": ['
        '{"category": "memory", "summary": "User prefers concise responses", '
        '"confidence": 0.95, "reason": "repeated explicit correction", "target": "user", '
        '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
    )

    with (
        patch.object(agent, "_resolve_auto_learning_actor_settings", return_value={
            "model": "claude-3-5-haiku-latest",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key": "verifier-key",
            "api_mode": "anthropic_messages",
            "max_iterations": 4,
            "timeout": None,
        }),
        patch("agent.auto_learning.build_auto_learning_verifier_prompt", return_value="verify prompt"),
        patch("run_agent.AIAgent") as mock_child_cls,
        patch("tools.memory_tool.memory_tool") as mock_memory_tool,
    ):
        mock_child = MagicMock()
        mock_child.run_conversation.return_value = {
            "final_response": '{"decisions": [{"index": 0, "disposition": "reject", "confidence": 0.0, "reason": "Single weak signal only."}]}'
        }
        mock_child_cls.return_value = mock_child

        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates()
    assert result["rejected"] == 1
    assert result["promoted"] == 0
    assert len(items) == 1
    assert items[0]["status"] == "rejected"
    assert items[0]["evidence"]["verifier"]["disposition"] == "reject"
    assert items[0]["evidence"]["critic"]["disposition"] == "reject"
    mock_memory_tool.assert_not_called()



def test_auto_learning_review_downscores_candidate_before_promotion(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
            "verifier": {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-latest",
            },
        }
    )
    agent._memory_store = MagicMock()

    review_text = (
        '{"candidates": ['
        '{"category": "memory", "summary": "User prefers concise responses", '
        '"confidence": 0.95, "reason": "repeated explicit correction", "target": "user", '
        '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
    )

    with (
        patch.object(agent, "_resolve_auto_learning_actor_settings", return_value={
            "model": "claude-3-5-haiku-latest",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key": "verifier-key",
            "api_mode": "anthropic_messages",
            "max_iterations": 4,
            "timeout": None,
        }),
        patch("agent.auto_learning.build_auto_learning_verifier_prompt", return_value="verify prompt"),
        patch("run_agent.AIAgent") as mock_child_cls,
        patch("tools.memory_tool.memory_tool") as mock_memory_tool,
    ):
        mock_child = MagicMock()
        mock_child.run_conversation.return_value = {
            "final_response": '{"decisions": [{"index": 0, "disposition": "downscore", "confidence": 0.41, "reason": "Useful, but only one observed instance."}]}'
        }
        mock_child_cls.return_value = mock_child

        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates()
    assert result["staged"] == 1
    assert result["promoted"] == 0
    assert len(items) == 1
    assert items[0]["status"] == "candidate"
    assert items[0]["confidence"] == 0.41
    assert items[0]["evidence"]["verifier"]["disposition"] == "downscore"
    mock_memory_tool.assert_not_called()



def test_auto_learning_role_records_promoter_audit_when_promoting(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
            "promoter": {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-latest",
            },
        }
    )
    agent._memory_store = MagicMock()

    with patch("tools.memory_tool.memory_tool", return_value='{"success": true, "message": "Entry added.", "target": "user"}') as mock_memory_tool:
        review_text = (
            '{"candidates": ['
            '{"category": "memory", "summary": "User prefers concise responses", '
            '"confidence": 0.95, "reason": "repeated explicit correction", "target": "user", '
            '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
        )

        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates(status="promoted")
    assert result["promoted"] == 1
    assert len(items) == 1
    assert items[0]["status"] == "promoted"
    assert items[0]["evidence"]["promoter"]["disposition"] == "promote"
    mock_memory_tool.assert_called_once()



def test_auto_learning_review_keeps_skill_candidate_staged_when_auto_skill_promotion_disabled(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    with patch("tools.skill_manager_tool.skill_manage") as mock_skill_manage:
        review_text = (
            '{"candidates": ['
            '{"category": "skill", "summary": "Patch outdated OpenVINO steps", '
            '"confidence": 0.99, "reason": "reusable workflow fix", "target": "openvino-qwen-no-think", '
            '"payload": {"action": "patch", "old_string": "old", "new_string": "new"}}]}'
        )

        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates()
    assert result["promoted"] == 0
    assert len(items) == 1
    assert items[0]["status"] == "candidate"
    mock_skill_manage.assert_not_called()



def test_auto_learning_review_promotes_skill_candidate_when_enabled(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": True,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    with (
        patch("tools.skill_manager_tool.replay_validate_skill_candidate", return_value={"valid": True, "action": "patch", "name": "openvino-qwen-no-think"}) as mock_validate,
        patch("tools.skill_manager_tool.skill_manage", return_value='{"success": true, "message": "Skill updated."}') as mock_skill_manage,
    ):
        review_text = (
            '{"candidates": ['
            '{"category": "skill", "summary": "Patch outdated OpenVINO steps", '
            '"confidence": 0.99, "reason": "reusable workflow fix", "target": "openvino-qwen-no-think", '
            '"payload": {"action": "patch", "old_string": "old", "new_string": "new"}}]}'
        )

        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates(status="promoted")
    assert result["promoted"] == 1
    assert len(items) == 1
    mock_validate.assert_called_once()
    mock_skill_manage.assert_called_once()



def test_auto_learning_review_blocks_skill_promotion_on_invalid_replay_validation(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": True,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    with (
        patch("tools.skill_manager_tool.replay_validate_skill_candidate", return_value={"valid": False, "action": "patch", "name": "openvino-qwen-no-think", "error": "old_string not found in the file."}) as mock_validate,
        patch("tools.skill_manager_tool.skill_manage") as mock_skill_manage,
    ):
        review_text = (
            '{"candidates": ['
            '{"category": "skill", "summary": "Patch outdated OpenVINO steps", '
            '"confidence": 0.99, "reason": "reusable workflow fix", "target": "openvino-qwen-no-think", '
            '"payload": {"action": "patch", "old_string": "old", "new_string": "new"}}]}'
        )

        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates(status="manual_review")
    assert result["manual_review"] == 1
    assert len(items) == 1
    assert items[0]["evidence"]["quality"]["skill_validation"]["valid"] is False
    assert "not found" in items[0]["evidence"]["quality"]["skill_validation"]["error"]
    mock_validate.assert_called_once()
    mock_skill_manage.assert_not_called()



def test_auto_learning_review_rejects_malformed_skill_payload(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": True,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    review_text = (
        '{"candidates": ['
        '{"category": "skill", "summary": "Bad skill payload", '
        '"confidence": 0.99, "reason": "broken payload", "target": "", '
        '"payload": {"action": "patch"}}]}'
    )

    result = agent._process_auto_learning_review_result(review_text)

    rejected = agent._auto_learning_store.list_candidates(status="rejected")
    assert result["rejected"] == 1
    assert len(rejected) == 1



def test_auto_learning_review_dedupes_duplicate_candidate(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": False,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    review_text = (
        '{"candidates": ['
        '{"category": "memory", "summary": "User prefers concise responses", '
        '"confidence": 0.7, "reason": "signal", "target": "user", '
        '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
    )

    first = agent._process_auto_learning_review_result(review_text)
    second = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates()
    assert first["staged"] == 1
    assert second["staged"] == 1
    assert len(items) == 1



def test_auto_learning_review_routes_contradictory_memory_to_manual_review(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )
    agent._memory_store = MagicMock()
    agent._memory_store.user_entries = ["User prefers concise responses"]
    agent._memory_store.memory_entries = []

    review_text = (
        '{"candidates": ['
        '{"category": "memory", "summary": "User prefers verbose responses", '
        '"confidence": 0.94, "reason": "new correction", "target": "user", '
        '"payload": {"action": "add", "content": "User prefers verbose responses."}}]}'
    )

    with patch("tools.memory_tool.memory_tool") as mock_memory_tool:
        result = agent._process_auto_learning_review_result(review_text)

    items = agent._auto_learning_store.list_candidates(status="manual_review")
    assert result["manual_review"] == 1
    assert len(items) == 1
    assert items[0]["evidence"]["quality"]["contradictions"]["has_contradiction"] is True
    mock_memory_tool.assert_not_called()



def test_auto_learning_review_records_semantic_supersession_for_variant_wording(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": False,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    first = agent._process_auto_learning_review_result(
        '{"candidates": [{"category": "memory", "summary": "User prefers concise responses", "confidence": 0.55, "reason": "signal", "target": "user", "payload": {"action": "add", "content": "User prefers concise responses."}}]}'
    )
    second = agent._process_auto_learning_review_result(
        '{"candidates": [{"category": "memory", "summary": "User likes brief answers", "confidence": 0.87, "reason": "stronger signal", "target": "user", "payload": {"action": "add", "content": "User likes brief answers."}}]}'
    )

    items = agent._auto_learning_store.list_candidates()
    assert first["staged"] == 1
    assert second["superseded"] == 1
    assert len(items) == 2
    assert len([item for item in items if item["status"] == "superseded"]) == 1



def test_run_conversation_spawns_background_auto_learning_review_when_triggered(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": False,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    agent._iters_since_skill = 1

    with patch.object(agent, "_spawn_auto_learning_review") as mock_auto_review:
        agent.run_conversation("hello")

    mock_auto_review.assert_called_once()


def test_auto_learning_review_notifies_on_successful_memory_promotion(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )
    agent._memory_store = MagicMock()
    agent.background_review_callback = MagicMock()

    with patch("tools.memory_tool.memory_tool", return_value='{"success": true, "message": "Entry added.", "target": "user"}'):
        review_text = (
            '{"candidates": ['
            '{"category": "memory", "summary": "User prefers concise responses", '
            '"confidence": 0.95, "reason": "repeated explicit correction", "target": "user", '
            '"payload": {"action": "add", "content": "User prefers concise responses."}}]}'
        )

        result = agent._process_auto_learning_review_result(review_text)

    assert result["promoted"] == 1
    agent.background_review_callback.assert_called_once()
    notification = agent.background_review_callback.call_args.args[0]
    assert "Memory upgraded" in notification
    assert "User prefers concise responses" in notification



def test_auto_learning_review_notifies_on_successful_skill_promotion(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 1,
            "min_tool_iterations": 1,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": True,
            "auto_promote_skills": True,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )
    agent.background_review_callback = MagicMock()

    with (
        patch("tools.skill_manager_tool.replay_validate_skill_candidate", return_value={"valid": True, "action": "patch", "name": "openvino-qwen-no-think"}),
        patch("tools.skill_manager_tool.skill_manage", return_value='{"success": true, "message": "Skill updated."}'),
    ):
        review_text = (
            '{"candidates": ['
            '{"category": "skill", "summary": "Patch outdated OpenVINO steps", '
            '"confidence": 0.99, "reason": "reusable workflow fix", "target": "openvino-qwen-no-think", '
            '"payload": {"action": "patch", "old_string": "old", "new_string": "new"}}]}'
        )

        result = agent._process_auto_learning_review_result(review_text)

    assert result["promoted"] == 1
    agent.background_review_callback.assert_called_once()
    notification = agent.background_review_callback.call_args.args[0]
    assert "Skill upgraded" in notification
    assert "openvino-qwen-no-think" in notification



def test_run_conversation_does_not_spawn_background_auto_learning_review_when_not_triggered(tmp_path):
    agent = _make_agent(
        {
            "enabled": True,
            "review_interval": 2,
            "min_tool_iterations": 3,
            "candidate_char_limit": 12000,
            "candidate_max_entries": 10,
            "promotion_threshold": 0.8,
            "auto_promote_memory": False,
            "auto_promote_skills": False,
            "store_path": str(tmp_path / "candidates.jsonl"),
            "debug": False,
        }
    )

    agent._iters_since_skill = 1

    with patch.object(agent, "_spawn_auto_learning_review") as mock_auto_review:
        agent.run_conversation("hello")

    mock_auto_review.assert_not_called()
