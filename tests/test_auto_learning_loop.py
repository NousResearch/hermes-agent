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



def _make_agent(auto_learning_config: dict):
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("memory", "skill_manage"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config",
            return_value={
                "memory": {"memory_enabled": False, "user_profile_enabled": False},
                "skills": {"creation_nudge_interval": 10},
                "auto_learning": auto_learning_config,
            },
        ),
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



def test_auto_learning_review_promotes_high_confidence_memory_candidate(tmp_path):
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

    with patch("tools.skill_manager_tool.skill_manage", return_value='{"success": true, "message": "Skill updated."}') as mock_skill_manage:
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
    mock_skill_manage.assert_called_once()



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
