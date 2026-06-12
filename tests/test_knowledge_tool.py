import json


def test_knowledge_answer_merges_memory_skills_vault_and_sessions(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override

    home = tmp_path / "hermes-home"
    set_hermes_home_override(home)

    memories = home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("PETRA uses clean-sheet portfolio notes\n§\nUnrelated memory", encoding="utf-8")
    (memories / "USER.md").write_text("Pete prefers concise answers", encoding="utf-8")

    skills = home / "skills" / "finance" / "petra"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: petra\ndescription: PETRA investment workflow\n---\n# PETRA\nUse clean-sheet comparison notes.",
        encoding="utf-8",
    )

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "PETRA Note.md").write_text("# PETRA\nClean-sheet portfolio is not a rebalance.", encoding="utf-8")

    session_payload = {
        "success": True,
        "results": [
            {
                "session_id": "s1",
                "title": "PETRA discussion",
                "snippet": "Discussed clean-sheet portfolio constraints",
                "messages": [{"id": 1, "role": "assistant", "content": "clean-sheet means new allocation"}],
            }
        ],
    }

    def fake_session_search(**kwargs):
        assert kwargs["query"] == "PETRA clean-sheet"
        assert kwargs["limit"] == 3
        return json.dumps(session_payload)

    monkeypatch.setattr("tools.knowledge_tool.session_search", fake_session_search)

    from tools.knowledge_tool import knowledge_answer

    result = json.loads(knowledge_answer("PETRA clean-sheet", vault_paths=[str(vault)], max_results=3))

    assert result["success"] is True
    assert result["query"] == "PETRA clean-sheet"
    assert "memory" in result["sources"]
    assert result["sources"]["memory"][0]["target"] == "memory"
    assert result["sources"]["skills"][0]["name"] == "petra"
    assert result["sources"]["vault"][0]["path"].endswith("PETRA Note.md")
    assert result["sources"]["sessions"][0]["session_id"] == "s1"
    assert result["answer_contract"] == ["Known", "Uncertain", "Missing", "Next action"]
    assert "Known / Uncertain / Missing / Next action" in result["synthesis_instruction"]


def test_knowledge_answer_skips_missing_sources(tmp_path):
    from hermes_constants import set_hermes_home_override
    from tools.knowledge_tool import knowledge_answer

    set_hermes_home_override(tmp_path / "empty-home")

    result = json.loads(knowledge_answer("nonexistent topic", vault_paths=[str(tmp_path / "missing")], include_sources=["memory", "vault"], max_results=2))

    assert result["success"] is True
    assert result["sources"]["memory"] == []
    assert result["sources"]["vault"] == []
    assert "skills" not in result["sources"]
    assert "sessions" not in result["sources"]
