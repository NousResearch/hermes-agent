import json

from gateway.xiaoxing_working_memory import build_prompt_block, record_event, resolve_person


def test_milky_dad_qq_id_is_a_builtin_alias(monkeypatch):
    monkeypatch.delenv("XIAOXING_PERSON_ALIASES", raising=False)

    assert resolve_person(
        platform="milky",
        chat_id="group:610066383",
        user_id="490008192",
        user_name="李文浩",
    ) == ("dad", "爸爸")


def test_confusing_group_qq_ids_do_not_collapse_without_explicit_aliases(monkeypatch):
    monkeypatch.delenv("XIAOXING_PERSON_ALIASES", raising=False)

    assert resolve_person(
        platform="milky",
        chat_id="group:610066383",
        user_id="864806375",
        user_name="铭阳",
    ) == ("milky:864806375", "铭阳")
    assert resolve_person(
        platform="napcat",
        chat_id="group:610066383",
        user_id="519434661",
        user_name="泽铭",
    ) == ("napcat:519434661", "泽铭")
    assert resolve_person(
        platform="milky",
        chat_id="group:610066383",
        user_id="1305601309",
        user_name="王泽铭",
    ) == ("milky:1305601309", "王泽铭")
    assert resolve_person(
        platform="napcat",
        chat_id="group:610066383",
        user_id="1772845984",
        user_name="Z大哥",
    ) == ("napcat:1772845984", "Z大哥")


def test_prompt_block_resolves_current_group_sender_by_qq_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("XIAOXING_WORKING_MEMORY_PATH", str(tmp_path / "recent_events.jsonl"))
    monkeypatch.setenv(
        "XIAOXING_PERSON_ALIASES",
        json.dumps([
            {
                "platform": "napcat",
                "external_id": "qq-alice",
                "person_id": "alice",
                "person_name": "Alice Real",
            },
        ]),
    )

    record_event(
        direction="turn",
        platform="napcat",
        chat_id="qq-alice",
        user_id="qq-alice",
        user_name="Alice Private",
        text="private context",
        response_text="private answer",
    )

    prompt = build_prompt_block(
        current_platform="napcat",
        current_chat_id="group:610066383",
        current_user_id="qq-alice",
        current_user_name="Alice Card",
    )

    assert "Current channel resolves to: Alice Real<alice>." in prompt
    assert "with Alice Real<alice>" in prompt
    assert "private answer" in prompt


def test_prompt_block_links_group_and_private_by_qq_without_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("XIAOXING_WORKING_MEMORY_PATH", str(tmp_path / "recent_events.jsonl"))
    monkeypatch.delenv("XIAOXING_PERSON_ALIASES", raising=False)

    record_event(
        direction="turn",
        platform="napcat",
        chat_id="qq-bob",
        user_id="qq-bob",
        user_name="Bob Private",
        text="private context",
        response_text="private answer",
    )

    prompt = build_prompt_block(
        current_platform="napcat",
        current_chat_id="group:610066383",
        current_user_id="qq-bob",
        current_user_name="Bob Card",
    )

    assert "Current channel resolves to: Bob Card<napcat:qq-bob>." in prompt
    assert "with Bob Private<napcat:qq-bob>" in prompt
    assert "private answer" in prompt
