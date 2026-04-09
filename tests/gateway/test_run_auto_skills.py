from agent.skill_commands import _build_skill_message
from gateway.run import _build_auto_skill_message


def test_build_auto_skill_message_supports_multiple_skills(monkeypatch):
    loaded = {
        "neomesh-core": ({"content": "NEOMESH CORE"}, None, "neomesh-core"),
        "backend": ({"content": "BACKEND ROLE"}, None, "backend"),
    }

    monkeypatch.setattr(
        "gateway.run._load_skill_payload",
        lambda skill_identifier, task_id=None: loaded.get(skill_identifier),
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._build_skill_message",
        lambda loaded_skill, skill_dir, activation_note, user_instruction="", runtime_note="": _build_skill_message(
            loaded_skill,
            skill_dir,
            activation_note,
            user_instruction=user_instruction,
            runtime_note=runtime_note,
        ),
        raising=False,
    )

    text = _build_auto_skill_message(["neomesh-core", "backend"], "Investigate API latency", task_id="t1")

    assert "neomesh-core" in text
    assert "backend" in text
    assert "NEOMESH CORE" in text
    assert "BACKEND ROLE" in text
    assert "Investigate API latency" in text


def test_build_auto_skill_message_returns_original_text_when_skills_missing(monkeypatch):
    monkeypatch.setattr("gateway.run._load_skill_payload", lambda skill_identifier, task_id=None: None, raising=False)

    text = _build_auto_skill_message(["missing-skill"], "Hello", task_id="t1")

    assert text == "Hello"


def test_build_auto_skill_message_preserves_user_text_when_last_skill_is_missing(monkeypatch):
    loaded = {
        "neomesh-core": ({"content": "NEOMESH CORE"}, None, "neomesh-core"),
    }

    monkeypatch.setattr(
        "gateway.run._load_skill_payload",
        lambda skill_identifier, task_id=None: loaded.get(skill_identifier),
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.run._build_skill_message",
        lambda loaded_skill, skill_dir, activation_note, user_instruction="", runtime_note="": _build_skill_message(
            loaded_skill,
            skill_dir,
            activation_note,
            user_instruction=user_instruction,
            runtime_note=runtime_note,
        ),
        raising=False,
    )

    text = _build_auto_skill_message(["neomesh-core", "missing-skill"], "Keep the user text", task_id="t1")

    assert "NEOMESH CORE" in text
    assert "Keep the user text" in text
