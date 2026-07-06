from gateway.topic_context import format_topic_context_prompt


def test_format_topic_context_prompt_renders_compact_metadata():
    prompt = format_topic_context_prompt(
        {
            "platform": "telegram",
            "chat_id": "-1001",
            "thread_id": "205",
            "chat_name": "Hermes",
            "topic_name": "Hermes docs patch",
            "purpose": "Keep topic context after /new.",
            "skills": ["hermes-agent", "obsidian"],
        }
    )

    assert "Telegram group topic context" in prompt
    assert "not chat history" in prompt
    assert 'Group: "Hermes"' in prompt
    assert "Topic: Hermes docs patch" in prompt
    assert "Topic ID: 205" in prompt
    assert "Purpose: Keep topic context after /new." in prompt
    assert "Topic-bound skills: hermes-agent, obsidian" in prompt


def test_format_topic_context_prompt_renders_workdir_when_set():
    prompt = format_topic_context_prompt(
        {
            "thread_id": "205",
            "purpose": "Project workspace.",
            "workdir": "/srv/project",
        }
    )
    assert "Workdir: /srv/project" in prompt
    assert "project rules are loaded from here" in prompt


def test_format_topic_context_prompt_omits_workdir_when_absent():
    prompt = format_topic_context_prompt(
        {
            "thread_id": "205",
            "purpose": "Project workspace.",
        }
    )
    assert "Workdir:" not in prompt


def test_format_topic_context_prompt_empty_context_is_empty():
    assert format_topic_context_prompt({}) == ""


def test_format_topic_context_prompt_truncates_long_values():
    prompt = format_topic_context_prompt(
        {
            "thread_id": "205",
            "purpose": "x" * 2000,
            "skills": ["y" * 2000],
        }
    )

    assert "x" * 900 not in prompt
    assert "y" * 900 not in prompt
    assert len(prompt) < 2500
