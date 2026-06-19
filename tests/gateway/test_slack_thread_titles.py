from gateway.slack_thread_titles import (
    apply_title_marker,
    apply_title_prefix,
    build_preview_fallback,
    build_thread_title_prompt,
    extract_retitle_request,
    generate_title,
    get_or_create_thread_title,
    get_thread_title,
    set_thread_title,
)


def test_get_or_create_persists_stable_title(tmp_path, monkeypatch):
    store = tmp_path / "slack_thread_titles.json"
    monkeypatch.setattr("gateway.slack_thread_titles.store_path", lambda: store)

    first = get_or_create_thread_title(
        "C123",
        "1781837641.289239",
        "When responding to a message lead with a 5-10 word title",
    )
    second = get_or_create_thread_title(
        "C123",
        "1781837641.289239",
        "A different later reply should not rename the thread",
    )

    assert first == second
    assert get_thread_title("C123", "1781837641.289239") == first
    assert 5 <= len(first.split()) <= 10


def test_explicit_retitle_overrides_existing_title(tmp_path, monkeypatch):
    store = tmp_path / "slack_thread_titles.json"
    monkeypatch.setattr("gateway.slack_thread_titles.store_path", lambda: store)

    set_thread_title("C123", "111.222", "Old Infrastructure Thread Title")
    requested = extract_retitle_request("retitle this thread: Slack Preview Thread Title Policy")
    assert requested == "Slack Preview Thread Title Policy"

    set_thread_title("C123", "111.222", requested)
    assert get_thread_title("C123", "111.222") == "Slack Preview Thread Title Policy"


def test_prompt_and_enforcement_use_exact_markdown_edges():
    prompt = build_thread_title_prompt("Slack Preview Thread Title Policy")
    assert "**Slack Preview Thread Title Policy:**" in prompt
    assert "Begin every user-visible reply" in prompt
    assert "final paragraph" in prompt

    assert apply_title_prefix("Done.", "Slack Preview Thread Title Policy") == (
        "**Slack Preview Thread Title Policy:**\n\nDone.\n\n**Slack Preview Thread Title Policy:**"
    )
    assert apply_title_marker(
        "Already titled.\n\n**Slack Preview Thread Title Policy:**",
        "Slack Preview Thread Title Policy",
        placement="last",
    ) == "Already titled.\n\n**Slack Preview Thread Title Policy:**"
    assert apply_title_marker(
        "**Slack Preview Thread Title Policy:**\n\nLegacy leading title.",
        "Slack Preview Thread Title Policy",
        placement="first",
    ) == "**Slack Preview Thread Title Policy:**\n\nLegacy leading title."


def test_preview_fallback_starts_with_title_and_body_excerpt():
    fallback = build_preview_fallback(
        "**Slack Preview Thread Title Policy:**\n\nImplemented the gateway fallback.\n\n**Slack Preview Thread Title Policy:**",
        "Slack Preview Thread Title Policy",
    )
    assert fallback == "Slack Preview Thread Title Policy: Implemented the gateway fallback."


def test_generated_title_uses_deterministic_five_to_ten_words():
    title = generate_title("Can you expand the Longhorn volume in prd k8s safely?")
    assert title == generate_title("Can you expand the Longhorn volume in prd k8s safely?")
    assert 5 <= len(title.split()) <= 10
