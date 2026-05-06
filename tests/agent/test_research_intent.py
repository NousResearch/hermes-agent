import pytest

from agent.research_search import classify_research_intent


def test_research_intent_classifies_current_fact_requests():
    intent = classify_research_intent("latest Detroit Pistons current starting five")

    assert intent["is_research"] is True
    assert intent["redirect_web_search"] is True
    assert intent["topic_type"] == "sports"
    assert intent["freshness"] == "latest"
    assert "latest" in intent["matched"]["latest"]


def test_research_intent_classifies_medical_pharma_before_current_events():
    intent = classify_research_intent(
        "latest GLP-1 GIP drugs in phase 3 clinical trials"
    )

    assert intent["is_research"] is True
    assert intent["redirect_web_search"] is True
    assert intent["topic_type"] == "medical_pharma"
    assert intent["freshness"] == "latest"
    assert "glp-1" in intent["matched"]["topic"]
    assert "phase 3" in intent["matched"]["topic"]


def test_research_intent_keeps_targeted_lookup_out_of_redirect():
    intent = classify_research_intent("site:example.com foobar")

    assert intent["redirect_web_search"] is False
    assert intent["topic_type"] == "general"
    assert "site:" in intent["matched"]["targeted_lookup"]


def test_research_intent_uses_word_boundaries_for_short_markers():
    intent = classify_research_intent("capital allocation overview")

    assert intent["topic_type"] == "general"
    assert "api" not in intent["matched"]["topic"]


def test_research_intent_does_not_treat_pipeline_alone_as_pharma():
    intent = classify_research_intent("CI pipeline bug in GitHub actions")

    assert intent["topic_type"] == "technical"
    assert intent["redirect_web_search"] is False


@pytest.mark.parametrize(
    ("query", "topic_type"),
    [
        ("Nvidia Blackwell GPU launch latest news", "technology"),
        ("Apple earnings guidance SEC filing", "finance"),
        ("ASME pressure vessel standard failure analysis", "engineering"),
        ("Elden Ring DLC patch notes performance issues", "gaming"),
        ("TikTok viral trend report with sources", "social_trends"),
        ("Billboard chart history for the latest Kendrick album", "music"),
        ("best OLED monitor deals and reviews", "shopping"),
        ("White House domestic news latest policy statement", "domestic_news"),
        ("NATO Ukraine ceasefire negotiations latest", "geopolitics"),
    ],
)
def test_research_intent_classifies_additional_topic_profiles(query, topic_type):
    intent = classify_research_intent(query)

    assert intent["topic_type"] == topic_type
    assert intent["is_research"] is True
    assert intent["redirect_web_search"] is True
