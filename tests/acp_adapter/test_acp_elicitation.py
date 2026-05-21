from types import SimpleNamespace

from acp_adapter.elicitation import (
    build_clarify_requested_schema,
    extract_clarify_answer,
    supports_form_elicitation,
)


def test_supports_form_elicitation_with_object_form():
    caps = SimpleNamespace(elicitation=SimpleNamespace(form=SimpleNamespace()))
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_with_dict_form():
    caps = SimpleNamespace(elicitation={"form": {}})
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_with_empty_elicitation_object():
    # ACP RFD-style shorthand: empty elicitation object means form supported.
    caps = SimpleNamespace(elicitation={})
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_missing_is_false():
    assert supports_form_elicitation(SimpleNamespace()) is False
    assert supports_form_elicitation(None) is False


def test_supports_form_elicitation_url_only_is_false():
    caps = SimpleNamespace(elicitation={"url": {}})
    assert supports_form_elicitation(caps) is False


def test_open_ended_clarify_schema_has_required_answer_string():
    schema = build_clarify_requested_schema(question="What branch?", choices=None)
    assert schema["type"] == "object"
    assert schema["required"] == ["answer"]
    assert schema["properties"]["answer"]["type"] == "string"


def test_choice_clarify_schema_preserves_other_path():
    schema = build_clarify_requested_schema(
        question="Which approach?",
        choices=["Strict", "Fallback"],
    )
    props = schema["properties"]
    assert "answer" in props
    assert "other_answer" in props
    assert "Strict" in props["answer"].get("enum", [])
    assert "Fallback" in props["answer"].get("enum", [])
    assert "Other (type your answer)" in props["answer"].get("enum", [])


def test_extract_answer_from_accept_dict():
    response = {"action": "accept", "content": {"answer": "Strict"}}
    assert extract_clarify_answer(response) == "Strict"


def test_extract_other_answer_from_accept_dict():
    response = {
        "action": "accept",
        "content": {"answer": "Other (type your answer)", "other_answer": "Use raw JSON-RPC"},
    }
    assert extract_clarify_answer(response) == "Use raw JSON-RPC"


def test_extract_answer_from_accept_object():
    response = SimpleNamespace(action="accept", content={"answer": "Fallback"})
    assert extract_clarify_answer(response) == "Fallback"


def test_decline_cancel_or_missing_returns_standard_sentinel():
    assert extract_clarify_answer({"action": "decline"}) == "[user declined the clarification]"
    assert extract_clarify_answer({"action": "cancel"}) == "[user cancelled the clarification]"
    assert extract_clarify_answer(None) == "[clarify prompt could not be delivered]"
