import json

from hermes_cli import agents_os
from hermes_cli.agents_os_idea_factory import draft_idea, idea_factory_schema


def test_schema_exposes_required_fields_and_risk_classes():
    schema = idea_factory_schema()

    assert "idea_text" in schema["input_fields"]
    assert "classification" in schema["output_fields"]
    assert "safe_local" in schema["risk_classes"]
    assert "credential_gated" in schema["risk_classes"]


def test_drafts_safe_youtube_intake_as_local_research():
    draft = draft_idea("Obradi YouTube video i spremi ga u vault")

    assert draft["classification"] == "research_intake"
    assert draft["risk_class"] == "safe_local"
    assert draft["approval_required"] is False
    assert draft["suggested_agent"] == "doni-local"
    assert draft["recommended_lane"] == "youtube-content-intake"
    assert draft["plan_steps"]
    assert draft["acceptance_criteria"]


def test_drafts_agent_os_build_as_local_write_approval():
    draft = draft_idea("Dodaj Memory Galaxy u Doni dashboard")

    assert draft["classification"] == "agent_os_build"
    assert draft["risk_class"] == "approval_local_write"
    assert draft["approval_required"] is True
    assert draft["recommended_lane"] == "mission-control-build"


def test_high_risk_ideas_are_approval_gated():
    cases = [
        ("Pošalji email klijentu", "public_outbound_gated", "public_gated"),
        ("Spoji Kraken live trading", "finance_gated", "finance_gated"),
        ("Skeniraj tuđi web za ranjivosti", "security_gated", "security_gated"),
        ("Obriši stare memorije", "memory_skill_ops", "destructive_gated"),
        ("Spoji API ključ i koristi credentiale", "unclear_needs_context", "credential_gated"),
    ]

    for text, classification, risk in cases:
        draft = draft_idea(text)
        assert draft["classification"] == classification
        assert draft["risk_class"] == risk
        assert draft["approval_required"] is True


def test_public_web_offer_is_safe_local_draft():
    draft = draft_idea("Napravi landing za web revenue audit")

    assert draft["classification"] == "web_seo_offer"
    assert draft["risk_class"] == "safe_local"
    assert draft["approval_required"] is False
    assert draft["recommended_lane"] == "web-seo-offer"


def test_agents_os_idea_schema_cli_returns_contract(capsys):
    assert agents_os.main(["idea", "schema", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "idea_text" in payload["input_fields"]
    assert "approval_required" in payload["output_fields"]
    assert payload["local_only"] is True


def test_agents_os_idea_draft_cli_outputs_draft(capsys):
    assert agents_os.main(["idea", "draft", "Obradi YouTube video", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "research_intake"
    assert payload["risk_class"] == "safe_local"
    assert payload["approval_required"] is False


def test_agents_os_idea_draft_cli_gates_public_action(capsys):
    assert agents_os.main(["idea", "draft", "Pošalji email klijentu", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "public_outbound_gated"
    assert payload["risk_class"] == "public_gated"
    assert payload["approval_required"] is True
