from agent.beta.router import Intent, route_request
from agent.beta.specialists import Specialist, SpecialistRegistry


def test_postgresql_diagnosis_selects_database_infra_and_monitoring():
    decision = route_request("Verifique por que o PostgreSQL está lento")
    assert decision.intent == Intent.DIAGNOSIS
    assert set(decision.specialists) == {"dba", "infra", "monitoring"}
    assert decision.initial_risk == "low"
    assert decision.delegation_needed is True
    assert decision.parallelizable is True


def test_contract_review_does_not_select_infrastructure_specialists():
    decision = route_request("Revise este contrato")
    assert decision.intent == Intent.AUDIT
    assert not {"infra", "dba", "monitoring", "devops"}.intersection(decision.specialists)


def test_simple_conversation_does_not_delegate():
    decision = route_request("Olá, tudo bem?")
    assert decision.intent == Intent.CONVERSATION
    assert decision.specialists == ()
    assert decision.delegation_needed is False


def test_selection_comes_from_manifest_capabilities():
    legal = Specialist(
        id="legal",
        name="Legal",
        description="Reviews contracts",
        capabilities=("contract", "legal-review"),
        keywords=("contract", "contrato"),
    )
    decision = route_request("Revise este contrato", SpecialistRegistry([legal]))
    assert decision.specialists == ("legal",)


def test_ambiguous_request_selects_multiple_specialists():
    decision = route_request("Audite a segurança do firewall e do pipeline de deploy")
    assert {"security", "devops"}.issubset(decision.specialists)
    assert decision.parallelizable is True
    assert 0 <= decision.confidence <= 1

