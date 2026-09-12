from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "clinical-evidence-guard"
    / "__init__.py"
)


def _load_plugin():
    name = "_test_clinical_evidence_guard"
    spec = importlib.util.spec_from_file_location(name, _PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._reset_for_tests()
    return module


@pytest.fixture
def plugin():
    module = _load_plugin()
    yield module
    module._reset_for_tests()


def test_registers_every_declared_delivery_hook(plugin):
    ctx = SimpleNamespace(register_hook=lambda name, callback: registered.append((name, callback)))
    registered = []

    plugin.register(ctx)

    assert [name for name, _ in registered] == [
        "pre_llm_call",
        "post_tool_call",
        "pre_delivery_scope",
        "pre_delivery",
    ]


def test_nonclinical_turn_keeps_streaming_and_delivery_open(plugin):
    assert plugin._on_pre_delivery_scope("Organiza mis archivos") is False
    plugin._on_pre_llm_call(session_id="ordinary", user_message="Organiza mis archivos")
    assert plugin._on_pre_delivery(
        session_id="ordinary", response_text="Los archivos quedaron ordenados."
    ) == {"action": "allow"}


def test_meta_prefixed_clinical_validation_turn_is_still_protected(plugin):
    prompt = (
        "Consulta de validación interna: responde brevemente, con evidencia recuperada en "
        "este turno, cuál es el principal riesgo posoperatorio temprano tras ritidectomía; "
        "coloca la respuesta antes del ledger."
    )

    assert plugin._on_pre_delivery_scope(prompt) is True


def test_contextual_clinical_follow_up_is_protected_before_streaming(plugin):
    history = [
        {"role": "user", "content": "¿Cuál es la dosis geriátrica de este medicamento?"},
        {"role": "assistant", "content": "Respuesta clínica anterior."},
    ]

    assert plugin._on_pre_delivery_scope(
        "¿Y en ancianos?", conversation_history=history
    ) is True


def test_classification_failure_fails_closed(plugin, monkeypatch):
    def fail(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(plugin, "_is_clinical", fail)

    assert plugin._on_pre_delivery_scope("consulta") is True
    plugin._on_pre_llm_call(session_id="classification-error", user_message="consulta")
    assert plugin._STATES[plugin._session_key("classification-error")].clinical is True


def test_positive_claim_is_not_rejected_by_unrelated_negation_in_evidence(plugin):
    assert plugin._polarity_matches(
        "Hematoma is the main early complication.",
        "Hematoma is the main early complication. There was no difference in infection.",
    ) is True
    assert plugin._polarity_matches(
        "Treatment does not reduce hematoma.",
        "Treatment reduces hematoma.",
    ) is False


def test_positive_claim_is_rejected_when_relevant_evidence_is_negative(plugin):
    assert plugin._polarity_matches(
        "La aspirina reduce la mortalidad.",
        "La aspirina no reduce la mortalidad.",
    ) is False


@pytest.mark.parametrize("artifact", ["sitio web", "plugin", "código", "workflow"])
def test_clinical_request_cannot_be_excluded_as_software_workflow(plugin, artifact):
    assert plugin._is_clinical(
        f"Revisa el tratamiento para mi paciente en el {artifact}"
    ) is True


def test_technical_guard_status_report_is_not_clinical(plugin):
    assert plugin._is_clinical(
        "Corregidos los bypasses de clasificación clínica, plugin, gateway y streaming."
    ) is False


def test_overlapping_turns_in_one_session_keep_separate_state(plugin):
    plugin._on_pre_llm_call(
        session_id="shared", turn_id="clinical",
        user_message="¿Qué tratamiento indico a este paciente?",
    )
    plugin._on_pre_llm_call(
        session_id="shared", turn_id="ordinary",
        user_message="Resume el estado del repositorio",
    )

    decision = plugin._on_pre_delivery(
        session_id="shared", turn_id="clinical",
        response_text="Se recomienda un tratamiento específico.",
    )

    assert decision["action"] == "block"
    assert plugin._STATES[plugin._session_key("shared", "clinical")].clinical is True
    assert plugin._STATES[plugin._session_key("shared", "ordinary")].clinical is False


def test_inverted_numeric_relation_is_not_certified(plugin):
    session_id = "inverted-relation"
    plugin._on_pre_llm_call(
        session_id=session_id,
        user_message="¿Qué dosis reduce el sangrado mayor?",
    )
    plugin._on_post_tool_call(
        session_id=session_id,
        tool_name="mcp__consensus__search",
        status="ok",
        result=(
            '{"title":"Dose comparison","abstract":"5 mg caused major bleeding. '
            '2.5 mg reduced major bleeding.","doi":"10.1000/example"}'
        ),
    )
    response = (
        "5 mg reduced major bleeding.\n\nFuentes declaradas\n"
        "Proveedores consultados: consensus\n"
        "Limitaciones de recuperación: ninguna.\n"
        "Estado de verificación: verificado."
    )

    assert plugin._on_pre_delivery(
        session_id=session_id, response_text=response
    )["action"] == "block"


def test_cross_language_clinical_terms_support_relevance(plugin):
    assert plugin._evidence_is_relevant(
        "¿Cuál es el principal riesgo posoperatorio tras ritidectomía?",
        "El hematoma es la principal complicación posoperatoria temprana tras ritidectomía.",
        "Hematoma remains the most common early postoperative complication after facelift.",
    ) is True


def test_clinical_turn_without_retrieved_evidence_blocks_for_repair(plugin):
    assert plugin._on_pre_delivery_scope("¿Cuál es la tasa de seroma tras abdominoplastia?") is True
    plugin._on_pre_llm_call(
        session_id="clinical",
        user_message="¿Cuál es la tasa de seroma tras abdominoplastia?",
    )

    decision = plugin._on_pre_delivery(
        session_id="clinical",
        response_text="La tasa es específica, pero no tiene respaldo recuperado.",
    )

    assert decision["action"] == "block"
    assert "message" in decision and "reason" in decision
    assert "tasa" not in decision["message"].lower()


def test_unknown_clinical_state_fails_closed(plugin):
    decision = plugin._on_pre_delivery(
        session_id="missing",
        response_text="Se recomienda iniciar un tratamiento específico.",
    )

    assert decision["action"] == "block"


def test_supported_clinical_response_with_visible_ledger_is_allowed(plugin):
    session_id = "clinical-supported"
    claim = "La aspirina inhibe la agregación plaquetaria."
    plugin._on_pre_llm_call(
        session_id=session_id,
        user_message="¿Cómo actúa la aspirina sobre las plaquetas?",
    )
    plugin._on_post_tool_call(
        session_id=session_id,
        tool_name="mcp__consensus__search",
        status="ok",
        result=(
            '{"title":"Aspirin and platelets","abstract":"La aspirina inhibe la agregación '
            'plaquetaria.","doi":"10.1000/example"}'
        ),
    )
    response = (
        claim
        + "\n\nFuentes declaradas\n"
        + "Proveedores consultados: consensus\n"
        + "Limitaciones de recuperación: ninguna en esta prueba.\n"
        + "Estado de verificación: verificado contra la fuente recuperada."
    )

    assert plugin._on_pre_delivery(session_id=session_id, response_text=response) == {
        "action": "allow"
    }


def test_pre_delivery_repair_preserves_clinical_state_and_evidence(plugin):
    plugin._on_pre_llm_call(session_id="s", user_message="¿Cuál es la tasa de seroma tras abdominoplastia?")
    plugin._on_post_tool_call(
        session_id="s",
        tool_name="mcp__consensus__search",
        result='{"title":"Aspirin review","abstract":"Aspirin reduces platelet aggregation.","doi":"10.1000/example"}',
    )

    plugin._on_pre_llm_call(
        session_id="s",
        user_message="[PRE_DELIVERY_REPAIR]\nRepair the answer without unsupported claims.",
    )

    state = plugin._STATES[plugin._session_key("s")]
    assert state.clinical is True
    assert len(state.evidence) == 1
