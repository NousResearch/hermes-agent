"""Spanish action intentions preserve the existing bounded anti-stall contract."""
from types import SimpleNamespace

import pytest

from agent.agent_runtime_intent import (
    looks_like_codex_intermediate_ack,
    promoted_reasoning_announces_action,
    trailing_continue_intent,
)


def ack(text, *, workspace=False, tool_result=False, user="continúa"):
    messages = [{"role": "user", "content": user}]
    if tool_result:
        messages.append({"role": "tool", "content": "ok", "tool_call_id": "t"})
    return looks_like_codex_intermediate_ack(
        SimpleNamespace(_strip_think_blocks=lambda value: value), user, text,
        messages, require_workspace=workspace,
    )


@pytest.mark.parametrize("text", [
    "Voy a lanzar un agente.", "Vamos a revisar los archivos.",
    "Ahora voy a ejecutar las pruebas.", "Déjame revisar el repositorio.",
    "Dejame comprobar el servicio.", "Déjanos buscar el archivo.",
    "Necesito inspeccionar el directorio.", "Tengo que abrir la carpeta.",
    "Primero, voy a analizar el proyecto.", "Listo. Voy a depurar el servicio.",
    "VOY A PROBAR EL SERVICIO.",
])
def test_spanish_first_person_intentions(text):
    assert ack(text)
    assert trailing_continue_intent(text)
    assert promoted_reasoning_announces_action(text)


@pytest.mark.parametrize("text", [
    "Lanzando el agente ahora.", "Relanzando un agente ahora.",
    "Iniciando el proceso ahora.", "Reiniciando el servicio ahora.",
    "Creando la sesión ahora.", "Creando la sesion ahora.",
    "Ejecutando las pruebas ahora.", "Corriendo los tests ahora.",
    "Revisando el registro ahora.", "Comprobando el estado ahora.",
    "Verificando el servicio ahora.", "Leyendo la salida ahora.",
    "Abriendo el archivo ahora.", "Inspeccionando el repositorio ahora.",
    "Listo. Lanzando el agente ahora.",
])
def test_spanish_pronounless_actions_have_real_transition_paths(text):
    assert ack(text)


@pytest.mark.parametrize("text", [
    "No voy a lanzar un agente.", "Nunca voy a ejecutar las pruebas.",
    "Jamás vamos a revisar los archivos.", "¿Voy a revisar el repositorio?",
    "Voy a revisar el repositorio?", "¿Lanzando el agente ahora?",
    "Sí, ya lancé un agente.", "Ya lance un agente.",
    "Ya revisé los archivos.", "El repositorio está revisado.",
    "Revisando el registro se entiende el fallo.",
    "Lanzando el agente.", "Voy a Madrid mañana.",
    "La frase voy a revisar indica futuro.",
    "Opciones:\n- Voy a revisar el repositorio.\n- Ejecutando las pruebas ahora.",
    "1. Voy a revisar los archivos.\n2. Lanzando el agente ahora.",
])
def test_spanish_non_intentions_stay_final(text):
    assert not ack(text)
    assert not trailing_continue_intent(text)
    assert not promoted_reasoning_announces_action(text)


def test_workspace_and_tool_result_gates_are_preserved():
    text = "Voy a lanzar un agente."
    assert not ack(text, workspace=True)
    assert ack(text, workspace=True, user="revisa la carpeta")
    assert ack("Voy a revisar los archivos.", workspace=True)
    assert not ack(text, tool_result=True)
    assert not ack("Lanzando el agente ahora.", tool_result=True)
    assert not ack(text + "x" * 1200)


def test_tail_only_and_short_visible_answer_limits():
    text = "Voy a revisar los archivos. La respuesta es correcta."
    assert not trailing_continue_intent(text)
    assert not promoted_reasoning_announces_action(text)
    long_plan = "Contexto. " * 60 + "Ahora voy a revisar los archivos."
    assert not trailing_continue_intent(long_plan)
    assert promoted_reasoning_announces_action(long_plan)


def test_original_english_reproductions_are_unchanged():
    assert ack("Launching the agent now.")
    assert ack("I will now check the files.")
    assert trailing_continue_intent("I will now check the files.")
    assert promoted_reasoning_announces_action("I will now check the files.")
    assert not ack("I already launched an agent.")
