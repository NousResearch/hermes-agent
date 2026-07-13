from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from gateway.config import Platform
from gateway.run import _sanitize_gateway_final_response


def test_provider_wait_interrupt_prefix_is_stable():
    assert (
        INTERRUPT_WAITING_FOR_MODEL_PREFIX
        == "Operation interrupted: waiting for model response ("
    )


def test_gateway_suppresses_provider_wait_interrupt_metadata_for_chat_surfaces():
    message = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}1.7s elapsed)."

    assert _sanitize_gateway_final_response(Platform.TELEGRAM, message) == ""


def test_gateway_preserves_provider_wait_interrupt_metadata_for_raw_surfaces():
    message = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}1.7s elapsed)."

    assert _sanitize_gateway_final_response("local", message) == message
