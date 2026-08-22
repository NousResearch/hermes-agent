import pytest

from hermes_cli.plugins import PluginManager


def _boom(**kwargs):
    raise RuntimeError("transport safety failed")


def test_fail_closed_hook_surfaces_callback_failure_as_skip():
    manager = PluginManager()
    manager._hooks["pre_gateway_transport"] = [_boom]

    assert manager.invoke_hook(
        "pre_gateway_transport", fail_closed=True
    ) == [{"action": "skip", "reason": "hook-callback-error"}]


def test_legacy_hook_callback_failure_remains_isolated():
    manager = PluginManager()
    manager._hooks["pre_gateway_dispatch"] = [_boom]

    assert manager.invoke_hook("pre_gateway_dispatch") == []


def test_required_text_hooks_chain_in_registration_order():
    manager = PluginManager()
    seen = []

    def _first(**kwargs):
        seen.append(kwargs["response_text"])
        return kwargs["response_text"] + " first"

    def _second(**kwargs):
        seen.append(kwargs["response_text"])
        return kwargs["response_text"] + " second"

    manager._hooks["finalize_llm_output"] = [_first, _second]

    assert manager.invoke_text_hook(
        "finalize_llm_output", response_text="original", platform="telegram"
    ) == ("original first second", True)
    assert seen == ["original", "original first"]


def test_required_text_hook_callback_failure_is_raised():
    manager = PluginManager()
    manager._hooks["finalize_llm_output"] = [_boom]

    with pytest.raises(RuntimeError, match="transport safety failed"):
        manager.invoke_text_hook(
            "finalize_llm_output", response_text="must not leak", platform="telegram"
        )


def test_terminal_validation_hooks_cannot_modify_output():
    manager = PluginManager()
    manager._hooks["validate_llm_output"] = [lambda **kwargs: "unsafe append"]

    with pytest.raises(TypeError, match="must return None or True"):
        manager.invoke_validation_hook(
            "validate_llm_output", response_text="safe", platform="telegram"
        )
