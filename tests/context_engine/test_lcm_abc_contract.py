"""LCM contract drift checks against the current ContextEngine ABC."""

from __future__ import annotations

import inspect

from agent.context_engine import ContextEngine
from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


REQUIRED_TOKEN_STATE_ATTRS = (
    "last_prompt_tokens",
    "last_completion_tokens",
    "last_total_tokens",
    "threshold_tokens",
    "context_length",
    "compression_count",
)


def _close_engine(engine) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def test_lcm_engine_satisfies_current_context_engine_abstract_methods() -> None:
    missing = sorted(getattr(LCMEngine, "__abstractmethods__", set()))

    assert issubclass(LCMEngine, ContextEngine)
    assert missing == []
    for method_name in sorted(ContextEngine.__abstractmethods__):
        member = getattr(LCMEngine, method_name, None)
        if isinstance(member, property):
            assert member.fget is not None, f"LCMEngine missing property getter {method_name}"
            assert not getattr(member.fget, "__isabstractmethod__", False), method_name
        else:
            assert callable(member), f"LCMEngine missing abstract method {method_name}"
            assert not getattr(member, "__isabstractmethod__", False), method_name


def test_context_engine_token_state_contract_declares_required_attrs() -> None:
    for attr in REQUIRED_TOKEN_STATE_ATTRS:
        assert attr in vars(ContextEngine), f"ContextEngine token-state attr drifted: {attr}"


def test_lcm_engine_initializes_required_token_state_attrs(tmp_path) -> None:
    cfg = LCMConfig(database_path=str(tmp_path / "abc-contract.db"))
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        for attr in REQUIRED_TOKEN_STATE_ATTRS:
            assert attr in engine.__dict__, f"LCMEngine does not initialize {attr}"
            value = getattr(engine, attr)
            assert isinstance(value, int), f"{attr} should be int-compatible, got {type(value).__name__}"
    finally:
        _close_engine(engine)


def test_lcm_public_method_signatures_accept_current_abc_parameters() -> None:
    for method_name in sorted(ContextEngine.__abstractmethods__):
        abc_member = getattr(ContextEngine, method_name)
        lcm_member = getattr(LCMEngine, method_name)
        if isinstance(abc_member, property):
            assert isinstance(lcm_member, property), f"LCMEngine.{method_name} is no longer a property"
            abc_callable = abc_member.fget
            lcm_callable = lcm_member.fget
        else:
            abc_callable = abc_member
            lcm_callable = lcm_member
        assert abc_callable is not None
        assert lcm_callable is not None
        abc_signature = inspect.signature(abc_callable)
        lcm_signature = inspect.signature(lcm_callable)
        for param_name in abc_signature.parameters:
            assert param_name in lcm_signature.parameters, (
                f"LCMEngine.{method_name} missing ABC parameter {param_name}"
            )
