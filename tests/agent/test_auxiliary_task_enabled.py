"""Tests for the first-class ``auxiliary.<task>.enabled`` config knob.

Setting ``auxiliary.<task>.enabled: false`` in config.yaml must disable the
task cleanly — no provider resolution, no network call — instead of relying
on workarounds like ``provider: none`` (which only works by accident of the
resolution error path).
"""

from unittest.mock import patch

import pytest

from agent.auxiliary_client import (
    AuxiliaryTaskDisabled,
    async_call_llm,
    call_llm,
)

_MESSAGES = [{"role": "user", "content": "hi"}]


def test_call_llm_disabled_task_raises_without_resolving_provider():
    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"enabled": False},
    ), patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        side_effect=AssertionError("provider resolution must not run for a disabled task"),
    ):
        with pytest.raises(AuxiliaryTaskDisabled, match="title_generation"):
            call_llm(task="title_generation", messages=_MESSAGES)


@pytest.mark.asyncio
async def test_async_call_llm_disabled_task_raises_without_resolving_provider():
    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"enabled": False},
    ), patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        side_effect=AssertionError("provider resolution must not run for a disabled task"),
    ):
        with pytest.raises(AuxiliaryTaskDisabled, match="title_generation"):
            await async_call_llm(task="title_generation", messages=_MESSAGES)


@pytest.mark.parametrize("cfg", [{}, {"enabled": True}, {"enabled": "false"}])
def test_call_llm_enabled_or_absent_proceeds_to_resolution(cfg):
    """Anything other than a literal False proceeds (absent = enabled;
    non-bool values are ignored rather than guessed)."""
    sentinel = RuntimeError("resolution reached")
    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=cfg,
    ), patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        side_effect=sentinel,
    ):
        with pytest.raises(RuntimeError, match="resolution reached"):
            call_llm(task="title_generation", messages=_MESSAGES)


def test_disabled_task_error_is_runtimeerror_subclass():
    """Existing callers catch broad Exception/RuntimeError — the new type
    must stay inside that hierarchy so disabling a task degrades cleanly."""
    assert issubclass(AuxiliaryTaskDisabled, RuntimeError)
