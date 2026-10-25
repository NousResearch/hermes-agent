"""Zed reads its pickers only from ``configOptions`` once an agent sends them, so another surface's mode
or effort change must reach it as ``config_option_update``, not ``current_mode_update`` alone."""
import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
async def test_another_surfaces_mode_and_effort_change_resend_the_config_options(monkeypatch):
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', lambda info: None)
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    info = {'edit_approval_policy': 'ask', 'model': 'fixture', 'reasoning_effort': 'medium'}
    await agent._project_mode('s', 's', info)  # the attach's snapshot: its reply shows it
    await agent._project_mode('s', 's', {**info, 'edit_approval_policy': 'session'})
    await agent._project_mode('s', 's', {'reasoning_effort': 'high'})  # a partial session.info
    shown = [{option.category: option.current_value for option in call.kwargs['update'].config_options}
             for call in agent._conn.session_update.await_args_list
             if call.kwargs['update'].session_update == 'config_option_update']
    assert shown == [{'mode': 'session', 'thought_level': 'medium'}, {'mode': 'session', 'thought_level': 'high'}]


@pytest.mark.asyncio
async def test_a_projection_superseded_during_its_catalog_build_never_lands_after_the_newer_one(monkeypatch):
    """The pump's 'ask' projection is still building its model catalog when a snapshot adoption installs
    the newer 'session' policy: the editor's last config_option_update must be the newer one."""
    building, release = threading.Event(), threading.Event()

    def model_state(info):
        if not building.is_set():  # the pump's projection: held until the adoption has sent its own
            building.set()
            release.wait(30)
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', model_state)
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    info = {'edit_approval_policy': 'session', 'model': 'fixture', 'reasoning_effort': 'medium'}
    await agent._project_mode('s', 's', info)
    older = asyncio.create_task(agent._project_mode('s', 's', {**info, 'edit_approval_policy': 'ask'}))
    await asyncio.to_thread(building.wait, 30)
    await agent._adopt('s', {'info': info})
    release.set()
    await older
    modes = [{o.category: o.current_value for o in call.kwargs['update'].config_options}['mode']
             for call in agent._conn.session_update.await_args_list
             if call.kwargs['update'].session_update == 'config_option_update']
    assert modes[-1] == 'session', modes


@pytest.mark.asyncio
async def test_an_equal_adoption_during_the_catalog_build_does_not_swallow_the_update(monkeypatch):
    """Another surface switched the mode to Don't Ask; while that event's projection builds its model
    catalog, an adoption with the same settings lands (a typed /model's own refresh, say). The editor
    must still receive one config_option_update showing the new mode."""
    building, release = threading.Event(), threading.Event()

    def model_state(info):
        if not building.is_set():  # the event's projection: held until the adoption has run
            building.set()
            release.wait(30)
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', model_state)
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    info = {'edit_approval_policy': 'ask', 'model': 'fixture', 'reasoning_effort': 'medium'}
    await agent._project_mode('s', 's', info)  # the attach's snapshot
    changed = {**info, 'edit_approval_policy': 'session'}
    event = asyncio.create_task(agent._project_mode('s', 's', changed))
    await asyncio.to_thread(building.wait, 30)
    await agent._adopt('s', {'info': dict(changed)})
    release.set()
    await event
    modes = [{o.category: o.current_value for o in call.kwargs['update'].config_options}['mode']
             for call in agent._conn.session_update.await_args_list
             if call.kwargs['update'].session_update == 'config_option_update']
    assert modes == ['session'], modes
