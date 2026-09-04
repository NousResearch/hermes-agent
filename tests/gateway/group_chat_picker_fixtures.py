"""Capture the core picker contract without exercising a native adapter."""

import pytest

from tests.gateway.test_group_home_consent import home
from tests.gateway.test_hosted_room_messaging import _PickerAdapter


class CorePicker(_PickerAdapter):
    supports_choice_pages = True

    def __init__(self, config):
        super().__init__()
        self.config = config


@pytest.fixture
def picker_home(home):
    adapter = CorePicker(home.runner.config.platforms[home.event.source.platform])
    home.runner.adapters[home.event.source.platform] = adapter
    home.adapter = adapter
    return home


async def choose_first(state):
    call = state.adapter.calls[-1]
    return await call['on_choice_selected'](
        state.event.source.chat_id, call['choices'][0]['value']
    )
