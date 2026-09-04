"""Only known bundled picker owners may render the Home consent prompt."""

import pytest

from gateway.group_home_consent import _verified_picker


@pytest.mark.parametrize('platform', ['telegram', 'discord'])
@pytest.mark.parametrize('owner', ['adapter', 'adapter_prompts'])
@pytest.mark.parametrize('prefix', ['plugins.platforms.', 'hermes_plugins.platforms__'])
def test_bundled_picker_defining_owner_is_recognized(platform, owner, prefix):
    async def send_choice_picker(self):
        pass

    send_choice_picker.__module__ = prefix + platform + '.' + owner
    adapter = type('Picker', (), {'send_choice_picker': send_choice_picker})()
    assert _verified_picker(adapter)
    if prefix.startswith('hermes_plugins'):
        send_choice_picker.__module__ = prefix + platform + '__home_012345abcdef.' + owner
        assert _verified_picker(adapter)


@pytest.mark.parametrize('module', [
    'plugins.platforms.matrix.adapter_prompts', 'other.adapter_prompts',
    'hermes_plugins.platforms__telegram_fake.adapter_prompts',
    'hermes_plugins.platforms__telegram__home_bad.adapter_prompts',
    'hermes_plugins.platforms__telegram.adapter_prompts.extra',
    'hermes_plugins.platforms__telegram.adapter_prompts\n',
])
def test_unrecognized_owner_does_not_gain_consent_capability(module):
    async def send_choice_picker(self):
        pass

    send_choice_picker.__module__ = module
    assert not _verified_picker(type('Picker', (), {'send_choice_picker': send_choice_picker})())
