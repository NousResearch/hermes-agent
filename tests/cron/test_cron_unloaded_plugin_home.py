"""A saved home channel whose platform plugin is not loaded is reported, never skipped silently.

``deliver: all`` and the origin-less ``deliver: origin`` fallback only see platforms a loaded plugin
registers; a home channel saved for a platform whose plugin is not installed or not enabled (a
platform that left core, a disabled bundled one) was dropped without a word, so the run read as
fine while that chat got nothing.
"""

import pytest

from hermes_constants import get_hermes_home


@pytest.fixture()
def unloaded_home():
    (get_hermes_home() / "config.yaml").write_text(
        "platforms:\n"
        "  yuanbao:\n"
        "    enabled: true\n"
        "    home_channel: {platform: yuanbao, chat_id: 'group:555', name: yb home}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("deliver", ["all", "origin"])
def test_unloaded_plugin_home_is_reported_with_remedy(unloaded_home, deliver):
    from cron.scheduler_delivery import _deliver_result

    job = {"id": "j1", "name": "report", "deliver": deliver, "origin": None}
    error = _deliver_result(job, "hello")

    assert error is not None and "yuanbao home channel skipped" in error
    assert "hermes plugins install yuanbao" in error
