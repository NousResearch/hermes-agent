"""restart_needed / adopt_selected against real generations: real uv, offline wheels, temp HERMES_HOME."""
from __future__ import annotations

from tests.hermes_cli.plugin_worker_support import (
    boot as boot, isolated_python as isolated_python, plugin_world as plugin_world, publish_plugins)


def test_a_process_on_a_superseded_generation_needs_a_restart_until_it_adopts(plugin_world, boot):
    from pm.environments_adopt import adopt_selected, restart_needed

    world = plugin_world
    first = publish_plugins(world, {"base": ["plugin-proof-dep==1.0"]})
    # This interpreter never booted from the install: nothing it could restart into.
    assert restart_needed(world.core) is None
    assert adopt_selected(world.core)
    boot(first)
    assert restart_needed(world.core) is None
    second = publish_plugins(world, {"base": ["plugin-proof-dep==1.0"], "adds": ["plugin-proof-other"]})
    assert second != first
    reason = restart_needed(world.core)
    assert reason and second.parent.name in reason
    assert adopt_selected(world.core)
    assert restart_needed(world.core) is None
