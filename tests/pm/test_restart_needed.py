"""restart_needed / adopt_selected against real generations: real uv, offline wheels, temp HERMES_HOME."""
from __future__ import annotations

from tests.hermes_cli.plugin_worker_support import (
    boot as boot, isolated_python as isolated_python, plugin_world as plugin_world, publish_plugins)


def test_adopt_refuses_a_selected_generation_built_for_another_interpreter(tmp_path, monkeypatch):
    """adopt() is the live-process sibling of activate_dependencies' generation check: an
    installer can publish a generation built for another interpreter while this process is
    already running, and swapping sys.path onto it must be refused the same way, not just
    at the boot-time check activate_dependencies already has."""
    import os
    import sys
    from pm.environments import site_packages, venv_bin_dir
    from pm.environments_adopt import adopt

    previous, selected = tmp_path / "first" / "venv", tmp_path / "second" / "venv"
    for venv, cfg in ((previous, "home = test\n"), (selected, "home = test\nversion = 9.9.0\n")):
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(cfg)
        site_packages(venv).mkdir(parents=True)
        venv_bin_dir(venv).mkdir(parents=True)
    monkeypatch.setenv("PATH", str(venv_bin_dir(previous)))
    original_path = list(sys.path)
    assert adopt(previous, selected, previous) is False
    assert sys.path == original_path
    assert os.environ["PATH"] == str(venv_bin_dir(previous))


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
