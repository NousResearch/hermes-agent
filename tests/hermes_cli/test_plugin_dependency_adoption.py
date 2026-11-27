"""A plugin whose Python dependencies PM published after this process booted loads without a restart.

A platform plugin with new dependencies (Feishu: ``lark-oapi``) is installed into a NEW dependency
generation: at gateway start (credentials present, plugin missing) or while the gateway runs (Desktop /
dashboard save, ``hermes plugins install``). Activation is a boot decision, so the gateway that did or saw
the install kept importing its boot generation: discovery logged "restart Hermes" and the adapter failed
with "lark-oapi is not installed" until somebody restarted it. PM already adopts a published generation
in place when nothing this process imported would change (``pm.environments_adopt``); discovery and the
platform dependency probe now use it. Real uv, offline wheels, temp HERMES_HOME.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil

from tests.hermes_cli.plugin_worker_support import (
    boot as boot, isolated_python as isolated_python, plugin_world as plugin_world, publish_plugins)  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def test_discovery_loads_a_plugin_whose_dependencies_were_published_after_boot(plugin_world, boot, caplog):
    from hermes_cli import plugins

    world = plugin_world
    boot(publish_plugins(world, {"base": ["plugin-proof-dep==1.0"]}))
    publish_plugins(world, {"base": ["plugin-proof-dep==1.0"], "adds": ["plugin-proof-other"]})
    plugins._reset_plugin_managers_for_tests()
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            plugins.discover_plugins(force=True)
        loaded = plugins.get_plugin_manager()._plugins["adds"]
    finally:
        plugins._reset_plugin_managers_for_tests()

    assert loaded.error is None and (world.home / "plugins" / "adds" / "imported").is_file(), loaded.error
    assert not [r.getMessage() for r in caplog.records if "restart" in r.getMessage()]


def test_a_platform_adapter_finds_dependencies_published_after_boot(plugin_world, boot):
    """The platform was loaded (its check imports lazily) before the generation carrying its SDK
    was published: building its adapter adopts that generation instead of reporting the SDK missing."""
    from gateway.platform_registry import PlatformEntry, PlatformRegistry

    world = plugin_world
    boot(publish_plugins(world, {"base": ["plugin-proof-dep==1.0"]}))
    registry = PlatformRegistry()
    registry.register(PlatformEntry(
        name="proofchat", label="Proof Chat", adapter_factory=lambda cfg: object(), source="plugin",
        check_fn=lambda: importlib.util.find_spec("plugin_proof_other") is not None))
    assert importlib.util.find_spec("plugin_proof_other") is None
    publish_plugins(world, {"base": ["plugin-proof-dep==1.0"], "adds": ["plugin-proof-other"]})

    assert registry.ensure_deps("proofchat") is True


def test_a_process_booted_from_the_installs_own_venv_adopts_the_first_generation(plugin_world, boot):
    """A container image runs from the venv baked into its tree until PM commits a generation (the
    first plugin install publishes one): that venv is the install's, so the process adopts the
    generation instead of asking for a container restart that would only boot into it."""
    from pm.environments import runtime_facts_path
    from pm.environments_adopt import adopt_published

    world = plugin_world
    first = publish_plugins(world, {"base": ["plugin-proof-dep==1.0"]})
    shipped = world.core / ".venv"
    shutil.copytree(first, shipped, symlinks=True)
    runtime_facts_path(world.core).unlink()  # nothing committed yet: the image's venv is what boots
    boot(shipped)
    publish_plugins(world, {"base": ["plugin-proof-dep==1.0"], "adds": ["plugin-proof-other"]})
    assert importlib.util.find_spec("plugin_proof_other") is None

    assert adopt_published(world.core) is True
    assert importlib.util.find_spec("plugin_proof_other") is not None
