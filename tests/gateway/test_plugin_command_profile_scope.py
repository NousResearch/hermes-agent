"""Plugin slash command dispatch must resolve and run under the chat's own profile (#133244).

Under ``gateway.multiplex_profiles`` the plugin command table follows ``get_hermes_home()``:
a bare lookup consults the launch profile's registry, so a plugin enabled only there fires
in another profile's chat and its handler runs with no secret scope — every credential
read fails closed with ``UnscopedSecretError``. The dispatch must scope itself from the
source (like the agent-turn path) instead of trusting the caller to have wrapped it.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture
def multiplex_homes(tmp_path, monkeypatch):
    """Launch home with one user plugin enabled, plus a live secondary profile without it."""
    launch = tmp_path / "hermes"
    launch.mkdir()
    (launch / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - probe-plugin\n", encoding="utf-8"
    )

    plugin_dir = launch / "plugins" / "probe-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: probe-plugin\nversion: 1.0.0\ndescription: probe\nauthor: probe\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def _handler(args):\n"
        "        import hermes_constants\n"
        "        return 'probe-ok@' + hermes_constants.get_hermes_home().name + ':' + args\n"
        "    ctx.register_command('probe', handler=_handler, description='probe')\n",
        encoding="utf-8",
    )

    secondary = launch / "profiles" / "infra"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text(
        "plugins:\n  enabled: []\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(launch))

    from hermes_cli import plugins as plugins_mod

    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load(force=True)
    assert "probe" in manager._plugin_commands, "probe plugin must register its command"

    yield {"launch": launch, "secondary": secondary}

    plugins_mod._reset_plugin_managers_for_tests()


def _multiplex_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.config.multiplex_profiles = True
    runner._draining = False
    return runner


def _probe_event(source_profile: str = "") -> tuple[MessageEvent, SessionSource]:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="c1",
        user_id="u1",
        user_name="tester",
        chat_type="dm",
        profile=source_profile,
    )
    return MessageEvent(text="/probe hi", source=source, message_id="m1"), source


@pytest.mark.asyncio
async def test_plugin_command_of_other_profile_is_not_dispatched(multiplex_homes):
    """A command registered only in the launch profile's registry must not fire in the
    secondary profile's chat: the lookup happens under the secondary's scope and returns
    None, so the command falls through to the unknown-command path instead of running a
    foreign plugin with no credential scope."""
    runner = _multiplex_runner()
    event, source = _probe_event(source_profile="infra")

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, source, "probe"
    )

    assert handled is False
    assert result is None
    assert command == "probe"


@pytest.mark.asyncio
async def test_plugin_command_runs_scoped_to_the_chat_profile(multiplex_homes):
    """In the launch profile's own chat the command still dispatches, and the handler sees
    the chat profile's HERMES_HOME (not an unscoped ambient home)."""
    runner = _multiplex_runner()
    event, source = _probe_event(source_profile="default")

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, source, "probe"
    )

    assert handled is True
    assert result == "probe-ok@hermes:hi"
    assert command == "probe"


@pytest.mark.asyncio
async def test_plugin_command_unscoped_entry_point_stays_profile_scoped(
    multiplex_homes,
):
    """The exact #133244 shape: the dispatch is reached with NO ambient profile scope
    (an entry point that calls the runner directly, e.g. interaction passthrough). The
    lookup must still consult the chat's own profile — not fall back to the launch
    profile's registry through the ambient HERMES_HOME."""
    from hermes_constants import get_hermes_home_override

    assert get_hermes_home_override() is None, (
        "test must run without an ambient override"
    )

    runner = _multiplex_runner()
    event, source = _probe_event(source_profile="infra")

    handled, _result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, source, "probe"
    )

    assert handled is False
