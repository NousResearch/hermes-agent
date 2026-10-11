"""Deferred confirmation and onboarding writes must belong to the routed profile."""

import asyncio
import queue
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

import gateway.run as run
import cli  # Import before per-test I/O guards; persistence below uses the real writer.
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


@pytest.fixture
def profile_homes(tmp_path, monkeypatch):
    root = tmp_path / "root"
    homes = [root / "profiles" / name for name in ("alpha", "beta")]
    for home in [root, *homes]:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("display:\n  tool_progress_command: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(run, "_hermes_home", root)
    return root, homes


def _read(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def test_busy_hint_persists_in_the_home_it_reads(profile_homes):
    from agent.onboarding import BUSY_INPUT_FLAG, busy_input_hint_gateway, is_seen
    from gateway.run import _profile_runtime_scope
    from gateway.run_busy import GatewayBusySessionMixin

    root, homes = profile_homes
    original_root = (root / "config.yaml").read_bytes()
    runner = GatewayBusySessionMixin()
    event = MessageEvent(text="follow up", source=SessionSource(platform=Platform.DISCORD, chat_id="chat"))
    for home in [homes[0], homes[1], homes[0]]:
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            was_seen = is_seen(_read(home), BUSY_INPUT_FLAG)
            message = runner._compose_busy_ack_message(
                event, 0, None, None, is_steer_mode=False, is_queue_mode=True,
                is_redirect_mode=False, demoted_for_subagents=False, demoted_for_compression=False,
            )
            assert (busy_input_hint_gateway("queue") in message) is not was_seen
        assert is_seen(_read(home), BUSY_INPUT_FLAG)
        assert (root / "config.yaml").read_bytes() == original_root
    assert all(is_seen(_read(home), BUSY_INPUT_FLAG) for home in homes)


def test_tool_progress_hint_persists_in_the_home_it_reads(profile_homes):
    from agent.onboarding import TOOL_PROGRESS_FLAG, is_seen
    from gateway.run import _profile_runtime_scope
    from gateway.run_turn_runner import TurnRunner

    root, homes = profile_homes
    original_root = (root / "config.yaml").read_bytes()
    for home in [homes[0], homes[1], homes[0]]:
        ctx = SimpleNamespace(
            _LONG_TOOL_THRESHOLD_S=10, progress_mode="all",
            long_tool_hint_fired=[False], progress_queue=queue.Queue(),
        )
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            was_seen = is_seen(_read(home), TOOL_PROGRESS_FLAG)
            TurnRunner(None, ctx)._progress_onboarding_hint({"duration": 15})
            assert ctx.progress_queue.empty() is was_seen
        assert is_seen(_read(home), TOOL_PROGRESS_FLAG)
        assert (root / "config.yaml").read_bytes() == original_root


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["new", "reset", "undo", "reload-mcp"])
async def test_always_confirmation_writes_only_registering_profile(profile_homes, command):
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret
    from tools import slash_confirm

    root, homes = profile_homes
    original_root = (root / "config.yaml").read_bytes()
    runner = object.__new__(run.GatewayRunner)
    runner._session_key_for_source = lambda source: f"agent:{source.profile}:discord:chat"
    runner._delivery_adapter_for = lambda source: None
    runner._thread_metadata_for_source = lambda *args: None
    runner._reply_anchor_for_event = lambda event: None
    runner._typed_command_prefix_for = lambda platform: "/"
    observations = []

    async def execute(event=None):
        observations.append((get_hermes_home(), get_secret("PROFILE_TEST_TOKEN")))
        await asyncio.sleep(0)
        observations.append((get_hermes_home(), get_secret("PROFILE_TEST_TOKEN")))
        return "executed"

    runner._execute_mcp_reload = execute
    flag = "mcp_reload_confirm" if command == "reload-mcp" else "destructive_slash_confirm"
    for home in [homes[0], homes[1], homes[0]]:
        # Force a fresh prompt on the second alpha visit, verifying persistence in A→B→A.
        with run._profile_runtime_scope(home, prepared_secret_scope={"PROFILE_TEST_TOKEN": home.name}):
            assert cli.save_config_value(f"approvals.{flag}", True)
            source = SessionSource(platform=Platform.DISCORD, chat_id="chat", profile=home.name)
            event = MessageEvent(text=f"/{command}", source=source)
            if command == "reload-mcp":
                await runner._handle_reload_mcp_command(event)
            else:
                await runner._maybe_confirm_destructive_slash(
                    event=event, command=command, title=command, detail="test", execute=execute,
                )
            key = runner._session_key_for_source(source)
            pending = slash_confirm.get_pending(key)
            assert pending is not None
        other = homes[1] if home == homes[0] else homes[0]
        untouched = (other / "config.yaml").read_bytes()
        # Discord buttons can resolve in the root or even another profile's task.
        resolver_home = root if home == homes[0] else other
        with run._profile_runtime_scope(resolver_home, prepared_secret_scope={"PROFILE_TEST_TOKEN": "wrong"}):
            reply = await slash_confirm.resolve(key, pending["confirm_id"], "always")
            assert get_hermes_home() == resolver_home
            assert get_secret("PROFILE_TEST_TOKEN") == "wrong"
        assert isinstance(reply, str) and "executed" in reply
        assert observations[-2:] == [(home, home.name), (home, home.name)]
        assert _read(home)["approvals"][flag] is False
        assert (other / "config.yaml").read_bytes() == untouched
        assert (root / "config.yaml").read_bytes() == original_root
        assert await slash_confirm.resolve(key, pending["confirm_id"], "always") is None
