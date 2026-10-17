"""Global emergency stop (`hermes pause` / `hermes resume`) — agent/estop.py.

The ESTOP sentinel is a resumable pause for NEW work only: cron dispatch,
kanban dispatch, and new gateway turns are halted while it is engaged; work
already in flight is never touched. Removing the sentinel (`hermes resume`)
restores normal operation with no restart.

Ported from: gastownhall/gastown estop.go (MIT); related prior art: #26778
(/panic — kill/exit semantics, deliberately different) and #44617
(interrupt in-flight cron — deliberately NOT done here).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from agent import estop


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and reset estop module log state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    estop._logged_components.clear()
    return tmp_path


# ── sentinel create / remove ────────────────────────────────────────────────


def test_engage_creates_sentinel_and_is_engaged(hermes_home):
    assert estop.is_engaged() is False
    estop.engage()
    assert (hermes_home / "ESTOP").exists()
    assert estop.is_engaged() is True


def test_disengage_removes_sentinel(hermes_home):
    estop.engage()
    assert estop.disengage() is True
    assert not (hermes_home / "ESTOP").exists()
    assert estop.is_engaged() is False
    # Disengaging when not engaged is a no-op that reports False.
    assert estop.disengage() is False


def test_reason_and_timestamp_stored(hermes_home):
    estop.engage(reason="runaway cron fan-out")
    state = estop.get_state()
    assert state is not None
    assert state["reason"] == "runaway cron fan-out"
    assert state["engaged_at"]  # ISO timestamp string

    raw = json.loads((hermes_home / "ESTOP").read_text(encoding="utf-8"))
    assert raw["reason"] == "runaway cron fan-out"




def test_corrupt_sentinel_still_engages(hermes_home):
    """A hand-touched/corrupt ESTOP file must still pause (fail safe)."""
    (hermes_home / "ESTOP").write_text("not json", encoding="utf-8")
    assert estop.is_engaged() is True
    state = estop.get_state()
    assert state is not None
    assert state.get("reason") is None


# ── paused notice for new gateway turns ─────────────────────────────────────




def test_paused_reply_surfaces_reason_and_resume_hint(hermes_home):
    estop.engage(reason="deploy window")
    notice = estop.paused_reply()
    assert notice is not None
    assert "paused" in notice.lower()
    assert "deploy window" in notice




# ── check_paused: cheap gate + log-once ─────────────────────────────────────




# ── cron scheduler integration ──────────────────────────────────────────────


def test_cron_tick_skips_dispatch_when_engaged(hermes_home, monkeypatch):
    from cron import scheduler

    calls = []

    def _fake_get_due_jobs():
        calls.append(1)
        return []

    monkeypatch.setattr(scheduler, "get_due_jobs", _fake_get_due_jobs)

    estop.engage(reason="test")
    assert scheduler.tick(verbose=False) == 0
    assert calls == [], "engaged ESTOP must skip the due-job scan entirely"


def test_cron_tick_resumes_after_disengage(hermes_home, monkeypatch):
    from cron import scheduler

    calls = []

    def _fake_get_due_jobs():
        calls.append(1)
        return []

    monkeypatch.setattr(scheduler, "get_due_jobs", _fake_get_due_jobs)

    estop.engage()
    scheduler.tick(verbose=False)
    assert calls == []

    estop.disengage()
    scheduler.tick(verbose=False)
    assert calls == [1], "resume must restore normal cron dispatch"


# ── kanban dispatcher integration ───────────────────────────────────────────


def test_kanban_dispatch_blocked_when_engaged(hermes_home):
    from gateway.kanban_watchers_common import _kanban_dispatch_allowed

    assert _kanban_dispatch_allowed() is True
    estop.engage(reason="test")
    assert _kanban_dispatch_allowed() is False
    estop.disengage()
    assert _kanban_dispatch_allowed() is True


# ── gateway turn-start integration ──────────────────────────────────────────


class _FakeSource:
    platform = None
    chat_id = "c1"
    user_id = "u1"
    user_name = "user"
    chat_type = "dm"
    profile = None


class _FakeEvent:
    internal = False
    text = "hello"

    def __init__(self):
        self.source = _FakeSource()

    def get_command(self) -> str | None:
        return None


@pytest.mark.asyncio
async def test_gateway_new_turn_gets_paused_reply(hermes_home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized = lambda source: True  # bare-instance stub
    estop.engage(reason="maintenance")
    reply = await runner._handle_message(_FakeEvent())
    assert reply is not None
    assert "paused" in reply.lower()
    assert "maintenance" in reply


@pytest.mark.asyncio
async def test_gateway_internal_events_bypass_estop(hermes_home):
    """Internal events (in-flight work completions) must NOT be paused."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    estop.engage()
    event = _FakeEvent()
    event.internal = True
    # An internal event proceeds past the estop gate; the bare runner then
    # blows up further down the pipeline on missing attributes — that error
    # (anything but a paused reply) proves the gate let it through.
    try:
        reply = await runner._handle_message(event)
    except Exception:
        return
    assert reply is None or "paused" not in (reply or "").lower()


# ── CLI: hermes pause / hermes resume ───────────────────────────────────────


def test_cli_pause_engages_with_reason(hermes_home, capsys):
    from hermes_cli.subcommands.pause import cmd_pause

    rc = cmd_pause(argparse.Namespace(reason="ops incident"))
    assert rc == 0
    assert estop.is_engaged() is True
    assert estop.get_state()["reason"] == "ops incident"
    assert "paused" in capsys.readouterr().out.lower()


def test_cli_pause_idempotent(hermes_home, capsys):
    from hermes_cli.subcommands.pause import cmd_pause

    assert cmd_pause(argparse.Namespace(reason=None)) == 0
    assert cmd_pause(argparse.Namespace(reason=None)) == 0
    assert estop.is_engaged() is True


def test_cli_resume_disengages(hermes_home, capsys):
    from hermes_cli.subcommands.pause import cmd_pause, cmd_resume

    cmd_pause(argparse.Namespace(reason=None))
    rc = cmd_resume(argparse.Namespace())
    assert rc == 0
    assert estop.is_engaged() is False
    assert "resumed" in capsys.readouterr().out.lower()






# ── hermes status surfacing ─────────────────────────────────────────────────


def test_status_line_when_paused(hermes_home):
    from hermes_cli.status import _estop_status_line

    assert _estop_status_line() is None
    estop.engage(reason="ops")
    line = _estop_status_line()
    assert line is not None
    assert "paused" in line.lower()
    assert "ops" in line
    estop.disengage()
    assert _estop_status_line() is None


# ── post-merge audit fixes (#81148 follow-up) ───────────────────────────────


def test_is_engaged_fails_safe_on_stat_error(hermes_home, monkeypatch):
    """A stat failure must report ENGAGED (fail safe) — the pause has to
    hold even when HERMES_HOME is misbehaving, matching the module's
    corrupt-sentinel doctrine."""
    class _BoomPath:
        def exists(self):
            raise OSError("permission denied")

    monkeypatch.setattr(estop, "sentinel_path", lambda: _BoomPath())
    assert estop.is_engaged() is True


class _FakeCmdEvent(_FakeEvent):
    text = "/status"

    def get_command(self):
        return "status"

    def get_command_args(self):
        return ""


@pytest.mark.asyncio
async def test_gateway_slash_commands_bypass_estop(hermes_home):
    """Recognized slash commands must pass the estop gate — /pause off is
    the in-band resume path for messaging-only users, and /status, /help
    and friends must keep working while paused."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized = lambda source: True
    estop.engage(reason="maintenance")
    # The command proceeds past the estop gate; the bare runner then blows
    # up further down on missing attributes — anything but the paused
    # notice proves the gate let it through.
    try:
        reply = await runner._handle_message(_FakeCmdEvent())
    except Exception:
        return
    assert reply is None or "hermes is paused" not in (reply or "").lower()


class _FakePauseEvent(_FakeEvent):
    def __init__(self, args=""):
        super().__init__()
        self._args = args
        self.text = f"/pause {args}".strip()

    def get_command(self):
        return "pause"

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_gateway_pause_command_engages_and_resumes(hermes_home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)

    reply = await runner._handle_pause_command(_FakePauseEvent("deploy window"))
    assert "paused" in reply.lower()
    assert estop.is_engaged() is True
    assert estop.get_state()["reason"] == "deploy window"

    # Re-issuing without args reports already-paused instead of clobbering.
    reply = await runner._handle_pause_command(_FakePauseEvent(""))
    assert "already paused" in reply.lower()

    reply = await runner._handle_pause_command(_FakePauseEvent("off"))
    assert "resumed" in reply.lower()
    assert estop.is_engaged() is False

    reply = await runner._handle_pause_command(_FakePauseEvent("off"))
    assert "wasn't paused" in reply.lower()


def test_pause_command_registered_for_gateway():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    cmd = resolve_command("pause")
    assert cmd is not None and cmd.name == "pause"
    assert "pause" in GATEWAY_KNOWN_COMMANDS
    # Must be dispatchable while an agent is running (in-band emergency stop).
    assert cmd.busy_policy == "dispatch"


def test_profile_gateway_honors_canonical_root_estop(tmp_path, monkeypatch):
    """fleet-analyst-class: HERMES_HOME is a profile dir; pause lives at root.

    A process launched with HERMES_HOME=~/.hermes/profiles/fleet-analyst must
    still treat ~/.hermes/ESTOP as engaged. Otherwise `hermes pause` is not
    a global emergency stop (t_7b65ff88).
    """
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "fleet-analyst"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    estop._logged_components.clear()

    assert estop.is_engaged() is False
    (root / "ESTOP").write_text("{\"reason\": \"thundering herd\"}\n", encoding="utf-8")
    assert estop.is_engaged() is True
    assert estop.paused_reply() is not None
    assert "paused" in estop.paused_reply().lower()
    # Profile-local engage still works and is independent.
    estop.engage(reason="local")
    assert (profile / "ESTOP").exists()
    assert estop.is_engaged() is True
    (root / "ESTOP").unlink()
    assert estop.is_engaged() is True  # still held by profile sentinel
    estop.disengage()
    assert estop.is_engaged() is False


# ── authority: only an operator may engage or lift the stop ─────────────────
#
# An agent that runs `hermes pause` from its terminal tool halts every profile: engage()
# writes the caller's home (the fleet root for the default profile, and for any remote
# backend whose shell lands there) and disengage() removes the fleet-root sentinel too,
# so an agent that can call either can stop or silently un-stop every profile.

# One marker per kind of agent/automation context; each alone must refuse.
_AGENT_CONTEXT_MARKERS = [
    ("HERMES_CRON_SESSION", "1"),               # cron job
    ("HERMES_KANBAN_TASK", "t_0000"),           # kanban worker
    ("HERMES_SINGLE_QUERY_SESSION", "1"),       # hermes chat -q
    ("HERMES_SESSION_PLATFORM", "telegram"),    # gateway turn / child of one
    ("HERMES_SESSION_ID", "20260101_000000_ab"),  # any agent session / child of one
    ("HERMES_GATEWAY_SESSION", "1"),            # TUI gateway
    ("HERMES_AGENT", "true"),                   # any process a Hermes agent started
]

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture
def operator_env(monkeypatch):
    """A plain operator shell: no agent marker, and no `hermes` entry point has run here."""
    import hermes_constants

    # raising=False: on a base without the marker the tests still run and fail on their assertions.
    monkeypatch.setattr(hermes_constants, "inherited_agent_marker", None, raising=False)
    return monkeypatch


def _cli(*args):
    return [sys.executable, "-m", "hermes_cli.main", *args]


@pytest.mark.parametrize("name,value", _AGENT_CONTEXT_MARKERS)
def test_agent_context_cannot_engage(hermes_home, operator_env, name, value):
    operator_env.setenv(name, value)
    with pytest.raises(estop.EstopRefused) as exc:
        estop.engage(reason="agent decided")
    assert not (hermes_home / "ESTOP").exists()
    assert estop.is_engaged() is False
    # The refusal must not hand the caller a way around it.
    assert "HERMES_" not in str(exc.value)


@pytest.mark.parametrize("name,value", _AGENT_CONTEXT_MARKERS)
def test_agent_context_cannot_lift_operator_stop(hermes_home, operator_env, name, value):
    estop.engage(reason="operator")
    operator_env.setenv(name, value)
    with pytest.raises(estop.EstopRefused) as exc:
        estop.disengage()
    assert (hermes_home / "ESTOP").exists()
    assert "/pause off" in str(exc.value)  # the chat command that lifts it, not /pause


def test_operator_can_engage_and_lift(hermes_home, operator_env):
    estop.engage(reason="operator")
    assert estop.is_engaged() is True
    assert estop.disengage() is True
    assert estop.is_engaged() is False


def test_hermes_that_set_its_own_agent_marker_is_an_operator(hermes_home, operator_env):
    """`hermes pause` from a plain shell: the entry point advertised HERMES_AGENT itself."""
    import hermes_constants

    operator_env.setenv("HERMES_AGENT", "true")
    operator_env.setattr(hermes_constants, "inherited_agent_marker", False)
    estop.engage(reason="operator")
    assert estop.disengage() is True
    operator_env.setattr(hermes_constants, "inherited_agent_marker", True)
    with pytest.raises(estop.EstopRefused):
        estop.engage(reason="agent decided")


def test_profile_agent_cannot_lift_fleet_root_stop(tmp_path, operator_env):
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    operator_env.setenv("HERMES_HOME", str(root))
    estop.engage(reason="fleet halt")
    operator_env.setenv("HERMES_HOME", str(profile))
    operator_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    with pytest.raises(estop.EstopRefused):
        estop.disengage()
    assert (root / "ESTOP").exists()
    assert estop.is_engaged() is True


def test_unclassifiable_context_fails_closed(hermes_home, operator_env):
    from tools import approval_context

    def _boom():
        raise RuntimeError("session context unavailable")

    operator_env.setattr(approval_context, "_is_cron_approval_context", _boom)
    with pytest.raises(estop.EstopRefused):
        estop.engage()
    assert not (hermes_home / "ESTOP").exists()


def test_agent_terminal_tool_cannot_pause_end_to_end(hermes_home, operator_env):
    """`hermes pause` run for real through the agent's terminal tool is refused."""
    from tools.environments.local import LocalEnvironment

    # cwd = checkout so ``-m`` resolves even where the terminal tool strips Hermes' PYTHONPATH.
    env = LocalEnvironment(cwd=_REPO_ROOT, timeout=120)
    try:
        result = env.execute(shlex.join(_cli("pause", "--reason", "agent decided")), timeout=120)
    finally:
        env.cleanup()
    assert not (hermes_home / "ESTOP").exists(), result
    assert result["returncode"] == 1, result
    assert "refusing" in result["output"].lower()


def _execute_code_env(hermes_home):
    from tools.code_execution_env import _build_child_env

    env = _build_child_env(rpc_endpoint="unused", rpc_token="unused", tmpdir=str(hermes_home),
                           child_python=sys.executable)
    return {**env, "PYTHONPATH": _REPO_ROOT}


def _mcp_stdio_env(hermes_home):
    from tools.mcp_tool_config import _build_safe_env

    return {**_build_safe_env({"HERMES_HOME": str(hermes_home)}), "PYTHONPATH": _REPO_ROOT}


@pytest.mark.parametrize("build_env", [_execute_code_env, _mcp_stdio_env], ids=["execute_code", "mcp_stdio"])
def test_scrubbed_agent_child_cannot_pause_end_to_end(hermes_home, operator_env, build_env):
    """The execute_code sandbox and stdio MCP servers get an allowlisted env and are marked as
    agent-started even when the parent never advertised HERMES_AGENT (e.g. the ACP entry point),
    so `hermes pause` started from them is refused."""
    env = build_env(hermes_home)
    assert env.get("HERMES_HOME") == str(hermes_home)  # never the real ~/.hermes
    paused = subprocess.run(_cli("pause"), env=env, capture_output=True, text=True, timeout=120,
                            cwd=_REPO_ROOT, stdin=subprocess.DEVNULL)
    assert paused.returncode == 1, paused.stdout + paused.stderr
    assert "refusing" in paused.stderr.lower()
    assert not (hermes_home / "ESTOP").exists()


def test_cli_bang_command_is_the_operators(hermes_home, operator_env):
    """`!hermes pause` typed in the interactive CLI after its agent started a session."""
    from hermes_cli.bang_shell import _bang_env
    from hermes_cli.main import _advertise_agent_env

    for name in ("AI_AGENT", "HERMES_AGENT"):
        operator_env.setenv(name, "x")  # recorded, so monkeypatch restores it at teardown
        operator_env.delenv(name)
    _advertise_agent_env()  # the operator's `hermes` CLI
    # Once its agent exists the CLI mirrors the session id into os.environ (set_current_session_id).
    operator_env.setenv("HERMES_SESSION_ID", "20260101_000000_ab")
    env = {**_bang_env(), "PYTHONPATH": _REPO_ROOT}
    paused = subprocess.run(_cli("pause"), env=env, capture_output=True, text=True, timeout=120,
                            cwd=_REPO_ROOT, stdin=subprocess.DEVNULL)
    assert paused.returncode == 0, paused.stdout + paused.stderr
    assert (hermes_home / "ESTOP").exists()


def test_operator_cli_can_pause_and_resume_end_to_end(hermes_home, operator_env):
    env = {k: v for k, v in os.environ.items() if k != "HERMES_AGENT"}
    paused = subprocess.run(_cli("pause", "--reason", "ops"), env=env, capture_output=True,
                            text=True, timeout=120, cwd=_REPO_ROOT, stdin=subprocess.DEVNULL)
    assert paused.returncode == 0, paused.stdout + paused.stderr
    assert (hermes_home / "ESTOP").exists()
    resumed = subprocess.run(_cli("resume"), env=env, capture_output=True, text=True, timeout=120,
                             cwd=_REPO_ROOT, stdin=subprocess.DEVNULL)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert not (hermes_home / "ESTOP").exists()


# ── chat /pause: a person's command, governed by security.estop_chat_control ──


@pytest.mark.asyncio
async def test_gateway_pause_allowed_by_default_inside_gateway(hermes_home, operator_env):
    """The gateway process carries session markers; /pause sent by an authorized person is
    still a human command and works by default (behaviour unchanged)."""
    from gateway.run import GatewayRunner

    operator_env.setenv("HERMES_SESSION_ID", "20260101_000000_ab")
    operator_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    runner = object.__new__(GatewayRunner)
    reply = await runner._handle_pause_command(_FakePauseEvent("deploy"))
    assert "paused" in reply.lower()
    assert estop.is_engaged() is True
    reply = await runner._handle_pause_command(_FakePauseEvent("off"))
    assert "resumed" in reply.lower()
    assert estop.is_engaged() is False


def _from_bot(event):
    event.source.is_bot = True


def _internal(event):
    event.internal = True


def _from_webhook(event):
    from gateway.config import Platform

    event.source.platform = Platform.WEBHOOK


@pytest.mark.asyncio
@pytest.mark.parametrize("make_automated", [_from_bot, _internal, _from_webhook],
                         ids=["bot", "internal", "webhook"])
async def test_gateway_pause_refused_from_automated_sender(hermes_home, operator_env, make_automated):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = _FakePauseEvent("agent decided")
    make_automated(event)
    reply = await runner._handle_pause_command(event)
    assert "hermes pause" in reply
    assert not (hermes_home / "ESTOP").exists()

    estop.engage(reason="operator")
    event = _FakePauseEvent("off")
    make_automated(event)
    reply = await runner._handle_pause_command(event)
    assert "hermes resume" in reply
    assert (hermes_home / "ESTOP").exists()


@pytest.mark.asyncio
async def test_gateway_pause_refused_when_chat_control_disabled(hermes_home, operator_env):
    from gateway.run import GatewayRunner

    (hermes_home / "config.yaml").write_text("security:\n  estop_chat_control: false\n", encoding="utf-8")
    runner = object.__new__(GatewayRunner)
    reply = await runner._handle_pause_command(_FakePauseEvent("deploy"))
    assert "hermes pause" in reply
    assert not (hermes_home / "ESTOP").exists()

    estop.engage(reason="operator")  # an operator at a terminal
    reply = await runner._handle_pause_command(_FakePauseEvent("off"))
    assert "hermes resume" in reply
    assert (hermes_home / "ESTOP").exists()


def test_profile_chat_cannot_lift_a_root_that_disabled_chat_control(tmp_path, operator_env):
    """Lifting removes the fleet-root sentinel, so the root's `false` binds every profile."""
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("security:\n  estop_chat_control: false\n", encoding="utf-8")
    operator_env.setenv("HERMES_HOME", str(root))
    estop.engage(reason="fleet halt")
    operator_env.setenv("HERMES_HOME", str(profile))
    with pytest.raises(estop.EstopRefused):
        estop.disengage(from_chat=True)
    assert (root / "ESTOP").exists()


def test_unreadable_chat_control_engages_but_never_lifts(hermes_home, operator_env):
    """The brake still works when the setting cannot be read; lifting fails closed."""
    import hermes_cli.config

    def _boom():
        raise OSError("config unreadable")

    operator_env.setattr(hermes_cli.config, "load_config_readonly", _boom)
    estop.engage(reason="chat", from_chat=True)
    assert estop.is_engaged() is True
    with pytest.raises(estop.EstopRefused):
        estop.disengage(from_chat=True)
    assert (hermes_home / "ESTOP").exists()
