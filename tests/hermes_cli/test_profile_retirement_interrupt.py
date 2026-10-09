"""An interrupted profile rename or delete must not leave the profile behind its own tombstone.

Retirement publishes the tombstone before the bounded holder census, which waits up to ~5 s
per pass while another process has a profile file open. A Ctrl-C or a failure in that window
used to leave the home tombstoned with its data intact: list/show/-p said the profile did not
exist, a rename retry was refused as "being deleted", create said the name was taken, and only
delete still reached it (PR #93508 review).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profile_cmd, profile_lifecycle, profiles
from hermes_constants import named_profile_is_deleted

_HOLDER = 424242


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # Never inspect or stop services and processes outside this disposable fixture.
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda *_: False)
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_profile_backends", "_stop_bot_desktop"):
        monkeypatch.setattr(profiles, name, lambda *_: None)
    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", lambda *_: [])
    return home


def _holder_wait_interrupted_once(monkeypatch, exc_type):
    """A process holds a profile file open and *exc_type* lands while the census waits on it."""
    answers = iter([[_HOLDER], exc_type("interrupted while waiting for the holder")])

    def census(_profile_dir, _candidates=None):
        answer = next(answers, [])
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", census)


def _mcp_shutdown_interrupted_once(monkeypatch, exc_type):
    """*exc_type* lands after retirement, while the old home's MCP transports close."""
    from tools import mcp_tool_lifecycle

    pending = [exc_type("interrupted while closing MCP transports")]

    def shutdown(*_args, **_kwargs):
        if pending:
            raise pending.pop()

    monkeypatch.setattr(mcp_tool_lifecycle, "shutdown_mcp_servers", shutdown)


def _backend_stop_interrupted_once(monkeypatch, exc_type):
    """*exc_type* lands after delete's tombstone, while the profile's backends stop."""
    pending = [exc_type("interrupted while stopping profile backends")]

    def stop(*_args):
        if pending:
            raise pending.pop()

    monkeypatch.setattr(profiles, "_stop_profile_backends", stop)


def _listed(name: str) -> bool:
    return name in [p.name for p in profiles.list_profiles()]


@pytest.mark.parametrize(("interrupt", "exc_type"), [
    pytest.param(_holder_wait_interrupted_once, KeyboardInterrupt, id="ctrl-c-in-holder-wait"),
    pytest.param(_holder_wait_interrupted_once, RuntimeError, id="holder-census-failed"),
    pytest.param(_mcp_shutdown_interrupted_once, KeyboardInterrupt, id="ctrl-c-in-mcp-shutdown"),
])
def test_interrupted_rename_leaves_the_profile_live_and_retryable(profile_env, monkeypatch, interrupt, exc_type):
    old_dir = profiles.create_profile("coder", no_alias=True, no_skills=True)
    interrupt(monkeypatch, exc_type)

    with pytest.raises(exc_type):
        profiles.rename_profile("coder", "dev")

    assert not named_profile_is_deleted(old_dir)
    assert profiles.profile_exists("coder") and _listed("coder")
    assert not profiles.profile_exists("dev")

    new_dir = profiles.rename_profile("coder", "dev")
    assert new_dir == profiles.get_profile_dir("dev")
    assert profiles.profile_exists("dev") and not profiles.profile_exists("coder")


@pytest.mark.parametrize(("interrupt", "exc_type"), [
    pytest.param(_holder_wait_interrupted_once, KeyboardInterrupt, id="ctrl-c-in-holder-wait"),
    pytest.param(_backend_stop_interrupted_once, KeyboardInterrupt, id="ctrl-c-stopping-backends"),
    pytest.param(_backend_stop_interrupted_once, RuntimeError, id="backend-stop-failed"),
])
def test_interrupted_delete_restores_the_profile(profile_env, monkeypatch, interrupt, exc_type):
    profile_dir = profiles.create_profile("coder", no_alias=True, no_skills=True)
    interrupt(monkeypatch, exc_type)

    with pytest.raises(exc_type):
        profiles.delete_profile("coder", yes=True)

    assert profile_dir.is_dir()
    assert not named_profile_is_deleted(profile_dir)
    assert profiles.profile_exists("coder") and _listed("coder")
    # Not refused as "being deleted": an action other than delete still reaches it.
    assert profiles.rename_profile("coder", "dev") == profiles.get_profile_dir("dev")


def test_delete_interrupted_once_removal_began_keeps_the_fence(profile_env, monkeypatch):
    profile_dir = profiles.create_profile("coder", no_alias=True, no_skills=True)

    def interrupted_rmtree(*_args):
        raise KeyboardInterrupt("interrupted while removing the profile")

    monkeypatch.setattr(profiles, "_rmtree_with_retry", interrupted_rmtree)

    with pytest.raises(KeyboardInterrupt):
        profiles.delete_profile("coder", yes=True)

    # rmtree may have removed part of the tree; a half-deleted home must stay hidden.
    assert named_profile_is_deleted(profile_dir)
    assert not profiles.profile_exists("coder") and not _listed("coder")


def test_cli_rename_refused_by_a_holder_prints_the_reason_without_a_traceback(profile_env, monkeypatch, capsys):
    profiles.create_profile("coder", no_alias=True, no_skills=True)
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", lambda *_: [_HOLDER])

    with pytest.raises(SystemExit) as exited:
        profile_cmd._profile_rename(SimpleNamespace(old_name="coder", new_name="dev"))

    assert exited.value.code == 1
    out, err = capsys.readouterr()
    assert str(_HOLDER) in out
    assert "Traceback" not in out + err
    assert profiles.profile_exists("coder")
