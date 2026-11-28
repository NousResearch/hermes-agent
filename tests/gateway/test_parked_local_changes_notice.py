"""A gateway restarted by an update that left local changes in a stash tells its operator.

``hermes update`` parks an autostash whose restore conflicted, records the owed ``local_changes``
user action (#122557) and restarts the gateway. Nothing outside the update's own terminal said so.
These tests run the real stash/park and receipt producers in a throwaway git checkout, the real
boot-send pass, and the real home-channel target selection against fake adapters.
"""

import asyncio
import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.run as gateway_run
import hermes_cli.update_cmd_stash as stash
from hermes_cli import update_receipt
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from hermes_cli.update_completion import _record_owed_user_action
from pm import receipt as pm_receipt

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

HEADING = "left local source changes in a git stash"


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _adapter(result=None):
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=result or SendResult(success=True, message_id="unit-test-notice")),
    )


def _home(platform, chat_id):
    return PlatformConfig(enabled=True, home_channel=HomeChannel(platform=platform, chat_id=chat_id, name=chat_id))


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A Hermes home, a checkout this gateway serves, and a booting gateway with one Discord home."""
    home = tmp_path / "hermes"
    home.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q", "-b", "main")
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "t")
    _git(checkout, "config", "commit.gpgsign", "false")
    (checkout / "tool.py").write_text("VALUE = 1\n")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-qm", "upstream 1")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    # The gateway under test runs from the throwaway checkout (raising=False: absent on main).
    monkeypatch.setattr("gateway.run_notifications_update._serving_checkout", lambda: checkout.resolve(),
                        raising=False)
    monkeypatch.setattr(stash, "_pending_autostash", None)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.DISCORD: _home(Platform.DISCORD, "operator-home")},
                                  sessions_dir=tmp_path / "sessions")
    runner.adapters = {Platform.DISCORD: _adapter()}
    runner._claim_pending_obligations = AsyncMock(return_value=[])
    runner._redeliver_claimed_obligations = AsyncMock(return_value=0)
    return SimpleNamespace(home=home, checkout=checkout, runner=runner)


def _update(checkout, *, local_edit=True, keep_stash=False, finalize=True, source=None) -> str:
    """One ``hermes update`` run through the real producers: autostash, upstream move, restore or
    park, then the completion child's user-action record. A local edit conflicts with upstream."""
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        update_id = update_receipt._current.get().data["update_id"]
        upstream = int(_git(checkout, "rev-list", "--count", "HEAD")) + 1
        if local_edit:
            (checkout / "tool.py").write_text("VALUE = 'my local patch'\n")
        ref = stash._stash_local_changes_if_needed(["git"], checkout)
        (checkout / "tool.py").write_text(f"VALUE = {upstream}\n")
        _git(checkout, "commit", "-qam", f"upstream {upstream}")
        if ref and keep_stash:
            stash._park_stashed_changes(ref)
        elif ref:
            assert stash._restore_stashed_changes(["git"], checkout, ref) is False, "the restore must conflict"
        # update_cmd._complete_source_update hands the unsettled-stash notice to the completion child.
        request = {"source": str(source or checkout)}
        notice = stash._unrestored_autostash_notice()
        if notice:
            request["completion_message"] = notice
        _record_owed_user_action(request)
        stash._clear_pending_autostash()
        if finalize:
            update_receipt.finalize_update_receipt("success")
    return update_id


def _boot(runner):
    asyncio.run(runner._await_startup_boot_sends(planned_restart_notification_pending=False))


def _sent(adapter):
    return [call.args[1] for call in adapter.send.await_args_list]


def test_boot_after_a_parked_update_tells_the_operator_once(host):
    _update(host.checkout)
    ref = _git(host.checkout, "rev-parse", "refs/stash")
    discord = host.runner.adapters[Platform.DISCORD]

    _boot(host.runner)
    _boot(host.runner)

    assert len(_sent(discord)) == 1, "one notice per update run, not one per start"
    assert discord.send.await_args.args[0] == "operator-home"
    message = _sent(discord)[0]
    assert HEADING in message and "did NOT restore" in message and ref in message


@pytest.mark.parametrize("profile", [None, "work"], ids=["default-home", "named-profile-home"])
def test_package_manager_sync_after_the_park_does_not_hide_it(host, monkeypatch, profile):
    """Every pm sync rewrites ``latest.json`` (#157231); the newest UPDATE run still decides."""
    if profile:
        (host.home / "config.yaml").write_text("{}\n")
        profile_home = host.home / "profiles" / profile
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setattr(gateway_run, "_hermes_home", profile_home)
    _update(host.checkout)
    pm_receipt.begin("sync")
    pm_receipt.finalize("ok")
    assert update_receipt.read_latest_receipt()["kind"] == "sync", "the pointer now names the sync"

    _boot(host.runner)

    assert len(_sent(host.runner.adapters[Platform.DISCORD])) == 1


def test_gateway_restarted_while_the_update_is_still_running_is_told(host):
    """The completion child records the action before it restarts the fleet, so the run is open."""
    _update(host.checkout, finalize=False)

    _boot(host.runner)

    assert len(_sent(host.runner.adapters[Platform.DISCORD])) == 1


@pytest.mark.parametrize("later", ["clean-update", "malformed-newest-run"])
def test_only_the_newest_update_run_counts(host, monkeypatch, later):
    _update(host.checkout)
    if later == "clean-update":
        # Run records are named by their start second: start the later run in a later one.
        monkeypatch.setattr(update_receipt, "_utc_now_iso", lambda: "2999-01-01T00:00:00+00:00")
        _update(host.checkout, local_edit=False)
    else:
        runs = host.home / "logs" / "update_receipts"
        (runs / "update_99991231_235959_1_ffff.json").write_text("{not json", encoding="utf-8")

    _boot(host.runner)

    host.runner.adapters[Platform.DISCORD].send.assert_not_called()


@pytest.mark.parametrize("case", ["keep-stash", "other-checkout", "satellite-only-host", "no-update-run"])
def test_no_operator_notice(host, tmp_path, case):
    """A chosen ``--keep-stash`` park owes nothing; another checkout's park is not this gateway's;
    a multiplexer's served profiles are tenants, so a host whose launch profile has no home channel
    stays quiet; a home with no update record stays quiet."""
    if case == "keep-stash":
        _update(host.checkout, keep_stash=True)
        assert _git(host.checkout, "stash", "list"), "the chosen park keeps its stash"
    elif case == "other-checkout":
        _update(host.checkout, source=tmp_path / "another-checkout")
    elif case == "satellite-only-host":
        _update(host.checkout)
        tenant = _adapter()
        host.runner.config = GatewayConfig(platforms={}, sessions_dir=tmp_path / "sessions")
        host.runner._profile_configs = {"tenant": GatewayConfig(platforms={Platform.TELEGRAM: _home(Platform.TELEGRAM, "t")})}
        host.runner._profile_adapters = {"tenant": {Platform.TELEGRAM: tenant}}
        host.runner.adapters = {Platform.DISCORD: _adapter()}
        _boot(host.runner)
        tenant.send.assert_not_called()

    _boot(host.runner)

    host.runner.adapters[Platform.DISCORD].send.assert_not_called()


def test_muted_warnings_send_nothing(host):
    (host.home / "config.yaml").write_text("display:\n  suppress_warning_notifications: true\n")
    _update(host.checkout)

    _boot(host.runner)
    (host.home / "config.yaml").write_text("{}\n")
    _boot(host.runner)

    # A muted home is settled, not owed: unmuting later does not replay the notice.
    host.runner.adapters[Platform.DISCORD].send.assert_not_called()


def test_a_home_that_was_down_at_boot_is_told_at_the_next_start_only(host):
    host.runner.config.platforms[Platform.TELEGRAM] = _home(Platform.TELEGRAM, "second-home")
    _update(host.checkout)
    discord, telegram = host.runner.adapters[Platform.DISCORD], _adapter()

    _boot(host.runner)  # Telegram not connected
    host.runner.adapters[Platform.TELEGRAM] = telegram
    _boot(host.runner)
    _boot(host.runner)

    assert len(_sent(discord)) == 1 and len(_sent(telegram)) == 1


def test_a_refused_send_is_retried_for_at_most_three_starts(host):
    refusing = _adapter(SendResult(success=False, error="chat not found"))
    host.runner.adapters[Platform.DISCORD] = refusing
    _update(host.checkout)

    for _ in range(5):
        _boot(host.runner)

    assert refusing.send.await_count == 3


def test_a_slow_send_does_not_hold_the_inbound_gate(host, monkeypatch):
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0.05)
    _update(host.checkout)

    async def scenario():
        release = asyncio.Event()

        async def slow_send(*_args, **_kwargs):
            await release.wait()
            return SendResult(success=True, message_id="late")

        host.runner.adapters[Platform.DISCORD].send = AsyncMock(side_effect=slow_send)
        await asyncio.wait_for(
            host.runner._await_startup_boot_sends(planned_restart_notification_pending=False), timeout=5)
        started = host.runner.adapters[Platform.DISCORD].send.await_count
        release.set()
        await asyncio.gather(*getattr(host.runner, "_background_tasks", ()))
        return started

    assert asyncio.run(scenario()) == 1, "the send was in flight when startup moved on"
    marker = json.loads((host.home / ".update_local_changes_notice.json").read_text(encoding="utf-8-sig"))
    assert marker["settled"] == [["discord", "operator-home", None]]


def test_an_unwritable_marker_still_sends_and_never_breaks_startup(host, caplog):
    """Best effort, not exactly once: without a marker the next start repeats the notice."""
    _update(host.checkout)
    (host.home / ".update_local_changes_notice.json").mkdir()  # neither readable nor replaceable as a file

    _boot(host.runner)
    _boot(host.runner)

    assert len(_sent(host.runner.adapters[Platform.DISCORD])) == 2
    assert "the next start may repeat it" in caplog.text
    host.runner._redeliver_claimed_obligations.assert_awaited()
