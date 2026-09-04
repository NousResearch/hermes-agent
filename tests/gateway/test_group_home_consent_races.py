"""Cancellation and replacement around real config persistence, without retries."""

import asyncio
import time
from dataclasses import replace
from threading import Event

import pytest

from gateway import group_home_consent as consent
from gateway.config import Platform
from hermes_cli import config as configuration
from tests.gateway.test_group_home_consent import home, command


def ack():
    return configuration.load_config()["platforms"]["telegram"]["home_channel"].get(
        "group_audience_ack"
    )


def hold(monkeypatch, stage):
    entered, release, finished = Event(), Event(), Event()
    owner, name = (
        (configuration, "load_config")
        if stage == "before_save"
        else (consent, "_persist")
    )
    if stage == "during_save":
        owner, name = configuration, "save_config"
    original = getattr(owner, name)
    held = False

    def controlled(*args, **kwargs):
        nonlocal held
        if held:
            return original(*args, **kwargs)
        held = True
        try:
            value = (
                original(*args, **kwargs)
                if stage in {"before_save", "after_save"}
                else None
            )
            entered.set()
            assert release.wait(10)
            return (
                value
                if stage in {"before_save", "after_save"}
                else original(*args, **kwargs)
            )
        finally:
            finished.set()

    monkeypatch.setattr(owner, name, controlled)
    return entered, release, finished


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before_persist", "before_save"])
@pytest.mark.parametrize("action", ["cancel", "replace"])
async def test_cancel_or_replace_before_commit_never_writes_or_deletes_new_prompt(
    home, monkeypatch, stage, action
):
    await command(home, "/group")
    old = next(iter(home.runner._group_home_confirmations.values()))
    entered, release, finished = hold(monkeypatch, stage)
    task = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        reply = await asyncio.wait_for(
            command(home, "/group cancel" if action == "cancel" else "/group"), 2
        )
        newer = home.runner._group_home_confirmations.get(old.key)
        if action == "cancel":
            assert reply == consent.text("cancel") and newer is None
        else:
            assert consent.text("warning") in reply and newer is not old
    finally:
        release.set()
    result = await task
    assert finished.is_set() and not ack()
    assert "Release room" not in result and "Research room" not in result
    assert home.runner._group_home_confirmations.get(old.key) is newer


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "replace"])
async def test_during_save_has_honest_boundary_no_old_chooser_and_unblocked_stop(
    home, monkeypatch, action
):
    await command(home, "/group")
    entered, release, _finished = hold(monkeypatch, "during_save")
    task = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        started = time.monotonic()
        reply = await asyncio.wait_for(
            command(home, "/group cancel" if action == "cancel" else "/group"), 2
        )
        stopped = await asyncio.wait_for(command(home, "/group 1 stop"), 2)
        assert time.monotonic() - started < 4
        assert reply == consent.text("cancel_late" if action == "cancel" else "saving")
        assert home.service.stopped and stopped == consent.text("stop_requested")
        assert "Release room" not in reply + stopped
    finally:
        release.set()
    result = await task
    assert ack() and "Release room" not in result and "Research room" not in result
    assert not home.runner._group_home_confirmations


@pytest.mark.asyncio
async def test_old_completed_writer_cannot_remove_reselected_prompt(home, monkeypatch):
    await command(home, "/group")
    entered, release, _finished = hold(monkeypatch, "after_save")
    task = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        await home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
        assert consent.text("warning") in await command(home, "/group")
        current = next(iter(home.runner._group_home_confirmations.values()))
    finally:
        release.set()
    result = await task
    assert home.runner._group_home_confirmations.get(current.key) is current
    assert not ack() and "Release room" not in result


@pytest.mark.asyncio
async def test_cancel_during_chooser_read_withholds_private_output(home, monkeypatch):
    from gateway import hosted_room_messaging as rooms

    await command(home, "/group")
    entered, release = Event(), Event()
    original = rooms.list_messaging_rooms

    def paused(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(rooms, "list_messaging_rooms", paused)
    task = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        assert await command(home, "/group cancel") == consent.text("cancel_late")
    finally:
        release.set()
    result = await task
    assert "Release room" not in result and "Research room" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before_persist", "during_save"])
async def test_task_cancellation_cannot_leave_a_queued_acceptance_or_retry_trap(
    home, monkeypatch, stage
):
    await command(home, "/group")
    entered, release, finished = hold(monkeypatch, stage)
    worker_finished = Event()
    original_persist = consent._persist

    def observed_persist(*args, **kwargs):
        try:
            return original_persist(*args, **kwargs)
        finally:
            worker_finished.set()

    monkeypatch.setattr(consent, "_persist", observed_persist)
    task = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(finished.wait, 5)
    assert await asyncio.to_thread(worker_finished.wait, 5)
    assert bool(ack()) is (stage == "during_save")
    if stage == "before_persist":
        assert consent.text("warning") in await command(home, "/group")
    else:
        assert "Release room" in await command(home, "/group list")
