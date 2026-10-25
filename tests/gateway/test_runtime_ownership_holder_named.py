"""A refused reservation names the lock record's last owner.

The refusal is what a crash-looping supervisor and the dashboard's 409 report, so it has to say
which process to look at. It must never claim more than the record can support: a live process that
already gave the home up is reported as the last recorded owner, not as the holder.
"""
import json
import os

import pytest


def _conflict_message(tmp_path, holder):
    """Reserve a home, run *holder* to take the flock, and return the contender's refusal text."""
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership

    home = tmp_path / 'home'
    contender = ProfileOwnership()
    holder(home)
    try:
        with pytest.raises(OwnershipConflict) as refused:
            contender.reserve([home])
        return home, str(refused.value)
    finally:
        contender.close()


@pytest.mark.platforms("posix")
def test_conflict_names_the_last_recorded_owner_and_its_purpose(tmp_path):
    from gateway.runtime_ownership import ProfileOwnership

    owner = ProfileOwnership()

    def hold(home):
        owner.reserve([home], purpose='session-maintenance')

    try:
        home, message = _conflict_message(tmp_path, hold)
        assert f'already owns profile {home.resolve()}' in message, message
        assert f'last recorded owner: pid {os.getpid()}' in message, message
        assert 'purpose session-maintenance' in message, message
    finally:
        owner.close()


@pytest.mark.platforms("posix")
def test_conflict_says_unidentified_when_the_record_names_no_live_process(tmp_path):
    from gateway.runtime_ownership import ProfileOwnership

    owner = ProfileOwnership()

    def hold(home):
        owner.reserve([home])
        (home / 'gateway.lock').write_text(json.dumps({'pid': 4194303, 'start_time': 1}), encoding='utf-8')

    try:
        home, message = _conflict_message(tmp_path, hold)
        assert f'already owns profile {home.resolve()}' in message, message
        assert 'unidentified' in message, message
        assert 'names no live process' in message, message
        assert 'pid 4194303' not in message.replace('names no live process', ''), message
    finally:
        owner.close()


@pytest.mark.platforms("posix")
def test_conflict_says_unidentified_for_an_empty_record(tmp_path):
    """The realistic mid-acquire case: the winner holds the flock and has not written yet."""
    from gateway.runtime_ownership import ProfileOwnership

    owner = ProfileOwnership()

    def hold(home):
        owner.reserve([home])
        (home / 'gateway.lock').write_text('', encoding='utf-8')

    try:
        home, message = _conflict_message(tmp_path, hold)
        assert f'already owns profile {home.resolve()}' in message, message
        assert 'unidentified' in message, message
        assert 'empty or unreadable' in message, message
    finally:
        owner.close()


@pytest.mark.platforms("posix")
def test_released_holder_is_named_as_last_recorded_owner_not_as_holder(tmp_path):
    """`release` leaves the record behind, so a live process that gave the home up is still readable.

    The next owner holds the flock before it rewrites the record; a contender that loses in that
    window must not be told the previous owner is the holder.
    """
    from gateway.runtime_ownership import ProfileOwnership
    from gateway.status import _release_file_lock, _try_acquire_file_lock

    previous = ProfileOwnership()
    previous.reserve([tmp_path / 'home'])
    previous.release(tmp_path / 'home')
    previous.close()

    winner = open(tmp_path / 'home' / 'gateway.lock', 'r+', encoding='utf-8')  # no record written yet
    assert _try_acquire_file_lock(winner)
    try:
        _home, message = _conflict_message(tmp_path, lambda _home: None)
        assert f'last recorded owner: pid {os.getpid()}' in message, message
        assert 'holder: ' not in message, message
    finally:
        _release_file_lock(winner)
        winner.close()
