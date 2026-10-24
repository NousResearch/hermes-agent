"""A refused reservation names its holder: the refusal is what an operator (and the dashboard's
status strip) reports, and "already owns profile X" alone leaves nothing to stop or clean."""
import json
import os

import pytest


@pytest.mark.platforms("posix")
def test_conflict_names_the_live_holder_and_its_purpose(tmp_path):
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership

    home = tmp_path / 'home'
    owner, contender = ProfileOwnership(), ProfileOwnership()
    owner.reserve([home], purpose='session-maintenance')
    try:
        with pytest.raises(OwnershipConflict) as refused:
            contender.reserve([home])
        message = str(refused.value)
        assert f'already owns profile {home.resolve()}' in message, message
        assert f'pid {os.getpid()}' in message, message
        assert 'purpose session-maintenance' in message, message
    finally:
        owner.close()
        contender.close()


@pytest.mark.platforms("posix")
def test_conflict_says_when_the_record_names_no_live_holder(tmp_path):
    """A held flock whose record names a dead PID (a torn write, or a holder mid-acquire) is still a
    refusal, but it must not borrow the language of a running gateway."""
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership

    home = tmp_path / 'home'
    owner, contender = ProfileOwnership(), ProfileOwnership()
    owner.reserve([home])
    try:
        (home / 'gateway.lock').write_text(json.dumps({'pid': 4194303, 'start_time': 1}), encoding='utf-8')
        with pytest.raises(OwnershipConflict) as refused:
            contender.reserve([home])
        message = str(refused.value)
        assert f'already owns profile {home.resolve()}' in message, message
        assert 'unidentified' in message, message
        assert 'names no live holder' in message, message
    finally:
        owner.close()
        contender.close()
