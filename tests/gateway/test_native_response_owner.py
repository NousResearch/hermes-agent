"""Native responses retain their creator's authority across public dashboard sockets."""
import asyncio
from pathlib import Path

from tests.gateway.fixtures.native_response_owner import owner_cases
from tests.gateway.fixtures.native_response_public import public_native_cases


def test_public_ticket_answer_releases_native_waiter_and_next_admission(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    result = public_native_cases(tmp_path)
    for case in result['cases']:
        assert case['public_reply_settled'], case
        assert case['first_receipt']['status'] == case['next_receipt']['status'] == 'terminal', case
        assert case['snapshot_clear'] and case['next_send_once'], case


def test_known_request_id_never_substitutes_another_owner_or_lifetime(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    for result in asyncio.run(owner_cases(tmp_path)):
        assert result['settled'] == result['expected'], result
