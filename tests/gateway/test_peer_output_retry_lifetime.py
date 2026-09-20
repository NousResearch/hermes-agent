"""Finite manual retry clocks and retained-task metadata lifetime."""
import json
import time

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import tick, pending, clock
from gateway.run import _profile_runtime_scope


@pytest.mark.asyncio
async def test_backoff_is_bounded_and_metadata_never_contains_payloads_or_credentials(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        attempts = []
        def unavailable(self, *args):
            attempts.append(now[0])
            raise PeerRunsHTTPError('secret error ' + c.issued['grant'] + str(c.home), retryable=True)
        monkeypatch.setattr(PeerOutputCustody, 'read', unavailable)
        for number, delay in enumerate([1, 2, 4, 8, 16, 32, 60, 60], 1):
            await tick(c)
            row, = pending(c)
            assert row['attempts'] == number
            assert row['next_attempt_at'] == now[0] + delay
            text = json.dumps(row)
            assert c.issued['grant'] not in text and str(c.home) not in text
            assert c.task['payload']['prompt'] not in text
            await tick(c)
            assert len(attempts) == number
            now[0] = row['next_attempt_at']
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_retry_protects_task_and_completion_prunes_only_after_task_retirement(files_target, monkeypatch):
    from gateway import hosted_room_driver as tasks
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        c.wire.faults.lost_ack = True
        await tick(c)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            assert tasks.prune_published_terminal_tasks(c.db.db_path, room_id='room-one', clock=time.time, retain=0) == 0
        now[0] = pending(c)[0]['next_attempt_at']
        await tick(c)
        assert pending(c) == []
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            assert tasks.prune_published_terminal_tasks(c.db.db_path, room_id='room-one', clock=time.time, retain=0) == 1
        await tick(c)
        assert c.db._conn.execute('SELECT count(*) FROM hosted_room_artifact_completions').fetchone()[0] == 0
