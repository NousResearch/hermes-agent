"""Test-only crash barriers around a native reply's terminal commit; every recovery boot is ordinary.

``KILL_BEFORE_LEDGER``: the owner dies after the terminal commit, before the adapter ledgers the reply.
``KILL_BEFORE_COMMIT``: the owner dies after the reply is in the transcript, before the terminal commit.
"""
import os
import runpy
import signal

from gateway import session_results
from gateway.platforms.base import BasePlatformAdapter

record = BasePlatformAdapter._record_delivery_obligation
finish = session_results.finish_result


async def die_before_ledger(self, event, session_key, text_content, *args, **kwargs):
    if 'KILL_BEFORE_LEDGER' in text_content:
        os.kill(os.getpid(), signal.SIGKILL)
    return await record(self, event, session_key, text_content, *args, **kwargs)


def die_before_commit(db, *, row, **kwargs):
    if row['payload'].get('text') == 'KILL_BEFORE_COMMIT':
        os.kill(os.getpid(), signal.SIGKILL)
    return finish(db, row=row, **kwargs)


BasePlatformAdapter._record_delivery_obligation = die_before_ledger
session_results.finish_result = die_before_commit
runpy.run_module('gateway.run', run_name='__main__')
