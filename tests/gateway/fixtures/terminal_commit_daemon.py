"""Stop the actual owner after the result commit, before any publication."""
import os
import runpy
import signal
from gateway import session_results

finish = session_results.finish_result


def stop_after_commit(db, *, row, **kwargs):
    receipt = finish(db, row=row, **kwargs)
    if row['payload'].get('text') == 'BLOCK_STARTED':
        os.kill(os.getpid(), signal.SIGSTOP)
    return receipt


session_results.finish_result = stop_after_commit
runpy.run_module('gateway.run', run_name='__main__')
