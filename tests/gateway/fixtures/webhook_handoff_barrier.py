"""Stop the owned daemon after source commit, before any Bot Chat transport call."""

import os
import runpy
import signal
from pathlib import Path

from gateway import webhook_final_intent


def stop_before_handoff(*args, **kwargs):
    Path(os.environ['TEST_HANDOFF_BARRIER']).write_text('before handoff')
    os.kill(os.getpid(), signal.SIGSTOP)


webhook_final_intent._handoff = stop_before_handoff
runpy.run_module('gateway.run', run_name='__main__')
