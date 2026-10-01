"""One-shot actual owner crash after durable file publication, before input admission."""
import os
from pathlib import Path
import runpy

from gateway import hosted_room_input_preparation
from tui_gateway.hosted_room_driver import HostedRoomRuntime

record_error = HostedRoomRuntime._record_error

def trace_error(self, message):
    record_error(self, message)
    if getattr(self, "_fixture_last_error", None) != message:
        self._fixture_last_error = message
        print("ROOM RECOVERY:", message, flush=True)

HostedRoomRuntime._record_error = trace_error

copy = hosted_room_input_preparation._copy


def crash_after_copy(data, path):
    copy(data, path)
    marker = Path(os.environ['HERMES_HOME']) / 'crash-document-preparation'
    if marker.exists():
        marker.unlink()
        os._exit(77)


hosted_room_input_preparation._copy = crash_after_copy
runpy.run_module('gateway.run', run_name='__main__')
