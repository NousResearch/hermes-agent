"""One-shot actual owner crash after durable file publication, before input admission."""
import os
from pathlib import Path
import runpy

from gateway import hosted_room_input_preparation

copy = hosted_room_input_preparation._copy


def crash_after_copy(data, path):
    copy(data, path)
    marker = Path(os.environ['HERMES_HOME']) / 'crash-document-preparation'
    if marker.exists():
        marker.unlink()
        os._exit(77)


hosted_room_input_preparation._copy = crash_after_copy
runpy.run_module('gateway.run', run_name='__main__')
