"""The local setup supervisor bounds child waits and terminates only its owner."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest


def _worker():
    file = Path(__file__).parents[3] / 'apps/desktop/public/hermestv-worker.py'
    spec = importlib.util.spec_from_file_location('hermestv_worker_under_test', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hung_setup_terminates_the_supervisor_owner(monkeypatch):
    worker = _worker()
    child = Mock()
    child.wait.side_effect = subprocess.TimeoutExpired('fixture', 600)
    monkeypatch.setattr(worker.subprocess, 'Popen', Mock(return_value=child))
    terminate = Mock()
    monkeypatch.setattr(worker.os, 'kill', terminate)
    with pytest.raises(ValueError, match='supervisor-timeout'):
        worker.supervise(json.dumps({'executable': sys.executable, 'args': []}))
    child.wait.assert_called_once_with(timeout=600)
    terminate.assert_called_once_with(worker.os.getpid(), signal.SIGTERM)


def test_setup_exit_code_is_preserved(monkeypatch):
    worker = _worker()
    child = Mock()
    child.wait.return_value = 7
    monkeypatch.setattr(worker.subprocess, 'Popen', Mock(return_value=child))
    assert worker.supervise(json.dumps({'executable': sys.executable, 'args': []})) == 7
