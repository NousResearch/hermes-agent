"""History readiness closes successful and malformed status responses."""

import io
import os
import runpy
import subprocess
import sys
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class ProbeComplete(BaseException):
    """Stop before the unrelated WebSocket assertions after readiness succeeds."""


@pytest.mark.parametrize("retry", [False, True])
def test_status_probe_closes_responses(tmp_path, monkeypatch, retry):
    import websockets.sync.client

    root = Path(__file__).resolve().parents[2]
    script = root / "evals/desktop_bug_campaign/history_live.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--repo", str(root),
                                    "--output", str(tmp_path / "receipt")])
    responses = ([io.BytesIO(b"bad json")] if retry else []) + [io.BytesIO(b'{"status": "ok"}')]
    urlopen = Mock(side_effect=responses)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(websockets.sync.client, "connect", Mock(side_effect=ProbeComplete))

    with patch.dict(os.environ), pytest.raises(ProbeComplete):
        runpy.run_path(str(script))

    assert urlopen.call_count == len(responses)
    assert all(response.closed for response in responses)
    process.terminate.assert_called_once()
