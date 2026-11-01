"""Cloud cleanup releases its response before returning to the orchestrator."""

import io
import json
import runpy
import sys
import urllib.request
from pathlib import Path
from unittest.mock import Mock


def test_browserbase_close_releases_response(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.json"
    tasks.write_text("{}")
    script = Path(__file__).resolve().parents[2] / "evals/browser_use/orchestrate_cloud.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--backend", "browserbase",
                        "--tasks", str(tasks), "--results", str(tmp_path / "results.jsonl")])
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "test-project")
    monkeypatch.setenv("BROWSERBASE_API_KEY", "test-key")
    # An empty task list loads the real class without provisioning a browser.
    module = runpy.run_path(str(script))
    response = io.BytesIO(b"unused")
    urlopen = Mock(return_value=response)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    module["Browserbase"]().close({"id": "test-session"})

    assert response.closed
    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/sessions/test-session")
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"projectId": "test-project", "status": "REQUEST_RELEASE"}
    assert urlopen.call_args.kwargs["timeout"] == 30
