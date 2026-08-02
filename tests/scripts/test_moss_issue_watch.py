from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "moss-issue-watch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("moss_issue_watch_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_has_linked_pr_accepts_current_list_shape(monkeypatch):
    module = load_module()
    responses = iter(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {"closedByPullRequestsReferences": [{"number": 123}]}
                ),
                stderr="",
            )
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(responses))

    assert module.has_linked_pr("NousResearch/hermes-agent", 456) == (
        True,
        "has closing PR reference",
    )


def test_has_linked_pr_keeps_legacy_mapping_shape(monkeypatch):
    module = load_module()
    responses = iter(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {"closedByPullRequestsReferences": {"totalCount": 1}}
                ),
                stderr="",
            )
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(responses))

    assert module.has_linked_pr("NousResearch/hermes-agent", 456) == (
        True,
        "has closing PR reference",
    )
