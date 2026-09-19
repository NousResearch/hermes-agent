"""Tests for the `jev` shadow-mode scoring in cron/scripts/classify_items.py.

Shadow mode must never affect the production decision (which items get surfaced, the exit
code). It only appends a comparison record per item to cron/logs/jev-shadow-classify.jsonl.
`jev` itself is always mocked here -- no real subprocess, no network, no credentials needed.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cron.scripts import classify_items


def _fake_llm_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


@pytest.fixture(autouse=True)
def isolated_shadow_log(tmp_path, monkeypatch):
    """Point the shadow log at a scratch file so tests never touch the real one."""
    log_path = tmp_path / "jev-shadow-classify.jsonl"
    monkeypatch.setattr(classify_items, "_JEV_SHADOW_LOG", log_path)
    return log_path


def _run_main(monkeypatch, items, llm_content, threshold=7):
    monkeypatch.setattr(
        "sys.argv",
        ["classify_items.py", "--criteria", "test criteria", "--threshold", str(threshold), "--format", "json"],
    )
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(items)))
    with patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response(llm_content),
    ):
        return classify_items.main()


def _jev_ok_row(item_id: str, ok_score_1to10: int, confidence: float = 0.7) -> dict:
    idx = ok_score_1to10 - 1
    legend = {str(i): str(i + 1) for i in range(10)}
    return {
        "index": 0,
        "id": item_id,
        "ok": True,
        "failed": False,
        "result": {
            "answers": {
                "urgency": {"type": "score", "score": idx, "confidence": confidence, "legend": legend}
            }
        },
    }


def test_jev_shadow_logs_agreement_when_both_surface(monkeypatch, isolated_shadow_log):
    items = [{"id": "a", "title": "prod is down"}]
    llm_content = json.dumps([{"index": 0, "score": 9, "reason": "urgent"}])
    jev_rows = [_jev_ok_row("a", 9)]

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["jev"], returncode=0, stdout=json.dumps(jev_rows), stderr="",
        )
        rc = _run_main(monkeypatch, items, llm_content)

    assert rc == 0
    lines = isolated_shadow_log.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["item_id"] == "a"
    assert record["llm_score"] == 9
    assert record["llm_surfaced"] is True
    assert record["jev"]["score"] == 9
    assert record["jev_surfaced"] is True
    assert record["agree"] is True


def test_jev_shadow_logs_disagreement(monkeypatch, isolated_shadow_log):
    items = [{"id": "b", "title": "quarterly newsletter"}]
    llm_content = json.dumps([{"index": 0, "score": 2, "reason": "not urgent"}])
    jev_rows = [_jev_ok_row("b", 8)]

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["jev"], returncode=0, stdout=json.dumps(jev_rows), stderr="",
        )
        rc = _run_main(monkeypatch, items, llm_content, threshold=7)

    assert rc == 0  # LLM said not urgent -> production path stays silent regardless of jev
    record = json.loads(isolated_shadow_log.read_text().strip())
    assert record["llm_surfaced"] is False
    assert record["jev_surfaced"] is True
    assert record["agree"] is False


@pytest.mark.parametrize(
    "run_side_effect",
    [
        FileNotFoundError("jev: command not found"),
        subprocess.TimeoutExpired(cmd=["jev"], timeout=60),
    ],
)
def test_jev_shadow_failure_is_swallowed_and_logged(monkeypatch, isolated_shadow_log, run_side_effect):
    """A broken/missing jev CLI must never break the production path (exit code, surfaced item)."""
    items = [{"id": "c", "title": "prod is down"}]
    llm_content = json.dumps([{"index": 0, "score": 9, "reason": "urgent"}])

    with patch("subprocess.run", side_effect=run_side_effect):
        rc = _run_main(monkeypatch, items, llm_content)

    assert rc == 0
    record = json.loads(isolated_shadow_log.read_text().strip())
    assert record["jev_call_failed"] is True
    assert "error" in record


def test_jev_shadow_never_raises_out_of_main(monkeypatch, isolated_shadow_log):
    """Even a totally malformed jev response must not propagate as an exception."""
    items = [{"id": "d", "title": "prod is down"}]
    llm_content = json.dumps([{"index": 0, "score": 9, "reason": "urgent"}])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["jev"], returncode=0, stdout="not valid json {{{", stderr="",
        )
        rc = _run_main(monkeypatch, items, llm_content)

    assert rc == 0
    record = json.loads(isolated_shadow_log.read_text().strip())
    assert record["jev_call_failed"] is True
