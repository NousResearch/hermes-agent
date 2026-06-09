from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_cli import swarm


def _args(tmp_path: Path, **overrides):
    defaults = dict(
        run_dir=str(tmp_path),
        worker=[],
        toolsets=None,
        skills=None,
        max_turns=None,
        timeout_seconds=None,
        kill_grace_seconds=0.01,
        poll_interval=0.001,
        status_file=None,
        source_prefix=None,
        hermes_bin="hermes",
        read_only=False,
        accept_hooks=False,
        yolo=False,
        allow_failures=False,
        dry_run=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_parse_worker_spec_defaults_handoff():
    assert swarm.parse_worker_spec("researcher") == {
        "profile": "researcher",
        "handoff": "researcher.md",
    }


def test_parse_worker_spec_accepts_colon_and_equals():
    assert swarm.parse_worker_spec("researcher:search.md") == {
        "profile": "researcher",
        "handoff": "search.md",
    }
    assert swarm.parse_worker_spec("reviewer=review.md") == {
        "profile": "reviewer",
        "handoff": "review.md",
    }


def test_dry_run_redacts_prompt_from_status(tmp_path: Path, capsys):
    handoffs = tmp_path / "handoffs"
    handoffs.mkdir()
    (handoffs / "worker.md").write_text("secret prompt body", encoding="utf-8")

    rc = swarm.run_swarm(
        _args(tmp_path, worker=["worker"], dry_run=True, toolsets="web,file")
    )

    assert rc == 0
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "planned"
    record = status["workers"]["worker"]
    assert record["cmd_preview"][-1] == "<prompt redacted>"
    assert "secret prompt body" not in (tmp_path / "status.json").read_text(encoding="utf-8")
    assert "secret prompt body" not in capsys.readouterr().out


def test_missing_handoff_is_skipped_and_marks_failure(tmp_path: Path):
    rc = swarm.run_swarm(_args(tmp_path, worker=["missing"]))

    assert rc == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "completed_with_failures"
    assert status["workers"]["missing"]["status"] == "skipped"
    assert status["workers"]["missing"]["reason"] == "missing_handoff"


def test_run_swarm_tracks_success_and_failure(tmp_path: Path, monkeypatch):
    handoffs = tmp_path / "handoffs"
    handoffs.mkdir()
    (handoffs / "ok.md").write_text("do ok", encoding="utf-8")
    (handoffs / "bad.md").write_text("do bad", encoding="utf-8")

    class FakePopen:
        next_pid = 100

        def __init__(self, cmd, stdout=None, stderr=None, text=None):
            self.cmd = cmd
            self.stdout = stdout
            self.stderr = stderr
            self.text = text
            self.pid = FakePopen.next_pid
            FakePopen.next_pid += 1
            profile = cmd[cmd.index("-p") + 1]
            self.returncode = 0 if profile == "ok" else 7
            self._polled = False
            if stdout:
                stdout.write(f"out:{profile}\n")
            if stderr and profile == "bad":
                stderr.write("err:bad\n")

        def poll(self):
            if not self._polled:
                self._polled = True
                return None
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(swarm.subprocess, "Popen", FakePopen)

    rc = swarm.run_swarm(_args(tmp_path, worker=["ok", "bad"], read_only=True))

    assert rc == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "completed_with_failures"
    assert status["workers"]["ok"]["status"] == "completed"
    assert status["workers"]["bad"]["status"] == "failed"
    assert status["workers"]["bad"]["exit_code"] == 7
    assert (tmp_path / "agents" / "ok.out.md").read_text(encoding="utf-8") == "out:ok\n"
    assert (tmp_path / "logs" / "bad.err.log").read_text(encoding="utf-8") == "err:bad\n"


def test_status_swarm_prints_existing_status(tmp_path: Path, capsys):
    (tmp_path / "status.json").write_text(
        json.dumps({"overall_status": "completed"}), encoding="utf-8"
    )

    rc = swarm.status_swarm(argparse.Namespace(run_dir=str(tmp_path), status_file=None))

    assert rc == 0
    assert '"overall_status": "completed"' in capsys.readouterr().out
