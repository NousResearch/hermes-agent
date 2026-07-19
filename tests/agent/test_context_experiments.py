"""Tests for session-scoped context experiments."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from agent import context_experiments
from agent.prompt_builder import build_context_files_prompt


class _FakeMsvcrt:
    LK_LOCK = 1
    LK_UNLCK = 2

    def __init__(self, *, fail_lock=False):
        self.fail_lock = fail_lock
        self.calls = []

    def locking(self, fd, mode, size):
        self.calls.append((mode, size, os.lseek(fd, 0, os.SEEK_CUR)))
        if self.fail_lock and mode == self.LK_LOCK:
            raise OSError("lock unavailable")


def test_msvcrt_lock_seeds_one_byte_and_uses_offset_zero(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(context_experiments, "fcntl", None)
    monkeypatch.setattr(context_experiments, "msvcrt", fake_msvcrt)

    context_experiments._round_robin_arm("experiment", "session-a", ["old", "new"])

    lock_path = home / "context_experiments" / "experiment.json.lock"
    assert lock_path.read_bytes() == b"\x00"
    assert [call[0] for call in fake_msvcrt.calls] == [
        fake_msvcrt.LK_LOCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert [call[2] for call in fake_msvcrt.calls] == [0, 0]


def test_msvcrt_lock_rewinds_existing_lock_file_before_locking(
    monkeypatch, tmp_path
):
    home = tmp_path / "home"
    lock_dir = home / "context_experiments"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "experiment.json.lock"
    lock_path.write_bytes(b"\x00")
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(context_experiments, "fcntl", None)
    monkeypatch.setattr(context_experiments, "msvcrt", fake_msvcrt)

    context_experiments._round_robin_arm("experiment", "session-a", ["old", "new"])

    assert [call[2] for call in fake_msvcrt.calls] == [0, 0]


def test_msvcrt_lock_failure_does_not_read_modify_write(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake_msvcrt = _FakeMsvcrt(fail_lock=True)
    monkeypatch.setattr(context_experiments, "fcntl", None)
    monkeypatch.setattr(context_experiments, "msvcrt", fake_msvcrt)

    with pytest.raises(OSError, match="lock unavailable"):
        context_experiments._round_robin_arm(
            "experiment", "session-a", ["old", "new"]
        )

    state_path = home / "context_experiments" / "experiment.json"
    assert not state_path.exists()
    assert [call[0] for call in fake_msvcrt.calls] == [fake_msvcrt.LK_LOCK]


@pytest.mark.skipif(
    context_experiments.fcntl is None,
    reason="POSIX fcntl/flock required",
)
def test_round_robin_assignment_is_cross_process_safe(tmp_path, monkeypatch):
    home = tmp_path / "home"
    ready = tmp_path / "first-writing"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo_root = Path(__file__).resolve().parents[2]
    worker = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        import sys
        import time

        sys.path.insert(0, {str(repo_root)!r})
        from agent import context_experiments
        from utils import atomic_json_write as real_atomic_json_write

        if sys.argv[1] == "session-a":
            def delayed_atomic_json_write(*args, **kwargs):
                Path(os.environ["READY"]).write_text("1")
                time.sleep(0.5)
                return real_atomic_json_write(*args, **kwargs)

            context_experiments.atomic_json_write = delayed_atomic_json_write

        arm = context_experiments._round_robin_arm(
            "experiment", sys.argv[1], ["old", "new"]
        )
        print(arm, flush=True)
        """
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["READY"] = str(ready)

    first = subprocess.Popen(
        [sys.executable, "-c", worker, "session-a"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "first process never entered the write"

        second = subprocess.Popen(
            [sys.executable, "-c", worker, "session-b"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        first_output, first_error = first.communicate(timeout=10)
        second_output, second_error = second.communicate(timeout=10)
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

    assert first.returncode == 0, first_error
    assert second.returncode == 0, second_error
    assert first_output.strip() == "old"
    assert second_output.strip() == "new"
    state = json.loads(
        (home / "context_experiments" / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["assignments"] == {"session-a": "old", "session-b": "new"}


def _experiment_config(old_file: Path, new_file: Path, *, new_skills=None):
    return {
        "context_experiments": {
            "agentsmd-split": {
                "enabled": True,
                "assignment": "round_robin",
                "arms": {
                    "old": {"context_file": str(old_file)},
                    "new": {
                        "context_file": str(new_file),
                        "skills": list(new_skills or []),
                    },
                },
            }
        }
    }


def test_context_experiment_round_robin_replaces_default_project_context(
    monkeypatch, tmp_path
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "AGENTS.md").write_text("DEFAULT AGENTS", encoding="utf-8")
    old_file = tmp_path / "AGENTS.old.md"
    new_file = tmp_path / "AGENTS.slim.md"
    old_file.write_text("OLD RULES", encoding="utf-8")
    new_file.write_text("NEW RULES", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _experiment_config(old_file, new_file),
    )

    first = build_context_files_prompt(cwd=str(tmp_path), session_id="session-a")
    second = build_context_files_prompt(cwd=str(tmp_path), session_id="session-b")
    first_again = build_context_files_prompt(cwd=str(tmp_path), session_id="session-a")

    assert "OLD RULES" in first
    assert "NEW RULES" not in first
    assert "NEW RULES" in second
    assert "OLD RULES" not in second
    assert first_again == first
    assert "DEFAULT AGENTS" not in first + second


def test_context_experiment_arm_can_preload_skills(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    old_file = tmp_path / "AGENTS.old.md"
    new_file = tmp_path / "AGENTS.slim.md"
    old_file.write_text("OLD RULES", encoding="utf-8")
    new_file.write_text("NEW RULES", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _experiment_config(old_file, new_file, new_skills=["eval-skill"]),
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        lambda skills, task_id=None: (
            "[loaded skill body]",
            ["eval-skill"],
            [],
        ),
    )

    # First new session receives old; second receives new + skill preload.
    build_context_files_prompt(cwd=str(tmp_path), session_id="session-a")
    prompt = build_context_files_prompt(cwd=str(tmp_path), session_id="session-b")

    assert "NEW RULES" in prompt
    assert "[loaded skill body]" in prompt
