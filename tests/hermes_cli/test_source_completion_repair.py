"""A failed product remains owed; repairing it finishes without moving HEAD."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import source_completion, venv_sync


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("install-stamp.json\n", encoding="utf-8")
    for args in (("init", "-q"), ("config", "user.name", "test"),
                 ("config", "user.email", "test@example.invalid"), ("add", ".gitignore"),
                 ("commit", "-q", "-m", "release")):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    return root, head


def _tail(monkeypatch, build):
    monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", lambda root: None)
    monkeypatch.setattr("hermes_cli.source_build.build_update_products", build)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._run_post_update_maintenance", lambda **kwargs: True)


def test_failed_build_never_announces_success(tmp_path, monkeypatch, capsys):
    root, _ = _repo(tmp_path)

    def broken(root, *, desktop):
        raise RuntimeError("TypeScript build failed: TS1484")

    _tail(monkeypatch, broken)
    owed = []
    assert not source_completion.complete_source_checkout(
        root, desktop=False, assume_yes=True, announce="Code updated!", followups=owed)
    assert "Code updated!" not in capsys.readouterr().out
    assert owed == [("build", "TypeScript build failed: TS1484")]
    assert not (root / "install-stamp.json").exists()


def test_repair_clears_pending_and_exhausted_retries_without_changing_head(tmp_path, monkeypatch, capsys):
    from pm import environments
    from hermes_cli import update_custody

    root, head = _repo(tmp_path)
    state = tmp_path / "state"
    monkeypatch.setattr(environments, "install_state_dir", lambda root: state)
    monkeypatch.setattr(environments, "activation_environment", lambda root: {})
    pending = venv_sync.arm_completion(root)
    attempts = pending.with_name("source-completion-attempts")
    builds = []

    def build(root, *, desktop):
        builds.append(root)
        if len(builds) == 1:
            raise RuntimeError("TypeScript build failed: TS1484")

    _tail(monkeypatch, build)

    def completion_child(*args, **kwargs):
        owed = []
        source_completion.complete_source_checkout(
            root, desktop=False, assume_yes=True, announce="Code updated!", followups=owed)
        return SimpleNamespace(returncode=int(bool(owed)))

    monkeypatch.setattr(update_custody, "run", completion_child)
    with pytest.raises(RuntimeError, match="source update completion failed"):
        venv_sync._finish_source_update(root, current=True, pending=pending)
    assert pending.exists() and not (root / "install-stamp.json").exists()
    attempts.write_text(str(venv_sync.COMPLETION_RETRY_MAX_ATTEMPTS), encoding="utf-8")
    assert venv_sync.completion_retry_state(root)[0] is False

    # An explicit completion call finishes the pending update without a user update command or launch retry.
    venv_sync._finish_source_update(root, current=True, pending=pending)
    assert not pending.exists() and not attempts.exists()
    stamp = json.loads((root / "install-stamp.json").read_text(encoding="utf-8"))
    assert stamp["commit"] == head
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip() == head
    assert capsys.readouterr().out.count("Code updated!") == 1

    # A leftover marker naming this already completed HEAD is removed quietly, without another build.
    pending = venv_sync.arm_completion(root)
    older = (root / "install-stamp.json").stat().st_mtime - 10
    os.utime(pending, (older, older))
    venv_sync._finish_source_update(root, current=True, pending=pending)
    assert not pending.exists() and len(builds) == 2
    assert capsys.readouterr().out == ""
