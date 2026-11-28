"""Update holds: an integration that patches the checkout refuses in-place update until it qualifies the next base."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hermes_cli.update_contract import evaluate_update_admission, update_holds_dir


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _hold(root: Path, name: str, value) -> Path:
    directory = update_holds_dir(root)
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return path


def test_holds_live_in_the_git_dir_outside_the_working_tree(tmp_path):
    root = _checkout(tmp_path)
    assert update_holds_dir(root) == root / ".git" / "hermes-update-holds"


def test_no_checkout_has_no_hold_directory(tmp_path):
    assert update_holds_dir(tmp_path) is None


def test_valid_hold_refuses_with_its_holder_and_message(tmp_path):
    root = _checkout(tmp_path)
    _hold(root, "lifeos", {"holder": "LifeOS", "message": "Remove LifeOS before updating Hermes."})
    refusal = evaluate_update_admission(root)
    assert refusal is not None and refusal.code == "update-hold"
    assert "LifeOS: Remove LifeOS before updating Hermes." in refusal.message


def test_every_hold_is_named(tmp_path):
    root = _checkout(tmp_path)
    _hold(root, "a", {"holder": "First", "message": "one"})
    _hold(root, "b", {"holder": "Second", "message": "two"})
    message = evaluate_update_admission(root).message
    assert "First: one" in message and "Second: two" in message


def test_malformed_hold_fails_closed(tmp_path):
    root = _checkout(tmp_path)
    for value in ("not json", {"holder": "LifeOS"}, {"holder": "", "message": "x"},
                  {"holder": "LifeOS", "message": "x" * 3000}):
        path = _hold(root, "lifeos", value)
        refusal = evaluate_update_admission(root)
        assert refusal is not None and refusal.code == "update-hold-invalid", value
        path.unlink()


def test_hold_path_that_is_not_a_directory_fails_closed(tmp_path):
    root = _checkout(tmp_path)
    update_holds_dir(root).write_text("{}", encoding="utf-8")
    assert evaluate_update_admission(root).code == "update-hold-invalid"


def test_empty_hold_directory_and_other_files_do_not_hold(tmp_path):
    root = _checkout(tmp_path)
    update_holds_dir(root).mkdir()
    (update_holds_dir(root) / "README").write_text("notes", encoding="utf-8")
    assert evaluate_update_admission(root) is None


def test_linked_worktree_reads_the_shared_hold(tmp_path):
    root = _checkout(tmp_path)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                    "commit", "-q", "--allow-empty", "-m", "base"], check=True)
    linked = tmp_path / "linked"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "-q", "--detach", str(linked)], check=True)
    _hold(root, "lifeos", {"holder": "LifeOS", "message": "held"})
    assert evaluate_update_admission(linked).code == "update-hold"
