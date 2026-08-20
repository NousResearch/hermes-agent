import json
import subprocess
from argparse import ArgumentParser
from types import SimpleNamespace
from pathlib import Path

import pytest

from hermes_cli import main as cli_main
from hermes_cli.main import _downstream_update_guard_status
from hermes_cli.main import _enforce_downstream_update_guard
from hermes_cli.subcommands.update import build_update_parser


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path) -> tuple[str, str]:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "base.txt").write_text("base\n")
    _git(root, "add", "base.txt")
    _git(root, "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "stage1.txt").write_text("stage1\n")
    _git(root, "add", "stage1.txt")
    _git(root, "commit", "-qm", "stage1")
    return base, _git(root, "rev-parse", "HEAD")


def _guard(root: Path, anchor: str) -> None:
    (root / ".hermes-update-guard.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "guards": [
                    {
                        "guard_id": "stage1",
                        "anchor_commit": anchor,
                    }
                ],
            }
        )
    )


def test_update_guard_blocks_until_anchor_is_in_remote_target(tmp_path):
    base, anchor = _repository(tmp_path)
    _guard(tmp_path, anchor)
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    assert _downstream_update_guard_status(tmp_path, "main") == ["stage1"]
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", anchor)
    assert _downstream_update_guard_status(tmp_path, "main") == []


def test_update_guard_requires_explicit_operator_override(tmp_path, capsys):
    base, anchor = _repository(tmp_path)
    _guard(tmp_path, anchor)
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)

    with pytest.raises(SystemExit) as exc:
        _enforce_downstream_update_guard(tmp_path, "main")
    assert exc.value.code == 2

    _enforce_downstream_update_guard(
        tmp_path,
        "main",
        force_downstream_guard=True,
    )
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "stage1" in output


def test_update_guard_override_is_explicit_for_invalid_manifest(tmp_path, capsys):
    _repository(tmp_path)
    (tmp_path / ".hermes-update-guard.json").write_text("{}")

    _enforce_downstream_update_guard(
        tmp_path,
        "main",
        force_downstream_guard=True,
    )
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "invalid" in output


def test_update_guard_override_requires_named_cli_flag():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda _args: None)

    assert parser.parse_args(["update"]).force_downstream_guard is False
    assert (
        parser.parse_args(
            ["update", "--force-downstream-guard"]
        ).force_downstream_guard
        is True
    )


def test_update_guard_allows_history_after_exact_anchor_is_rebased_away(tmp_path):
    base, anchor = _repository(tmp_path)
    _guard(tmp_path, anchor)
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    _git(tmp_path, "reset", "--hard", base)
    assert _downstream_update_guard_status(tmp_path, "main") == []


def test_update_guard_fails_closed_on_invalid_manifest(tmp_path):
    _repository(tmp_path)
    (tmp_path / ".hermes-update-guard.json").write_text("{}")
    with pytest.raises(RuntimeError, match="invalid"):
        _downstream_update_guard_status(tmp_path, "main")


def test_update_guard_fails_closed_when_missing(tmp_path):
    _repository(tmp_path)
    with pytest.raises(RuntimeError, match="guard is missing"):
        _downstream_update_guard_status(tmp_path, "main")


def test_update_guard_fails_closed_when_deleted(tmp_path):
    _, anchor = _repository(tmp_path)
    _guard(tmp_path, anchor)
    (tmp_path / ".hermes-update-guard.json").unlink()
    with pytest.raises(RuntimeError, match="guard is missing"):
        _downstream_update_guard_status(tmp_path, "main")


def test_update_guard_fails_closed_when_symlinked(tmp_path):
    _, anchor = _repository(tmp_path)
    real = tmp_path / "guard-target.json"
    _guard(tmp_path, anchor)
    (tmp_path / ".hermes-update-guard.json").replace(real)
    (tmp_path / ".hermes-update-guard.json").symlink_to(real)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        _downstream_update_guard_status(tmp_path, "main")


def test_update_guard_fails_closed_without_git_metadata(tmp_path):
    _guard(tmp_path, "1" * 40)
    with pytest.raises(RuntimeError, match="Git metadata is missing"):
        _downstream_update_guard_status(tmp_path, "main")


def test_update_guard_fails_closed_on_permission_error(tmp_path, monkeypatch):
    _, anchor = _repository(tmp_path)
    guard = tmp_path / ".hermes-update-guard.json"
    _guard(tmp_path, anchor)
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == guard:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(RuntimeError, match="guard is unreadable"):
        _downstream_update_guard_status(tmp_path, "main")


def test_update_guard_blocks_before_service_pause_or_checkout_mutation(
    tmp_path, monkeypatch
):
    class GuardReached(Exception):
        pass

    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_main, "_is_windows", lambda: False)
    monkeypatch.setattr(cli_main, "_run_pre_update_backup", lambda _args: None)
    monkeypatch.setattr(
        cli_main,
        "_enforce_downstream_update_guard",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GuardReached()),
    )
    monkeypatch.setattr(
        cli_main,
        "_pause_windows_gateways_for_update",
        lambda: pytest.fail("services paused before downstream guard"),
    )
    with pytest.raises(GuardReached):
        cli_main._cmd_update_impl(
            SimpleNamespace(
                yes=True,
                force=False,
                force_venv=False,
                force_downstream_guard=False,
                branch=None,
            ),
            gateway_mode=False,
        )
