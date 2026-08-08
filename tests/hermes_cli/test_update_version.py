from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd as hermes_update_cmd
from hermes_cli.subcommands.update import build_update_parser


def _handler(_args):  # pragma: no cover - parser identity only
    return None


def _update_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=_handler)
    return parser


def test_update_version_parses_and_excludes_branch():
    parser = _update_parser()

    ns = parser.parse_args(["update", "--version", "v2026.7.30"])
    assert ns.version == "v2026.7.30"
    assert ns.branch is None

    with pytest.raises(SystemExit):
        parser.parse_args([
            "update",
            "--branch",
            "release-candidate",
            "--version",
            "v2026.7.30",
        ])


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (SimpleNamespace(), ("branch", "main")),
        (
            SimpleNamespace(branch=" release-candidate "),
            ("branch", "release-candidate"),
        ),
        (SimpleNamespace(version=" v2026.7.30 "), ("tag", "v2026.7.30")),
    ],
)
def test_resolve_update_target(args, expected):
    assert hermes_main._resolve_update_target(args) == expected


def test_cmd_update_rejects_non_release_version_before_backup(monkeypatch):
    backup_called = False

    def fake_backup(_args):
        nonlocal backup_called
        backup_called = True

    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", fake_backup)

    with pytest.raises(SystemExit, match="1"):
        hermes_main._cmd_update_impl(
            SimpleNamespace(version="*", branch=None), gateway_mode=False
        )

    assert not backup_called


def test_update_check_version_fetches_and_compares_tag(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_config, "detect_install_method", lambda _root: "git")

    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        joined = " ".join(str(part) for part in cmd)
        if "fetch" in joined:
            assert "--no-tags" in cmd
            assert hermes_update_cmd.OFFICIAL_REPO_URL in cmd
            assert cmd[-1] == "+refs/tags/v2026.7.30:refs/tags/v2026.7.30"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if joined.endswith("rev-parse --verify --quiet refs/tags/v2026.7.30^{commit}"):
            return SimpleNamespace(returncode=0, stdout="tag-sha\n", stderr="")
        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if "rev-parse --abbrev-ref HEAD" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if "rev-parse --is-shallow-repository" in joined:
            return SimpleNamespace(returncode=0, stdout="false\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main._cmd_update_check(version="v2026.7.30")

    out = capsys.readouterr().out
    assert (
        "Update available: target version v2026.7.30 differs from current checkout."
        in out
    )
    assert any(
        "refs/tags/v2026.7.30:refs/tags/v2026.7.30" in " ".join(call) for call in calls
    )


def test_cmd_update_version_checks_out_detached_tag(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda _args: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(
        hermes_update_cmd, "_discard_lockfile_churn", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        hermes_update_cmd, "_normalize_managed_eol", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        hermes_main,
        "_get_origin_url",
        lambda *_a, **_k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        hermes_update_cmd,
        "_validate_critical_files_syntax",
        lambda _root: (False, "hermes_cli/main.py", "synthetic syntax failure"),
    )

    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        joined = " ".join(str(part) for part in cmd)
        if "fetch" in joined:
            assert "--no-tags" in cmd
            assert hermes_update_cmd.OFFICIAL_REPO_URL in cmd
            assert cmd[-1] == "+refs/tags/v2026.7.30:refs/tags/v2026.7.30"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "rev-parse --abbrev-ref HEAD" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if joined.endswith("rev-parse --verify --quiet refs/tags/v2026.7.30^{commit}"):
            # HEAD is attached to main at the release commit. Explicit
            # --version must still perform a detached checkout.
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if "checkout --detach refs/tags/v2026.7.30" in joined:
            return SimpleNamespace(
                returncode=0, stdout="HEAD is now at tag-sha\n", stderr=""
            )
        if "reset --hard old-sha" in joined:
            return SimpleNamespace(
                returncode=0, stdout="HEAD is now at old-sha\n", stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main._cmd_update_impl(
            SimpleNamespace(version="v2026.7.30", branch=None), gateway_mode=False
        )

    flattened = [" ".join(str(part) for part in call) for call in calls]
    assert any("checkout --detach refs/tags/v2026.7.30" in call for call in flattened)
    assert not any("merge --ff-only origin/" in call for call in flattened)
    assert not any("checkout -B" in call for call in flattened)


def test_update_zip_version_is_rejected_before_download(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda _root: 0)
    monkeypatch.setattr(hermes_main, "_record_bytecode_fingerprint", lambda: None)

    downloaded = False

    def fake_urlretrieve(url, path):
        nonlocal downloaded
        downloaded = True
        raise RuntimeError("stop after URL capture")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(SystemExit, match="1"):
        hermes_main._update_via_zip(SimpleNamespace(version="v2026.7.30", branch=None))

    out = capsys.readouterr().out
    assert "--version is not supported on the Windows ZIP-fallback" in out
    assert "hermes update --version <release>" in out
    assert not downloaded
