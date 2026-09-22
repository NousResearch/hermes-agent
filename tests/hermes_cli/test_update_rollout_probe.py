"""Behavioral coverage for the read-only update rollout observation protocol."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import update_rollout_probe as probe


SHA = "a" * 40
INSTALL_ID = "b" * 32


def _patch_common(monkeypatch, root: Path, *, identity_text: str | None):
    monkeypatch.setattr(probe, "get_default_hermes_root", lambda: root)
    monkeypatch.setattr(probe, "_CODE_ROOT", root / "checkout")
    monkeypatch.setattr(probe, "_read_git_metadata", lambda _root: {
        "origin": "https://example.test/hermes.git",
        "trackingBranch": "origin/main",
        "checkoutSha": SHA,
        "checkoutState": "clean",
    })
    monkeypatch.setattr(probe, "_read_image_marker", lambda: {"state": "unavailable"})
    monkeypatch.setattr(probe, "_read_runtime_evidence", lambda: {
        "generation": 7,
        "requiredScopes": [],
        "processes": [{"pid": 123, "generation": 7, "identity": "gateway"}],
        "processGeneration": {"state": "matched", "observed": 7},
    })
    monkeypatch.setattr(probe, "_read_recovery", lambda: {
        "state": "clear",
        "markers": {"updateIncomplete": {"state": "unavailable"}},
    })
    monkeypatch.setattr(probe, "_dependency_evidence", lambda: {
        "state": "ready", "python": "fixture", "checked": [], "missing": [],
    })
    if identity_text is not None:
        (root / "install_id").write_text(identity_text, encoding="utf-8")


def _snapshot_files(root: Path):
    return sorted(
        (path.relative_to(root), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    )


def test_existing_raw_identity_is_observed_without_mutation(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path, identity_text=INSTALL_ID + "\n")
    before = _snapshot_files(tmp_path)

    result = probe.collect_observation(mode="inventory")

    after = _snapshot_files(tmp_path)
    assert result["metadata"] == {"protocol": 1}
    assert result["observation"]["installId"] == INSTALL_ID
    assert result["observation"]["codeRoot"] == str(tmp_path / "checkout")
    assert result["observation"]["checkoutSha"] == SHA
    assert before == after


def test_probe_never_emits_credentials_embedded_in_origin(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path, identity_text=INSTALL_ID)
    monkeypatch.setattr(probe, "_read_git_metadata", lambda _root: {
        "origin": "https://operator:probe-secret@example.test/hermes.git",
        "trackingBranch": "origin/main", "checkoutSha": SHA,
        "checkoutState": "clean",
    })

    rendered = json.dumps(probe.collect_observation())
    assert "probe-secret" not in rendered
    assert "operator" not in rendered
    assert "https://example.test/hermes.git" in rendered


@pytest.mark.parametrize("missing", ["identity", "recovery", "generation"])
def test_readiness_refuses_missing_identity_recovery_or_generation(
    tmp_path, monkeypatch, missing,
):
    _patch_common(
        monkeypatch, tmp_path,
        identity_text=None if missing == "identity" else INSTALL_ID,
    )
    monkeypatch.setattr(probe, "_read_image_marker", lambda: {"state": "absent"})
    monkeypatch.setattr(probe, "_read_recovery", lambda: {
        "state": "live" if missing == "recovery" else "clear", "markers": {},
    })
    if missing == "generation":
        monkeypatch.setattr(probe, "_read_runtime_evidence", lambda: {
            "generation": None, "requiredScopes": None, "processes": [],
            "processGeneration": {"state": "unknown", "observed": None},
        })

    observation = probe.collect_observation(mode="health")["observation"]
    assert observation["readiness"]["state"] == "blocked"


@pytest.mark.parametrize("identity_text", [None, "not-an-install-id\n", "B" * 32 + "\n"])
def test_missing_or_invalid_raw_identity_stays_unknown(tmp_path, monkeypatch, identity_text):
    _patch_common(monkeypatch, tmp_path, identity_text=identity_text)
    result = probe.collect_observation(mode="health")
    assert result["observation"]["installId"] is None
    assert not (tmp_path / "install_id").exists() if identity_text is None else True


def test_marker_states_and_runtime_evidence_are_distinct(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path, identity_text=INSTALL_ID)
    monkeypatch.setattr(probe, "_read_image_marker", lambda: {
        "state": "malformed", "reason": "invalid-json"
    })
    monkeypatch.setattr(probe, "_read_recovery", lambda: {
        "state": "live", "markers": [".update-incomplete"]
    })
    monkeypatch.setattr(probe, "_read_runtime_evidence", lambda: {
        "generation": None,
        "requiredScopes": None,
        "processes": [],
        "processGeneration": {"state": "unknown", "observed": None},
    })
    monkeypatch.setattr(probe, "_dependency_evidence", lambda: {
        "state": "ready", "python": "fixture", "checked": [], "missing": [],
    })

    observation = probe.collect_observation(mode="health")["observation"]

    assert observation["markers"]["imageProvenance"]["state"] == "malformed"
    assert observation["recovery"]["state"] == "live"
    assert observation["runtime"]["requiredScopes"] is None
    assert observation["runtime"]["generation"] is None


def test_stdout_is_one_json_object_and_diagnostics_use_stderr(tmp_path, monkeypatch, capsys):
    _patch_common(monkeypatch, tmp_path, identity_text=INSTALL_ID)
    assert probe.main(["--mode", "health", "--correlation-id", "test-1"]) == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["metadata"] == {"protocol": 1}
    assert payload["request"] == {"mode": "health", "correlationId": "test-1"}
    assert captured.err == ""


def _build_protocol_package(repo_root: Path, tmp_path: Path):
    build_lib = tmp_path / "build-lib"
    egg_base = tmp_path / "metadata"
    home = tmp_path / "home"
    egg_base.mkdir(parents=True)
    home.mkdir()
    # build-lib alone does not redirect egg_info, which build_py also runs.
    # Keep the actual project configuration; redirect both output owners.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "setup.py",
                "--no-user-cfg",
                "egg_info",
                f"--egg-base={egg_base}",
                "build_py",
                f"--build-lib={build_lib}",
            ],
            cwd=repo_root,
            env=dict(
                os.environ, HOME=str(home), USERPROFILE=str(home),
                PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"Package-data build timed out after {exc.timeout}s; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"Package-data build failed ({result.returncode}); "
            f"stdout={result.stdout}; stderr={result.stderr}",
            pytrace=False,
        )

    return build_lib / "hermes_cli" / "update_rollout_protocol.json"


def test_protocol_fixture_is_registered_as_package_data(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    packaged = _build_protocol_package(repo_root, tmp_path)
    protocol = json.loads(packaged.read_text(encoding="utf-8"))
    assert protocol == {"protocol": 1}
    assert set(protocol) == {"protocol"}
    assert type(protocol["protocol"]) is int
    assert protocol["protocol"] == 1


def test_protocol_package_build_leaves_source_unchanged(tmp_path, monkeypatch):
    # Exercise the real setup.py/config and package, but keep a regressed builder
    # from writing metadata into the developer's checkout even on the RED path.
    repo_root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    for path in repo_root.glob("*.py"):
        shutil.copy2(path, source / path.name)
    for name in ("pyproject.toml", "setup.cfg", "MANIFEST.in", "README.md", "LICENSE"):
        if (repo_root / name).is_file():
            shutil.copy2(repo_root / name, source / name)
    shutil.copytree(
        repo_root / "hermes_cli", source / "hermes_cli",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    before = _snapshot_files(source)
    real_run = subprocess.run

    def bounded_run(*args, **kwargs):
        # Bound the real regression process even if the harness loses its timeout.
        kwargs.setdefault("timeout", 60)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", bounded_run)
    packaged = _build_protocol_package(source, tmp_path / "output")
    assert packaged.read_bytes() == (source / "hermes_cli/update_rollout_protocol.json").read_bytes()
    after = _snapshot_files(source)
    added = sorted(str(path) for path in dict(after).keys() - dict(before).keys())
    assert before == after, f"Packaging modified its source tree; added: {added}"


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_protocol_package_build_reports_bounded_failure(tmp_path, monkeypatch, failure):
    def fail_build(command, **kwargs):
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, (int, float)) and math.isfinite(timeout) and 0 < timeout <= 60, (
            "Packaging subprocess must have a finite timeout of at most 60 seconds"
        )
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout, output=b"build started")
        return subprocess.CompletedProcess(command, 2, stdout="build started", stderr="build broken")

    monkeypatch.setattr(subprocess, "run", fail_build)
    message = "Package-data build timed out" if failure == "timeout" else "build broken"
    with pytest.raises(pytest.fail.Exception, match=message):
        _build_protocol_package(tmp_path, tmp_path / "output")
