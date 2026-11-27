"""Part B: Catalog CI linter for ``default: true`` plugins that would trigger a PM venv rebuild."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hermes_yaml as yaml
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "catalog_validate_pinned_entries.py"
REPO_URL = "https://fixture.invalid/repo"

sys.path.insert(0, str(ROOT / "scripts" / "ci"))
from catalog_default_plugin_lint import check_default_plugin_venv_rebuild
import catalog_validate_pinned_entries as pinned_gate


def _write_plugin(plugin_dir: Path, *, pyproject: str | None = None, manifest: str | None = None) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        manifest if manifest is not None else "name: fixture-plugin\n",
        encoding="utf-8",
    )
    if pyproject is not None:
        (plugin_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return plugin_dir


# ── Reusable check_default_plugin_venv_rebuild unit tests ────────────────────


@pytest.mark.parametrize("deps", [
    ["cryptography>=50,<51"],
    ["httpx>=0.27,<1", "PyJWT[crypto]>=2.8", "cryptography>=50,<51"],
    [],
], ids=["mobile-connector-shape", "google-shape", "empty-deps"])
def test_core_satisfied_default_plugin_passes_cleanly(tmp_path, deps):
    plugin = _write_plugin(
        tmp_path / "ok-default",
        pyproject=(
            '[project]\nname = "ok-default"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            f"dependencies = {json.dumps(deps)}\n"
        ),
    )
    result = check_default_plugin_venv_rebuild(plugin, project_root=ROOT)
    assert result.triggers_rebuild is False
    assert result.unsatisfied_requirements == []
    assert result.pyproject_installs is False
    assert result.reason == ""


def test_unsatisfied_dep_triggers_rebuild_diagnostic(tmp_path):
    plugin = _write_plugin(
        tmp_path / "langfuse-default",
        pyproject=(
            '[project]\nname = "langfuse-default"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51", "langfuse==4.15.2"]\n'
        ),
    )
    result = check_default_plugin_venv_rebuild(plugin, project_root=ROOT)
    assert result.triggers_rebuild is True
    assert result.unsatisfied_requirements == ["langfuse==4.15.2"]
    assert result.pyproject_installs is False
    assert "langfuse==4.15.2" in result.reason
    assert "uv-cache" in result.guidance
    assert "pyproject.toml" in result.guidance
    assert "[build-system]" in result.guidance
    assert "httpx" in result.guidance
    assert "cryptography" in result.guidance


@pytest.mark.parametrize("extra_toml, expected_token", [
    (
        '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n',
        "[build-system]",
    ),
    (
        "[tool.uv]\npackage = true\n",
        "tool.uv.package = true",
    ),
], ids=["build-system", "uv-package-true"])
def test_pyproject_installs_triggers_rebuild_diagnostic(tmp_path, extra_toml, expected_token):
    plugin = _write_plugin(
        tmp_path / "buildable-default",
        pyproject=(
            '[project]\nname = "buildable-default"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["cryptography>=50,<51"]\n' + extra_toml
        ),
    )
    result = check_default_plugin_venv_rebuild(plugin, project_root=ROOT)
    assert result.triggers_rebuild is True
    assert result.pyproject_installs is True
    assert result.unsatisfied_requirements == []
    assert expected_token in result.reason
    assert "[build-system]" in result.guidance


def test_windows_only_unsatisfied_dep_is_caught_on_linux_ci(tmp_path):
    plugin = _write_plugin(
        tmp_path / "win-only",
        pyproject=(
            '[project]\nname = "win-only"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["win-only-third-party>=1; sys_platform == \'win32\'"]\n'
        ),
    )
    result = check_default_plugin_venv_rebuild(plugin, project_root=ROOT)
    assert result.triggers_rebuild is True
    assert result.unsatisfied_requirements == ["win-only-third-party>=1; sys_platform == 'win32'"]


def test_ci_default_check_uses_real_pinned_python_version_from_lock_json(tmp_path):
    """Finding #5: CI check uses the pinned Python version from ``pm/lock.json`` (e.g. 3.14.9),
    not hardcoded 3.14.0, so ``python_full_version >= '3.14.1'`` triggers on Windows."""
    plugin = _write_plugin(
        tmp_path / "win-py314-patch",
        pyproject=(
            '[project]\nname = "win-py314-patch"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
            'dependencies = ["fixturedep>=1; sys_platform == \'win32\' and python_full_version >= \'3.14.1\'"]\n'
        ),
    )
    result = check_default_plugin_venv_rebuild(plugin, project_root=ROOT, platforms=["windows"])
    assert result.triggers_rebuild is True
    assert result.unsatisfied_requirements == [
        "fixturedep>=1; sys_platform == 'win32' and python_full_version >= '3.14.1'"
    ]


def test_gha_escape_helpers():
    """GitHub Actions workflow-command property and data escaping runs on every platform."""
    assert pinned_gate._escape_gha_property("a%b:c,d\re\nf") == "a%25b%3Ac%2Cd%0De%0Af"
    assert pinned_gate._escape_gha_data("100%\r\nline:two,ok") == "100%25%0D%0Aline:two,ok"


# ── End-to-end catalog_validate_pinned_entries.py gate tests (Linux CI only) ─
# catalog_validate_pinned_entries.py runs on ubuntu-latest and invokes GNU
# `timeout`, `grep -rlZE`, and `xargs -r0` (matching test_plugin_catalog_ci_confinement.py).


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, msg: str = "x") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def _run_pinned_gate(
    tmp_path: Path,
    origin: Path,
    entry: Path,
    *,
    extra_args: list[str] = (),
) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "hermes"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    mktmp = tmp_path / "mktmp"
    mktmp.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "CHANGED_FILES": str(entry),
        "TMPDIR": str(mktmp),
        "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "GIT_TERMINAL_PROMPT": "0",
        "GITHUB_WORKSPACE": str(ROOT),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{origin.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": REPO_URL,
    }
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *extra_args],
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _make_origin(tmp_path: Path, pyproject: str) -> tuple[Path, str]:
    repo = tmp_path / "origin"
    (repo / "plugin").mkdir(parents=True)
    (repo / "plugin" / "plugin.yaml").write_text("name: fixture\n", encoding="utf-8")
    (repo / "plugin" / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    _git(repo, "init", "-q")
    return repo, _commit(repo)


def _write_catalog_entry(tmp_path: Path, name: str, sha: str, **extra) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"name": name, "repo": REPO_URL, "sha": sha, "subdir": "plugin", **extra}),
        encoding="utf-8",
    )
    return path


@pytest.mark.platforms("linux")
def test_ci_default_entry_with_core_satisfied_deps_passes_with_zero_warnings(tmp_path):
    origin, sha = _make_origin(
        tmp_path,
        '[project]\nname = "mobile"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["cryptography>=50,<51"]\n',
    )
    entry = _write_catalog_entry(tmp_path, "mobile", sha, tier="official", default=True)
    res = _run_pinned_gate(tmp_path, origin, entry)
    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined
    assert f"PASS: {entry}" in res.stdout
    assert "::warning" not in combined


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("pyproject, expected_trigger", [
    (
        '[project]\nname = "obs"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["langfuse==4.15.2"]\n',
        "langfuse==4.15.2",
    ),
    (
        '[project]\nname = "buildable"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = []\n'
        '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n',
        "[build-system]",
    ),
], ids=["unsatisfied-dep", "build-system"])
def test_ci_default_entry_triggering_rebuild_emits_actionable_warning(
    tmp_path, pyproject, expected_trigger
):
    origin, sha = _make_origin(tmp_path, pyproject)
    entry = _write_catalog_entry(tmp_path, "heavy", sha, tier="official", default=True)
    res = _run_pinned_gate(tmp_path, origin, entry)
    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined
    assert f"PASS: {entry}" in res.stdout
    warning_lines = [line for line in res.stdout.splitlines() if line.startswith(f"::warning file={entry}::")]
    assert len(warning_lines) == 1, combined
    warning = warning_lines[0]
    assert expected_trigger in warning
    assert "uv-cache" in warning
    assert "pyproject.toml" in warning
    assert "httpx" in warning
    assert "[build-system]" in warning

    # Blocking switch turns the same trigger into a hard CI failure.
    strict = _run_pinned_gate(tmp_path, origin, entry, extra_args=["--fail-on-default-rebuild"])
    assert strict.returncode == 1, strict.stdout + strict.stderr
    assert f"::error file={entry}::" in strict.stdout
    assert expected_trigger in strict.stdout


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("default_fields", [{}, {"default": False}], ids=["omitted", "default-false"])
def test_ci_non_default_entry_with_unsatisfied_dep_emits_no_default_rebuild_warning(
    tmp_path, default_fields
):
    origin, sha = _make_origin(
        tmp_path,
        '[project]\nname = "opt"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["langfuse==4.15.2"]\n',
    )
    entry = _write_catalog_entry(tmp_path, "opt", sha, tier="official", **default_fields)
    res = _run_pinned_gate(tmp_path, origin, entry)
    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined
    assert f"PASS: {entry}" in res.stdout
    assert "::warning" not in combined


@pytest.mark.platforms("linux")
def test_ci_default_rebuild_warning_escapes_gha_property_and_data_chars(tmp_path):
    """Finding #6: ``::warning`` escapes ``%``, ``\\r``, ``\\n``, ``:``, ``,`` in ``file=`` and
    ``%``, ``\\r``, ``\\n`` in the message body per GitHub Actions workflow-command syntax."""
    origin, sha = _make_origin(
        tmp_path,
        '[project]\nname = "obs"\nversion = "1.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["langfuse==4.15.2"]\n',
    )
    entry = tmp_path / "heavy,v2.yaml"
    entry.write_text(
        yaml.safe_dump({"name": "heavy", "repo": REPO_URL, "sha": sha, "subdir": "plugin", "tier": "official", "default": True}),
        encoding="utf-8",
    )
    res = _run_pinned_gate(tmp_path, origin, entry)
    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined
    escaped_file = (
        str(entry)
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )
    warning_lines = [line for line in res.stdout.splitlines() if line.startswith("::warning ")]
    assert len(warning_lines) == 1, combined
    assert warning_lines[0].startswith(f"::warning file={escaped_file}::"), warning_lines[0]
    assert "heavy%2Cv2.yaml" in warning_lines[0]
    assert "heavy,v2.yaml" not in warning_lines[0].split("::", 2)[1]
