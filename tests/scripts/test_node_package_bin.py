from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-node-package-bin.mjs"


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    return node


def test_resolves_hoisted_package_bin_from_nested_workspace(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "apps" / "desktop"
    package = root / "node_modules" / "fake-tool"
    nested.mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"name": "fake-tool", "bin": {"fake": "bin/cli.mjs"}}),
        encoding="utf-8",
    )
    cli = package / "bin" / "cli.mjs"
    cli.parent.mkdir()
    cli.write_text(
        "console.log(JSON.stringify({cwd: process.cwd(), args: process.argv.slice(2)}))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            _node(),
            str(RUNNER),
            "fake-tool",
            "--bin",
            "fake",
            "--cwd",
            str(nested),
            "--",
            "alpha",
            "beta",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert Path(payload["cwd"]) == nested
    assert payload["args"] == ["alpha", "beta"]


def test_executes_native_package_bin_directly(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "ui-tui"
    package = root / "node_modules" / "native-tool"
    nested.mkdir(parents=True)
    package.mkdir(parents=True)

    node = Path(_node())
    suffix = node.suffix if node.suffix else ""
    bin_relative = f"bin/native{suffix}"
    native = package / bin_relative
    native.parent.mkdir()
    shutil.copy2(node, native)
    (package / "package.json").write_text(
        json.dumps({"name": "native-tool", "bin": {"native": bin_relative}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            _node(),
            str(RUNNER),
            "native-tool",
            "--bin",
            "native",
            "--cwd",
            str(nested),
            "--",
            "-e",
            "process.stdout.write('native-ok')",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "native-ok"


def test_missing_package_fails_without_using_path_shims(tmp_path: Path) -> None:
    result = subprocess.run(
        [_node(), str(RUNNER), "definitely-not-installed", "--cwd", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 127
    assert "is not installed" in result.stderr
