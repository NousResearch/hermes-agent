"""Behavioral tests for the SQLite backend import guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_backend_imports.py"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_clean_root_passes(tmp_path):
    _write(tmp_path, {"gateway/thing.py": "from persistence_assembly import open_persistence\n"})

    result = _run(tmp_path)

    assert result.returncode == 0, result.stdout
    assert "No persistence.sqlite imports" in result.stdout


@pytest.mark.parametrize("source", [
    "from persistence.sqlite.conversation.messages import SQLiteMessageRepository\n",
    "import persistence.sqlite\n",
    "import persistence.sqlite.gateway as backend\n",
    "from persistence import sqlite\n",
    "def load():\n    from persistence.sqlite.gateway import hosted_room_attachments\n",
    "import importlib\nbackend = importlib.import_module('persistence.sqlite.gateway')\n",
])
def test_backend_import_outside_the_storage_layer_fails(tmp_path, source):
    _write(tmp_path, {"evals/probe.py": source})

    result = _run(tmp_path)

    assert result.returncode == 1, result.stdout
    assert "evals/probe.py:" in result.stdout
    assert "persistence_assembly" in result.stdout


@pytest.mark.parametrize("rel", [
    "persistence/sqlite/conversation/messages.py",
    "persistence/assembly_helpers.py",
    "persistence_assembly.py",
    "tests/persistence_sqlite/test_store.py",
    "plugins/tests/test_plugin.py",
])
def test_the_storage_layer_and_tests_may_import_the_backend(tmp_path, rel):
    _write(tmp_path, {rel: "from persistence.sqlite import SQLiteBackend\n"})

    assert _run(tmp_path).returncode == 0


@pytest.mark.parametrize("source", [
    "# from persistence.sqlite import SQLiteBackend\n",
    "DOC = 'from persistence.sqlite import SQLiteBackend'\n",
    "from persistence import open_store\n",
    "import persistence.sqlite_helpers\n",
])
def test_mentions_and_neighbouring_names_pass(tmp_path, source):
    _write(tmp_path, {"hermes_cli/thing.py": source})

    assert _run(tmp_path).returncode == 0


def test_each_offending_line_is_named(tmp_path):
    _write(tmp_path, {
        "evals/a.py": "import os\nfrom persistence.sqlite import _holder\n",
        "tools/b.py": "import persistence.sqlite\n",
    })

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "evals/a.py:2" in result.stdout
    assert "tools/b.py:1" in result.stdout
