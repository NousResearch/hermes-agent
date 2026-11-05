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
    "from persistence import core, sqlite\n",
    "from persistence import  sqlite\n",
    "from persistence import (\n    core,\n    sqlite,\n    conversation,\n)\n",
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


@pytest.mark.parametrize("source", [
    "from hermes_constants import get_hermes_home\n",
    "import hermes_constants\n\ndef root():\n    return hermes_constants.get_default_hermes_root()\n",
    "def config():\n    from hermes_cli.config import load_config_readonly\n    return load_config_readonly()\n",
    "from hermes_state import SessionDB\n",
    "import hermes_state\n",
])
def test_a_backend_module_reading_the_process_home_or_the_stand_in_fails(tmp_path, source):
    _write(tmp_path, {"persistence/sqlite/conversation/_store/turns.py": source})

    result = _run(tmp_path)

    assert result.returncode == 1, result.stdout
    assert "persistence/sqlite/conversation/_store/turns.py:" in result.stdout
    assert "_HOME_READS" in result.stdout


def test_a_listed_home_read_passes_only_in_its_file_and_for_its_name(tmp_path):
    listed = "import hermes_constants\n\ndef root():\n    return hermes_constants.get_default_hermes_root()\n"
    _write(tmp_path, {"persistence/sqlite/admin/_reconcile.py": listed})
    assert _run(tmp_path).returncode == 0

    _write(tmp_path, {"persistence/sqlite/admin/_reconcile.py": listed + "HOME = hermes_constants.get_hermes_home()\n"})
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "persistence/sqlite/admin/_reconcile.py:5 get_hermes_home" in result.stdout

    _write(tmp_path, {"persistence/sqlite/admin/_reconcile.py": "\n", "persistence/sqlite/admin/_repair.py": listed})
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "persistence/sqlite/admin/_repair.py:4 get_default_hermes_root" in result.stdout


@pytest.mark.parametrize(("rel", "source"), [
    ("persistence/sqlite/conversation/_store/turns.py", "from hermes_constants import named_profile_home\n"),
    ("persistence/sqlite/conversation/_store/turns.py", "from session_rules import turns\n"),
    ("gateway/run_helpers.py", "from hermes_constants import get_hermes_home\n"),
    ("persistence/conversation/messages.py", "from hermes_constants import get_hermes_home\n"),
])
def test_profile_scoped_reads_and_home_reads_outside_the_backend_pass(tmp_path, rel, source):
    _write(tmp_path, {rel: source})

    assert _run(tmp_path).returncode == 0
