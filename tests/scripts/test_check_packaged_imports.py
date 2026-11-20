"""scripts/check_packaged_imports.py: a package that packaged code imports but the wheel drops is flagged."""

import subprocess
from pathlib import Path

from scripts.check_packaged_imports import unshipped_imports

_PYPROJECT = '[tool.setuptools.packages.find]\ninclude = ["agent", "agent.*", "acp_adapter"]\n'


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in {"pyproject.toml": _PYPROJECT, "setup.py": "", **files}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=60)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, timeout=60)
    return tmp_path


def test_a_root_module_turned_into_a_folder_is_flagged(tmp_path):
    repo = _repo(tmp_path, {
        "run_agent.py": "import toolsets\n",
        "toolsets/__init__.py": "",
    })
    assert unshipped_imports(repo) == ["toolsets"]


def test_a_submodule_imported_by_from_import_is_flagged(tmp_path):
    repo = _repo(tmp_path, {
        "run_agent.py": "from acp_adapter import server\n",
        "acp_adapter/__init__.py": "",
        "acp_adapter/server/__init__.py": "",
    })
    assert unshipped_imports(repo) == ["acp_adapter.server"]


def test_a_relative_submodule_import_is_flagged(tmp_path):
    repo = _repo(tmp_path, {
        "acp_adapter/__init__.py": "",
        "acp_adapter/entry.py": "from . import server\n",
        "acp_adapter/server/__init__.py": "",
    })
    assert unshipped_imports(repo) == ["acp_adapter.server"]


def test_shipped_packages_and_plain_names_pass(tmp_path):
    repo = _repo(tmp_path, {
        "run_agent.py": "import json\nfrom agent import redact\nfrom agent.redact import scrub\n",
        "agent/__init__.py": "",
        "agent/redact/__init__.py": "def scrub(): ...\n",
        "plugins/browser/adapter.py": "",
    })
    assert unshipped_imports(repo) == []


def test_files_under_a_namespace_directory_are_scanned(tmp_path):
    # pyproject's packages.find defaults to namespaces = true, so plugins/browser/ (no __init__.py)
    # ships under "plugins.*" and its imports count.
    repo = _repo(tmp_path, {
        "pyproject.toml": '[tool.setuptools.packages.find]\ninclude = ["plugins", "plugins.*"]\n',
        "plugins/__init__.py": "",
        "plugins/browser/vendor/adapter.py": "import toolsets\n",
        "toolsets/__init__.py": "",
    })
    assert unshipped_imports(repo) == ["toolsets"]
