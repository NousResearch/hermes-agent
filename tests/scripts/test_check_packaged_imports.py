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


def test_optional_imports_are_not_wheel_dependencies(tmp_path):
    # A missing module is caught by the except ImportError handler, and TYPE_CHECKING never runs.
    repo = _repo(tmp_path, {
        "agent/__init__.py": "",
        "agent/a.py": "try:\n    import optpkg\nexcept ImportError:\n    optpkg = None\n",
        "agent/b.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from optpkg import thing\n",
        "agent/c.py": "try:\n    import optpkg\nexcept (ModuleNotFoundError, OSError):\n    optpkg = None\n",
        "agent/d.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    def f():\n        import optpkg\n",
        "agent/e.py": "try:\n    import optpkg\nexcept ImportError:\n    def fail():\n        raise\n",
        "optpkg/__init__.py": "",
    })
    assert unshipped_imports(repo) == []


def test_a_package_under_an_unshipped_namespace_directory_is_flagged(tmp_path):
    repo = _repo(tmp_path, {
        "pyproject.toml": '[tool.setuptools.packages.find]\ninclude = ["agent"]\n',
        "agent/__init__.py": "",
        "agent/a.py": "import agent.ns.missing\n",
        "agent/ns/missing/__init__.py": "",
    })
    assert unshipped_imports(repo) == ["agent.ns.missing"]


def test_imports_an_import_guard_does_not_protect_stay_checked(tmp_path):
    # The handler, else and finally, a broader catch, a plain if, a handler that re-raises, and a
    # function the try body defines (it runs when called, outside the handler) can all fail.
    repo = _repo(tmp_path, {
        "agent/__init__.py": "",
        "agent/clauses.py": "try:\n    import json\nexcept ImportError:\n    import p1\n"
                            "else:\n    import p6\nfinally:\n    import p7\n",
        "agent/broad.py": "try:\n    import p2\nexcept Exception:\n    pass\n",
        "agent/plain_if.py": "import sys\nif sys:\n    import p3\n",
        "agent/reraise.py": "try:\n    import p4\nexcept ImportError:\n    raise\n",
        "agent/deferred.py": "try:\n    def load():\n        import p5\nexcept ImportError:\n    pass\n",
        **{f"p{i}/__init__.py": "" for i in range(1, 8)},
    })
    assert unshipped_imports(repo) == [f"p{i}" for i in range(1, 8)]
