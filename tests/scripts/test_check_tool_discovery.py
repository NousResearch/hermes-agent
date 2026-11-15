"""The tool-discovery guard names every registering module that discovery would never import."""

import subprocess
from pathlib import Path

from scripts.check_tool_discovery import unreached_registrations

_REGISTERS = (
    "from tools.registry import registry\n"
    "registry.register(name='t', toolset='x', schema={}, handler=lambda *_a, **_k: '{}')\n"
)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=60)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, timeout=60)
    return tmp_path


def test_registrations_outside_discovery_are_named(tmp_path):
    repo = _repo(tmp_path, {
        "tools/__init__.py": "",
        "tools/flat_tool.py": _REGISTERS,
        "tools/pkg/__init__.py": "",
        "tools/pkg/tool.py": _REGISTERS,
        "tools/pkg/helper.py": _REGISTERS,
        "tools/pkg/sub/tool.py": _REGISTERS,
        "tools/no_init/tool.py": _REGISTERS,
    })
    assert unreached_registrations(repo) == [
        "tools/no_init/tool.py", "tools/pkg/helper.py", "tools/pkg/sub/tool.py",
    ]


def test_library_modules_and_discovered_tools_pass(tmp_path):
    repo = _repo(tmp_path, {
        "tools/__init__.py": "",
        "tools/flat_tool.py": _REGISTERS,
        "tools/pkg/__init__.py": "",
        "tools/pkg/tool.py": _REGISTERS,
        "tools/pkg/helper.py": "def helper():\n    return 1\n",
    })
    assert unreached_registrations(repo) == []
