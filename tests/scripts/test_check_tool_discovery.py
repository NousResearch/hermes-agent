"""scripts/check_tool_discovery.py: a tool registered where discovery never looks is flagged."""

import subprocess
from pathlib import Path

from scripts.check_tool_discovery import unreached_registrations

_REGISTERS = (
    "from tools.registry import registry\n"
    "registry.register(name='x', toolset='x', schema={}, handler=lambda *_a, **_k: '{}')\n"
)


def _tracked_tools(tmp_path: Path, files: dict[str, str]) -> Path:
    tools_dir = tmp_path / "tools"
    for rel, text in files.items():
        (tools_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (tools_dir / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=60)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, timeout=60)
    return tools_dir


def test_registration_outside_tool_py_in_a_package_is_flagged(tmp_path):
    tools_dir = _tracked_tools(tmp_path, {
        "__init__.py": "",
        "flat_tool.py": _REGISTERS,
        "pkg/__init__.py": "",
        "pkg/tool.py": _REGISTERS,
        "pkg/extra.py": _REGISTERS,
    })
    assert unreached_registrations(tools_dir) == [tools_dir / "pkg/extra.py"]


def test_flat_modules_and_package_tool_py_pass(tmp_path):
    tools_dir = _tracked_tools(tmp_path, {
        "__init__.py": "",
        "flat_tool.py": _REGISTERS,
        "pkg/__init__.py": "",
        "pkg/tool.py": _REGISTERS,
        "pkg/helpers.py": "def helper():\n    return 1\n",
    })
    assert unreached_registrations(tools_dir) == []


def test_package_tool_py_without_init_is_flagged(tmp_path):
    tools_dir = _tracked_tools(tmp_path, {"__init__.py": "", "pkg/tool.py": _REGISTERS})
    assert unreached_registrations(tools_dir) == [tools_dir / "pkg/tool.py"]
