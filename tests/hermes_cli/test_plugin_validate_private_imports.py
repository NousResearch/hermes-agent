"""``hermes plugins validate`` warns (never fails) on imports of private Hermes core modules."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml

from hermes_cli.plugin_validate import validate_plugin_dir


def _plugin(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text(yaml.safe_dump({"name": root.name, "version": "1.0.0", "description": "x"}),
                                      encoding="utf-8")
    files = {"__init__.py": "def register(ctx):\n    pass\n", **files}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _private_warnings(report) -> list[str]:
    return [w for w in report.warnings if w.startswith("private core imports")]


def test_private_core_import_warns_with_file_and_line_but_validate_passes(tmp_path):
    plugin = _plugin(tmp_path / "router", {"lib/agent_runner.py": (
        "import json\n"
        "def run():\n"
        "    from tools.tirith_security import ensure_installed\n"
        "    from hermes_cli.config import _private_helper\n"
        "    return ensure_installed, _private_helper\n"
    )})

    report = validate_plugin_dir(plugin)

    assert report.ok, report.failures
    [warning] = _private_warnings(report)
    assert "lib/agent_runner.py:3 tools.tirith_security.ensure_installed" in warning
    assert "lib/agent_runner.py:4 hermes_cli.config._private_helper" in warning
    assert "catalog rule 9" in warning


def test_documented_public_stdlib_and_own_package_imports_are_not_flagged(tmp_path):
    plugin = _plugin(tmp_path / "clean", {
        "__init__.py": (
            "import os, json\n"
            "from typing import TYPE_CHECKING\n"
            "import requests\n"
            "from agent.memory_provider import MemoryProvider\n"  # documented provider base class
            "from gateway.platforms import base\n"  # documented module, imported from its package
            "from hermes_cli.plugins import PluginContext\n"  # what register() receives
            "from clean_lib.helpers import local_helper\n"  # the plugin's own package
            "from .client import Client\n"
            "if TYPE_CHECKING:\n"
            "    from run_agent import AIAgent\n"
            "def register(ctx):\n"
            "    from hermes_cli.plugins import get_plugin_manager\n"  # same module, undocumented name
            "    return get_plugin_manager\n"
        ),
        "clean_lib/__init__.py": "",
        "clean_lib/helpers.py": "def local_helper():\n    pass\n",
        "client.py": "class Client:\n    pass\n",
    })

    [warning] = _private_warnings(validate_plugin_dir(plugin))

    sites = warning.split("release): ", 1)[1].split(". Use ctx.*")[0].split("; ")
    assert sites == ["__init__.py:12 hermes_cli.plugins.get_plugin_manager"], sites


_PLATFORM_PLUGIN_SEAMS = (
    ("gateway.platforms.access_policy_mixin", "OwnAccessPolicyMixin"),
    ("tools.approval", "resolve_gateway_approval"),
    ("tools.clarify_gateway", "get_pending"),
    ("tools.clarify_gateway", "resolve_gateway_clarify"),
    ("tools.clarify_gateway", "mark_awaiting_text"),
    ("gateway.platforms.helpers", "platform_http_client"),
    ("gateway.platforms.base", "ssrf_redirect_guard"),
    ("tools.url_safety", "create_ssrf_safe_async_client"),
    ("tools.url_safety", "is_safe_url"),
    ("gateway.platforms._http_client_limits", "platform_httpx_limits"),
    ("hermes_cli.gateway_setup_wizard", "print_setup_header"),
    ("hermes_cli.gateway_setup_wizard", "confirm_reconfigure"),
    ("hermes_cli.gateway_setup_wizard", "offer_home_channel"),
    ("hermes_cli.gateway_setup_wizard", "save_env_values"),
    ("hermes_cli.gateway_setup_wizard", "prompt_csv"),
)


def test_platform_plugin_seams_are_public_and_exist(tmp_path):
    """The documented platform-plugin helpers import cleanly AND raise no private-import finding."""
    import importlib

    for module, name in _PLATFORM_PLUGIN_SEAMS:
        assert hasattr(importlib.import_module(module), name), f"{module}.{name}"
    imports = "".join(f"from {module} import {name}\n" for module, name in _PLATFORM_PLUGIN_SEAMS)
    plugin = _plugin(tmp_path / "chatplat", {"adapter.py": imports})

    assert _private_warnings(validate_plugin_dir(plugin)) == []


_SKILL_CATALOG_SEAMS = (
    ("tools.skills_tool", "find_all_skills"), ("tools.skills_tool", "sort_skills"),
    ("tools.skills_tool", "skill_search_dirs"), ("tools.skills_tool", "locate_skill"),
    ("tools.skills_tool", "skill_lookup_path_error"), ("tools.skills_sync", "dir_hash"),
    ("tools.skill_usage", "read_skill_name"),
)


def test_skill_catalog_seams_are_public_and_drive_the_skills_tools(tmp_path, monkeypatch):
    """The documented skill-catalog helpers import cleanly, raise no private-import finding, and are
    the names ``skills_list`` reads, so a plugin sees exactly what the agent sees."""
    import importlib
    import json

    from tools import skills_tool

    for module, name in _SKILL_CATALOG_SEAMS:
        assert hasattr(importlib.import_module(module), name), f"{module}.{name}"
    imports = "".join(f"from {module} import {name}\n" for module, name in _SKILL_CATALOG_SEAMS)
    assert _private_warnings(validate_plugin_dir(_plugin(tmp_path / "catalog", {"view.py": imports}))) == []

    marker = {"name": "seam-probe-skill", "description": "d", "category": None}
    monkeypatch.setattr(skills_tool, "find_all_skills", lambda **_kw: [dict(marker)])
    listed = json.loads(skills_tool.skills_list())
    assert [s["name"] for s in listed["skills"] if s["name"] == "seam-probe-skill"] == ["seam-probe-skill"]
