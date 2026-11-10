"""``hermes plugins validate`` warns (never fails) on imports of private Hermes core modules."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml

from hermes_agent.compat.paths import current_module
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
    ("hermes_agent.gateway.platforms.access_policy_mixin", "OwnAccessPolicyMixin"),
    ("tools.approval", "resolve_gateway_approval"),
    ("tools.clarify_gateway", "get_pending"),
    ("tools.clarify_gateway", "resolve_gateway_clarify"),
    ("tools.clarify_gateway", "mark_awaiting_text"),
    ("hermes_agent.gateway.platforms.helpers", "platform_http_client"),
    ("hermes_agent.gateway.platforms.base", "ssrf_redirect_guard"),
    ("tools.url_safety", "create_ssrf_safe_async_client"),
    ("tools.url_safety", "is_safe_url"),
    ("hermes_agent.gateway.platforms._http_client_limits", "platform_httpx_limits"),
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
        assert hasattr(importlib.import_module(current_module(module)), name), f"{module}.{name}"
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

    from hermes_agent.tools import skills_tool

    for module, name in _SKILL_CATALOG_SEAMS:
        assert hasattr(importlib.import_module(current_module(module)), name), f"{module}.{name}"
    imports = "".join(f"from {module} import {name}\n" for module, name in _SKILL_CATALOG_SEAMS)
    assert _private_warnings(validate_plugin_dir(_plugin(tmp_path / "catalog", {"view.py": imports}))) == []

    marker = {"name": "seam-probe-skill", "description": "d", "category": None}
    monkeypatch.setattr(skills_tool, "find_all_skills", lambda **_kw: [dict(marker)])
    listed = json.loads(skills_tool.skills_list())
    assert [s["name"] for s in listed["skills"] if s["name"] == "seam-probe-skill"] == ["seam-probe-skill"]


def test_documented_session_helper_and_own_install_path_fallback_are_not_flagged(tmp_path):
    """onebot-platform reads session identity through the documented ``get_session_env``; openltm
    falls back to its own install path (``plugins.memory.openltm._db``) when the relative import
    fails. Neither is a core internal. A real core path beside them still warns."""
    plugin = _plugin(tmp_path / "openltm", {
        "__init__.py": (
            "def register(ctx):\n"
            "    from gateway.session_context import get_session_env\n"
            "    try:\n"
            "        from . import _db\n"
            "    except ImportError:\n"
            "        import plugins.memory.openltm._db as _db\n"
            "        from plugins.memory.openltm import _db\n"
            "    from hermes_cli.gateway_client import _session_ticket\n"
            "    return get_session_env, _db, _session_ticket\n"
        ),
        "_db.py": "",
    })

    [warning] = _private_warnings(validate_plugin_dir(plugin))

    sites = warning.split("release): ", 1)[1].split(". Use ctx.*")[0].split("; ")
    assert sites == ["__init__.py:8 hermes_cli.gateway_client._session_ticket"], sites


def test_plugin_guide_session_identity_example_validates_clean(tmp_path):
    """Regression for #148311: the plugin guide's ``get_session_env`` example uses the import spelling
    ``hermes plugins validate`` accepts, so a plugin that copies it gets no private-import warning."""
    import ast
    import re

    guide = Path(__file__).resolve().parents[2] / "website/docs/developer-guide/plugins/index.md"
    blocks = [b for b in re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), re.DOTALL)
              if "get_session_env" in b]
    assert blocks, "plugins/index.md shows no code example importing get_session_env"
    for block in blocks:  # validate sees only imports, and skips code that does not parse
        assert any(isinstance(node, ast.ImportFrom) and node.module == "hermes_agent.gateway.session_context"
                   and "get_session_env" in {alias.name for alias in node.names}
                   for node in ast.walk(ast.parse(block))), block
    plugin = _plugin(tmp_path / "sessid", {f"example_{i}.py": b for i, b in enumerate(blocks)})

    assert _private_warnings(validate_plugin_dir(plugin)) == []


def test_plugin_guide_file_safety_and_yaml_example_validates_clean(tmp_path):
    """The path-check and YAML helpers the guide (and catalog reviews) tell authors to call are
    public: a plugin copying the guide's example gets no private-import warning, while the same
    module imported whole still warns."""
    import re

    guide = Path(__file__).resolve().parents[2] / "website/docs/developer-guide/plugins/index.md"
    [block] = [b for b in re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), re.DOTALL)
               if "get_read_block_error" in b]
    clean = _plugin(tmp_path / "guide", {"example.py": block})
    assert _private_warnings(validate_plugin_dir(clean)) == []

    whole = _plugin(tmp_path / "whole", {"example.py": "import agent.file_safety\n"})
    assert _private_warnings(validate_plugin_dir(whole)), "a whole-module import of file_safety must still warn"


def test_private_method_calls_on_core_handles_warn_but_own_helpers_do_not(tmp_path):
    """opencode-sync calls SessionDB / GatewayRunner private methods on the objects it is handed;
    no import names them, so only the attribute pass can see it. The plugin's own ``_helper`` and
    ``self._x`` calls are not core reach."""
    plugin = _plugin(tmp_path / "sync", {"engine.py": (
        "class Store:\n"
        "    def _execute_write(self, fn):\n"
        "        return self._execute_write(fn)\n"
        "def apply(db, runner, store, route):\n"
        "    rows = db._row_to_message_dict(None)\n"
        "    runner._evict_cached_agent(route)\n"
        "    return store._execute_write(rows)\n"
    )})

    report = validate_plugin_dir(plugin)

    assert report.ok, report.failures
    [warning] = [w for w in report.warnings if w.startswith("private core attribute access")]
    sites = warning.split("release): ", 1)[1].split(". Use ctx.*")[0].split("; ")
    assert sites == ["engine.py:5 db._row_to_message_dict", "engine.py:6 runner._evict_cached_agent"], sites


_ROOT = Path(__file__).resolve().parents[2]


def _published_entries() -> list[str]:
    import json

    data = json.loads((_ROOT / "hermes_cli" / "data" / "plugin_api.json").read_text(encoding="utf-8-sig"))
    return [entry for group in data["groups"] for entry in group["entries"]]


def test_every_published_plugin_api_entry_passes_validate(tmp_path):
    """hermes_cli/data/plugin_api.json is the list validate enforces: every entry imports without a
    warning, while an unlisted name of a listed name-only module still warns."""
    imports = [f"from {m} import {n}" if n else f"import {m}"
               for m, _, n in (entry.partition(":") for entry in _published_entries())]
    imports.append("from hermes_cli.plugins import get_plugin_manager")
    plugin = _plugin(tmp_path / "api", {"uses_api.py": "\n".join(imports) + "\n"})

    [warning] = _private_warnings(validate_plugin_dir(plugin))

    sites = warning.split("release): ", 1)[1].split(". Use ctx.*")[0].split("; ")
    assert sites == [f"uses_api.py:{len(imports)} hermes_cli.plugins.get_plugin_manager"], sites


def test_public_api_docs_page_is_generated_from_the_data_file():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_plugin_api_docs", _ROOT / "scripts" / "gen_plugin_api_docs.py")
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    assert gen.check() == 0, "website/docs/developer-guide/plugins/public-api.md is stale"


def test_a_published_module_moved_under_hermes_agent_is_public_in_every_spelling(tmp_path, monkeypatch):
    """Once ``hermes_constants`` moves, ``from hermes_agent import hermes_constants`` names the same
    published module as ``import hermes_constants`` and must not warn."""
    from hermes_agent.compat.paths import PathMoves
    from hermes_cli import plugin_validate_private_imports as private_imports

    monkeypatch.setattr(private_imports, "original_module",
                        PathMoves({"hermes_constants": "hermes_agent.hermes_constants"}).original_module)
    plugin = _plugin(tmp_path / "moved", {"uses_moved.py": (
        "from hermes_agent import hermes_constants\n"
        "from hermes_agent.hermes_constants import get_hermes_home\n"
        "import hermes_agent.hermes_constants\n"
    )})

    assert _private_warnings(validate_plugin_dir(plugin)) == []
