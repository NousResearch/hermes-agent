from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mcp"
    / "fastmcp"
    / "scripts"
    / "scaffold_fastmcp.py"
)


def load_scaffold_module():
    spec = importlib.util.spec_from_file_location("fastmcp_scaffold", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_template_escapes_server_name_inside_python_string_literal():
    scaffold = load_scaffold_module()
    malicious_name = '");__import__("os").system("id");#'

    rendered = scaffold.render_template("api_wrapper", malicious_name)

    ast.parse(rendered)
    assert 'FastMCP("\\");__import__(\\"os\\").system(\\"id\\");#")' in rendered
    assert 'FastMCP("");__import__("os").system("id");#")' not in rendered


def test_render_template_rejects_unknown_or_path_traversal_templates():
    scaffold = load_scaffold_module()

    with pytest.raises(SystemExit):
        scaffold.render_template("../fastmcp/scripts/scaffold_fastmcp", "Server")
