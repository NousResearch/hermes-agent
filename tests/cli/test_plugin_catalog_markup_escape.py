"""Plugin-supplied text must print verbatim — Rich markup characters in
manifest data (e.g. ``uvicorn[standard]``, ``requests[security]`` or a
``[sigh]`` inside a catalog description) are data, not style tags.

Regression coverage for https://github.com/NousResearch/hermes-agent/issues/151743
"""
import io
from rich.console import Console

from hermes_cli import plugins_cmd_install


def test_python_deps_with_extras_brackets_print_verbatim():
    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    plugins_cmd_install._consent_python_deps(
        "demo", ["uvicorn[standard]", "requests[security]"], console, assume_yes=True
    )
    out = console.file.getvalue()
    assert "uvicorn[standard]" in out
    assert "requests[security]" in out
