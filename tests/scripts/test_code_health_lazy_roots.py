"""Every module the audited hosted roots import inside a function is a maintained decision.

The hosted graph follows only imports that run at import time. A module a root imports lazily
(inside a function) is on the hosted path only if it is listed: as a root in ``HOSTED_ENTRY``,
or in ``OFF_HOSTED_PATH`` with the reason it never runs while the process serves tenants. A new
lazy import from an audited root fails here until it is placed in one of the two.
"""

from __future__ import annotations

from scripts.code_health import config
from scripts.code_health.hosted_graph import function_imports, reachable
from tests.scripts.test_code_health import REPO


def test_every_lazy_import_of_an_audited_root_is_decided():
    graph = reachable(REPO, None)
    undecided = sorted({target for root in config.LAZY_IMPORTS_AUDITED
                        for target in function_imports(REPO, None, root)
                        if target not in graph and target not in config.OFF_HOSTED_PATH})
    assert not undecided, (f"add each to HOSTED_ENTRY, or to OFF_HOSTED_PATH with why it never runs"
                           f" while the process serves tenants: {undecided}")


def test_off_hosted_path_entries_are_current_and_reasoned():
    graph = reachable(REPO, None)
    lazy = {t for root in config.LAZY_IMPORTS_AUDITED for t in function_imports(REPO, None, root)}
    for path, why in config.OFF_HOSTED_PATH.items():
        assert why.strip(), path
        assert path not in graph, f"{path} is on the hosted graph now; drop it from OFF_HOSTED_PATH"
        assert path in lazy, f"{path} is no longer imported by an audited root; drop it"


def test_function_imports_lists_function_body_targets_only(tmp_path):
    (tmp_path / "pkg").mkdir()
    for name in ("root", "pkg/__init__", "pkg/lazy", "top", "unused"):
        (tmp_path / f"{name}.py").write_text("")
    (tmp_path / "root.py").write_text(
        "import top\ndef f():\n    from pkg import lazy\n    import pkg.lazy\n    from . import nothing\n")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert function_imports(tmp_path, None, "root") == {"pkg/__init__.py", "pkg/lazy.py"}
