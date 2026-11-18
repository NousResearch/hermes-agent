"""The launcher's code-fence check stays inside the launcher's startup closure.

``hermes_cli/_launchers.py`` runs before application code, from a tree whose startup files
(pyproject ``[tool.hermes] update-startup-files``) are all it may rely on. Its hosted
provenance check (``hosted_code_refusal``) is one of those files and imports only the
stdlib: in a local process it answers without importing ``hermes_cli.hosted_mode``, which
is not a startup file. On Windows, a source launcher that republished itself from such a
tree failed with ``ModuleNotFoundError`` (tests/hermes_cli/test_source_launcher_publication.py).
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_a_local_refusal_check_imports_only_the_startup_closure(tmp_path):
    tree = tmp_path / "tree"
    for relative in ("hermes_cli/__init__.py", "hermes_cli/hosted_code_fence.py"):
        (tree / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, tree / relative)
    probe = (
        "import sys; from pathlib import Path\n"
        "from hermes_cli.hosted_code_fence import hosted_code_refusal\n"
        "assert hosted_code_refusal(Path(sys.argv[1])) == ''\n"
        "assert 'hermes_cli.hosted_mode' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(tree)!r})\n{probe}",
                           str(tmp_path)], capture_output=True, text=True, timeout=60, cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr


def test_a_hosted_mode_still_initialising_reads_as_no_frozen_verdict(tmp_path, monkeypatch):
    """Another thread importing ``hosted_mode`` leaves it in ``sys.modules`` before its reader
    is bound; the import it replaced would have waited and read no verdict there either."""
    import types

    from hermes_cli.hosted_code_fence import hosted_code_refusal

    monkeypatch.setitem(sys.modules, "hermes_cli.hosted_mode", types.ModuleType("hermes_cli.hosted_mode"))
    monkeypatch.setattr(sys, "argv", ["hermes"])
    assert hosted_code_refusal(tmp_path) == ""
