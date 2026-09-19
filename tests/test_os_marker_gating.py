"""Run the real collection hook; skip-all and unregistered guards must fail."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("invalid,message", [("", ""), ("stacked", "at most one platforms()"),
                                               ("keyword", "unexpected keyword")])
def test_native_collection_witnesses(tmp_path, invalid, message):
    root = Path(__file__).resolve().parents[1]
    host = {"linux": "linux", "darwin": "macos", "win32": "windows"}[sys.platform]
    (tmp_path / "conftest.py").write_text(
        f"import sys; sys.path.insert(0, {str(root)!r})\n"
        "from tests.conftest import pytest_configure, pytest_collection_modifyitems\n", encoding="utf-8")
    suite = "import pytest\nfrom pathlib import Path\n"
    for name, marker in [
        ("plain", ""), ("any", "@pytest.mark.platforms('any')"),
        ("native", f"@pytest.mark.platforms({host!r})"),
        ("foreign", f"@pytest.mark.platforms('not {host}')"),
        ("arch", "@pytest.mark.platforms('any', arch='nonexistent-architecture')"),
    ]:
        suite += f"{marker}\ndef test_{name}():\n    Path({name!r}).touch()\n"
    (tmp_path / "test_valid.py").write_text(suite, encoding="utf-8")
    if invalid:
        marker = "@pytest.mark.platforms('any')" if invalid == "stacked" else "@pytest.mark.platforms('any', bogus=True)"
        module_mark = "pytestmark = pytest.mark.platforms('any')\n" if invalid == "stacked" else ""
        (tmp_path / "test_bad.py").write_text(
            f"import pytest\n{module_mark}{marker}\ndef test_bad():\n    raise AssertionError('must reject collection')\n",
            encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", str(tmp_path)],
                            cwd=tmp_path, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    witnesses = {p.name for p in tmp_path.iterdir() if p.name in {"plain", "any", "native", "foreign", "arch"}}
    if invalid:
        assert result.returncode == 4, output
        assert message in output and "test_bad.py::test_bad" in output
        assert "test_valid.py::" not in output
        assert not witnesses
    else:
        assert result.returncode == 0, output
        assert "3 passed, 2 skipped" in output
        assert witnesses == {"plain", "any", "native"}
