"""The repo's scripts package wins over an installed top-level ``scripts`` package."""
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def test_installed_scripts_package_does_not_shadow_repo_scripts(tmp_path):
    # Some distributions (gguf, for one) install a regular top-level
    # ``scripts`` package into site-packages.
    shadow = tmp_path / "site-packages"
    (shadow / "scripts").mkdir(parents=True)
    (shadow / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    code = (
        "import sys\n"
        f"sys.path.append({str(shadow)!r})\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import scripts.code_health\n"
        "print(scripts.__file__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == REPO / "scripts" / "__init__.py"
