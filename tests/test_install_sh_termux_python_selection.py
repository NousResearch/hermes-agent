from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"


def _check_python_source() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(r"^check_python\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert match, "check_python() missing from install.sh"
    return match.group(0)


def _fake_python(path: Path, version: str, supported: bool) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit " + ("0" if supported else "1") + "; fi\n"
        f"if [ \"${{1:-}}\" = \"--version\" ]; then echo 'Python {version}'; exit 0; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_check_python(tmp_path: Path, *, with_python311: bool) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_python(bin_dir / "python", "3.14.6", supported=False)
    if with_python311:
        _fake_python(bin_dir / "python3.11", "3.11.15", supported=True)
    pkg_log = tmp_path / "pkg.log"
    pkg = bin_dir / "pkg"
    pkg.write_text(f"#!/bin/sh\necho \"$*\" >> {pkg_log!s}\nexit 0\n", encoding="utf-8")
    pkg.chmod(0o755)

    harness = f"""
set -u
DISTRO=termux
PYTHON_VERSION=3.11
PYTHON_PATH=""
PYTHON_FOUND_VERSION=""
log_info() {{ :; }}
log_success() {{ :; }}
log_error() {{ echo "$*" >&2; }}
{_check_python_source()}
check_python
echo "SELECTED=$PYTHON_PATH"
echo "VERSION=$PYTHON_FOUND_VERSION"
"""
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    return subprocess.run(
        ["/bin/bash", "-c", harness],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_termux_prefers_supported_python311_over_default_python314(tmp_path: Path) -> None:
    result = _run_check_python(tmp_path, with_python311=True)
    assert result.returncode == 0, result.stderr
    assert f"SELECTED={tmp_path / 'bin' / 'python3.11'}" in result.stdout
    assert "VERSION=Python 3.11.15" in result.stdout
    assert not (tmp_path / "pkg.log").exists()


def test_termux_rejects_python314_after_package_install_attempt(tmp_path: Path) -> None:
    result = _run_check_python(tmp_path, with_python311=False)
    assert result.returncode != 0
    assert "Python >=3.11,<3.14" in result.stderr
    assert (tmp_path / "pkg.log").read_text(encoding="utf-8").strip() == "install -y python"
