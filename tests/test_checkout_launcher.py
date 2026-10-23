import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "hermes"


@pytest.fixture
def install_root() -> Path:
    with tempfile.TemporaryDirectory(prefix=".launcher-test-", dir=REPO_ROOT) as root:
        yield Path(root)


@pytest.mark.skipif(os.name == "nt", reason="managed uv venvs use symlinks on POSIX")
def test_checkout_launcher_reexecs_adjacent_venv_python(
    install_root: Path, tmp_path: Path
) -> None:
    """Legacy service units that invoke ./hermes must still enter the venv.

    A systemd unit installed by older Hermes versions could run the checkout's
    top-level ``hermes`` script directly.  If its shebang resolves to the system
    Python, gateway-only dependencies from the managed venv are invisible.  The
    launcher should detect an adjacent venv and exec that interpreter before
    importing Hermes modules.
    """

    launcher = install_root / "hermes"
    os.link(LAUNCHER, launcher)

    venv = install_root / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--symlinks", str(venv)],
        check=True,
    )
    venv_python = venv / "bin" / "python"
    site_packages = Path(
        subprocess.run(
            [
                str(venv_python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    probe_package = site_packages / "hermes_cli"
    probe_package.mkdir()
    (probe_package / "__init__.py").touch()
    marker = tmp_path / "launcher-result.json"
    (probe_package / "main.py").write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "def main():\n"
        "    result = {\n"
        "        'argv': sys.argv,\n"
        "        'executable': sys.executable,\n"
        "        'prefix': sys.prefix,\n"
        "        'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "        'pythonhome': os.environ.get('PYTHONHOME'),\n"
        "    }\n"
        "    Path(os.environ['HERMES_LAUNCHER_TEST_RESULT']).write_text(\n"
        "        json.dumps(result), encoding='utf-8'\n"
        "    )\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HERMES_LAUNCHER_TEST_RESULT"] = str(marker)
    env["PYTHONPATH"] = "should-be-cleared"
    env["PYTHONHOME"] = sys.base_prefix
    result = subprocess.run(
        [sys.executable, str(launcher), "gateway", "run"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert Path(recorded["executable"]).resolve() == Path(sys.executable).resolve()
    assert Path(recorded["prefix"]).resolve() == venv.resolve()
    assert recorded["argv"] == [
        str(launcher.resolve()),
        "gateway",
        "run",
    ]
    assert recorded["pythonpath"] is None
    assert recorded["pythonhome"] is None
