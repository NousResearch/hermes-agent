import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "hermes"


def test_checkout_launcher_reexecs_adjacent_venv_python(tmp_path: Path) -> None:
    """Legacy service units that invoke ./hermes must still enter the venv.

    A systemd unit installed by older Hermes versions could run the checkout's
    top-level ``hermes`` script directly.  If its shebang resolves to the system
    Python, gateway-only dependencies from the managed venv are invisible.  The
    launcher should detect an adjacent venv and exec that interpreter before
    importing Hermes modules.
    """

    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    launcher = install_root / "hermes"
    launcher.write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")

    fake_python = install_root / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    marker = tmp_path / "argv.txt"
    fake_python.write_text(
        "#!/usr/bin/env sh\n"
        f"printf '%s\\n' \"$@\" > {marker}\n"
        "exit 42\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PYTHONPATH"] = "should-be-cleared"
    result = subprocess.run(
        [sys.executable, str(launcher), "gateway", "run"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 42
    assert marker.read_text(encoding="utf-8").splitlines() == [
        str(launcher.resolve()),
        "gateway",
        "run",
    ]
