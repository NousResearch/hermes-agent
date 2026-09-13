"""Launch the packaged desktop with a competing mutable install."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


desktop, expected = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix="hermes-desktop-backend-") as temporary:
    home = Path(temporary)
    hermes_home = home / ".hermes"
    legacy = hermes_home / "hermes-agent"
    (legacy / "hermes_cli").mkdir(parents=True)
    (legacy / "hermes_cli" / "main.py").touch()
    launcher = legacy / "venv" / "bin" / "hermes"
    launcher.parent.mkdir(parents=True)
    # The old runtime passes discovery but cannot initialize a session. It
    # must not win merely because it was installed before the Nix desktop.
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('Hermes legacy fixture')\n"
        "    sys.exit(0)\n"
        "print('WRONG_BACKEND_SELECTED', flush=True)\n"
        "sys.exit(73)\n"
    )
    launcher.chmod(0o755)
    runtime = home / "runtime"
    runtime.mkdir(mode=0o700)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
        "HERMES_DESKTOP_USER_DATA_DIR": str(home / "electron"),
        "XDG_RUNTIME_DIR": str(runtime),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "WLR_BACKENDS": "headless",
        "WLR_RENDERER": "pixman",
        "WLR_NO_HARDWARE_CURSORS": "1",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
    }
    log_path = home / "desktop-output.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(
            ["cage", "--", desktop, "--no-sandbox", "--disable-gpu", "--ozone-platform=wayland"],
            cwd=home, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                output = log_path.read_text()
                desktop_log = hermes_home / "logs" / "desktop.log"
                if desktop_log.exists():
                    output += desktop_log.read_text()
                assert "WRONG_BACKEND_SELECTED" not in output, output
                assert child.poll() is None, output
                if "HERMES_BACKEND_READY port=" in output:
                    assert f"existing Hermes CLI at {expected}" in output, output
                    print("PASS: packaged desktop selected its pinned backend over the mutable install")
                    break
                time.sleep(0.1)
            else:
                raise AssertionError(f"Desktop did not start its backend:\n{log_path.read_text()}")
        finally:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
