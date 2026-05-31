import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "setup_open_webui.sh"


def test_write_launcher_secures_open_webui_data_dir(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    data_dir = home / ".local" / "share" / "open-webui" / "data"

    command = f"""
set -euo pipefail
source <(sed '/^main "\\$@"$/d' {SCRIPT})
umask 022
write_launcher
python3 - <<'PY2'
from pathlib import Path
import os
import stat
home = Path(os.environ['HOME'])
data_dir = Path(os.environ['OPEN_WEBUI_DATA_DIR'])
launcher = home / '.local' / 'bin' / 'start-open-webui-hermes.sh'
print(oct(stat.S_IMODE(data_dir.stat().st_mode)))
print('umask 077' in launcher.read_text())
print(oct(stat.S_IMODE(launcher.stat().st_mode)))
PY2
"""

    result = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "HOME": str(home),
            "OPEN_WEBUI_DATA_DIR": str(data_dir),
            "OPEN_WEBUI_VENV": str(home / ".local" / "open-webui-venv"),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    mode, has_umask, launcher_mode = result.stdout.strip().splitlines()
    assert mode == oct(stat.S_IRWXU)
    assert has_umask == "True"
    assert launcher_mode == oct(stat.S_IRWXU)
