import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backup-health-check.sh"


def test_shell_entrypoint_delegates_to_canonical_python_probe(tmp_path):
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf 'delegated=%s\\n' \"$1\"\n"
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    environment = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == f"delegated={SCRIPT_PATH.with_suffix('.py')}\n"
    assert result.stderr == ""
