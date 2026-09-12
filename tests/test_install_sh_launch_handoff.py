"""Post-PM installer stages use the real installation-bound publication."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.platforms('posix')
@pytest.mark.parametrize('stage, expected', [('setup', ['setup']), ('gateway', ['gateway', 'install']), ('desktop', ['desktop', '--build-only'])])
def test_installer_post_pm_stages(tmp_path: Path, stage: str, expected: list[str]) -> None:
    install = tmp_path / 'source tree'
    calls = tmp_path / 'calls.json'
    publish_fixture_launcher(install, "import json, os, sys\nfrom pathlib import Path\ndef main():\n    Path(os.environ['CALLS']).write_text(json.dumps(sys.argv[1:])); return int(os.environ['STAGE_EXIT'])\n")
    command = ['bash', '-c', 'source "$1" --manifest >/dev/null; INSTALL_DIR="$2"; NON_INTERACTIVE=false; "stage_$3"', 'test', str(ROOT / 'scripts/install.sh'), str(install), stage]
    env = {**os.environ, 'HOME': str(tmp_path), 'HERMES_HOME': str(tmp_path / 'home'), 'HERMES_RUNTIME_DIR': str(tmp_path / 'store'), 'CALLS': str(calls)}
    for code in [0, 9]:
        result = subprocess.run(command, cwd=tmp_path, env={**env, 'STAGE_EXIT': str(code)}, capture_output=True, text=True, timeout=20)
        assert (result.returncode == 0) == (code == 0), result.stdout + result.stderr
        assert json.loads(calls.read_text()) == expected
    assert not (install / 'venv').exists()


@pytest.mark.platforms("posix")
def test_desktop_flag_only_inserts_desktop_before_completion(tmp_path):
    def manifest(*flags):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/install.sh"), "--manifest", *flags],
            cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True,
        )
        return [row["name"] for row in json.loads(result.stdout)["stages"]]

    plain = manifest()
    desktop = manifest("--include-desktop")
    assert "desktop" not in plain
    assert plain[-1] == "complete"
    assert desktop == [*plain[:-1], "desktop", "complete"]