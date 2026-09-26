import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"


def test_setup_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_setup_stages_uv_into_pms_store_root(tmp_path, monkeypatch):
    """setup-hermes.sh stages uv where pm.paths.store_root() resolves it (#101269).

    The script hardcoded ~/.hermes/tools while pm's own store follows
    HERMES_HOME, so with HERMES_HOME set the two never met: the script staged a
    sha256-verified uv PM could not see, and PM re-downloaded it.
    """
    bash = shutil.which("bash")
    assert bash, "setup-hermes.sh is a bash script"

    # Deliberately different from $HOME/.hermes, so a hardcoded default in the
    # script stages somewhere pm will never look.
    home = tmp_path / "custom-home" / ".hermes"
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)

    from pm.paths import store_root

    store = Path(store_root())
    assert store == home / "tools", "pm must resolve its store under HERMES_HOME"

    lock = json.loads((REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))
    uv_version = lock["packages"]["uv"]["version"]
    system = "darwin" if sys.platform == "darwin" else "linux"
    arch = "arm64" if platform.machine() in {"arm64", "aarch64"} else "x64"
    uv = store / f"uv-{uv_version}-{system}-{arch}" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        '  --version) echo "uv 0.0.0-test"; exit 0 ;;\n'
        '  python) exit 1 ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    # A staging attempt must fail locally instead of reaching the network.
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    curl = stub_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 6\n", encoding="utf-8")
    curl.chmod(0o755)

    env = {**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        [bash, str(SETUP_SCRIPT)], env=env, cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    assert "Staging pinned uv" not in output, output
    assert "pinned uv found" in output, output
