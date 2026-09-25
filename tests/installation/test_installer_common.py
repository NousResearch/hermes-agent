"""Driver setup against private Git/script artifacts, never an installed app."""
import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "install/e2e-assets/installer-common.sh"


@pytest.mark.platforms("posix")
def test_redirect_and_historical_installer_keep_transport_flags_and_exit(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    repo, work, serve = (tmp_path / name for name in ("repo", "work", "serve.git"))
    work.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], env=env, text=True).strip()

    subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True)
    git("remote", "add", "origin", "https://example.invalid/fork.git")
    script = repo / "scripts/install.sh"
    script.parent.mkdir()
    script.write_text('printf "%s\\n" "$@"\nread -r value && exit 99\nexit 0\n', encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "old")
    old = git("rev-parse", "HEAD")
    source_body = script.read_text(encoding="utf-8")
    script.write_text('case "$1" in --help) printf "%s\\n" "--skip-browser --include-desktop"; exit 0;; esac\n'
                      '# --skip-browser --include-desktop\n' + source_body, encoding="utf-8")
    git("commit", "-qam", "new")
    new = git("rev-parse", "HEAD")
    subprocess.run(["git", "clone", "--bare", "-q", str(repo), str(serve)], env=env, check=True)
    setup = '''set -euo pipefail
source "$1"
fail() { printf '%s\n' "$*" >&2; exit 1; }
ok() { :; }
ts_prefix() { cat; }
log_group() { :; }
arm_source_redirect "$2" "$3" "$4"
'''

    def run(body):
        return subprocess.run(["bash", "-c", setup + body, "bash", str(HELPER), str(repo), str(work), str(serve), old, new],
                              env=env, text=True, capture_output=True, timeout=30)

    result = run('''[ "$(git -C "$2" remote get-url origin)" = 'https://github.com/NousResearch/hermes-agent.git' ]
git clone -q https://github.com/NousResearch/hermes-agent.git "$3/clone"
[ "$(git -C "$3/clone" rev-parse HEAD)" = "$6" ]
run_source_installer "$2" "$3" "$3" "$5" old
run_source_installer "$2" "$3" "$3" "$6" new desktop
''')
    assert result.returncode == 0, result.stderr
    assert (work / "install-old.log").read_text(encoding="utf-8").splitlines() == ["--skip-setup"]
    assert (work / "install-new.log").read_text(encoding="utf-8").splitlines() == ["--skip-setup", "--skip-browser", "--include-desktop"]
    assert (work / "install-new.sh").read_bytes() == script.read_bytes()
    refused = run('run_source_installer "$2" "$3" "$3" "$5" unavailable desktop')
    assert refused.returncode != 0 and "does not support --include-desktop" in refused.stderr
    assert not (work / "install-unavailable.log").exists()
    script.write_text('case "$1" in\n'
                      '  --help) printf "%s\\n" "--include-desktop"; exit 0;;\n'
                      '  --skip-browser) exit 23;;\n'
                      'esac\n'
                      '# --skip-browser no longer skips browser installation; --include-desktop stays supported\n'
                      + source_body, encoding="utf-8")
    git("commit", "-qam", "reject retired browser flag")
    current = run('run_source_installer "$2" "$3" "$3" HEAD current desktop')
    assert current.returncode == 0, current.stderr
    assert (work / "install-current.log").read_text(encoding="utf-8").splitlines() == [
        "--skip-setup", "--include-desktop",
    ]
    # A PM installer supports --skip-browser as a real opt-out; the leg must
    # still install what users get by default, so the driver withholds it.
    script.write_text('case "$1" in --help) printf "%s\\n" "--skip-browser --include-desktop"; exit 0;; esac\n'
                      '# "$boot_py" -m pm.cli install\n' + source_body, encoding="utf-8")
    git("commit", "-qam", "pm installer")
    pm_installer = run('run_source_installer "$2" "$3" "$3" HEAD pm')
    assert pm_installer.returncode == 0, pm_installer.stderr
    assert (work / "install-pm.log").read_text(encoding="utf-8").splitlines() == ["--skip-setup"]
    script.write_text("exit 23\n", encoding="utf-8")
    git("commit", "-qam", "failed installer")
    failed = run('run_source_installer "$2" "$3" "$3" HEAD broken')
    assert failed.returncode != 0 and "exited 23" in failed.stderr
