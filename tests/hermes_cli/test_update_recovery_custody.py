"""Broken custody imports never authorize repair of a possibly live checkout."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("live_owner", [True, False])
def test_torn_custody_modules_cannot_authorize_uncontained_repair(tmp_path, live_owner):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    for name in ("__init__.py", "_early_recovery.py", "update_lock.py", "update_custody.py"):
        shutil.copy2(source / "hermes_cli" / name, package / name)
    env = {**os.environ, "HOME": str(tmp_path / "home"), "HERMES_HOME": str(tmp_path / "state"),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "PYTHONDONTWRITEBYTECODE": "1"}
    # The child owns a newly created Git repo, not the checkout running pytest.
    env.pop("PYTEST_CURRENT_TEST", None)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], env=env, text=True).strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "review")
    git("config", "user.email", "review@example.invalid")
    payload = root / "payload"
    payload.write_bytes(b"before\n")
    git("add", "-A")
    git("commit", "-qm", "before")
    before = git("rev-parse", "HEAD")
    payload.write_bytes(b"after\n")
    git("commit", "-qam", "after")
    after = git("rev-parse", "HEAD")
    git("reset", "-q", "--hard", before)
    payload.write_bytes(b"after\n")
    marker = root / ".git/hermes-update-pull"
    marker.write_text(f"pid=0\npre={before}\ntarget={after}\ngit={shutil.which('git')}\n", encoding="utf-8")
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(
        "import sys,time\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
        "from hermes_cli.update_lock import UpdateLock\n"
        "lock=UpdateLock(path=Path(sys.argv[2])/'owner',install_root=Path(sys.argv[2]))\n"
        "assert lock.acquire()\nprint('HELD',flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    recovery_script = tmp_path / "recover.py"
    recovery_script.write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
        "from hermes_cli._early_recovery import restore_interrupted_pull\n"
        "restore_interrupted_pull(Path(sys.argv[1]))\n", encoding="utf-8")
    owner = subprocess.Popen([sys.executable, "-I", str(holder_script), str(source), str(root)],
                             env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        assert owner.stdout.readline().strip() == "HELD"
        damaged = ("update_lock.py", "update_custody.py") if live_owner else ("update_custody.py",)
        if not live_owner:
            owner.kill()
            owner.wait()
        for name in damaged:
            (package / name).write_bytes(b'"""torn')
        result = subprocess.run([sys.executable, "-I", str(recovery_script), str(root)],
                                cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=30, check=False)
        assert (owner.poll() is None) == live_owner
        assert payload.read_bytes() == b"after\n", "repair wrote without verifiable child custody"
        assert marker.exists(), "unverified custody must retain the interrupted-update record"
        assert result.returncode != 0 and "Cannot safely repair" in result.stderr
    finally:
        owner.kill()
        owner.wait()
