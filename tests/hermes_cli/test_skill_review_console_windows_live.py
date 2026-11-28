"""Real keyboard input through the updater's Windows completion transport.

A disposable completion peer runs the real bundled-skill sync/review, not an
update. No dependencies, gateways or installed skills are changed.
"""
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from tests.waiting import wait_until


@pytest.mark.live_system_guard_bypass  # ConPTY's child is parented by its console host.
@pytest.mark.platforms("windows")
@pytest.mark.parametrize("keys,accepted", [
    pytest.param(None, False, id="headless-no-prompt"),
    pytest.param("\r", False, id="enter-leaves-pending"),
    pytest.param("1\r", True, id="accept-reviewed-merged"),
    pytest.param("2\r", True, id="keep-local"),
    pytest.param("3\r", True, id="accept-upstream"),
    pytest.param("4\r", False, id="leave-pending"),
])
def test_completion_skill_review_keyboard(tmp_path, keys, accepted):
    from winpty import PtyProcess

    repo = Path(__file__).resolve().parents[2]
    peer = tmp_path / "checkout" / "hermes_cli" / "update_completion.py"
    peer.parent.mkdir(parents=True)
    peer.write_text(
        f"import sys; sys.path[:0] = {sys.path!r}\n"
        "import json, os, subprocess\nfrom pathlib import Path\n"
        "if '--prepared' not in sys.argv:\n"
        "    sys.exit(subprocess.call([sys.executable, '-I', '-S', '-u', __file__, *sys.argv[1:], '--prepared']))\n"
        "from hermes_cli.update_completion import _ForwardedOutput\n"
        "request=json.loads(Path(sys.argv[1]).read_text())\n"
        "sys.stdout=_ForwardedOutput(sys.stdout, request['stdout_isatty'])\n"
        "from tools import skills_sync as ss\n"
        "from tools.skills_sync_merge import state_dir\n"
        "source=Path(request['home']).parent/'bundle'/'example'\n"
        "source.mkdir(parents=True)\n"
        "base=b'---\\nname: example\\n---\\nbase\\n'\n"
        "(source/'SKILL.md').write_bytes(base)\n"
        "ss._get_bundled_dir=lambda:source.parent\n"
        "ss._get_optional_dir=lambda:source.parent/'absent'\n"
        "ss.sync_skills(quiet=True)\n"
        "local=Path(request['home'])/'skills'/'example'/'SKILL.md'\n"
        "local.write_bytes(base.replace(b'base',b'ours'))\n"
        "(source/'SKILL.md').write_bytes(base.replace(b'base',b'theirs'))\n"
        "from hermes_cli.update_cmd_maint import _print_bundled_skills_sync_report\n"
        "_print_bundled_skills_sync_report()\n"
        "pending=(state_dir(Path(request['home']),'example')/'pending').exists()\n"
        "print('REVIEW_PENDING',pending,flush=True)\n"
        "Path(sys.argv[2]).write_text(json.dumps({'schema':1,'update_id':'console-test',"
        "'exit_code':0,'receipt':{'update_id':'console-test','finished_at':'test',"
        "'outcome':'success'}}))\n",
        encoding="utf-8", newline="\n",
    )
    parent = tmp_path / "parent.py"
    request = {"source": str(peer.parents[1]), "home": str(tmp_path / "home"),
               "receipt": {"update_id": "console-test"}}
    parent.write_text(
        f"import sys; sys.path.insert(0, {str(repo)!r})\n"
        "import ctypes\nfrom ctypes import wintypes\n"
        "k=ctypes.windll.kernel32\n"
        "k.GetStdHandle.restype=wintypes.HANDLE\n"
        "h=k.GetStdHandle(-10); before=wintypes.DWORD(); after=wintypes.DWORD()\n"
        "if sys.stdin.isatty() and sys.stdout.isatty(): assert k.GetConsoleMode(h,ctypes.byref(before))\n"
        "from hermes_cli.update_completion import run_completion\n"
        f"result=run_completion({request!r})\n"
        "assert result['exit_code']==0, result\n"
        "if sys.stdin.isatty() and sys.stdout.isatty(): assert k.GetConsoleMode(h,ctypes.byref(after))\n"
        "print('MODE_RESTORED',before.value==after.value,flush=True)\n"
        "print('PARENT_READY',flush=True)\n"
        "if sys.stdin.isatty() and sys.stdout.isatty(): print('PARENT_INPUT',input(),flush=True)\n",
        encoding="utf-8", newline="\n",
    )
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"), TERM="xterm-256color")
    if keys is None:
        result = subprocess.run([sys.executable, str(parent)], cwd=repo, env=env,
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=20, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert b"Accept this reviewed skill update?" not in result.stdout
        assert b"REVIEW_PENDING True" in result.stdout
        return
    # PowerShell waits for the native updater exactly as the external update terminal does.
    command = "& " + " ".join("'" + arg.replace("'", "''") + "'"
                              for arg in (sys.executable, str(parent)))
    launch = tmp_path / "launch.ps1"
    launch.write_text(command + "\nexit $LASTEXITCODE\n", encoding="utf-8", newline="\n")
    proc = PtyProcess.spawn(subprocess.list2cmdline([
        "powershell.exe", "-NoLogo", "-NoProfile", "-File", str(launch),
    ]), cwd=str(repo), env=env, dimensions=(40, 160))
    chunks = []

    def pump():
        try:
            while True:
                chunks.append(proc.read(4096))
        except (EOFError, OSError):
            return

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        try:
            wait_until(lambda: "Choice" in "".join(chunks), timeout=20)
        except AssertionError:
            pytest.fail("Review did not appear:\n" + "".join(chunks))
        if keys == "1\r":
            from tools.skills_sync_merge import state_dir
            reviewed = state_dir(tmp_path / "home", "example") / "pending" / "merged" / "SKILL.md"
            reviewed.write_bytes(b"---\nname: example\n---\nreviewed\n")
        proc.write(keys)
        try:
            wait_until(lambda: "PARENT_READY" in "".join(chunks), timeout=5)
        except AssertionError:
            pytest.fail("Terminal keyboard did not release completion review:\n" + "".join(chunks))
        proc.write("parent-owns-input\r")
        wait_until(lambda: "PARENT_INPUT parent-owns-input" in "".join(chunks), timeout=5)
        screen = "".join(chunks)
        assert f"REVIEW_PENDING {not accepted}" in screen, screen
        assert "MODE_RESTORED True" in screen, screen
        expected = {"1\r": b"reviewed", "3\r": b"theirs"}.get(keys, b"ours")
        assert (tmp_path / "home" / "skills" / "example" / "SKILL.md").read_bytes() == (
            b"---\nname: example\n---\n" + expected + b"\n")
        wait_until(lambda: not proc.isalive(), timeout=5)
    finally:
        if proc.isalive():
            proc.terminate(force=True)
        reader.join(3)
