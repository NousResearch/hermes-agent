"""The passive update check's remote git dies with the CLI that started it.

``hermes chat`` starts ``check_for_updates`` in a daemon thread at startup; on an official
checkout it resolves the stable release and verifies the tag with ``git ls-remote --tags origin``,
which forks ``git remote-https``. A ``/exit`` while that lookup waits on GitHub abandoned the
thread, and the git process and its helper reparented to init and kept running (the 0.23.2
release E2E caught them in ``test_terminal_transcript_pty.py[cli-turns]``).

Real processes, no mocks of process semantics: a child interpreter runs the real
``resolve_source_release`` from a daemon thread against a fake ``git`` that forks a helper and
hangs, then exits the way the CLI does. Both the git and its helper must be gone afterwards.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import pid_exists_stdlib
from tests.waiting import wait_until

pytestmark = [
    pytest.mark.platforms("posix"),  # the owner-death supervisor is POSIX
    # Cleanup SIGKILLs the fake git's pids if they outlived the child: reparented to init, they are
    # outside this test's process subtree but still ours.
    pytest.mark.live_system_guard_bypass,
]

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 40

_CHILD = textwrap.dedent('''
    import sys, threading
    from hermes_cli import source_releases

    fake_git, cwd, marker = sys.argv[1:4]
    answers = {
        "https://api.github.com/repos/o/r/releases/latest":
            {"draft": False, "prerelease": False, "tag_name": "v1.2.3"},
        "https://api.github.com/repos/o/r/commits/v1.2.3": {"sha": "@SHA@"},
    }
    source_releases._json = answers.__getitem__  # GitHub's answers; only git is left to wait on

    threading.Thread(target=source_releases.resolve_source_release, daemon=True,
                     args=("stable", [fake_git], cwd), kwargs={"repository": "o/r", "pointer": False}).start()
    # Exit as /exit does, once the lookup's git and its helper are both running and the lookup is
    # waiting on them (its group handed to the owner-death supervisor): the marker lands from the
    # fake git before the spawning thread registers, so exiting on the marker alone races that.
    import pathlib
    from hermes_cli import process_identity
    from tests.waiting import wait_until
    pids = wait_until(lambda: (lambda p: len(p) >= 2 and p)(pathlib.Path(marker).read_text().split()),
                      timeout=30, what="the fake git and its helper to start")
    wait_until(lambda: int(pids[0]) in process_identity._supervised_pgids, timeout=30,
               what="the lookup's git group to be handed to the owner-death supervisor")
    with process_identity._supervisor_lock:  # the register line is written under this lock
        pass
''').replace("@SHA@", SHA)



def test_exit_during_the_stable_tag_lookup_leaves_no_git_behind(tmp_path):
    marker = tmp_path / "pids"
    marker.write_text("")
    fake_git = tmp_path / "git"
    # ls-remote's shape: the git process plus a forked transport helper, both waiting on the network.
    fake_git.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        sleep 300 &
        echo $$ $! >> {marker}
        wait
    """))
    fake_git.chmod(0o755)
    home = tmp_path / "hermes-home"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}
    pids: list[int] = []
    try:
        child = subprocess.run([sys.executable, "-c", _CHILD, str(fake_git), str(tmp_path), str(marker)],
                               cwd=REPO, env=env, capture_output=True, text=True, timeout=60, check=False)
        pids = [int(p) for p in marker.read_text().split()]
        assert child.returncode == 0, child.stderr
        assert len(pids) == 2, f"the fake git never ran: {child.stderr}"
        wait_until(lambda: not any(map(pid_exists_stdlib, pids)), timeout=15,
                   what="the update check's git and its helper to die with the CLI")
    finally:
        for pid in pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
