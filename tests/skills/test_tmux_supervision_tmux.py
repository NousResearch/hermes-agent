"""Installed-copy CLI acceptance with an isolated real tmux server and real PTYs."""

import sys

import pytest

if sys.platform != "linux":
    pytest.skip("Linux-only skill", allow_module_level=True)

import json
import os
import shutil
import subprocess
import tempfile
import time
import venv
from pathlib import Path

SKILL = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/autonomous-ai-agents/tmux-supervision"
)


def wait_for(check):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if result := check():
            return result
        time.sleep(0.02)
    pytest.fail("isolated tmux fixture did not reach the expected state")


@pytest.mark.parametrize("interactive,exit_code", [(True, 0), (True, 23), (False, 0)])
def test_installed_cli_launches_real_tmux_and_reports_only_process_exit(
    interactive, exit_code
):
    tmux = os.environ.get("TMUX_TEST_EXECUTABLE") or shutil.which("tmux")
    if tmux is None:
        pytest.skip("tmux is required for real terminal acceptance")
    with tempfile.TemporaryDirectory(
        prefix="t", dir=os.environ.get("TMPDIR")
    ) as directory:
        root = Path(directory).resolve()
        home, sockets, workspace = (root / name for name in ("h", "s", "w"))
        for path in (home, sockets, workspace):
            path.mkdir(mode=0o700)
        installed = root / "i"
        shutil.copytree(SKILL, installed, ignore=shutil.ignore_patterns("__pycache__"))
        venv.EnvBuilder(with_pip=False).create(root / "p")
        python = str(root / "p/bin/python")
        command = [python, str(installed / "scripts/tmux_supervise.py")]
        # A private socket directory and empty config exclude all existing servers
        # and user hooks, even when the host already runs a different tmux version.
        client = root / "tmux-client"
        client.write_text(
            f"#!{sys.executable}\nimport os, sys\nos.execv({tmux!r}, "
            f"[{tmux!r}, '-f', os.devnull, *sys.argv[1:]])\n"
        )
        client.chmod(0o700)
        env = {
            "PATH": os.defpath,
            "HOME": str(home),
            "SHELL": "/bin/sh",
            "TMPDIR": os.environ.get("TMPDIR", str(root)),
            "TMUX_TMPDIR": str(sockets),
            "HERMES_HOME": str(home),
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_KEY": "synthetic-tmux-key",
            "HERMES_SESSION_ID": "synthetic-tmux-generation",
            "HERMES_SESSION_CHAT_ID": "synthetic-chat",
            "HERMES_SESSION_THREAD_ID": "",
        }

        def call(argv, *, check=True):
            result = subprocess.run(
                argv,
                cwd=workspace,
                env=env,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            if check:
                assert result.returncode == 0, result.stderr
            return result

        prepared = json.loads(
            call([
                *command,
                "prepare",
                "--workspace",
                str(workspace),
                "--tmux-session",
                "synthetic-command",
                "--state-root",
                str(root / "r"),
            ]).stdout
        )
        run = Path(prepared["run_dir"])
        argument = "spaces ' and $();\nsecond line"
        ready = workspace / "ready.json"
        source = (
            "import json, os, sys; from pathlib import Path; "
            "assert 'hermes_cli' not in sys.modules; "
            f"pending = Path({str(ready.with_suffix('.pending'))!r}); "
            "pending.write_text(json.dumps({"
            "'tty': [os.isatty(fd) for fd in (0, 1, 2)], "
            "'cwd': os.getcwd(), 'argument': sys.argv[1]})); "
            f"pending.replace({str(ready)!r}); "
            + ("assert input() == 'finish'; " if interactive else "")
            + f"sys.exit({exit_code})"
        )
        program = workspace / "argv.json"
        program.write_text(json.dumps([python, "-c", source, argument]))
        watcher = subprocess.Popen(
            [*command, "watch", "--run-dir", str(run), "--timeout", "20"],
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:

            def status():
                return json.loads(
                    call([*command, "status", "--run-dir", str(run)]).stdout
                )

            wait_for(lambda: status()["observer_active"])
            launched = json.loads(
                call([
                    *command,
                    "launch",
                    "--run-dir",
                    str(run),
                    "--command-file",
                    str(program),
                    "--tmux-executable",
                    str(client),
                ]).stdout
            )
            assert launched["status"] == "launched"
            wait_for(ready.exists)
            observed = json.loads(ready.read_text())
            assert observed == {
                "tty": [True, True, True],
                "cwd": str(workspace),
                "argument": argument,
            }
            if interactive:
                wait_for(lambda: status()["cursor"]["seq"] == 1)
                assert watcher.poll() is None, (
                    "idle interactive apps must not imply completion"
                )
                pane = call([
                    str(client),
                    "list-panes",
                    "-t",
                    "=synthetic-command",
                    "-F",
                    "#{pane_id}",
                ]).stdout.strip()
                assert pane.startswith("%") and pane[1:].isdigit()
                call([str(client), "send-keys", "-t", pane, "-l", "finish"])
                call([str(client), "send-keys", "-t", pane, "Enter"])
            output, errors = watcher.communicate(timeout=15)
            assert watcher.returncode == 0, errors
            receipt = json.loads(output)
            assert (receipt["kind"], receipt["exit_code"]) == (
                "process_exited",
                exit_code,
            )
            final = status()
            assert final["cursor"]["terminal"] is True
            assert final["journal"]["state"] == "closed"
            assert source not in (run / "journal.json").read_text()
            assert argument not in (run / "journal.json").read_text()
            replay = call(
                [*command, "watch", "--run-dir", str(run), "--timeout", "1"],
                check=False,
            )
            assert replay.returncode == 2
            assert json.loads(replay.stderr) == {"error": "observation_closed"}
        finally:
            if watcher.poll() is None:
                watcher.kill()
                watcher.communicate(timeout=5)
            # This server belongs only to this fixture; never use the default socket.
            call([str(client), "kill-server"], check=False)
