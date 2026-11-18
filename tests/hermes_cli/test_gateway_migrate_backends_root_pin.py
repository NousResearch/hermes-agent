"""Every printed step targets the dry run's own home, also under a Hermes root other than ~/.hermes.

``hermes -p NAME`` resolves NAME against the root the shell's own HERMES_HOME names
(``profiles.resolve_profile_env``), so a home under ``HERMES_HOME=/srv/team-hermes`` must pin that root, or the
steps would act on the user's ordinary ~/.hermes. The printed line runs through ``/bin/sh`` here, with a fake
``hermes`` that resolves its ``-p`` the way the real CLI does and prints the home it lands in.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_migrate_backends_plan as plan
from hermes_cli import gateway_migrate_backends_windows as win

_POSIX = pytest.mark.platforms("linux", "macos")  # runs the printed line through /bin/sh

_REPO = Path(__file__).resolve().parents[2]
_FAKE = """#!{python}
import json, os, sys
sys.path.insert(0, {repo!r})
from hermes_cli.profiles import resolve_profile_env
args = sys.argv[1:]
name = args[args.index("-p") + 1] if "-p" in args else None
home = resolve_profile_env(name) if name else os.environ.get("HERMES_HOME")
print(json.dumps({{"home": home, "args": args}}))
"""


@pytest.fixture
def team_root(tmp_path, monkeypatch):
    """A user whose standard root is ~/.hermes, and a team root with a space and a quote in its path."""
    user = tmp_path / "user"
    (user / ".hermes").mkdir(parents=True)
    root = tmp_path / "team's hermes root"
    for home in (root, root / "profiles" / "work"):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("model:\n  default: fake\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("HERMES_HOME", str(root))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "hermes"
    fake.write_text(_FAKE.format(python=sys.executable, repo=str(_REPO)), encoding="utf-8")
    fake.chmod(0o755)
    return SimpleNamespace(user=user, root=root, bin=bin_dir)


def _run(line: str, env_home: Path, bin_dir: Path) -> dict:
    """*line* as a user types it in a fresh shell: their HOME, no HERMES_HOME of their own."""
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    env.update(HOME=str(env_home), PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', '')}")
    done = subprocess.run(["/bin/sh", "-c", line], env=env, cwd=env_home, capture_output=True, text=True,
                          timeout=60, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


@_POSIX
@pytest.mark.parametrize("profile", ["default", "work"])
def test_a_printed_step_under_another_root_runs_against_that_home(team_root, profile):
    home = team_root.root if profile == "default" else team_root.root / "profiles" / "work"

    line = f"{plan.hermes_command(str(home))} gateway status"
    ran = _run(line, team_root.user, team_root.bin)

    assert Path(ran["home"]) == home, line
    assert ran["args"] == ["-p", profile, "gateway", "status"]


@_POSIX
def test_every_by_hand_hermes_step_runs_against_that_home(team_root):
    home = team_root.root / "profiles" / "work"
    item = SimpleNamespace(supervisor="systemd", scope="user", unit="hermes-dashboard.service", pid=None)

    steps = [s for s in plan.manual_steps("127.0.0.1", 9119, item, env_keys=(), home=str(home))
             if not s.startswith("#") and "hermes -p" in s]

    assert steps, steps
    for step in steps:
        assert Path(_run(step, team_root.user, team_root.bin)["home"]) == home, step


@_POSIX
def test_the_standard_root_needs_no_pin(team_root, monkeypatch):
    work = team_root.user / ".hermes" / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text("model:\n  default: fake\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_HOME")

    assert plan.hermes_command(str(work)) == "hermes -p work"
    assert plan.hermes_command(str(team_root.user / ".hermes")) == "hermes -p default"


def _ps_unquote(literal: str) -> str:
    """The value of a PowerShell single-quoted string: each doubled quote mark (', ‘, ’, ‚, ‛) is one."""
    assert literal[0] == "'" and literal[-1] == "'"
    body, out, i = literal[1:-1], [], 0
    while i < len(body):
        out.append(body[i])
        if body[i] in "'\u2018\u2019\u201a\u201b":
            assert body[i + 1] == body[i], literal  # an undoubled quote would end the string
            i += 1
        i += 1
    return "".join(out)


@pytest.mark.parametrize("root", ["C:\\Teams\\team's hermes root", "C:\\Teams\\o\u2019brien $x $(Get-Date)"])
def test_the_windows_pin_is_one_powershell_single_quoted_string(root):
    line = plan.pin_hermes_home(root, "hermes -p work", powershell=True)

    assert line.startswith("$env:HERMES_HOME=") and line.endswith("'; hermes -p work")
    assert _ps_unquote(line[len("$env:HERMES_HOME="):-len("; hermes -p work")]) == root


@_POSIX  # the team root is a POSIX path here; the pin's quoting is tested above on any host
def test_the_windows_recipe_pins_the_root_on_every_hermes_line(team_root):
    home = str(team_root.root / "profiles" / "work")
    item = SimpleNamespace(supervisor="scheduled-task", unit="Hermes Dashboard", unit_path=None, pid=11)

    steps = win.manual_steps("127.0.0.1", 9119, item, home=home)

    pin = plan.pin_hermes_home(team_root.root, "hermes -p work", powershell=True)
    commands = [s for s in steps if not s.startswith("#") and "hermes" in s]
    assert commands and all(s.startswith(pin + " ") for s in commands), steps
    assert steps[0].startswith("# ") and "PowerShell" in steps[0]


@_POSIX
def test_the_windows_status_branch_still_reads_as_three_pinned_commands(team_root):
    from tests.hermes_cli.test_gateway_migrate_backends_steps_wording import _groups
    home = str(team_root.root / "profiles" / "work")
    item = SimpleNamespace(supervisor="scheduled-task", unit="Hermes Dashboard", unit_path=None, pid=11)

    groups = _groups(win.manual_steps("127.0.0.1", 9119, item, home=home))

    pin = plan.pin_hermes_home(team_root.root, "hermes -p work", powershell=True)
    assert [command for _quoted, command in groups] == [f"{pin} gateway {verb}"
                                                        for verb in ("install", "start", "restart")]


def _make_homes(root: Path) -> Path:
    for home in (root, root / "profiles" / "work"):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model:\n  default: fake\n", encoding="utf-8")
    return root / "profiles" / "work"


@_POSIX
@pytest.mark.parametrize("profile", ["default", "work"])
def test_a_root_nested_inside_the_standard_root_is_pinned(team_root, monkeypatch, profile):
    """~/.hermes/team/profiles/work: `hermes -p work` alone would select ~/.hermes/profiles/work."""
    nested = team_root.user / ".hermes" / "team"
    work = _make_homes(nested)
    _make_homes(team_root.user / ".hermes")  # the profile a bare `-p work` would wrongly pick
    home = nested if profile == "default" else work
    monkeypatch.setenv("HERMES_HOME", str(home))

    line = f"{plan.hermes_command(str(home))} gateway status"

    assert line.startswith("HERMES_HOME="), line
    assert Path(_run(line, team_root.user, team_root.bin)["home"]) == home, line


@_POSIX
@pytest.mark.parametrize("profile", ["default", "work"])
def test_a_relative_root_is_pinned_as_an_absolute_path(team_root, tmp_path, monkeypatch, profile):
    """`HERMES_HOME=team` means another directory in a fresh shell, which starts in the user's home."""
    cwd = tmp_path / "where the dry run ran"
    _make_homes(cwd / "team")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HERMES_HOME", "team")
    home = "team" if profile == "default" else "team/profiles/work"

    line = f"{plan.hermes_command(home)} gateway status"

    assert line.startswith(f"HERMES_HOME={shlex.quote(str(cwd / 'team'))} "), line
    assert Path(_run(line, team_root.user, team_root.bin)["home"]) == cwd / home, line


@_POSIX
def test_a_path_that_resolves_to_the_standard_root_is_not_pinned(team_root, tmp_path, monkeypatch):
    alias = tmp_path / "alias-of-standard-root"
    alias.symlink_to(team_root.user / ".hermes")
    monkeypatch.setenv("HERMES_HOME", str(alias))

    assert plan.hermes_command(str(alias)) == "hermes -p default"


def test_the_windows_owner_blocker_prints_the_powershell_pin(team_root):
    home = team_root.root / "profiles" / "work"
    host = SimpleNamespace(home=home, gateway_owner=lambda: ("unknown", None), os_family=lambda: "win32")

    problem = plan._owner_problem(host)

    pin = plan.pin_hermes_home(team_root.root, "hermes -p work", powershell=True)
    assert f"({pin} gateway status)" in problem, problem
    assert "HERMES_HOME=" not in problem.replace("$env:HERMES_HOME=", ""), problem


def test_every_message_naming_a_hermes_command_follows_the_host_shell():
    """Each hermes_command() call in the plan passes powershell= unless it builds the POSIX recipe."""
    import inspect
    source = inspect.getsource(plan)
    calls = re.findall(r"hermes_command\(([^()]*(?:\([^()]*\))?[^()]*)\)", source)
    unqualified = [c for c in calls if "powershell" not in c and not c.startswith("home: Optional")]
    assert unqualified == ["home"], unqualified  # manual_steps: the POSIX recipe (Windows has its own)
