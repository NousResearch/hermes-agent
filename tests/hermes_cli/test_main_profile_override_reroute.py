"""Regression: an explicit `-p` routing must survive the module's re-entrant import.

`python -m hermes_cli.main` executes main.py as `__main__`; the first
_apply_profile_override() consumes `-p <name>` from sys.argv. A later
`from hermes_cli.main import ...` (e.g. main_web_build's bytecode sweep inside
main()) re-executes the file in a SECOND module object with `-p` already
stripped — the re-scan must not fall through to the sticky active_profile file
and re-home the process. That leak made plain /api/status resolve
profiles/<active>/gateway_state.json instead of the top-level file
(gateway_running=false, gateway_state=null).
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = (
    "import sys\n"
    "sys.argv = {argv!r}\n"
    "import hermes_cli.main\n"
    "del sys.modules['hermes_cli.main']\n"
    "import hermes_cli.main\n"
    "import os\n"
    "print(os.environ.get('HERMES_HOME', ''))\n"
)

_SCRUB = (
    "HERMES_DEV",
    "HERMES_SUPERVISED_CHILD",
    "HERMES_S6_SUPERVISED_CHILD",
    "HERMES_GATEWAY_EXTERNAL_SUPERVISOR",
    "HERMES_DESKTOP",
)


def _make_home(root: Path, active: str) -> Path:
    home = root / ".hermes"
    (home / "profiles" / "researcher").mkdir(parents=True)
    (home / "profiles" / "coordinator").mkdir(parents=True)
    (home / "active_profile").write_text(active, encoding="utf-8")
    return home


def _run_probe(home: Path, argv: list) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    env["HERMES_HOME"] = str(home)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(argv=argv)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout.strip()


def test_explicit_default_survives_reentrant_import(tmp_path):
    home = _make_home(tmp_path, "researcher")
    assert _run_probe(home, ["probe", "-p", "default"]) == str(home)


def test_explicit_named_survives_reentrant_import(tmp_path):
    home = _make_home(tmp_path, "coordinator")
    assert _run_probe(home, ["probe", "-p", "researcher"]) == str(
        home / "profiles" / "researcher"
    )


def test_sticky_routing_without_flag_unchanged(tmp_path):
    home = _make_home(tmp_path, "researcher")
    assert _run_probe(home, ["probe"]) == str(home / "profiles" / "researcher")


def test_default_sticky_leaves_home_alone(tmp_path):
    home = _make_home(tmp_path, "default")
    assert _run_probe(home, ["probe"]) == str(home)
