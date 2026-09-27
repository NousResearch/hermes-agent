"""The installer's bootstrap uv writes state under the machine root, not the user's (#101269).

uv's defaults (``~/.cache/uv``, ``~/.local/share/uv``) belong to the uv the user
installed themselves: a Hermes download landing there is visible to their
``uv python list``, removable by their ``uv python uninstall``, and fills a cache
they own. Both pins must name slots under ``get_default_hermes_root()`` — the
cache in its OWN ``cache/uv-bootstrap`` slot (``pm.packages.uv_cache_dir()`` is
the payload-seeded directory; bootstrap bytes there first would make that seed
skip subtrees yet still mark itself done), the python dir shared with PM. The
profile case matters: the state is machine-scoped, so staging it under a named
profile home would hide it from every other profile and from PM.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"
pytestmark = pytest.mark.platforms("posix")

# Only the shapes whose expected root differs from ``$HERMES_HOME`` are worth
# running twice over: a default home makes the old rule and the new one agree,
# which is exactly how a regression here would slip through.
HOME_KINDS = ("default", "profile")


def _fake_uv(path: Path, *, bash: str, record: Path, boot_py: Path) -> None:
    """A uv that logs the state dirs it was given and answers the bootstrap ladder."""
    path.write_text(
        f"#!{bash}\n"
        'case "$1" in\n'
        '  --version) echo "uv 99.0.0"; exit 0 ;;\n'
        f'  python) printf "%s|%s\\n" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" '
        f'>> {shlex.quote(str(record))} ;;\n'
        'esac\n'
        'case "$1 $2" in\n'
        f'  "python find") printf "%s\\n" {shlex.quote(str(boot_py))} ;;\n'
        '  "python install") exit 0 ;;\n'
        'esac\n'
        'exit 0\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize("home_kind", HOME_KINDS)
def test_bootstrap_python_pins_uv_state_under_the_machine_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    """Every uv call the bootstrap makes carries Hermes-owned cache and python dirs."""
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap requires bash"
    home = tmp_path / "home" / ".hermes"
    hermes_home = home if home_kind == "default" else home / "profiles" / "coder"
    # Point pm and the bootstrap at one home; HOME rides along so the platform
    # default the fold anchors to is this fixture's, not the developer's.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    from hermes_constants import get_default_hermes_root

    expected_root = get_default_hermes_root()
    assert expected_root == home, (
        f"the fixture's {home_kind} home must fold to {home}, got {expected_root}"
    )

    checkout = tmp_path / "checkout"
    (checkout / "pm").mkdir(parents=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    (checkout / "pm" / "lock.json").write_text(
        json.dumps({"packages": {"python": {"version": version}}}), encoding="utf-8"
    )

    record = tmp_path / "uv-state"
    boot_py = Path(sys._base_executable).resolve()
    assert boot_py.is_file()
    _fake_uv(tmp_path / "uv", bash=bash, record=record, boot_py=boot_py)

    env = {**os.environ, "HOME": str(tmp_path / "home"), "HERMES_HOME": str(hermes_home)}
    env.pop("HERMES_RUNTIME_DIR", None)
    # Source the real script for its functions, then run the real bootstrap.
    script = ('source "$1" --manifest; INSTALL_DIR="$2"; UV_CMD="$3"; bootstrap_python; '
              'printf "%s\\n" "$boot_py"')
    result = subprocess.run(
        [bash, "-c", script, "test", str(INSTALL_SH), str(checkout), str(tmp_path / "uv")],
        env=env, cwd=checkout, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == str(boot_py), result.stdout + result.stderr

    assert record.is_file(), "the bootstrap never invoked uv's python commands"
    lines = record.read_text(encoding="utf-8").splitlines()
    assert lines, "the bootstrap invoked uv without recording its state dirs"
    for line in lines:
        cache, python_dir = line.split("|")
        # The machine root, never the (possibly profile) home -- and the cache
        # keeps its own slot so pm's payload seeding of <root>/cache/uv finds
        # it untouched (a pre-created top-level entry there is skipped by the
        # seed yet the seed still marks itself done).
        assert cache == str(expected_root / "cache" / "uv-bootstrap"), line
        assert python_dir == str(expected_root / "cache" / "uv-python"), line
