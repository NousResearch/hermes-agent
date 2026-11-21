"""Dependency activation never crosses a CPython minor-version boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    (
        "minor_delta",
        "site_exists",
        "selection_mode",
        "stderr_mode",
        "config_mode",
        "expected",
        "warns",
        "error",
    ),
    [
        (-1, True, "facts", "open", "valid", ["False", "True", "True", "True"], True, None),
        (0, True, "facts", "open", "valid", ["True", "False", "False", "False"], False, None),
        (1, True, "facts", "open", "valid", ["False", "True", "True", "True"], True, None),
        (1, False, "facts", "open", "valid", ["False", "True", "True", "True"], True, None),
        (-1, True, "facts", "none", "valid", ["False", "True", "True", "True"], True, None),
        (-1, True, "facts", "closed", "valid", ["False", "True", "True", "True"], False, None),
        (0, False, "facts", "open", "valid", None, False, "has no site-packages"),
        (0, False, "direct", "open", "valid", ["False", "True", "True", "True"], False, None),
        (-1, True, "facts", "open", "invalid", ["False", "True", "True", "True"], True, None),
        (-1, True, "payload", "open", "valid", ["False", "True", "True", "True"], True, None),
    ],
)
def test_dependency_generation_activation_respects_the_running_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minor_delta: int,
    site_exists: bool,
    selection_mode: str,
    stderr_mode: str,
    config_mode: str,
    expected: list[str] | None,
    warns: bool,
    error: str | None,
) -> None:
    """Only matching activates; every rejected selection releases its generation."""
    from pm import environments

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    if selection_mode == "payload":
        generation = tmp_path / "payload"
        root = generation / "hermes-agent"
        root.mkdir(parents=True)
        (generation / "manifest.json").write_text(
            json.dumps(
                {
                    "repo": root.name,
                    "venv": "venv",
                    "store": "tools",
                    "runtime": {"commands": {}},
                }
            ),
            encoding="utf-8",
        )
    else:
        root = tmp_path / "repo"
        state = environments.install_state_dir(root)
        generation = state / "environments" / "foreign"
    venv = generation / "venv"
    foreign = (sys.version_info.major, sys.version_info.minor + minor_delta)
    site = venv / (
        "Lib/site-packages"
        if os.name == "nt"
        else f"lib/python{foreign[0]}.{foreign[1]}/site-packages"
    )
    venv.mkdir(parents=True)
    if config_mode == "invalid":
        (venv / "pyvenv.cfg").write_bytes(
            f"version_info = {foreign[0]}.{foreign[1]}.0\n".encode() + b"\xff\xfebad\n"
        )
    else:
        (venv / "pyvenv.cfg").write_text(
            f"version_info = {foreign[0]}.{foreign[1]}.0\n", encoding="utf-8"
        )
    if site_exists:
        site.mkdir(parents=True)
        (site / "foreign_abi_only.py").write_text("ORIGIN = 'foreign'\n", encoding="utf-8")
    if selection_mode != "payload":
        (generation / ".lease-managed").touch()
    if selection_mode == "facts":
        (state / "facts.json").write_text(
            json.dumps({"schema": 1, "packages": {"venv": {"environment": str(venv)}}}),
            encoding="utf-8",
        )

    code = """
import importlib.util
import os
import sys
from pathlib import Path

repo, root, foreign_site, generation = map(Path, sys.argv[1:5])
sys.path.insert(0, str(repo))
from pm import environments

if sys.argv[6] == "direct":
    environments.committed_venv = lambda _root: generation / "venv"
if sys.argv[5] == "none":
    sys.stderr = None
elif sys.argv[5] == "closed":
    sys.stderr.close()
leases = generation / ".leases"
try:
    environments.activate_dependencies(root)
    environments.activate_dependencies(root)
except RuntimeError as exc:
    print(f"ERROR: {exc}")
    print(not leases.is_dir() or not any(leases.iterdir()))
    raise SystemExit(0)
paths = [Path(entry).resolve() for entry in sys.path if entry]
print(foreign_site.resolve() in paths)
print(importlib.util.find_spec("foreign_abi_only") is None)
print(str(foreign_site.resolve()) not in os.environ.get("PYTHONPATH", "").split(os.pathsep))
print(not leases.is_dir() or not any(leases.iterdir()))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(Path(__file__).resolve().parents[2]),
            str(root),
            str(site),
            str(generation),
            stderr_mode,
            selection_mode,
        ],
        cwd=root.parent,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    if error is not None:
        lines = result.stdout.splitlines()
        assert lines[0].startswith("ERROR: ") and error in lines[0]
        assert lines[1:] == ["True"]
    else:
        assert result.stdout.splitlines() == expected
    assert ("keeping the Python" in result.stderr) is warns


def _windows_style_venv(tmp_path: Path, cfg: bytes) -> Path:
    """A venv with Lib/site-packages and no lib/pythonX.Y, so only pyvenv.cfg dates it."""
    venv = tmp_path / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_bytes(cfg)
    return venv


@pytest.mark.parametrize(
    "cfg",
    [
        b"version_info = 3.99.0\n\xff\xfebad\n",
        b"\xff\xfebad\nversion_info = 3.99.0\n",
        b"home = /x/\xff\xfe\nversion = 3.99.1\n",
        b"\xef\xbb\xbfversion_info = 3.99.0\n\xff\n",
    ],
)
def test_corrupt_pyvenv_cfg_keeps_parseable_version_without_a_posix_lib_dir(tmp_path, cfg):
    from pm import environments

    venv = _windows_style_venv(tmp_path, cfg)
    assert environments.venv_python_version(venv) == (3, 99)


def test_pm_built_marker_survives_undecodable_pyvenv_cfg_bytes(tmp_path):
    from pm import extras
    from pm.environment import PM_BUILT_KEY

    venv = _windows_style_venv(tmp_path, f"{PM_BUILT_KEY} = true\n".encode() + b"\xff\xfebad\n")
    assert extras._built_by_pm(venv) is True


def test_windows_scheduler_cfg_reader_keeps_parseable_lines_through_corrupt_bytes(tmp_path):
    from cron import scheduler_script

    venv = _windows_style_venv(tmp_path, b"home = C:\\py\nversion = 3.99.1\n\xff\xfebad\n")
    assert scheduler_script._read_windows_pyvenv_cfg(venv)["version"] == "3.99.1"


def test_scheduler_cfg_reader_keeps_the_keys_its_consumer_reads(tmp_path):
    from cron import scheduler_script

    venv = _windows_style_venv(tmp_path, b"home = C:\\py\nuv = true\n\xff\xfebad\n")
    cfg = scheduler_script._read_windows_pyvenv_cfg(venv)
    assert cfg["home"] == "C:\\py"
    assert "uv" in cfg


def test_macos_anchor_interpreter_source_survives_undecodable_cfg_bytes(tmp_path):
    from hermes_cli import macos_tcc_anchor

    base = tmp_path / "store" / "python-3.13.1" / "bin"
    base.mkdir(parents=True)
    (base / "python3.13").write_bytes(b"")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_bytes(b"")  # a real file, so only the cfg names the base
    (venv / "pyvenv.cfg").write_bytes(f"home = {base}\nuv = true\n".encode() + b"\xff\xfebad\n")
    assert macos_tcc_anchor._interpreter_source(venv) == str(base / "python3.13")


def test_posix_lib_dir_outranks_a_drifted_cfg_version_when_cfg_is_corrupt(tmp_path):
    from pm import environments

    venv = tmp_path / "venv"
    (venv / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_bytes(b"version_info = 3.99.0\n\xffbad\n")
    assert environments.venv_python_version(venv) == (3, 13)
