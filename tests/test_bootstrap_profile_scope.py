"""CLI dependency bootstrap must inherit the same profile as the application."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_PROBE = r"""
import importlib, json, os, runpy, sys
from pathlib import Path
root, entry, argv = sys.argv[1:]
sys.path.insert(0, root)
sys.argv = ["hermes", *json.loads(argv)]
from hermes_cli import venv_sync
from pm.runtime import runtime_environment
import hermes_constants
hermes_constants._get_platform_default_hermes_home = lambda: Path(os.environ["HOME"]) / ".hermes"
class Boundary(BaseException):
    pass
observed = {}
def prepare(root, argv):
    observed.update(home=runtime_environment()["HERMES_HOME"], argv=argv)
    raise Boundary()
venv_sync.prepare_launch = prepare
def launch_entry():
    from hermes_cli._launchers import _launcher_script
    if entry != "launcher":
        target = "hermes_cli.main" if entry == "launcher-module" else "gateway.run"
        sys.argv[1:1] = ["--run-module", target]
    exec(_launcher_script("hermes", Path(root), None))
def runtime_entry():
    from hermes_cli._launchers import runtime_command
    code = "pass" if entry == "runtime-code" else None
    target = "gateway.run" if entry == "runtime-other" else "hermes_cli.main"
    command = runtime_command(Path(root), python=sys.executable, code=code, module=target)
    exec(command[3])
def sealed_entry():
    from scripts.build import launcher_wrapper
    launcher_wrapper.configure = lambda _here: None
    launcher_wrapper.HERMES_ENTRY_MODULE = "hermes_cli.main" if entry == "sealed" else "acp_adapter.entry"
    launcher_wrapper.main()
entries = {
    "import": lambda: importlib.import_module("hermes_cli.main"),
    "module": lambda: runpy.run_module("hermes_cli.main", run_name="__main__", alter_sys=True),
    "script": lambda: runpy.run_path(str(Path(root) / "hermes_cli/main.py"), run_name="__main__"),
    "other": lambda: importlib.import_module("hermes_bootstrap"),
    **dict.fromkeys(("launcher", "launcher-module", "launcher-other"), launch_entry),
    **dict.fromkeys(("runtime", "runtime-code", "runtime-other"), runtime_entry),
    **dict.fromkeys(("sealed", "sealed-other"), sealed_entry),
}
try:
    entries[entry]()
except Boundary:
    if entry not in {"other", "launcher-other", "runtime-code", "runtime-other", "sealed-other"}:
        from hermes_cli._startup_profile import finish_profile_override, explicit_cli_profile
        finish_profile_override()
        finish_profile_override()
        observed.update(dispatch_home=runtime_environment()["HERMES_HOME"],
                        dispatch_argv=sys.argv[1:], explicit=explicit_cli_profile())
    print(json.dumps(observed))
"""


def _probe(tmp_path, *, entry="import", argv=None, env_home=None, extra_env=None):
    env = {key: value for key, value in os.environ.items() if key not in {
        "HERMES_HOME", "HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD",
        "HERMES_GATEWAY_EXTERNAL_SUPERVISOR", "INVOCATION_ID",
    }}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               HERMES_RUNTIME_DIR=str(tmp_path / "runtime"))
    if env_home is not None:
        env["HERMES_HOME"] = str(env_home)
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _PROBE,
         str(Path(__file__).resolve().parents[1]), entry, json.dumps(argv or ["update", "--plan"])],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), result.stderr


@pytest.mark.parametrize("entry", ["import", "module", "script", "launcher", "launcher-module", "runtime", "sealed"])
def test_pm_bootstrap_follows_selected_profile_without_changing_relaunch_argv(tmp_path, entry):
    root = tmp_path / ".hermes"
    for name in ("alpha", "beta"):
        profile = root / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("alpha", "beta", "alpha"):
        (root / "active_profile").write_text(name + "\n", encoding="utf-8")
        for argv, expected, explicit in [
            (["update", "--plan"], root / "profiles" / name, None),
            (["-p", "beta", "update", "--plan"], root / "profiles" / "beta", "beta"),
            (["--profile=default", "update", "--plan"], root, "default"),
        ]:
            observed, stderr = _probe(tmp_path, entry=entry, argv=argv)
            assert observed == {
                "home": str(expected), "argv": argv,
                "dispatch_home": str(expected), "dispatch_argv": ["update", "--plan"],
                "explicit": explicit,
            }
            assert "HERMES_HOME fallback" not in stderr


@pytest.mark.parametrize("entry,argv,extra_env", [
    ("import", ["gateway", "run"], {"HERMES_SUPERVISED_CHILD": "1"}),
    ("import", ["serve", "--ssh-session-token-file", "/unused/token"], {}),
    ("other", ["-p", "alpha", "gateway", "run"], {}),
    ("launcher-other", ["-p", "alpha", "gateway", "run"], {}),
    ("runtime-code", ["-p", "alpha", "gateway", "run"], {}),
    ("runtime-other", ["-p", "alpha", "gateway", "run"], {}),
    ("sealed-other", ["-p", "alpha", "gateway", "run"], {}),
])
def test_fixed_identity_entrypoints_do_not_adopt_cli_sticky_profile(tmp_path, entry, argv, extra_env):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "alpha"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("alpha\n", encoding="utf-8")
    observed, _stderr = _probe(tmp_path, entry=entry, argv=argv,
                               env_home=root, extra_env=extra_env)
    assert observed["home"] == str(root)
    expected_argv = ["--run-module", "gateway.run", *argv] if entry == "launcher-other" else argv
    assert observed["argv"] == expected_argv
    if entry not in {"other", "launcher-other", "runtime-code", "runtime-other", "sealed-other"}:
        assert observed["dispatch_home"] == str(root)
        assert observed["dispatch_argv"] == argv
        assert observed["explicit"] is None
