"""Check the shipped Nix agent, not a reconstruction of its assembly recipe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


package, project, inputs_file = map(Path, sys.argv[1:])
inputs = json.loads(inputs_file.read_text())
entries = tomllib.loads(project.read_text())["project"]["scripts"]
command_map = json.loads((package / "command-map.json").read_text())
assert command_map["commands"].keys() == entries.keys()
assert {p.name for p in (package / "bin").iterdir()} == entries.keys()

repo = package / inputs["repo"]
for name, source in inputs["resources"].items():
    resource = repo / name
    assert resource.is_symlink(), resource
    assert resource.samefile(source), (resource, source)
for name, key in (("tui", "HERMES_TUI_DIR"), ("web", "HERMES_WEB_DIST")):
    resource = Path(command_map["env"][key])
    assert resource.is_symlink(), resource
    assert resource.samefile(inputs["frontends"][name])
assert not (repo / "run_agent.py").exists(), "installed Python code must not be copied"
assert not list(repo.glob("*.dist-info")), "Nix keeps the uv2nix wheel's metadata"
assert command_map["env"]["HERMES_PYTHON"] == inputs["python"]
assert command_map["env"]["HERMES_INSTALL_ROOT"] == str(repo)
stamp = json.loads((repo / "install-stamp.json").read_text())
assert stamp["pmRuntime"] == inputs["pm_runtime"]
assert (Path(inputs["pm_runtime"]) / "pm-runtime.json").is_file()

# Observe the environment the *actual wrapper* hands Python and import its
# declared entrypoint, without starting an LLM session. The legacy run_agent
# entrypoint does not parse --help and swallows initialization errors, so its
# exit status alone is not an import check. No parsing makeWrapper's shell text.
with tempfile.TemporaryDirectory() as temporary:
    home = Path(temporary)
    probe = home / "probe"
    probe.mkdir()
    (probe / "sitecustomize.py").write_text(
        "import importlib, json, os\n"
        "observed = dict(os.environ)\n"
        "module, function = os.environ['NIX_CHECK_ENTRY'].split(':')\n"
        "assert callable(getattr(importlib.import_module(module), function))\n"
        "print(json.dumps(observed), flush=True)\n"
        "os._exit(0)\n"
    )
    env = {"HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
           "PATH": os.defpath, "LANG": "C.UTF-8", "TZ": "UTC"}
    for name, entry in entries.items():
        command = command_map["commands"][name]
        assert command["entry"] == entry
        assert Path(command["source"]).samefile(Path(inputs["command_dir"]) / name)
        executable = package / command["destination"]
        assert os.access(executable, os.X_OK), executable
        result = subprocess.run([str(executable), "--help"], cwd=home,
                                env={**env, "PYTHONPATH": str(probe), "NIX_CHECK_ENTRY": entry},
                                text=True, capture_output=True, check=True, timeout=60)
        observed = json.loads(result.stdout)
        for key, value in command_map["env"].items():
            assert observed[key] == value, (name, key, observed.get(key), value)
        assert observed["HERMES_BIN"] == str(package / "bin/hermes")
        print(f"PASS: {name} imports from a clean cwd and receives the shared environment")

    # These public interfaces parse --help without starting an agent session.
    for name in ("hermes", "hermes-acp"):
        subprocess.run([str(package / "bin" / name), "--help"], cwd=home, env=env,
                       check=True, timeout=60, stdout=subprocess.DEVNULL)
        print(f"PASS: {name} --help")

print("PASS: referenced resources, independent PM runtime, and uv2nix code placement")
