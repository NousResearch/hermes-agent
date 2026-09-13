"""Native setup-pm smoke: verify PATH, PM identity and installed extras."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.lock import Facts, Lockfile
from pm.package import machine_matches_binary
from pm.paths import facts_path, lockfile_path, runtime_facts_path, store_root
from pm.registry import get_package
from pm.store import current_target, tree_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", action="store_true")
    args = parser.parse_args()
    lock = Lockfile(lockfile_path())
    facts = Facts(facts_path())
    target = current_target()
    rows = {}
    for name in ("python", "python3", "node", "npm", "npx"):
        binary = shutil.which(name)
        if binary is None:
            raise RuntimeError(f"{name} is missing from PATH")
        result = subprocess.check_output([binary, "--version"], text=True, encoding="utf-8", timeout=60).strip()
        package = {"python3": "python", "npx": "npm"}.get(name, name)
        pin = lock.version(package)
        expected = pin.partition("+")[0]
        actual = result.split()[1] if package == "python" else result.removeprefix("v")
        if actual != expected:
            raise RuntimeError(f"{name} on PATH is {result}, expected {pin}: {binary}")
        entry = store_root() / facts.get(package)["entry"]
        artifact = get_package(package).binary(entry, target)
        if facts.get(package)["artifacts"] != [a["sha256"] for a in lock.artifacts(package, target)]:
            raise RuntimeError(f"{name} has the wrong pinned artifact identity")
        if machine_matches_binary(artifact, target) is False:
            raise RuntimeError(f"{name} is not native {target}")
        if package == name and facts.get(package)["digest"] != tree_digest(entry):
            raise RuntimeError(f"{name} store was mutated after verification")
        rows[name] = {"path": binary, "version": result, "target": target}
    if args.extras:
        if Facts(runtime_facts_path()).get("venv") is not None:
            raise RuntimeError("CI test dependencies must not publish a runtime generation")
        if Path(sys.prefix).resolve() != Path(os.environ["VIRTUAL_ENV"]).resolve():
            raise RuntimeError("PATH Python did not select the isolated test environment")
        import pytest
        import ruamel.yaml

        rows["dependencies"] = {"pytest": pytest.__version__, "ruamel.yaml": ruamel.yaml.__version__}
        code = "import sys,pytest; print(sys.prefix); print(pytest.__version__)"
        probe = subprocess.check_output([shutil.which("python3"), "-c", code], text=True, encoding="utf-8", timeout=60)
        if pytest.__version__ not in probe:
            raise RuntimeError("python3 did not inherit the installed dev extra")
    print(json.dumps(rows, indent=2))
    destination = Path(os.environ["RUNNER_TEMP"]) / "pm-toolchain-proof.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
