"""Executed by the isolated MSIX's Python in the opt-in Windows CI test."""

import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import Mock


def main():
    source, output = (Path(value).resolve() for value in sys.argv[1:3])
    package_root = Path(sys.executable).resolve().parent.parent
    home = output.parent / "probe-home"
    windows_env = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR", "ComSpec", "TEMP", "TMP")
        if key in os.environ
    }
    os.environ.clear()
    os.environ.update(windows_env)
    os.environ.update(
        HERMES_HOME=str(home),
        HOME=str(home),
        USERPROFILE=str(home),
        LOCALAPPDATA=str(home / "AppData/Local"),
        PATH=str(Path(os.environ["SystemRoot"]) / "System32"),
        NPM_CONFIG_CACHE=str(home / "npm-cache"),
        NPM_CONFIG_USERCONFIG=str(home / "empty.npmrc"),
        NPM_CONFIG_UPDATE_NOTIFIER="false",
        NPM_CONFIG_OFFLINE="true",
    )
    home.mkdir(parents=True)
    (home / "empty.npmrc").write_text("", encoding="utf-8")
    sys.path.insert(0, str(source))

    socket.create_connection = Mock(
        side_effect=AssertionError("The MSIX fixture must remain offline")
    )
    socket.socket.connect = Mock(
        side_effect=AssertionError("The MSIX fixture must remain offline")
    )
    import hermes_constants
    from pm import paths

    # Inject only the fixture's installation paths. Resolution, verification, copying
    # and execution use production code and real Windows package identity.
    paths.store_root = Mock(return_value=package_root / "tools")
    paths.writable_store_root = Mock(return_value=home / "tools")
    paths.lockfile_path = Mock(return_value=package_root / "lock.json")
    length = ctypes.c_uint32()
    identity_result = ctypes.windll.kernel32.GetCurrentPackageFullName(
        ctypes.byref(length), None
    )
    assert identity_result == 122, (
        f"Packaged Python identity was not established: {identity_result}"
    )
    rows = []

    def run(label, argv, env, cwd=home):
        result = subprocess.run(
            argv,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        rows.append(
            {
                "case": label,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        output.write_text(
            json.dumps(
                {
                    "python": sys.executable,
                    "identity_result": identity_result,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    npm = hermes_constants.find_node_executable("npm")
    node = hermes_constants.find_node_executable("node")
    baseline = hermes_constants.with_hermes_node_path()
    direct = run("packaged Node directly", [node, "--version"], baseline)
    old = run("original packaged npm batch", [npm, "--version"], baseline)
    assert direct.returncode == 0, rows
    assert old.returncode != 0 and "denied" in old.stderr.lower(), rows
    repaired = hermes_constants.with_hermes_node_path(
        for_subprocess=True, explicit=True
    )
    assert hermes_constants.with_hermes_node_path(for_subprocess=True) == repaired
    assert (
        run("repaired packaged npm batch", [npm, "--version"], repaired).returncode == 0
    ), rows
    dependency, project = home / "dependency", home / "project"
    dependency.mkdir()
    project.mkdir()
    (dependency / "package.json").write_text(
        json.dumps(
            {
                "name": "local-dependency",
                "version": "1.0.0",
                "scripts": {"postinstall": "node postinstall.js"},
            }
        ),
        encoding="utf-8",
    )
    (dependency / "postinstall.js").write_text(
        "require('fs').writeFileSync('installed.txt', process.execPath)",
        encoding="utf-8",
    )
    (project / "package.json").write_text(
        json.dumps(
            {
                "name": "probe-project",
                "version": "1.0.0",
                "dependencies": {"local-dependency": "file:../dependency"},
            }
        ),
        encoding="utf-8",
    )
    for label, arguments in (
        ("offline install", ["install", "--offline", "--no-audit", "--no-fund"]),
        (
            "approve synthetic dependency",
            ["install-scripts", "approve", "local-dependency"],
        ),
        ("offline lifecycle rebuild", ["rebuild", "--offline", "--foreground-scripts"]),
    ):
        assert run(label, [npm, *arguments], repaired, project).returncode == 0, rows
    assert Path(
        (dependency / "installed.txt").read_text(encoding="utf-8")
    ).is_relative_to(home / "tools")


if __name__ == "__main__":
    main()
