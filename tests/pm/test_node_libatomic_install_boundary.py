"""Regression tests for #124926: libatomic repair belongs to install, not verify."""

from pathlib import Path

import pytest

from pm.packages import Nodejs


_LOADER = (
    "node: error while loading shared libraries: libatomic.so.1: "
    "cannot open shared object file: No such file or directory"
)


def _node_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.platforms("posix")
def test_staged_repair_installs_libatomic_then_reprobes(tmp_path, monkeypatch):
    node = tmp_path / "bin" / "node"
    _node_script(node, f'#!/bin/sh\necho "{_LOADER}" >&2\nexit 127\n')
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")

    calls = {"n": 0}

    def install():
        calls["n"] += 1
        _node_script(node, "#!/bin/sh\necho v26.7.0\n")
        return True, "install with sudo dnf install -y libatomic"

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", install)
    package = Nodejs()
    first = package.verify(tmp_path, "linux-x64")
    repaired = package.repair_staged_verification(tmp_path, "linux-x64", first)

    assert calls["n"] == 1
    assert repaired == ""


def test_auto_repair_never_reads_a_tty(monkeypatch):
    from pm import libatomic

    monkeypatch.setattr(libatomic, "_is_root", lambda: False)
    monkeypatch.setattr(libatomic, "_host_install_command", lambda: ("dnf", "install", "-y", "libatomic"))
    monkeypatch.setattr(libatomic.shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None)

    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return type("Result", (), {"returncode": 1})()

    monkeypatch.setattr(libatomic.subprocess, "run", run)

    attempted, _remedy = libatomic.try_install_libatomic()

    assert attempted is False
    assert captured["argv"][:2] == ["/usr/bin/sudo", "-n"]
    assert captured["stdin"] is libatomic.subprocess.DEVNULL
