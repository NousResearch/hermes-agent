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
    assert repaired == ("", "")


def test_auto_repair_never_reads_a_tty(monkeypatch):
    from pm import libatomic

    monkeypatch.setattr(libatomic, "_ATTEMPT", None)
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


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("flags", ["", "NON_INTERACTIVE=true"])
def test_installer_sudo_warmup_honours_non_interactive(tmp_path, flags):
    """The prerequisites warm-up may prompt for sudo, but never under --non-interactive."""
    import os
    import pty

    script = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"
    bin_dir = tmp_path / "bin"
    log = tmp_path / "sudo.log"
    _node_script(bin_dir / "ldconfig", "#!/bin/sh\nexit 0\n")
    _node_script(bin_dir / "id", "#!/bin/sh\necho 1000\n")
    _node_script(bin_dir / "sudo", f'#!/bin/sh\necho "$*" >> "{log}"\nexit 0\n')
    probe = f'source "$1" --manifest\n{flags}\nuv_bootstrap_target() {{ echo linux-x64; }}\nstage_prerequisites\n'
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(tmp_path))
    # A real controlling terminal: has_terminal() and `sudo -v </dev/tty` open it.
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe("bash", ["bash", "-c", probe, "probe", str(script)], env)
    while True:
        try:
            if not os.read(fd, 4096):
                break
        except OSError:
            break
    os.waitpid(pid, 0)

    prompted = log.is_file() and "-v" in log.read_text().split()
    assert prompted is (flags == "")


def test_update_completion_installs_before_detaching(monkeypatch):
    """run_completion's child has no controlling terminal; the pre-install runs first, in-session."""
    from hermes_cli import update_completion

    monkeypatch.setattr(update_completion.sys, "platform", "linux")
    calls = []
    monkeypatch.setattr(update_completion.subprocess, "run",
                        lambda argv, **kw: calls.append(("run", argv, kw)))

    def popen(argv, **kw):
        calls.append(("popen", argv, kw))
        raise RuntimeError("stop after spawn")

    monkeypatch.setattr(update_completion.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="stop after spawn"):
        update_completion.run_completion({"source": "/nonexistent", "home": "/nonexistent",
                                          "receipt": {"update_id": "u"}})
    assert [kind for kind, _argv, _kw in calls] == ["run", "popen"]
    _kind, argv, kw = calls[0]
    assert "install_before_lock" in " ".join(argv)
    assert "start_new_session" not in kw and kw.get("check") is False
