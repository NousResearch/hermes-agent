"""litco-slot-probe: argument handling and the command it builds. nsenter never runs here.

The probe's real proof runs in the machine smoke (deploy/host/smoke.sh --topology machine).
"""

from __future__ import annotations

import json
import pwd
import subprocess
import types

import pytest

from tests.host._load import load_script

probe = load_script("litco_slot_probe", "litco-slot-probe")


class FakeSystem:
    """systemctl show answers for two slots; records the probe's own subprocess.run."""

    def __init__(self, monkeypatch, *, states=None, reader_out=None):
        self.states = {"a": ("active", 101), "b": ("active", 202), **(states or {})}
        self.argv = None
        self.reader_out = reader_out or {k: ["denied", "PermissionError: Operation not permitted"]
                                         for k in probe.PROBES}
        monkeypatch.setattr(probe, "_run", self._show)
        monkeypatch.setattr(probe, "unit_cgroup", lambda pid: probe.Path(f"/sys/fs/cgroup/unit-{pid}"))
        monkeypatch.setattr(probe.pwd, "getpwnam", lambda name: types.SimpleNamespace(
            pw_uid=20001, pw_gid=20001, pw_dir=f"/srv/litco/m/{name[2:]}"))
        monkeypatch.setattr(probe.os, "geteuid", lambda: 0)
        monkeypatch.setattr(probe.subprocess, "run", self._exec)

    def _show(self, argv):
        slot = argv[-1].split("@", 1)[1].split(".", 1)[0]
        state, pid = self.states.get(slot, ("inactive", 0))
        return subprocess.CompletedProcess(argv, 0, f"ActiveState={state}\nMainPID={pid}\n", "")

    def _exec(self, argv, **kwargs):
        self.argv, self.kwargs = argv, kwargs
        return subprocess.CompletedProcess(argv, 0, json.dumps(self.reader_out) + "\n", "")


def test_all_denied_prints_one_json_object_and_exits_0(monkeypatch, capsys):
    fake = FakeSystem(monkeypatch)
    assert probe.main(["a", "b"]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out) == {"env": "denied", "proc": "denied", "home": "denied", "metadata": "denied"}
    assert "a -> b metadata: denied (PermissionError: Operation not permitted)" in err
    # The reads run as m_a, inside a's namespaces, from a's cgroup; the targets are b's.
    argv = fake.argv
    assert argv[:6] == ["nsenter", "--target=101", "--mount", "--pid", "--ipc", "--"]
    assert argv[6:12] == ["setpriv", "--reuid=20001", "--regid=20001", "--init-groups", "--no-new-privs", "--"]
    assert argv[12:15] == ["/usr/bin/python3", "-I", "-c"] and argv[15] == probe.READER
    assert argv[16:] == ["/run/litco-agent/b.env", "/proc/202/environ", "/srv/litco/m/b",
                         "http://169.254.169.254/metadata/v1/user-data"]
    assert callable(fake.kwargs["preexec_fn"])
    assert set(fake.kwargs["env"]) == {"PATH", "HOME", "LANG"}


def test_any_read_fails_the_probe(monkeypatch, capsys):
    out = {k: ["denied", "x"] for k in probe.PROBES}
    out["metadata"] = ["READ", "HTTP 200"]
    FakeSystem(monkeypatch, reader_out=out)
    assert probe.main(["a", "b"]) == 1
    assert json.loads(capsys.readouterr().out)["metadata"] == "READ"


@pytest.mark.parametrize("args", [["a"], ["a", "b", "c"], ["--help", "b"], ["A", "b"], ["a", "b;id"],
                                  ["a", "a" * 31], ["a", "a"]])
def test_bad_arguments_exit_2_and_run_nothing(monkeypatch, args):
    fake = FakeSystem(monkeypatch)
    assert probe.main(args) == 2
    assert fake.argv is None


def test_a_slot_that_is_not_running_exits_2(monkeypatch, capsys):
    fake = FakeSystem(monkeypatch, states={"b": ("inactive", 0)})
    assert probe.main(["a", "b"]) == 2
    assert "slot b is not running" in capsys.readouterr().err
    assert fake.argv is None


def test_a_reader_that_did_not_run_exits_3(monkeypatch, capsys):
    fake = FakeSystem(monkeypatch)
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "nsenter: cannot open"))
    assert probe.main(["a", "b"]) == 3
    assert "the reader did not run" in capsys.readouterr().err
    assert fake.argv is None


def test_reader_classifies_what_it_can_and_cannot_read(tmp_path):
    """The fixed reader, run locally as this user: a readable file is READ, a missing one denied."""
    readable = tmp_path / "b.env"
    readable.write_text("SECRET=x")
    out = subprocess.run(["python3", "-I", "-c", probe.READER, str(readable), str(tmp_path / "missing"),
                          str(tmp_path), "http://127.0.0.1:9/never"], capture_output=True, text=True, timeout=30)
    found = json.loads(out.stdout)
    assert found["env"][0] == "READ" and found["home"][0] == "READ"
    assert found["proc"][0] == "denied" and found["metadata"][0] == "denied"


def test_probe_needs_root(monkeypatch, capsys):
    monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
    assert probe.main(["a", "b"]) == 2
    assert "must run as root" in capsys.readouterr().err


def test_pwd_is_the_real_module():
    assert probe.pwd is pwd
