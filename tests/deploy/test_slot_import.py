"""litco-slot-import: restoring a migrated matter into a slot's home, in a scratch root."""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

from tests.host._load import HOST, load_script

imp = load_script("litco_slot_import", "litco-slot-import")
SCRIPT = HOST / "litco-slot-import"
SLOT = "a1b2c3d4e5f6"
FIXED_NOW = _dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)


@pytest.fixture
def root(tmp_path):
    home_root = tmp_path / "srv" / "m"
    state_dir = tmp_path / "state"
    (home_root / SLOT).mkdir(parents=True)
    state_dir.mkdir()
    (state_dir / "slots.json").write_text(json.dumps({"version": 1, "slots": {SLOT: {
        "matterId": "m-1", "port": 8800, "unixUser": f"m_{SLOT}", "home": str(home_root / SLOT),
        "desired": "stopped"}}}))
    return {"tmp": tmp_path, "home_root": home_root, "state_dir": state_dir, "home": home_root / SLOT}


def make_tar(path, entries, mode="w:gz"):
    """entries: (name, kind, payload): kind file|dir|symlink|hardlink|chardev|fifo."""
    with tarfile.open(path, mode) as tar:
        for name, kind, payload in entries:
            info = tarfile.TarInfo(name)
            data = None
            if kind == "file":
                data = payload.encode()
                info.size = len(data)
            elif kind == "dir":
                info.type, info.mode = tarfile.DIRTYPE, 0o755
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, payload
            elif kind == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, payload
            elif kind == "chardev":
                info.type, info.devmajor, info.devminor = tarfile.CHRTYPE, 1, 3
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


GOOD = [
    ("matter", "dir", None),
    ("matter/inbox", "dir", None),
    ("matter/inbox/complaint.txt", "file", "complaint text"),
    ("matter/deliverables/memo.md", "file", "memo"),
    ("matter/latest", "symlink", "deliverables/memo.md"),
    ("hermes", "dir", None),
    ("hermes/memories/MEMORY.md", "file", "the matter's memory"),
    ("hermes/state.db", "file", "sqlite bytes"),
    ("hermes/state-copy.db", "hardlink", "hermes/state.db"),
]


def run(root, tarball, *, dry_run=False, state="inactive", lines=None):
    out = lines if lines is not None else []
    return imp.run_import(SLOT, tarball, home_root=root["home_root"], state_dir=root["state_dir"], dry_run=dry_run,
                          state_of=lambda slot: state, user_of=lambda user: (os.getuid(), os.getgid()),
                          out=out.append, now=lambda: FIXED_NOW)


def test_import_lands_matter_and_hermes_in_the_slot_home(root):
    tarball = make_tar(root["tmp"] / "m.tar.gz", GOOD)
    lines = []
    plan = run(root, tarball, lines=lines)
    home = root["home"]
    assert (home / "matter" / "inbox" / "complaint.txt").read_text() == "complaint text"
    assert os.readlink(home / "matter" / "latest") == "deliverables/memo.md"
    assert (home / ".hermes" / "memories" / "MEMORY.md").read_text() == "the matter's memory"
    assert (home / ".hermes" / "state-copy.db").read_text() == "sqlite bytes"
    assert sorted(p.name for p in home.iterdir()) == [".hermes", "matter"]  # staging is gone
    assert (plan.counts["matter"].files, plan.counts["matter"].links) == (2, 1)
    assert (plan.counts["hermes"].files, plan.counts["hermes"].links) == (2, 1)
    assert any(line.startswith("importing matter/ -> ") and "2 files" in line for line in lines)
    assert any(line.startswith("chown -R m_") for line in lines)


def test_existing_homes_are_moved_aside_not_merged_or_deleted(root):
    home = root["home"]
    (home / ".hermes").mkdir()
    (home / ".hermes" / "config.yaml").write_text("rendered at the slot's first start")
    (home / "matter").mkdir()
    run(root, make_tar(root["tmp"] / "m.tar.gz", GOOD))
    aside = home / ".hermes.pre-import-20261001T120000Z"
    assert (aside / "config.yaml").read_text() == "rendered at the slot's first start"
    assert not (home / ".hermes" / "config.yaml").exists()
    assert (home / "matter.pre-import-20261001T120000Z").is_dir()


def test_dry_run_reports_and_writes_nothing(root):
    lines = []
    run(root, make_tar(root["tmp"] / "m.tar", GOOD, mode="w"), dry_run=True, lines=lines)
    assert list(root["home"].iterdir()) == []
    assert any(line.startswith("would import hermes/ -> ") for line in lines)
    assert any(line.startswith("would chown -R") for line in lines)


@pytest.mark.parametrize("bad, message", [
    (("/etc/passwd", "file", "x"), "absolute path"),
    (("matter/../../etc/cron.d/x", "file", "x"), "'..'"),
    (("other/file", "file", "x"), "outside matter/ and hermes/"),
    (("..", "dir", None), "'..'"),
    (("matter/dev", "chardev", None), "device node"),
    (("matter/pipe", "fifo", None), "device node or FIFO"),
    (("matter/out", "symlink", "/etc/shadow"), "symbolic link"),
    (("matter/out", "symlink", "../../../../etc/shadow"), "symbolic link"),
    (("matter/a/b/out", "symlink", "../../../x"), "symbolic link"),
    (("matter/cross", "symlink", "../hermes/state.db"), "symbolic link"),
    (("matter/hard", "hardlink", "/etc/shadow"), "hard link"),
    (("matter/hard", "hardlink", "hermes/state.db"), "hard link"),
    (("matter/hard", "hardlink", "matter/../../x"), "hard link"),
])
def test_dangerous_members_refuse_the_whole_tarball(root, bad, message):
    tarball = make_tar(root["tmp"] / "bad.tar.gz", [*GOOD, bad])
    with pytest.raises(imp.ImportError_, match=message):
        run(root, tarball)
    assert list(root["home"].iterdir()) == [], "nothing may be written before every member passes"


def test_a_tarball_missing_either_directory_is_refused(root):
    only_matter = make_tar(root["tmp"] / "m.tar.gz", [m for m in GOOD if m[0].startswith("matter")])
    with pytest.raises(imp.ImportError_, match="no hermes/"):
        run(root, only_matter)


@pytest.mark.parametrize("state", ["active", "activating", "deactivating", "reloading", "unknown"])
def test_a_running_or_unknown_slot_is_refused(root, state):
    with pytest.raises(imp.ImportError_, match=f"slot {SLOT} is {state}"):
        run(root, make_tar(root["tmp"] / "m.tar.gz", GOOD), state=state)


def test_a_failed_unit_counts_as_stopped(root):
    run(root, make_tar(root["tmp"] / "m.tar.gz", GOOD), state="failed")
    assert (root["home"] / "matter").is_dir()


def test_an_unregistered_or_removing_slot_is_refused(root):
    tarball = make_tar(root["tmp"] / "m.tar.gz", GOOD)
    with pytest.raises(imp.ImportError_, match="not registered"):
        imp.run_import("zz99", tarball, home_root=root["home_root"], state_dir=root["state_dir"], dry_run=False,
                       state_of=lambda s: "inactive", user_of=lambda u: (0, 0))
    doc = json.loads((root["state_dir"] / "slots.json").read_text())
    doc["slots"][SLOT]["desired"] = "removing"
    (root["state_dir"] / "slots.json").write_text(json.dumps(doc))
    with pytest.raises(imp.ImportError_, match="being removed"):
        run(root, tarball)


def test_a_home_that_is_a_symlink_is_refused(root):
    home = root["home"]
    home.rmdir()
    elsewhere = root["tmp"] / "elsewhere"
    elsewhere.mkdir()
    home.symlink_to(elsewhere)
    with pytest.raises(imp.ImportError_, match="is not a directory"):
        run(root, make_tar(root["tmp"] / "m.tar.gz", GOOD))
    assert list(elsewhere.iterdir()) == []


def test_bad_slot_id_and_bad_tarball(root):
    with pytest.raises(imp.ImportError_, match="slot id"):
        imp.run_import("../x", root["tmp"] / "none", home_root=root["home_root"], state_dir=root["state_dir"],
                       dry_run=True, state_of=lambda s: "inactive", user_of=lambda u: (0, 0))
    junk = root["tmp"] / "junk.tar"
    junk.write_text("not a tarball")
    with pytest.raises(imp.ImportError_, match="cannot open"):
        run(root, junk)


def test_cli_dry_run_uses_the_scratch_root_overrides(root):
    tarball = make_tar(root["tmp"] / "m.tar.gz", GOOD)
    bin_dir = root["tmp"] / "bin"
    bin_dir.mkdir()
    # The CLI asks systemctl for the unit state; a fake answers "inactive".
    (bin_dir / "systemctl").write_text("#!/bin/sh\necho inactive\nexit 3\n")
    (bin_dir / "systemctl").chmod(0o755)
    env = dict(os.environ, LITCO_HOME_ROOT=str(root["home_root"]), LITCO_STATE_DIR=str(root["state_dir"]))
    code = (f"import sys; sys.path.insert(0, {str(HOST.parents[1])!r})\n"
            "from tests.host._load import load_script\n"
            "m = load_script('imp', 'litco-slot-import')\n"
            f"m.COMMAND_PATH = {str(bin_dir)!r} + ':/usr/bin:/bin'\n"
            "m.lookup_user = lambda name: (0, 0)\n"
            "sys.exit(m.main(sys.argv[1:], user_of=m.lookup_user))\n")
    out = subprocess.run([sys.executable, "-c", code, "--dry-run", SLOT, str(tarball)], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "would import matter/" in out.stdout and list(root["home"].iterdir()) == []
    refused = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "nope", str(tarball)], capture_output=True,
                             text=True, env=env)
    assert refused.returncode == 2 and "not registered" in refused.stderr
