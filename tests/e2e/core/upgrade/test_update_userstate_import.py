"""`hermes import` of a broken backup archive must report failure, never "restored".

The archive is made by the real ``hermes backup`` from a populated home, then damaged the ways
archives really break (a download cut off half-way, a file that is not a zip at all, one member
whose bytes rotted, a target directory the importer cannot write). Each import runs as a fresh
sandboxed CLI over a home that already holds the user's current config; the contract is the
exit code and wording a script or the dashboard keys on, and that a failed import leaves the
existing home untouched.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
]

PY = sys.executable
RESTORED = "has been restored"
CURRENT_CONFIG = "# the user's CURRENT config, which a failed import must not touch\nmodel:\n  default: current-model\n"
SKILL = "---\nname: demo\ndescription: demo skill from the backup\n---\n" + ("body line\n" * 400)


class Gap(Exception):
    """Contract breach (not an AssertionError, so a crash reads differently from a wrong answer)."""


def _home(root: Path, name: str) -> dict:
    env = H.isolated_env(root / name, pythonpath=H.WORKTREE)
    (root / name / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(root / name / "tmp")
    return env


def _cli(root: Path, env: dict, *args: str):
    return H.run([PY, "-m", "hermes_cli.main", *args], env=env, cwd=root, writable=[root], timeout=300)


@pytest.fixture(scope="module")
def archive(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("import")
    env = _home(root, "src")
    hh = Path(env["HERMES_HOME"])
    (hh / "config.yaml").write_text("# from the backup\nmodel:\n  default: backup-model\n", encoding="utf-8")
    (hh / "SOUL.md").write_text("soul from the backup\n", encoding="utf-8")
    (hh / "skills" / "demo").mkdir(parents=True)
    (hh / "skills" / "demo" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    out = root / "backup.zip"
    cp = _cli(root, env, "backup", "-o", str(out))
    assert cp.returncode == 0 and out.exists(), H.describe(cp)
    assert {"config.yaml", "SOUL.md", "skills/demo/SKILL.md"} <= set(zipfile.ZipFile(out).namelist())
    return root, out


def _truncated(src: Path, dst: Path) -> None:
    data = src.read_bytes()
    dst.write_bytes(data[: len(data) // 2])


def _not_a_zip(src: Path, dst: Path) -> None:
    dst.write_bytes(b"<html>502 Bad Gateway</html>\n" * 40)


def _rotten_member(src: Path, dst: Path) -> None:
    """Flip bytes inside skills/demo/SKILL.md's compressed data: the central directory still
    lists every member, but that one fails its CRC on read."""
    shutil.copy(src, dst)
    with zipfile.ZipFile(dst) as zf:
        info = zf.getinfo("skills/demo/SKILL.md")
    raw = bytearray(dst.read_bytes())
    # local header: 30 fixed bytes + name + extra, then the compressed payload
    name_len = int.from_bytes(raw[info.header_offset + 26: info.header_offset + 28], "little")
    extra_len = int.from_bytes(raw[info.header_offset + 28: info.header_offset + 30], "little")
    start = info.header_offset + 30 + name_len + extra_len
    for i in range(start + 4, start + min(info.compress_size, 64) - 4):
        raw[i] ^= 0x5A
    dst.write_bytes(bytes(raw))


BROKEN = {"truncated": _truncated, "not-a-zip": _not_a_zip, "rotten-member": _rotten_member}


@pytest.mark.parametrize("kind", list(BROKEN))
def test_import_of_a_broken_archive_fails_and_leaves_the_home_alone(archive, kind):
    root, good = archive
    bad = root / f"{kind}.zip"
    BROKEN[kind](good, bad)
    env = _home(root, f"dst-{kind}")
    hh = Path(env["HERMES_HOME"])
    (hh / "config.yaml").write_text(CURRENT_CONFIG, encoding="utf-8")
    cp = _cli(root, env, "import", str(bad), "--force")
    out = cp.stdout + cp.stderr
    if I.TRACEBACK in out:
        raise Gap(f"import of a {kind} archive crashed instead of reporting it:\n" + H.describe(cp))
    assert cp.returncode != 0, f"import of a {kind} archive exited 0:\n" + H.describe(cp)
    assert RESTORED not in out, f"import of a {kind} archive claims success:\n" + H.describe(cp)
    if kind != "rotten-member":
        # Nothing readable in the archive: nothing may change.
        assert (hh / "config.yaml").read_text(encoding="utf-8") == CURRENT_CONFIG
    skill = hh / "skills" / "demo" / "SKILL.md"
    assert not skill.exists() or skill.read_text(encoding="utf-8") == SKILL, "a corrupt member was written as the skill"


def test_import_that_skips_members_reports_incomplete(archive):
    root, good = archive
    env = _home(root, "dst-partial")
    hh = Path(env["HERMES_HOME"])
    locked = hh / "skills" / "demo"
    locked.mkdir(parents=True)
    locked.chmod(0o555)  # left behind root-owned by an earlier `sudo hermes ...`
    try:
        cp = _cli(root, env, "import", str(good), "--force")
    finally:
        locked.chmod(0o755)
    out = cp.stdout + cp.stderr
    assert I.TRACEBACK not in out, H.describe(cp)
    assert (hh / "config.yaml").read_text(encoding="utf-8").startswith("# from the backup"), "readable members not restored"
    assert not (locked / "SKILL.md").exists(), "harness: the locked directory was writable"
    if cp.returncode == 0 or RESTORED in out:
        raise Gap("import skipped skills/demo/SKILL.md yet reported success:\n" + H.describe(cp))
