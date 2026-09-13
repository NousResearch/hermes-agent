"""Behavior contracts for pm's bionic target + DebPackage + stage_only.

The lock VALUES are bumped by review; these tests pin the relationships:
the linux-arm64-bionic rows exist and agree with their suppliers' shapes,
DebPackage extraction is hardened, and cross-target staging never touches
this host's installed facts.
"""

from __future__ import annotations

import json
import tarfile
import io
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pm():
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    import pm

    return pm


@pytest.fixture(scope="module")
def lock():
    return json.loads((REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))


def test_all_targets_includes_bionic():
    from pm.store import ALL_TARGETS

    assert "linux-arm64-bionic" in ALL_TARGETS
    assert ALL_TARGETS.count("linux-arm64-bionic") == 1


def _assert_pinned_bionic_row(lock, pkg, url_suffix_re):
    """The bionic rows are DELIBERATE explicit pins: the version axis follows
    main (the desktop artifacts), while the termux/TUR suppliers rotate or
    lag, so the bionic row pins whatever the supplier actually ships today.
    The stage path consumes the row's url+sha256 directly. The contract:
    the row exists, is https, matches the supplier's filename shape, and
    fetch_url AGREES with the pinned row for this target."""
    import re as _re

    row = lock["packages"][pkg]["artifacts"].get("linux-arm64-bionic")
    assert row, f"{pkg} has no linux-arm64-bionic artifact"
    assert row["url"].startswith("https://"), f"{pkg} bionic row not https"
    m = _re.search(url_suffix_re, row["url"])
    assert m, f"{pkg} bionic url shape: {row['url']}"
    assert _re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    from pm.registry import get_package

    cls = get_package(pkg)
    # fetch_url must reproduce the pinned row exactly for this target
    # (the pinned version is recovered from the URL itself).
    assert cls.fetch_url(m.group("ver"), "linux-arm64-bionic") == row["url"]


def test_python_bionic_row_matches_supplier(lock):
    """The Python package definition and lock must agree on the termux-main artifact."""
    _assert_pinned_bionic_row(lock, "python", r"/p/python/python_(?P<ver>[0-9.]+(?:-[0-9]+)?)_aarch64\.deb$")
def test_node_bionic_row_matches_supplier(lock):
    """The node bionic row is an explicit pin of the termux-main nodejs .deb;
    the row and Nodejs.fetch_url(bionic arm) must agree."""
    _assert_pinned_bionic_row(lock, "node", r"nodejs_(?P<ver>[0-9.]+)-1_aarch64\.deb$")
def test_termux_docker_row_pins_digest(lock):
    row = lock["packages"]["termux-docker"]["artifacts"]["linux-arm64-bionic"]
    version = lock["packages"]["termux-docker"]["version"]
    assert version.startswith("sha256:")
    assert len(version) == 7 + 64
    assert row["url"] == f"docker://termux/termux-docker@{version}"


def test_python_bionic_pin_is_independent_of_desktop_build_version(lock):
    from pm.registry import get_package

    py = get_package("python")
    package = lock["packages"]["python"]
    assert py.fetch_url(package["version"], "linux-arm64-bionic") == package["artifacts"]["linux-arm64-bionic"]["url"]
    assert py.deb_package == "python"


def test_uv_bionic_row_matches_supplier(lock):
    """The uv bionic row is an explicit pin of the termux-main pool .deb;
    the row and Uv.fetch_url(bionic arm) must agree."""
    _assert_pinned_bionic_row(lock, "uv", r"/u/uv/uv_(?P<ver>[0-9.]+)_aarch64\.deb$")
def _build_fake_deb(path: Path, control: dict[str, str], files: dict[str, bytes]) -> None:
    def ar_member(name: str, data: bytes) -> bytes:
        hdr = (
            name.ljust(16).encode()
            + b"0".ljust(12)
            + b"0".ljust(6)
            + b"0".ljust(6)
            + b"100644".ljust(8)
            + str(len(data)).encode().ljust(10)
            + b"`\n"
        )
        pad = b"\n" if len(data) % 2 else b""
        return hdr + data + pad

    ctrl_buf = io.BytesIO()
    with tarfile.open(fileobj=ctrl_buf, mode="w:gz") as tf:
        body = "".join(f"{k}: {v}\n" for k, v in control.items()).encode()
        info = tarfile.TarInfo("control")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
    data_buf = io.BytesIO()
    with tarfile.open(fileobj=data_buf, mode="w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    path.write_bytes(
        b"!<arch>\n"
        + ar_member("debian-binary", b"2.0\n")
        + ar_member("control.tar.gz", ctrl_buf.getvalue())
        + ar_member("data.tar.gz", data_buf.getvalue())
    )


def test_debpackage_unpack_hardened(tmp_path: Path):
    """DebPackage.unpack extracts data members and refuses traversal."""
    from pm.package import DebPackage

    class _P(DebPackage):
        name = "test-deb"

    deb = tmp_path / "test.deb"
    _build_fake_deb(
        deb,
        {"Package": "test-deb", "Version": "1.0"},
        {"data/data/com.termux/files/usr/bin/tool": b"\x7fELF"},
    )
    staged = tmp_path / "staged"
    staged.mkdir()
    _P().unpack(deb, staged, "linux-arm64-bionic")
    assert (staged / "data/data/com.termux/files/usr/bin/tool").read_bytes() == b"\x7fELF"

    # traversal member must be refused
    evil = tmp_path / "evil.deb"
    _build_fake_deb(
        evil, {"Package": "evil", "Version": "1.0"}, {"../escape": b"x"}
    )
    with pytest.raises(Exception):
        _P().unpack(evil, tmp_path / "staged2", "linux-arm64-bionic")


def test_python_bionic_verify_is_file_evidence(tmp_path: Path):
    """bionic verify never executes the staged binary; presence is the
    contract (the digest already proved the bytes)."""
    from pm.registry import get_package

    py = get_package("python")
    bin_rel = Path(py.prefix_rel) / py.main_rel("linux-arm64-bionic")
    entry = tmp_path / "entry"
    (entry / bin_rel).parent.mkdir(parents=True)
    (entry / bin_rel).write_bytes(b"bionic-elf-bytes")
    assert py.verify(entry, "linux-arm64-bionic") == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "missing" in py.verify(empty, "linux-arm64-bionic")


def test_bionic_binary_and_env_contract(tmp_path: Path):
    """On bionic, _BionicDebArm.binary() must return the staged deb's main
    binary path (file evidence, no exec), so the base Package.env contract
    exposes the tool through PATH like every other pm package."""
    from pm.registry import get_package

    for name in ("uv", "python", "node"):
        pkg = get_package(name)
        entry = tmp_path / name
        main = entry / pkg.prefix_rel / pkg.main_rel("linux-arm64-bionic")
        main.parent.mkdir(parents=True)
        main.write_bytes(b"bionic-elf")

        binary = pkg.binary(entry, "linux-arm64-bionic")
        assert binary == main, f"{name}.binary() on bionic: {binary}"

        env = pkg.env(entry, "linux-arm64-bionic")
        assert env.get("PATH") == [str(main.parent)], (
            f"{name}.env() on bionic does not follow the Package.env PATH contract"
        )


def test_stage_only_does_not_record_host_facts(tmp_path, monkeypatch):
    """stage_only publishes the entry but must not touch this machine's
    installed facts -- the fact slot belongs to the HOST target."""
    pm = _pm()
    from pm.ensure import stage_only
    from pm.lock import Facts
    from pm.paths import facts_path

    def snapshot() -> dict:
        path = facts_path()
        if not path.is_file():
            return {}
        return Facts(path)._packages

    before = snapshot()
    entry = stage_only("termux-docker", "linux-arm64-bionic")
    after = snapshot()
    assert before == after
    # termux-docker is a pin_only package: stage_only returns the would-be
    # entry path (store root + entry name) without staging bytes.
    assert "termux-docker" in str(entry)
