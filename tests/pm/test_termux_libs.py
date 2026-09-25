"""The termux pool pin authority: parse the index, retire, repin.

No network and no payload: the index is text, the pool rows are data, and the
one step that would touch bytes (hashing the replacement archive) is injected.

`--termux` exists because the pool is rolling — rebuilding a package DELETES
the previous archive — so the invariant every case here pins is *only retired
rows move, and only to the archive the index describes*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pm import cli, termux_libs
from pm.lock import SCHEMA, Lockfile

POOL = termux_libs.POOL
ICONV = f"{POOL}pool/main/libi/libiconv/libiconv_1.19_aarch64.deb"
ICONV_OLD = f"{POOL}pool/main/libi/libiconv/libiconv_1.18-1_aarch64.deb"
GLIB = f"{POOL}pool/main/g/glib/glib_2.90.0_aarch64.deb"
LICENSES = f"{POOL}pool/main/t/termux-licenses/termux-licenses_2.2_all.deb"
UV_OLD = f"{POOL}pool/main/u/uv/uv_0.12.10_aarch64.deb"
UV_NEW = f"{POOL}pool/main/u/uv/uv_0.12.15_aarch64.deb"

INDEX = (
    "Package: libiconv\n"
    "Version: 1.19\n"
    "Architecture: aarch64\n"
    f"Filename: {ICONV[len(POOL):]}\n"
    "Size: 562724\n"
    f"SHA256: {'i' * 64}\n"
    "Description: An implementation of iconv()\n"
    " Depends: libc\n"
    " This continuation line is not a stanza.\n"
    "\n"
    "Package: glib\n"
    "Version: 2.90.0\n"
    f"Filename: {GLIB[len(POOL):]}\n"
    f"SHA256: {'g' * 64}\n"
)


def _lockfile(tmp_path: Path, packages: dict) -> Lockfile:
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"schema": SCHEMA, "packages": packages}), encoding="utf-8")
    return Lockfile(path)


def _uv(bionic_url: str) -> dict:
    """uv's two axes: astral's tarball on the desktop, the pool's .deb on bionic."""
    return {
        "version": "0.12.3",
        "artifacts": {
            "darwin-x64": {"url": "https://github.com/astral-sh/uv/releases/download/0.12.3/uv.tar.gz",
                           "sha256": "d" * 64},
            "linux-arm64-bionic": {"url": bionic_url, "sha256": "o" * 64},
        },
    }


def _table(libiconv_url: str = ICONV_OLD) -> dict:
    return {
        "licenses": {"sha256": "l" * 64, "url": LICENSES, "version": "2.2"},
        "libs": {
            "glib": {"sha256": "g" * 64, "url": GLIB, "version": "2.90.0"},
            "libiconv": {"sha256": "o" * 64, "url": libiconv_url, "version": "1.18-1"},
        },
    }


def _pool(**rows: termux_libs.PoolPackage) -> dict:
    """The pool the faked index reports, keyed by package."""
    return {
        "libiconv": termux_libs.PoolPackage("libiconv", "1.19", ICONV, "i" * 64),
        "glib": termux_libs.PoolPackage("glib", "2.90.0", GLIB, "g" * 64),
        "termux-licenses": termux_libs.PoolPackage("termux-licenses", "2.2", LICENSES, "l" * 64),
        **rows,
    }


# ------------------------------------------------------------------ the index

def test_parse_index_reads_stanzas_and_ignores_continuations():
    pool = termux_libs.parse_index(INDEX)
    assert set(pool) == {"libiconv", "glib"}
    assert pool["libiconv"] == termux_libs.PoolPackage("libiconv", "1.19", ICONV, "i" * 64)
    # The indented "Depends: libc" continuation must not become a field, and
    # must not split the stanza it belongs to.
    assert pool["glib"].sha256 == "g" * 64


def test_parse_index_keeps_the_pool_epoch_and_drops_rows_without_a_hash():
    text = (
        f"Package: libvpx\nVersion: 1:1.17.0\nFilename: pool/main/libv/libvpx/libvpx_1:1.17.0_aarch64.deb\n"
        f"SHA256: {'v' * 64}\n\n"
        "Package: half\nVersion: 1.0\nFilename: pool/main/h/half/half_1.0_aarch64.deb\n\n"
    )
    pool = termux_libs.parse_index(text)
    assert pool["libvpx"].version == "1:1.17.0"
    assert pool["libvpx"].url.endswith("libvpx_1:1.17.0_aarch64.deb")
    assert "half" not in pool


@pytest.mark.parametrize("url, expected", [
    (GLIB, "glib"),
    (LICENSES, "termux-licenses"),
    ("https://github.com/astral-sh/uv/releases/download/0.12.3/uv.tar.gz", None),
    ("docker://termux/termux-docker@sha256:" + "a" * 64, None),
    (f"{POOL}dists/stable/main/binary-aarch64/Packages", None),
    (f"{POOL}pool/main/libi/libiconv/deeper/thing.deb", None),
])
def test_package_of_recognizes_only_pool_archives(url, expected):
    assert termux_libs.package_of(url) == expected


# ---------------------------------------------------------------------- pins

def test_pins_span_the_table_and_the_pool_lock_rows(tmp_path):
    lockfile = _lockfile(tmp_path, {"uv": _uv(UV_OLD)})
    names = {(pin.scope, pin.name) for pin in termux_libs.pins(_table(), lockfile)}
    assert names == {
        ("library", "glib"),
        ("library", "libiconv"),
        ("license", "termux-licenses"),
        ("tool", "uv@linux-arm64-bionic"),
    }


def test_pins_ignore_lock_sources_the_pool_does_not_serve(tmp_path):
    """A GitHub tarball or an OCI digest is not the pool's to repair."""
    lockfile = _lockfile(tmp_path, {
        "uv": _uv(UV_OLD),
        "chromium": {"version": "145", "artifacts": {"darwin-x64": {
            "url": "https://cdn.playwright.dev/builds/cft/145.0.7632.6/mac-x64/chrome-mac-x64.zip",
            "sha256": "c" * 64}}},
    })
    assert [pin.name for pin in termux_libs.pins({"libs": {}}, lockfile)] == ["uv@linux-arm64-bionic"]


# ----------------------------------------------------------------- retirement

def test_retirement_is_decided_by_the_filename_not_the_version_string(tmp_path):
    """Termux version strings carry epochs and -N rebuild suffixes, and the
    pin table's own version field can lag what the pool now reports. The
    archive filename is the evidence that the pool still serves the pin."""
    lockfile = _lockfile(tmp_path, {"uv": _uv(UV_OLD)})
    table = _table(ICONV)
    table["libs"]["libiconv"]["version"] = "1.18-1"
    table["libs"]["glib"]["version"] = "2.90.0-1"
    pool = _pool(uv=termux_libs.PoolPackage("uv", "0.12.10-1", UV_OLD, "o" * 64))
    assert termux_libs.retired(table, lockfile, pool) == []


def test_retired_reports_the_replacement_and_a_dropped_package(tmp_path):
    lockfile = _lockfile(tmp_path, {"uv": _uv(f"{POOL}pool/main/u/uv/uv_0.12.9_aarch64.deb")})
    table = _table()
    table["libs"]["nolonger"] = {"sha256": "n" * 64, "version": "1",
                                 "url": f"{POOL}pool/main/n/nolonger/nolonger_1_aarch64.deb"}
    pool = _pool(uv=termux_libs.PoolPackage("uv", "0.12.15", UV_NEW, "u" * 64))
    stale = {entry.pin.name: entry for entry in termux_libs.retired(table, lockfile, pool)}
    assert set(stale) == {"libiconv", "uv@linux-arm64-bionic", "nolonger"}
    assert stale["libiconv"].replacement.url == ICONV
    assert stale["uv@linux-arm64-bionic"].replacement.url == UV_NEW
    assert stale["nolonger"].replacement is None  # dropped, not renamed


# --------------------------------------------------------------------- repair

def test_repair_repoints_only_retired_rows(tmp_path):
    lockfile = _lockfile(tmp_path, {"uv": _uv(UV_OLD)})
    table = _table()
    served = {ICONV: "i" * 64, UV_NEW: "u" * 64}
    stale = termux_libs.retired(table, lockfile, _pool(uv=termux_libs.PoolPackage("uv", "0.12.15", UV_NEW, "u" * 64)))

    assert termux_libs.repair(table, lockfile, stale, verify=served.__getitem__) == 2
    assert table["libs"]["libiconv"] == {"sha256": "i" * 64, "url": ICONV, "version": "1.19"}
    assert table["libs"]["glib"] == {"sha256": "g" * 64, "url": GLIB, "version": "2.90.0"}
    assert table["licenses"] == {"sha256": "l" * 64, "url": LICENSES, "version": "2.2"}
    row = lockfile.artifacts("uv", "linux-arm64-bionic")[0]
    assert (row["url"], row["sha256"]) == (UV_NEW, "u" * 64)
    # uv's desktop axis is a different supplier and stays exactly as pinned.
    assert lockfile.artifacts("uv", "darwin-x64")[0]["url"].startswith("https://github.com/")


def test_repair_refuses_bytes_the_index_does_not_describe(tmp_path):
    lockfile = _lockfile(tmp_path, {"uv": _uv(UV_OLD)})
    table = _table()
    before = json.dumps(table, sort_keys=True)
    stale = termux_libs.retired(table, lockfile, _pool(uv=termux_libs.PoolPackage("uv", "0.12.15", UV_NEW, "u" * 64)))

    with pytest.raises(termux_libs.PinMismatch):
        termux_libs.repair(table, lockfile, stale, verify=lambda url: "z" * 64)
    assert json.dumps(table, sort_keys=True) == before
    assert lockfile.artifacts("uv", "linux-arm64-bionic")[0]["url"] == UV_OLD


# ---------------------------------------------------------------- the command

@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A retired library and a retired bionic tool row, with the pool faked."""
    table_path = tmp_path / "termux_runtime_libs.json"
    termux_libs.save_table(_table(), table_path)
    lockfile = _lockfile(tmp_path, {"uv": _uv(UV_OLD)})
    monkeypatch.setattr(termux_libs, "table_path", lambda: table_path)
    monkeypatch.setattr(cli, "_lockfile", lambda: lockfile)
    monkeypatch.setattr(termux_libs, "index",
                        lambda: _pool(uv=termux_libs.PoolPackage("uv", "0.12.15", UV_NEW, "u" * 64)))
    # The only bytes the run touches: the replacements, served with the hash
    # the index declares for them.
    monkeypatch.setattr(termux_libs, "hash_url", {ICONV: "i" * 64, UV_NEW: "u" * 64}.__getitem__)
    return table_path, lockfile


def _args(**over):
    base = dict(names=[], check=False, target=None, uv=False, npm=False, termux=True)
    base.update(over)
    return argparse.Namespace(**base)


def test_termux_pass_check_reports_without_writing(capsys, deployment):
    table_path, _ = deployment
    before = table_path.read_bytes()
    assert cli._termux_pass(check=True) == 1
    out = capsys.readouterr().out
    assert "libiconv" in out and "1.19" in out
    assert "uv@linux-arm64-bionic" in out and "0.12.15" in out
    assert table_path.read_bytes() == before


def test_termux_pass_repins_the_retired_rows_and_saves(deployment):
    table_path, lockfile = deployment
    assert cli._termux_pass(check=False) == 0
    written = json.loads(table_path.read_text(encoding="utf-8"))
    assert written["libs"]["libiconv"] == {"sha256": "i" * 64, "url": ICONV, "version": "1.19"}
    assert written["libs"]["glib"] == _table()["libs"]["glib"]
    assert lockfile.artifacts("uv", "linux-arm64-bionic")[0]["url"] == UV_NEW


def test_termux_pass_reports_a_dropped_package_as_a_failure(tmp_path, monkeypatch, capsys):
    table_path = tmp_path / "termux_runtime_libs.json"
    row = {"sha256": "n" * 64, "version": "1", "url": f"{POOL}pool/main/g/gone/gone_1_aarch64.deb"}
    termux_libs.save_table({"libs": {"gone": row}}, table_path)
    monkeypatch.setattr(termux_libs, "table_path", lambda: table_path)
    monkeypatch.setattr(cli, "_lockfile", lambda: _lockfile(tmp_path, {}))
    monkeypatch.setattr(termux_libs, "index", lambda: {})
    assert cli._termux_pass(check=False) == 1
    assert "no longer carries it" in capsys.readouterr().out
    assert json.loads(table_path.read_text(encoding="utf-8"))["libs"]["gone"] == row


def test_update_command_routes_termux_to_its_own_pass(monkeypatch, capsys):
    """--termux is a repair pass, so the resolve/install path must not run."""
    seen = []

    def fake_pass(*, check):
        seen.append(check)
        return 0

    monkeypatch.setattr(cli, "_termux_pass", fake_pass)
    monkeypatch.setattr(cli, "resolve_package", lambda *a, **k: pytest.fail("resolved a package"))
    assert cli.cmd_update(_args(check=True, names=["node"])) == 0
    assert seen == [True]
    assert "ignoring names" in capsys.readouterr().out


def test_save_table_round_trips_with_lf_and_keeps_row_order(tmp_path):
    path = tmp_path / "table.json"
    table = _table()
    termux_libs.save_table(table, path)
    assert b"\r\n" not in path.read_bytes()
    assert termux_libs.load_table(path) == table
    assert list(termux_libs.load_table(path)) == ["licenses", "libs"]
