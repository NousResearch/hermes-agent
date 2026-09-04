"""Tests: check-updates — address resolution, saved-tag security, feeds.

All network seams injected (fetch, ls-remote, PyPI probe) — hermetic.
The local bare-repo fixture exercises the REAL git ls-remote command
shape without network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli.plugins_provenance import Provenance, ProvenanceClass
from hermes_cli.plugins_updates import (
    CheckResult,
    check_pip_plugins,
    check_provenanced,
    parse_feed_yml,
    run_checks,
)


def _prov(
    klass=ProvenanceClass.GIT,
    row=None,
    path=None,
):
    return Provenance(
        name="plug",
        klass=klass,
        path=path or Path("/x/plug"),
        row=row or {},
    )


def _git_prov(**row):
    row.setdefault("pinned", False)
    row.setdefault("revision", "a" * 40)
    row.setdefault("source", "https://example/o/r")
    return _prov(ProvenanceClass.GIT, row=row)


# ── provenance-class short-circuits ────────────────────────────────


def test_manual_reports_not_updatable():
    r = check_provenanced(_prov(ProvenanceClass.MANUAL), fetch=_no, ls_remote=_no)
    assert r.update_available is None
    assert "not auto-updatable" in r.reason


def test_drift_reports_reinstall_remedy():
    r = check_provenanced(
        _prov(ProvenanceClass.DRIFT, row={"source": "https://x/y"}),
        fetch=_no, ls_remote=_no,
    )
    assert "reinstall" in r.reason


def test_self_cloned_reports_adopt():
    r = check_provenanced(_prov(ProvenanceClass.SELF_CLONED), fetch=_no, ls_remote=_no)
    assert "adopt" in r.reason


def _no(*a, **k):
    raise AssertionError("should not be called")


# ── pinned ──────────────────────────────────────────────────────────


def test_pinned_never_auto_moves():
    r = check_provenanced(_git_prov(pinned=True), fetch=_no, ls_remote=_no)
    assert r.update_available is False
    assert "pinned" in r.reason


# ── the saved-tag security heart ────────────────────────────────────


def test_url_appearing_where_none_saved_is_needs_fixing(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://evil.example/feed.yml\n",
        encoding="utf-8",
    )
    prov = _git_prov()  # no update_url in the row
    prov.path = plug
    r = check_provenanced(prov, fetch=_no, ls_remote=_no)
    assert r.needs_fixing and "trust-update-url" in r.needs_fixing
    assert r.update_available is None  # refused, not unknown


def test_url_mismatch_is_needs_fixing(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://new.example/feed.yml\n",
        encoding="utf-8",
    )
    prov = _git_prov(update_url="https://old.example/feed.yml")
    prov.path = plug
    r = check_provenanced(prov, fetch=_no, ls_remote=_no)
    assert "mismatch" in r.needs_fixing
    assert "trust-update-url" in r.needs_fixing


# ── feed path (matching tag) ────────────────────────────────────────


FEED = """\
version: 1.2.0
released: 2026-09-03T00:00:00Z
min_hermes: 0.27.0
artifacts:
  git: https://example/o/r
  bundle: https://example/o/r/plug-1.2.0.zip
  bundle_sha256: abc123
"""


def test_matching_tag_fetches_feed(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://feed.example/f.yml\n", encoding="utf-8"
    )
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug

    fetched = []

    def fetch(url):
        fetched.append(url)
        return FEED

    r = check_provenanced(prov, fetch=fetch, ls_remote=_no)
    assert fetched == ["https://feed.example/f.yml"]
    assert r.latest == "1.2.0"
    assert r.min_hermes == "0.27.0"
    assert r.update_available is True  # revision sha != 1.2.0


def test_feed_version_equal_to_current_means_no_update(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://feed.example/f.yml\n", encoding="utf-8"
    )
    prov = _git_prov(update_url="https://feed.example/f.yml", revision="1.2.0")
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: FEED, ls_remote=_no)
    assert r.update_available is False


def test_feed_fetch_failure_is_row_level_reason(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://feed.example/f.yml\n", encoding="utf-8"
    )
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug

    def boom(url):
        raise OSError("timeout")

    r = check_provenanced(prov, fetch=boom, ls_remote=_no)
    assert r.reason.startswith("feed fetch failed")
    assert r.update_available is None


# ── ls-remote fallback (no update_url anywhere) ─────────────────────


def test_ls_remote_fallback(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    prov = _git_prov()  # no update_url in row or manifest
    prov.path = plug
    calls = []

    def ls(source):
        calls.append(source)
        return "b" * 40

    r = check_provenanced(prov, fetch=_no, ls_remote=ls)
    assert calls == ["https://example/o/r"]
    assert r.update_available is True  # b != a


# ── the real git ls-remote, against a local bare repo ───────────────


def test_real_ls_remote_against_bare_repo(tmp_path):
    import os
    import shutil

    # The host PATH may resolve git to the MSIX payload (package-boundary
    # spawn denial, WinError 5) — prefer a conventional install.
    git_bin = shutil.which("git") or "git"
    for candidate in (
        "C:/Program Files/Git/cmd/git.exe",
        "/usr/bin/git",
    ):
        if Path(candidate).is_file():
            git_bin = candidate
            break

    source = tmp_path / "bare.git"
    env = {k: v for k, v in os.environ.items() if k != "GIT_DIR"}
    subprocess.run(
        [git_bin, "init", "--bare", "-q", str(source)],
        check=True, capture_output=True, env=env,
    )
    # ls-remote on an empty bare repo: exit 0, empty HEAD — the command
    # SHAPE works; empty maps to unknown, never a crash
    proc = subprocess.run(
        [git_bin, "ls-remote", str(source), "HEAD"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


# ── feed parsing ────────────────────────────────────────────────────


def test_parse_feed_requires_version():
    with pytest.raises(ValueError):
        parse_feed_yml("released: 2026-01-01\n")
    assert parse_feed_yml("version: 2.0.0\n")["version"] == "2.0.0"


# ── pip world ───────────────────────────────────────────────────────


class _EP:
    def __init__(self, name, value, dist_name):
        self.name = name
        self.value = value
        self.dist_name = dist_name


def test_pip_check_stateless():
    eps = [_EP("mnemosyne", "mnemosyne_hermes:register", "mnemosyne-hermes")]
    rs = check_pip_plugins(
        installed_version=lambda d: "0.5.0",
        pypi_latest=lambda d: "0.6.0",
        entry_points=eps,
    )
    assert rs[0].klass == "pip"
    assert rs[0].current == "0.5.0"
    assert rs[0].latest == "0.6.0"
    assert rs[0].update_available is True


def test_pip_not_on_pypi_reports_unknown():
    eps = [_EP("local-only", "x:y", "x")]
    rs = check_pip_plugins(
        installed_version=lambda d: "1.0",
        pypi_latest=lambda d: None,
        entry_points=eps,
    )
    assert rs[0].update_available is None
    assert "not on PyPI" in rs[0].reason


# ── run_checks composition ───────────────────────────────────────────


def test_run_checks_never_mutates(tmp_path):
    plugins = tmp_path / "plugins"
    plug = plugins / "plug"
    (plug / ".git").mkdir(parents=True)  # git-class, not drift
    (plugins / ".install-metadata.json").write_text(
        json.dumps({"plug": {"pinned": False, "revision": "a" * 40,
                             "source": "https://example/o/r"}}),
        encoding="utf-8",
    )
    before = (plugins / ".install-metadata.json").read_text(encoding="utf-8")

    results = run_checks(
        plugins,
        fetch=_no,
        ls_remote=lambda s: "b" * 40,
        include_pip=False,
    )
    assert results[0].update_available is True
    assert (plugins / ".install-metadata.json").read_text(encoding="utf-8") == before