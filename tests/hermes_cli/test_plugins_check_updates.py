"""Tests: check-updates — address resolution, saved-tag security, feeds.

All network seams injected (fetch, ls-remote, PyPI probe) — hermetic.
The local bare-repo fixture exercises the REAL git ls-remote command
shape without network.
"""

from __future__ import annotations

import json
import os
import shutil
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


# ── local git, one resolver for every repo-spawning test ────────────


def _git_exe():
    """Conventional git from PATH — tests run with Git for Windows' cmd
    dir leading PATH, never the MSIX payload (spawn denial, WinError 5)."""
    return shutil.which("git") or "git"


def _git_env():
    return {k: v for k, v in os.environ.items() if k != "GIT_DIR"}


def _local_repo(path: Path, *init_args):
    """A real local git repo; returns (run, head) bound to it."""
    git = _git_exe()
    env = _git_env()

    def run(*args):
        subprocess.run(
            [git, *args], cwd=path, check=True, capture_output=True, env=env
        )

    def head():
        return subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=path, check=True,
            capture_output=True, text=True, env=env,
        ).stdout.strip()

    path.mkdir(parents=True)
    run("init", "-q", *init_args)
    run("-c", "user.email=t@e", "-c", "user.name=t",
        "commit", "--allow-empty", "-qm", "one")
    return run, head


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
        "name: plug\nversion: 1.0.0\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
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
    assert r.update_available is True  # installed 1.0.0 vs feed 1.2.0


def test_feed_version_equal_to_installed_version_means_no_update(tmp_path):
    """C17 convergence: feed semantic version vs installed semantic version."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nversion: 1.2.0\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
    )
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: FEED, ls_remote=_no)
    assert r.update_available is False


def test_feed_version_with_no_installed_version_is_unknown(tmp_path):
    """A semantic feed vs a versionless manifest cannot be compared — report
    unknown, never a revision-vs-version inequality."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://feed.example/f.yml\n", encoding="utf-8"
    )
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: FEED, ls_remote=_no)
    assert r.update_available is None
    assert r.latest == "1.2.0"


def test_feed_git_sha_artifact_compares_sha_to_sha(tmp_path):
    """C17 like-for-like: a full git SHA in the feed's artifacts.git is
    compared against the recorded revision, never to a semantic version —
    and both reported fields are the shas actually compared."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nversion: 9.9.9\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
    )
    feed = "version: 1.2.0\nartifacts:\n  git: %s\n" % ("c" * 40)
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug

    r = check_provenanced(prov, fetch=lambda u: feed, ls_remote=_no)
    assert r.update_available is True  # c*40 != a*40
    assert r.current == "a" * 40
    assert r.latest == "c" * 40

    prov2 = _git_prov(update_url="https://feed.example/f.yml", revision="c" * 40)
    prov2.path = plug
    r2 = check_provenanced(prov2, fetch=lambda u: feed, ls_remote=_no)
    assert r2.update_available is False  # SHA vs SHA: converged
    assert r2.current == "c" * 40
    assert r2.latest == "c" * 40


def test_feed_git_sha_case_equivalent_is_not_an_update(tmp_path):
    """Hex identity is case-insensitive — but only after format validation."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nupdate_url: https://feed.example/f.yml\n", encoding="utf-8"
    )
    feed = "version: 1.2.0\nartifacts:\n  git: %s\n" % ("C" * 40)
    prov = _git_prov(update_url="https://feed.example/f.yml", revision="c" * 40)
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: feed, ls_remote=_no)
    assert r.update_available is False


def test_feed_git_tag_artifact_falls_to_semantic_compare(tmp_path):
    """A non-sha artifacts.git (repo URL, tag) is not a comparable git
    identity — the feed's semantic version is compared against the
    installed manifest's version instead."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nversion: 1.0.0\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
    )
    feed = "version: 9.9.9\nartifacts:\n  git: https://example/o/r\n"
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: feed, ls_remote=_no)
    assert r.update_available is True
    assert r.current == "1.0.0"
    assert r.latest == "9.9.9"


def test_feed_git_sha_with_no_recorded_revision_is_unknown(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nversion: 9.9.9\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
    )
    feed = "version: 1.2.0\nartifacts:\n  git: %s\n" % ("c" * 40)
    prov = _git_prov(update_url="https://feed.example/f.yml", revision="")
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: feed, ls_remote=_no)
    assert r.update_available is None
    assert "no full revision sha" in r.reason


def test_feed_git_sha_against_tagged_revision_is_unknown(tmp_path):
    """A recorded tag is not a full sha — never compared by inequality."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nversion: 9.9.9\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
    )
    feed = "version: 1.2.0\nartifacts:\n  git: %s\n" % ("c" * 40)
    prov = _git_prov(
        update_url="https://feed.example/f.yml", revision="v1.2.0"
    )
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: feed, ls_remote=_no)
    assert r.update_available is None


def test_feed_semantic_branch_reports_installed_version_as_current(tmp_path):
    """The semantic branch compares version vs version, so the reported
    current is the installed version — not the unrelated revision sha."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: plug\nversion: 1.0.0\nupdate_url: https://feed.example/f.yml\n",
        encoding="utf-8",
    )
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug
    r = check_provenanced(prov, fetch=lambda u: FEED, ls_remote=_no)
    assert r.update_available is True
    assert r.current == "1.0.0"
    assert r.latest == "1.2.0"


def test_malformed_yaml_feed_reports_unknown_row(tmp_path):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text("update_url: https://feed.example/f.yml\n", encoding="utf-8")
    prov = _git_prov(update_url="https://feed.example/f.yml")
    prov.path = plug
    result = check_provenanced(prov, fetch=lambda _: "version: [", ls_remote=_no)
    assert result.update_available is None
    assert "unparseable" in result.reason


def test_parse_feed_rejects_non_string_git_artifact():
    feed = "version: 1.2.0\nartifacts:\n  git:\n    url: x\n"
    with pytest.raises(ValueError):
        parse_feed_yml(feed)


def test_ls_remote_lifecycle_current_available_applied(tmp_path):
    """The git-path gate, end to end against a real local repo: current ->
    ls-remote sees a new HEAD -> revision recorded -> current. No network,
    no manual manifest rewriting — every comparison is the real command."""
    repo = tmp_path / "repo"
    run, head = _local_repo(repo, "-b", "main")
    sha1 = head()
    ls_remote = _real_ls_remote()

    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / ".git").mkdir()
    prov = _git_prov(revision=sha1, source=str(repo))
    prov.path = plug

    def check():
        return check_provenanced(prov, fetch=_no, ls_remote=ls_remote)

    assert check().update_available is False  # current

    run("-c", "user.email=t@e", "-c", "user.name=t",
        "commit", "--allow-empty", "-qm", "two")
    sha2 = head()
    assert sha2 != sha1
    r = check()
    assert r.update_available is True  # available
    assert r.latest == sha2

    # once the update pipeline has recorded the new revision, converged
    prov3 = _git_prov(revision=sha2, source=str(repo))
    prov3.path = plug
    assert check_provenanced(prov3, fetch=_no, ls_remote=ls_remote).update_available is False


def _real_ls_remote():
    git = _git_exe()
    env = _git_env()

    def ls_remote(source):
        proc = subprocess.run(
            [git, "ls-remote", source, "HEAD"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "ls-remote failed").strip()[:200])
        out = (proc.stdout or "").strip()
        return out.split("\t")[0] if out else ""
    return ls_remote


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
    source = tmp_path / "bare.git"
    git = _git_exe()
    env = _git_env()
    subprocess.run(
        [git, "init", "--bare", "-q", str(source)],
        check=True, capture_output=True, env=env,
    )
    # ls-remote on an empty bare repo: exit 0, empty HEAD — the command
    # SHAPE works; empty maps to unknown, never a crash
    proc = subprocess.run(
        [git, "ls-remote", str(source), "HEAD"],
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


class _Dist:
    def __init__(self, name):
        self._name = name

    @property
    def metadata(self):
        return {"Name": self._name}


def test_pip_uses_owning_distribution_not_import_root():
    """C17: the dist name comes from the entry point's owning distribution,
    never guessed from the value's first import module."""
    ep = _EP("mnemosyne", "some.module:register", None)
    ep.dist = _Dist("mnemosyne-hermes")
    seen = []

    rs = check_pip_plugins(
        installed_version=lambda d: (seen.append(d), "0.5.0")[1],
        pypi_latest=lambda d: None,
        entry_points=[ep],
    )
    assert seen == ["mnemosyne-hermes"]
    assert rs[0].current == "0.5.0"


def test_pip_entry_point_without_distribution_is_unknown_not_guessed():
    ep = _EP("orphan", "some.module:register", None)
    rs = check_pip_plugins(
        installed_version=lambda d: pytest.fail(f"guessed dist {d!r}"),
        pypi_latest=lambda d: pytest.fail("no fetch without a dist"),
        entry_points=[ep],
    )
    assert rs[0].update_available is None
    assert rs[0].latest is None
    assert "distribution" in rs[0].reason


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