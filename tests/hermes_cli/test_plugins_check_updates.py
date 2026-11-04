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
    assert r.current == "1.0.0"
    assert r.min_hermes == "0.27.0"
    assert r.update_available is True  # installed 1.0.0 vs feed 1.2.0


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


def test_ls_remote_lifecycle_current_available_applied(tmp_path, monkeypatch):
    """The git-path gate, end to end against a real local repo: current ->
    ls-remote sees a new HEAD -> revision recorded -> current. No network,
    no manual manifest rewriting — every comparison is the real command."""
    repo = tmp_path / "repo"
    run, head = _local_repo(repo, "-b", "main")
    sha1 = head()
    from hermes_cli.plugins_updates import default_ls_remote
    monkeypatch.setattr('hermes_cli.plugins_cmd._resolve_git_executable', _git_exe)
    ls_remote = default_ls_remote

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


def test_real_ls_remote_against_bare_repo(tmp_path, monkeypatch):
    source = tmp_path / "bare.git"
    git = _git_exe()
    env = _git_env()
    subprocess.run(
        [git, "init", "--bare", "-q", str(source)],
        check=True, capture_output=True, env=env,
    )
    from hermes_cli.plugins_updates import default_ls_remote
    monkeypatch.setattr('hermes_cli.plugins_cmd._resolve_git_executable', _git_exe)
    assert default_ls_remote(str(source)) == ""


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


@pytest.mark.parametrize("url", ["http://feed.example/f.yml", "file:///etc/passwd", "ftp://x/f.yml", ""])
def test_default_fetch_refuses_non_https_feeds_before_any_request(monkeypatch, url):
    """Rows saved before the https rule (or hand-edited) still reach the real fetcher from the
    gateway tick; the sink refuses them instead of opening the URL."""
    import urllib.request
    from hermes_cli.plugins_updates import default_fetch

    def never(*a, **k):
        raise AssertionError("urlopen must not be reached")

    monkeypatch.setattr(urllib.request, "urlopen", never)
    with pytest.raises(ValueError, match="https://"):
        default_fetch(url)


@pytest.fixture
def feed_redirect_server(monkeypatch):
    """Exercise urllib's redirect machinery without requiring a TLS certificate."""
    import http.client
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    visited = []

    class FeedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            visited.append(self.path)
            if self.path == "/start":
                target = f"http://127.0.0.1:{self.server.server_port}/middle"
            elif self.path == "/middle" or self.path == "/secure":
                target = f"https://127.0.0.1:{self.server.server_port}/feed"
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"version: 1.2.0\n")
                return
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()

        def log_message(self, *_args):
            pass

    # Only the HTTPS transport is replaced; the redirect handler and HTTP
    # transport stay real, so the server can observe a forbidden HTTP hop.
    monkeypatch.setattr(
        urllib.request.HTTPSHandler, "https_open",
        lambda self, req: self.do_open(http.client.HTTPConnection, req),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), FeedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}", visited
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_default_fetch_refuses_intermediate_plaintext_redirect(feed_redirect_server):
    from hermes_cli.plugins_updates import default_fetch

    base, visited = feed_redirect_server
    with pytest.raises(ValueError, match="https://"):
        default_fetch(base + "/start")
    assert visited == ["/start"]


def test_default_fetch_follows_https_redirect(feed_redirect_server):
    from hermes_cli.plugins_updates import default_fetch

    base, visited = feed_redirect_server
    assert default_fetch(base + "/secure") == "version: 1.2.0\n"
    assert visited == ["/secure", "/feed"]


# ── artifact-pinned: the update answer comes from the catalog ──
# (UPD-5, est-qrazy: archive installs are class=proven, outdatedness is
# the live catalog pin, and the remedy is never a git-clone downgrade.)


def _artifact_prov(**kw):
    row = kw.pop("row", {"revision": "9a6f8ac", "source": "https://github.com/jacobhausler/hermes-workflows"})
    marker = kw.pop("marker", {"version": "1.4.3", "source_commit": "9a6f8ac",
                              "source_repo": "https://github.com/jacobhausler/hermes-workflows"})
    return Provenance(name="plug", klass=ProvenanceClass.ARTIFACT, path=Path("/x/plug"),
                      row=row, install_json=marker, **kw)


def _entry(**kw):
    from hermes_cli.plugin_catalog import PluginCatalogEntry

    kw.setdefault("name", "hermes-workflows")
    kw.setdefault("repo", "https://github.com/jacobhausler/hermes-workflows")
    kw.setdefault("sha", "f" * 40)
    kw.setdefault("description", "")
    kw.setdefault("maintainer", "nous")
    kw.setdefault("version", "1.4.4")
    return PluginCatalogEntry(**kw)


def _stub_catalog(monkeypatch, entry, removed=None):
    import hermes_cli.plugin_catalog as pc

    monkeypatch.setattr(pc, "get_live_catalog_entry", lambda name: entry)
    monkeypatch.setattr(pc, "find_removed", lambda cand, **kw: removed)
    monkeypatch.setattr(pc, "load_catalog_live", lambda: [entry] if entry else [])


def test_artifact_version_marker_behind_catalog_pin(monkeypatch):
    """The epic's own case: sidecar 9a6f8ac, marker 1.4.3, catalog 1.4.4 →
    update available, computed like-for-like on versions, remedy is the
    archive — the text must not send the user to git clone."""
    from hermes_cli.plugins_updates import check_provenanced

    _stub_catalog(monkeypatch, _entry())
    result = check_provenanced(_artifact_prov(), fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert result.klass == "artifact-pinned"
    assert result.current == "1.4.3"
    assert result.latest == "1.4.4"
    assert result.update_available is True
    assert "git clone" not in result.reason.lower()
    assert "install the updated archive" in result.reason


def test_artifact_full_sha_marker_matches_catalog_pin(monkeypatch):
    from hermes_cli.plugins_updates import check_provenanced

    _stub_catalog(monkeypatch, _entry(sha="a" * 40))
    prov = _artifact_prov(marker={"version": "1.4.4", "source_commit": "a" * 40,
                                  "source_repo": "https://github.com/jacobhausler/hermes-workflows"})
    result = check_provenanced(prov, fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert result.update_available is False


def test_artifact_full_sha_marker_moved_pin_reports_update_on_shas(monkeypatch):
    from hermes_cli.plugins_updates import check_provenanced

    _stub_catalog(monkeypatch, _entry(sha="f" * 40, version=""))
    prov = _artifact_prov(marker={"version": "1.4.3", "source_commit": "a" * 40})
    result = check_provenanced(prov, fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert result.update_available is True
    assert result.current == "a" * 40 and result.latest == "f" * 40  # shas — no human label to move to


def test_artifact_uncomparable_marker_is_unknown_not_an_update(monkeypatch):
    """A short-sha marker with no version to fall back on: like-for-like
    discipline — unknown, not evidence of an update."""
    from hermes_cli.plugins_updates import check_provenanced

    _stub_catalog(monkeypatch, _entry(version=""))
    prov = _artifact_prov(marker={"source_commit": "9a6f8ac"})
    result = check_provenanced(prov, fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert result.update_available is None
    assert "cannot be compared" in result.reason


def test_artifact_missing_catalog_entry_retains_the_pin(monkeypatch):
    from hermes_cli.plugins_updates import check_provenanced

    _stub_catalog(monkeypatch, None)
    result = check_provenanced(_artifact_prov(), fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert result.update_available is None
    assert "catalog entry unavailable" in result.reason


def test_artifact_removed_entry_blocks(monkeypatch):
    from hermes_cli.plugin_catalog import RemovedEntry
    from hermes_cli.plugins_updates import check_provenanced

    _stub_catalog(monkeypatch, _entry(), removed=RemovedEntry(name="plug", reason="sunset"))
    result = check_provenanced(_artifact_prov(), fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert result.update_available is None
    assert "removed from catalog" in result.reason


def test_artifact_catalog_name_from_the_row_wins(monkeypatch):
    """A row that records its catalog key needs no reverse lookup."""
    from hermes_cli.plugins_updates import check_provenanced

    seen = []
    import hermes_cli.plugin_catalog as pc

    def fake_entry(name):
        seen.append(name)
        return _entry(name="wf", sha="a" * 40)

    monkeypatch.setattr(pc, "get_live_catalog_entry", fake_entry)
    monkeypatch.setattr(pc, "find_removed", lambda cand, **kw: None)
    prov = _artifact_prov(row={"catalog": {"name": "wf"}, "source": "https://example/o/r"},
                          marker={"version": "1.0.0", "source_commit": "a" * 40})
    result = check_provenanced(prov, fetch=lambda u: "{}", ls_remote=lambda s: "")
    assert seen == ["wf"]
    assert result.update_available is False


def test_local_provenance_artifact_is_proven_not_drift():
    """check_local_provenance (the offline surface the doctor rides) reads
    the artifact cell as proven with its recorded identity — no drift
    warning, no reinstall remedy."""
    from hermes_cli.plugins_updates import check_local_provenance

    result = check_local_provenance(_artifact_prov())
    assert result.klass == "artifact-pinned"
    assert result.current == "9a6f8ac"  # marker commit over the (possible stale) row revision
    assert result.needs_fixing is None
    assert "reinstall" not in result.reason and "drift" not in result.reason
