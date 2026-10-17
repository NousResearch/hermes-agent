# Author: Sandip Kumar <scorpiosk@gmail.com>
"""Pre-write baseline reuse for an unchanged, already-checked file.

Reproduced with the real typescript-language-server: a file that had been checked clean
was opened again unchanged, and ``snapshot_baseline`` (call BEFORE every write) re-paid
the full configured wait budget (5.0 s) to re-derive the verdict the client already held
freshly.  The repair: ``LSPClient.baseline_reusable`` reuses the client's fresh stores
ONLY when the tracked document is fresh for its current version AND its tracked text
still matches the on-disk text under the same read rules as ``open_file`` (utf-8-sig —
BOM stripped, errors=replace, newline translation).

Contract under test — deterministic store-state transitions (verdicts, counters,
fall-through calls), never wall-clock timing; the real TypeScript NEW-error latency is
independently gated by the owner's cached_baseline_owner.py:

- fresh checked-clean AND fresh error verdicts are reused,
- an EXTERNAL content change invalidates a PREVIOUSLY FRESH store (None, not the old verdict),
- stale, seed-only and no-verdict stores never reuse,
- a fresh clean pull excludes an older push's error (merged fresh_only view),
- the text match follows the document's BOM/newline decoding,
- a deleted/untracked/never-opened document cannot reuse a previous verdict,
- the manager gate reuses only through a running client, and a no-verdict store falls
  through to the normal open+wait path with the configured budget (paid, not skipped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient, _DocState, file_uri
from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec
from agent.lsp.workspace import clear_cache

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")
ERROR_DIAG = {"code": "MOCK001", "message": "synthetic", "severity": 1,
              "range": {"start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 5}}}


def _tracked(tmp_path: Path):
    """A client holding one tracked document (version 1, text 'x=1\\n') — no server needed."""
    path = tmp_path / "x.py"
    path.write_text("x=1\n", encoding="utf-8")
    client = LSPClient(server_id="fixture", workspace_root=str(tmp_path), command=[sys.executable])
    doc = _DocState(version=1, text="x=1\n")
    client._docs[str(path)] = doc
    return path, client, doc


def test_fresh_clean_and_error_verdicts_are_reused(tmp_path: Path):
    """Both fresh verdicts are reused verbatim: [] legitimately means checked-clean."""
    path, client, doc = _tracked(tmp_path)
    doc.push_version, doc.push = 1, []
    assert client.baseline_reusable(str(path)) == []
    doc.push = [ERROR_DIAG]
    assert client.baseline_reusable(str(path)) == [ERROR_DIAG]


def test_external_change_invalidates_a_previously_fresh_store(tmp_path: Path):
    """The real invalidation transition: the SAME live store flips from reused ([]) to
    None the moment the file changes behind the client's back (text differs, version same)."""
    path, client, doc = _tracked(tmp_path)
    doc.push_version, doc.push = 1, []
    assert client.baseline_reusable(str(path)) == []
    path.write_text("x=2\n", encoding="utf-8")
    assert client.baseline_reusable(str(path)) is None
    path.write_text("x=1\n", encoding="utf-8")  # restored: the fresh verdict is valid again
    assert client.baseline_reusable(str(path)) == []


def test_stale_seed_only_and_no_verdict_stores_are_never_reused(tmp_path: Path):
    """Version-ahead doc (push tags stale), seed-only store, and no-verdict store: all None."""
    path, client, doc = _tracked(tmp_path)
    doc.version = 2  # didChange moved on: the v1 push is now stale
    doc.push_version, doc.push = 1, [ERROR_DIAG]
    assert client.baseline_reusable(str(path)) is None
    # Seed-on-first-push tagging (as _handle_publish_diagnostics stores it): data present,
    # no freshness tag, seed_seen set — baseline data only, never a verdict.
    doc.push_version, doc.pull_version, doc.seed_seen = -1, -1, True
    assert client.baseline_reusable(str(path)) is None
    # No verdict anywhere: fresh doc, both stores untagged, no seed.
    doc.version, doc.seed_seen = 1, False
    assert client.baseline_reusable(str(path)) is None


def test_fresh_clean_pull_excludes_an_older_push_error(tmp_path: Path):
    """A fresh CLEAN pull (v2) must win over the stale v1 push error in the merged view."""
    path, client, doc = _tracked(tmp_path)
    doc.version = 2
    doc.push_version, doc.push = 1, [ERROR_DIAG]
    doc.pull_version, doc.pull = 2, []
    assert client.baseline_reusable(str(path)) == []


def test_reuse_follows_document_bom_and_newline_decoding(tmp_path: Path):
    """utf-8-sig + text-mode newline decoding: BOM + CRLF bytes on disk ARE the tracked text."""
    path, client, doc = _tracked(tmp_path)
    doc.push_version, doc.push = 1, []
    path.write_bytes(b"\xef\xbb\xbfx=1\r\n")
    assert client.baseline_reusable(str(path)) == []


def test_deleted_or_untracked_document_cannot_reuse(tmp_path: Path):
    """Deleted file, never-tracked file, and a relatedDocuments spillover entry
    (version -1): the full fresh path must run for all three."""
    path, client, doc = _tracked(tmp_path)
    doc.push_version, doc.push = 1, []
    other = tmp_path / "other.py"
    other.write_text("y = 1\n", encoding="utf-8")
    client._docs[str(other)] = _DocState(version=-1, text="y = 1\n", push=[], push_version=-1)
    assert client.baseline_reusable(str(other)) is None  # never opened: version < 0
    assert client.baseline_reusable(str(tmp_path / "missing.py")) is None  # no store at all
    path.unlink()
    assert client.baseline_reusable(str(path)) is None  # unreadable: never matches tracked text


# ---- manager gate: LSPService._reusable_baseline ----


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    return repo


def _install_mock(script: str, monkeypatch, repo: Path, *, seed: bool = False):
    """Install the mock as ``pyright`` (so .py routes to it) over a git+pyproject root."""
    idx = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[idx]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER], workspace_root=root, cwd=root,
            env={"MOCK_LSP_SCRIPT": script}, initialization_options={},
        )

    SERVERS[idx] = ServerDef(
        server_id="pyright", extensions=original.extensions,
        resolve_root=lambda fp, ws: ws, build_spawn=_spawn,
        seed_first_push=seed, description="mock pyright " + script,
    )
    monkeypatch.chdir(str(repo))
    clear_cache()

    def _restore():
        SERVERS[idx] = original
        clear_cache()

    return _restore


def _service(repo: Path) -> LSPService:
    # wait_timeout is the BUDGET value under test (what fall-through must request),
    # not a timing assertion — this suite asserts outcomes and counters, never elapsed time.
    return LSPService(enabled=True, wait_mode="document", wait_timeout=1.0,
                      install_strategy="manual", trusted_workspaces=[str(repo)])


def test_manager_gate_reuses_only_live_and_matching(monkeypatch, tmp_path):
    """Warm a clean verdict through the real client: the gate reuses [] while the text
    matches, flips to None on an EXTERNAL change (and back on restore), and returns None
    with no running client after shutdown."""
    repo = _repo(tmp_path)
    restore = _install_mock("clean", monkeypatch, repo)
    f = repo / "x.py"
    f.write_text("x = 1\n")
    svc = _service(repo)
    try:
        assert svc.get_diagnostics_sync(str(f), delta=False) == []  # checked clean, tracked
        assert svc._reusable_baseline(str(f)) == []
        f.write_text("x = 1\ny = 2\n")  # external change behind the client's back
        assert svc._reusable_baseline(str(f)) is None
        f.write_text("x = 1\n")  # restored: the same live store is valid again
        assert svc._reusable_baseline(str(f)) == []
    finally:
        svc.shutdown()
        restore()
    assert svc._reusable_baseline(str(f)) is None  # no running client after shutdown


def test_no_verdict_store_falls_through_to_full_path(monkeypatch, tmp_path):
    """Deterministic fall-through (counted at the seam production calls): a no-verdict
    store must reach the normal open+wait path with the configured budget — and the
    baseline it stores is [] (empty baseline PAID for), never a reused verdict."""
    repo = _repo(tmp_path)
    restore = _install_mock("silent", monkeypatch, repo)
    f = repo / "x.py"
    f.write_text("x = 1\n")
    svc = _service(repo)
    budgets: list = []
    original_open_wait = svc._open_and_wait_async

    async def counting_open_wait(file_path, *, snapshot=False, budget=None):
        budgets.append(budget)
        return await original_open_wait(file_path, snapshot=snapshot, budget=budget)

    svc._open_and_wait_async = counting_open_wait
    try:
        # Warm-up contact (its own full path), then the file is tracked with NO verdict.
        svc.get_diagnostics_sync(str(f), delta=False)
        assert svc._reusable_baseline(str(f)) is None, "no-verdict stores must not reuse"
        budgets.clear()
        svc.snapshot_baseline(str(f))
        assert budgets == [1.0], f"fall-through must pay the configured budget exactly once: {budgets}"
        assert svc._delta_baseline[str(f.resolve())] == []  # empty baseline, but paid for honestly
    finally:
        svc.shutdown()
        restore()


def test_seed_store_never_reuses_through_the_real_handler(monkeypatch, tmp_path):
    """Seed tagging through the REAL client and REAL publishDiagnostics handler on the
    mock 'silent' script (no pull channel either): the first push is stored WITHOUT a
    freshness tag, so the manager gate refuses to reuse it even though the store is full
    of data."""
    repo = _repo(tmp_path)
    restore = _install_mock("silent", monkeypatch, repo, seed=True)
    f = repo / "x.py"
    f.write_text("x = 1\n")
    svc = _service(repo)
    try:
        client = svc._loop.run(svc._get_or_spawn(str(f)))
        assert client is not None and client._seed_first_push, "mock must be installed as a seed server"
        version = svc._loop.run(client.open_file(str(f), language_id="python"))
        assert version == 0
        client._handle_publish_diagnostics({  # the mock's didOpen error, via the real handler
            "uri": file_uri(str(f.resolve())), "version": 0, "diagnostics": [ERROR_DIAG],
        })
        doc = client._docs[str(f.resolve())]
        assert doc.push == [ERROR_DIAG] and doc.push_version == -1 and doc.seed_seen, \
            "seed push must be stored WITHOUT a freshness tag"
        assert svc._reusable_baseline(str(f)) is None, "a seed-only store must never be reused"
    finally:
        svc.shutdown()
        restore()
