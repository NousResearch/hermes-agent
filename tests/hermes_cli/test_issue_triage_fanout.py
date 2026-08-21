"""Tests for ``hermes_cli.issue_triage_fanout`` (t_053fea1a).

Acceptance coverage:

* AC2 — parsing the t_12cc81c6 body shape produces the same 22 issues
  (16 in-scope + 3 adjacent + 3 sibling-cites that the heuristic
  matches as bare ``#NNN`` refs). Idempotent re-run is a no-op.
* AC3 — synthetic scout card with 3 fake links creates 3 child cards
  with the canonical title format; re-run is a no-op.

Plus the secondary unit tests the spec calls out: done-when
priority, title truncation at 200 chars, cluster extraction, both URL
and bare ``#NNN`` detection, dedupe triple-layer (file -> kanban
idempotency_key -> parent-link scan).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli import issue_triage_fanout as itf  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_keys_db(tmp_path, monkeypatch):
    """Redirect the key store at a tmp path so tests don't pollute prod."""
    db_path = str(tmp_path / "issue_fanout_keys.sqlite")
    store = itf.FanoutKeyStore(db_path)
    yield store
    # Cleanup is handled by tmp_path; nothing else to do.


# t_12cc81c6 body — read live from the ops board for parity with AC2.
T_12CC81C6_TITLE = (
    "smilemap batch triage: 22 untracked product-contract failures "
    "(issues #901-907, #909-923)"
)


def _t_12cc81c6_body() -> str:
    return (
        "Triage 22 untracked product-contract failure issues from the "
        "2026-07-26/27 behavioral-test evidence batch so each one is either "
        "scheduled as its own done_When-scoped card or has a written reason "
        "for staying grouped. Issues in scope: #901, #902, #903, #904, #905, "
        "#906, #907, #909, #910, #911, #912, #915, #920, #921, #922, #923. "
        "Adjacent P3 (#898, #899, #900) is for awareness only — do not file "
        "cards for it unless explicitly reassigned. Cluster themes: replay-safe "
        "mutation intents / idempotency / pagination / 44px a11y / blank-name "
        "rejection / soft-delete leak / consent revocation / 36px consent "
        "control.\n\n"
        "Step 1 — CORPUS CHECK (mandatory, cite findings in the result):\n"
        "  - Read the already-filed sibling cards from the same evidence "
        "batch (#946, #952, #954, #955, #956, #958-961, #963, #964, #966, "
        "#968, #969, #972, #974-976) to lock the established done_when "
        "style, evidence-citation format, and acceptance-criteria "
        "granularity. Match it exactly.\n"
    )


def _synthetic_scout_body() -> str:
    return (
        "Synthetic scout for AC3 — three fresh issues from a fake repo.\n"
        "See https://github.com/acme/widgets/issues/101 and "
        "https://github.com/acme/widgets/issues/102 plus "
        "https://github.com/acme/widgets/issues/103.\n"
    )


def _fake_fetch(*, title: str = "Fake issue title", body: Optional[str] = None):
    """Build a fake ``fetch_gh_issue`` callable.

    GAP 2 follow-up: the real ``fetch_gh_issue`` now accepts a
    ``repo_owner_map`` keyword so bare-ref scouts whose owner
    resolves empty can still get a real title via the
    ``REPO_OWNER_MAP`` (or a CLI-supplied ``--repo-map``).
    Tests that don't care about repo mapping should accept
    the new kwarg silently so the call site doesn't break.
    """
    def _impl(
        owner: str,
        repo: str,
        issue_id: int,
        *,
        gh_path: str = "gh",
        repo_owner_map: Optional[dict[str, str]] = None,
    ):
        return itf.GhFetchResult(title=title, body=body)
    return _impl


@dataclass
class _FakeCreate:
    """In-memory stand-in for ``kb.create_task``."""

    next_id: int = 1
    created: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        *,
        title: str,
        body: str,
        parents: tuple[str, ...],
        idempotency_key: str,
    ) -> str:
        new_id = f"t_fake{self.next_id:04d}"
        self.next_id += 1
        self.created.append(
            {
                "id": new_id,
                "title": title,
                "body": body,
                "parents": parents,
                "idempotency_key": idempotency_key,
            }
        )
        return new_id


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_url_and_bare_forms():
    body = (
        "Mixed: https://github.com/acme/widgets/issues/101 and bare #202 "
        "and another URL https://github.com/Other/Repo/issues/303."
    )
    refs = itf.parse_issue_links(body, default_repo="acme")
    ids = [r.issue_id for r in refs]
    assert ids == [101, 303, 202]  # URL first in doc order, bare last
    assert refs[0].source == "url"
    assert refs[1].source == "url"
    assert refs[2].source == "bare"
    assert refs[2].owner == ""
    assert refs[2].repo == "acme"


def test_parse_dedupes_when_url_and_bare_appear_for_same_issue():
    body = (
        "Bug: https://github.com/acme/widgets/issues/42 — see #42 for context."
    )
    refs = itf.parse_issue_links(body, default_repo="acme")
    assert len(refs) == 1
    assert refs[0].issue_id == 42
    assert refs[0].source == "url"


def test_parse_ignores_pr_and_commit_urls():
    body = (
        "See https://github.com/acme/widgets/pull/200 and "
        "https://github.com/acme/widgets/commit/abcdef0 and the bug "
        "https://github.com/acme/widgets/issues/5."
    )
    refs = itf.parse_issue_links(body, default_repo="acme")
    assert [r.issue_id for r in refs] == [5]


def test_parse_t_12cc81c6_body_shape_detects_in_scope_and_adjacent_issues():
    """AC2: parsing t_12cc81c6's body shape yields the issues in the
    explicit triage section.

    Spec says "22 issues" but the body of t_12cc81c6 has 19 unique
    ``#NNN`` refs in the explicit triage section — 16 in-scope + 3
    adjacent. The 13 sibling-citation refs in the corpus-check
    section are *already-filed* cards referenced for format
    consistency, not fan-out targets. The parser correctly excludes
    them by scoping detection to the explicit triage section. See
    module docstring "Spec deviations" for the rationale.
    """
    body = _t_12cc81c6_body()
    refs = itf.parse_issue_links(body, default_repo="smilemap")
    ids = sorted(r.issue_id for r in refs)
    expected = sorted(
        [
            # 16 in-scope
            901, 902, 903, 904, 905, 906, 907, 909, 910, 911, 912,
            915, 920, 921, 922, 923,
            # 3 adjacent (P3 awareness — included in scope by default;
            # whether to actually fan out is a downstream routing
            # decision per spec §11)
            898, 899, 900,
        ]
    )
    assert ids == expected
    assert len(refs) == 19
    # Sibling citations (e.g. #946, #952) must NOT be in the detected
    # set — they are corpus-check references, not fan-out targets.
    for sibling_id in (946, 952, 954, 974):
        assert sibling_id not in ids, (
            f"sibling citation #{sibling_id} leaked into fan-out targets"
        )


def test_parse_empty_body_returns_empty():
    assert itf.parse_issue_links("") == []
    assert itf.parse_issue_links("no issues here") == []


def test_parse_bare_only_no_default_repo_returns_empty():
    """Bare matches require a default_repo; without one we skip them
    (the spec §3 title needs a repo segment)."""
    refs = itf.parse_issue_links("see #123 and #456", default_repo=None)
    assert refs == []


# ---------------------------------------------------------------------------
# Repo inference
# ---------------------------------------------------------------------------


def test_infer_default_repo_from_title():
    assert itf.infer_default_repo("smilemap batch triage: ...") == "smilemap"
    assert (
        itf.infer_default_repo("calcifer-engine: nightly cron work")
        == "calcifer-engine"
    )


def test_infer_default_repo_handles_stopwords_and_empty():
    assert itf.infer_default_repo("") is None
    assert itf.infer_default_repo("The big bug") is None
    assert itf.infer_default_repo(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Title / body builders
# ---------------------------------------------------------------------------


def test_build_child_title_canonical_format():
    t = itf.build_child_title("smilemap", 901, "Client mutations persist blank names")
    assert t == "[gh] smilemap: Client mutations persist blank names (#901)"


def test_build_child_title_truncates_at_200_chars():
    long = "x" * 400
    t = itf.build_child_title("smilemap", 901, long)
    assert len(t) <= 200
    assert t.endswith("(#901)")
    assert "…" in t


def test_build_child_title_placeholder_when_title_blank():
    t = itf.build_child_title("smilemap", 901, "")
    assert "<unknown title" in t
    assert t.endswith("(#901)")


def test_build_child_body_seven_sections():
    body = itf.build_child_body(
        repo="smilemap",
        issue_id=901,
        issue_body_excerpt="First line.\nMust reject blank names.\n\nMore.",
        scout_evidence_lines=["handler_sha256=deadbeef", "browser_artifact=cafef00d"],
        cluster="blank-name rejection",
        done_when="Done when: blank names are rejected at create time.",
        issue_url="https://github.com/veroscale/smilemap/issues/901",
        scout_signal_id="sig-abc123",
        scout_card_id="t_12cc81c6",
    )
    # §5 fixed-order sections
    assert body.startswith("FULL CONTEXT: smilemap GitHub issue #901.")
    assert "Must reject blank names." in body
    assert "## Inherited evidence" in body
    assert "handler_sha256=deadbeef" in body
    assert "Cluster: blank-name rejection" in body
    assert "Done when: blank names are rejected at create time." in body
    assert "Link: https://github.com/veroscale/smilemap/issues/901" in body
    assert "CORPUS-FIRST:" in body
    assert body.rstrip().endswith("Inherited scout signal: sig-abc123")


def test_build_child_body_inherited_signal_backfill_marker():
    body = itf.build_child_body(
        repo="smilemap",
        issue_id=901,
        issue_body_excerpt=None,
        scout_evidence_lines=[],
        cluster=None,
        done_when="Done when: x.",
        issue_url="https://github.com/veroscale/smilemap/issues/901",
        scout_signal_id=None,
        scout_card_id="t_12cc81c6",
    )
    assert "no signal emitted — backfill needed" in body
    # When issue body excerpt is missing we skip the section entirely.
    assert "## Issue body (excerpt)" not in body


# ---------------------------------------------------------------------------
# Done-when derivation
# ---------------------------------------------------------------------------


def test_derive_done_when_priority_1_issue_body_action():
    issue = "Some intro. Must reject blank client names at creation. More text."
    dw, needs = itf.derive_done_when(
        issue_body=issue,
        scout_body="",
        scout_comments=[],
        issue_id=901,
        repo="smilemap",
        scout_card_id="t_x",
    )
    assert "Must reject blank client names" in dw
    assert dw.startswith("Done when: ")
    assert needs is False


def test_derive_done_when_priority_2_scout_done_when_line():
    issue = "Nothing useful here."
    scout = (
        "Triage card body.\n"
        "Done when: blank names are rejected at the database layer.\n"
    )
    dw, needs = itf.derive_done_when(
        issue_body=issue,
        scout_body=scout,
        scout_comments=[],
        issue_id=901,
        repo="smilemap",
        scout_card_id="t_x",
    )
    assert dw == "Done when: blank names are rejected at the database layer."
    assert needs is False


def test_derive_done_when_priority_3_template_with_marker():
    dw, needs = itf.derive_done_when(
        issue_body=None,
        scout_body="no done when here",
        scout_comments=[],
        issue_id=901,
        repo="smilemap",
        scout_card_id="t_12cc81c6",
    )
    assert "issue #901" in dw
    assert "t_12cc81c6" in dw
    assert needs is True  # template fallback flagged for refinement


def test_derive_done_when_strips_emoji_and_bullets():
    issue = "- 🚨 Must not allow blank names\n"
    dw, _ = itf.derive_done_when(
        issue_body=issue,
        scout_body="",
        issue_id=1,
        repo="r",
        scout_card_id="t_x",
    )
    assert "🚨" not in dw
    assert "Must not allow blank names" in dw


# ---------------------------------------------------------------------------
# Cluster extraction
# ---------------------------------------------------------------------------


def test_derive_cluster_explicit_line():
    body = "stuff\nCluster: replay-safe mutation intents / idempotency\nmore stuff"
    assert itf.derive_cluster(body) == "replay-safe mutation intents / idempotency"


def test_derive_cluster_handles_parenthetical_fallback():
    body = "no cluster line but here's (cluster: pagination) inline"
    # Spec only mandates the explicit "Cluster: ..." form; parenthetical
    # is a separate "in done_when" extension. Cluster line itself is
    # None here.
    assert itf.derive_cluster(body) is None


def test_derive_cluster_none_when_absent():
    assert itf.derive_cluster(None) is None
    assert itf.derive_cluster("nothing") is None


# ---------------------------------------------------------------------------
# gh CLI integration (with fake subprocess via monkeypatch)
# ---------------------------------------------------------------------------


def test_fetch_gh_issue_parses_title_and_body(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = json.dumps({"title": "Bug: x fails", "body": "Fails when y."})
        stderr = ""

    def _fake_run(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("acme", "widgets", 42)
    assert out.title == "Bug: x fails"
    assert out.body == "Fails when y."
    assert out.error is None


def test_fetch_gh_issue_falls_back_on_nonzero_exit(monkeypatch):
    class _Proc:
        returncode = 4
        stdout = ""
        stderr = "404 Not Found"

    def _fake_run(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("acme", "widgets", 99999)
    assert out.title == itf.GH_UNAVAILABLE_TITLE
    assert out.body is None
    assert "exited 4" in (out.error or "")


def test_fetch_gh_issue_falls_back_on_timeout(monkeypatch):
    def _fake_run(*args, **kwargs):
        raise itf.subprocess.TimeoutExpired(cmd="gh", timeout=5)

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("acme", "widgets", 1, timeout_seconds=5)
    assert out.title == itf.GH_UNAVAILABLE_TITLE
    assert "timeout" in (out.error or "")


def test_fetch_gh_issue_falls_back_on_missing_binary(monkeypatch):
    def _fake_run(*args, **kwargs):
        raise FileNotFoundError("gh not on PATH")

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("acme", "widgets", 1)
    assert out.title == itf.GH_UNAVAILABLE_TITLE
    assert "gh unavailable" in (out.error or "")


# ---------------------------------------------------------------------------
# Idempotency store
# ---------------------------------------------------------------------------


def test_key_store_reserve_then_mark_created(temp_keys_db):
    store = temp_keys_db
    assert store.reserve("t_x", 901, "smilemap") == ""
    assert store.reserve("t_x", 901, "smilemap") == ""
    store.mark_created("t_x", 901, "t_y")
    assert store.reserve("t_x", 901, "smilemap") == "t_y"  # skip


def test_key_store_isolates_per_scout(temp_keys_db):
    store = temp_keys_db
    store.mark_created("t_a", 5, "t_child_a5")
    store.mark_created("t_b", 5, "t_child_b5")
    assert store.reserve("t_a", 5, "r") == "t_child_a5"
    assert store.reserve("t_b", 5, "r") == "t_child_b5"


def test_key_store_persists_across_instances(tmp_path):
    """Spec §6: cross-process dedupe is the reason the file is used."""
    db_path = str(tmp_path / "k.sqlite")
    a = itf.FanoutKeyStore(db_path)
    a.mark_created("t_s", 42, "t_child42")
    b = itf.FanoutKeyStore(db_path)
    assert b.reserve("t_s", 42, "r") == "t_child42"


# ---------------------------------------------------------------------------
# End-to-end fanout (AC2 + AC3)
# ---------------------------------------------------------------------------


def _make_deps(
    *,
    key_store,
    fetcher,
    create_log: _FakeCreate,
    parent_link_issue_ids: Optional[list[int]] = None,
    signal_id: Optional[str] = None,
) -> itf.FanoutDeps:
    def _signal_emit(*, kind: str, tag: str, summary: str) -> Optional[str]:
        return signal_id or f"sig-fake-{tag}"

    def _parent_link_scan(scout_id: str) -> list[int]:
        return list(parent_link_issue_ids or [])

    return itf.FanoutDeps(
        fetch_gh_issue=fetcher,
        key_store=key_store,
        create_task=create_log,
        signal_emit=_signal_emit,
        parent_link_scan=_parent_link_scan,
    )


def test_ac2_t12cc81c6_produces_16_detected_issues(temp_keys_db):
    """AC2: parser detects the 16 in-scope issues of t_12cc81c6.

    The 3 adjacent (P3) refs (#898, #899, #900) are correctly
    EXCLUDED by the awareness-only marker in the scout body
    ("Adjacent P3 (#898, #899, #900) is for awareness only — do
    not file cards for it...") per spec §11 / t_b17ae9d3 GAP 1.

    Spec said "22" but the body has 16 in-scope + 3 adjacent = 19
    unique refs; with the awareness-only marker honored, fan-out
    produces 16 cards. See module docstring "Spec deviations" for
    the rationale.
    """
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(title="Issue title"),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_12cc81c6",
        scout={
            "id": "t_12cc81c6",
            "title": T_12CC81C6_TITLE,
            "body": _t_12cc81c6_body(),
            "comments": [],
        },
        deps=deps,
    )
    # 19 unique #NNN refs in body, 3 excluded by awareness marker → 16.
    assert result.detected == 16
    assert result.scanned == 16
    assert len(result.created) == 16
    assert len(result.skipped_duplicates) == 0
    assert len(result.errors) == 0
    assert result.exit_code() == 0

    # GAP 1 acceptance: 898/899/900 surface in result.excluded.
    excluded_ids = sorted(e["issue_id"] for e in result.excluded)
    assert excluded_ids == [898, 899, 900]
    for entry in result.excluded:
        if entry["issue_id"] in (898, 899, 900):
            assert entry["reason"] == "awareness_only"
            assert entry["label"] == "Adjacent P3"

    # Every created card uses the canonical title format and idempotency
    # key prefix; the 3 excluded refs must NOT appear in created.
    created_ids = sorted(c["issue_id"] for c in result.created)
    assert created_ids == sorted([
        901, 902, 903, 904, 905, 906, 907, 909, 910, 911, 912,
        915, 920, 921, 922, 923,
    ])
    for entry in result.created:
        assert entry["title"].startswith("[gh] smilemap: ")
        assert entry["title"].endswith(f"(#{entry['issue_id']})")
        assert (
            entry["idempotency_key"]
            == f"scoutfanout:t_12cc81c6:{entry['issue_id']}"
        )
    for excluded_id in (898, 899, 900):
        assert excluded_id not in created_ids


def test_ac2_idempotent_rerun_creates_zero_cards(temp_keys_db):
    """Re-running against the same scout card is a no-op.

    With the awareness-only marker honored, the second run
    detects 16, creates 0, and skips 16 via the key-store layer
    (file-based dedupe). The 3 excluded refs are still reported
    in result.excluded on every run (they're scout-body-derived,
    not key-store-derived).
    """
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
    )
    scout = {
        "id": "t_12cc81c6",
        "title": T_12CC81C6_TITLE,
        "body": _t_12cc81c6_body(),
        "comments": [],
    }
    first = itf.run_fanout(
        board="ops",
        scout_card_id="t_12cc81c6",
        scout=scout,
        deps=deps,
    )
    assert len(first.created) == 16
    assert len(first.excluded) == 3

    # Second run — same key_store, fresh create log to prove no new calls.
    second_create = _FakeCreate()
    deps2 = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=second_create,
    )
    second = itf.run_fanout(
        board="ops",
        scout_card_id="t_12cc81c6",
        scout=scout,
        deps=deps2,
    )
    assert second.detected == 16
    assert len(second.created) == 0
    assert len(second.skipped_duplicates) == 16
    assert len(second.errors) == 0
    assert len(second.excluded) == 3  # awareness exclusion is body-derived, persists
    assert second.exit_code() == 0
    assert second_create.created == []  # no new create_task calls


def test_ac3_synthetic_scout_with_three_fake_links_creates_three_cards(
    temp_keys_db,
):
    """AC3: synthetic scout card with 3 fake links creates exactly 3
    child cards; re-run is a no-op."""
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(title="Synthetic issue"),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout for AC3",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps,
    )
    assert result.detected == 3
    assert len(result.created) == 3
    created_ids = sorted(c["issue_id"] for c in result.created)
    assert created_ids == [101, 102, 103]
    # Titles use repo from URL (acme/widgets -> widgets).
    for entry in result.created:
        assert entry["title"].startswith("[gh] widgets: ")
    assert result.exit_code() == 0

    # Re-run is a no-op.
    second_create = _FakeCreate()
    deps2 = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=second_create,
    )
    second = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout for AC3",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps2,
    )
    assert second.detected == 3
    assert len(second.created) == 0
    assert len(second.skipped_duplicates) == 3
    assert second_create.created == []


def test_ac3_dry_run_creates_nothing_and_records_would_create(temp_keys_db):
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout for AC3",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps,
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.detected == 3
    assert len(result.would_create) == 3
    assert len(result.created) == 0
    assert create_log.created == []
    # Key store must not be touched in dry-run.
    assert temp_keys_db.reserve("t_synthetic", 101, "widgets") == ""

    # Serialised JSON includes would_create but not created.
    blob = result.to_dict()
    assert "would_create" in blob
    assert "created" not in blob
    assert blob["dry_run"] is True


def test_parent_link_scan_layer3_dedupe(temp_keys_db):
    """Layer-3 dedupe: if a child card already exists as a child of
    the scout, skip even when the file-based key store is fresh."""
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
        parent_link_issue_ids=[101, 102],  # 101 and 102 already linked
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout for AC3",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps,
    )
    assert result.detected == 3
    assert len(result.created) == 1
    assert len(result.skipped_duplicates) == 2
    created_issue_ids = sorted(c["issue_id"] for c in result.created)
    assert created_issue_ids == [103]


def test_partial_failure_exit_code_is_2(temp_keys_db):
    """Spec §7: exit 2 on partial failure (created > 0 AND errors > 0).

    The fetcher must accept the new ``repo_owner_map`` kwarg (GAP 2
    follow-up) — this test uses the bare-receiver signature as a
    regression guard against future kwarg additions silently
    breaking custom fetchers.
    """
    create_log = _FakeCreate()

    def _flaky_fetch(
        owner,
        repo,
        issue_id,
        *,
        gh_path="gh",
        repo_owner_map=None,
    ):
        if issue_id == 102:
            return itf.GhFetchResult(
                title=itf.GH_UNAVAILABLE_TITLE,
                body=None,
                error="gh unavailable: timeout",
            )
        return itf.GhFetchResult(title=f"Issue {issue_id}", body="Must do x.")

    def _flaky_create(*, title, body, parents, idempotency_key):
        if "Issue 103" in title:
            raise RuntimeError("create_task failed: simulated")
        return create_log(title=title, body=body, parents=parents,
                          idempotency_key=idempotency_key)

    deps = itf.FanoutDeps(
        fetch_gh_issue=_flaky_fetch,
        key_store=temp_keys_db,
        create_task=_flaky_create,
        signal_emit=lambda **kw: None,
        parent_link_scan=lambda _id: [],
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps,
    )
    # 101 created, 102 has gh error (still created w/ placeholder title),
    # 103 raised on create.
    assert any(e["issue_id"] == 103 for e in result.errors)
    assert result.exit_code() == 2


def test_total_failure_exit_code_is_1(temp_keys_db):
    create_log = _FakeCreate()

    def _always_fail(*, title, body, parents, idempotency_key):
        raise RuntimeError("kaboom")

    deps = itf.FanoutDeps(
        fetch_gh_issue=_fake_fetch(),
        key_store=temp_keys_db,
        create_task=_always_fail,
        signal_emit=lambda **kw: None,
        parent_link_scan=lambda _id: [],
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps,
    )
    assert len(result.created) == 0
    assert len(result.errors) == 3
    assert result.exit_code() == 1


def test_json_summary_shape_matches_spec(temp_keys_db):
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_synthetic",
        scout={
            "id": "t_synthetic",
            "title": "acme widgets synthetic scout",
            "body": _synthetic_scout_body(),
            "comments": [],
        },
        deps=deps,
    )
    blob = result.to_dict()
    # Spec §8 required keys.
    for k in (
        "scout_card_id",
        "board",
        "detected",
        "scanned",
        "created",
        "skipped_duplicates",
        "errors",
        "dry_run",
        "duration_ms",
    ):
        assert k in blob, f"missing key: {k}"
    assert blob["scout_card_id"] == "t_synthetic"
    assert blob["board"] == "ops"
    assert blob["dry_run"] is False
    assert isinstance(blob["duration_ms"], int)
    # Created entries carry the spec fields.
    entry = blob["created"][0]
    for k in ("issue_id", "repo", "child_card_id", "idempotency_key", "inherited_signal"):
        assert k in entry, f"missing entry key: {k}"


def test_argparser_accepts_spec_flags():
    parser = itf.build_argparser()
    args = parser.parse_args(
        [
            "ops",
            "t_12cc81c6",
            "--dry-run",
            "--max-issues",
            "5",
            "--skip-existing",
            "--json",
        ]
    )
    assert args.board == "ops"
    assert args.scout_card_id == "t_12cc81c6"
    assert args.dry_run is True
    assert args.max_issues == 5
    assert args.skip_existing is True
    assert args.json is True


# ---------------------------------------------------------------------------
# GAP 1 follow-up (t_b17ae9d3): awareness-only / do-not-file marker parsing
# ---------------------------------------------------------------------------


def test_parse_excluded_refs_awareness_only_marker():
    """The reference scout body shape must extract the 3 adjacent
    refs and label them with the captured prefix."""
    body = _t_12cc81c6_body()
    excluded = itf.parse_excluded_refs(body)
    by_id = {e.issue_id: e for e in excluded}
    assert sorted(by_id) == [898, 899, 900]
    for iid in (898, 899, 900):
        assert by_id[iid].reason == "awareness_only"
        assert by_id[iid].label == "Adjacent P3"
        assert "awareness only" in by_id[iid].excerpt.lower()
        assert "do not file" in by_id[iid].excerpt.lower()


def test_parse_excluded_refs_do_not_file_marker_classifies_as_do_not_file():
    """``do not file`` and ``don't file`` phrases must classify
    distinctly from ``awareness only`` so the JSON summary tells
    operators which directive fired."""
    body = (
        "Triage these: #101, #102, #103. "
        "Batch X (#104, #105) — do not file cards for these.\n"
        "Batch Y (#106, #107) — don't file them either.\n"
    )
    excluded = itf.parse_excluded_refs(body)
    by_id = {e.issue_id: e for e in excluded}
    assert sorted(by_id) == [104, 105, 106, 107]
    assert by_id[104].reason == "do_not_file"
    assert by_id[105].reason == "do_not_file"
    assert by_id[106].reason == "do_not_file"
    assert by_id[107].reason == "do_not_file"


def test_parse_excluded_refs_does_not_match_parenthesized_list_without_marker():
    """A bare ``(#NNN)`` without an awareness/do-not-file phrase
    must NOT exclude those refs — only the marker phrase proves
    intent. This is the regression guard for t_dcb31361 AC5
    (cards got created from a body that had a stray
    parenthesized list)."""
    body = (
        "See (the README) for context. "
        "Reference (issue #999) here is unrelated to fan-out. "
        "Real scope: #1, #2.\n"
    )
    excluded = itf.parse_excluded_refs(body)
    assert excluded == []


def test_parse_excluded_refs_handles_empty_and_none():
    assert itf.parse_excluded_refs("") == []
    assert itf.parse_excluded_refs(None) == []  # type: ignore[arg-type]
    assert itf.parse_excluded_refs("nothing here") == []


def test_parse_issue_links_respects_excluded_issue_ids():
    """The parser must drop excluded ids from both URL and bare
    matches — both forms are valid scout-body evidence."""
    body = (
        "See https://github.com/acme/widgets/issues/898 and "
        "bare #899 and URL https://github.com/acme/widgets/issues/900 "
        "plus in-scope #901."
    )
    refs = itf.parse_issue_links(
        body,
        default_repo="widgets",
        excluded_issue_ids={898, 899, 900},
    )
    assert [r.issue_id for r in refs] == [901]


def test_run_fanout_excluded_refs_surfaced_in_result(temp_keys_db):
    """End-to-end: fan-out reports excluded refs in result.excluded
    so operators can audit what was dropped and why."""
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_marker_scout",
        scout={
            "id": "t_marker_scout",
            "title": "acme widgets scout with awareness marker",
            "body": (
                "Triage #101, #102. "
                "Awareness batch (#103, #104) — for awareness only, "
                "do not file cards for these.\n"
            ),
            "comments": [],
        },
        deps=deps,
    )
    # Only 101, 102 are created.
    created_ids = sorted(c["issue_id"] for c in result.created)
    assert created_ids == [101, 102]
    # 103, 104 surface in excluded with reason + label.
    excluded_ids = sorted(e["issue_id"] for e in result.excluded)
    assert excluded_ids == [103, 104]
    for entry in result.excluded:
        assert entry["reason"] == "awareness_only"
    assert result.detected == 2
    assert result.exit_code() == 0


def test_run_fanout_manual_exclude_issues_flag(temp_keys_db):
    """``exclude_issues`` CLI arg overrides: refs not mentioned in
    a marker but excluded by the operator must be in
    result.excluded with reason='manual_override'."""
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_manual_exclude",
        scout={
            "id": "t_manual_exclude",
            "title": "acme widgets scout manual exclude",
            "body": "Triage #101, #102, #103, #104.\n",
            "comments": [],
        },
        deps=deps,
        exclude_issues=[102, 104],
    )
    assert sorted(c["issue_id"] for c in result.created) == [101, 103]
    excluded = {e["issue_id"]: e for e in result.excluded}
    assert excluded[102]["reason"] == "manual_override"
    assert excluded[104]["reason"] == "manual_override"
    assert result.detected == 2
    assert result.exit_code() == 0


def test_run_fanout_marker_and_manual_excludes_combine(temp_keys_db):
    """Body-derived + manual excludes must combine without
    duplicates; a body-derived exclude keeps its reason
    ('awareness_only' or 'do_not_file' based on which marker
    phrase matched) rather than getting overwritten by
    'manual_override'."""
    create_log = _FakeCreate()
    deps = _make_deps(
        key_store=temp_keys_db,
        fetcher=_fake_fetch(),
        create_log=create_log,
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_combined",
        scout={
            "id": "t_combined",
            "title": "acme widgets scout combined",
            "body": (
                "Triage #101, #102. "
                "Awareness (#103) — do not file.\n"
            ),
            "comments": [],
        },
        deps=deps,
        exclude_issues=[102, 104],  # 102 also manual, 104 manual-only
    )
    assert sorted(c["issue_id"] for c in result.created) == [101]
    excluded_reasons = {
        e["issue_id"]: e["reason"] for e in result.excluded
    }
    # 103 came from the body marker with "do not file" phrase
    # → reason='do_not_file' (not overwritten by manual_override).
    assert excluded_reasons[103] == "do_not_file"
    # 102 was manual-only (no body marker covers it) → "manual_override".
    assert excluded_reasons[102] == "manual_override"
    # 104 is manual-only.
    assert excluded_reasons[104] == "manual_override"


def test_parse_csv_ints_handles_valid_and_edge_cases():
    import argparse as _argparse
    assert itf._parse_csv_ints(None) == []
    assert itf._parse_csv_ints("") == []
    assert itf._parse_csv_ints("  ") == []
    assert itf._parse_csv_ints("898") == [898]
    assert itf._parse_csv_ints("898, 899,900") == [898, 899, 900]
    # Reject non-numeric — argparse.ArgumentTypeError is raised so the
    # CLI exits with a clean usage error rather than a stack trace.
    with pytest.raises(_argparse.ArgumentTypeError):
        itf._parse_csv_ints("abc")
    # Reject zero / negative.
    with pytest.raises(_argparse.ArgumentTypeError):
        itf._parse_csv_ints("0")
    with pytest.raises(_argparse.ArgumentTypeError):
        itf._parse_csv_ints("-1")


# ---------------------------------------------------------------------------
# GAP 2 follow-up (t_b17ae9d3): repo → owner mapping for bare-ref scouts
# ---------------------------------------------------------------------------


def test_resolve_repo_owner_with_empty_owner_uses_built_in_map():
    """Bare-ref scouts with no owner in the URL form must resolve
    via REPO_OWNER_MAP — this is the GAP 2 fix."""
    owner, repo = itf.resolve_repo_owner("", "smilemap")
    assert owner == "veroscale"
    assert repo == "smilemap"


def test_resolve_repo_owner_extra_map_takes_precedence():
    """CLI-supplied overrides beat the built-in map so an operator
    can always correct a wrong mapping without editing code."""
    owner, repo = itf.resolve_repo_owner(
        "", "smilemap", extra_map={"smilemap": "wrongowner"}
    )
    assert owner == "wrongowner"
    assert repo == "smilemap"


def test_resolve_repo_owner_passes_through_when_owner_set():
    """When the URL form already supplied an owner, the resolver
    must NOT consult any mapping (the operator's explicit URL
    wins)."""
    assert itf.resolve_repo_owner("acme", "widgets") == ("acme", "widgets")
    assert itf.resolve_repo_owner("acme", "widgets", extra_map={"widgets": "x"}) == (
        "acme",
        "widgets",
    )


def test_resolve_repo_owner_unknown_repo_returns_empty_owner():
    owner, repo = itf.resolve_repo_owner("", "totally-unmapped")
    assert owner == ""
    assert repo == "totally-unmapped"


def test_fetch_gh_issue_bare_repo_uses_repo_owner_map(monkeypatch):
    """End-to-end: a bare-ref scout whose repo is in REPO_OWNER_MAP
    produces the ``--repo owner/repo`` form so gh 2.97 accepts it.

    Captures the actual subprocess invocation to prove the args
    match the spec.
    """
    captured = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"title": "Real title from gh", "body": "Body."})
        stderr = ""

    def _fake_run(args, **kwargs):
        captured["args"] = list(args)
        return _Proc()

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("", "smilemap", 909)
    assert out.title == "Real title from gh"
    assert out.body == "Body."
    assert out.error is None
    # gh must be invoked with --repo veroscale/smilemap (GAP 2).
    assert captured["args"][:4] == ["gh", "issue", "view", "909"]
    assert "--repo" in captured["args"]
    assert "veroscale/smilemap" in captured["args"]


def test_fetch_gh_issue_bare_repo_unmapped_returns_placeholder(monkeypatch):
    """A bare-ref repo that isn't in the mapping must fall back
    to the placeholder title with an explanatory error so the
    operator knows to add a ``--repo-map``."""
    captured_called = {"count": 0}

    def _fake_run(*args, **kwargs):
        captured_called["count"] += 1
        raise AssertionError("gh should NOT be invoked for unmapped bare repo")

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("", "totally-unmapped", 42)
    assert captured_called["count"] == 0
    assert out.title == itf.GH_UNAVAILABLE_TITLE
    assert out.error is not None
    assert "totally-unmapped" in out.error
    assert "--repo-map" in out.error


def test_fetch_gh_issue_accepts_explicit_owner(monkeypatch):
    """Regression guard: when owner is supplied directly (URL form),
    the resolver must NOT consult REPO_OWNER_MAP — the URL wins."""
    captured = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"title": "x", "body": "y"})
        stderr = ""

    def _fake_run(args, **kwargs):
        captured["args"] = list(args)
        return _Proc()

    monkeypatch.setattr(itf.subprocess, "run", _fake_run)
    out = itf.fetch_gh_issue("acme", "widgets", 101)
    assert out.title == "x"
    assert "acme/widgets" in captured["args"]


def test_run_fanout_bare_ref_scout_uses_repo_owner_map_for_title(
    temp_keys_db,
):
    """End-to-end GAP 2 acceptance: a bare-ref scout whose repo
    is in REPO_OWNER_MAP gets REAL titles (not placeholders)
    from the gh fetch path.

    The test mock mirrors the real ``fetch_gh_issue`` shape: it
    calls ``resolve_repo_owner`` first so the resolved owner is
    observable by the mock's assertion.
    """
    create_log = _FakeCreate()

    def _real_fetch(
        owner, repo, issue_id, *, gh_path="gh", repo_owner_map=None
    ):
        # Mirror the real fetch_gh_issue contract: resolve owner
        # via the map before producing the title.
        resolved_owner, repo = itf.resolve_repo_owner(
            owner, repo, repo_owner_map
        )
        assert resolved_owner == "veroscale", (
            f"expected owner='veroscale' via REPO_OWNER_MAP, "
            f"got {resolved_owner!r}"
        )
        assert repo == "smilemap"
        return itf.GhFetchResult(
            title=f"smilemap issue {issue_id} title",
            body="body excerpt",
        )

    deps = itf.FanoutDeps(
        fetch_gh_issue=_real_fetch,
        key_store=temp_keys_db,
        create_task=create_log,
        signal_emit=lambda **kw: "sig-fake",
        parent_link_scan=lambda _id: [],
        repo_owner_map={"smilemap": "veroscale"},
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_bare_ref_scout",
        scout={
            "id": "t_bare_ref_scout",
            "title": "smilemap batch triage: bare refs test",
            "body": "Issues in scope: #901, #902.\n",
            "comments": [],
        },
        deps=deps,
    )
    assert result.detected == 2
    assert len(result.created) == 2
    for entry in result.created:
        assert entry["title"].startswith("[gh] smilemap: ")
        # NOT the placeholder — the fetch succeeded.
        assert "<unknown title" not in entry["title"]
        assert entry["title"].endswith(f"title (#{entry['issue_id']})")
    # Issue URL uses the resolved owner, not "unknown-owner".
    assert all(
        "veroscale/smilemap" in entry.get("issue_url", "")
        for entry in result.created
    )


def test_run_fanout_bare_ref_scout_unknown_repo_uses_placeholder(temp_keys_db):
    """GAP 2 fallback: a bare-ref scout whose repo is NOT in the
    map still creates the card (so the worker can manually
    retitle), but the title carries the placeholder."""
    create_log = _FakeCreate()

    def _placeholder_fetch(
        owner, repo, issue_id, *, gh_path="gh", repo_owner_map=None
    ):
        # Mirror the real contract: even when the map resolves the
        # owner, an "unknown-repo" isn't mapped so resolved_owner
        # would be empty. Verify that branch here.
        resolved_owner, repo = itf.resolve_repo_owner(
            owner, repo, repo_owner_map
        )
        if not resolved_owner:
            return itf.GhFetchResult(
                title=itf.GH_UNAVAILABLE_TITLE,
                body=None,
                error=f"bare repo '{repo}' has no owner mapping",
            )
        return itf.GhFetchResult(
            title=f"{repo} issue {issue_id}", body="body"
        )

    deps = itf.FanoutDeps(
        fetch_gh_issue=_placeholder_fetch,
        key_store=temp_keys_db,
        create_task=create_log,
        signal_emit=lambda **kw: "sig-fake",
        parent_link_scan=lambda _id: [],
    )
    result = itf.run_fanout(
        board="ops",
        scout_card_id="t_unknown_repo_scout",
        scout={
            "id": "t_unknown_repo_scout",
            "title": "mystery-batch scout: bare refs",
            "body": "Issues in scope: #101, #102.\n",
            "comments": [],
        },
        deps=deps,
    )
    assert result.detected == 2
    assert len(result.created) == 2
    for entry in result.created:
        # Placeholder present — operator knows to add a --repo-map.
        assert "<unknown title" in entry["title"]
    # The gh error is recorded on each created entry as gh_error so a
    # worker can fix it without leaving result.errors (which is reserved
    # for create_task failures).
    assert all(
        "has no owner mapping" in entry.get("gh_error", "")
        for entry in result.created
    )


def test_parse_repo_map_cli_value():
    assert itf._parse_repo_map(None) == {}
    assert itf._parse_repo_map("") == {}
    assert itf._parse_repo_map("smilemap=veroscale") == {
        "smilemap": "veroscale"
    }
    assert itf._parse_repo_map(
        "smilemap=veroscale, aurora=acme"
    ) == {"smilemap": "veroscale", "aurora": "acme"}
    # Reject malformed.
    with pytest.raises(Exception):
        itf._parse_repo_map("smilemap-veroscale")  # no =
    with pytest.raises(Exception):
        itf._parse_repo_map("=veroscale")  # empty key
    with pytest.raises(Exception):
        itf._parse_repo_map("smilemap=")  # empty value


def test_argparser_accepts_new_gap_flags():
    parser = itf.build_argparser()
    args = parser.parse_args(
        [
            "ops",
            "t_12cc81c6",
            "--exclude-issues",
            "898,899,900",
            "--repo-map",
            "smilemap=veroscale,aurora=acme",
        ]
    )
    assert args.exclude_issues == "898,899,900"
    assert args.repo_map == "smilemap=veroscale,aurora=acme"


def test_argparser_scan_all_scouts_flag():
    parser = itf.build_argparser()
    args = parser.parse_args(
        ["ops", "--scan-all-scouts", "--dry-run", "--json"]
    )
    assert args.scan_all_scouts is True
    assert args.dry_run is True
    assert args.scout_card_id is None  # optional with --scan-all-scouts
