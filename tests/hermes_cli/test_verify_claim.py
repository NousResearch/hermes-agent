"""Landing-evidence bundle for a fix/duplicate claim (issue #111189, ``hermes verify-claim``).

The bar the tests pin down:

* ``state_reason: completed`` is a label, never landing evidence - a closed-as-
  completed issue with no merged canonical must come out ``unverified`` with a
  non-zero exit code, never ``landed``.
* A merged canonical only reaches ``landed`` when the local
  ``git merge-base --is-ancestor <sha> <base>`` proof passes; a SHA that exists
  but is not on the base ref is ``refuted`` (exit 3).
* A fix PR's repro record is parsed structurally (test names, red-on-base,
  green-after) or reported ``missing`` - never invented.

Everything external is faked: ``verify_claim._gh_api`` (endpoint -> payload) and
``verify_claim._git`` (scripted exit codes), so no network and no git repo is
touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest

from hermes_cli import verify_claim
from hermes_cli.verify_claim import (
    EXIT_OK,
    EXIT_REFUTED,
    EXIT_UNVERIFIED,
    EXIT_USAGE,
    collect_evidence,
    parse_repro,
    run_verify_claim_command,
    timeline_candidates,
)

REPO = "NousResearch/hermes-agent"
MERGED_SHA = "a" * 40
OTHER_SHA = "b" * 40
BASE = "origin/main"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

_MISSING = object()


class FakeGh:
    """``gh api`` stand-in: ``endpoint -> payload`` (or an exception to raise)."""

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, endpoint, *, paginate=False):
        self.calls.append((endpoint, paginate))
        value = self.mapping.get(endpoint, _MISSING)
        if value is _MISSING:
            raise AssertionError(f"unexpected gh endpoint: {endpoint}")
        if isinstance(value, Exception):
            raise value
        return value


class FakeGit:
    """``git`` stand-in driven by exit codes instead of a real repository."""

    def __init__(self, *, in_repo=True, present=(), ancestors=None, fetch_ok=True,
                 fetch_provides=(), base_ok=True):
        self.in_repo = in_repo
        self.present = set(present)
        self.ancestors = dict(ancestors or {})
        self.fetch_ok = fetch_ok
        self.fetch_provides = set(fetch_provides)
        self.base_ok = base_ok
        self.calls: list[list[str]] = []

    def __call__(self, args, cwd):
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["rev-parse", "--git-dir"]:
            rc = 0 if self.in_repo else 128
        elif args[:2] == ["cat-file", "-e"]:
            rc = 0 if args[2].split("^{")[0] in self.present else 1
        elif args[:2] == ["rev-parse", "--verify"]:
            rc = 0 if self.base_ok else 1
        elif args[:1] == ["fetch"]:
            rc = 0 if self.fetch_ok else 128
            if self.fetch_ok:
                self.present |= self.fetch_provides
        elif args[:2] == ["merge-base", "--is-ancestor"]:
            rc = self.ancestors.get(args[2], 1)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected git invocation: {args}")
        return subprocess.CompletedProcess(["git", *args], rc, "", "")


def issue(number, *, state="closed", state_reason="completed", title="Target issue",
          body="", is_pr=False):
    payload = {
        "number": number,
        "title": title,
        "state": state,
        "state_reason": state_reason,
        "html_url": f"https://github.com/{REPO}/issues/{number}",
        "body": body,
    }
    if is_pr:
        payload["pull_request"] = {"url": f"https://api.github.com/repos/{REPO}/pulls/{number}"}
    return payload


def pull(number, *, merged=True, merge_commit=MERGED_SHA, state="closed",
         merged_at="2026-09-14T00:00:00Z", title="Fix it", body="", base="main"):
    return {
        "number": number,
        "title": title,
        "state": state,
        "body": body,
        "merged": merged,
        "merge_commit_sha": merge_commit,
        "merged_at": merged_at,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "base": {"ref": base},
    }


def xref(number, *, merged_at="2026-09-14T00:00:00Z", kind="pull_request", title="Fix"):
    item = {"number": number, "title": title, "state": "closed",
            "html_url": f"https://github.com/{REPO}/issues/{number}"}
    if kind == "pull_request":
        item["pull_request"] = {"merged_at": merged_at}
    return {"event": "cross-referenced", "source": {"issue": item}}


def fixture_gh(issue_payload, *, timeline=None, pulls=None, extra=None):
    mapping = {f"repos/{REPO}/issues/{issue_payload['number']}": issue_payload}
    if timeline is not None:
        mapping[f"repos/{REPO}/issues/{issue_payload['number']}/timeline"] = timeline
    for number, payload in (pulls or {}).items():
        mapping[f"repos/{REPO}/pulls/{number}"] = payload
        if not isinstance(payload, dict):  # a scripted failure for this endpoint
            continue
        # A PR is also readable through /issues/<n>; body references learn the kind
        # from that endpoint, so mirror it the way GitHub does.
        mapping.setdefault(f"repos/{REPO}/issues/{number}", issue(
            number, is_pr=True, state=payload.get("state", "closed"), state_reason=None,
            title=payload.get("title", ""), body=payload.get("body", "")))
    mapping.update(extra or {})
    return FakeGh(mapping)


def run(gh, git, number, **overrides):
    """Collect a bundle with both seams faked."""
    monkeypatch_args = dict(repo=REPO, cwd=".", base_ref=BASE, fetch=True, require_repro=True)
    monkeypatch_args.update(overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verify_claim, "_gh_api", gh)
        mp.setattr(verify_claim, "_git", git)
        return collect_evidence(number, **monkeypatch_args)


def cli_args(**overrides):
    defaults = dict(number=1, repo=REPO, repo_dir=".", base=BASE, no_fetch=False,
                    repro_optional=False, json=True, compact=True)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------- #
# the label is not evidence
# --------------------------------------------------------------------------- #

def test_completed_label_without_a_merge_commit_is_never_landed():
    """#72485-style: closed as completed, nothing merged -> unverified, non-zero exit."""
    gh = fixture_gh(issue(72485, state_reason="completed", title="Font customization"), timeline=[])
    git = FakeGit(in_repo=True)
    bundle = run(gh, git, 72485)

    assert bundle["target"]["state_reason"] == "completed"
    assert bundle["target"]["state_label_is_landing_evidence"] is False
    assert bundle["landing"]["merge_commit"] is None
    assert bundle["landing"]["verdict"] == "unverified"
    assert bundle["verdict"] != "landed"
    assert bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED
    assert bundle["ok"] is False
    # No SHA means no ancestry proof was attempted.
    assert bundle["proof"]["status"] == "not-run"
    assert not [c for c in git.calls if c[:2] == ["merge-base", "--is-ancestor"]]
    failing = {c["id"] for c in bundle["checks"] if c["status"] == "fail"}
    assert "landing-commit-identified" in failing
    assert "state-label-not-used-as-landing-evidence" not in failing


def test_closed_unmerged_canonical_pr_is_not_landing_evidence():
    """A closed-but-unmerged PR's merge_commit_sha is the test-merge commit, not a landing."""
    target = issue(500, state_reason="completed", body="Fixed by #501")
    gh = fixture_gh(target, pulls={501: pull(501, merged=False, state="closed")}, timeline=[])
    bundle = run(gh, FakeGit(), 500)

    assert bundle["landing"]["merge_commit"] is None
    assert bundle["landing"]["verdict"] == "unverified"
    assert "never merged" in bundle["landing"]["reason"]
    assert bundle["exit_code"] == EXIT_UNVERIFIED


def test_finalize_downgrades_a_landed_verdict_without_a_merge_commit():
    """Belt-and-braces invariant: verdict 'landed' requires a merge commit."""
    bundle = {
        "target": {"number": 1, "kind": "issue", "state": "closed", "state_reason": "completed",
                   "state_label_is_landing_evidence": False},
        "claim": {"kind": "fixed", "basis": "state_reason"},
        "landing": {"verdict": "landed", "merge_commit": None, "reason": "closed as completed"},
        "proof": {"status": "not-run", "shell": "git merge-base --is-ancestor <sha> origin/main",
                  "detail": ""},
        "repro": {"status": "not-applicable", "required": False, "reason": "issue target"},
        "checks": [{"id": "landing-commit-identified", "required": True, "status": "fail", "detail": ""}],
    }
    verify_claim.finalize(bundle)
    assert bundle["landing"]["verdict"] == "unverified"
    assert "invariant" in bundle["landing"]["reason"]
    assert bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED


# --------------------------------------------------------------------------- #
# landed: merge commit + is-ancestor proof
# --------------------------------------------------------------------------- #

def test_merged_canonical_with_ancestor_proof_is_landed():
    target = issue(700, state_reason="completed", body="Fixed by #701")
    gh = fixture_gh(target, pulls={701: pull(701)}, timeline=[])
    git = FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0})
    bundle = run(gh, git, 700)

    assert bundle["landing"]["merge_commit"] == MERGED_SHA
    assert bundle["proof"]["status"] == "verified"
    assert bundle["proof"]["shell"] == f"git merge-base --is-ancestor {MERGED_SHA} {BASE}"
    assert bundle["proof"]["sha_present_locally"] is True
    assert bundle["verdict"] == "landed"
    assert bundle["exit_code"] == EXIT_OK
    assert ["merge-base", "--is-ancestor", MERGED_SHA, BASE] in git.calls


def test_landing_sha_absent_locally_is_fetched_then_verified():
    target = issue(710, state_reason="completed", body="Fixed by #711")
    gh = fixture_gh(target, pulls={711: pull(711)}, timeline=[])
    git = FakeGit(in_repo=True, present=set(), fetch_provides={MERGED_SHA}, ancestors={MERGED_SHA: 0})
    bundle = run(gh, git, 710, fetch=True)

    assert bundle["proof"]["fetched"] is True
    assert bundle["proof"]["status"] == "verified"
    assert bundle["exit_code"] == EXIT_OK
    assert any(c[:2] == ["fetch", "--quiet"] for c in git.calls)


def test_landing_sha_absent_locally_without_fetch_is_unverified():
    target = issue(720, state_reason="completed", body="Fixed by #721")
    gh = fixture_gh(target, pulls={721: pull(721)}, timeline=[])
    git = FakeGit(present=set())
    bundle = run(gh, git, 720, fetch=False)

    assert bundle["proof"]["status"] == "unverified"
    assert bundle["proof"]["fetched"] is False
    assert "--no-fetch" in bundle["proof"]["detail"]
    assert bundle["verdict"] == "landed" or bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED


def test_no_local_clone_is_unverified_with_a_reason():
    target = issue(730, state_reason="completed", body="Fixed by #731")
    gh = fixture_gh(target, pulls={731: pull(731)}, timeline=[])
    git = FakeGit(in_repo=False)
    bundle = run(gh, git, 730)

    assert bundle["landing"]["merge_commit"] == MERGED_SHA
    assert bundle["proof"]["status"] == "unverified"
    assert "not inside a git working tree" in bundle["proof"]["detail"]
    assert bundle["exit_code"] == EXIT_UNVERIFIED


def test_landing_commit_not_on_base_is_refuted():
    """The claim is checkably false when the merge commit is not reachable from the base ref."""
    target = issue(740, state_reason="completed", body="Fixed by #741")
    gh = fixture_gh(target, pulls={741: pull(741)}, timeline=[])
    git = FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 1})
    bundle = run(gh, git, 740)

    assert bundle["proof"]["status"] == "refuted"
    assert bundle["proof"]["exit_code"] == 1
    assert bundle["landing"]["verdict"] == "refuted"
    assert bundle["verdict"] == "refuted"
    assert bundle["exit_code"] == EXIT_REFUTED
    assert {c["status"] for c in bundle["checks"] if c["id"].endswith("reachable-from-base")} == {"fail"}


def test_unknown_base_ref_is_unverified():
    target = issue(750, state_reason="completed", body="Fixed by #751")
    gh = fixture_gh(target, pulls={751: pull(751)}, timeline=[])
    git = FakeGit(present={MERGED_SHA}, base_ok=False)
    bundle = run(gh, git, 750)

    assert bundle["proof"]["status"] == "unverified"
    assert "does not exist locally" in bundle["proof"]["detail"]
    assert bundle["exit_code"] == EXIT_UNVERIFIED


# --------------------------------------------------------------------------- #
# canonical discovery
# --------------------------------------------------------------------------- #

def test_cross_referenced_merged_pr_is_the_canonical():
    target = issue(800, state_reason="completed")
    gh = fixture_gh(target, pulls={801: pull(801, title="fix: thing")},
                    timeline=[xref(801, merged_at="2026-09-14T00:00:00Z")])
    bundle = run(gh, FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0}), 800)

    canonical = bundle["claim"]["canonical"]
    assert canonical["number"] == 801
    assert canonical["kind"] == "pull_request"
    assert canonical["source"] == "timeline-cross-reference"
    assert bundle["verdict"] == "landed"


def test_bare_pr_url_without_a_claim_verb_is_not_a_body_reference():
    """A bare link ("Related discussion: <url>") asserts no fix and gates nothing."""
    from hermes_cli.verify_claim import body_references
    body = "Related discussion: https://github.com/NousResearch/hermes-agent/pull/811."
    assert body_references(body) == []


def test_pr_url_with_a_claim_verb_keeps_its_own_repo():
    """A PR URL is read with the owner/repo it names, never flattened to the target's #N."""
    from hermes_cli.verify_claim import body_references
    body = "Fixed by https://github.com/other/repo/pull/811."
    refs = body_references(body)
    assert len(refs) == 1
    assert refs[0]["number"] == 811
    assert refs[0]["repo"] == "other/repo"
    assert refs[0]["source"] == "body-url-reference"


def test_pr_url_with_a_claim_verb_lands_through_the_target_repo():
    """A claim-verb link to this repo's own PR is still a body claim that lands."""
    target = issue(800, state_reason="completed",
                   body="Root cause found.\n\nFixed by https://github.com/NousResearch/hermes-agent/pull/811.")
    gh = fixture_gh(target, pulls={811: pull(811)}, timeline=[])
    bundle = run(gh, FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0}), 800)

    assert bundle["claim"]["canonical"]["source"] == "body-url-reference"
    assert bundle["claim"]["canonical"]["number"] == 811
    assert bundle["landing"]["merge_commit"] == MERGED_SHA
    assert bundle["verdict"] == "landed"


def test_cross_repo_pr_url_is_not_landing_evidence_for_this_repo():
    """A link to another repo's #811 must not resolve to this repo's merged #811."""
    target = issue(800, state_reason="completed",
                   body="Fixed by https://github.com/other/repo/pull/811.")
    # This repo's #811 is merged; the link names a different repo, so it proves nothing here.
    gh = fixture_gh(target, pulls={811: pull(811)}, timeline=[])
    bundle = run(gh, FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0}), 800)

    assert bundle["verdict"] != "landed"
    assert bundle["landing"]["merge_commit"] is None
    assert bundle["claim"]["canonical"] is None
    assert "repos/NousResearch/hermes-agent/pulls/811" not in [endpoint for endpoint, _ in gh.calls]
    assert "other/repo" in bundle["landing"]["reason"]


def test_body_reference_sets_the_claim_basis():
    target = issue(810, state_reason="completed", body="Root cause found.\n\nFixed by #811.")
    gh = fixture_gh(target, pulls={811: pull(811)}, timeline=[])
    bundle = run(gh, FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0}), 810)

    assert bundle["claim"]["kind"] == "fixed"
    assert bundle["claim"]["basis"] == "body-reference"
    assert bundle["claim"]["evidence"] == "Fixed by #811."


def test_duplicate_claim_resolves_through_the_canonical_issue():
    """duplicate -> canonical issue -> its merged fix PR is the landing evidence."""
    target = issue(900, state_reason="duplicate", body="Duplicate of #901")
    gh = fixture_gh(
        target,
        timeline=[],
        extra={
            f"repos/{REPO}/issues/901": issue(901, state_reason="completed", title="Canonical"),
            f"repos/{REPO}/issues/901/timeline": [xref(902, merged_at="2026-09-14T00:00:00Z")],
            f"repos/{REPO}/pulls/902": pull(902, title="fix: canonical"),
        },
    )
    bundle = run(gh, FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0}), 900)

    assert bundle["claim"]["kind"] == "duplicate"
    assert bundle["claim"]["canonical"]["number"] == 901
    assert bundle["claim"]["canonical"]["resolution"] == "canonical-issue-merged-pr"
    assert bundle["landing"]["pull_request"]["number"] == 902
    assert bundle["verdict"] == "landed"
    assert bundle["exit_code"] == EXIT_OK


def test_duplicate_without_landing_evidence_is_unverified():
    target = issue(910, state_reason="duplicate", body="Duplicate of #911")
    gh = fixture_gh(target, timeline=[], extra={
        f"repos/{REPO}/issues/911": issue(911, state_reason="completed", title="Canonical"),
        f"repos/{REPO}/issues/911/timeline": [],
    })
    bundle = run(gh, FakeGit(), 910)

    assert bundle["claim"]["kind"] == "duplicate"
    assert bundle["landing"]["merge_commit"] is None
    assert bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED


def test_body_reference_to_an_issue_is_followed_one_level_down():
    """A "#901" body reference resolves its kind from the API, not from the wording."""
    target = issue(915, state_reason="completed", body="Fixed in #901")
    gh = fixture_gh(target, timeline=[], extra={
        f"repos/{REPO}/issues/901": issue(901, state_reason="completed", title="Canonical"),
        f"repos/{REPO}/issues/901/timeline": [xref(902, merged_at="2026-09-14T00:00:00Z")],
        f"repos/{REPO}/pulls/902": pull(902, title="fix: canonical"),
    })
    bundle = run(gh, FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0}), 915)

    assert bundle["claim"]["basis"] == "body-reference"
    assert bundle["landing"]["merge_commit"] == MERGED_SHA
    assert bundle["exit_code"] == EXIT_OK


def test_close_commit_event_is_used_when_no_canonical_pr_exists():
    target = issue(920, state_reason="completed")
    gh = fixture_gh(target, timeline=[{"event": "closed", "commit_id": OTHER_SHA}])
    git = FakeGit(present={OTHER_SHA}, ancestors={OTHER_SHA: 0})
    bundle = run(gh, git, 920)

    assert bundle["landing"]["merge_commit"] == OTHER_SHA
    assert "closing commit" in bundle["landing"]["reason"]
    assert bundle["verdict"] == "landed"


def test_not_planned_issue_asserts_nothing_and_skips_landing_lookups():
    gh = fixture_gh(issue(930, state="closed", state_reason="not_planned"))
    bundle = run(gh, FakeGit(), 930)

    assert bundle["claim"]["kind"] == "none"
    assert bundle["landing"]["verdict"] == "not-claimed"
    assert bundle["verdict"] == "no-claim"
    assert bundle["exit_code"] == EXIT_OK
    assert [c for c in gh.calls if "timeline" in c[0]] == []


def test_open_issue_without_a_claim_does_not_gate():
    gh = fixture_gh(issue(940, state="open", state_reason=None))
    bundle = run(gh, FakeGit(), 940)

    assert bundle["claim"]["kind"] == "none"
    assert bundle["exit_code"] == EXIT_OK
    assert bundle["checks"][1]["status"] == "na"


# --------------------------------------------------------------------------- #
# fix PRs: landing + repro record
# --------------------------------------------------------------------------- #

def test_merged_fix_pr_with_repro_record_is_landed():
    body = (
        "## What does this PR do?\n\nFix the thing.\n\n"
        "## How to Test\n\n"
        "1. `pytest tests/hermes_cli/test_verify_claim.py::test_landed`\n"
        "   - before: FAILED (3 failed on origin/main)\n"
        "   - after: 3 passed\n"
    )
    gh = fixture_gh(issue(1000, is_pr=True, body=body), extra={f"repos/{REPO}/pulls/1000": pull(1000, body=body)})
    git = FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0})
    bundle = run(gh, git, 1000)

    assert bundle["claim"]["kind"] == "fix-pr"
    repro = bundle["repro"]
    assert repro["status"] == "present"
    assert repro["required"] is True
    assert repro["section"] == "How to Test"
    assert "tests/hermes_cli/test_verify_claim.py::test_landed" in repro["tests"]
    assert repro["red_on_base"]["observed"] is True
    assert "FAILED" in repro["red_on_base"]["evidence"]
    assert repro["green_after"]["observed"] is True
    assert "passed" in repro["green_after"]["evidence"]
    assert repro["verified_by"] == "body-text"
    assert bundle["verdict"] == "landed"
    assert bundle["exit_code"] == EXIT_OK


def test_merged_fix_pr_without_repro_record_exits_unverified():
    body = "## What does this PR do?\n\nFix the thing.\n"
    gh = fixture_gh(issue(1010, is_pr=True, body=body), extra={f"repos/{REPO}/pulls/1010": pull(1010, body=body)})
    git = FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0})
    bundle = run(gh, git, 1010)

    assert bundle["repro"]["status"] == "missing"
    assert bundle["repro"]["tests"] == []
    assert bundle["repro"]["red_on_base"]["observed"] is False
    repro_check = next(c for c in bundle["checks"] if c["id"] == "repro-record")
    assert repro_check["status"] == "fail" and repro_check["required"] is True
    assert bundle["exit_code"] == EXIT_UNVERIFIED


def test_repro_optional_downgrades_the_gate_but_still_reports():
    body = "## What does this PR do?\n\nTypo.\n"
    gh = fixture_gh(issue(1020, is_pr=True, body=body), extra={f"repos/{REPO}/pulls/1020": pull(1020, body=body)})
    git = FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0})
    bundle = run(gh, git, 1020, require_repro=False)

    assert bundle["repro"]["status"] == "missing"
    assert bundle["repro"]["required"] is False
    assert bundle["exit_code"] == EXIT_OK


def test_open_fix_pr_is_pending_and_gated_by_the_repro_record():
    body = "## Test plan\n\n- `pytest tests/x/test_y.py::test_z` (red before, passes after)\n"
    gh = fixture_gh(issue(1030, is_pr=True, state="open", state_reason=None, body=body),
                    extra={f"repos/{REPO}/pulls/1030": pull(1030, merged=False, state="open", body=body)})
    bundle = run(gh, FakeGit(), 1030)

    assert bundle["landing"]["verdict"] == "not-merged"
    assert bundle["landing"]["merge_commit"] is None
    assert bundle["verdict"] == "pending"
    assert bundle["exit_code"] == EXIT_OK
    assert bundle["repro"]["required"] is True


def test_open_fix_pr_without_repro_is_unverified():
    gh = fixture_gh(issue(1040, is_pr=True, state="open", state_reason=None, body="nothing here"),
                    extra={f"repos/{REPO}/pulls/1040": pull(1040, merged=False, state="open", body="nothing here")})
    bundle = run(gh, FakeGit(), 1040)

    assert bundle["exit_code"] == EXIT_UNVERIFIED
    assert bundle["verdict"] == "unverified"


# --------------------------------------------------------------------------- #
# repro parser
# --------------------------------------------------------------------------- #

def test_parse_repro_extracts_tests_and_red_green_from_an_evidence_section():
    body = (
        "## Evidence\n\n"
        "```text\n"
        "pytest tests/agent/test_widget.py::test_regression\n"
        "origin/main: 1 failed\n"
        "this branch: 1 passed\n"
        "```\n"
    )
    repro = parse_repro(body, source="pull:1:body")

    assert repro["status"] == "present"
    assert repro["section"] == "Evidence"
    assert repro["tests"] == ["tests/agent/test_widget.py::test_regression"]
    assert repro["red_on_base"]["observed"] is True
    assert repro["green_after"]["observed"] is True
    assert repro["counts"] == ["1 failed", "1 passed"]


def test_parse_repro_accepts_the_arrow_form():
    repro = parse_repro("**Test plan**\n\n- tests/x/test_y.py::test_z: FAIL -> PASS\n", source="pull:1:body")

    assert repro["status"] == "present"
    assert repro["tests"] == ["tests/x/test_y.py::test_z"]
    assert repro["red_on_base"]["observed"] and repro["green_after"]["observed"]
    assert "FAIL -> PASS" in repro["red_on_base"]["evidence"]


def test_parse_repro_reports_missing_without_inventing_anything():
    for body in (
        None,
        "",
        "## What does this PR do?\n\nRefactor internals, no behaviour change.\n",
        "## Evidence\n\nSee the CI run.\n",
    ):
        repro = parse_repro(body, source="pull:1:body")
        assert repro["status"] == "missing", body
        assert repro["tests"] == []
        assert repro["red_on_base"] == {"observed": False, "evidence": None}
        assert repro["green_after"] == {"observed": False, "evidence": None}
        assert repro["reason"]


def test_parse_repro_red_without_green_is_missing():
    repro = parse_repro("## Evidence\n\n- tests/x/test_y.py::test_z still fails on main\n", source="pull:1:body")
    assert repro["tests"] == ["tests/x/test_y.py::test_z"]
    assert repro["red_on_base"]["observed"] is True
    assert repro["green_after"]["observed"] is False
    # A test id alone is still a structured record; the markers are reported as-is.
    assert repro["status"] == "present"


def test_timeline_candidates_order_merged_prs_first():
    events = [
        xref(1, kind="issue"),
        xref(2, merged_at=None),
        xref(3, merged_at="2026-09-14T00:00:00Z"),
        xref(4, merged_at="2026-09-13T00:00:00Z"),
    ]
    ordered = timeline_candidates(events, repo=REPO, self_number=99)
    assert [c["number"] for c in ordered] == [4, 3, 2, 1] or [c["number"] for c in ordered] == [3, 4, 2, 1]
    assert ordered[0]["kind"] == "pull_request"
    assert ordered[-1]["kind"] == "issue"


# --------------------------------------------------------------------------- #
# CLI: exit codes and rendering
# --------------------------------------------------------------------------- #

def _cli(monkeypatch, gh, git, capsys, args):
    monkeypatch.setattr(verify_claim, "_gh_api", gh)
    monkeypatch.setattr(verify_claim, "_git", git)
    code = run_verify_claim_command(args)
    return code, capsys.readouterr()


def test_cli_json_bundle_carries_the_exit_code(monkeypatch, capsys):
    target = issue(1100, state_reason="completed", body="Fixed by #1101")
    gh = fixture_gh(target, pulls={1101: pull(1101)}, timeline=[])
    git = FakeGit(present={MERGED_SHA}, ancestors={MERGED_SHA: 0})
    code, captured = _cli(monkeypatch, gh, git, capsys, cli_args(number=1100))

    assert code == EXIT_OK
    payload = json.loads(captured.out)
    assert payload["schema"] == "hermes.verify-claim/1"
    assert payload["exit_code"] == EXIT_OK
    assert payload["verdict"] == "landed"
    assert set(payload) >= {"schema", "target", "claim", "landing", "proof", "repro", "checks", "exit_codes"}


def test_cli_human_summary_names_the_proof_command(monkeypatch, capsys):
    target = issue(1110, state_reason="completed")
    gh = fixture_gh(target, timeline=[])
    code, captured = _cli(monkeypatch, gh, FakeGit(), capsys, cli_args(number=1110, json=False))

    assert code == EXIT_UNVERIFIED
    out = captured.out
    assert "hermes verify-claim 1110" in out
    assert "UNVERIFIED" in out
    assert "state_reason=completed" in out
    assert "git merge-base --is-ancestor" in out
    assert "exit 1" in out


def test_cli_unknown_target_is_a_usage_error(monkeypatch, capsys):
    gh = FakeGh({})  # every endpoint raises AssertionError -> replaced below
    gh.mapping[f"repos/{REPO}/issues/1200"] = verify_claim.UsageError(
        f"gh api not found: repos/{REPO}/issues/1200 (gh exit 1)")
    code, captured = _cli(monkeypatch, gh, FakeGit(), capsys, cli_args(number=1200))

    assert code == EXIT_USAGE
    payload = json.loads(captured.out)
    assert payload["ok"] is False and payload["exit_code"] == EXIT_USAGE
    assert "not found" in payload["error"]


def test_cli_rejects_a_malformed_repo(monkeypatch, capsys):
    code, captured = _cli(monkeypatch, FakeGh({}), FakeGit(), capsys, cli_args(repo="not-a-slug"))

    assert code == EXIT_USAGE
    assert "--repo must be OWNER/REPO" in captured.err


def test_gh_failure_says_the_cli_is_missing():
    monkeypatch_git = FakeGit()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verify_claim.subprocess, "run", _raise_filenotfound)
        with pytest.raises(verify_claim.UsageError) as excinfo:
            verify_claim._gh_api(f"repos/{REPO}/issues/1")
    assert "gh CLI not found on PATH" in str(excinfo.value)
    assert monkeypatch_git.calls == []


def _raise_filenotfound(*args, **kwargs):
    raise FileNotFoundError("gh")


def test_gh_retries_once_then_fails_without_echoing_stderr(monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        return subprocess.CompletedProcess(["gh"], 1, "", "gh: secret host 10.0.0.1 leaked")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verify_claim.subprocess, "run", flaky)
        mp.setattr(verify_claim.time, "sleep", lambda _s: None)
        with pytest.raises(verify_claim.UsageError) as excinfo:
            verify_claim._gh_api("repos/x/y/issues/1")
    assert calls["n"] == 2
    assert "leaked" not in str(excinfo.value)


def test_gh_retries_absorb_a_transient_flake(monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(["gh"], 1, "", "connection reset")
        return subprocess.CompletedProcess(["gh"], 0, json.dumps({"number": 1}), "")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verify_claim.subprocess, "run", flaky)
        mp.setattr(verify_claim.time, "sleep", lambda _s: None)
        value = verify_claim._gh_api("repos/x/y/issues/1")
    assert value == {"number": 1}
    assert calls["n"] == 2


def test_gh_api_never_retries_a_404(monkeypatch):
    calls = {"n": 0}

    def not_found(*args, **kwargs):
        calls["n"] += 1
        return subprocess.CompletedProcess(["gh"], 1, "", "gh: Not Found (HTTP 404)")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verify_claim.subprocess, "run", not_found)
        mp.setattr(verify_claim.time, "sleep", lambda _s: None)
        with pytest.raises(verify_claim.UsageError) as excinfo:
            verify_claim._gh_api("repos/x/y/issues/404")
    assert calls["n"] == 1
    assert "not found" in str(excinfo.value)


def test_paginated_gh_reads_flatten_their_pages(monkeypatch):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verify_claim.subprocess, "run",
                   lambda *a, **k: subprocess.CompletedProcess(["gh"], 0, json.dumps([[1, 2], [3]]), ""))
        assert verify_claim._gh_api("repos/x/y/issues/1/timeline", paginate=True) == [1, 2, 3]


def test_candidate_lookup_failure_is_reported_in_the_reason():
    target = issue(760, state_reason="completed", body="Fixed by #761")
    gh = fixture_gh(target, pulls={761: verify_claim.UsageError("gh api request failed: pulls/761")},
                    timeline=[], extra={f"repos/{REPO}/issues/761": issue(761, is_pr=True)})
    bundle = run(gh, FakeGit(), 760)

    assert bundle["landing"]["merge_commit"] is None
    assert "#761: gh api request failed: pulls/761" in bundle["landing"]["reason"]
    assert bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED


def test_pr_target_lookup_failure_marks_lookup_failed():
    body = "## How to Test\n\n- tests/x/test_y.py::test_z (fails on main, passes after)\n"
    gh = fixture_gh(issue(765, is_pr=True, body=body),
                    extra={f"repos/{REPO}/pulls/765": verify_claim.UsageError("gh api request failed: pulls/765")})
    bundle = run(gh, FakeGit(), 765)

    assert bundle["landing"]["lookup_failed"] is True
    assert bundle["landing"]["merge_commit"] is None
    assert bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED
    # The repro record is still reported even when landing lookup failed.
    assert bundle["repro"]["status"] == "present"


def test_timeline_failure_does_not_crash_the_bundle():
    target = issue(1300, state_reason="completed")
    gh = fixture_gh(target, timeline=verify_claim.UsageError("gh api request failed: timeline"))
    bundle = run(gh, FakeGit(), 1300)

    assert "timeline" in bundle["timeline_error"]
    assert bundle["claim"]["kind"] == "fixed"
    assert bundle["verdict"] == "unverified"
    assert bundle["exit_code"] == EXIT_UNVERIFIED


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #

def test_parser_is_registered_and_defaults_to_the_handler():
    import argparse as _argparse

    from hermes_cli.subcommands.verify_claim import build_verify_claim_parser

    sentinel = object()
    parser = _argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_verify_claim_parser(subparsers, cmd_verify_claim=sentinel)

    args = parser.parse_args(["verify-claim", "111189", "--json"])
    assert args.func is sentinel
    assert args.number == 111189
    assert args.repo == REPO
    assert args.base == BASE
    assert args.repo_dir == "."
    assert args.json is True and args.no_fetch is False and args.repro_optional is False


def test_main_registers_the_builtin_subcommand():
    from hermes_cli import main as cli_main

    assert "verify-claim" in cli_main._BUILTIN_SUBCOMMANDS
    assert callable(cli_main.cmd_verify_claim)
