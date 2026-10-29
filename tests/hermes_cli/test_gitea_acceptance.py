"""Pure Gitea Actions acceptance (``hermes_cli.gitea_acceptance``).

The fixture is a real Gitea 1.2x API capture of a merged pull request whose
required job ran on the pinned runner, sanitized (host, owner, users, workflow
bytes and hash replaced). Review listings and production-grade protection values
(``required_approvals``/``block_on_rejected_reviews``) are SYNTHETIC: the capture
had no reviews and a qualification-only protection rule.
"""
import copy
import json
from pathlib import Path

import pytest

from hermes_cli.gitea_acceptance import assess, gate_receipt, parse_pr_url, validate_policy

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "gitea_acceptance_pr8.json").read_text())
SHA = "a3d67bdc249e00ec13576ceee99c20cee4e33aba"
RP = "/repos/acme/widget"
PR, PROT = RP + "/pulls/8", RP + "/branch_protections/rootless-v1"
RUNS, JOBS = RP + "/actions/runs?limit=50&page=1", RP + "/actions/runs/8/jobs?limit=50"
REVIEWS = RP + "/pulls/8/reviews?limit=50&page=1"
ORIGIN = "https://forge.example.test"


def _policy(qualification_only):
    raw = FIXTURE["policy"]
    return {"policy_id": "widget-v1", "qualification_only": qualification_only, "repo": raw["repo"],
            "repository_id": raw["repository_id"], "base": raw["base"], "runner_ids": [raw["runner_id"]],
            "mergers": [{"login": "merge-bot", "id": 4}], "workflows": copy.deepcopy(raw["workflows"])}


def _review(review_id=31, login="bob", user_id=7, state="APPROVED", commit=SHA, **extra):
    return {"id": review_id, "user": {"id": user_id, "login": login}, "state": state, "commit_id": commit,
            "stale": False, "dismissed": False, "official": True, "submitted_at": "2000-01-01T00:00:00Z", **extra}


class Forge:
    def __init__(self, production=False):
        self.rows = copy.deepcopy(FIXTURE["responses"])
        self.reads = {}
        self.mutate = {}
        if production:
            self.rows[PROT].update(required_approvals=1, block_on_rejected_reviews=True)
            self.rows[REVIEWS] = [_review()]

    def __call__(self, path):
        self.reads[path] = self.reads.get(path, 0) + 1
        if path not in self.rows:
            raise KeyError(path)
        row = copy.deepcopy(self.rows[path])
        hook = self.mutate.get((path, self.reads[path]))
        return hook(row) if hook else row


def run(forge, policy, **kwargs):
    kwargs.setdefault("custody_ok", True)
    return assess(forge, policy, 8, **kwargs)


# --- qualification: the captured evidence itself ---

def test_real_capture_passes_qualification_and_says_so():
    receipt = run(Forge(), _policy(True))
    assert receipt["ok"] and receipt["classification"] == "success"
    assert receipt["qualification_only"] is True and receipt["policy_id"] == "widget-v1"
    assert receipt["checks"] == [{"name": "Rootless CI / rootless-verify (pull_request)", "id": 11, "run_id": 8,
                                  "run_attempt": 3, "runner_id": 2, "head_sha": SHA, "classification": "success",
                                  "completed_at": "2026-10-13T20:28:37Z", "run_completed_at": "2026-10-13T20:28:37Z"}]
    assert "merge" not in receipt and "review" not in receipt


def test_qualification_may_assess_an_open_pr_but_not_a_closed_unmerged_one():
    forge = Forge()
    forge.rows[PR].update(state="open", merged=False, merged_by=None, merge_commit_sha=None)
    assert run(forge, _policy(True))["ok"]
    forge.rows[PR].update(state="closed")
    assert run(forge, _policy(True))["classification"] == "closed"


def test_real_capture_protection_is_not_production_grade():
    # The captured rule requires no approvals: production must refuse it.
    forge = Forge()
    forge.rows[REVIEWS] = [_review()]
    assert run(forge, _policy(False))["classification"] == "protection"


@pytest.mark.parametrize("case,expected", [
    ("custody", "custody"), ("workflow_missing", "infra"), ("workflow_modified", "workflow"),
    ("writer_status_only", "missing"), ("wrong_runner", "untrusted"), ("pending", "pending"),
    ("skipped", "failure"), ("failed_run", "failure"), ("old_attempt", "untrusted"),
    ("incomplete_jobs", "infra"), ("duplicate_job", "infra"), ("duplicate_runs", "infra"),
    ("missing_run_page", "infra"), ("total_changed", "infra"), ("no_status_check", "protection"),
    ("wrong_context", "protection"), ("extra_merger", "protection"), ("merge_team", "protection"),
    ("admin_override", "protection"), ("push_allowed", "protection"), ("wrong_repo_id", "binding"),
    ("wrong_repo_name", "binding"), ("wrong_base", "binding"), ("bad_head", "infra"), ("wrong_number", "infra"),
    ("required_job_absent", "missing"),
])
def test_fail_closed(case, expected):
    forge, custody = Forge(), True
    rows = forge.rows
    contents = next(k for k in rows if "/contents/" in k)
    job = rows[JOBS]["jobs"][0]
    run8 = next(r for r in rows[RUNS]["workflow_runs"] if r["id"] == 8)
    edits = {
        "custody": lambda: None,
        "workflow_missing": lambda: rows.pop(contents),
        "workflow_modified": lambda: rows[contents].update(content="Zm9yZ2Vk"),
        "writer_status_only": lambda: rows.__setitem__(RUNS, {"workflow_runs": [], "total_count": 0}),
        "wrong_runner": lambda: job.update(runner_id=999),
        "pending": lambda: job.update(status="in_progress"),
        "skipped": lambda: job.update(conclusion="skipped"),
        "failed_run": lambda: run8.update(conclusion="failure"),
        "old_attempt": lambda: job.update(run_attempt=1),
        "incomplete_jobs": lambda: rows[JOBS].update(total_count=2),
        "duplicate_job": lambda: rows[JOBS].update(jobs=[job, dict(job, id=99)], total_count=2),
        "duplicate_runs": lambda: rows[RUNS]["workflow_runs"].append(rows[RUNS]["workflow_runs"][0]),
        "missing_run_page": lambda: rows[RUNS].update(total_count=13),
        "total_changed": lambda: (rows[RUNS].update(total_count=60),
                                  rows.__setitem__(RUNS.replace("page=1", "page=2"),
                                                   {"workflow_runs": [], "total_count": 61})),
        "no_status_check": lambda: rows[PROT].update(enable_status_check=False),
        "wrong_context": lambda: rows[PROT].update(status_check_contexts=["forged"]),
        "extra_merger": lambda: rows[PROT]["merge_whitelist_usernames"].append("writer"),
        "merge_team": lambda: rows[PROT].update(merge_whitelist_teams=["writers"]),
        "admin_override": lambda: rows[PROT].update(block_admin_merge_override=False),
        "push_allowed": lambda: rows[PROT].update(enable_push=True),
        "wrong_repo_id": lambda: rows[PR]["base"]["repo"].update(id=2),
        "wrong_repo_name": lambda: rows[PR]["base"]["repo"].update(full_name="acme/other"),
        "wrong_base": lambda: rows[PR]["base"].update(ref="main"),
        "bad_head": lambda: rows[PR]["head"].update(sha="A" * 40),
        "wrong_number": lambda: rows[PR].update(number=9),
        "required_job_absent": lambda: job.update(name="other"),
    }
    edits[case]()
    if case == "custody":
        custody = False
    receipt = run(forge, _policy(True), custody_ok=custody)
    assert receipt["ok"] is False and receipt["classification"] == expected


def test_run_pagination_is_bounded():
    forge = Forge()
    forge.rows[RUNS] = {"workflow_runs": [dict(r, id=1000 + i) for i, r in
                                          enumerate(forge.rows[RUNS]["workflow_runs"][:1] * 50)],
                        "total_count": 10_000}
    for page in range(2, 50):
        forge.rows[RUNS.replace("page=1", f"page={page}")] = {
            "workflow_runs": [dict(forge.rows[RUNS]["workflow_runs"][0], id=page * 1000 + i) for i in range(50)],
            "total_count": 10_000}
    assert run(forge, _policy(True))["classification"] == "infra"
    assert max(int(p.rsplit("=", 1)[1]) for p in forge.reads if "actions/runs?" in p) == 40


@pytest.mark.parametrize("path,field,classification", [
    (PR, ("head", "sha"), "stale"), (PR, ("base", "sha"), "stale"), (PR, ("state",), "stale"),
    (PROT, ("required_approvals",), "protection"),
])
def test_readback_race_fails_closed(path, field, classification):
    forge = Forge()

    def change(row):
        target = row
        for key in field[:-1]:
            target = target[key]
        target[field[-1]] = "b" * 40 if field[-1] == "sha" else ("open" if field[-1] == "state" else 5)
        return row
    forge.mutate[(path, 2)] = change
    assert run(forge, _policy(True))["classification"] == classification


def test_expected_head_mismatch_is_stale():
    assert run(Forge(), _policy(True), expected_head="b" * 40)["classification"] == "stale"


# --- production: merged + independent exact-head approval (synthetic review rows) ---

def test_production_requires_merge_provenance_and_exact_head_approval():
    receipt = run(Forge(production=True), _policy(False))
    assert receipt["ok"] and receipt["qualification_only"] is False
    assert receipt["merge"] == {"merged": True, "merge_commit_sha": "05c7847890efedf9c96af76ec09fd12e83b7fe35",
                                "merged_by": "merge-bot", "merged_by_id": 4, "merged_at": "2026-10-13T20:33:09Z"}
    assert receipt["review"] == {"id": 31, "reviewer": "bob", "reviewer_id": 7, "commit_id": SHA,
                                 "submitted_at": "2000-01-01T00:00:00Z"}


@pytest.mark.parametrize("case,expected", [
    ("open", "unmerged"), ("merger_id_spoof", "merge"), ("merger_unlisted", "merge"),
    ("no_merge_commit", "merge"), ("no_reviews", "review"), ("self_approval", "review"),
    ("old_commit", "review"), ("stale", "review"), ("dismissed", "review"), ("unofficial", "review"),
    ("comment_only", "review"), ("changes_requested", "review"), ("duplicate_review", "infra"),
    ("reviewer_not_allowlisted", "review"), ("block_rejected_off", "protection"),
])
def test_production_fail_closed(case, expected):
    forge = Forge(production=True)
    policy = _policy(False)
    pr, reviews = forge.rows[PR], forge.rows[REVIEWS]
    author = pr["user"]
    edits = {
        "open": lambda: pr.update(state="open", merged=False),
        "merger_id_spoof": lambda: pr["merged_by"].update(id=99),
        "merger_unlisted": lambda: pr["merged_by"].update(login="alice", id=author["id"]),
        "no_merge_commit": lambda: pr.update(merge_commit_sha=None),
        "no_reviews": lambda: reviews.clear(),
        "self_approval": lambda: reviews[0].update(user={"id": author["id"], "login": author["login"]}),
        "old_commit": lambda: reviews[0].update(commit_id="c" * 40),
        "stale": lambda: reviews[0].update(stale=True),
        "dismissed": lambda: reviews[0].update(dismissed=True),
        "unofficial": lambda: reviews[0].update(official=False),
        "comment_only": lambda: reviews[0].update(state="COMMENT"),
        "changes_requested": lambda: reviews.append(_review(32, "carol", 8, "REQUEST_CHANGES")),
        "duplicate_review": lambda: reviews.append(copy.deepcopy(reviews[0])),
        "reviewer_not_allowlisted": lambda: policy.update(reviewers=["dave"]),
        "block_rejected_off": lambda: forge.rows[PROT].update(block_on_rejected_reviews=False),
    }
    edits[case]()
    receipt = run(forge, policy)
    assert receipt["ok"] is False and receipt["classification"] == expected


def test_production_later_approval_supersedes_own_earlier_change_request():
    forge = Forge(production=True)
    forge.rows[REVIEWS] = [_review(30, state="REQUEST_CHANGES"), _review(31)]
    assert run(forge, _policy(False))["ok"]


def test_production_review_pages_and_readback():
    def forge_with_pages():
        forge = Forge(production=True)
        forge.rows[REVIEWS] = [_review(100 + i, "carol", 8, "COMMENT") for i in range(50)]
        forge.rows[REVIEWS.replace("page=1", "page=2")] = [_review()]
        return forge
    assert run(forge_with_pages(), _policy(False))["ok"]
    forge = forge_with_pages()
    forge.mutate[(REVIEWS.replace("page=1", "page=2"), 2)] = lambda rows: rows + [_review(500, "dave", 9, "REQUEST_CHANGES")]
    assert run(forge, _policy(False))["classification"] == "stale"


def test_production_merge_readback_change_is_stale():
    forge = Forge(production=True)
    forge.mutate[(PR, 2)] = lambda row: {**row, "merge_commit_sha": "d" * 40}
    assert run(forge, _policy(False))["classification"] == "stale"


# --- policy and URL binding ---

@pytest.mark.parametrize("edit", [
    lambda p: p.update(allow_qualification=True), lambda p: p.pop("qualification_only"),
    lambda p: p.update(qualification_only="false"), lambda p: p.update(runner_ids=[]),
    lambda p: p.update(mergers=[{"login": "merge-bot"}]), lambda p: p.update(workflows=[]),
    lambda p: p["workflows"][0].update(sha256="x"), lambda p: p.update(policy_id=""),
])
def test_policy_schema_is_strict(edit):
    policy = _policy(False)
    edit(policy)
    with pytest.raises(ValueError):
        validate_policy(policy)


@pytest.mark.parametrize("url,parsed", [
    ("https://forge.example.test/acme/widget/pulls/8", (ORIGIN, "acme/widget", 8)),
    ("https://forge.example.test:3000/acme/widget/pulls/8", (ORIGIN + ":3000", "acme/widget", 8)),
    ("http://forge.example.test/acme/widget/pulls/8", None),
    ("https://forge.example.test/acme/widget/pulls/8?x=1", None),
    ("https://forge.example.test/acme/widget/pulls/8/", None),
    ("https://forge.example.test/acme/widget/pulls/08", None),
    ("https://user@forge.example.test/acme/widget/pulls/8", None),
    ("https://Forge.example.test/acme/widget/pulls/8", None),
    ("https://forge.example.test/../widget/pulls/8", None),
    ("https://forge.example.test/acme/../pulls/8", None),
    ("https://github.com/acme/widget/pulls/8", None),
    ("https://github.com/acme/widget/pull/8", None),
])
def test_pr_url_parsing(url, parsed):
    assert parse_pr_url(url) == parsed


def test_gate_receipt_binds_url_origin_and_repo():
    url = ORIGIN + "/acme/widget/pulls/8"
    receipt = gate_receipt(Forge(), _policy(True), url, ORIGIN, custody_ok=True)
    assert receipt["ok"] and (receipt["pr_url"], receipt["provider"], receipt["origin"]) == (url, "gitea", ORIGIN)
    assert gate_receipt(Forge(), _policy(True), url, "https://other.example.test", custody_ok=True)["classification"] == "binding"
    assert gate_receipt(Forge(), _policy(True), ORIGIN + "/acme/other/pulls/8", ORIGIN,
                        custody_ok=True)["classification"] == "binding"
    bad = _policy(True)
    bad["allow_qualification"] = True
    assert gate_receipt(Forge(), bad, url, ORIGIN, custody_ok=True)["classification"] == "infra"


def test_post_merge_approval_is_not_premerge_review():
    forge = Forge(production=True)
    forge.rows[REVIEWS][0]["submitted_at"] = "2099-01-01T00:00:00Z"
    assert run(forge, _policy(False))["classification"] == "review"


def test_postmerge_ci_cannot_qualify_merge():
    forge = Forge(production=True)
    forge.rows[JOBS]["jobs"][0]["completed_at"] = "2099-01-01T00:00:00Z"
    assert run(forge, _policy(False))["classification"] == "merge"


def test_workflow_basename_and_job_identity_must_be_unique():
    policy = _policy(False)
    policy["workflows"].append(copy.deepcopy(policy["workflows"][0]))
    policy["workflows"][1]["path"] = "other/" + policy["workflows"][0]["path"].split("/")[-1]
    policy["workflows"][1]["jobs"][0]["context"] = "other"
    with pytest.raises(ValueError): validate_policy(policy)
    policy = _policy(False)
    policy["workflows"][0]["jobs"].append(dict(policy["workflows"][0]["jobs"][0], context="other"))
    with pytest.raises(ValueError): validate_policy(policy)


def test_strict_qualification_merge_never_becomes_production():
    receipt = run(Forge(production=True), _policy(True), require_review=True)
    assert receipt["ok"] and receipt["qualification_only"] is True
    assert receipt["merge"]["merged"] and receipt["review"]["commit_id"] == SHA
    forge = Forge(production=True)
    forge.rows[REVIEWS] = []
    assert run(forge, _policy(True), require_review=True)["classification"] == "review"


def test_premerge_requires_reviews_but_cannot_be_completion():
    forge = Forge(production=True)
    forge.rows[PR].update(state="open", merged=False, merged_by=None, merge_commit_sha=None)
    assert run(forge, _policy(False))["classification"] == "unmerged"
    receipt = run(forge, _policy(False), require_merged=False)
    assert receipt["ok"] and "merge" not in receipt
    forge.rows[REVIEWS] = []
    assert run(forge, _policy(False), require_merged=False)["classification"] == "review"


def test_comment_cannot_erase_rejection():
    forge = Forge(production=True)
    forge.rows[REVIEWS] = [_review(31, state="REQUEST_CHANGES"), _review(32, state="COMMENT"),
                          _review(33, login="carol", user_id=8)]
    assert run(forge, _policy(False))["classification"] == "review"


def test_new_attempt_after_collection_refused():
    forge = Forge(production=True)
    def changed(row):
        row["workflow_runs"][0]["run_attempt"] += 1
        return row
    forge.mutate[(RUNS, 2)] = changed
    assert run(forge, _policy(False))["classification"] == "stale"

