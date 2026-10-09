"""Acceptance invariants for GitHub completion contracts.

Regression for the two reproduced defects: Repository Rules was read BEFORE
Check Runs and its 403 on a private/free repository aborted collection (the
receipt then claimed ``checks: []`` as if CI had reported nothing), and the
exact PR had to be re-supplied on the reviewer's completion because nothing
bound it at the implementer's review handoff. Plus the follow-up defect in the
fix for the second: the binding ran BEFORE ``request_review``'s own
compare-and-swap, and since a normal ``return`` out of ``write_txn`` commits, a
refused handoff could still pin the card's PR permanently.

GitHub is mocked at ``kanban_pr_acceptance._api`` — the single process
boundary the gate talks to GitHub through — so the real collector, the real
config loader, real SQLite and the real ``complete_task`` /
``request_review`` lifecycle run on every host.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_pr_acceptance as pra
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_pr_acceptance_store import PublishedPrBindingError

PR_URL = "https://github.com/acme/repo/pull/7"
HEAD = "a" * 40
OTHER_HEAD = "b" * 40

#: "No override installed", so a test can still override with ``None``.
_UNSET = object()


class FakeGitHub:
    """``gh api`` responses in the shapes GitHub actually returns.

    ``rules_forbidden`` reproduces a private repository without a paid plan:
    ``gh`` exits non-zero on the 403 from the Rules endpoint.
    """

    def __init__(self):
        self.head = HEAD
        self.base = "main"
        self.state = "OPEN"
        self.merged = False
        self.protection_contexts = []
        self.rules_contexts = []
        self.rules_forbidden = False
        self.check_runs = []
        self.statuses = []
        self.noise_runs = 0
        self.total_count_override = None
        #: Stands in for the ``/pulls/{n}`` record verbatim, so a test can hand
        #: the collector a shape GitHub would never send.
        self.pull_override = None
        #: The same for the GraphQL answer: the whole response envelope
        #: verbatim, including the levels above ``pullRequest``.
        self.graphql_override = _UNSET
        self.calls = []
        self.hooks = {}

    # --- helpers used by the tests ---

    def run(self, name, conclusion, *, status="completed", head=None, app_id=None, run_id=42):
        self.check_runs.append({
            "id": run_id, "name": name, "head_sha": head or self.head,
            "app": {"id": app_id if app_id is not None else 1},
            "status": status, "conclusion": conclusion,
            "html_url": f"https://github.com/acme/repo/actions/runs/{run_id}",
        })
        return self

    def legacy_status(self, context, state, *, status_id=5):
        self.statuses.append({
            "id": status_id, "context": context, "state": state,
            "target_url": "https://ci.example/1",
        })
        return self

    # --- transport ---

    def __call__(self, endpoint, *, query=None, paginate=False, profile_home=None):
        self.calls.append(endpoint)
        hook = next((fn for key, fn in self.hooks.items() if key in endpoint), None)
        if endpoint == "graphql" and self.graphql_override is not _UNSET:
            # ``None`` is itself a shape GitHub can answer, so the override has
            # its own sentinel rather than using None for "not overridden".
            value = self.graphql_override
        elif endpoint == "graphql":
            protection = (
                {"requiredStatusChecks": [{"context": c, "app": {"databaseId": a}}
                                          for c, a in self.protection_contexts]}
                if self.protection_contexts else None
            )
            value = {"data": {"repository": {"pullRequest": {
                "headRefOid": self.head, "baseRefName": self.base, "state": self.state,
                "baseRef": {"branchProtectionRule": protection}}}}}
        elif "/rules/branches/" in endpoint:
            if self.rules_forbidden == "auth":
                # What the real ``_api`` raises for the 403 itself: a private
                # repository on a free plan is an identity refusal, not a
                # transport failure, and it must not abort collection either.
                raise pra._GateAuthError("HTTP 403 on " + endpoint.split("?")[0])
            if self.rules_forbidden:
                raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
            value = [[{"type": "required_status_checks", "parameters": {
                "required_status_checks": [{"context": c, "integration_id": a}
                                           for c, a in self.rules_contexts]}}]] \
                if self.rules_contexts else [[]]
        elif "/check-runs" in endpoint:
            value = self._check_run_pages()
        elif "/statuses" in endpoint:
            value = [self.statuses] if self.statuses else [[]]
        elif "/pulls/" in endpoint:
            value = self.pull_override if self.pull_override is not None else {
                "head": {"sha": self.head}, "base": {"ref": self.base},
                "state": "closed" if self.state != "OPEN" else "open",
                "merged": self.merged}
        else:
            raise AssertionError(f"unexpected endpoint {endpoint}")
        if hook is not None:
            hook()
        return value

    def _check_run_pages(self):
        """Noise runs fill whole 100-item pages so a required run can only be
        found by following pagination past the first page."""
        noise = [{"id": 1000 + i, "name": "noise", "head_sha": self.head,
                  "app": {"id": 1}, "status": "completed", "conclusion": "skipped"}
                 for i in range(self.noise_runs)]
        runs = noise + list(self.check_runs)
        total = self.total_count_override if self.total_count_override is not None else len(runs)
        pages = [runs[i:i + 100] for i in range(0, len(runs), 100)] or [[]]
        return [{"total_count": total, "check_runs": page} for page in pages]


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(pra, "_api", fake)
    return fake


def _declare_checks(entry):
    """Write ``kanban.completion_checks`` to the temp home's real config.yaml."""
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"kanban": {"completion_checks": entry}}), encoding="utf-8")


def _live_profile(name):
    """A profile the gate can resolve. ``request_review(reviewer=…)`` reassigns the
    card, and acceptance reads the repo as the ASSIGNEE profile's gh login
    (#122689) — so the reviewer needs a home, not just a name."""
    home = Path(os.environ["HERMES_HOME"]) / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("", encoding="utf-8")
    return name


def _receipt(conn, task_id):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' "
        "ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


def _card(conn, *, contract="acme/repo", title="Publish"):
    return kb.create_task(conn, title=title, completion_contract=contract)


def _pinned_events(conn, task_id):
    return [json.loads(row["payload"]) for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_pinned' ORDER BY id",
        (task_id,))]


def test_rules_403_accepts_declared_green_check_and_separates_its_evidence(github):
    """A private/free repository's Rules 403 must not stop the Check Runs read."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is True
        assert kb.get_task(conn, tid).status == "done"
        receipt = _receipt(conn, tid)
    assert receipt["ok"] and receipt["phase"] == "accepted"
    assert receipt["head_sha"] == HEAD
    # Rules unreadable is recorded as such, and the checks endpoint was still read.
    assert receipt["rules"]["available"] is False and receipt["rules"]["reason"]
    assert receipt["checks_endpoint"]["fetched"] is True
    assert receipt["required_sources"] == ["configured"]
    assert [c["classification"] for c in receipt["checks"]] == ["success"]


def test_rules_403_raised_as_an_auth_refusal_still_reaches_the_checks(github):
    """``_api`` turns the Rules 403 into ``_GateAuthError`` (#122689's profile
    isolation), which is the shape production actually raises — the optional
    Rules read must survive that one too, not only a bare transport failure."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = "auth"
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, tid)
    assert receipt["ok"] and receipt["classification"] != "auth"
    assert receipt["rules"]["available"] is False and receipt["rules"]["reason"]
    assert receipt["checks_endpoint"]["fetched"] is True


def test_rules_403_without_any_declared_policy_fails_before_the_checks_read(github):
    """No declared policy is fail-closed, and the receipt says so without
    pretending the checks endpoint answered empty."""
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert not receipt["ok"] and receipt["phase"] == "declared_policy"
    assert receipt["rules"]["available"] is False
    assert receipt["checks_endpoint"]["fetched"] is False
    assert receipt["checks"] == []
    assert "completion_checks" in receipt["detail"]
    assert not any("check-runs" in call for call in github.calls)


def test_rules_403_with_declared_policy_and_no_ci_at_all_fails(github):
    """Declared policy + an empty checks endpoint is distinguishable from the
    case above: the endpoint was read and returned nothing."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["classification"] == "missing"
    assert receipt["checks_endpoint"] == {"fetched": True, "total_count": 0, "runs": 0,
                                          "statuses": 0, "pages": 1}
    assert [c["classification"] for c in receipt["checks"]] == ["missing"]


def test_declared_check_absent_while_other_checks_are_green_fails(github):
    """All-green-observed is never a substitute for the declared check."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("some-other-job", "success", run_id=9)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["classification"] == "missing"
    assert [c["name"] for c in receipt["checks"]] == ["ci/test"]


@pytest.mark.parametrize("conclusion,status", [
    ("failure", "completed"), ("cancelled", "completed"), ("timed_out", "completed"),
    ("skipped", "completed"), ("neutral", "completed"), ("action_required", "completed"),
    ("stale", "completed"), (None, "in_progress"), ("some_future_conclusion", "completed"),
])
def test_no_non_success_conclusion_family_can_complete(github, conclusion, status):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", conclusion, status=status)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["classification"] != "success"
    assert receipt["checks"][0]["conclusion"] == conclusion


def test_required_check_past_the_first_page_is_found_and_truncation_fails(github):
    """The required run only exists on page 2 of the paginated response."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.noise_runs = 100
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, tid)
        assert receipt["checks_endpoint"]["pages"] == 2
        assert receipt["checks_endpoint"]["runs"] == 101

        # A page set that does not add up to total_count is incomplete evidence.
        github.total_count_override = 500
        stale_pages = _card(conn, title="truncated")
        assert kb.complete_task(conn, stale_pages, summary="handoff", metadata={"published_pr": PR_URL}) is False
        truncated = _receipt(conn, stale_pages)
    assert truncated["classification"] == "infra" and truncated["phase"] == "check_runs"


def test_legacy_statuses_are_paginated_for_the_exact_head(github):
    _declare_checks({"acme/repo": {"required_checks": ["legacy-ci"]}})
    github.rules_forbidden = True
    github.legacy_status("legacy-ci", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, tid)
    assert receipt["checks_endpoint"]["statuses"] == 1
    assert receipt["checks"][0]["head_sha"] == HEAD


@pytest.mark.parametrize("field,value", [("head", OTHER_HEAD), ("base", "release")])
def test_head_or_base_change_during_collection_fails_stale(github, field, value):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")

    def mutate():
        setattr(github, "head" if field == "head" else "base", value)

    github.hooks["/check-runs"] = mutate
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["classification"] == "stale" and receipt["phase"] == "stale_recheck"


def test_readable_rules_and_branch_protection_contexts_are_still_enforced(github):
    """Readable GitHub policy keeps working, with no config entry at all."""
    github.protection_contexts = [("protected-ci", 1)]
    github.rules_contexts = [("ruleset-ci", None)]
    github.run("protected-ci", "success", run_id=11)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        missing_ruleset = _receipt(conn, tid)
        assert missing_ruleset["classification"] == "missing"
        assert sorted(missing_ruleset["required_sources"]) == ["branch_protection", "repository_rules"]

        github.run("ruleset-ci", "success", run_id=12)
        done = _card(conn, title="both green")
        assert kb.complete_task(conn, done, summary="handoff", metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, done)
    assert receipt["rules"]["available"] is True and receipt["rules"]["reason"] is None
    assert {c["name"] for c in receipt["checks"]} == {"protected-ci", "ruleset-ci"}


def test_review_handoff_pins_the_pr_so_the_reviewer_completes_without_repeating_it(github):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.request_review(conn, tid, summary="implemented", metadata={"published_pr": PR_URL})
        assert kb.get_task(conn, tid).completion_contract == PR_URL
        # The reviewer's completion carries no PR evidence of its own.
        assert kb.complete_task(conn, tid, summary="approved") is True
        receipt = _receipt(conn, tid)
    assert receipt["ok"] and receipt["pr_url"] == PR_URL


def _refused_handoff(conn, refusal):
    """A card whose next ``request_review`` must be refused, plus the kwargs that
    refuse it. Every case leaves ``completion_contract`` un-pinned, so a binding
    the refusal should not have made is visible as a changed contract."""
    tid = _card(conn)
    if refusal == "stale_run_id":
        return tid, {"expected_run_id": 987654}
    if refusal == "ineligible_status":
        # ``blocked`` is not a source status for a review handoff.
        assert kb.block_task(conn, tid, reason="waiting on an answer") is True
        return tid, {}
    if refusal == "live_claim":
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        # This process stands in for the spawned worker: alive and fingerprinted.
        kbd._set_worker_pid(conn, tid, os.getpid())
        return tid, {}
    if refusal == "unsatisfied_parent":
        parent = kb.create_task(conn, title="parent that is not done")
        kb.link_tasks(conn, parent, tid)
        return tid, {}
    # A re-review whose durable reviewer provenance is corrupt. The first handoff
    # names no PR, so the contract is still the bare repository here; the card
    # needs an assignee for ``request_changes`` to have an implementer to route
    # back to.
    conn.execute("UPDATE tasks SET assignee = 'builder' WHERE id = ?", (tid,))
    conn.commit()
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="v1", reviewer="reviewer",
                             expected_run_id=claimed.current_run_id) is True
    review = kb.claim_review_task(conn, tid)
    assert kb.request_changes(conn, tid, reason="fix",
                              expected_run_id=review.current_run_id)[0] is True
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET payload = '{}' WHERE task_id = ? "
                     "AND kind = 'changes_requested'", (tid,))
    retry = kb.claim_task(conn, tid, claimer="builder:retry")
    return tid, {"expected_run_id": retry.current_run_id}


@pytest.mark.parametrize("refusal", [
    "stale_run_id", "ineligible_status", "live_claim", "unsatisfied_parent",
    "no_reviewer_provenance",
])
def test_a_refused_review_handoff_never_pins_the_cards_pr(github, refusal):
    """A normal ``return`` out of ``write_txn`` COMMITS.

    So binding the exact PR before the transition's own compare-and-swap let a
    handoff that was then refused still pin the card — permanently, since the
    pin is immutable — from a run it never owned, and append a ``pr_pinned``
    event claiming it did. Every refusal must leave the contract and the event
    log exactly as it found them.
    """
    with connect_closing() as conn:
        tid, kwargs = _refused_handoff(conn, refusal)
        before = kb.get_task(conn, tid)

        assert kb.request_review(conn, tid, summary="handoff",
                                 metadata={"published_pr": PR_URL}, **kwargs) is False

        after = kb.get_task(conn, tid)
        assert after.completion_contract == before.completion_contract == "acme/repo"
        assert after.status == before.status
        assert _pinned_events(conn, tid) == []
    # Nothing about a refused handoff talks to GitHub either.
    assert github.calls == []


def test_the_pin_still_happens_on_the_handoff_that_is_accepted(github):
    """The fence must not cost the feature: once the CAS wins, the PR is pinned
    in that same transaction, so the reviewer never repeats the URL."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        # Refused first, on a run the card never had.
        assert kb.request_review(conn, tid, summary="v1", metadata={"published_pr": PR_URL},
                                 expected_run_id=987654) is False
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"

        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary="v1", reviewer=_live_profile("reviewer"),
                                 metadata={"published_pr": PR_URL},
                                 expected_run_id=claimed.current_run_id) is True
        assert kb.get_task(conn, tid).completion_contract == PR_URL
        # Exactly one binding, on the run that actually made the handoff.
        assert [p["pr_url"] for p in _pinned_events(conn, tid)] == [PR_URL]
        assert kb.complete_task(conn, tid, summary="approved") is True
        assert _receipt(conn, tid)["ok"] is True


def test_pr_named_under_the_wrong_key_is_an_explicit_error_on_both_transitions(github):
    with connect_closing() as conn:
        completing = _card(conn)
        with pytest.raises(PublishedPrBindingError) as complete_err:
            kb.complete_task(conn, completing, summary="handoff", metadata={"pr_url": PR_URL})
        assert "published_pr" in str(complete_err.value) and "pr_url" in str(complete_err.value)
        # Nothing was mutated and no GitHub call was made for an unbindable handoff.
        assert kb.get_task(conn, completing).status == "ready"
        assert kb.get_task(conn, completing).completion_contract == "acme/repo"

        reviewing = _card(conn, title="review handoff")
        with pytest.raises(PublishedPrBindingError):
            kb.request_review(conn, reviewing, summary="done", metadata={"pr_url": PR_URL})
        assert kb.get_task(conn, reviewing).status == "ready"
        assert kb.get_task(conn, reviewing).completion_contract == "acme/repo"
    assert github.calls == []


def test_a_sibling_pr_cannot_replace_the_pinned_one_after_a_failed_attempt(github):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "failure")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).completion_contract == PR_URL

        # The retry names a different PR of the same repository.
        sibling = "https://github.com/acme/repo/pull/8"
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": sibling}) is False
        assert kb.get_task(conn, tid).completion_contract == PR_URL
        rejection = _receipt(conn, tid)
        assert rejection["pr_url"] == PR_URL and rejection["rejected_pr"] == sibling

        # And a PR from another repository is rejected outright.
        with pytest.raises(PublishedPrBindingError):
            kb.request_review(conn, tid, summary="x",
                              metadata={"published_pr": "https://github.com/other/repo/pull/7"})

        github.check_runs = []
        github.run("ci/test", "success")
        assert kb.complete_task(conn, tid, summary="green now") is True
        assert kb.get_task(conn, tid).completion_contract == PR_URL


def test_declared_checks_load_from_config_yaml_and_bad_entries_report_themselves(github):
    """The loader's own data flow: real config.yaml -> real load_config()."""
    from hermes_cli.kanban_completion_policy import configured_required_checks

    _declare_checks({"ACME/Repo": ["from-bare-list", "from-bare-list", " spaced "]})
    assert configured_required_checks("acme/repo") == (
        ["from-bare-list", "spaced"], None)
    assert configured_required_checks("other/repo") == ([], None)

    _declare_checks({"acme/repo": {"required_checks": "ci/test"}})
    names, problem = configured_required_checks("acme/repo")
    assert names == [] and "must be a list" in problem

    _declare_checks({"acme/repo": {"note": "todo"}})
    names, problem = configured_required_checks("acme/repo")
    assert names == [] and "required_checks" in problem

    # A malformed entry blocks completion and names itself in the receipt.
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["phase"] == "declared_policy"
    assert "required_checks" in receipt["config_problem"]


def _rerun_after_the_first_collection(github, conclusion, *, status="completed", run_id=43):
    """Queue one more run of the required check against the SAME head, once,
    after the first collection pass has read the check-run pages.

    The stale recheck is the first thing that happens after that pass, so its
    ``/pulls/`` call is the exact moment a rerun becomes invisible to evidence
    already collected — and the PR itself never moves, which is all that
    recheck can see.
    """
    fired = []

    def rerun():
        if fired:
            return
        fired.append(True)
        github.run("ci/test", conclusion, status=status, run_id=run_id)

    github.hooks["/pulls/"] = rerun


@pytest.mark.parametrize("conclusion,status", [
    (None, "queued"), (None, "in_progress"), ("failure", "completed"),
    ("cancelled", "completed"), ("timed_out", "completed"), ("skipped", "completed"),
    ("neutral", "completed"), ("action_required", "completed"),
    ("some_future_conclusion", "completed"),
])
def test_a_rerun_on_the_same_head_after_collection_fails_closed(github, conclusion, status):
    """The window an exact-head recheck cannot see.

    A rerun queues a NEW required run against the SAME sha: the PR's head,
    base and state are untouched, so collecting once and then only rechecking
    the PR records an acceptance that was already out of date. Every
    non-success family a newer run can be in has to refuse.
    """
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    _rerun_after_the_first_collection(github, conclusion, status=status)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["ok"] is False and receipt["classification"] != "success"
    assert receipt["phase"] == "recheck_evaluate"
    # The first pass is kept as history; the recorded evidence is the re-read
    # one, which is the only one that saw the newer run.
    assert [c["classification"] for c in receipt["recheck"]["first_pass"]] == ["success"]
    assert {c["id"] for c in receipt["checks"]} == {42, 43}


def test_a_rerun_that_also_succeeded_still_accepts(github):
    """The recheck is a fail-closed re-read, not a new way to refuse: a rerun
    that went green on the same head is still green."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    _rerun_after_the_first_collection(github, "success", run_id=44)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, tid)
    assert receipt["ok"] and receipt["phase"] == "accepted"
    assert receipt["recheck"]["performed"] is True
    assert [c["classification"] for c in receipt["checks"]] == ["success", "success"]


def test_the_evidence_is_collected_exactly_twice_and_never_in_a_loop(github):
    """Fail-closed must not become a retry loop: however the card ends, the
    checks are collected at most twice and the PR rechecked at most twice."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        accepted = _card(conn)
        assert kb.complete_task(conn, accepted, summary="handoff", metadata={"published_pr": PR_URL}) is True
    assert len([c for c in github.calls if "/check-runs" in c]) == 2
    assert len([c for c in github.calls if "/statuses" in c]) == 2
    assert len([c for c in github.calls if "/pulls/" in c]) == 2

    # A card refused on the first pass never pays for the second one.
    github.calls.clear()
    github.check_runs = []
    github.run("ci/test", "failure")
    with connect_closing() as conn:
        refused = _card(conn, title="red")
        assert kb.complete_task(conn, refused, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert _receipt(conn, refused)["recheck"]["performed"] is False
    assert len([c for c in github.calls if "/check-runs" in c]) == 1
    assert len([c for c in github.calls if "/pulls/" in c]) == 1


@pytest.mark.parametrize("record", [
    "merged", ["merged"], 7,
    {"base": {"ref": "main"}, "state": "open", "merged": False},
    {"head": None, "base": {"ref": "main"}, "state": "open", "merged": False},
    {"head": {"sha": 42}, "base": {"ref": "main"}, "state": "open", "merged": False},
    {"head": {"sha": HEAD}, "base": "main", "state": "open", "merged": False},
    {"head": {"sha": HEAD}, "base": ["main"], "state": "open", "merged": False},
    {"head": {"sha": HEAD}, "base": {"ref": "  "}, "state": "open", "merged": False},
    {"head": {"sha": HEAD}, "base": {"ref": "main"}, "state": "open", "merged": "true"},
    {"head": {"sha": HEAD}, "base": {"ref": "main"}, "state": "open", "merged": "false"},
    # REST reports exactly ``open`` or ``closed``: a proxy's "unknown", a blank,
    # the GraphQL spelling, a number, a list or a null are all states no gate
    # condition can act on, and none of them is ``closed``.
    *({"head": {"sha": HEAD}, "base": {"ref": "main"}, "state": state, "merged": False}
      for state in ("unknown", "proxy-error", "", "OPEN", 7, ["open"], None)),
])
def test_a_malformed_pr_record_blocks_acceptance_without_raising(github, record):
    """The head recheck dereferences this record field by field, and
    ``merged: "false"`` is a truthy string. A shape that cannot be read is
    infrastructure trouble, never an acceptance and never a traceback."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    github.pull_override = record
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["ok"] is False and receipt["classification"] == "infra"
    assert receipt["phase"] == "stale_recheck"
    # The receipt names which field could not be read, never the body itself.
    assert "malformed" in receipt["detail"]
    assert "pull request" in receipt["detail"]


def test_a_state_gone_unreadable_by_the_final_recheck_still_refuses(github):
    """The PR is rechecked twice — once after collection and once more after the
    second pass — and the final recheck is the last thing between an otherwise
    green card and acceptance, so it reads the state as strictly as the first."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")

    def answer_the_next_read_with_an_unknown_state():
        github.pull_override = {"head": {"sha": HEAD}, "base": {"ref": "main"},
                                "state": "proxy-error", "merged": False}

    github.hooks["/pulls/"] = answer_the_next_read_with_an_unknown_state
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["ok"] is False and receipt["classification"] == "infra"
    assert receipt["phase"] == "final_stale_recheck"
    assert receipt["recheck"]["performed"] is True
    assert "malformed" in receipt["detail"] and "proxy-error" not in receipt["detail"]


@pytest.mark.parametrize("state,merged,accepted", [
    ("open", False, True),
    ("closed", True, True),
    ("closed", False, False),
])
def test_both_states_github_does_send_keep_their_meaning(github, state, merged, accepted):
    """Strictness about ``state`` must not cost the two values REST reports: a
    merged PR is ``closed`` and still acceptable evidence, while closed without
    a merge is the stale case the recheck exists for."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    github.pull_override = {"head": {"sha": HEAD}, "base": {"ref": "main"},
                            "state": state, "merged": merged}
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is accepted
        receipt = _receipt(conn, tid)
    assert receipt["ok"] is accepted
    assert receipt["classification"] == ("success" if accepted else "stale")
    assert "malformed" not in (receipt.get("detail") or "")


def _graphql(pull_request):
    """The acceptance query's envelope around one ``pullRequest`` value."""
    return {"data": {"repository": {"pullRequest": pull_request}}}


def _pull_request(**overrides):
    """GitHub's answer for an ordinary open PR with no protection rule."""
    return {"headRefOid": HEAD, "baseRefName": "main", "state": "OPEN",
            "baseRef": {"branchProtectionRule": None}, **overrides}


def _protected(required_status_checks):
    return _pull_request(baseRef={"branchProtectionRule": {
        "requiredStatusChecks": required_status_checks}})


#: Every level of the consumed GraphQL envelope, each broken the way a proxy's
#: error page, an enterprise host's older schema or a partial-error answer
#: breaks it: a string, a list or a number where an object, a sha, a branch
#: name, a check list or an app id belongs.
_MALFORMED_GRAPHQL = {
    "response is a string": "not-an-object",
    "response is a list": [],
    "response carries no data": {},
    "data is a string": {"data": "bad"},
    "data carries no repository": {"data": {}},
    "repository is a list": {"data": {"repository": []}},
    "repository is a string": {"data": {"repository": "bad"}},
    "pullRequest is a string": _graphql("bad"),
    "pullRequest is null": _graphql(None),
    "headRefOid is null": _graphql(_pull_request(headRefOid=None)),
    "headRefOid is not a sha": _graphql(_pull_request(headRefOid="HEAD")),
    "headRefOid is a number": _graphql(_pull_request(headRefOid=7)),
    "baseRefName is blank": _graphql(_pull_request(baseRefName="   ")),
    "baseRefName is a list": _graphql(_pull_request(baseRefName=["main"])),
    "state is null": _graphql(_pull_request(state=None)),
    "state is a number": _graphql(_pull_request(state=7)),
    "baseRef is a string": _graphql(_pull_request(baseRef="not-an-object")),
    "baseRef is a list": _graphql(_pull_request(baseRef=[])),
    "branchProtectionRule is a list": _graphql(
        _pull_request(baseRef={"branchProtectionRule": []})),
    "branchProtectionRule is a string": _graphql(
        _pull_request(baseRef={"branchProtectionRule": "bad"})),
    "requiredStatusChecks is a string": _graphql(_protected("bad")),
    "requiredStatusChecks is an object": _graphql(_protected({"nodes": []})),
    "a required check is a string": _graphql(_protected(["ci/test"])),
    "a required check is null": _graphql(_protected([None])),
    "a required check context is blank": _graphql(_protected([{"context": "  "}])),
    "a required check context is a number": _graphql(_protected([{"context": 7}])),
    "a required check has no context": _graphql(_protected([{"app": {"databaseId": 1}}])),
    "a required check app is a string": _graphql(
        _protected([{"context": "ci/test", "app": "bad"}])),
    "a required check app is a list": _graphql(
        _protected([{"context": "ci/test", "app": []}])),
    "app databaseId is a string": _graphql(
        _protected([{"context": "ci/test", "app": {"databaseId": "1"}}])),
    "app databaseId is a bool": _graphql(
        _protected([{"context": "ci/test", "app": {"databaseId": True}}])),
    "app databaseId is a list": _graphql(
        _protected([{"context": "ci/test", "app": {"databaseId": []}}])),
    "a later required check is malformed": _graphql(_protected(
        [{"context": "ci/test", "app": {"databaseId": 1}}, {"context": "ci/lint", "app": 7}])),
}


def test_an_explicit_null_repository_is_an_identity_refusal_not_malformed(github):
    """GitHub answers ``repository: null`` for a repo the login cannot see, so
    that one shape is the ``auth`` classification an operator fixes by signing
    the assignee profile in (#122689) — never ``infra``, which reads as "retry",
    and never a traceback. A MISSING repository key stays malformed."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.run("ci/test", "success")
    github.graphql_override = {"data": {"repository": None}}
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["ok"] is False and receipt["classification"] == "auth"
    assert "acme/repo" in receipt["detail"]
    assert not any("check-runs" in call for call in github.calls)


@pytest.mark.parametrize("payload", list(_MALFORMED_GRAPHQL.values()),
                         ids=list(_MALFORMED_GRAPHQL))
def test_a_malformed_graphql_envelope_blocks_acceptance_without_raising(github, payload):
    """The first hop's answer nests four levels deep before the first field the
    gate reads, and every level is one GitHub may answer ``null`` for.

    Each of these shapes used to be dereferenced on trust: the object-or-null
    levels raised ``AttributeError`` (``"not-an-object".get(...)``), which is not
    an API failure the collector converts, so a malformed answer aborted
    ``complete_task`` with a traceback instead of blocking it with a receipt.
    None of them may read as acceptance either — the card below is otherwise
    green and declared, so a partial read would accept it.
    """
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.run("ci/test", "success")
    github.graphql_override = payload
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["ok"] is False and receipt["classification"] == "infra"
    assert receipt["phase"] == "pr_resolve"
    # The receipt names the field that could not be read…
    assert "malformed" in receipt["detail"]
    # …and no evidence was collected against an unread head.
    assert receipt["head_sha"] is None and receipt["base_ref"] is None
    assert receipt["checks"] == [] and receipt["required"] == []
    assert receipt["checks_endpoint"]["fetched"] is False
    assert [c for c in github.calls if "graphql" not in c] == []


def test_a_malformed_graphql_receipt_never_quotes_the_response_body(github):
    """A GraphQL answer can carry a token or host detail (a proxy's auth error
    page lands in exactly this position), and receipts are immutable board
    history: the receipt reports which field was unreadable, never its value."""
    secret = "ghp_averyrealisticlookingtokenvalue"
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.run("ci/test", "success")
    github.graphql_override = _graphql(_pull_request(baseRef=secret))
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
        task = kb.get_task(conn, tid)
    assert secret not in json.dumps(receipt)
    assert secret not in (task.last_failure_error or "")
    assert "baseRef" in receipt["detail"]


def test_a_null_protection_rule_and_null_app_id_stay_ordinary_answers(github):
    """Fail-closed must not mean fail-on-``null``: GitHub legitimately answers
    ``null`` for ``baseRef``, for a repository with no protection rule, for a
    rule that requires no checks, and for a check pinned to no app."""
    _declare_checks({"acme/repo": {"required_checks": []}})
    github.run("ci/test", "success")
    github.legacy_status("ci/test", "success")
    for pull_request in (
        _pull_request(baseRef=None),
        _pull_request(baseRef={"branchProtectionRule": None}),
        _protected(None),
        _protected([{"context": "ci/test", "app": None}]),
        _protected([{"context": "ci/test"}]),
    ):
        github.graphql_override = _graphql(pull_request)
        with connect_closing() as conn:
            tid = _card(conn)
            accepted = kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL})
            receipt = _receipt(conn, tid)
        # No declared policy for the first three: fail closed on the POLICY,
        # never on the shape — and the shape must not be the thing that refused.
        assert "malformed" not in (receipt.get("detail") or "")
        assert receipt["head_sha"] == HEAD and receipt["base_ref"] == "main"
        assert accepted is bool(receipt["required"])


def test_local_only_cards_never_reach_github(github):
    with connect_closing() as conn:
        tid = _card(conn, contract="local-only")
        assert kb.complete_task(conn, tid, summary=f"{PR_URL} is background context") is True
        assert _receipt(conn, tid) is None
    assert github.calls == []
