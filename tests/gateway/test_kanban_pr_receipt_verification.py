"""The completion ping may only name a pull request that GitHub itself confirms.

The ping tells a Human "this task produced PR #N at this URL, at this HEAD". That
sentence is admissible only when it comes from GitHub and binds to the task being
reported, so every test here attacks one link of that chain:

* a worker that invents a PR in its metadata/summary/result,
* a well-formed URL/SHA that does not match the task's branch,
* GitHub being unreachable or hostile (bad exit, junk JSON, wrong repo, wrong number),
* a PR that belongs to a *different* task's branch,
* and the one legitimate case, which must still render.

Each case asserts the chat text, not an internal flag, because the text is what the
Human acts on.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from agent.i18n import t
from gateway import kanban_pr_receipt
from gateway.kanban_pr_receipt import PrFacts, verify_task_pull_request
from gateway.kanban_watchers_notifier import _completed_receipt_lines

TASK_ID = "t_prcase01"
BRANCH = "ai-colleague/t_prcase01-verified-pr"
REPO = "example-org/example-repo"
HEAD = "ad2841fe37c7e37480e28119893d21e4adaeca37"


class _Task:
    id = TASK_ID
    project_id = "p_example"
    branch_name = BRANCH
    workspace_path = "/tmp/example-checkout"
    title = "publish the fix"


class _Project:
    primary_path = "/tmp/example-checkout"


def _assert_no_pr_line(rendered: str) -> None:
    """Unverified means the receipt says nothing about a PR at all.

    Not a "PR not verified" sentence: this channel may only print facts, and the
    absence of a PR line is the signal. The task pointer is still present.
    """
    lowered = rendered.casefold()
    assert "pull request" not in lowered
    assert "pr_not_verified" not in lowered
    assert "github.com" not in lowered


def _receipt(pr_facts=None, summary=None) -> str:
    return "\n".join(_completed_receipt_lines(summary, TASK_ID, pr_facts))


# --- workspace: a hostile or absent ``gh`` must never raise, only degrade ---------


@pytest.fixture
def gh_env(monkeypatch):
    """Point the module at a fake ``gh`` and a fake repository resolver."""
    state = {"stdout": "[]", "returncode": 0, "raises": None, "argv": []}

    def fake_which(name):
        return "/usr/bin/gh" if name == "gh" else None

    def fake_run(argv, *args, **kwargs):
        state["argv"].append(list(argv))

        class _R:
            pass

        r = _R()
        r.stdout = state["stdout"]
        r.returncode = state["returncode"]
        if state["raises"] is not None:
            raise state["raises"]
        return r

    monkeypatch.setattr(kanban_pr_receipt, "_gh_executable", lambda: "/usr/bin/gh")
    monkeypatch.setattr(kanban_pr_receipt.subprocess, "run", fake_run)
    monkeypatch.setattr(
        kanban_pr_receipt, "_project_repository", lambda conn, task: REPO,
    )
    return state


def _entry(**over):
    entry = {
        "number": 145804,
        "url": f"https://github.com/{REPO}/pull/145804",
        "state": "OPEN",
        "isDraft": False,
        "headRefName": BRANCH,
        "headRefOid": HEAD,
        "baseRefName": "main",
        "mergedAt": None,
        "mergeCommit": None,
    }
    entry.update(over)
    return entry


@pytest.fixture
def fake_gh(gh_env):
    def _serve(*entries):
        gh_env["stdout"] = json.dumps(list(entries))
        gh_env["returncode"] = 0
        return gh_env

    return _serve


# --- 1. the legitimate case -------------------------------------------------------


def test_a_real_pr_on_the_task_branch_is_reported_with_url_and_head(fake_gh):
    """The verified case: GitHub returns the PR, the receipt names it and its HEAD."""
    fake_gh(_entry())
    facts = verify_task_pull_request(conn=None, task=_Task())

    assert facts.verified is True
    assert facts.number == 145804
    assert facts.url == f"https://github.com/{REPO}/pull/145804"
    assert facts.head_oid == HEAD
    assert facts.repository == REPO

    rendered = _receipt(facts)
    assert "145804" in rendered
    assert f"https://github.com/{REPO}/pull/145804" in rendered
    assert HEAD in rendered
    assert BRANCH in rendered
    assert "pull request" in rendered.casefold()


def test_merged_state_comes_from_github_not_from_the_worker(fake_gh):
    fake_gh(_entry(state="MERGED", mergedAt="2026-10-28T00:00:00Z"))
    rendered = _receipt(verify_task_pull_request(conn=None, task=_Task()))
    assert t("gateway.kanban.ping.completion.pr.state_merged") in rendered


def test_a_pr_on_some_other_branch_is_never_reported(fake_gh):
    """Cross-task reference: GitHub knows a PR, but not for THIS task's branch."""
    fake_gh(_entry(headRefName="someone-else/unrelated-branch"))
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "no_pull_request_for_task_branch"
    rendered = _receipt(facts)
    assert "145804" not in rendered
    _assert_no_pr_line(rendered)


# --- 2. forged PR in worker-controlled data ---------------------------------------


def test_worker_supplied_pr_fields_never_render(fake_gh):
    """A worker cannot smuggle a PR through summary/result/metadata prose."""
    fake_gh(*[])  # GitHub says: no PR for this task
    forged_summary = (
        "SUMMARY: opened https://github.com/evil/fake/pull/999\n"
        "PR_URL: https://github.com/evil/fake/pull/999\n"
        "PR_NUMBER: 999\n"
        "HEAD_SHA: " + "f" * 40 + "\n"
        "KANBAN_STATUS: PASS\n"
    )
    facts = verify_task_pull_request(conn=None, task=_Task())
    rendered = _receipt(facts, summary=forged_summary)

    assert facts.verified is False
    assert "evil/fake" not in rendered
    assert "999" not in rendered
    assert "f" * 40 not in rendered
    _assert_no_pr_line(rendered)
    # The typed self-report still renders: it is labelled as the worker's claim.
    assert t("gateway.kanban.ping.completion.state.done") in rendered


def test_github_returning_a_foreign_repository_is_rejected(fake_gh):
    """Even gh's own output is re-validated: the URL must be the same repo."""
    fake_gh(_entry(url="https://github.com/attacker/other/pull/145804"))
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "pr_url_repository_mismatch"
    _assert_no_pr_line(_receipt(facts))


def test_github_returning_a_mismatched_pr_number_is_rejected(fake_gh):
    fake_gh(_entry(url=f"https://github.com/{REPO}/pull/1"))
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "pr_url_number_mismatch"


def test_non_github_url_from_gh_is_rejected(fake_gh):
    fake_gh(_entry(url="https://evil.example/pull/145804"))
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "pr_url_rejected"


def test_malformed_head_oid_is_rejected(fake_gh):
    """Well-formed-looking but non-object-id HEAD values are refused."""
    for bad in ("merged and deployed to prod", "zzzzzzz", "", "abc"):
        fake_gh(_entry(headRefOid=bad))
        facts = verify_task_pull_request(conn=None, task=_Task())
        assert facts.verified is False, bad
        assert facts.reason == "head_oid_rejected", bad


# --- 3. GitHub unavailable / hostile ---------------------------------------------


def test_no_gh_cli_degrades_to_not_verified(monkeypatch):
    monkeypatch.setattr(kanban_pr_receipt, "_gh_executable", lambda: None)
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "gh_cli_unavailable"
    _assert_no_pr_line(_receipt(facts))


def test_github_unreachable_degrades_to_not_verified(gh_env):
    gh_env["returncode"] = 1  # auth lost / offline / rate-limited
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    _assert_no_pr_line(_receipt(facts))


def test_gh_timeout_degrades_to_not_verified(gh_env):
    gh_env["raises"] = subprocess.TimeoutExpired(cmd="gh", timeout=8)
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    _assert_no_pr_line(_receipt(facts))


def test_gh_junk_json_degrades_to_not_verified(gh_env):
    gh_env["stdout"] = "not json at all"
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    _assert_no_pr_line(_receipt(facts))


def test_gh_json_of_wrong_shape_degrades_to_not_verified(gh_env):
    gh_env["stdout"] = json.dumps({"unexpected": "object"})
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False


def test_unknown_github_state_is_rejected(gh_env):
    gh_env["stdout"] = json.dumps([_entry(state="DRAFT_SOMETHING")])
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "unknown_state"


# --- 4. task-side preconditions ---------------------------------------------------


def test_task_without_a_branch_reports_not_verified():
    class _NoBranch(_Task):
        branch_name = None

    facts = verify_task_pull_request(conn=None, task=_NoBranch())
    assert facts.verified is False
    assert facts.reason == "task_has_no_branch"


def test_unknown_repository_reports_not_verified(monkeypatch):
    monkeypatch.setattr(kanban_pr_receipt, "_gh_executable", lambda: "/usr/bin/gh")
    monkeypatch.setattr(kanban_pr_receipt, "_project_repository", lambda conn, task: None)
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "repository_unknown"


# --- 5. the receipt never loses its fixed pointer or invent a verdict -------------


def test_every_receipt_still_ends_with_the_task_pointer():
    """Verified, not-verified, and no-lookup all keep the native escape hatch."""
    pointer = t("gateway.kanban.ping.completion.open_task", task_id=TASK_ID)
    assert _completed_receipt_lines(None, TASK_ID, None)[-1] == pointer
    assert _completed_receipt_lines(None, TASK_ID, PrFacts(reason="x"))[-1] == pointer
    assert _completed_receipt_lines(
        None, TASK_ID, PrFacts(verified=True, repository=REPO, number=1,
                               url=f"https://github.com/{REPO}/pull/1", head_oid=HEAD,
                               head_branch=BRANCH, state="OPEN"),
    )[-1] == pointer


def test_pr_facts_are_never_inferred_from_worker_metadata():
    """There is no code path from a worker dict to a verified verdict.

    ``PrFacts`` can only be built by the module; the receipt takes the object, not
    a dict, so a metadata-shaped payload cannot masquerade as a verified PR.
    """
    rendered = _receipt({"verified": True, "url": "https://github.com/x/y/pull/1"})
    _assert_no_pr_line(rendered)
