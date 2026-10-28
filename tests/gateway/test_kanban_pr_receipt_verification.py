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
import re
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
    """Point the module at a fake ``gh`` that answers REST-shaped pull payloads."""
    state = {
        "pulls": [], "returncode": 0, "raises": None, "argv": [],
        "identity": "task-owner",
    }

    def fake_run(argv, *args, **kwargs):
        state["argv"].append(list(argv))
        if state["raises"] is not None:
            raise state["raises"]

        class _R:
            stdout = ""
            returncode = 0

        r = _R()
        if "api" in argv and "user" in argv:
            r.stdout = state["identity"]
            r.returncode = 0 if state["identity"] else 1
            return r
        path = next((a for a in argv if a.startswith("repos/")), "")
        r.returncode = state["returncode"]
        if r.returncode != 0:
            return r
        entries = list(state["pulls"])
        m = re.search(r"[?&]head=([^&]+)", path)
        if m:
            want = m.group(1)

            def head_of(entry):
                head = entry.get("head") if isinstance(entry.get("head"), dict) else {}
                repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
                owner = repo.get("owner") if isinstance(repo.get("owner"), dict) else {}
                return head.get("ref"), owner.get("login")

            if ":" in want:
                owner, want_branch = want.split(":", 1)
                entries = [
                    e for e in entries
                    if head_of(e) == (want_branch, owner)
                ]
            else:
                entries = [e for e in entries if head_of(e)[0] == want]
        r.stdout = json.dumps(entries)
        return r

    monkeypatch.setattr(kanban_pr_receipt, "_gh_executable", lambda: "/usr/bin/gh")
    monkeypatch.setattr(kanban_pr_receipt.subprocess, "run", fake_run)
    monkeypatch.setattr(
        kanban_pr_receipt, "_project_repository", lambda conn, task: REPO,
    )
    return state


def _rest(number=145804, *, state="open", draft=False, merged_at=None, owner="task-owner",
          ref=None, sha=None, repo=None):
    """A GitHub REST pull payload (the shape ``gh api .../pulls`` returns)."""
    ref = BRANCH if ref is None else ref
    sha = HEAD if sha is None else sha
    repo = repo if repo is not None else f"{owner}/{REPO.split('/')[1]}"
    return {
        "number": number,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "state": state,
        "draft": draft,
        "merged_at": merged_at,
        "head": {
            "ref": ref,
            "sha": sha,
            "repo": {"full_name": repo, "owner": {"login": owner}},
        },
        "base": {"ref": "main"},
    }


def _entry(**over):
    """Backwards-compatible builder used by the earlier tests."""
    merged = over.pop("mergedAt", None)
    owner = over.pop("headRepositoryOwner", None)
    owner_login = owner.get("login") if isinstance(owner, dict) else None
    repo = over.pop("headRepository", None)
    repo_name = repo.get("nameWithOwner") if isinstance(repo, dict) else None
    return _rest(
        over.pop("number", 145804),
        state=str(over.pop("state", "OPEN")).lower(),
        draft=bool(over.pop("isDraft", False)),
        merged_at=merged,
        owner=owner_login or "task-owner",
        ref=over.pop("headRefName", BRANCH),
        sha=over.pop("headRefOid", HEAD),
        repo=repo_name,
    )


@pytest.fixture
def fake_gh(gh_env):
    def _serve(*entries):
        gh_env["pulls"] = list(entries)
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
    entry = _rest()
    entry["html_url"] = "https://github.com/attacker/other/pull/145804"
    fake_gh(entry)
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "pr_url_repository_mismatch"
    _assert_no_pr_line(_receipt(facts))


def test_github_returning_a_mismatched_pr_number_is_rejected(fake_gh):
    entry = _rest()
    entry["html_url"] = f"https://github.com/{REPO}/pull/1"
    fake_gh(entry)
    facts = verify_task_pull_request(conn=None, task=_Task())
    assert facts.verified is False
    assert facts.reason == "pr_url_number_mismatch"


def test_non_github_url_from_gh_is_rejected(fake_gh):
    entry = _rest()
    entry["html_url"] = "https://evil.example/pull/145804"
    fake_gh(entry)
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
    gh_env["pulls"] = [_rest(state="draft_something")]
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


# --- 6. fork identity binding ----------------------------------------------------
#
# A branch name is NOT an identity: it is derived from the project slug and task
# id, so it is predictable, and two different forks can both hold a branch with
# that name. Before this binding, a bare ``--head <branch>`` query returned both,
# and the newest-state preference adopted the MERGED one — reporting another
# fork's old work as this task's pull request.


def test_an_old_merged_pr_from_another_fork_is_never_adopted(fake_gh):
    """Two forks, same branch name: the task's own OPEN PR wins, not the stale MERGED one."""
    foreign_merged = _rest(number=900, state="merged", merged_at="2026-10-01T00:00:00Z",
                           owner="other-fork", sha="a" * 40)
    own_open = _rest(number=950, state="open")
    # GitHub holds both; the per-owner query must return only the task owner's.
    gh_env_state = fake_gh(foreign_merged, own_open)

    facts = verify_task_pull_request(conn=None, task=_Task())

    assert facts.verified is True
    assert facts.number == 950, "adopted the wrong fork's PR"
    assert facts.head_owner == "task-owner"
    rendered = _receipt(facts)
    assert "pull/950" in rendered
    assert "pull/900" not in rendered


def test_the_query_is_scoped_to_the_task_owner_not_a_bare_branch(fake_gh):
    """Every published query must carry ``owner:branch``; a bare branch is never used.

    A bare ``head=<branch>`` returns PRs from *any* fork, so using it would put a
    stranger's same-named branch in play. The owner comes from identity, not from
    whatever branches happen to exist.
    """
    gh_env_state = fake_gh(_entry())
    verify_task_pull_request(conn=None, task=_Task())

    paths = [a for argv in gh_env_state["argv"] for a in argv if a.startswith("repos/")]
    assert paths, "no pull query was made"
    for path in paths:
        assert f"head=task-owner:{BRANCH}" in path, path
        assert f"head={BRANCH}&" not in path and not path.endswith(f"head={BRANCH}"), path


def test_an_impostor_pr_is_dropped_while_the_tasks_own_pr_still_verifies(fake_gh):
    """The impostor is not merely "not found" — the real PR must still win.

    A test that only asserts "the impostor renders nothing" passes on the
    vulnerable tree too, so it proves nothing. This asserts the positive: with a
    hostile fork's PR present, the receipt still names the task's own PR.
    """
    impostor = _rest(number=777, owner="someone-else")
    own = _rest(number=950, owner="task-owner")
    gh_env_state = fake_gh(impostor, own)
    gh_env_state["identity"] = "task-owner"

    facts = verify_task_pull_request(conn=None, task=_Task())

    assert facts.verified is True
    assert facts.number == 950
    assert facts.head_owner == "task-owner"
    rendered = _receipt(facts)
    assert "pull/950" in rendered
    assert "pull/777" not in rendered


def test_a_deleted_fork_fails_closed_without_raising(fake_gh):
    """``head.repo`` is null when a fork is deleted; that must not crash or verify."""
    deleted = _rest()
    deleted["head"]["repo"] = None
    gh_env_state = fake_gh(deleted)
    gh_env_state["identity"] = "task-owner"

    facts = verify_task_pull_request(conn=None, task=_Task())

    assert facts.verified is False
    _assert_no_pr_line(_receipt(facts))


def test_a_pr_whose_owner_field_is_null_fails_closed(fake_gh):
    """``head.repo.owner`` may also be null; same rule."""
    entry = _rest()
    entry["head"]["repo"]["owner"] = None
    gh_env_state = fake_gh(entry)
    gh_env_state["identity"] = "task-owner"

    facts = verify_task_pull_request(conn=None, task=_Task())

    assert facts.verified is False
    _assert_no_pr_line(_receipt(facts))


def test_unresolvable_fork_identity_fails_closed(gh_env):
    """No PR and no authenticated identity: the owner is unknown, so say nothing."""
    gh_env["identity"] = ""
    gh_env["pulls"] = []
    facts = verify_task_pull_request(conn=None, task=_Task())

    assert facts.verified is False
    assert facts.reason == "head_owner_unresolved"
    _assert_no_pr_line(_receipt(facts))


def test_upstream_owner_cannot_be_used_as_the_head_fork(fake_gh):
    """A same-named branch in the upstream org itself is someone else's work."""
    upstream_pr = _rest(number=555, owner=REPO.split("/")[0],
                        repo=f"{REPO.split('/')[0]}/{REPO.split('/')[1]}")
    gh_env_state = fake_gh(upstream_pr)
    facts = verify_task_pull_request(conn=None, task=_Task())

    # The upstream owner is excluded from the identity candidates, so this PR is
    # never adopted even though its branch name matches.
    assert facts.verified is False
    assert facts.head_owner is None


def test_verified_pr_reports_the_fork_it_came_from(fake_gh):
    """The receipt names the fork, so a Human can see where the PR was pushed from."""
    fake_gh(_entry())
    rendered = _receipt(verify_task_pull_request(conn=None, task=_Task()))
    assert "task-owner" in rendered
