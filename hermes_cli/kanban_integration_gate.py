"""Integration-gate verification — the external-evidence half.

A gate card sits below an Implementation card and its QA card and above the
next Implementation card. Completing it asserts the one thing the board cannot
see by itself: the implementation QA passed is *actually integrated* — merged
by a human into the configured integration branch, and present in that
branch's freshly fetched tip. Nothing here merges, pushes or writes to GitHub.

Every condition is proven from outside the board (GitHub's PR record + the
local clone's git objects) and **any condition that cannot be proven blocks
the gate**. "The API was unreachable" is therefore never "not merged yet": the
receipt's phase separates them, because an operator waiting for a merge and an
operator whose ``gh`` auth expired need different actions. Evidence that cannot
even be parsed as a pull request joins that first family — a malformed record
is checked for shape before any field is read, so it blocks with a receipt
rather than raising out of the middle of a completion.

Two things the gate is careful not to trust: the PR record's identity (the head
it reports must still be the head acceptance passed and QA reviewed, or the
merge integrated something nobody judged) and a git REF (mutable — the tracking
ref is resolved once to an exact commit, and that captured commit is what the
ancestry question and the receipt both name).

No SQLite transaction is open while this runs; the store module snapshots the
board facts first and rechecks that snapshot before the terminal transition.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import ClassVar

from hermes_cli.kanban_github_evidence import is_sha, nonblank_str, pull_request_problem
from hermes_cli.kanban_pr_acceptance import _API_FAILURES, _PR

_GIT_TIMEOUT = 120
#: Phases whose failure means "GitHub/git could not answer", not "not yet integrated".
UNPROVABLE_PHASES = frozenset({
    "pr_unreadable", "pr_evidence_malformed", "fetch_failed", "ancestry_unprovable",
})

_RECOVERY = {
    "waiting_for_merge": "Merge the implementation PR into the configured integration branch, "
                         "then retry the gate.",
    "pr_unreadable": "Check gh authentication/API access for the implementation PR, then retry "
                     "the gate; nothing about the merge is known yet.",
    "pr_evidence_malformed": "GitHub's answer for the implementation PR could not be read as a "
                             "pull request. Check gh/API access (a proxy or error envelope in "
                             "place of the record), then retry; nothing about the merge is known.",
    "fetch_failed": "Fix the configured remote/branch or network access for the gate repository "
                    "(hermes kanban integration-gate show), then retry.",
    "ancestry_unprovable": "The merge commit is not reachable from the fetched integration branch "
                           "(force-push, wrong branch, or a revert). Investigate before retrying.",
    "declared_parents_linked": "Restore the declared edge with `hermes kanban link <parent> "
                               "<gate>`, or drop the declaration with `hermes kanban "
                               "integration-gate rm <gate>` if the card is an ordinary one now.",
    "pr_head_matches_accepted_head": "The PR moved on after the acceptance and QA this gate "
                                     "proves. Re-run the implementation's acceptance and its QA "
                                     "against the new head, then retry the gate.",
}
_DEFAULT_RECOVERY = ("Satisfy the failing gate condition above, then retry completion. Use "
                     "kanban_block if human input is needed; receipts remain on the task event log.")


@dataclass(frozen=True)
class GateConfig:
    """One declared gate (a row of ``integration_gates``)."""

    gate_task_id: str
    implementation_task_id: str
    qa_task_id: str
    repository_path: str
    integration_remote: str
    integration_branch: str

    #: A human merger is MANDATORY for every declared gate — a class constant,
    #: not a field, so there is no per-gate value to configure, persist, or get
    #: wrong. Automation merging its own unreviewed work is precisely the
    #: failure a gate exists to catch, so "a bot merged it" can never be the
    #: thing that lets a downstream card start. Rows an earlier build wrote with
    #: ``require_human_merge = 0`` are therefore read as mandatory-human too:
    #: the store never reads that column (see
    #: ``kanban_integration_gate_store._GATE_COLUMNS``).
    require_human_merge: ClassVar[bool] = True

    def as_dict(self) -> dict:
        return {
            "gate_task_id": self.gate_task_id,
            "implementation_task_id": self.implementation_task_id,
            "qa_task_id": self.qa_task_id,
            "repository_path": self.repository_path,
            "integration_remote": self.integration_remote,
            "integration_branch": self.integration_branch,
            # Reported on every receipt and every ``show``: a reader of an old
            # receipt should not have to guess whether the policy applied.
            "require_human_merge": self.require_human_merge,
        }

    @property
    def tracking_ref(self) -> str:
        return f"refs/remotes/{self.integration_remote}/{self.integration_branch}"


#: Synthetic exit code for a git invocation that never produced one. 128 is
#: git's own "fatal", and it reads as "could not decide" rather than the
#: ``is-ancestor`` answer 1 ("decided: no") — the gate must not report a
#: missing git binary or a hung fetch as "that commit is not in the branch".
_GIT_UNAVAILABLE = 128


def _git(repository_path: str, *args: str) -> subprocess.CompletedProcess:
    """Run one git command in the gate's clone.

    Never raises: a non-zero exit, a missing git binary and a timeout all come
    back as a ``CompletedProcess`` so the caller decides whether the code means
    "false" or "unprovable". Letting any of them escape would abort
    ``complete_task`` with a traceback instead of the blocking receipt that
    tells an operator which condition could not be proven.
    """
    try:
        return subprocess.run(
            ["git", "-C", repository_path, *args], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=_GIT_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # Never surface git's stderr/output (host and credential detail); the
        # unproven condition is the actionable fact.
        return subprocess.CompletedProcess(args, _GIT_UNAVAILABLE, stdout="", stderr="")


def collect_gate_acceptance(config: GateConfig, board: dict) -> dict:
    """Verify every gate condition; returns the ``integration_acceptance`` receipt.

    ``board`` is the snapshot the store read under its connection:
    ``implementation_linked``, ``qa_linked``, ``implementation_status``,
    ``implementation_assignee``, ``qa_status``, ``qa_run_id``, ``qa_decision``,
    ``qa_revision``,
    ``qa_summary_mentions_pass``, ``accepted_head_sha``, ``pr_url``,
    ``contract``.
    """
    receipt: dict = {
        "ok": False, "phase": "board_state", "conditions": [],
        **config.as_dict(),
        "pr_url": board.get("pr_url"),
        "accepted_head_sha": board.get("accepted_head_sha"),
        "qa_run_id": board.get("qa_run_id"),
        "qa_verdict": board.get("qa_decision"),
        "qa_revision": board.get("qa_revision"),
        "base_ref": None, "pr_head_sha": None, "merge_commit_sha": None, "merged_at": None,
        "merged_by": None, "fetched_branch_tip": None,
        "verified_at": int(time.time()), "recovery": _DEFAULT_RECOVERY,
    }

    def condition(name: str, ok: bool, detail: str, *, phase: str | None = None) -> bool:
        receipt["conditions"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            receipt["phase"] = phase or name
            receipt["detail"] = detail
            receipt["recovery"] = _RECOVERY.get(receipt["phase"], _DEFAULT_RECOVERY)
        return bool(ok)

    # The declaration names two PARENTS, and an edge is an ordinary row: it can
    # be unlinked (or never restored after a graph edit) while the declaration
    # survives. A gate whose declared parents are no longer its parents has
    # nothing to gate — and with zero parent edges the board's own dependency
    # check is vacuously satisfied, which is exactly the case that must not
    # read as "ready to integrate".
    missing_edges = [parent for parent, linked in (
        (config.implementation_task_id, board.get("implementation_linked")),
        (config.qa_task_id, board.get("qa_linked")),
    ) if not linked]
    if not condition(
        "declared_parents_linked", not missing_edges,
        f"gate {config.gate_task_id} declares implementation "
        f"{config.implementation_task_id} and QA {config.qa_task_id} as its parents, but "
        f"{' and '.join(missing_edges)} {'is' if len(missing_edges) == 1 else 'are'} no longer "
        f"linked to it; a gate is never satisfied by a parent edge that is gone",
    ):
        return receipt
    if not condition(
        "implementation_done", board.get("implementation_status") == "done",
        f"implementation {config.implementation_task_id} is "
        f"{board.get('implementation_status') or 'missing'}, not done",
    ):
        return receipt
    if not condition(
        "qa_done", board.get("qa_status") == "done",
        f"QA {config.qa_task_id} is {board.get('qa_status') or 'missing'}, not done",
    ):
        return receipt
    # Structured verdict only: a run whose prose says "PASS" but whose metadata
    # carries no decision is an unverifiable handoff, not a pass.
    if not condition(
        "qa_verdict_pass", board.get("qa_decision") == "PASS",
        f"latest completed QA run {board.get('qa_run_id')} has decision="
        f"{board.get('qa_decision')!r}; the gate reads metadata.decision == \"PASS\" and never "
        f"prose" + (" (its summary mentions PASS, which is not evidence)"
                    if board.get("qa_summary_mentions_pass") and not board.get("qa_decision") else ""),
    ):
        return receipt
    accepted = board.get("accepted_head_sha")
    if not condition(
        "accepted_head_known", is_sha(accepted),
        f"no accepted exact-head PR acceptance receipt on implementation "
        f"{config.implementation_task_id}; the gate compares QA's revision against the head "
        f"GitHub acceptance actually passed",
    ):
        return receipt
    if not condition(
        "qa_revision_matches_accepted_head", board.get("qa_revision") == accepted,
        f"QA reviewed revision {board.get('qa_revision')!r}, but the accepted implementation head "
        f"is {accepted}",
    ):
        return receipt
    pr_url = board.get("pr_url") or ""
    match = _PR.fullmatch(pr_url)
    if not condition(
        "implementation_pr_pinned", bool(match) and pr_url == board.get("contract"),
        f"implementation {config.implementation_task_id} is not pinned to an exact GitHub PR "
        f"(contract={board.get('contract')!r}, accepted receipt pr_url={pr_url!r})",
    ):
        return receipt

    pr = _read_pull_request(match[1], int(match[2]), board.get("implementation_assignee"))
    if not condition("implementation_pr_readable", pr is not None,
                     "GitHub could not be read for the implementation PR, so nothing about its "
                     "merge state is known", phase="pr_unreadable"):
        return receipt
    # Every field below is dereferenced, and one of them (``merged``) is read
    # for truth: ``merged: "false"`` is a TRUTHY string. So the record's shape
    # is checked BEFORE anything reads it, and an answer that is not a pull
    # request blocks as unprovable instead of raising mid-completion.
    structure = pull_request_problem(pr)
    if not condition(
        "pr_evidence_well_formed", structure is None,
        f"GitHub's record for {pr_url} is not usable evidence: {structure}",
        phase="pr_evidence_malformed",
    ):
        return receipt
    # GitHub sends ``null`` for an unmerged PR; anything that is not an object
    # is treated as "no known actor", which fails the human-merge condition.
    merged_by_raw = pr.get("merged_by")
    merged_by = merged_by_raw if isinstance(merged_by_raw, dict) else {}
    receipt.update(
        base_ref=pr["base"]["ref"],
        pr_head_sha=pr["head"]["sha"],
        merge_commit_sha=pr.get("merge_commit_sha"),
        merged_at=pr.get("merged_at"),
        merged_by={"login": merged_by.get("login"), "type": merged_by.get("type")},
    )
    # The gate proves the integration of the head that was ACCEPTED and
    # QA-reviewed. A PR that moved on afterwards and was then merged integrated
    # something nobody judged, and the merge commit alone cannot show that: a
    # squash merge of the newer head looks exactly like a squash merge of the
    # accepted one.
    if not condition(
        "pr_head_matches_accepted_head", receipt["pr_head_sha"] == accepted,
        f"PR {pr_url} now has head {receipt['pr_head_sha']}, but the accepted and QA-reviewed "
        f"head is {accepted}; the PR moved on after the evidence this gate rests on",
    ):
        return receipt
    # ``is True``, not truthiness: the shape check above already refuses a
    # non-boolean, and this keeps that the only reading of a merge.
    if not condition("implementation_pr_merged", pr["merged"] is True,
                     f"PR {pr_url} is {pr['state']} and not merged",
                     phase="waiting_for_merge"):
        return receipt
    if not condition(
        "pr_base_is_integration_branch", receipt["base_ref"] == config.integration_branch,
        f"PR {pr_url} merged into {receipt['base_ref']!r}, not the configured integration branch "
        f"{config.integration_branch!r}",
    ):
        return receipt
    # Unconditional: no flag, config key or stored column can turn this off.
    if not condition(
        "human_merge_actor", _is_human_actor(merged_by_raw),
        f"PR {pr_url} was merged by {merged_by.get('login')!r} "
        f"(type={merged_by.get('type')!r}); an integration gate always requires a human merger",
    ):
        return receipt
    merge_sha = receipt["merge_commit_sha"]
    # Squash merges rewrite the commit, so the PR head is NOT in the branch —
    # merge_commit_sha is the only commit that is.
    if not condition(
        "merge_commit_sha_valid", is_sha(merge_sha),
        f"GitHub reports merge_commit_sha={merge_sha!r} for {pr_url}; without an exact commit sha "
        f"the merge cannot be located in the integration branch",
    ):
        return receipt

    fetch = _git(config.repository_path, "fetch", "--quiet", config.integration_remote,
                 f"+refs/heads/{config.integration_branch}:{config.tracking_ref}")
    if not condition(
        "integration_branch_fetched", fetch.returncode == 0,
        f"git fetch {config.integration_remote} {config.integration_branch} failed in "
        f"{config.repository_path} (exit {fetch.returncode})", phase="fetch_failed",
    ):
        return receipt
    # Resolve the tracking ref ONCE, to an exact commit, and name that commit
    # from here on. A ref is mutable: another process fetching or pushing in the
    # same clone can move it between this read and the ancestry question, so
    # asking git about the REF would prove the merge against a tip the receipt
    # never recorded — and ``^{commit}`` plus the sha check keeps a tag or a
    # tree from standing in for one.
    tip = _git(config.repository_path, "rev-parse", "--verify", "--quiet",
               f"{config.tracking_ref}^{{commit}}")
    captured_tip_sha = tip.stdout.strip() if tip.returncode == 0 else ""
    if not condition(
        "integration_branch_tip_readable", is_sha(captured_tip_sha),
        f"{config.tracking_ref} did not resolve to an exact commit sha after fetching "
        f"(exit {tip.returncode})", phase="fetch_failed",
    ):
        return receipt
    receipt["fetched_branch_tip"] = captured_tip_sha
    branch_label = f"{config.integration_remote}/{config.integration_branch} tip {captured_tip_sha}"
    ancestry = _git(config.repository_path, "merge-base", "--is-ancestor",
                    merge_sha, captured_tip_sha)
    if ancestry.returncode not in (0, 1):
        # Exit 0/1 are the answer; anything else (unknown object, corrupt repo)
        # is "cannot prove", which must not read as "not merged".
        condition("merge_commit_in_integration_branch", False,
                  f"git merge-base --is-ancestor could not decide whether {merge_sha} is in the "
                  f"fetched {branch_label} (exit {ancestry.returncode})",
                  phase="ancestry_unprovable")
        return receipt
    if not condition(
        "merge_commit_in_integration_branch", ancestry.returncode == 0,
        f"merge commit {merge_sha} is not an ancestor of the fetched {branch_label}",
    ):
        return receipt
    receipt.update(ok=True, phase="integrated", detail="every gate condition verified",
                   recovery=_DEFAULT_RECOVERY)
    return receipt


def describe_receipt(receipt: dict | None) -> list[str]:
    """Human-readable condition-by-condition rendering of a gate verification.

    Shared by ``integration-gate show`` and the diagnostics detail text so
    "waiting for a human merge" and "GitHub/git could not be read" never look
    the same to an operator.
    """
    from hermes_cli.kanban_output import _fmt_ts

    if not receipt:
        return ["  Latest verification: none yet (the gate has not been completed)."]
    headline = ("verified — every condition proven" if receipt.get("ok")
                else f"{'UNPROVABLE' if receipt.get('phase') in UNPROVABLE_PHASES else 'BLOCKED'} "
                     f"at {receipt.get('phase')}")
    lines = [f"  Latest verification ({_fmt_ts(receipt.get('verified_at'))}): {headline}"]
    for condition in receipt.get("conditions") or []:
        mark = "✓" if condition.get("ok") else "✗"
        lines.append(f"    {mark} {condition.get('name')}: {condition.get('detail')}")
    if not receipt.get("ok"):
        lines.append("    · later conditions were not evaluated (the gate stops at the first "
                     "unproven one)")
        if receipt.get("recovery"):
            lines.append(f"  Next step: {receipt['recovery']}")
    for label, key in (("PR", "pr_url"), ("PR head", "pr_head_sha"),
                       ("accepted head", "accepted_head_sha"),
                       ("QA revision", "qa_revision"), ("merge commit", "merge_commit_sha"),
                       ("merged by", "merged_by"), ("fetched tip", "fetched_branch_tip")):
        value = receipt.get(key)
        if isinstance(value, dict):
            value = value.get("login") and f"{value.get('login')} ({value.get('type')})"
        if value:
            lines.append(f"  {label}: {value}")
    return lines


def _read_pull_request(repo: str, number: int, assignee: str | None):
    """GitHub's answer for the PR, or None when it could not be read at all.

    The answer is returned as decoded, without a shape check: "the API did not
    answer" and "the API answered something that is not a pull request" are
    different facts with different receipts, so judging the shape stays with
    the condition that reports it.

    ``_api`` is resolved at call time from its defining module so both halves
    of the completion gate talk to GitHub through one patchable seam.

    The read runs as the IMPLEMENTATION card's assignee profile, exactly like
    the acceptance half (#122689): this is that card's PR, and a gate card may
    be completed by a reviewer, an operator or the dispatcher, none of whose
    ambient ``gh`` logins is the identity that can see a private repo.
    :class:`_GateAuthError` joins the transport failures here because the gate
    reports "GitHub could not be read" as one unprovable condition rather than
    a second verdict — an expired login and an unreachable API need the same
    operator action, which ``_RECOVERY["pr_unreadable"]`` already names.
    """
    from hermes_cli.kanban_pr_acceptance import _api, _assignee_profile_home, _GateAuthError

    try:
        return _api(f"repos/{repo}/pulls/{number}",
                    profile_home=_assignee_profile_home(assignee))
    except (_GateAuthError, *_API_FAILURES):
        # Never persist gh stderr (credentials/host details).
        return None


_BOT_LOGIN = re.compile(r".*\[bot\]$", re.IGNORECASE)


def _is_human_actor(actor) -> bool:
    """A merger GitHub attributes to a Bot (or a ``…[bot]`` login, or nobody)
    is not a human approval. Unknown actor type fails closed.

    The whole value is typed here rather than upstream: GitHub sends an object
    or ``null``, so a string, a number or a list is "no known actor" — the same
    answer as a bot, and reported as the human-merge condition it fails, which
    is what an operator needs to read.
    """
    if not isinstance(actor, dict):
        return False
    login = nonblank_str(actor.get("login"))
    if login is None or _BOT_LOGIN.match(login):
        return False
    return actor.get("type") == "User"
