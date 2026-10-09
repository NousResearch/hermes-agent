"""Shape of the GitHub JSON both completion gates consume.

``gh api`` output is JSON decoded from a subprocess, and the gates dereference
it field by field: ``pr["head"]["sha"]``, ``pr["base"]["ref"]``,
``pr["merged"]``. A proxy's error envelope, an enterprise host's older schema or
a future field rename can put a string, a list or ``null`` where an object
belongs — and ``merged: "false"`` is a TRUTHY string, so an untyped read of it
accepts a merge that never happened.

Every field is therefore checked for shape HERE, once, before anything
dereferences it, so a malformed answer blocks the gate with a receipt instead
of raising a ``KeyError``/``TypeError`` out of the middle of a completion. The
problem comes back as a short operator-facing sentence that never quotes the
response body (which can carry tokens or host detail).

Fields only a MERGED pull request carries (``merged_by``,
``merge_commit_sha``, ``merged_at``) are ``null`` on an open PR, so they are
NOT validated here: each is the subject of its own gate condition, which is
what lets a receipt say "merged by a bot" or "no usable merge commit" instead
of a flat "malformed".

The acceptance gate's first hop is GraphQL rather than REST, and a GraphQL
answer nests four levels deep (``data.repository.pullRequest.baseRef``) before
the first field the gate reads. Every level is an object GitHub *may* answer
``null`` for, and an enterprise host, a proxy or a partial-error envelope can
put a string or a list there instead — so that envelope gets the same
before-any-dereference treatment, and the validator hands back the
``(context, app id)`` projection the gate keys its required set by rather than
letting a second traversal re-read the response under weaker assumptions.
"""
from __future__ import annotations

import re

#: A git object name as GitHub reports it: exactly 40 lowercase hex digits.
SHA_RE = re.compile(r"[0-9a-f]{40}")

#: Every value REST reports for a pull request's ``state`` — the whole
#: contract, lowercase. A tuple, not a set: membership then compares by
#: equality, so an unhashable answer (a list) is refused rather than raising.
_REST_PR_STATES = ("open", "closed")


def is_sha(value) -> bool:
    """True only for an exact 40-character hex object name."""
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def nonblank_str(value):
    """``value`` when it is a non-blank string, else None."""
    return value if isinstance(value, str) and value.strip() else None


def _object(container: dict, key: str):
    value = container.get(key)
    return value if isinstance(value, dict) else None


class _Malformed(Exception):
    """The first unreadable field, described for an operator.

    Internal to this module: every public entry point converts it into the
    ``problem`` string its caller reports, so no shape check ever escapes as an
    exception into the middle of a completion.
    """


def _required_object(container: dict, key: str, label: str) -> dict:
    value = _object(container, key)
    if value is None:
        raise _Malformed(f"{label} is not an object")
    return value


def _nullable_object(container: dict, key: str, label: str) -> dict:
    """The object under ``key``, or ``{}`` when GitHub sends ``null``/omits it.

    A null here is ordinary (no base branch, no protection rule, no app), and
    reading it as "no fields" is what the gate already does. Anything that is
    neither an object nor null is NOT ordinary: that is the error envelope or
    older host schema this module exists to stop.
    """
    value = container.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _Malformed(f"{label} is neither an object nor null")
    return value


def pull_request_problem(pr) -> str | None:
    """The first structural problem in a ``repos/{repo}/pulls/{n}`` answer.

    ``None`` means every field the gates read unconditionally is present and of
    the right type, so they may be dereferenced without a guard.
    """
    if not isinstance(pr, dict):
        return f"GitHub answered with {type(pr).__name__}, not a pull request object"
    head = _object(pr, "head")
    if head is None:
        return "pull request head is not an object"
    if not is_sha(head.get("sha")):
        return "pull request head.sha is not a 40-character commit sha"
    base = _object(pr, "base")
    if base is None:
        return "pull request base is not an object"
    if nonblank_str(base.get("ref")) is None:
        return "pull request base.ref is not a branch name"
    if not isinstance(pr.get("merged"), bool):
        # The one that matters most: "false" is a non-empty string, so a
        # truthiness test on it reads an unmerged PR as merged.
        return "pull request merged is not a boolean"
    if pr.get("state") not in _REST_PR_STATES:
        # REST answers exactly one of two lowercase words. Any other value is a
        # state no gate condition can act on: "unknown"/"proxy-error" is a
        # non-blank string that is not ``closed``, so the stale recheck reads it
        # as an open PR and the merge condition reports a state nobody sent.
        return 'pull request state is neither "open" nor "closed"'
    if pr.get("merged_at") is not None and not isinstance(pr.get("merged_at"), str):
        return "pull request merged_at is neither null nor a timestamp string"
    return None


def graphql_pull_request_evidence(payload):
    """``(pull_request, required_checks, problem)`` for the acceptance query.

    ``problem`` is ``None`` only when the whole consumed envelope is readable:
    the response object, ``data``, ``repository``, ``pullRequest``, the 40-hex
    ``headRefOid``, the ``baseRefName``/``state`` strings, the object-or-null
    ``baseRef`` and ``branchProtectionRule``, and every required status check's
    object, non-blank ``context`` and object-or-null ``app`` with an integer or
    null ``databaseId``. Only then may the caller dereference the PR.

    ``required_checks`` is the ``(context, app databaseId)`` projection the gate
    keys its required set by — validated here, once, so the collector never
    walks the response itself.

    On a problem the pull request comes back as ``None`` and the description
    names the field that could not be read, never the response body: it can
    carry tokens and host detail that must not reach a persisted receipt.
    """
    try:
        pr, checks = _graphql_pull_request(payload)
        return pr, checks, None
    except _Malformed as bad:
        return None, (), str(bad)


def _graphql_pull_request(payload) -> tuple[dict, tuple]:
    if not isinstance(payload, dict):
        raise _Malformed(
            f"GitHub answered with {type(payload).__name__}, not a GraphQL response object")
    data = _required_object(payload, "data", "GraphQL response data")
    repository = _required_object(data, "repository", "GraphQL data.repository")
    pr = _required_object(repository, "pullRequest", "GraphQL repository.pullRequest")
    if not is_sha(pr.get("headRefOid")):
        raise _Malformed("pull request headRefOid is not a 40-character commit sha")
    if nonblank_str(pr.get("baseRefName")) is None:
        raise _Malformed("pull request baseRefName is not a branch name")
    if nonblank_str(pr.get("state")) is None:
        raise _Malformed("pull request state is not a status string")
    base_ref = _nullable_object(pr, "baseRef", "pull request baseRef")
    rule = _nullable_object(base_ref, "branchProtectionRule", "branch protection rule")
    return pr, _required_status_checks(rule)


def _required_status_checks(rule: dict) -> tuple:
    """Every required status check a branch protection rule declares.

    GitHub's schema types this selection as a list of objects that is itself
    nullable, so a missing/``null`` value means "this rule requires no checks"
    — never "accept whatever is green". Each entry's ``context`` is the name the
    gate matches a check run by and ``app.databaseId`` pins it to one app, so a
    blank context or a non-integer id would silently widen or void the match and
    is refused as malformed instead.
    """
    checks = rule.get("requiredStatusChecks")
    if checks is None:
        return ()
    if not isinstance(checks, list):
        raise _Malformed("branch protection requiredStatusChecks is not a list")
    projected = []
    for index, check in enumerate(checks):
        label = f"required status check {index}"
        if not isinstance(check, dict):
            raise _Malformed(f"{label} is not an object")
        context = nonblank_str(check.get("context"))
        if context is None:
            raise _Malformed(f"{label} context is not a check-name string")
        app_id = _nullable_object(check, "app", f"{label} app").get("databaseId")
        # ``bool`` is an ``int`` in Python and never a GitHub app id.
        if app_id is not None and (isinstance(app_id, bool) or not isinstance(app_id, int)):
            raise _Malformed(f"{label} app databaseId is neither an integer id nor null")
        projected.append((context, app_id))
    return tuple(projected)
