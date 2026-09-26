"""Publication authority and terminal release evidence validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import (
    ACTIVE_PUBLICATION_STATES,
    CANDIDATE_STATES,
    GITHUB_LOGIN,
    HEX40,
    HEX64,
    MAIN_REQUIRED_STATES,
    PUBLICATION_KINDS,
    PUBLICATION_ROLES,
    PUBLICATION_STATES,
    _canonical_repo_path,
    _github_url,
    _is_int,
    _required_list,
)

def _validate_publications(
    row: Mapping[str, Any],
    capability_id: str,
    delivery_state: Any,
    repository: str,
    merged_sha: str,
    errors: list[str],
) -> tuple[list[tuple[int, str]], Mapping[str, Any] | None]:
    publications = _required_list(
        row,
        "publications",
        f"{capability_id}.publications",
        errors,
    )
    authoritative: list[Mapping[str, Any]] = []
    authoritative_prs: list[tuple[int, str]] = []

    for index, publication in enumerate(publications):
        prefix = f"{capability_id}.publications[{index}]"
        if not isinstance(publication, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        role = publication.get("role")
        kind = publication.get("kind")
        state = publication.get("state")
        if role not in PUBLICATION_ROLES:
            errors.append(f"{prefix}.role must be one of {sorted(PUBLICATION_ROLES)}")
        if kind not in PUBLICATION_KINDS:
            errors.append(f"{prefix}.kind must be one of {sorted(PUBLICATION_KINDS)}")
        if state not in PUBLICATION_STATES:
            errors.append(f"{prefix}.state must be one of {sorted(PUBLICATION_STATES)}")

        number = publication.get("number")
        if kind in {"issue", "pull_request"}:
            if not _is_int(number) or number <= 0:
                errors.append(f"{prefix}.number must be a positive integer")
        if role == "authoritative":
            authoritative.append(publication)
            if kind != "pull_request":
                errors.append(f"{prefix} authoritative publication must be a pull request")
            if _is_int(number):
                authoritative_prs.append((number, capability_id))
            author = publication.get("author")
            if not isinstance(author, str) or not GITHUB_LOGIN.fullmatch(author):
                errors.append(f"{prefix}.author must be a GitHub login")

            if delivery_state in CANDIDATE_STATES and state != "open":
                errors.append(f"{prefix}.state must be open for candidate delivery")
            if delivery_state in MAIN_REQUIRED_STATES and state != "merged":
                errors.append(f"{prefix}.state must be merged for main delivery")

            if kind == "pull_request" and _is_int(number) and repository:
                expected = f"pull/{number}"
                _github_url(
                    publication.get("url"),
                    f"{prefix}.url",
                    repository,
                    errors,
                    expected_path_prefix=expected,
                )

            head_sha = publication.get("head_sha")
            if delivery_state == "candidate_open":
                if not isinstance(head_sha, str) or not HEX40.fullmatch(head_sha):
                    errors.append(
                        f"{prefix}.head_sha must be lowercase 40-hex for candidate_open"
                    )
            merge_commit_sha = publication.get("merge_commit_sha")
            if delivery_state in MAIN_REQUIRED_STATES:
                if merge_commit_sha != merged_sha:
                    errors.append(
                        f"{prefix}.merge_commit_sha must equal merged.commit_sha"
                    )

    if delivery_state in ACTIVE_PUBLICATION_STATES and len(authoritative) != 1:
        errors.append(
            f"{capability_id} delivery_state={delivery_state!r} requires exactly one authoritative publication"
        )
    if delivery_state in {"gap", "superseded"} and authoritative:
        errors.append(
            f"{capability_id} delivery_state={delivery_state!r} cannot retain authoritative publication ownership"
        )

    return authoritative_prs, authoritative[0] if len(authoritative) == 1 else None


def _validate_release_evidence(
    row: Mapping[str, Any],
    capability_id: str,
    repository: str,
    merged_sha: str,
    authoritative: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    release = row.get("release_evidence")
    if not isinstance(release, Mapping):
        errors.append(f"{capability_id}.release_evidence is required for released")
        return

    ci = release.get("ci")
    if not isinstance(ci, Mapping):
        errors.append(f"{capability_id}.release_evidence.ci must be an object")
    else:
        _github_url(
            ci.get("url"),
            f"{capability_id}.release_evidence.ci.url",
            repository,
            errors,
            expected_path_prefix="actions/runs/",
        )
        if ci.get("commit_sha") != merged_sha:
            errors.append(
                f"{capability_id}.release_evidence.ci.commit_sha must equal merged.commit_sha"
            )

    receipt = release.get("live_receipt")
    if not isinstance(receipt, Mapping):
        errors.append(
            f"{capability_id}.release_evidence.live_receipt must be an object"
        )
    else:
        _canonical_repo_path(
            receipt.get("path"),
            f"{capability_id}.release_evidence.live_receipt.path",
            errors,
        )
        digest = receipt.get("sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            errors.append(
                f"{capability_id}.release_evidence.live_receipt.sha256 must be lowercase 64-hex"
            )
        if receipt.get("commit_sha") != merged_sha:
            errors.append(
                f"{capability_id}.release_evidence.live_receipt.commit_sha must equal merged.commit_sha"
            )

    reviews = release.get("reviews")
    if not isinstance(reviews, list) or len(reviews) < 2:
        errors.append(
            f"{capability_id}.release_evidence.reviews requires at least two reviews"
        )
        return

    reviewers: list[str] = []
    author = authoritative.get("author") if authoritative else None
    publication_number = authoritative.get("number") if authoritative else None
    for index, review in enumerate(reviews):
        prefix = f"{capability_id}.release_evidence.reviews[{index}]"
        if not isinstance(review, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        reviewer = review.get("reviewer")
        if not isinstance(reviewer, str) or not GITHUB_LOGIN.fullmatch(reviewer):
            errors.append(f"{prefix}.reviewer must be a GitHub login")
        else:
            reviewers.append(reviewer.casefold())
            if isinstance(author, str) and reviewer.casefold() == author.casefold():
                errors.append(f"{prefix}.reviewer must be independent of the PR author")
        expected = (
            f"pull/{publication_number}"
            if _is_int(publication_number)
            else "pull/"
        )
        url = _github_url(
            review.get("url"),
            f"{prefix}.url",
            repository,
            errors,
            expected_path_prefix=expected,
        )
        if url and "#pullrequestreview-" not in url:
            errors.append(f"{prefix}.url must identify a submitted pull request review")
        if review.get("commit_sha") != merged_sha:
            errors.append(f"{prefix}.commit_sha must equal merged.commit_sha")

    if len(set(reviewers)) != len(reviewers):
        errors.append(f"{capability_id}.release_evidence.reviews must be independent")
