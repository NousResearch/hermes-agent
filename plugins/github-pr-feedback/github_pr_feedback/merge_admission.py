"""Opt-in owned-PR admission; merge eligibility remains independently gated."""
from datetime import UTC, datetime


def enroll_owned_pulls(policy, merge_policy, ledger, pulls):
    if not merge_policy.auto_enroll_owned_prs:
        return 0
    enrolled = set(ledger.enrolled_merge_pr_numbers(merge_policy.repository))
    count = 0
    for pull in pulls:
        if (pull.number in enrolled or not policy.admit_pull_request(pull).admitted
                or pull.base_repository != merge_policy.repository
                or pull.head_repository != merge_policy.repository
                or pull.author_login.casefold() != merge_policy.author_login.casefold()
                or pull.base_branch != merge_policy.base_branch or pull.state != "OPEN"):
            continue
        count += int(ledger.enroll_merge_pr(
            pull.base_repository, pull.number, enrolled_at=datetime.now(UTC),
            enrolled_by="configured-owned-pr-admission", automatic=True,
        ))
    return count
