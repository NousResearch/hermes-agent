"""Keep exact-head audits behind unresolved work on their own PR."""

from .github_client import GitHubClient, GitHubClientError, MergeStateStillComputingError, PullRequestMergeState
from .ledger import FeedbackLedger
from .policy import PullRequest


def local_ci_admission_blocker(
    github: GitHubClient, ledger: FeedbackLedger, current: PullRequest
) -> str | None:
    if ledger.has_pending_mutation(current.base_repository, current.number):
        return "mutation_pending"
    try:
        state = github.get_merge_state(current.base_repository, current.number)
    except MergeStateStillComputingError:
        return "mergeable_state_still_computing"
    except GitHubClientError:
        return "github_error"
    if (
        state.repository != current.base_repository
        or state.number != current.number
        or state.head_sha != current.head_sha
        or state.base_sha != current.base_sha
        or state.state != "OPEN"
        or state.merged
    ):
        return "head_changed"
    return audit_execution_blocker(ledger, state)


def audit_execution_blocker(ledger: FeedbackLedger, state: PullRequestMergeState) -> str | None:
    """Queued cards must recheck prerequisites before starting expensive CI."""
    if ledger.has_pending_mutation(state.repository, state.number):
        return "mutation_pending"
    if not state.mergeable or state.merge_state_status == "DIRTY":
        return "merge_conflict"
    return None
