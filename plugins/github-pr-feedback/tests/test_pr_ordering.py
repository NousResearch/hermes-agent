from dataclasses import replace

from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import PullRequest
from github_pr_feedback.pr_ordering import order_pull_requests, repair_window


def pull(number, base='main', repository='acme/widgets'):
    return PullRequest(number, 'OPEN', 'acme/widgets', repository, 'owner',
                       f'codex/{number}', f'{number:040x}', base_branch=base, base_sha='b' * 40)


def test_intake_orders_dependencies_before_children_without_serializing_independent_prs():
    independent, child, parent = pull(1), pull(2, 'codex/9'), pull(9)
    ordered = order_pull_requests((child, parent, independent))
    assert ordered == (independent, parent, child)
    # A fork branch cannot be the child's base branch in the upstream repo.
    assert order_pull_requests((replace(parent, head_repository='fork/widgets'), child))[0] == child
    # Malformed cyclic branches remain inspectable; ordering grants no merge authority.
    assert {p.number for p in order_pull_requests((pull(2, 'codex/9'), pull(9, 'codex/2')))} == {2, 9}


def test_repair_scans_cover_backlog_across_process_restarts(tmp_path):
    pulls = tuple(pull(n) for n in range(1, 30))
    seen = set()
    for scan in range(3):
        ledger = FeedbackLedger(tmp_path / 'ledger.sqlite3')
        selected = repair_window(ledger, 'acme/widgets', tuple(reversed(pulls)), 12)
        assert len(selected) == 12
        seen.update(p.number for p in selected)
        if scan == 0:
            assert [p.number for p in selected] == list(range(1, 13))
        ledger.close()
    assert seen == {p.number for p in pulls}
