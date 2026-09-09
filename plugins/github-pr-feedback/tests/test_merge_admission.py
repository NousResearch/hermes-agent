from dataclasses import replace
from datetime import UTC, datetime

from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.merge_admission import enroll_owned_pulls
from github_pr_feedback.policy import PullRequest, load_policy


def configured(tmp_path, auto=True):
    import subprocess
    if not (tmp_path / ".git").exists():
        subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    return load_policy({
        'enabled': True,
        'repositories': [{'base_repository': 'acme/widgets', 'head_repository': 'acme/widgets',
            'owner_login': 'owner', 'local_path': str(tmp_path), 'branch_prefixes': ['codex/']}],
        'reviewer_logins': ['reviewer'], 'reviewer_associations': [],
        'not_before': '2026-08-25T00:00:00Z', 'assignee': 'repair', 'board': 'repairs',
        'merge_maintainer': {'enabled': True, 'assignee': 'merge', 'repository': 'acme/widgets',
            'author_login': 'owner', 'base_branch': 'stable', 'merge_methods': ['squash'],
            'receipt_max_age_seconds': 3600, 'report_only': False, 'post_merge': {'enabled': False},
            'auto_enroll_owned_prs': auto},
    })


def test_owned_admission_respects_lane_scope_and_explicit_opt_in(tmp_path):
    policy = configured(tmp_path)
    pull = PullRequest(17, 'OPEN', 'acme/widgets', 'acme/widgets', 'owner', 'codex/fix',
                       'a' * 40, base_branch='stable', base_sha='b' * 40)
    others = (replace(pull, number=18, author_login='other'),
              replace(pull, number=19, head_repository='fork/widgets'),
              replace(pull, number=20, base_branch='other'),
              replace(pull, number=21, head_ref_name='unmanaged/fix'))
    ledger = FeedbackLedger(tmp_path / 'ledger.db')
    try:
        disabled = configured(tmp_path, auto=False)
        assert enroll_owned_pulls(disabled, disabled.merge_maintainer, ledger, (pull, *others)) == 0
        assert enroll_owned_pulls(policy, policy.merge_maintainer, ledger, (pull, *others)) == 1
        assert ledger.enrolled_merge_pr_numbers('acme/widgets') == (17,)
        assert ledger.latest_ci_receipt_for_head('acme/widgets', 17, pull.head_sha) is None
        assert ledger.completed_merge_receipt('acme/widgets', 17) is None
    finally:
        ledger.close()


def test_operator_disable_survives_rescan_and_restart_until_explicit_enable(tmp_path):
    policy = configured(tmp_path)
    pull = PullRequest(17, 'OPEN', 'acme/widgets', 'acme/widgets', 'owner', 'codex/fix',
                       'a' * 40, base_branch='stable', base_sha='b' * 40)
    path = tmp_path / 'ledger.db'
    ledger = FeedbackLedger(path)
    ledger.unenroll_merge_pr('acme/widgets', 17)
    ledger.close()
    ledger = FeedbackLedger(path)
    try:
        assert enroll_owned_pulls(policy, policy.merge_maintainer, ledger, (pull,)) == 0
        assert not ledger.is_merge_enrolled('acme/widgets', 17)
        ledger.enroll_merge_pr('acme/widgets', 17, enrolled_at=datetime.now(UTC), enrolled_by='operator')
        assert ledger.is_merge_enrolled('acme/widgets', 17)
    finally:
        ledger.close()
