"""Tests for ``hermes_cli.kanban_output`` — covers the parked-card surfacing
added in #t_cfbbb112: ``hermes kanban list`` should show why a ready card is
not being dispatched (``quota_cooldown`` / ``blocker_auth``) without an
operator having to run ``hermes kanban dispatch --json``."""

from __future__ import annotations

import dataclasses

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_output as ko


def _mk_task(**overrides) -> kb.Task:
    """Build a minimal ``kb.Task`` with sensible defaults, then apply overrides
    via ``dataclasses.replace``. Avoids hand-typing the full field list —
    adding a new ``Task`` column won't silently drop a default."""
    base = kb.Task(
        id="t_test1",
        title="Some task",
        body=None,
        assignee="friday",
        status="ready",
        priority=1,
        created_by="user",
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    return dataclasses.replace(base, **overrides)


class TestFmtTaskLineParked:
    """``hermes kanban list`` should annotate ready cards with their parked
    reason so operators can spot quota walls vs auth failures without invoking
    the dispatcher."""

    def test_ready_without_failure_text_has_no_parked_tag(self):
        t = _mk_task()
        line = ko._fmt_task_line(t)
        assert "[parked:" not in line, line
        assert t.id in line
        assert "Some task" in line

    def test_ready_with_quota_text_is_tagged_quota_cooldown(self):
        t = _mk_task(last_failure_error=(
            "pid 4121338 not alive Worker's last output: ' account entitlement "
            "is exhausted for z-ai/glm-5.2. Add credits or update billing'"
        ))
        line = ko._fmt_task_line(t)
        assert "[parked: quota_cooldown]" in line, line

    def test_ready_with_auth_text_is_tagged_blocker_auth(self):
        t = _mk_task(last_failure_error="HTTP 401 unauthorized: invalid api key")
        line = ko._fmt_task_line(t)
        assert "[parked: blocker_auth]" in line, line

    def test_non_ready_status_with_failure_text_is_not_tagged(self):
        """A blocked task has its failure text on the comment thread; we only
        annotate ready cards (which is what the dispatcher would refuse to
        spawn)."""
        for s in ("blocked", "done", "running", "archived"):
            t = _mk_task(status=s, last_failure_error="out of credits")
            line = ko._fmt_task_line(t)
            assert "[parked:" not in line, f"status={s!r} should not be tagged: {line!r}"

    def test_unknown_failure_text_is_not_tagged(self):
        t = _mk_task(last_failure_error="Worker segfaulted at line 42 of libfoo.so")
        line = ko._fmt_task_line(t)
        assert "[parked:" not in line, line

    def test_auth_in_quota_flavoured_message_wins(self):
        """If a message contains BOTH auth and quota words, surface the auth
        reason — auth is terminal, quota is transient, and the operator needs
        to know the more urgent one."""
        t = _mk_task(last_failure_error=(
            "HTTP 401 unauthorized: 429 quota exceeded, billing for API key failed"
        ))
        line = ko._fmt_task_line(t)
        assert "[parked: blocker_auth]" in line, line


class TestParkedReasonHelper:
    """The helper itself is what the dispatcher logic should agree with
    (``kanban_db_dispatch._RESPAWN_QUOTA_RE`` / ``_RESPAWN_AUTH_RE``). A
    divergence here means operators and dispatchers disagree about why a card
    is sitting in ``ready``."""

    def test_returns_none_for_no_failure_text(self):
        assert ko._parked_reason(_mk_task()) is None

    def test_returns_none_for_non_ready(self):
        t = _mk_task(status="blocked", last_failure_error="out of credits")
        assert ko._parked_reason(t) is None

    def test_quota_sample(self):
        cases = [
            "HTTP 429: Token Plan rate limit reached",
            "out of credits on OpenRouter",
            "billing error: card declined",
            "subscription expired",
            "rate-limited every one of 1 attempts",
            "entitlement exhausted for z-ai/glm-5.2",
            "token plan upgrade required",
        ]
        for err in cases:
            t = _mk_task(last_failure_error=err)
            assert ko._parked_reason(t) == "quota_cooldown", f"text={err!r}"

    def test_auth_sample(self):
        cases = [
            "HTTP 401 unauthorized",
            "HTTP 403 forbidden",
            "invalid api key passed",
            "permission denied",
            "access denied",
            "Run 'hermes auth add anthropic'",
        ]
        for err in cases:
            t = _mk_task(last_failure_error=err)
            assert ko._parked_reason(t) == "blocker_auth", f"text={err!r}"
