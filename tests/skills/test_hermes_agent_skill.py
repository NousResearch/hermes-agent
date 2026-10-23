"""Regression checks for issue #87552: the bundled hermes-agent skill's
security-sensitive claims must stay in sync with actual runtime behavior.

Scoped to the specific mismatches verified and fixed by that issue's PR:
Portal credential-relay guidance, background-execution durability wording,
and YOLO/approval-mode overstatement. Not a comprehensive check of every
category the issue raised -- see the issue for the full list of
mismatches, some of which are tracked separately.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "autonomous-ai-agents" / "hermes-agent"


def _read(rel_path: str) -> str:
    path = SKILL_DIR / rel_path
    assert path.exists(), f"expected reference file missing: {path}"
    return path.read_text(encoding="utf-8")


class TestPortalAuthGuidance:
    """references/portal-auth-for-third-party-apps.md must point agents at
    the existing `hermes proxy` command, not a DIY credential-relay that
    reads the raw Portal bearer out of ~/.hermes/auth.json."""

    def _content(self) -> str:
        return _read("references/portal-auth-for-third-party-apps.md")

    def test_recommends_hermes_proxy_not_a_diy_relay(self):
        content = self._content()
        assert "hermes proxy" in content, (
            "must point agents at the existing, supported hermes proxy "
            "command for the third-party-app credential-sharing use case"
        )

    def test_does_not_instruct_reading_raw_auth_json_credential(self):
        content = self._content()
        assert "Read Hermes's existing Portal credential out of" not in content, (
            "must not instruct agents to read the raw bearer directly "
            "out of ~/.hermes/auth.json for forwarding to another app"
        )

    def test_notes_hermes_login_is_deprecated(self):
        content = self._content()
        assert "deprecated" in content.lower(), (
            "must not present `hermes login` as the current, non-deprecated "
            "way to authenticate -- `hermes auth`/`hermes model`/`hermes "
            "setup` are the current commands"
        )


class TestBackgroundExecutionDurabilityGuidance:
    """references/background-systems.md must not recommend a process-local
    mechanism (terminal background execution) for work that needs to
    survive a process restart -- only cronjob is durable."""

    def _content(self) -> str:
        return _read("references/background-systems.md")

    def test_terminal_background_not_recommended_for_durable_work(self):
        content = self._content()
        assert (
            "use `cronjob` or\n  `terminal(background=True, notify_on_complete=True)`"
            not in content
        ), (
            "terminal(background=True) is process-local, same as "
            "delegate_task's own background mode this same section "
            "correctly calls out as 'Not durable' -- it must not be "
            "offered as an alternative durable option"
        )

    def test_cron_timeout_defaults_are_not_the_stale_three_minute_claim(self):
        content = self._content()
        assert "3-minute hard interrupt" not in content, (
            "stale claim -- current defaults (cron/scheduler.py) are a "
            "600s (10min) inactivity timeout for the agent run and a "
            "separate 3600s (1hr) timeout for pre-run scripts, both "
            "configurable, not a fixed 3-minute interrupt"
        )


class TestApprovalYoloGuidance:
    """references/security-privacy.md must not describe YOLO/off as
    bypassing every safety mechanism -- hardline command blocks and
    approvals.deny rules remain active unconditionally."""

    def _content(self) -> str:
        return _read("references/security-privacy.md")

    def test_does_not_claim_yolo_skips_all_approval_prompts(self):
        content = self._content()
        assert "skip all approval prompts" not in content, (
            "overstated -- off/--yolo disables the interactive prompt for "
            "recoverable dangerous commands only; hardline blocks and "
            "approvals.deny rules fire unconditionally, before the yolo "
            "bypass, per tools/approval.py's own design"
        )

    def test_mentions_hardline_or_deny_rules_remaining_active(self):
        content = self._content()
        assert "hardline" in content.lower() or "approvals.deny" in content, (
            "must document that some protections (hardline floor, "
            "user-defined deny rules) remain active even under yolo"
        )

    def test_does_not_claim_file_writes_have_no_protection(self):
        content = self._content()
        assert "file writes never\ngo through the approval prompt, only shell commands do" not in content, (
            "write_file/patch DO block writes to a fixed set of sensitive "
            "paths (system-sensitive locations, the active config.yaml) "
            "unconditionally -- see get_write_denied_error() in "
            "tools/file_operations.py"
        )
