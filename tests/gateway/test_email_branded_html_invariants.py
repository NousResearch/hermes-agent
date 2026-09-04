"""Regression guard for cron branded-HTML email delivery.

The original 21-case suite (test_email_branded_html.py) verifies the helpers
behave correctly in isolation. This file is the *structural* guard: it fails
loudly if any of the four required helpers stops being referenced from the
adapter or if the EmailAdapter.splits_long_messages flag gets reverted.

History: this guard exists because PR #90147 (commit 2d401855a, "fix(email):
render cron branded HTML reports as text/html with correct subject") shipped
correctly, was unmerged on a local backup branch, and the production adapter
silently regressed to the broken state for weeks. Each refactor that touched
adapter.py was a chance to re-introduce the bug. These assertions catch that
regression class at test-time instead of in the inbox.
"""
import inspect
import unittest

from cron import scheduler as cron_scheduler
from cron import email_contract as email_contract_module
from plugins.platforms.email import adapter as adapter_module
from plugins.platforms.email.adapter import EmailAdapter


class TestBrandedHtmlInvariants(unittest.TestCase):
    """Structural guards against silent regression of the branded-HTML fix."""

    def test_email_adapter_splits_long_messages_is_true(self):
        """The gateway slices outbound messages at MAX_PLATFORM_OUTPUT (4000
        chars) by default. Branded cron HTML routinely exceeds 10KB. If this
        flag is ever flipped back to False, every branded cron email arrives
        truncated mid-<td>."""
        self.assertTrue(
            getattr(EmailAdapter, "splits_long_messages", False),
            "EmailAdapter.splits_long_messages must remain True — branded "
            "cron HTML exceeds the 4KB gateway slice otherwise.",
        )

    def test_send_email_uses_html_subtype_helper(self):
        """The in-process send path must select MIME subtype via _html_subtype(),
        not hardcode 'plain'. A naive refactor that re-introduces the hardcoded
        MIMEText(body, 'plain') will make raw HTML appear as literal tags in
        the user's inbox. (Since the facade/siblings refactor, _send_email and
        the attachment senders all funnel through _new_reply — that builder is
        the seam the helpers must appear in.)"""
        src = inspect.getsource(adapter_module.EmailAdapter._new_reply)
        self.assertIn(
            "_html_subtype",
            src,
            "_new_reply must call _html_subtype() to select MIME subtype. "
            "Hardcoded 'plain' regression: raw HTML renders as literal tags.",
        )
        self.assertIn(
            "_subject_for_body",
            src,
            "_new_reply must call _subject_for_body() so HTML report subjects "
            "match <title> exactly without a Re: prefix.",
        )

    def test_all_in_process_send_paths_use_no_re_subject_helper(self):
        """Every in-process HTML path must avoid adding Re: to report titles.

        The three public send methods must still funnel through _new_reply,
        which is where the subject helper lives."""
        new_reply_src = inspect.getsource(adapter_module.EmailAdapter._new_reply)
        self.assertIn("_subject_for_body", new_reply_src, "_new_reply bypasses subject helper")
        self.assertNotIn(
            'subject = f"Re: {subject}"',
            new_reply_src,
            "_new_reply still prefixes branded report subjects with Re:",
        )
        for method_name in (
            "_send_email",
            "_send_with_files",
        ):
            src = inspect.getsource(getattr(adapter_module.EmailAdapter, method_name))
            self.assertIn(
                "_new_reply",
                src,
                f"{method_name} no longer funnels through _new_reply — the "
                "subject/MIME invariants bypass it.",
            )

    def test_standalone_send_uses_html_subtype_helper(self):
        """_standalone_send (the out-of-process fallback for cron deliveries
        when the gateway runner is absent) must also use the helpers. The
        2026-08-12 two-path trap taught us: patching _send_email without
        _standalone_send (or vice versa) leaves one of the two delivery
        paths broken."""
        src = inspect.getsource(adapter_module._standalone_send)
        for helper in ("_html_subtype", "_extract_title_from_html", "_strip_leading_prose"):
            self.assertIn(
                helper,
                src,
                f"_standalone_send must call {helper}() — out-of-process "
                "cron delivery path is broken without it (two-path trap).",
            )

    def test_scheduler_enforces_generation_contract_before_delivery(self):
        """Email correctness starts before the adapter: the cron scheduler must
        reject/repair malformed report generation instead of treating any
        non-empty model response as a successful email. (The failure-notice
        composition lives in _compose_run_delivery since the refactor split
        delivery composition out of _run_one_job_body.)"""
        run_src = inspect.getsource(cron_scheduler.run_job)
        compose_src = inspect.getsource(cron_scheduler._compose_run_delivery)
        self.assertIn("validate_email_output(job, final_response)", run_src)
        self.assertIn("_run_email_contract_repair_turn", run_src)
        self.assertIn("render_contract_failure_email", compose_src)
        self.assertTrue(
            callable(getattr(email_contract_module, "validate_email_output", None))
        )

    def test_canonical_email_palette_is_enforced_centrally(self):
        """All explicitly palette-bound cron reports must fail closed when
        generation drifts back to an older blue or ad-hoc template."""
        contract_src = inspect.getsource(email_contract_module.validate_email_output)
        expected = {
            "header_start": "#6C27D7",
            "header_end": "#4F1E9C",
            "heading": "#4C1D95",
            "text": "#1F2430",
            "callout": "#F7F5FC",
            "callout_border": "#E0D7F5",
            "table_header": "#F3F0FB",
            "badge": "#EDE9FE",
            "border": "#E5E7EB",
            "body": "#FFFFFF",
            "outer": "#F4F5F7",
            "footer": "#FAFAFA",
        }
        self.assertEqual(email_contract_module.CANONICAL_PURPLE_PALETTE, expected)
        self.assertIn("CANONICAL_PURPLE_PALETTE_ID", contract_src)
        self.assertIn("linear-gradient", contract_src)

    def test_html_helpers_are_importable(self):
        """The four helpers must remain at module scope so the inspection
        checks above resolve. A refactor that moves them into a nested
        namespace (e.g. _helpers.html) without updating the call sites will
        fail here before failing in production."""
        for helper in (
            "_html_subtype",
            "_extract_title_from_html",
            "_strip_leading_prose",
            "_subject_for_body",
        ):
            self.assertTrue(
                hasattr(adapter_module, helper),
                f"adapter.{helper} is missing — branded-HTML helpers must "
                "remain at module scope.",
            )
            self.assertTrue(
                callable(getattr(adapter_module, helper, None)),
                f"adapter.{helper} exists but is not callable.",
            )


if __name__ == "__main__":
    unittest.main()
