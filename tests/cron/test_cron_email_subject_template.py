"""Cron email subject template rendering.

A job may carry ``subject_template`` (e.g. ``"Daily Report {date}"``); cron email
delivery renders it into ``route_metadata["subject"]``, which the email adapter
turns into a fresh (non-reply) message. It applies to EMAIL targets only.
"""

import re
from types import SimpleNamespace
from typing import Any

from gateway.config import Platform

from cron.scheduler_delivery import _live_route_metadata, _render_subject_template


def _target(platform, job=None, thread_id=None) -> Any:
    """Minimal _TargetDelivery view: only attributes _live_route_metadata reads."""
    return SimpleNamespace(
        job=job or {"id": "j1", "name": "daily"},
        platform=platform,
        platform_name=platform.value,
        chat_id="ops@example.com",
        thread_id=thread_id,
        runtime_adapter=None,
        loop=None,
        notify_delivery=True,
        origin_target=False,
        origin={},
    )


class TestRenderSubjectTemplate:
    def test_renders_date_placeholder(self):
        out = _render_subject_template({"id": "j", "subject_template": "Daily Report {date}"})
        assert out is not None
        assert re.fullmatch(r"Daily Report \d{4}/\d{2}/\d{2}", out)

    def test_missing_or_empty_template_returns_none(self):
        assert _render_subject_template({"id": "j"}) is None
        assert _render_subject_template({"id": "j", "subject_template": ""}) is None

    def test_unknown_placeholder_falls_back_to_raw(self):
        assert _render_subject_template({"id": "j", "subject_template": "Report {who}"}) == "Report {who}"


class TestLiveRouteMetadataSubject:
    def test_email_target_carries_rendered_subject_only_on_text_route(self):
        job = {"id": "j1", "name": "daily", "subject_template": "Report {date}"}
        _, route_metadata, media_metadata = _live_route_metadata(_target(Platform.EMAIL, job))
        assert route_metadata["subject"].startswith("Report ")
        assert re.search(r"\d{4}/\d{2}/\d{2}$", route_metadata["subject"])
        # Media (attachments) has no subject concept.
        assert "subject" not in media_metadata

    def test_non_email_target_has_no_subject(self):
        job = {"id": "j1", "name": "daily", "subject_template": "Report {date}"}
        _, route_metadata, _ = _live_route_metadata(_target(Platform.FEISHU, job))
        assert "subject" not in route_metadata

    def test_email_target_without_template_has_no_subject(self):
        _, route_metadata, _ = _live_route_metadata(_target(Platform.EMAIL, {"id": "j1", "name": "plain"}))
        assert "subject" not in route_metadata
