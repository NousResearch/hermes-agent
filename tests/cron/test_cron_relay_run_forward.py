"""Manual `hermes cron run` forwarding for relay-fronted delivery targets.

A standalone CLI process has no live relay adapter and no standalone sender,
so a manual run that targets a relay-fronted platform must forward to the
running gateway (whose live relay adapter owns that delivery) rather than
execute in-process and hit the native standalone fallback.
"""

import json
from unittest.mock import patch

from tools import cronjob_tools


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.text = ""


class TestRelayFrontedDeliveryPlatforms:
    def test_empty_when_nothing_fronted(self):
        with patch("gateway.relay.relay_fronted_platforms", return_value=set()):
            assert cronjob_tools._relay_fronted_delivery_platforms({"id": "j1"}) == set()

    def test_detects_fronted_delivery_platform(self):
        job = {"id": "j1", "deliver": "discord"}
        with patch("gateway.relay.relay_fronted_platforms", return_value={"discord"}), patch(
            "cron.scheduler._resolve_delivery_targets",
            return_value=[{"platform": "discord", "chat_id": "123"}],
        ):
            assert cronjob_tools._relay_fronted_delivery_platforms(job) == {"discord"}

    def test_ignores_non_fronted_platform(self):
        job = {"id": "j1", "deliver": "discord"}
        with patch("gateway.relay.relay_fronted_platforms", return_value={"telegram"}), patch(
            "cron.scheduler._resolve_delivery_targets",
            return_value=[{"platform": "discord", "chat_id": "123"}],
        ):
            assert cronjob_tools._relay_fronted_delivery_platforms(job) == set()


class TestForwardRelayFrontedRun:
    def test_none_on_native_topology(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value=set()
        ):
            assert cronjob_tools._forward_relay_fronted_run({"id": "j1"}) is None

    def test_forwards_on_success(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", return_value=_Resp(200)):
            out = json.loads(cronjob_tools._forward_relay_fronted_run({"id": "j1"}))
        assert out["success"] is True
        assert out["forwarded_to_gateway"] is True

    def test_errors_when_gateway_unreachable(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=Exception("down")):
            out = json.loads(cronjob_tools._forward_relay_fronted_run({"id": "j1"}))
        assert out["success"] is False
        assert "relay-fronted" in out["error"]

    def test_errors_on_gateway_4xx(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", return_value=_Resp(401)):
            out = json.loads(cronjob_tools._forward_relay_fronted_run({"id": "j1"}))
        assert out["success"] is False
        assert "relay-fronted" in out["error"]

    def test_posts_to_run_route_with_bearer(self):
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent["url"] = url
            sent["headers"] = headers
            return _Resp(200)

        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=fake_post), patch(
            "agent.secret_scope.get_secret", return_value="secret-key-16chars"
        ):
            cronjob_tools._forward_relay_fronted_run({"id": "abc123"})
        assert sent["url"].endswith("/api/jobs/abc123/run")
        assert sent["url"].startswith("http://127.0.0.1:")
        assert sent["headers"]["Authorization"] == "Bearer secret-key-16chars"

    def test_honors_api_server_host_env(self, monkeypatch):
        """API_SERVER_HOST must reach the forward URL (adapter bind parity)."""
        monkeypatch.setenv("API_SERVER_HOST", "10.9.8.7")
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent["url"] = url
            return _Resp(200)

        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=fake_post):
            cronjob_tools._forward_relay_fronted_run({"id": "j1"})
        assert sent["url"].startswith("http://10.9.8.7:")

    def test_wildcard_bind_dials_loopback(self, monkeypatch):
        """0.0.0.0 is a bind address, not a dial address — use loopback."""
        monkeypatch.setenv("API_SERVER_HOST", "0.0.0.0")
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent["url"] = url
            return _Resp(200)

        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=fake_post):
            cronjob_tools._forward_relay_fronted_run({"id": "j1"})
        assert sent["url"].startswith("http://127.0.0.1:")

    def test_forwards_transient_prompt_in_body(self):
        """cronjob(action='run', prompt=...) context rides the forward body."""
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent["json"] = json
            return _Resp(200)

        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=fake_post):
            cronjob_tools._forward_relay_fronted_run(
                {"id": "j1"}, extra_prompt="focus on EU numbers"
            )
        assert sent["json"] == {"prompt": "focus on EU numbers"}

    def test_empty_body_without_prompt(self):
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent["json"] = json
            return _Resp(200)

        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=fake_post):
            cronjob_tools._forward_relay_fronted_run({"id": "j1"})
        assert sent["json"] == {}


class TestManualRunPromptConsumption:
    """The stamped transient prompt reaches the fire that consumes the
    manual occurrence, and only that fire."""

    def test_run_one_job_consumes_stamped_prompt(self):
        from cron import scheduler

        captured = {}

        def fake_body(job, **kwargs):
            captured["extra_prompt"] = kwargs.get("extra_prompt")
            return True

        job = {
            "id": "j1",
            "manual_run_at": "2026-08-28T00:00:00+00:00",
            "manual_run_prompt": "focus on EU numbers",
        }
        with patch.object(scheduler, "_run_one_job_body", side_effect=fake_body), patch.object(
            scheduler, "_run_with_fire_claim_heartbeat", side_effect=lambda j, fn: fn(None, None)
        ):
            assert scheduler.run_one_job(job) is True
        assert captured["extra_prompt"] == "focus on EU numbers"

    def test_explicit_extra_prompt_wins_over_stamp(self):
        from cron import scheduler

        captured = {}

        def fake_body(job, **kwargs):
            captured["extra_prompt"] = kwargs.get("extra_prompt")
            return True

        job = {
            "id": "j1",
            "manual_run_at": "2026-08-28T00:00:00+00:00",
            "manual_run_prompt": "stale stamp",
        }
        with patch.object(scheduler, "_run_one_job_body", side_effect=fake_body), patch.object(
            scheduler, "_run_with_fire_claim_heartbeat", side_effect=lambda j, fn: fn(None, None)
        ):
            scheduler.run_one_job(job, extra_prompt="direct context")
        assert captured["extra_prompt"] == "direct context"

    def test_stamp_ignored_without_manual_run_intent(self):
        """A leftover prompt with no manual_run_at (defensive) is not injected."""
        from cron import scheduler

        captured = {}

        def fake_body(job, **kwargs):
            captured["extra_prompt"] = kwargs.get("extra_prompt")
            return True

        job = {"id": "j1", "manual_run_prompt": "orphaned"}
        with patch.object(scheduler, "_run_one_job_body", side_effect=fake_body), patch.object(
            scheduler, "_run_with_fire_claim_heartbeat", side_effect=lambda j, fn: fn(None, None)
        ):
            scheduler.run_one_job(job)
        assert captured["extra_prompt"] is None


class TestTriggerJobPromptStamp:
    def test_trigger_stamps_and_mark_run_clears(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import cron.jobs as jobs_mod
        import importlib

        importlib.reload(jobs_mod)
        job = jobs_mod.create_job(
            prompt="daily report", schedule="0 9 * * *", name="stamp-test"
        )
        triggered = jobs_mod.trigger_job(job["id"], extra_prompt="just EU today")
        assert triggered["manual_run_prompt"] == "just EU today"
        assert triggered["manual_run_at"] == triggered["next_run_at"]

        jobs_mod.mark_job_run(job["id"], success=True)
        after = jobs_mod.get_job(job["id"])
        assert "manual_run_prompt" not in after
        assert "manual_run_at" not in after

    def test_retrigger_without_prompt_clears_stale_stamp(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import cron.jobs as jobs_mod
        import importlib

        importlib.reload(jobs_mod)
        job = jobs_mod.create_job(
            prompt="daily report", schedule="0 9 * * *", name="stamp-test-2"
        )
        jobs_mod.trigger_job(job["id"], extra_prompt="first context")
        retriggered = jobs_mod.trigger_job(job["id"])
        assert retriggered.get("manual_run_prompt") is None
