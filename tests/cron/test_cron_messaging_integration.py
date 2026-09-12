"""Job persistence and scheduler-to-transport contracts (no real external sends)."""
import argparse
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

from cron import outbound
from cron.jobs import create_job, get_job, list_jobs
from cron.scheduler import run_job
from tools.registry import registry
from tools import send_message_tool  # registers the real model entry point


def test_cli_persists_opt_in_and_unrelated_edits_preserve_it(tmp_path, monkeypatch):
    from hermes_cli.cron import cron_create, cron_edit
    from hermes_cli.subcommands.cron import build_cron_parser
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron/jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron/output")
    monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda _: None)
    assert cron_create(parser.parse_args([
        "cron", "create", "every 15m", "report", "--allow-messaging", "--paused",
    ])) == 0
    job = list_jobs(include_disabled=True)[0]
    assert get_job(job["id"])["allow_messaging"] is True
    for flags, expected in [(["--name", "renamed"], True), (["--no-allow-messaging"], False),
                            (["--allow-messaging"], True)]:
        assert cron_edit(parser.parse_args(["cron", "edit", job["id"], *flags])) == 0
        assert get_job(job["id"])["allow_messaging"] is expected


def test_scheduler_registry_native_transport_and_scope(tmp_path, monkeypatch):
    from agent.delegation_context import delegated_child_context
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset

    monkeypatch.setattr("cron.scheduler._hermes_home", tmp_path)
    # Env strings never confer authority, even if all old fields are forged.
    for key in ("SESSION", "ALLOW_MESSAGING", "JOB_ID", "RUN_ID"):
        monkeypatch.setenv("HERMES_CRON_" + key, "1")
    monkeypatch.setenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "999")
    args = {"target": "origin", "message": "hello", "message_key": "report"}
    assert json.loads(registry.dispatch("send_message", args))["error"]
    for platform in ("cli", "telegram", "cron"):
        enabled = _get_platform_tools({}, platform)
        assert "send_message" not in {t for name in enabled for t in resolve_toolset(name)}

    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")})
    transport = AsyncMock(return_value={"success": True, "message_id": "receipt-1"})
    copied_contexts = []
    seen = []

    class FakeAgent:
        def __init__(self, **kw):
            seen.append(kw)

        def run_conversation(self, *a, **kw):
            run = outbound.current_run()
            copied_contexts.append(contextvars.copy_context())
            if run.allowed:
                assert run.origin["chat_id"] == "123"
                assert "messaging" in seen[-1]["enabled_toolsets"]
                assert "messaging" not in seen[-1]["disabled_toolsets"]
                with delegated_child_context():
                    assert json.loads(registry.dispatch("send_message", args))["error"]
                for action in ("react", "unreact", "list"):
                    assert json.loads(registry.dispatch("send_message", {**args, "action": action}))["error"]
                assert json.loads(registry.dispatch("send_message", {**args, "target": "telegram:999"}))["error"]
                first = json.loads(registry.dispatch("send_message", {**args, "account": "other"}))
                assert first["status"] == "verified", first
                assert json.loads(registry.dispatch("send_message", args))["skipped"]
                second = json.loads(registry.dispatch("send_message", {**args, "message_key": "second"}))
                assert second["status"] == "verified"
            else:
                assert "messaging" in seen[-1]["disabled_toolsets"]
                assert json.loads(registry.dispatch("send_message", args))["error"]
            return {"final_response": "leftover summary"}

    with patch("cron.scheduler._preflight_job_config", return_value=None), \
         patch("cron.scheduler._init_cron_mcp_tools"), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "test-key", "base_url": "https://example.invalid/v1",
             "provider": "openrouter", "api_mode": "chat_completions",
         }), \
         patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.send_message_tool._send_to_platform", transport), \
         patch("tools.send_message_tool._mirror_sent_message", return_value=False), \
         patch("run_agent.AIAgent", FakeAgent):
        job = {"id": "job", "name": "report", "prompt": "report", "deliver": "telegram:123:42",
               "allow_messaging": True, "enabled_toolsets": ["web"]}
        ok, output, final, error = run_job(job, execution_id="execution-1")
        assert ok and error is None, error
        assert final == "[SILENT]" and "leftover summary" in output
        assert transport.await_count == 2
        assert transport.call_args.args[2] == "123"
        assert transport.call_args.kwargs["thread_id"] == "42"
        assert copied_contexts[0].run(outbound.current_run) is None
        assert outbound.current_run() is None
        ok, _, final, error = run_job({**job, "allow_messaging": False}, execution_id="execution-2")
        assert ok and error is None, error
        assert final == "leftover summary" and transport.await_count == 2


def test_ledger_atomic_claim_retry_fencing_and_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(outbound, "OUTBOUND_FILE", tmp_path / "outbound.db")
    params = dict(job_id="j", run_id="r", message_key="k", target="origin", body="hello",
                  platform="telegram", chat_id="123", thread_id=None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: outbound.claim_or_reuse(**params), range(16)))
    assert sum(r["action"] == "claim" for r in results) == 1
    outbound.mark_result(job_id="j", run_id="r", message_key="k", status="failed")
    retry = outbound.claim_or_reuse(**params)
    assert retry["action"] == "claim" and retry["record"]["attempt"] == 2
    stale = outbound.mark_result(job_id="j", run_id="r", message_key="k", status="verified", attempt=1)
    assert stale["status"] == "queued"
    outbound.mark_result(job_id="j", run_id="r", message_key="k", status="ambiguous", attempt=2)
    assert outbound.claim_or_reuse(**params)["action"] == "reuse"
    assert outbound.classify_send_result({"error": "timeout"})["status"] == "ambiguous"
    assert outbound.classify_send_result({"error": "not started", "delivery_not_attempted": True})["status"] == "failed"


def test_ledger_is_private_before_wal_writes(tmp_path, monkeypatch):
    import os
    import stat
    if os.name == "nt":
        return
    path = tmp_path / "cron" / "outbound.db"
    monkeypatch.setattr(outbound, "OUTBOUND_FILE", path)
    with outbound._transaction() as conn:
        conn.execute("SELECT 1")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
