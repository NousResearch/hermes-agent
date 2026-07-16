"""End-to-end proof of the generic scheduled plugin host contract."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


def test_plugin_context_scheduler_and_delivery_form_one_private_receipt_pipeline(
    tmp_path, monkeypatch
):
    import hermes_cli.plugins as plugins
    from cron.scheduler import (
        _build_job_prompt,
        _deliver_result_with_receipt,
        _notify_cron_delivery,
    )
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    manager = plugins.PluginManager()
    context = plugins.PluginContext(
        plugins.PluginManifest(name="test-companion", key="test-companion"),
        manager,
    )
    delivered = []

    def provide_context(provider, **_kwargs):
        if provider != "companion-test":
            return None
        return {
            "provider": provider,
            "context": "Preferred language: zh",
            "metadata": {"attempt_id": "private-attempt-42"},
        }

    context.register_hook("cron_context", provide_context)
    context.register_hook(
        "cron_delivery", lambda receipt, **_kwargs: delivered.append(receipt)
    )
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    job = {
        "id": "job-1",
        "name": "companion-checkin",
        "prompt": "Compose one natural message.",
        "context_provider": "companion-test",
        "response_mode": "text_only",
        "deliver": "origin",
        "origin": {
            "platform": "telegram",
            "chat_id": "chat-1",
            "session_id": "session-private-9",
        },
    }
    prompt = _build_job_prompt(job)
    pconfig = MagicMock(enabled=True)
    config = MagicMock(platforms={Platform.TELEGRAM: pconfig})

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch(
            "tools.send_message_tool._send_to_platform",
            new=AsyncMock(return_value={"success": True}),
        ) as send,
    ):
        receipt = _deliver_result_with_receipt(job, "今天训练完感觉挺轻松的。")
        _notify_cron_delivery(receipt)

    sent = send.call_args.args[3]
    assert "Preferred language: zh" in prompt
    assert "private-attempt-42" not in prompt
    assert "session-private-9" not in prompt
    assert sent == "今天训练完感觉挺轻松的。"
    assert "private-attempt-42" not in sent
    assert delivered == [receipt.as_dict()]
    assert delivered[0]["metadata"] == {"attempt_id": "private-attempt-42"}


def test_plugin_context_reports_versioned_runtime_capabilities():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    context = PluginContext(
        PluginManifest(name="capability-consumer", key="capability-consumer"),
        PluginManager(),
    )

    capabilities = context.get_host_capabilities()

    assert capabilities["contract_version"] >= 1
    assert capabilities["scheduler"]["scheduled_context"] is True
    assert capabilities["scheduler"]["delivery_receipts"] is True
    assert capabilities["scheduler"]["text_only"] is True
    assert capabilities["tools"]["names"] == sorted(capabilities["tools"]["names"])
    assert isinstance(capabilities["tools"]["web_search"], bool)
    assert isinstance(capabilities["tools"]["browser"], bool)
