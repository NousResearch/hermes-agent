"""End-to-end conservative cron transport-receipt integration tests."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import subprocess
from concurrent.futures import Future
from unittest.mock import AsyncMock, patch

from cron.scheduler import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult, TransportReceipt, TransportTarget


class _PlannedTelegramAdapter:
    supports_inchannel_continuable = False

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.send_calls = 0
        self.last_metadata = None

    def plan_transport_text(self, content: str) -> list[str]:
        assert content
        return list(self.chunks)

    async def send(self, chat_id: str, content: str, metadata=None) -> SendResult:
        self.send_calls += 1
        metadata = metadata or {}
        self.last_metadata = dict(metadata)
        actual_thread = str(metadata.get("thread_id")) if metadata.get("thread_id") else None
        requested_identity = metadata.get("_transport_receipt_requested_target") or {}
        requested = TransportTarget(
            str(requested_identity.get("platform") or "telegram"),
            str(requested_identity.get("chat_id") or chat_id),
            (str(requested_identity["thread_id"]) if requested_identity.get("thread_id") else None),
        )
        actual = TransportTarget("telegram", str(chat_id), actual_thread)
        return SendResult(
            success=True,
            message_id="provider-0",
            receipts=tuple(
                TransportReceipt(
                    outcome="delivered",
                    provider_message_id=f"provider-{ordinal}",
                    requested_target=requested,
                    actual_target=actual,
                    component="text",
                    ordinal=ordinal,
                )
                for ordinal, _chunk in enumerate(self.chunks)
            ),
        )


class _PartialTelegramAdapter(_PlannedTelegramAdapter):
    async def send(self, chat_id: str, content: str, metadata=None) -> SendResult:
        self.send_calls += 1
        target = TransportTarget("telegram", str(chat_id))
        return SendResult(
            success=False,
            error="second chunk failed",
            receipts=(TransportReceipt(
                outcome="delivered", provider_message_id="provider-0",
                requested_target=target, actual_target=target,
                component="text", ordinal=0,
            ),),
        )


class _TypedMediaTelegramAdapter(_PlannedTelegramAdapter):
    async def send_document(
        self, chat_id: str, file_path: str, metadata=None,
    ) -> SendResult:
        metadata = metadata or {}
        requested_identity = metadata["_transport_receipt_requested_target"]
        requested = TransportTarget(
            str(requested_identity["platform"]),
            str(requested_identity["chat_id"]),
            (str(requested_identity["thread_id"]) if requested_identity.get("thread_id") else None),
        )
        actual = TransportTarget("telegram", str(chat_id))
        ordinal = metadata["_transport_receipt_ordinal"]
        return SendResult(
            success=True,
            message_id=f"media-provider-{ordinal}",
            receipts=(TransportReceipt(
                outcome="delivered",
                provider_message_id=f"media-provider-{ordinal}",
                requested_target=requested,
                actual_target=actual,
                component="media",
                ordinal=ordinal,
            ),),
        )


class _UnknownMediaTelegramAdapter(_PlannedTelegramAdapter):
    async def send_document(
        self, chat_id: str, file_path: str, metadata=None,
    ) -> SendResult:
        metadata = metadata or {}
        requested_identity = metadata["_transport_receipt_requested_target"]
        requested = TransportTarget(
            str(requested_identity["platform"]),
            str(requested_identity["chat_id"]),
            (str(requested_identity["thread_id"]) if requested_identity.get("thread_id") else None),
        )
        ordinal = metadata["_transport_receipt_ordinal"]
        return SendResult(
            success=False,
            error="media delivery outcome is unknown",
            error_kind="unknown",
            receipts=(TransportReceipt(
                outcome="unknown",
                requested_target=requested,
                component="media",
                ordinal=ordinal,
            ),),
        )


class _FailedMediaWithDeliveredReceiptAdapter(_TypedMediaTelegramAdapter):
    async def send_document(
        self, chat_id: str, file_path: str, metadata=None,
    ) -> SendResult:
        delivered = await super().send_document(chat_id, file_path, metadata)
        return SendResult(
            success=False,
            error="provider reported media failure",
            receipts=delivered.receipts,
        )


def _gateway_config() -> GatewayConfig:
    return GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )


def _running_loop():
    loop = type("Loop", (), {"is_running": lambda self: True})()
    return loop


def _run_coroutine_threadsafe(coro, _loop):
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except BaseException as exc:  # noqa: BLE001 - test transports exception faithfully
        future.set_exception(exc)
    return future


def _job() -> dict:
    return {
        "id": "receipt-e2e",
        "name": "receipt-e2e",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "123"},
    }


def test_standalone_telegram_preregistration_uses_exact_formatted_chunks():
    from cron.scheduler import _receipt_text_chunks_for_target
    from tools.send_message_tool import _plan_standalone_telegram_text

    content = ("- bounded item!\n" * 300).strip()
    _formatted, actual_chunks, _has_html, _caption = (
        _plan_standalone_telegram_text(content)
    )
    planned_chunks = _receipt_text_chunks_for_target(None, "telegram", content)

    assert len(actual_chunks) > 1
    assert planned_chunks == actual_chunks
    assert planned_chunks != [content]


def test_scheduler_rejects_hostile_adapter_plan_and_receipt_containers():
    from cron.scheduler import (
        _confirm_adapter_delivery,
        _persist_target_text_receipts,
        _receipt_text_chunks_for_target,
    )

    class HostileList(list):
        def __bool__(self):
            raise AssertionError("hostile plan truthiness was evaluated")

        def __iter__(self):
            raise AssertionError("hostile plan was iterated")

    class HostileTuple(tuple):
        def __bool__(self):
            raise AssertionError("hostile receipts truthiness was evaluated")

        def __iter__(self):
            raise AssertionError("hostile receipts were iterated")

    class HostileResult:
        def __getattribute__(self, _name):
            raise AssertionError("hostile adapter result attribute was read")

    descriptor_calls = []

    class HostileFieldsDescriptor:
        @property
        def __dict__(self):
            descriptor_calls.append("called")
            return {"success": True}

    adapter = _PlannedTelegramAdapter(["unused"])
    adapter.plan_transport_text = lambda _content: HostileList(["private"])
    with __import__("pytest").raises(ValueError, match="planner"):
        _receipt_text_chunks_for_target(
            {Platform.TELEGRAM: adapter}, "telegram", "bounded",
        )

    assert _persist_target_text_receipts(
        HostileTuple(()),
        {("telegram", "123", "", "text", 0): "attempt"},
        {"platform": "telegram", "chat_id": "123", "thread_id": ""},
        components={"text"},
    ) is False
    assert _confirm_adapter_delivery(HostileResult()) is False
    assert _confirm_adapter_delivery(HostileFieldsDescriptor()) is False
    assert descriptor_calls == []


def test_persisted_non_delivered_receipts_do_not_satisfy_component_plan():
    from cron.scheduler import _persist_target_text_receipts

    target = TransportTarget("telegram", "123")
    attempts = {("telegram", "123", "", "media", 0): "attempt"}
    requested = {"platform": "telegram", "chat_id": "123", "thread_id": ""}
    receipts = (
        TransportReceipt(
            outcome="unknown",
            requested_target=target,
            component="media",
            ordinal=0,
        ),
        TransportReceipt(
            outcome="failed",
            requested_target=target,
            failure_kind="pre_dispatch",
            component="media",
            ordinal=0,
        ),
    )

    for receipt in receipts:
        with patch("cron.scheduler.record_transport_receipt", return_value=True):
            assert _persist_target_text_receipts(
                (receipt,), attempts, requested, components={"media"}
            ) is False


def test_delivered_receipt_to_different_actual_target_does_not_satisfy_plan():
    from cron.scheduler import _persist_target_text_receipts

    requested_target = TransportTarget("telegram", "123", "topic-7")
    actual_target = TransportTarget("telegram", "123")
    attempts = {("telegram", "123", "topic-7", "text", 0): "attempt"}
    requested = {
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "topic-7",
    }
    receipt = TransportReceipt(
        outcome="delivered",
        requested_target=requested_target,
        actual_target=actual_target,
        provider_message_id="provider-ack-1",
        component="text",
        ordinal=0,
    )

    with patch("cron.scheduler.record_transport_receipt", return_value=True):
        assert _persist_target_text_receipts(
            (receipt,), attempts, requested, components={"text"}
        ) is False


def test_standalone_telegram_caption_preregistration_omits_unsent_text_component():
    from cron.scheduler import _receipt_text_chunks_for_target
    from tools.send_message_tool import _plan_standalone_telegram_text

    media = [("/tmp/bounded-image.jpg", False)]
    content = "bounded caption"
    _formatted, actual_chunks, _has_html, caption = (
        _plan_standalone_telegram_text(content, media_files=media)
    )
    planned_chunks = _receipt_text_chunks_for_target(
        None, "telegram", content, media_files=media,
    )

    assert caption is not None
    assert actual_chunks == []
    assert planned_chunks == actual_chunks


def test_scheduler_uses_standalone_telegram_plan_when_adapter_loop_is_unavailable(
    monkeypatch, tmp_path
):
    import cron.executions as executions
    from gateway.config import Platform
    from tools.send_message_tool import _plan_standalone_telegram_text

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    execution = executions.create_execution("receipt-e2e", source="direct")
    content = ("- standalone item!\n" * 300).strip()
    expected_chunks = _plan_standalone_telegram_text(content)[1]
    adapter = _PlannedTelegramAdapter(["live-adapter-only"])

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch(
            "tools.send_message_tool._send_to_platform",
            return_value={"error": "standalone send rejected before provider ack"},
        ),
    ):
        _deliver_result(
            _job(), content, adapters={Platform.TELEGRAM: adapter}, loop=None,
            execution_id=execution["id"], fire_identity="fire-no-loop",
        )

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        hashes = [row[0] for row in conn.execute(
            "SELECT content_hash FROM delivery_components ORDER BY ordinal"
        )]
    assert hashes == [
        hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        for chunk in expected_chunks
    ]


def test_scheduler_preregisters_and_persists_every_planned_text_chunk(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _PlannedTelegramAdapter(["chunk-one", "chunk-two"])

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
    ):
        result = _deliver_result(
            _job(), "one long logical report",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-chunks",
        )

    assert result is None
    assert adapter.send_calls == 1
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 2,
        "failed": 0,
        "unknown": 0,
        "targets_delivered": 1,
    }


def test_receipt_preregistration_failure_stops_before_adapter_send(monkeypatch):
    adapter = _PlannedTelegramAdapter(["chunk-one"])

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("cron.scheduler.preregister_receipt_plan", side_effect=RuntimeError("private database path")),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
    ):
        result = _deliver_result(
            _job(), "report body",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id="execution-id", fire_identity="fire-preregister-fail",
        )

    assert result == "delivery receipt plan could not be persisted; no delivery was sent"
    assert adapter.send_calls == 0
    assert "private database path" not in result


def test_scheduler_rejects_target_subclasses_before_magic_methods(monkeypatch):
    adapter = _PlannedTelegramAdapter(["chunk-one"])

    class HostileText(str):
        def __bool__(self):
            raise AssertionError("hostile target truthiness was evaluated")

        def __str__(self):
            raise AssertionError("hostile target was stringified")

        def __hash__(self):
            raise AssertionError("hostile target was hashed")

        def __eq__(self, _other):
            raise AssertionError("hostile target was compared")

    job = _job()
    job["origin"] = {"platform": "telegram", "chat_id": HostileText("123")}
    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
    ):
        result = _deliver_result(
            job, "report body",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id="execution-id", fire_identity="fire-id",
        )

    assert result == "delivery target is invalid; no delivery was sent"
    assert adapter.send_calls == 0


def test_scheduler_rejects_execution_identity_subclasses_before_magic_methods():
    adapter = _PlannedTelegramAdapter(["chunk-one"])

    class HostileText(str):
        def __bool__(self):
            raise AssertionError("hostile identity truthiness was evaluated")

        def __str__(self):
            raise AssertionError("hostile identity was stringified")

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
    ):
        result = _deliver_result(
            _job(), "report body",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=HostileText("execution-id"), fire_identity="fire-id",
        )

    assert result == "delivery receipt identity is invalid; no delivery was sent"
    assert adapter.send_calls == 0


def test_partial_chunk_ack_is_retained_without_standalone_retry(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _PartialTelegramAdapter(["chunk-one", "chunk-two"])
    standalone_calls = AsyncMock(return_value={"success": True})

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
        patch("tools.send_message_tool._send_to_platform", new=standalone_calls),
    ):
        result = _deliver_result(
            _job(), "partial report",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-partial-chunks",
        )

    assert result is not None
    standalone_calls.assert_not_awaited()
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1, "failed": 0, "unknown": 1, "targets_delivered": 0,
    }


def test_text_ack_with_opaque_media_remains_partial_and_target_unknown(monkeypatch, tmp_path):
    import cron.executions as executions
    import gateway.platforms.base as base

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    media = tmp_path / "receipt-report.pdf"
    media.write_bytes(b"test-pdf")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _PlannedTelegramAdapter(["text"])

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
        patch("cron.scheduler._send_media_via_adapter", return_value=[]),
    ):
        result = _deliver_result(
            _job(), f"report\nMEDIA:{media}",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-media",
        )

    assert result is not None
    assert "media acknowledgement" in result
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1,
        "failed": 0,
        "unknown": 1,
        "targets_delivered": 0,
    }


def test_live_adapter_typed_media_ack_completes_planned_target(monkeypatch, tmp_path):
    import cron.executions as executions
    import gateway.platforms.base as base

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    media = tmp_path / "typed-receipt-report.pdf"
    media.write_bytes(b"test-pdf")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _TypedMediaTelegramAdapter(["text"])

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
    ):
        result = _deliver_result(
            _job(), f"report\nMEDIA:{media}",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-typed-media",
        )

    assert result is None
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 2,
        "failed": 0,
        "unknown": 0,
        "targets_delivered": 1,
    }


def test_live_adapter_unknown_media_ack_stays_partial_without_session_side_effects(
    monkeypatch, tmp_path
):
    import cron.executions as executions
    import gateway.platforms.base as base

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    media = tmp_path / "unknown-receipt-report.pdf"
    media.write_bytes(b"test-pdf")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _UnknownMediaTelegramAdapter(["text"])
    job = _job()
    job["attach_to_session"] = True

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
        patch("cron.scheduler._open_continuable_cron_thread", return_value="created-thread"),
        patch("cron.scheduler._seed_cron_thread_session") as seed_thread,
        patch("cron.scheduler._maybe_mirror_cron_delivery") as mirror_delivery,
    ):
        result = _deliver_result(
            job, f"report\nMEDIA:{media}",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-unknown-media",
        )

    assert result is not None
    assert "media" in result
    seed_thread.assert_not_called()
    mirror_delivery.assert_not_called()
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1,
        "failed": 0,
        "unknown": 1,
        "targets_delivered": 0,
    }


def test_live_adapter_media_error_prevents_delivery_side_effects_even_with_delivered_receipt(
    monkeypatch, tmp_path
):
    import cron.executions as executions
    import gateway.platforms.base as base

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    media = tmp_path / "failed-media-report.pdf"
    media.write_bytes(b"test-pdf")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _FailedMediaWithDeliveredReceiptAdapter(["text"])
    standalone_calls = AsyncMock(return_value={"success": True})
    job = _job()
    job["attach_to_session"] = True

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
        patch("cron.scheduler._open_continuable_cron_thread", return_value=None),
        patch("cron.scheduler._maybe_mirror_cron_delivery") as mirror_delivery,
        patch("tools.send_message_tool._send_to_platform", new=standalone_calls),
    ):
        result = _deliver_result(
            job, f"report\nMEDIA:{media}",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-failed-media",
        )

    assert result is not None
    assert "provider reported media failure" in result
    mirror_delivery.assert_not_called()
    standalone_calls.assert_not_awaited()
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 2,
        "failed": 0,
        "unknown": 0,
        "targets_delivered": 1,
    }


def test_new_continuation_thread_preserves_preregistered_requested_target(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("receipt-e2e", source="direct")
    adapter = _PlannedTelegramAdapter(["brief"])
    job = _job()
    job["attach_to_session"] = True

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coroutine_threadsafe),
        patch("cron.scheduler._open_continuable_cron_thread", return_value="created-thread") as open_thread,
        patch("cron.scheduler._seed_cron_thread_session"),
        patch("cron.scheduler._maybe_mirror_cron_delivery"),
    ):
        result = _deliver_result(
            job, "continuable brief",
            adapters={Platform.TELEGRAM: adapter}, loop=_running_loop(),
            execution_id=execution["id"], fire_identity="fire-thread",
        )

    assert result is None
    open_thread.assert_called_once()
    assert adapter.last_metadata["_transport_receipt_requested_target"] == {
        "platform": "telegram", "chat_id": "123", "thread_id": "",
    }
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1,
        "failed": 0,
        "unknown": 0,
        "targets_delivered": 0,
    }


def test_standalone_typed_receipt_is_persisted(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("receipt-e2e", source="direct")
    target = TransportTarget("telegram", "123")
    standalone_result = {
        "success": True,
        "message_id": "standalone-1",
        "receipts": (
            TransportReceipt(
                outcome="delivered", provider_message_id="standalone-1",
                requested_target=target, actual_target=target,
                component="text", ordinal=0,
            ),
        ),
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", return_value=standalone_result),
    ):
        result = _deliver_result(
            _job(), "standalone report", adapters=None, loop=None,
            execution_id=execution["id"], fire_identity="fire-standalone-typed",
        )

    assert result is None
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1, "failed": 0, "unknown": 0, "targets_delivered": 1,
    }


def test_standalone_legacy_success_remains_unknown(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("receipt-e2e", source="direct")

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", return_value={"success": True, "message_id": "legacy"}),
    ):
        result = _deliver_result(
            _job(), "standalone report", adapters=None, loop=None,
            execution_id=execution["id"], fire_identity="fire-standalone-legacy",
        )

    assert result is not None
    assert "typed receipt" in result
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 0, "failed": 0, "unknown": 1, "targets_delivered": 0,
    }


def test_bot_chat_exit_zero_persists_observed_unknown_for_exact_query_bytes(
    monkeypatch, tmp_path
):
    import cron.executions as executions
    import cron.scheduler as scheduler

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("bot-receipt", source="direct")
    captured = {}

    def fake_run(argv, **_kwargs):
        query_file = argv[argv.index("--query-file") + 1]
        with open(query_file, encoding="utf-8") as handle:
            captured["message"] = handle.read()
        return subprocess.CompletedProcess(argv, 0, "", "")

    job = {"id": "bot-receipt", "name": "bot receipt", "deliver": "bot-chat:research"}
    with (
        patch("gateway.config.load_gateway_config", return_value=GatewayConfig()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("cron.scheduler._resolve_delivery_targets", return_value=[{
            "platform": "bot-chat", "chat_id": "research", "thread_id": None,
        }]),
        patch.object(scheduler.shutil, "which", return_value="/usr/bin/hermes"),
        patch.object(scheduler.subprocess, "run", side_effect=fake_run),
    ):
        result = _deliver_result(
            job, "bot payload", adapters=None, loop=None,
            execution_id=execution["id"], fire_identity="fire-bot-chat",
        )

    assert result == "bot-chat delivery confirmation unavailable"
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 0, "failed": 0, "unknown": 1, "targets_delivered": 0,
    }
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        row = conn.execute(
            """SELECT c.component, c.ordinal, c.content_hash,
                      a.outcome, a.observed_at
               FROM delivery_attempts a
               JOIN delivery_components c ON c.id=a.component_id"""
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM delivery_components").fetchone()[0]
    assert count == 1
    assert row == (
        "text", 0,
        hashlib.sha256(captured["message"].encode("utf-8")).hexdigest(),
        "unknown", row[4],
    )
    assert row[4] is not None


def test_standalone_text_ack_with_opaque_media_remains_partial(monkeypatch, tmp_path):
    import cron.executions as executions
    import gateway.platforms.base as base

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    media = tmp_path / "standalone-report.pdf"
    media.write_bytes(b"test-pdf")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    execution = executions.create_execution("receipt-e2e", source="direct")
    target = TransportTarget("telegram", "123")
    standalone_result = {
        "success": True,
        "receipts": (TransportReceipt(
            outcome="delivered", provider_message_id="standalone-text",
            requested_target=target, actual_target=target,
            component="text", ordinal=0,
        ),),
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", return_value=standalone_result),
    ):
        result = _deliver_result(
            _job(), f"{'report ' + ('x' * 1100)}\nMEDIA:{media}",
            adapters=None, loop=None,
            execution_id=execution["id"], fire_identity="fire-standalone-media",
        )

    assert result is not None
    assert "media acknowledgement" in result
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1, "failed": 0, "unknown": 1, "targets_delivered": 0,
    }


def test_standalone_typed_text_and_media_acks_complete_target(monkeypatch, tmp_path):
    import cron.executions as executions
    import gateway.platforms.base as base

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    media = tmp_path / "standalone-confirmed.pdf"
    media.write_bytes(b"test-pdf")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    execution = executions.create_execution("receipt-e2e", source="direct")
    target = TransportTarget("telegram", "123")
    standalone_result = {
        "success": True,
        "receipts": (
            TransportReceipt(
                outcome="delivered", provider_message_id="standalone-text",
                requested_target=target, actual_target=target,
                component="text", ordinal=0,
            ),
            TransportReceipt(
                outcome="delivered", provider_message_id="standalone-media",
                requested_target=target, actual_target=target,
                component="media", ordinal=0,
            ),
        ),
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", return_value=standalone_result),
    ):
        result = _deliver_result(
            _job(), f"{'report ' + ('x' * 1100)}\nMEDIA:{media}",
            adapters=None, loop=None,
            execution_id=execution["id"], fire_identity="fire-standalone-confirmed-media",
        )

    assert result is None
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 2, "failed": 0, "unknown": 0, "targets_delivered": 1,
    }
