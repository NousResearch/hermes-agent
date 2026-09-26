"""Buzz destination allowlist (item 2)."""
from __future__ import annotations

import asyncio

import pytest

from tests.gateway.buzz_forward_support import (
    CHANNEL,
    DM_CHANNEL,
    _DESTINATION_DENIED,
    _ScriptedCli,
    _buzz_mod,
    _make_adapter,
    _standalone_send,
)

pytest_plugins = ["tests.gateway.buzz_forward_support"]


class TestDestinationAllowlist:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["reaction", "delete"])
    @pytest.mark.parametrize("extra", [
        {"allowed_destinations": [CHANNEL]},
        {"allowed_destinations": []},
        {"forward_only": True},
    ])
    async def test_event_writes_refuse_disallowed_destination(self, operation, extra):
        adapter = _make_adapter(extra)
        adapter.cli_path = "/synthetic/buzz"
        adapter._run_cli = _ScriptedCli()
        if operation == "reaction":
            result = await adapter.send_reaction(DM_CHANNEL, "event-1", "eyes")
        else:
            result = await adapter.delete_message(DM_CHANNEL, "event-1")
        assert result is False
        assert adapter._run_cli.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["reaction", "delete"])
    @pytest.mark.parametrize("extra", [{}, {"allowed_destinations": [CHANNEL]}])
    async def test_event_writes_preserve_allowed_and_default_behavior(self, operation, extra):
        adapter = _make_adapter(extra)
        adapter.cli_path = "/synthetic/buzz"
        cli = adapter._run_cli = _ScriptedCli()
        group, command = ("reactions", "add") if operation == "reaction" else ("messages", "delete")
        cli.script(group, command, {"accepted": True})
        if operation == "reaction":
            result = await adapter.send_reaction(CHANNEL, "event-1", "eyes")
        else:
            result = await adapter.delete_message(CHANNEL, "event-1")
        assert result is True
        assert len(cli.calls) == 1
        assert cli.calls[0][0][:2] == [group, command]

    @pytest.mark.asyncio
    async def test_send_refuses_empty_allowlist(self, caplog):
        adapter = _make_adapter({"allowed_destinations": []})
        adapter._run_cli = _ScriptedCli()
        result = await adapter.send(CHANNEL, "hello")
        assert result.success is False
        assert result.error == _DESTINATION_DENIED
        assert adapter._run_cli.calls == []
        assert CHANNEL[:8] in caplog.text
        assert CHANNEL not in caplog.text

    @pytest.mark.asyncio
    async def test_send_refuses_unknown_destination(self):
        adapter = _make_adapter({"allowed_destinations": [CHANNEL]})
        adapter._run_cli = _ScriptedCli()
        result = await adapter.send(DM_CHANNEL, "hello")
        assert result.success is False
        assert result.error == _DESTINATION_DENIED
        assert adapter._run_cli.calls == []

    @pytest.mark.asyncio
    async def test_send_allows_listed_destination(self):
        adapter = _make_adapter({"allowed_destinations": [CHANNEL]})
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "ok1"})
        adapter._run_cli = cli
        result = await adapter.send(CHANNEL, "hello")
        assert result.success is True
        assert result.message_id == "ok1"

    @pytest.mark.asyncio
    async def test_edit_and_document_use_the_same_gate(self, tmp_path):
        adapter = _make_adapter({"allowed_destinations": [CHANNEL]})
        adapter._run_cli = _ScriptedCli()
        denied = await adapter.edit_message(DM_CHANNEL, "e1", "edit")
        assert denied.success is False
        assert denied.error == _DESTINATION_DENIED
        doc = tmp_path / "note.txt"
        doc.write_text("x", encoding="utf-8")
        denied_doc = await adapter.send_document(DM_CHANNEL, str(doc))
        assert denied_doc.success is False
        assert denied_doc.error == _DESTINATION_DENIED

    def test_standalone_send_refuses_empty_allowlist(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        cli = tmp_path / "buzz"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")

        async def fake_exec(*_a, **_k):
            raise AssertionError("CLI must not run for a denied destination")

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        result = asyncio.run(
            _standalone_send(
                PlatformConfig(
                    enabled=True,
                    extra={
                        "relay_url": "https://test.relay",
                        "cli_path": str(cli),
                        "home_channel": CHANNEL,
                        "allowed_destinations": [],
                    },
                ),
                CHANNEL,
                "hello",
            )
        )
        assert "destination not allowed" in str(result)

    def test_standalone_send_denies_unset_dest_in_forward_only(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        cli = tmp_path / "buzz"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")

        async def fake_exec(*_a, **_k):
            raise AssertionError("CLI must not run for a denied destination")

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        monkeypatch.setenv("BUZZ_FORWARD_ONLY", "1")
        result = asyncio.run(
            _standalone_send(
                PlatformConfig(
                    enabled=True,
                    extra={
                        "relay_url": "https://test.relay",
                        "cli_path": str(cli),
                        "home_channel": CHANNEL,
                        "forward_only": True,
                    },
                ),
                CHANNEL,
                "hello",
            )
        )
        assert "destination not allowed" in str(result)

    def test_env_csv_is_honored(self, monkeypatch):
        monkeypatch.setenv("BUZZ_ALLOWED_DESTINATIONS", CHANNEL)
        adapter = _make_adapter()
        assert adapter._allowed_destinations == {CHANNEL}
