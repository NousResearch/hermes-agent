from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.discord_command_projection import DiscordProjectionMismatch
from plugins.platforms.discord import projected_adapter


class _RegistrationContext:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_platform(self, **kwargs):
        self.calls.append(kwargs)


class _LocalCommand:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self, _tree) -> dict:
        return deepcopy(self.payload)


class _RemoteCommand:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self) -> dict:
        return deepcopy(self.payload)


class _Tree:
    def __init__(self, local: list[dict], remote: list[dict]) -> None:
        self._local = [_LocalCommand(row) for row in local]
        self._remote = [_RemoteCommand(row) for row in remote]
        self.fetch_commands = AsyncMock(return_value=self._remote)

    def get_commands(self):
        return list(self._local)


def test_registration_reuses_adapter_metadata_and_swaps_only_factory() -> None:
    ctx = _RegistrationContext()

    projected_adapter.register(ctx)

    assert len(ctx.calls) == 1
    call = ctx.calls[0]
    assert call["name"] == "discord"
    assert call["label"] == "Discord"
    assert call["adapter_factory"] is projected_adapter._build_projected_adapter
    assert call["standalone_sender_fn"] is not None
    assert call["required_env"] == ["DISCORD_BOT_TOKEN"]


def test_projected_factory_returns_the_bounded_subclass(monkeypatch) -> None:
    monkeypatch.setattr(
        projected_adapter.ProjectedDiscordAdapter,
        "__init__",
        lambda self, config: None,
    )

    created = projected_adapter._build_projected_adapter(object())

    assert isinstance(created, projected_adapter.ProjectedDiscordAdapter)


@pytest.mark.asyncio
async def test_native_sync_reads_back_the_exact_projection(monkeypatch) -> None:
    payloads = [
        {"name": "new", "description": "Start a new conversation"},
        {
            "name": "model",
            "description": "Show or change the model",
            "options": [
                {
                    "type": 3,
                    "name": "name",
                    "description": "Model name",
                    "required": False,
                }
            ],
        },
    ]
    tree = _Tree(payloads, list(reversed(payloads)))
    adapter = object.__new__(projected_adapter.ProjectedDiscordAdapter)
    adapter._client = SimpleNamespace(tree=tree)
    adapter._existing_command_to_payload = lambda command: command.to_dict()
    adapter._READBACK_ATTEMPTS = 1

    async def base_sync(_self):
        return {
            "total": 2,
            "unchanged": 2,
            "updated": 0,
            "recreated": 0,
            "created": 0,
            "deleted": 0,
        }

    monkeypatch.setattr(
        projected_adapter._adapter.DiscordAdapter,
        "_safe_sync_slash_commands",
        base_sync,
    )

    result = await adapter._safe_sync_slash_commands()

    assert result["total"] == 2
    assert tree.fetch_commands.await_count == 1
    assert adapter._discord_command_projection_revision
    assert (
        adapter._discord_command_projection_verified_revision
        == adapter._discord_command_projection_revision
    )


@pytest.mark.asyncio
async def test_native_sync_refuses_false_completion_after_remote_drift(
    monkeypatch,
) -> None:
    local = [{"name": "new", "description": "Start a new conversation"}]
    remote = [{"name": "new", "description": "drifted"}]
    tree = _Tree(local, remote)
    adapter = object.__new__(projected_adapter.ProjectedDiscordAdapter)
    adapter._client = SimpleNamespace(tree=tree)
    adapter._existing_command_to_payload = lambda command: command.to_dict()
    adapter._READBACK_ATTEMPTS = 1

    async def base_sync(_self):
        return {
            "total": 1,
            "unchanged": 0,
            "updated": 1,
            "recreated": 0,
            "created": 0,
            "deleted": 0,
        }

    monkeypatch.setattr(
        projected_adapter._adapter.DiscordAdapter,
        "_safe_sync_slash_commands",
        base_sync,
    )

    with pytest.raises(DiscordProjectionMismatch):
        await adapter._safe_sync_slash_commands()


def test_native_sync_uses_the_shared_projection_revision() -> None:
    payloads = [{"name": "new", "description": "Start a new conversation"}]
    tree = _Tree(payloads, payloads)
    adapter = object.__new__(projected_adapter.ProjectedDiscordAdapter)
    adapter._client = SimpleNamespace(tree=tree)

    assert (
        adapter._desired_command_sync_fingerprint()
        == adapter._desired_command_projection().revision
    )


def test_same_local_fingerprint_does_not_bypass_remote_readback(monkeypatch) -> None:
    adapter = object.__new__(projected_adapter.ProjectedDiscordAdapter)

    monkeypatch.setattr(
        projected_adapter._adapter.DiscordAdapter,
        "_command_sync_skip_reason",
        lambda _self, _app_id, _fingerprint: (
            "same slash-command fingerprint already synced"
        ),
    )

    assert adapter._command_sync_skip_reason("app", "revision") is None


def test_rate_limit_backoff_still_skips_reconciliation(monkeypatch) -> None:
    adapter = object.__new__(projected_adapter.ProjectedDiscordAdapter)

    monkeypatch.setattr(
        projected_adapter._adapter.DiscordAdapter,
        "_command_sync_skip_reason",
        lambda _self, _app_id, _fingerprint: "retry after 12.0s",
    )

    assert (
        adapter._command_sync_skip_reason("app", "revision")
        == "retry after 12.0s"
    )


@pytest.mark.asyncio
async def test_native_readback_retries_bounded_remote_propagation() -> None:
    desired_rows = [
        {"name": "new", "description": "Start a new conversation"}
    ]
    drifted_rows = [{"name": "new", "description": "stale"}]
    tree = _Tree(desired_rows, drifted_rows)
    tree.fetch_commands = AsyncMock(
        side_effect=[
            [_RemoteCommand(row) for row in drifted_rows],
            [_RemoteCommand(row) for row in desired_rows],
        ]
    )
    adapter = object.__new__(projected_adapter.ProjectedDiscordAdapter)
    adapter._client = SimpleNamespace(tree=tree)
    adapter._existing_command_to_payload = lambda command: command.to_dict()
    adapter._READBACK_ATTEMPTS = 2
    adapter._sleep_between_command_sync_mutations = AsyncMock()

    observed = await adapter._verify_remote_projection(
        adapter._desired_command_projection()
    )

    assert observed.revision == adapter._desired_command_projection().revision
    assert tree.fetch_commands.await_count == 2
    adapter._sleep_between_command_sync_mutations.assert_awaited_once()
