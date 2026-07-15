from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import discord_connector_bootstrap as bootstrap
from gateway.discord_connector_protocol import canonical_json_bytes, sha256_json
from gateway.discord_connector_service import (
    DiscordConnectorAcceptedMessage,
    DiscordConnectorHistoryReaderPeer,
    DiscordConnectorRuntime,
    DiscordConnectorUnixServer,
    DurableDiscordConnectorJournal,
)
from plugins.platforms.discord.public_connector import DiscordPublicConnectorPolicy


@pytest.fixture
def unix_socket_tmp_path():
    # Keep the literal AF_UNIX path below macOS' 104-byte limit while retaining
    # the real socket/listener behavior exercised by these readiness tests.
    base = "/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
    with tempfile.TemporaryDirectory(prefix="mdcb-", dir=base) as directory:
        yield Path(directory)


class _Backend:
    def prove_public_target(self, _channel_id):  # pragma: no cover - not used here
        raise AssertionError("unexpected target proof")

    def fetch_guild_history(self, *_args, **_kwargs):  # pragma: no cover - not used
        raise AssertionError("unexpected history read")

    def send_public_message(self, *_args, **_kwargs):  # pragma: no cover - not used
        return DiscordConnectorAcceptedMessage("500", True)


class _ReadyClient:
    def readiness_identity(self):
        return {
            "discord_gateway_ready": True,
            "bot_user_id": "500",
            "intents": ["guilds", "guild_messages", "message_content"],
            "dm_messages": False,
            "require_mention": True,
            "auto_thread": True,
            "thread_require_mention": False,
            "public_only": True,
            "author_policy": "exact_ids_or_roles",
            "free_response_channel_ids": [],
            "allowed_user_ids": ["400"],
            "allowed_role_ids": ["500"],
            "allowed_channel_ids": ["200"],
            "reviewed_cron_history_targets_sha256": (
                DiscordPublicConnectorPolicy.build(
                    allowed_guild_ids=["100"],
                    allowed_channel_ids=["200"],
                    allowed_user_ids=["400"],
                    allowed_role_ids=["500"],
                ).reviewed_cron_history_targets_sha256
            ),
            "public_target_proofs": [
                {
                    "target_type": "public_guild_channel",
                    "guild_id": "100",
                    "channel_id": "200",
                }
            ],
        }

    def stop(self):
        return None


def _running_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    canary_history_reader: DiscordConnectorHistoryReaderPeer | None = None,
):
    uid = os.geteuid()
    gid = os.getegid()
    os.chown(tmp_path, uid, gid)
    config_path = tmp_path / "connector.json"
    config_raw = b'{"sealed":true}\n'
    config_path.write_bytes(config_raw)
    os.chown(config_path, uid, gid)
    config_path.chmod(0o440)
    token = tmp_path / "bot-token"
    token.write_text("never-record-this-token", encoding="ascii")
    os.chown(token, uid, gid)
    token.chmod(0o400)
    journal_path = tmp_path / "journal.sqlite3"
    journal = DurableDiscordConnectorJournal.bootstrap(journal_path)
    os.chown(journal_path, uid, gid)
    journal_path.chmod(0o600)
    policy = DiscordPublicConnectorPolicy.build(
        allowed_guild_ids=["100"],
        allowed_channel_ids=["200"],
        allowed_user_ids=["400"],
        allowed_role_ids=["500"],
    )
    config = bootstrap.DiscordConnectorConfig(
        config_path=config_path,
        socket_path=tmp_path / "connector.sock",
        gateway_unit="hermes-cloud-gateway.service",
        connector_unit="muncho-discord-connector.service",
        gateway_uid=max(uid, 1),
        connector_uid=uid,
        connector_gid=gid,
        connection_timeout_seconds=1.0,
        token_file=token,
        credentials_directory=tmp_path,
        policy=policy,
        ready_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        journal_path=journal_path,
        journal_busy_timeout_ms=1_000,
        canary_history_reader=canary_history_reader,
    )
    runtime = DiscordConnectorRuntime(backend=_Backend(), journal=journal)
    server = DiscordConnectorUnixServer(
        config.socket_path,
        runtime=runtime,
        expected_gateway_uid=max(uid, 1),
        history_reader_peer=canary_history_reader,
        connection_timeout_seconds=1,
    )
    running = bootstrap.DiscordConnectorBootstrap(
        config=config,
        journal=journal,
        client=_ReadyClient(),
        runtime=runtime,
        server=server,
    )
    trusted_file = bootstrap._trusted_file

    def trusted(path, **kwargs):
        if Path(path) == config_path:
            return config_raw
        return trusted_file(path, **kwargs)

    monkeypatch.setattr(bootstrap, "_trusted_file", trusted)
    return running, config_raw


def _write_receipt(path: Path, value: dict) -> None:
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o400)


def test_canary_history_reader_is_exact_account_and_not_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _account(name):
        calls.append(name)
        return SimpleNamespace(
            pw_name=bootstrap.CANARY_HISTORY_READER_SERVICE_USER,
            pw_uid=2203,
            pw_gid=2303,
            pw_dir="/nonexistent",
            pw_shell="/usr/sbin/nologin",
        )

    monkeypatch.setattr(bootstrap.pwd, "getpwnam", _account)
    assert bootstrap._canary_history_reader(None) is None
    assert calls == []
    peer = bootstrap._canary_history_reader({
        "service_unit": bootstrap.CANARY_HISTORY_READER_SERVICE_UNIT,
        "service_user": bootstrap.CANARY_HISTORY_READER_SERVICE_USER,
        "requester_user_id": bootstrap.CANARY_REQUESTER_USER_ID,
    })
    assert peer is not None
    assert peer.expected_uid == 2203
    assert peer.service_unit == bootstrap.CANARY_HISTORY_READER_SERVICE_UNIT
    assert peer.requester_user_id == bootstrap.CANARY_REQUESTER_USER_ID
    assert calls == [bootstrap.CANARY_HISTORY_READER_SERVICE_USER]

    with pytest.raises(ValueError, match="not the pinned canary peer"):
        bootstrap._canary_history_reader({
            "service_unit": "hermes-cloud-gateway.service",
            "service_user": bootstrap.CANARY_HISTORY_READER_SERVICE_USER,
            "requester_user_id": bootstrap.CANARY_REQUESTER_USER_ID,
        })


def test_readiness_is_live_secret_free_and_revalidates_process_identity(
    unix_socket_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = unix_socket_tmp_path
    running, _config_raw = _running_boundary(tmp_path, monkeypatch)
    readiness = tmp_path / "readiness.json"
    try:
        receipt = bootstrap.publish_readiness(
            running,
            path=readiness,
            now_unix=1_700_000_000,
            start_time_reader=lambda _pid: 123,
            notifier=lambda *_args, **_kwargs: True,
        )
        serialized = readiness.read_bytes()
        assert b"never-record-this-token" not in serialized
        assert b"token_sha" not in serialized
        assert receipt["discord"]["public_target_proofs"][0]["channel_id"] == "200"
        assert receipt["allowed_role_ids"] == ["500"]

        loaded = bootstrap.load_readiness_receipt(
            running.config,
            readiness,
            start_time_reader=lambda _pid: 123,
        )
        assert loaded == receipt
        with pytest.raises(ValueError, match="readiness receipt is invalid"):
            bootstrap.load_readiness_receipt(
                running.config,
                readiness,
                start_time_reader=lambda _pid: 124,
            )
    finally:
        running.close()


def test_readiness_attests_exact_canary_history_reader_without_raw_user_id(
    unix_socket_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = DiscordConnectorHistoryReaderPeer(
        service_unit=bootstrap.CANARY_HISTORY_READER_SERVICE_UNIT,
        expected_uid=max(os.geteuid(), 1) + 100,
        requester_user_id=bootstrap.CANARY_REQUESTER_USER_ID,
    )
    running, _config_raw = _running_boundary(
        unix_socket_tmp_path,
        monkeypatch,
        canary_history_reader=peer,
    )
    readiness = unix_socket_tmp_path / "reader-readiness.json"
    try:
        receipt = bootstrap.publish_readiness(
            running,
            path=readiness,
            now_unix=1_700_000_000,
            start_time_reader=lambda _pid: 123,
            notifier=lambda *_args, **_kwargs: True,
        )
        assert receipt["canary_history_reader"] == peer.readiness_mapping()
        assert bootstrap.CANARY_REQUESTER_USER_ID not in readiness.read_text(
            encoding="utf-8"
        )
        loaded = bootstrap.load_readiness_receipt(
            running.config,
            readiness,
            start_time_reader=lambda _pid: 123,
        )
        assert loaded == receipt
    finally:
        running.close()


def test_readiness_recomputes_cleanup_invariants_and_notify_failure_removes_file(
    unix_socket_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = unix_socket_tmp_path
    running, _config_raw = _running_boundary(tmp_path, monkeypatch)
    running.server.start()
    good = bootstrap.build_readiness_receipt(
        running,
        now_unix=1_700_000_000,
        start_time_reader=lambda _pid: 123,
    )
    bad = json.loads(json.dumps(good))
    bad["journal_cleanup"]["send_state_counts"] = {"uncertain": 1}
    unsigned = {key: value for key, value in bad.items() if key != "receipt_sha256"}
    bad["receipt_sha256"] = sha256_json(unsigned)
    bad_path = tmp_path / "bad-readiness.json"
    _write_receipt(bad_path, bad)
    try:
        with pytest.raises(ValueError, match="readiness receipt is invalid"):
            bootstrap.load_readiness_receipt(
                running.config,
                bad_path,
                start_time_reader=lambda _pid: 123,
            )

        failed_path = tmp_path / "failed-readiness.json"
        with pytest.raises(RuntimeError, match="requires systemd"):
            bootstrap.publish_readiness(
                running,
                path=failed_path,
                now_unix=1_700_000_000,
                start_time_reader=lambda _pid: 123,
                notifier=lambda *_args, **_kwargs: False,
            )
        assert failed_path.exists() is False
    finally:
        running.close()


def test_post_ready_discord_failure_stops_the_connector_service() -> None:
    shutdown = threading.Event()
    stop = threading.Event()
    failed = threading.Event()
    running = SimpleNamespace(
        client=SimpleNamespace(wait_for_health_failure=lambda _timeout: True),
        server=SimpleNamespace(shutdown=shutdown.set),
    )

    bootstrap._watch_discord_health(running, stop, failed)
    assert failed.is_set() is True
    assert shutdown.is_set() is True
