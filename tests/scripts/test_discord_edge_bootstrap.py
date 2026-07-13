from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.canonical_writer_boundary import (
    DEFAULT_DISCORD_EDGE_SOCKET_PATH,
    DEFAULT_DISCORD_EDGE_UNIT,
    DEFAULT_GATEWAY_UNIT,
)
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.discord_edge_runtime import (
    DiscordEdgeRuntime,
    DiscordEdgeRuntimeError,
    DurableDiscordEdgeJournal,
)
from gateway import discord_edge_bootstrap as bootstrap_module
from gateway.discord_edge_bootstrap import (
    DiscordEdgeBootstrap,
    bootstrap_journal,
    build_service,
    load_service_config,
    serve_service,
)
from gateway.discord_edge_service import DiscordEdgeUnixServer
from gateway.discord_rest_edge import DiscordRestEdgeError
from scripts import discord_edge_bootstrap as compatibility_bootstrap
from scripts import discord_edge_service as compatibility_service


class _FakeAdapter:
    def __init__(self) -> None:
        self.closed = False

    def close(self):
        self.closed = True

    def prove_public_message_send(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def prove_public_message_edit(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def prove_public_thread_create(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def prove_public_readback(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def send_public_message(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def edit_public_message(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def create_public_thread(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def read_public_message(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")

    def read_created_public_thread(self, *_args, **_kwargs):
        raise AssertionError("not exercised during bootstrap")


def test_source_compatibility_modules_delegate_to_packaged_boundary():
    assert compatibility_bootstrap.main is bootstrap_module.main
    assert compatibility_service.DiscordEdgeUnixServer is DiscordEdgeUnixServer


def _write_file(path: Path, body: bytes, mode: int) -> None:
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(body)
    path.chmod(mode)


def _config_value(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    writer_private = Ed25519PrivateKey.generate()
    edge_private = Ed25519PrivateKey.generate()
    writer_public_path = tmp_path / "writer-capability-public.pem"
    edge_private_path = tmp_path / "edge-receipt-private.pem"
    _write_file(
        writer_public_path,
        writer_private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
        0o440,
    )
    _write_file(
        edge_private_path,
        edge_private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        0o400,
    )
    credentials_directory = tmp_path / "credentials"
    credentials_directory.mkdir(exist_ok=True)
    credentials_directory.chmod(0o700)
    token_file = credentials_directory / "discord-bot-token"
    _write_file(token_file, b"test.discord.bot.token.123456789\n", 0o400)
    journal_directory = tmp_path / "journal"
    journal_directory.mkdir(exist_ok=True)
    journal_directory.chmod(0o700)
    socket_directory = tmp_path / "socket"
    socket_directory.mkdir(exist_ok=True)
    socket_directory.chmod(0o700)
    return {
        "service": {
            "socket_path": str(socket_directory / "edge.sock"),
            "gateway_unit": DEFAULT_GATEWAY_UNIT,
            "edge_unit": DEFAULT_DISCORD_EDGE_UNIT,
            "gateway_uid": os.geteuid() + 1,
            "edge_uid": os.geteuid(),
            "edge_gid": os.getegid(),
            "connection_timeout_seconds": 20,
            "max_connections": 4,
        },
        "keys": {
            "writer_capability_public_key_file": str(writer_public_path),
            "writer_capability_public_key_id": ed25519_public_key_id(
                writer_private.public_key()
            ),
            "edge_receipt_private_key_file": str(edge_private_path),
            "edge_receipt_public_key_id": ed25519_public_key_id(
                edge_private.public_key()
            ),
        },
        "discord": {
            "token_file": str(token_file),
            "credentials_directory": str(credentials_directory),
            "api_timeout_seconds": 5,
        },
        "journal": {
            "path": str(journal_directory / "discord-edge.sqlite3"),
            "busy_timeout_ms": 5_000,
        },
        "runtime": {"max_proof_age_ms": 5_000},
    }


def _write_config(tmp_path: Path, value: dict, *, mode: int = 0o440) -> Path:
    path = tmp_path / "discord-edge.json"
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(mode)
    return path


def _load(path: Path):
    return load_service_config(
        path,
        _expected_owner_uid=os.geteuid(),
        _require_root_owned_parents=False,
        _expected_socket_path=None,
        _expected_journal_path=None,
    )


def _loaded_config(tmp_path: Path):
    return _load(_write_config(tmp_path, _config_value(tmp_path)))


def test_loads_exact_local_config_and_pinned_distinct_ed25519_keys(tmp_path):
    config = _loaded_config(tmp_path)

    assert config.gateway_uid != config.edge_uid
    assert config.gateway_unit == DEFAULT_GATEWAY_UNIT
    assert config.edge_unit == DEFAULT_DISCORD_EDGE_UNIT
    assert ed25519_public_key_id(config.writer_public_key) == (
        config.writer_capability_public_key_id
    )
    assert ed25519_public_key_id(config.edge_private_key.public_key()) == (
        config.edge_receipt_public_key_id
    )
    assert config.writer_capability_public_key_id != (
        config.edge_receipt_public_key_id
    )
    assert config.token_file.parent == config.credentials_directory


def test_production_loader_pins_socket_and_journal_paths(tmp_path):
    value = _config_value(tmp_path)
    path = _write_config(tmp_path, value)
    with pytest.raises(ValueError, match="pinned edge socket"):
        load_service_config(
            path,
            _expected_owner_uid=os.geteuid(),
            _require_root_owned_parents=False,
        )

    value["service"]["socket_path"] = str(DEFAULT_DISCORD_EDGE_SOCKET_PATH)
    path = _write_config(tmp_path, value)
    with pytest.raises(ValueError, match="pinned edge journal"):
        load_service_config(
            path,
            _expected_owner_uid=os.geteuid(),
            _require_root_owned_parents=False,
        )


@pytest.mark.parametrize("mode", [0o400, 0o600, 0o640, 0o644, 0o660])
def test_config_rejects_mutable_or_world_readable_mode(tmp_path, mode):
    path = _write_config(tmp_path, _config_value(tmp_path), mode=mode)

    with pytest.raises(ValueError, match="mode"):
        _load(path)


def test_config_rejects_symlink_duplicate_unknown_and_embedded_token(tmp_path):
    value = _config_value(tmp_path)
    original = _write_config(tmp_path, value)
    link = tmp_path / "edge-config-link.json"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _load(link)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"service":{},"service":{}}', encoding="utf-8")
    duplicate.chmod(0o440)
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _load(duplicate)

    value = _config_value(tmp_path)
    value["discord"]["token"] = "must-not-be-embedded"
    with pytest.raises(ValueError, match="must not embed secret material"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize(
    "key_field,mode",
    [
        ("writer_capability_public_key_file", 0o400),
        ("writer_capability_public_key_file", 0o444),
        ("edge_receipt_private_key_file", 0o440),
        ("edge_receipt_private_key_file", 0o600),
    ],
)
def test_key_files_require_exact_owner_group_and_mode(tmp_path, key_field, mode):
    value = _config_value(tmp_path)
    Path(value["keys"][key_field]).chmod(mode)

    with pytest.raises(ValueError, match="mode must"):
        _load(_write_config(tmp_path, value))


def test_key_files_reject_symlinks_hardlinks_and_missing_files(tmp_path):
    value = _config_value(tmp_path)
    writer_path = Path(value["keys"]["writer_capability_public_key_file"])
    link = tmp_path / "writer-key-link.pem"
    link.symlink_to(writer_path)
    value["keys"]["writer_capability_public_key_file"] = str(link)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    edge_path = Path(value["keys"]["edge_receipt_private_key_file"])
    os.link(edge_path, tmp_path / "edge-key-hardlink.pem")
    with pytest.raises(ValueError, match="exactly one link"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    Path(value["keys"]["edge_receipt_private_key_file"]).unlink()
    with pytest.raises(ValueError, match="private key is unavailable"):
        _load(_write_config(tmp_path, value))


def test_key_ids_are_lowercase_pinned_and_pem_is_canonical(tmp_path):
    value = _config_value(tmp_path)
    value["keys"]["writer_capability_public_key_id"] = "f" * 64
    with pytest.raises(ValueError, match="pinned key ID"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["keys"]["edge_receipt_public_key_id"] = value["keys"][
        "edge_receipt_public_key_id"
    ].upper()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    edge_path = Path(value["keys"]["edge_receipt_private_key_file"])
    body = edge_path.read_bytes()
    _write_file(edge_path, body + b"trailing data\n", 0o400)
    with pytest.raises(ValueError, match="exact PKCS#8 PEM"):
        _load(_write_config(tmp_path, value))


def test_key_identity_policy_is_bound_to_configured_edge_uid_and_gid(tmp_path):
    value = _config_value(tmp_path)
    value["service"]["edge_uid"] = os.geteuid() + 10
    value["service"]["gateway_uid"] = os.geteuid() + 11
    with pytest.raises(ValueError, match="private key owner is not trusted"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["service"]["edge_gid"] = os.getegid() + 10
    with pytest.raises(ValueError, match="edge GID"):
        _load(_write_config(tmp_path, value))


def test_loader_never_falls_back_to_environment_or_remote_secret_lookup(
    tmp_path,
    monkeypatch,
):
    value = _config_value(tmp_path)
    missing = Path(value["keys"]["edge_receipt_private_key_file"])
    missing.unlink()
    monkeypatch.setenv("MUNCHO_DISCORD_EDGE_PRIVATE_KEY", "ignored")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ignored")

    with pytest.raises(ValueError, match="private key is unavailable"):
        _load(_write_config(tmp_path, value))


def test_token_must_be_direct_child_of_explicit_credentials_directory(tmp_path):
    value = _config_value(tmp_path)
    value["discord"]["credentials_directory"] = str(tmp_path)

    with pytest.raises(ValueError, match="directly inside"):
        _load(_write_config(tmp_path, value))


def test_explicit_journal_bootstrap_then_default_open_succeeds(tmp_path):
    config = _loaded_config(tmp_path)

    created = bootstrap_journal(config)
    opened = DurableDiscordEdgeJournal(
        config.journal_path,
        busy_timeout_ms=config.journal_busy_timeout_ms,
    )

    assert created.path == config.journal_path
    assert opened.path == config.journal_path
    assert Path(f"{config.journal_path}.initialized").is_file()
    database_stat = config.journal_path.stat()
    marker_stat = Path(f"{config.journal_path}.initialized").stat()
    assert stat.S_IMODE(config.journal_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(database_stat.st_mode) == 0o600
    assert stat.S_IMODE(marker_stat.st_mode) == 0o600
    assert database_stat.st_uid == marker_stat.st_uid == config.edge_uid


def test_normal_restart_reopens_same_journal_without_reinit_or_resigning(tmp_path):
    config = _loaded_config(tmp_path)
    bootstrap_journal(config)
    marker = Path(f"{config.journal_path}.initialized")
    original_database_identity = (
        config.journal_path.stat().st_dev,
        config.journal_path.stat().st_ino,
    )
    original_marker_identity = (marker.stat().st_dev, marker.stat().st_ino)
    original_marker = marker.read_bytes()
    adapters = []

    def adapter_factory(*_args, **_kwargs):
        adapter = _FakeAdapter()
        adapters.append(adapter)
        return adapter

    first = build_service(config, _adapter_factory=adapter_factory)
    first.close()
    second = build_service(config, _adapter_factory=adapter_factory)
    second.close()

    assert (
        config.journal_path.stat().st_dev,
        config.journal_path.stat().st_ino,
    ) == original_database_identity
    assert (marker.stat().st_dev, marker.stat().st_ino) == original_marker_identity
    assert marker.read_bytes() == original_marker
    assert len(adapters) == 2
    assert all(adapter.closed for adapter in adapters)
    assert ed25519_public_key_id(
        first.runtime.edge_private_key.public_key()
    ) == ed25519_public_key_id(second.runtime.edge_private_key.public_key())


def test_journal_bootstrap_refuses_existing_database_or_orphan_marker(tmp_path):
    config = _loaded_config(tmp_path)
    bootstrap_journal(config)
    with pytest.raises(FileExistsError, match="existing Discord edge journal"):
        bootstrap_journal(config)

    other = _loaded_config(tmp_path / "other")
    marker = Path(f"{other.journal_path}.initialized")
    marker.write_text("orphan\n", encoding="ascii")
    marker.chmod(0o600)
    with pytest.raises(FileExistsError, match="existing Discord edge journal"):
        bootstrap_journal(other)


def test_normal_startup_refuses_missing_database_or_marker(tmp_path):
    config = _loaded_config(tmp_path)
    adapter_calls = []

    with pytest.raises(DiscordEdgeRuntimeError, match="explicit bootstrap"):
        build_service(
            config,
            _adapter_factory=lambda *_args, **_kwargs: adapter_calls.append(True),
        )
    assert adapter_calls == []

    bootstrap_journal(config)
    Path(f"{config.journal_path}.initialized").unlink()
    with pytest.raises(DiscordEdgeRuntimeError, match="marker"):
        build_service(
            config,
            _adapter_factory=lambda *_args, **_kwargs: adapter_calls.append(True),
        )
    assert adapter_calls == []


def test_build_wires_existing_journal_token_adapter_runtime_and_exact_peer(
    tmp_path,
):
    config = _loaded_config(tmp_path)
    bootstrap_journal(config)
    calls = []
    adapter = _FakeAdapter()

    def adapter_factory(path, **kwargs):
        calls.append((path, kwargs))
        return adapter

    built = build_service(config, _adapter_factory=adapter_factory)

    assert calls == [(
        config.token_file,
        {
            "credentials_directory": config.credentials_directory,
            "expected_owner_uid": config.edge_uid,
            "timeout_seconds": config.api_timeout_seconds,
        },
    )]
    assert isinstance(built.runtime, DiscordEdgeRuntime)
    assert isinstance(built.server, DiscordEdgeUnixServer)
    assert built.server.expected_client_uid == config.gateway_uid
    assert built.server.gateway_unit == DEFAULT_GATEWAY_UNIT
    assert built.runtime.target_prover is adapter
    assert built.runtime.transport is adapter

    built.close()
    built.close()
    assert adapter.closed is True


def test_build_rejects_runtime_identity_drift_before_opening_journal(tmp_path):
    # Disable the key-owner mismatch only for this process-identity test by
    # loading first and then replacing the immutable scalar in a test object.
    config = _loaded_config(tmp_path)
    config = replace(config, edge_uid=os.geteuid() + 10)

    with pytest.raises(PermissionError, match="UID/GID"):
        build_service(config)


def test_real_token_boundary_rejects_wrong_mode_before_http_adapter_creation(tmp_path):
    config = _loaded_config(tmp_path)
    bootstrap_journal(config)
    config.token_file.chmod(0o600)

    with pytest.raises(DiscordRestEdgeError, match="exact mode 0400"):
        build_service(config)


def test_real_token_boundary_accepts_exact_owner_mode_without_outbound_call(tmp_path):
    config = _loaded_config(tmp_path)
    bootstrap_journal(config)

    built = build_service(config)
    built.close()

    assert built.adapter is not None


def test_build_failure_closes_adapter(tmp_path):
    config = _loaded_config(tmp_path)
    bootstrap_journal(config)
    adapter = _FakeAdapter()

    def fail_server(*_args, **_kwargs):
        raise RuntimeError("server construction failed")

    with pytest.raises(RuntimeError, match="server construction failed"):
        build_service(
            config,
            _adapter_factory=lambda *_args, **_kwargs: adapter,
            _server_factory=fail_server,
        )
    assert adapter.closed is True


def test_serve_service_always_shuts_down_and_closes_adapter():
    events = []

    class _Server:
        def serve_forever(self):
            events.append("serve")
            raise RuntimeError("stop")

        def shutdown(self):
            events.append("shutdown")

    class _Adapter:
        def close(self):
            events.append("adapter_close")

    bootstrap = DiscordEdgeBootstrap(
        config=SimpleNamespace(),
        journal=SimpleNamespace(),
        adapter=_Adapter(),
        runtime=SimpleNamespace(),
        server=_Server(),
    )

    with pytest.raises(RuntimeError, match="stop"):
        serve_service(bootstrap)

    assert events == ["serve", "shutdown", "adapter_close"]


def test_cli_bootstrap_journal_is_explicit_and_does_not_build_service(monkeypatch):
    config = SimpleNamespace()
    calls = []
    monkeypatch.setattr(bootstrap_module, "load_service_config", lambda _path: config)
    monkeypatch.setattr(
        bootstrap_module,
        "bootstrap_journal",
        lambda observed: calls.append(("bootstrap", observed)),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "build_service",
        lambda _config: calls.append(("build", _config)),
    )

    result = bootstrap_module.main(
        ["--config", "/root/edge.json", "--bootstrap-journal"]
    )

    assert result == 0
    assert calls == [("bootstrap", config)]
