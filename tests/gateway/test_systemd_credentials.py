from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Iterable

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import full_canary_discord_edge_bootstrap as readiness
from gateway import systemd_credentials as credentials
from gateway.canonical_writer_boundary import (
    DEFAULT_DISCORD_EDGE_UNIT,
    DEFAULT_GATEWAY_UNIT,
)
from gateway.discord_edge_bootstrap import load_service_config
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.discord_rest_edge import (
    DiscordRestEdgeAdapter,
    DiscordRestEdgeError,
)
from gateway.mac_ops_edge_service import (
    DEFAULT_PROJECT_ID,
    MacOpsEdgeConfig,
    UrllibGitLabApi,
)


SERVICE_UID = 42_424
TOKEN = b"test.systemd.discord.token.123456789\n"
GITLAB_ENV = b"GITLAB_BASE_URL=https://gitlab.example\nGITLAB_TOKEN=secret-value\n"


def test_gateway_api_credentials_have_exact_production_bindings() -> None:
    directory = credentials.SYSTEMD_CREDENTIAL_ROOT / credentials.GATEWAY_API_UNIT
    assert directory == Path("/run/credentials/hermes-cloud-gateway.service")
    assert directory / credentials.GATEWAY_API_BEARER_CREDENTIAL == Path(
        "/run/credentials/hermes-cloud-gateway.service/api-server.key"
    )
    assert directory / credentials.GATEWAY_API_APPROVAL_CREDENTIAL == Path(
        "/run/credentials/hermes-cloud-gateway.service/api-approval-passkey"
    )


def _stat_with(
    item: os.stat_result,
    *,
    uid: int,
    gid: int,
    permission: int | None = None,
    inode_delta: int = 0,
) -> os.stat_result:
    values = list(item)
    if permission is not None:
        values[0] = stat.S_IFMT(item.st_mode) | permission
    values[1] = item.st_ino + inode_delta
    values[4] = uid
    values[5] = gid
    return os.stat_result(values)


def _credential_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unit: str,
    values: dict[str, bytes],
) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "run" / "credentials"
    directory = root / unit
    directory.mkdir(parents=True, mode=0o700)
    paths: dict[str, Path] = {}
    for name, value in values.items():
        path = directory / name
        path.write_bytes(value)
        path.chmod(0o400)
        paths[name] = path
    directory.chmod(0o500)
    monkeypatch.setattr(credentials, "SYSTEMD_CREDENTIAL_ROOT", root)
    return directory, paths


def _patch_provenance(
    monkeypatch: pytest.MonkeyPatch,
    *,
    directory: Path,
    files: Iterable[Path],
    owner_uid: int,
    directory_owner_uid: int | None = None,
    file_owner_uid: int | None = None,
    directory_gid: int = 0,
    file_gid: int = 0,
    directory_mode: int = 0o500,
    file_mode: int = 0o400,
) -> None:
    file_set = set(files)
    observed_directory_uid = (
        owner_uid if directory_owner_uid is None else directory_owner_uid
    )
    observed_file_uid = owner_uid if file_owner_uid is None else file_owner_uid
    real_lstat = credentials._lstat
    real_fstat = credentials._fstat

    def fake_lstat(path: str | os.PathLike[str]) -> os.stat_result:
        candidate = Path(path)
        item = real_lstat(path)
        if candidate == directory:
            return _stat_with(
                item,
                uid=observed_directory_uid,
                gid=directory_gid,
                permission=directory_mode,
            )
        if candidate in file_set:
            return _stat_with(
                item,
                uid=observed_file_uid,
                gid=file_gid,
                permission=file_mode,
            )
        return item

    def fake_fstat(descriptor: int) -> os.stat_result:
        return _stat_with(
            real_fstat(descriptor),
            uid=observed_file_uid,
            gid=file_gid,
            permission=file_mode,
        )

    monkeypatch.setattr(credentials, "_lstat", fake_lstat)
    monkeypatch.setattr(credentials, "_fstat", fake_fstat)


@pytest.mark.parametrize("owner_uid", [SERVICE_UID, 0])
def test_real_read_accepts_service_or_root_owned_systemd_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_uid: int,
) -> None:
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.DISCORD_EDGE_UNIT,
        values={credentials.DISCORD_TOKEN_CREDENTIAL: TOKEN},
    )
    token = paths[credentials.DISCORD_TOKEN_CREDENTIAL]
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=[token],
        owner_uid=owner_uid,
    )

    raw = credentials.read_systemd_credential(
        token,
        unit=credentials.DISCORD_EDGE_UNIT,
        name=credentials.DISCORD_TOKEN_CREDENTIAL,
        service_uid=SERVICE_UID,
        maximum=512,
        credentials_directory=directory,
    )
    provenance = credentials.systemd_credential_provenance(
        token,
        unit=credentials.DISCORD_EDGE_UNIT,
        name=credentials.DISCORD_TOKEN_CREDENTIAL,
        service_uid=SERVICE_UID,
        maximum=512,
        credentials_directory=directory,
    )

    assert raw == TOKEN
    assert provenance == {
        "path": str(token),
        "device": os.lstat(token).st_dev,
        "inode": os.lstat(token).st_ino,
        "uid": owner_uid,
        "gid": 0,
        "mode": "0400",
        "size": len(TOKEN),
    }


@pytest.mark.parametrize(
    (
        "directory_owner_uid",
        "file_owner_uid",
        "directory_gid",
        "file_gid",
        "directory_mode",
        "file_mode",
        "code",
    ),
    [
        (SERVICE_UID + 9, 0, 0, 0, 0o500, 0o400, "directory_provenance"),
        (0, SERVICE_UID + 9, 0, 0, 0o500, 0o400, "file_provenance"),
        (0, 0, 1, 0, 0o500, 0o400, "directory_provenance"),
        (0, 0, 0, 1, 0o500, 0o400, "file_provenance"),
        (0, 0, 0, 0, 0o700, 0o400, "directory_provenance"),
        (0, 0, 0, 0, 0o500, 0o600, "file_provenance"),
    ],
)
def test_rejects_wrong_gid_or_exact_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_owner_uid: int,
    file_owner_uid: int,
    directory_gid: int,
    file_gid: int,
    directory_mode: int,
    file_mode: int,
    code: str,
) -> None:
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.MAC_OPS_UNIT,
        values={credentials.MAC_OPS_GITLAB_CREDENTIAL: GITLAB_ENV},
    )
    path = paths[credentials.MAC_OPS_GITLAB_CREDENTIAL]
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=[path],
        owner_uid=0,
        directory_owner_uid=directory_owner_uid,
        file_owner_uid=file_owner_uid,
        directory_gid=directory_gid,
        file_gid=file_gid,
        directory_mode=directory_mode,
        file_mode=file_mode,
    )

    with pytest.raises(credentials.SystemdCredentialError, match=code):
        credentials.read_systemd_credential(
            path,
            unit=credentials.MAC_OPS_UNIT,
            name=credentials.MAC_OPS_GITLAB_CREDENTIAL,
            service_uid=SERVICE_UID,
            maximum=32 * 1024,
        )


def test_rejects_wrong_unit_name_path_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run" / "credentials"
    root.mkdir(parents=True)
    monkeypatch.setattr(credentials, "SYSTEMD_CREDENTIAL_ROOT", root)
    wrong_unit = root / "other.service" / credentials.DISCORD_TOKEN_CREDENTIAL
    wrong_name = root / credentials.DISCORD_EDGE_UNIT / "other-token"
    for candidate in (wrong_unit, wrong_name):
        with pytest.raises(
            credentials.SystemdCredentialError, match="binding_invalid"
        ):
            credentials.is_expected_systemd_credential(
                candidate,
                unit=credentials.DISCORD_EDGE_UNIT,
                name=credentials.DISCORD_TOKEN_CREDENTIAL,
            )

    actual = root / "actual"
    actual.mkdir(mode=0o700)
    (actual / credentials.DISCORD_TOKEN_CREDENTIAL).write_bytes(TOKEN)
    (actual / credentials.DISCORD_TOKEN_CREDENTIAL).chmod(0o400)
    actual.chmod(0o500)
    expected_directory = root / credentials.DISCORD_EDGE_UNIT
    expected_directory.symlink_to(actual, target_is_directory=True)
    expected_path = expected_directory / credentials.DISCORD_TOKEN_CREDENTIAL
    with pytest.raises(
        credentials.SystemdCredentialError, match="directory_provenance"
    ):
        credentials.read_systemd_credential(
            expected_path,
            unit=credentials.DISCORD_EDGE_UNIT,
            name=credentials.DISCORD_TOKEN_CREDENTIAL,
            service_uid=SERVICE_UID,
            maximum=512,
        )


def test_rejects_file_symlink_and_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run" / "credentials"
    directory = root / credentials.DISCORD_EDGE_UNIT
    directory.mkdir(parents=True, mode=0o700)
    target = directory / "target"
    target.write_bytes(TOKEN)
    target.chmod(0o400)
    expected = directory / credentials.DISCORD_TOKEN_CREDENTIAL
    expected.symlink_to(target)
    directory.chmod(0o500)
    monkeypatch.setattr(credentials, "SYSTEMD_CREDENTIAL_ROOT", root)
    with pytest.raises(
        credentials.SystemdCredentialError, match="file_provenance"
    ):
        credentials.read_systemd_credential(
            expected,
            unit=credentials.DISCORD_EDGE_UNIT,
            name=credentials.DISCORD_TOKEN_CREDENTIAL,
            service_uid=SERVICE_UID,
            maximum=512,
        )

    directory.chmod(0o700)
    expected.unlink()
    expected.write_bytes(TOKEN)
    expected.chmod(0o400)
    os.link(expected, directory / "second-link")
    directory.chmod(0o500)
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=[expected],
        owner_uid=0,
    )
    with pytest.raises(
        credentials.SystemdCredentialError, match="file_provenance"
    ):
        credentials.read_systemd_credential(
            expected,
            unit=credentials.DISCORD_EDGE_UNIT,
            name=credentials.DISCORD_TOKEN_CREDENTIAL,
            service_uid=SERVICE_UID,
            maximum=512,
        )


def test_rejects_file_identity_change_during_real_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.DISCORD_EDGE_UNIT,
        values={credentials.DISCORD_TOKEN_CREDENTIAL: TOKEN},
    )
    path = paths[credentials.DISCORD_TOKEN_CREDENTIAL]
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=[path],
        owner_uid=0,
    )
    prior_fstat = credentials._fstat
    calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        item = prior_fstat(descriptor)
        if calls == 2:
            return _stat_with(item, uid=0, gid=0, inode_delta=1)
        return item

    monkeypatch.setattr(credentials, "_fstat", changed_fstat)
    with pytest.raises(credentials.SystemdCredentialError, match="changed"):
        credentials.read_systemd_credential(
            path,
            unit=credentials.DISCORD_EDGE_UNIT,
            name=credentials.DISCORD_TOKEN_CREDENTIAL,
            service_uid=SERVICE_UID,
            maximum=512,
        )


def test_discord_token_adapter_accepts_root_owned_acl_like_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.DISCORD_EDGE_UNIT,
        values={credentials.DISCORD_TOKEN_CREDENTIAL: TOKEN},
    )
    path = paths[credentials.DISCORD_TOKEN_CREDENTIAL]
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=[path],
        owner_uid=0,
    )

    adapter = DiscordRestEdgeAdapter.from_credential_file(
        path,
        credentials_directory=directory,
        expected_owner_uid=SERVICE_UID,
    )
    adapter.close()


def _write_edge_config(
    tmp_path: Path,
    *,
    credentials_directory: Path,
    private_key_path: Path,
) -> tuple[Path, Ed25519PrivateKey]:
    writer_private = Ed25519PrivateKey.generate()
    edge_private = Ed25519PrivateKey.generate()
    writer_public_path = tmp_path / "writer-public.pem"
    writer_public_path.write_bytes(
        writer_private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    os.chown(writer_public_path, -1, os.getgid())
    writer_public_path.chmod(0o440)
    socket_directory = tmp_path / "socket"
    journal_directory = tmp_path / "journal"
    socket_directory.mkdir()
    journal_directory.mkdir()
    value = {
        "service": {
            "socket_path": str(socket_directory / "edge.sock"),
            "gateway_unit": DEFAULT_GATEWAY_UNIT,
            "edge_unit": DEFAULT_DISCORD_EDGE_UNIT,
            "gateway_uid": SERVICE_UID + 1,
            "edge_uid": SERVICE_UID,
            "edge_gid": os.getgid(),
            "connection_timeout_seconds": 10,
            "max_connections": 4,
        },
        "keys": {
            "writer_capability_public_key_file": str(writer_public_path),
            "writer_capability_public_key_id": ed25519_public_key_id(
                writer_private.public_key()
            ),
            "edge_receipt_private_key_file": str(private_key_path),
            "edge_receipt_public_key_id": ed25519_public_key_id(
                edge_private.public_key()
            ),
        },
        "discord": {
            "token_file": str(
                credentials_directory / credentials.DISCORD_TOKEN_CREDENTIAL
            ),
            "credentials_directory": str(credentials_directory),
            "api_timeout_seconds": 5,
        },
        "journal": {
            "path": str(journal_directory / "edge.sqlite3"),
            "busy_timeout_ms": 5_000,
        },
        "runtime": {"max_proof_age_ms": 5_000},
    }
    config_path = tmp_path / "discord-edge.json"
    config_path.write_text(json.dumps(value), encoding="utf-8")
    os.chown(config_path, -1, os.getgid())
    config_path.chmod(0o440)
    return config_path, edge_private


def test_discord_private_key_loader_accepts_root_owned_systemd_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge_private = Ed25519PrivateKey.generate()
    private_bytes = edge_private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.DISCORD_EDGE_UNIT,
        values={
            credentials.DISCORD_TOKEN_CREDENTIAL: TOKEN,
            credentials.DISCORD_PRIVATE_KEY_CREDENTIAL: private_bytes,
        },
    )
    private_path = paths[credentials.DISCORD_PRIVATE_KEY_CREDENTIAL]
    config_path, _unused_key = _write_edge_config(
        tmp_path,
        credentials_directory=directory,
        private_key_path=private_path,
    )
    value = json.loads(config_path.read_text(encoding="utf-8"))
    value["keys"]["edge_receipt_public_key_id"] = ed25519_public_key_id(
        edge_private.public_key()
    )
    config_path.chmod(0o600)
    config_path.write_text(json.dumps(value), encoding="utf-8")
    config_path.chmod(0o440)
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=paths.values(),
        owner_uid=0,
    )

    loaded = load_service_config(
        config_path,
        _expected_owner_uid=os.getuid(),
        _require_root_owned_parents=False,
        _expected_socket_path=None,
        _expected_journal_path=None,
    )

    assert ed25519_public_key_id(loaded.edge_private_key.public_key()) == (
        value["keys"]["edge_receipt_public_key_id"]
    )


def test_mac_gitlab_loader_accepts_root_owned_acl_like_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.MAC_OPS_UNIT,
        values={credentials.MAC_OPS_GITLAB_CREDENTIAL: GITLAB_ENV},
    )
    path = paths[credentials.MAC_OPS_GITLAB_CREDENTIAL]
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=[path],
        owner_uid=0,
    )
    config = MacOpsEdgeConfig(
        socket_path=tmp_path / "edge.sock",
        gateway_uid=SERVICE_UID + 1,
        socket_gid=SERVICE_UID + 2,
        service_identity_sha256="a" * 64,
        max_connections=4,
        gitlab_env_file=path,
        gitlab_project_id=DEFAULT_PROJECT_ID,
        gitlab_timeout_seconds=5,
        journal_path=tmp_path / "journal.sqlite3",
        journal_busy_timeout_ms=1_000,
    )

    api = UrllibGitLabApi.from_config(config, service_uid=SERVICE_UID)

    assert api._base_url == "https://gitlab.example"
    assert repr(api._token) == "<GitLab credential: redacted>"


def test_readiness_provenance_accepts_root_owned_systemd_files_without_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, paths = _credential_files(
        tmp_path,
        monkeypatch,
        unit=credentials.DISCORD_EDGE_UNIT,
        values={
            credentials.DISCORD_TOKEN_CREDENTIAL: TOKEN,
            credentials.DISCORD_PRIVATE_KEY_CREDENTIAL: b"private-key-bytes",
        },
    )
    _patch_provenance(
        monkeypatch,
        directory=directory,
        files=paths.values(),
        owner_uid=0,
    )

    for name, path in paths.items():
        observed = readiness._file_provenance(
            path,
            expected_uid=SERVICE_UID,
            expected_mode=0o400,
            systemd_credential_name=name,
            maximum=512,
        )
        assert observed["uid"] == 0
        assert observed["gid"] == 0
        assert observed["mode"] == "0400"
        assert "sha256" not in observed


def test_legacy_fixed_files_keep_their_original_strict_checks(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "legacy"
    directory.mkdir(mode=0o700)
    token = directory / "token"
    token.write_bytes(TOKEN)
    token.chmod(0o400)
    with pytest.raises(DiscordRestEdgeError, match="owner"):
        DiscordRestEdgeAdapter.from_credential_file(
            token,
            credentials_directory=directory,
            expected_owner_uid=os.getuid() + 1,
        )

    gitlab = directory / "gitlab.env"
    gitlab.write_bytes(GITLAB_ENV)
    gitlab.chmod(0o640)
    config = MacOpsEdgeConfig(
        socket_path=tmp_path / "edge.sock",
        gateway_uid=os.getuid() + 1,
        socket_gid=os.getgid(),
        service_identity_sha256="a" * 64,
        max_connections=4,
        gitlab_env_file=gitlab,
        gitlab_project_id=DEFAULT_PROJECT_ID,
        gitlab_timeout_seconds=5,
        journal_path=tmp_path / "journal.sqlite3",
        journal_busy_timeout_ms=1_000,
    )
    with pytest.raises(ValueError, match="protected_file_identity_invalid"):
        UrllibGitLabApi.from_config(config, service_uid=os.getuid())
