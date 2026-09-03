from __future__ import annotations

import multiprocessing
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from hermes_cli import profiles
from hermes_cli import bot_transfer
from hermes_cli.bot_transfer import (
    BOT_ID_FILENAME,
    export_bot_profile,
    get_profile_bot_id,
    import_bot_profile,
    pull_bot_profile,
    profile_is_cloneable,
    push_bot_profile,
    set_profile_cloneable,
)


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile_root = root / "profiles"
    profile_root.mkdir(parents=True)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profile_root)
    return root


def _source_profile(root: Path, name: str = "helper") -> Path:
    source = root / "profiles" / name
    (source / "skills" / "demo").mkdir(parents=True)
    (source / "memories").mkdir()
    (source / "sessions").mkdir()
    (source / "SOUL.md").write_text("Helpful bot\n", encoding="utf-8")
    (source / "config.yaml").write_text(
        "model: test\ngateway:\n  bot_sharing:\n    allow_push: true\n",
        encoding="utf-8",
    )
    (source / "skills" / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (source / "memories" / "MEMORY.md").write_text("private memory\n", encoding="utf-8")
    (source / "sessions" / "chat.json").write_text("private chat\n", encoding="utf-8")
    (source / ".env").write_text("OPENAI_API_KEY=private\n", encoding="utf-8")
    (source / "auth.json").write_text('{"token":"private"}\n', encoding="utf-8")
    return source


def _hold_clone_lock(profiles_root: str, ready) -> None:
    profiles._get_profiles_root = lambda: Path(profiles_root)
    with bot_transfer._clone_import_lock():
        ready.set()
        time.sleep(30)


def test_bot_export_is_definition_only_and_identity_is_stable(profile_root, tmp_path):
    source = _source_profile(profile_root)

    first, first_id = export_bot_profile("helper", str(tmp_path / "first.tar.gz"))
    second, second_id = export_bot_profile("helper", str(tmp_path / "second.tar.gz"))

    assert first_id == second_id == get_profile_bot_id(source)
    with tarfile.open(first, "r:gz") as archive:
        names = set(archive.getnames())
    assert "helper/SOUL.md" in names
    assert "helper/skills/demo/SKILL.md" in names
    assert f"helper/{BOT_ID_FILENAME}" in names
    assert not any("memories" in name for name in names)
    assert not any("sessions" in name for name in names)
    assert not any(name.endswith(("/.env", "/auth.json")) for name in names)
    with tarfile.open(first, "r:gz") as archive:
        config = archive.extractfile("helper/config.yaml").read().decode("utf-8")
        profile_meta = archive.extractfile("helper/profile.yaml").read().decode("utf-8")
    assert "bot_sharing" not in config
    assert "cloneable: false" in profile_meta


def test_bot_export_excludes_nested_credentials_and_cron_runtime(profile_root, tmp_path):
    source = _source_profile(profile_root)
    (source / "skills" / "demo" / ".env").write_text(
        "TOKEN=private\n", encoding="utf-8"
    )
    (source / "plugins" / "demo").mkdir(parents=True)
    (source / "plugins" / "demo" / "credentials.bin").write_bytes(b"\x00private")
    (source / "cron" / "output").mkdir(parents=True)
    (source / "cron" / "jobs.json").write_text("{}\n", encoding="utf-8")
    (source / "cron" / "output" / "last.txt").write_text(
        "private\n", encoding="utf-8"
    )
    (source / "cron" / "ticker-heartbeat").write_text("runtime\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported binary file"):
        export_bot_profile("helper", str(tmp_path / "binary.tar.gz"))

    (source / "plugins" / "demo" / "credentials.bin").unlink()
    archive, _ = export_bot_profile("helper", str(tmp_path / "safe.tar.gz"))
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "helper/cron/jobs.json" in names
    assert not any(name.endswith("/.env") for name in names)
    assert not any("cron/output" in name for name in names)
    assert "helper/cron/ticker-heartbeat" not in names


def test_pull_policy_is_per_profile_and_fails_closed_for_malformed_yaml(profile_root):
    source = _source_profile(profile_root)

    assert profile_is_cloneable("helper") is False
    bot_id = set_profile_cloneable("helper", True)
    assert profile_is_cloneable("helper") is True
    assert bot_id == get_profile_bot_id(source)

    (source / "profile.yaml").write_text("cloneable: 'false'\n", encoding="utf-8")
    assert profile_is_cloneable("helper") is False


def test_clone_policy_rejects_profile_path_traversal(profile_root):
    outside = profile_root / "outside"
    outside.mkdir()

    assert profile_is_cloneable("../outside") is False
    with pytest.raises(ValueError, match="Invalid profile name"):
        set_profile_cloneable("../outside", True)

    assert not (outside / BOT_ID_FILENAME).exists()
    assert not (outside / "profile.yaml").exists()


def test_bot_identity_is_published_atomically_for_concurrent_exporters(
    profile_root,
):
    source = _source_profile(profile_root)

    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = set(pool.map(lambda _: bot_transfer.ensure_profile_bot_id(source), range(8)))

    assert len(identities) == 1
    assert get_profile_bot_id(source) == identities.pop()
    assert not list(source.glob(f".{BOT_ID_FILENAME}.*.tmp"))


def test_bot_identity_publish_failure_leaves_no_partial_file(
    profile_root, monkeypatch
):
    source = _source_profile(profile_root)
    monkeypatch.setattr(bot_transfer.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("fail")))

    with pytest.raises(OSError, match="fail"):
        bot_transfer.ensure_profile_bot_id(source)

    assert not (source / BOT_ID_FILENAME).exists()
    assert not list(source.glob(f".{BOT_ID_FILENAME}.*.tmp"))


def test_bot_import_allows_rename_but_rejects_name_and_identity_collisions(profile_root, tmp_path):
    source = _source_profile(profile_root)
    archive, bot_id = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    shutil.rmtree(source)

    imported, imported_id = import_bot_profile(str(archive), name="helper-copy")

    assert imported.name == "helper-copy"
    assert imported_id == bot_id == get_profile_bot_id(imported)
    assert (imported / "SOUL.md").read_text(encoding="utf-8") == "Helpful bot\n"
    assert profile_is_cloneable("helper-copy") is False

    with pytest.raises(FileExistsError, match="already exists as profile"):
        import_bot_profile(str(archive), name="another-name")

    other = _source_profile(profile_root, "occupied")
    (other / BOT_ID_FILENAME).unlink(missing_ok=True)
    other_archive, _ = export_bot_profile("occupied", str(tmp_path / "occupied.tar.gz"))
    shutil.rmtree(other)
    (profile_root / "profiles" / "occupied").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        import_bot_profile(str(other_archive), name="occupied")


def test_bot_import_rejects_user_state_even_from_safe_tar(profile_root, tmp_path):
    staged = tmp_path / "staged" / "unsafe"
    staged.mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text("a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8")
    (staged / "sessions").mkdir()
    (staged / "sessions" / "chat.json").write_text("private\n", encoding="utf-8")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="unsafe")

    with pytest.raises(ValueError, match="disallowed profile data: sessions"):
        import_bot_profile(str(archive))


def test_bot_import_resets_sender_sharing_policies(profile_root, tmp_path):
    staged = tmp_path / "staged" / "shared"
    staged.mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "profile.yaml").write_text("cloneable: true\n", encoding="utf-8")
    (staged / "config.yaml").write_text(
        "model: test\ngateway:\n  bot_sharing:\n    allow_push: true\n",
        encoding="utf-8",
    )
    archive = tmp_path / "shared.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="shared")

    imported, _ = import_bot_profile(str(archive))

    assert profile_is_cloneable("shared") is False
    assert "bot_sharing" not in (imported / "config.yaml").read_text(encoding="utf-8")


def test_bot_import_bounds_expanded_archive_and_leaves_no_profile(
    profile_root, tmp_path, monkeypatch
):
    staged = tmp_path / "staged" / "large"
    staged.mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "SOUL.md").write_bytes(b"0" * 2048)
    archive = tmp_path / "large.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="large")
    assert archive.stat().st_size < 1024
    monkeypatch.setattr(bot_transfer, "MAX_BOT_CLONE_BYTES", 1024)

    with pytest.raises(ValueError, match="expanded-size limit"):
        import_bot_profile(str(archive))

    assert not (profile_root / "profiles" / "large").exists()


def test_bot_export_rejects_source_outside_import_budget(
    profile_root, tmp_path, monkeypatch
):
    source = _source_profile(profile_root)
    (source / "SOUL.md").write_bytes(b"0" * 2048)
    monkeypatch.setattr(bot_transfer, "MAX_BOT_CLONE_BYTES", 1024)

    with pytest.raises(ValueError, match="expanded-size limit"):
        export_bot_profile("helper", str(tmp_path / "large.tar.gz"))

    assert not (tmp_path / "large.tar.gz").exists()


def test_bot_export_rejects_too_many_archive_members(
    profile_root, tmp_path, monkeypatch
):
    _source_profile(profile_root)
    monkeypatch.setattr(bot_transfer, "MAX_BOT_CLONE_MEMBERS", 3)

    with pytest.raises(ValueError, match="member limit"):
        export_bot_profile("helper", str(tmp_path / "many.tar.gz"))

    assert not (tmp_path / "many.tar.gz").exists()


def test_clone_import_lock_cannot_be_stolen_and_recovers_after_owner_death(
    profile_root,
):
    profiles_root = profile_root / "profiles"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    holder = context.Process(target=_hold_clone_lock, args=(str(profiles_root), ready))
    holder.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(TimeoutError):
            with bot_transfer._clone_import_lock(timeout=0.1):
                pass
    finally:
        holder.terminate()
        holder.join(timeout=10)
    assert not holder.is_alive()

    with bot_transfer._clone_import_lock(timeout=1):
        pass


def test_pull_downloads_and_imports_remote_clone(profile_root, tmp_path, monkeypatch):
    source = _source_profile(profile_root)
    archive, bot_id = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    payload = archive.read_bytes()
    shutil.rmtree(source)
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://remote.example/v1/bots/helper/clone"
        assert request.headers["Authorization"] == "Bearer remote-secret"
        return httpx.Response(200, content=payload, headers={"Content-Type": "application/gzip"})

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    imported, imported_id = pull_bot_profile(
        "helper", remote="https://remote.example", name="local-helper"
    )

    assert imported.name == "local-helper"
    assert imported_id == bot_id


def test_push_uploads_clone_with_requested_remote_name(profile_root, monkeypatch):
    _source_profile(profile_root)
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://remote.example/v1/bots/clone?name=remote-helper"
        assert request.headers["Authorization"] == "Bearer remote-secret"
        assert request.headers["Content-Type"] == "application/gzip"
        assert 0 < len(request.content) <= bot_transfer.MAX_BOT_CLONE_BYTES
        return httpx.Response(
            201,
            json={
                "object": "hermes.bot_clone",
                "name": "remote-helper",
                "bot_id": "a8c214f7-37ee-4f50-95a4-939a51631283",
            },
        )

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    name, bot_id = push_bot_profile(
        "helper", remote="https://remote.example/v1", name="remote-helper"
    )

    assert name == "remote-helper"
    assert bot_id == "a8c214f7-37ee-4f50-95a4-939a51631283"


def test_remote_clone_refuses_cleartext_bearer_auth_off_loopback(monkeypatch):
    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")

    with pytest.raises(ValueError, match="must use HTTPS"):
        pull_bot_profile("helper", remote="http://gateway.example")


def test_pull_wraps_network_failures(profile_root, monkeypatch):
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(bot_transfer.BotTransferError, match="request failed"):
        pull_bot_profile("helper", remote="https://remote.example")


def test_push_wraps_network_failures(profile_root, monkeypatch):
    _source_profile(profile_root)
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(bot_transfer.BotTransferError, match="request failed"):
        push_bot_profile("helper", remote="https://remote.example")


def test_push_rejects_malformed_success_response(profile_root, monkeypatch):
    _source_profile(profile_root)
    real_client = httpx.Client

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"object": "hermes.bot_clone"})

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(bot_transfer.BotTransferError, match="invalid bot clone response"):
        push_bot_profile("helper", remote="https://remote.example")
