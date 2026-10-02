"""Secret stores Hermes writes vs the surfaces that hand files out.

Each store's path comes from the code that writes it, so a store that moves or is added
without reaching the shared list fails here instead of shipping readable on one surface.
A store is refused for what it is, not for how the path to it is spelled: a hardlink, a
store directory behind a symlink, or a case variant on a case-insensitive filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _written_secret_stores(home: Path) -> list[tuple[Path, Path]]:
    """``(store file, store root)``: the root is the file itself for an exact-file store, else
    the store directory holding it."""
    from agent.file_safety import SECRET_STORE_DIRS, SECRET_STORE_FILES
    from agent.secret_sources import bitwarden, onepassword
    from agent.vault_store import VaultStore
    from gateway.config import PlatformConfig
    from gateway.pairing import _default_pairing_dir
    from hermes_cli.config import load_config_readonly
    from gateway.platforms import weixin
    from plugins.platforms.google_chat import oauth as google_chat_oauth
    from plugins.platforms.matrix.adapter import _recovery_key_output_path
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    vault = VaultStore()
    whatsapp = WhatsAppAdapter(PlatformConfig.from_dict(load_config_readonly()["platforms"]["whatsapp"]))
    stores = [
        onepassword._STORE.disk.path(),
        bitwarden._DISK_CACHE.path(),
        bitwarden._encrypted_disk_cache_path(),
        vault._key_path,
        vault._vault_path,
        _default_pairing_dir() / "telegram-pending.json",
        home / "slack_tokens.json",
        google_chat_oauth._token_path("user@example.com"),
        google_chat_oauth._token_path(),
        google_chat_oauth._client_secret_path(),
        weixin._account_dir(str(home)) / "fake-account.json",
    ]
    # The same stores in a sibling profile.
    stores += [home / "profiles" / "beta" / p.relative_to(home) for p in stores]

    def root_of(path: Path) -> Path:
        base = home / "profiles" / "beta" if path.is_relative_to(home / "profiles") else home
        rel = path.relative_to(base)
        if str(rel) in SECRET_STORE_FILES:
            return path
        return base / next(d for d in SECRET_STORE_DIRS if rel.is_relative_to(d))

    # Plus the two stores whose location is configured.
    return [*((p, root_of(p)) for p in stores),
            (whatsapp._session_path / "creds.json", whatsapp._session_path),
            (_recovery_key_output_path(),) * 2]


def _reached_another_way(stores: list[tuple[Path, Path]], reach: str, tmp_path: Path) -> list[Path]:
    """The stores' paths as a consumer may be handed them: as written, through a hardlink (an
    exact-file store) or a symlinked store directory, or in swapped case."""
    if reach == "as written":
        return [path for path, _root in stores]
    if reach == "case variant":
        return [Path(str(path).swapcase()) for path, _root in stores]
    outside = tmp_path / "outside"
    outside.mkdir()
    paths = []
    for i, (path, root) in enumerate(stores):
        if root == path:
            paths.append(outside / f"notes-{i}.txt")
            os.link(path, paths[-1])
        else:
            # A store directory moved behind a symlink: the resolved path has no store name in it.
            external = tmp_path / f"external-{i}"
            root.rename(external)
            root.symlink_to(external, target_is_directory=True)
            paths.append(external / path.relative_to(root))
    return paths


@pytest.mark.parametrize("reach", [
    "as written",
    "hardlink or symlinked store directory",
    pytest.param("case variant", marks=pytest.mark.platforms("macos", "windows")),
])
def test_every_written_secret_store_is_refused_by_file_tools_delivery_and_dashboard(reach, monkeypatch, tmp_path):
    from agent.file_safety import get_read_block_error, is_write_denied
    from gateway.platforms.base import validate_media_delivery_path
    from hermes_cli.web_routers.files import _is_sensitive_path
    from hermes_constants import get_hermes_home

    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    home = get_hermes_home()
    monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", home)  # frozen at import
    (home / "config.yaml").write_text(f"platforms:\n  whatsapp:\n    session_path: {home / 'accounts' / 'work'}\n")
    monkeypatch.setenv("MATRIX_RECOVERY_KEY_OUTPUT_FILE", str(home / "recovery.txt"))

    stores = _written_secret_stores(home)
    for path, _root in stores:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"k": "FAKE-secret-value"}')  # fresh mtime: recency trust must not apply

    def refused_by(path: Path) -> dict[str, bool]:
        return {
            "read_file": get_read_block_error(str(path)) is not None,
            "write_file": is_write_denied(str(path)),
            "chat delivery": validate_media_delivery_path(str(path)) is None,
            "dashboard files": _is_sensitive_path(path.resolve()),
        }

    leaks = [f"{path} via {name}" for path in _reached_another_way(stores, reach, tmp_path)
             for name, refused in refused_by(path).items() if not refused]
    # Not created yet: on a case-insensitive volume a write to .ENV creates the .env the loader reads.
    (home / ".env").unlink(missing_ok=True)
    if reach == "case variant" and not is_write_denied(str(home / ".ENV")):
        leaks.append(".ENV via write_file")
    assert leaks == []
    # Ordinary files next to those stores, and outside any Hermes home, stay usable.
    for ordinary in (home / "accounts" / "notes.txt", home / "profiles" / "beta" / "notes.md"):
        ordinary.write_text("ordinary")
        assert get_read_block_error(str(ordinary)) is None and not is_write_denied(str(ordinary))
    plain = tmp_path / "report.txt"
    plain.write_text("ordinary")
    assert not any(refused_by(plain).values())


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["delivery filter", "simplex send"])
async def test_a_credential_store_inside_an_allowlisted_root_is_not_delivered(consumer, monkeypatch):
    """Credential denies beat an operator allow root, at the shared filter and at a direct-send
    consumer: SimpleX ``send()`` gets text whose refused ``MEDIA:`` tags were left in on purpose,
    so re-parsing it with a local regex would upload exactly what the filter refused."""
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult, validate_media_delivery_path
    from hermes_constants import get_hermes_home
    from plugins.platforms.simplex.adapter import SimplexAdapter

    home = get_hermes_home()
    # An operator allow root that contains the stores (here: the whole Hermes home).
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(home))
    (home / ".env").write_text("FAKE_KEY=FAKE-secret-value")
    session = home / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True)
    (session / "creds.json").write_text('{"token": "FAKE-secret-value"}')
    (home / "auth.json").write_text('{"token": "FAKE-secret-value"}')
    artifact = home / "exports" / "report.pdf"
    artifact.parent.mkdir()
    artifact.write_bytes(b"%PDF-1.4 ordinary")
    offered = (home / ".env", session / "creds.json", home / "auth.json", artifact)

    async def delivered() -> list[str]:
        if consumer == "delivery filter":
            return [kept for p in offered if (kept := validate_media_delivery_path(str(p)))]
        adapter = SimplexAdapter(PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"}))
        adapter._ws = AsyncMock()
        adapter.send_document = AsyncMock(return_value=SendResult(success=True))
        adapter.send_voice = AsyncMock(return_value=SendResult(success=True))
        # A fenced tag is an example, never an attachment, even for an ordinary file.
        text = "".join(f"MEDIA:{p}\n" for p in offered) + f"example:\n```\nMEDIA:{artifact}\n```\n"
        assert (await adapter.send("contact-42", text)).success is True
        calls = adapter.send_document.await_args_list + adapter.send_voice.await_args_list
        return [str(Path(call.args[1]).resolve()) for call in calls]

    kept = {}
    for strict in ("0", "1"):
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", strict)
        kept[strict] = await delivered()
    assert kept == {s: [str(artifact.resolve())] for s in ("0", "1")}
