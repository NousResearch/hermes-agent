"""Secret stores Hermes writes vs the surfaces that hand files out.

Each store's path comes from the code that writes it, so a store that moves or is added
without reaching the shared list fails here instead of shipping readable on one surface.
"""

from __future__ import annotations

from pathlib import Path


def _written_secret_stores(home: Path) -> list[Path]:
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
    # The same stores in a sibling profile, and the two stores whose location is configured.
    return [*stores, *(home / "profiles" / "beta" / p.relative_to(home) for p in stores),
            whatsapp._session_path / "creds.json", _recovery_key_output_path()]


def test_every_written_secret_store_is_refused_by_file_tools_delivery_and_dashboard(monkeypatch):
    from agent.file_safety import get_read_block_error, is_write_denied
    from gateway.platforms.base import validate_media_delivery_path
    from hermes_cli.web_routers.files import _is_sensitive_path
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "config.yaml").write_text(f"platforms:\n  whatsapp:\n    session_path: {home / 'accounts' / 'work'}\n")
    monkeypatch.setenv("MATRIX_RECOVERY_KEY_OUTPUT_FILE", str(home / "recovery.txt"))

    leaks = []
    for path in _written_secret_stores(home):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"k": "FAKE-secret-value"}')  # fresh mtime: recency trust must not apply
        surfaces = {
            "read_file": get_read_block_error(str(path)) is not None,
            "write_file": is_write_denied(str(path)),
            "chat delivery": validate_media_delivery_path(str(path)) is None,
            "dashboard files": _is_sensitive_path(path.resolve()),
        }
        leaks += [f"{path} via {name}" for name, blocked in surfaces.items() if not blocked]
    assert leaks == []
    # Ordinary files next to those stores stay usable.
    for ordinary in (home / "accounts" / "notes.txt", home / "profiles" / "beta" / "notes.md"):
        ordinary.write_text("ordinary")
        assert get_read_block_error(str(ordinary)) is None and not is_write_denied(str(ordinary))


def test_a_credential_store_inside_an_allowlisted_root_is_not_delivered(monkeypatch):
    from gateway.platforms.base import validate_media_delivery_path
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    # An operator allow root that contains the stores (here: the whole Hermes home).
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(home))
    (home / ".env").write_text("FAKE_KEY=FAKE-secret-value")
    session = home / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True)
    (session / "creds.json").write_text('{"token": "FAKE-secret-value"}')
    artifact = home / "exports" / "chart.png"
    artifact.parent.mkdir()
    artifact.write_bytes(b"\x89PNG")
    kept = {}
    for strict in ("0", "1"):
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", strict)
        kept[strict] = [validate_media_delivery_path(str(p)) for p in (home / ".env", session / "creds.json", artifact)]
    assert kept == {s: [None, None, str(artifact.resolve())] for s in ("0", "1")}
