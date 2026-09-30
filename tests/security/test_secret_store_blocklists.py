"""Secret stores Hermes writes vs the surfaces that hand files out.

Each store's path comes from the code that writes it, so a store that moves or is added
without reaching the shared list fails here instead of shipping readable on one surface.
"""

from __future__ import annotations

from pathlib import Path


def _written_secret_stores() -> list[Path]:
    from agent.secret_sources import bitwarden, onepassword
    from agent.vault_store import VaultStore
    from gateway.pairing import _default_pairing_dir
    from hermes_cli.web_server_messaging import _whatsapp_session_path

    vault = VaultStore()
    return [
        onepassword._STORE.disk.path(),
        bitwarden._DISK_CACHE.path(),
        bitwarden._encrypted_disk_cache_path(),
        vault._key_path,
        vault._vault_path,
        _default_pairing_dir() / "telegram-pending.json",
        _whatsapp_session_path() / "creds.json",
    ]


def test_every_written_secret_store_is_refused_by_file_tools_delivery_and_dashboard():
    from agent.file_safety import get_read_block_error, is_write_denied
    from gateway.platforms.base import _path_under_denied_prefix
    from hermes_cli.web_routers.files import _is_sensitive_path

    leaks = []
    for path in _written_secret_stores():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"k": "FAKE-secret-value"}')  # fresh mtime: recency trust must not apply
        resolved = path.resolve()
        surfaces = {
            "read_file": get_read_block_error(str(path)) is not None,
            "write_file": is_write_denied(str(path)),
            "chat delivery": _path_under_denied_prefix(resolved),
            "dashboard files": _is_sensitive_path(resolved),
        }
        leaks += [f"{path.name} via {name}" for name, blocked in surfaces.items() if not blocked]
    assert leaks == []
