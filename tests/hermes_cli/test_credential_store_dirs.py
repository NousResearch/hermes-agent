"""The vault, pairing, messaging-session stores and the 1Password ``.op.env`` bootstrap are refused by
every surface that hands out files: file-tool read and write, gateway chat delivery and the dashboard
Files API (#128812).

Each path comes from the code that writes or loads it, so a store that moves (as pairing did to
``platforms/pairing``) without reaching the shared list fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    return home


def _written_stores(home: Path) -> list[Path]:
    from agent.vault_store import VaultStore
    from gateway import pairing
    from hermes_cli.web_server_messaging import _whatsapp_session_path
    from plugins.platforms.matrix.adapter import MatrixAdapter

    assert pairing.PAIRING_DIR is None
    vault = VaultStore()
    matrix_store = MatrixAdapter.__new__(MatrixAdapter)._resolve_store_dir()
    return [
        vault._key_path, vault._vault_path,
        pairing._default_pairing_dir() / "telegram-approved.json",
        _whatsapp_session_path() / "creds.json",
        matrix_store / "crypto.db",
        home / ".op.env",
    ]


def test_op_env_is_the_file_the_loader_bootstraps_from(home, monkeypatch):
    from hermes_cli.env_loader import _bootstrap_env

    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    (home / ".op.env").write_text("OP_SERVICE_ACCOUNT_TOKEN=FAKE-op-token\n", encoding="utf-8")
    assert _bootstrap_env(home).get("OP_SERVICE_ACCOUNT_TOKEN") == "FAKE-op-token"


def test_every_surface_refuses_the_stores(home, monkeypatch):
    _assert_refused_everywhere(home, monkeypatch, _written_stores(home))


def _assert_refused_everywhere(home, monkeypatch, paths):
    import agent.file_safety as fs
    from gateway.platforms import base
    from hermes_cli.web_routers.files import _is_sensitive_path

    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: home)
    monkeypatch.setattr(base, "_HERMES_HOME", home)
    monkeypatch.setattr(base, "_HERMES_ROOT", home)
    leaks = []
    for path in paths:
        assert path.is_relative_to(home), path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FAKE-secret-value", encoding="utf-8")  # fresh mtime: recency trust must not apply
        refused = {
            "read": fs.get_read_block_error(str(path)) is not None,
            "write": fs.is_write_denied(str(path)),
            "chat delivery": base.validate_media_delivery_path(str(path)) is None,
            "dashboard files": _is_sensitive_path(path),
        }
        leaks += [f"{path.relative_to(home)} via {name}" for name, ok in refused.items() if not ok]
    assert leaks == []


@pytest.mark.platforms("macos", "windows")
def test_case_variants_of_the_stores_are_refused(home, monkeypatch):
    """Case-insensitive filesystems open ``PLATFORMS/PAIRING/`` and ``.OP.ENV`` for the real stores."""
    variants = [home / str(path.relative_to(home)).upper() for path in _written_stores(home)]
    _assert_refused_everywhere(home, monkeypatch, variants)


def test_ordinary_files_with_similar_names_stay_usable(home):
    from hermes_cli.web_routers.files import _is_sensitive_path

    for rel in ("notes/vault/todo.md", "projects/session/log.txt", "docs/store/readme.md"):
        assert not _is_sensitive_path(home / rel), rel


def test_terminal_write_to_op_env_needs_approval():
    from tools.approval_detection import detect_dangerous_command

    assert detect_dangerous_command("echo X > ~/.hermes/.op.env")[0] is True
