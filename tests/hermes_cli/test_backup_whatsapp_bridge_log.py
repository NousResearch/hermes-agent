"""Backups leave the WhatsApp bridge log out (#149860).

bridge.log is troubleshooting output that records chat ids and the Signal library's console
output for the linked account; a restore never needs it, so full and pre-update backups skip it.
"""
from pathlib import Path


def _selected(root: Path) -> set[str]:
    from hermes_cli.backup import _iter_backup_files

    return {rel.as_posix() for _, rel in _iter_backup_files(root, root.parent / "out.zip")}


def _touch(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_backup_skips_whatsapp_bridge_logs_but_keeps_the_session(tmp_path):
    root = tmp_path / "home"
    excluded = [
        "platforms/whatsapp/bridge.log", "platforms/whatsapp/bridge.log.1", "whatsapp/bridge.log",
        "profiles/work/platforms/whatsapp/bridge.log", "profiles/work/whatsapp/bridge.log.1",
    ]
    kept = [
        "platforms/whatsapp/session/creds.json",
        "profiles/work/platforms/whatsapp/session/creds.json",
        # Same name elsewhere is user data, not the bridge log.
        "skills/mine/whatsapp/bridge.log",
        "bridge.log",
    ]
    for rel in excluded + kept:
        _touch(root, rel)

    selected = _selected(root)

    assert not selected & set(excluded)
    assert set(kept) <= selected
