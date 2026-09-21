"""P13 isolation + inventory proof for scripts/discord-bot-health.py.

Verifies:
- HERMES_HOME parameterisation: HERMES resolves under HERMES_HOME, not
  /home/kensei/.hermes.
- Service inventory includes kensei-review and quan (previously missed).
- discover_services falls back to the known inventory when systemd is
  unavailable, so every gateway bot is still checked.
- import-safe: importing the module does not touch the filesystem.
"""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "discord-bot-health.py"


def _load_module(monkeypatch, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "discord_bot_health_under_test", str(SCRIPT)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_hermes_resolves_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.HERMES).startswith(str(fake_home))


def test_inventory_includes_kensei_review_and_quan(monkeypatch, fake_home):
    """The known gateway inventory must include kensei-review and quan,
    which were previously missed by the glob-based discovery."""
    mod = _load_module(monkeypatch, fake_home)
    names = [n for n, _ in mod.KNOWN_GATEWAY_SERVICES]
    assert "kensei-review" in names, "kensei-review missing from inventory"
    assert "quan" in names, "quan missing from inventory"
    assert "kensei" in names


def test_discover_services_falls_back_to_inventory(monkeypatch, fake_home):
    """When systemd is unavailable, discover_services must return the
    full known inventory (sorted), so every bot is still checked."""
    mod = _load_module(monkeypatch, fake_home)

    def boom(*a, **k):
        raise FileNotFoundError("systemctl not found")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    services = mod.discover_services()
    names = [n for n, _ in services]
    # Every known bot present.
    for n, _ in mod.KNOWN_GATEWAY_SERVICES:
        assert n in names, f"{n} missing from fallback discovery"
    # Sorted.
    assert names == sorted(names)


def test_discover_services_merges_inventory_with_systemd(monkeypatch, fake_home):
    """When systemd reports some services, the inventory fills the gaps
    (e.g. kensei-review and quan are added even if not in systemctl output)."""
    mod = _load_module(monkeypatch, fake_home)

    class FakeResult:
        stdout = (
            "hermes-gateway.service   static\n"
            "hermes-gateway-octacon.service  static\n"
        )
        returncode = 0

    def fake_run(cmd, *a, **k):
        return FakeResult()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    services = mod.discover_services()
    names = [n for n, _ in services]
    # systemd-discovered bots present.
    assert "kensei" in names
    assert "octacon" in names
    # Inventory-filled gaps present (not in systemctl output).
    assert "kensei-review" in names, "inventory did not fill kensei-review gap"
    assert "quan" in names, "inventory did not fill quan gap"


def test_dry_import_no_filesystem_touch(monkeypatch, fake_home):
    """Importing must not create any file under the fake home."""
    _load_module(monkeypatch, fake_home)
    # The fake home should only contain the dir we made.
    assert list(fake_home.iterdir()) == []
