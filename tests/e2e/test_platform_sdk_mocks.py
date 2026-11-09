"""Guards for the platform-SDK mocks installed by ``tests/e2e/conftest.py``.

The conftest fakes the ``telegram`` / ``discord`` / ``slack`` packages so the
adapters import without their SDKs installed. Faking is only correct when the
SDK really is absent: a real, installed SDK must never be swapped for a mock,
otherwise the e2e suite exercises a MagicMock while claiming to exercise the
real client interface (issue #147239).
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.e2e import conftest as e2e_conftest

# platform id -> module names the conftest fakes for it
FAKED_SDKS = {
    "telegram": ("telegram",),
    "discord": ("discord",),
    "slack": ("slack_bolt", "slack_sdk"),
}


def _ensure(platform_name: str) -> None:
    {
        "telegram": e2e_conftest._ensure_telegram_mock,
        "discord": e2e_conftest._ensure_discord_mock,
        "slack": e2e_conftest._ensure_slack_mock,
    }[platform_name]()


@pytest.fixture
def platform_modules():
    """Snapshot the faked platform modules and restore them after the test.

    Tests here delete entries from :data:`sys.modules` to simulate "not imported
    yet" / "not installed"; without the restore the rest of the session would run
    without the conftest mocks it depends on.
    """
    roots = {root for names in FAKED_SDKS.values() for root in names}  # module roots, not platform ids

    def _owned(name: str) -> bool:
        return name.split(".", 1)[0] in roots

    snapshot = {name: mod for name, mod in sys.modules.items() if _owned(name)}
    yield
    for name in [name for name in sys.modules if _owned(name)]:
        del sys.modules[name]
    sys.modules.update(snapshot)


def _drop(*names: str) -> None:
    for name in [name for name in sys.modules if name.split(".", 1)[0] in names]:
        del sys.modules[name]


def _write_fake_sdk(root: Path, name: str, body: str = "") -> None:
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text(
        body or f'"""Installed stand-in for the {name} SDK."""\n', encoding="utf-8"
    )


@pytest.mark.parametrize("platform_name", sorted(FAKED_SDKS))
def test_installed_sdk_is_not_replaced_by_the_mock(platform_name, tmp_path, monkeypatch, platform_modules):
    """An installed-but-not-yet-imported SDK must survive the conftest."""
    names = FAKED_SDKS[platform_name]
    for name in names:
        _write_fake_sdk(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop(*names)

    _ensure(platform_name)

    for name in names:
        assert name not in sys.modules, f"{name} is installed but conftest installed a fake"
        mod = importlib.import_module(name)
        assert Path(mod.__file__).resolve().is_relative_to(tmp_path)
        assert not isinstance(mod, MagicMock)


@pytest.mark.parametrize("platform_name", sorted(FAKED_SDKS))
def test_missing_sdk_still_gets_the_mock(platform_name, monkeypatch, platform_modules):
    """With no SDK on the path the fake is still installed — that is the legit case."""
    names = FAKED_SDKS[platform_name]
    _drop(*names)
    # Pretend nothing of this name is installed, whatever the test env has.
    monkeypatch.setattr(importlib.util, "find_spec", lambda *args, **kwargs: None)

    _ensure(platform_name)

    for name in names:
        assert isinstance(sys.modules[name], MagicMock), f"{name} was not mocked"
