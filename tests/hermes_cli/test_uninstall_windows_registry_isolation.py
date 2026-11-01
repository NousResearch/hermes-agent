"""A Windows test run must never mutate the operator's real ``HKCU\\Environment``.

Test isolation redirects ``HERMES_HOME`` to a per-session sandbox, but the uninstaller's two
``HKCU\\Environment`` mutators take the home as an argument and derive their match patterns
from it:

- ``remove_path_from_windows_registry`` **deletes** every User-PATH entry that starts with
  ``<home>\\{hermes-agent,git,node,venv}`` (plus ``<home>\\bin`` when ``include_managed_bin``);
- ``remove_hermes_env_vars_windows`` **deletes** the ``HERMES_HOME`` / ``HERMES_GIT_BASH_PATH``
  values.

Both are reached only through ``_perform_uninstall``'s step list, behind ``_is_windows()`` --
so what keeps a test run off the real registry is the per-test
``monkeypatch.setattr(uninstall, "_is_windows", lambda: False)`` in
``test_uninstall_gui_userdata.py``, not the write point itself. Move that seam (the module
starts reading ``sys.platform`` directly, or a new test drives ``_perform_uninstall``) and the
operator's real PATH loses ``%LOCALAPPDATA%\\hermes\\bin``: measured 2026-10-10 (card
``t_0cf0aa7e``, probe ``evals/background_review/uninstall_path_stripper_probe.py``), a real
root with ``include_managed_bin=True`` removes that entry, and a real root alone removes
``...\\hermes\\git\\cmd`` / ``...\\hermes\\node`` on a pre-``bin`` install.ps1 layout.

Unlike the two *prependers* (``test_windows_user_path_isolation.py``), the red half here cannot
be "drive the real writer and watch the stored value change": a red run would DELETE a real
entry and break the operator's ``hermes`` command, and a deleted entry cannot be
reverse-engineered from what is left. So the contract is asserted one step earlier and is just
as behavioural -- under isolation the registry key must not be opened at all. Run either test
against a tree without its guard and it fails with ``HKCU\\Environment opened during a test
run``, before any write can happen (red half: ``evals/background_review/uninstall_guard_red_half.py``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")


def _real_default_home() -> Path:
    """The machine's real default ``HERMES_HOME`` -- where the match patterns come from.

    Deliberately the REAL root rather than ``get_hermes_home()``: the case under test is a
    caller that hands over the real home, and under pytest ``get_hermes_home()`` is already
    the sandbox. Mirrors ``hermes_constants._get_platform_default_hermes_home``'s Windows rule.
    """
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return base / "hermes"


@pytest.fixture
def sandbox_home(tmp_path, monkeypatch):
    """The isolation pair pytest installs: a redirected home AND the marker that names it."""
    home = tmp_path / "hermes_test"
    home.mkdir(exist_ok=True)  # the hermetic conftest already creates this sandbox home
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(home))
    return home


@pytest.fixture
def refuse_registry(monkeypatch):
    """Make any open of ``HKCU\\Environment`` fail the test instead of reaching the value.

    ``_edit_user_environment`` imports ``winreg`` at call time, so patching the module
    attribute intercepts it. Raising -- rather than recording and calling through -- is what
    keeps a red run harmless: an unguarded tree fails before anything is read or written.
    """
    import winreg

    def refuse(*args, **kwargs):
        raise AssertionError(
            "HKCU\\Environment opened during a test run -- the uninstaller's "
            "HERMES_TEST_ISOLATION guard is missing (see this module's docstring)"
        )

    monkeypatch.setattr(winreg, "OpenKey", refuse)


def test_strip_user_path_does_not_touch_the_registry_under_test_isolation(
    sandbox_home, refuse_registry
):
    """The PATH deleter must be inert -- even for the most dangerous root/flags pair.

    ``include_managed_bin=True`` is the full-uninstall-of-the-default-home configuration
    (``_perform_uninstall`` sets it when ``full_uninstall and _is_default_hermes_home(...)``),
    i.e. exactly the case that removes the real ``%LOCALAPPDATA%\\hermes\\bin``.
    """
    from hermes_cli import uninstall

    removed = uninstall.remove_path_from_windows_registry(
        _real_default_home(), include_managed_bin=True
    )

    assert removed == [], "an inert strip must report nothing removed"


def test_remove_user_env_vars_does_not_touch_the_registry_under_test_isolation(
    sandbox_home, refuse_registry
):
    """The sibling mutator in the same ``_perform_uninstall`` step list, same contract."""
    from hermes_cli import uninstall

    removed = uninstall.remove_hermes_env_vars_windows()

    assert removed == [], "an inert delete must report nothing removed"


class _FakeKey:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_winreg(store: dict[str, str]):
    """In-memory stand-in for the ``winreg`` module holding one ``Path`` value.

    ``_edit_user_environment`` imports ``winreg`` at call time, so patching the
    ``sys.modules`` entry intercepts the real code path without opening the key.
    """
    import types
    import winreg as real

    mod = types.ModuleType("winreg")
    for name, value in (
        ("HKEY_CURRENT_USER", real.HKEY_CURRENT_USER),
        ("Environment", "Environment"),
        ("KEY_READ", real.KEY_READ),
        ("KEY_WRITE", real.KEY_WRITE),
        ("REG_SZ", real.REG_SZ),
        ("REG_EXPAND_SZ", real.REG_EXPAND_SZ),
        ("OpenKey", lambda *a, **k: _FakeKey()),
        ("QueryValueEx", lambda key, name: (store[name], real.REG_EXPAND_SZ)),
        ("SetValueEx", lambda key, name, reserved, kind, value: store.__setitem__(name, value)),
    ):
        setattr(mod, name, value)
    return mod


def test_strip_is_prefix_scoped_to_hermes_owned_entries(monkeypatch, tmp_path):
    """Only ``<home>\\{...}`` entries go; a lookalike with the same leaf survives.

    This is the property the production path relies on, and the one a silent edit to
    ``_hermes_path_markers`` breaks -- drop the ``root`` prefix, or switch
    ``startswith`` for a substring test, and an unrelated ``...\\Git\\cmd`` or a
    different drive's ``hermes\\bin`` starts disappearing. It is also why the
    deleter cannot double as a leak cleaner: it cannot tell a leaked entry from a
    legitimately installed ``%LOCALAPPDATA%\\hermes\\bin``.

    The registry is an in-memory dict (no key is opened), and the marker is cleared
    so the deletion logic itself runs.
    """
    import sys

    from hermes_cli import uninstall

    home = tmp_path / "hermes"
    owned = [str(home) + "\\bin", str(home) + "\\git\\cmd", str(home) + "\\node"]
    lookalikes = [
        "C:\\Program Files\\Git\\cmd",  # a real Git install, not Hermes's copy
        "C:\\tools\\hermes-agent",  # same leaf, unrelated parent
        "D:\\hermes\\bin",  # same leaf, other drive
        str(tmp_path / "hermes-setup") + "\\bin",  # sibling dir sharing the home's name prefix
    ]
    store = {"Path": ";".join(owned + lookalikes)}
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(store))
    monkeypatch.delenv("HERMES_TEST_ISOLATION", raising=False)

    removed = uninstall.remove_path_from_windows_registry(home, include_managed_bin=True)

    assert sorted(removed) == sorted(owned)
    assert store["Path"].split(";") == lookalikes
