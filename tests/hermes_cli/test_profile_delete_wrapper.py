"""``hermes profile delete`` must remove the command wrapper it created — on every platform.

The wrapper is written at ``_wrapper_path(alias)`` (``<alias>.bat`` on Windows, bare name
elsewhere) and removed by ``remove_wrapper_script()``, which already probes both candidates.
``delete_profile`` gated the removal on its own extensionless ``_get_wrapper_dir() / canon``
probe, so on Windows the gate never matched the ``.bat`` file that actually exists, and a
profile carrying a CUSTOM alias (``hermes profile alias demo --name bot``) was never probed at
all: ``demo`` names no file, ``bot`` does. Either way the profile directory was deleted while the
wrapper survived, and the surviving wrapper re-created the deleted profile home on its next
invocation.

Regression for #126210. Host-native: the suffix is a real platform fact, so each OS asserts its
own wrapper name rather than a simulated one.
"""

import builtins
import sys
from pathlib import Path

import pytest

from hermes_cli.profiles import (
    _wrapper_path,
    create_profile,
    create_wrapper_script,
    delete_profile,
    find_alias_for_profile,
)


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated profile root. ``Path.home()`` and ``HERMES_HOME`` must agree — see AGENTS.md."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _make_wrapper(alias: str, target: str) -> Path:
    """Write the wrapper ``hermes profile alias <target> --name <alias>`` would have written."""
    wrapper = _wrapper_path(alias)
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        f"@echo off\r\nhermes -p {target} %*\r\n"
        if sys.platform == "win32"
        else f'#!/bin/sh\nexec hermes -p {target} "$@"\n',
        encoding="utf-8",
    )
    return wrapper


def _silence_delete_side_effects(monkeypatch):
    """Neutralize the process-global teardown ``delete_profile`` performs alongside the unlink.

    Wrapper removal is step 3 of the delete; gateway/identity/logging teardown is unrelated
    machinery that would only add real subprocess scans to this contract. ``_purge_identity``
    must report settled (True) or the delete raises on a partial settlement.
    """
    for name in (
        "_cleanup_gateway_service",
        "_maybe_unregister_gateway_service",
        "_check_gateway_running",
        "_stop_profile_backends",
        "_stop_bot_desktop",
        "_notify_multiplexer",
    ):
        monkeypatch.setattr(f"hermes_cli.profiles.{name}", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.profiles._purge_identity", lambda *a, **k: True)
    monkeypatch.setattr("hermes_cli.profiles._retarget_active_profile", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.profiles._rmtree_with_retry", lambda path, *a, **k: None)
    monkeypatch.setattr("hermes_cli.profiles.mark_named_profile_deleted", lambda *a, **k: None)


class TestDeleteRemovesOwnWrapper:
    """A deleted profile leaves no wrapper that would resurrect its home directory."""

    def test_profile_named_wrapper_is_removed(self, profile_env, monkeypatch):
        """The wrapper written for the profile's own name is gone after delete.

        Host-native: on Windows that file is ``<name>.bat`` and the extensionless probe never
        sees it; on POSIX the bare name matches, so this row is the Windows-specific half.
        """
        create_profile("demo", no_alias=True)
        wrapper = create_wrapper_script("demo")
        assert wrapper is not None and wrapper.is_file(), f"precondition: {wrapper} should exist"

        _silence_delete_side_effects(monkeypatch)
        delete_profile("demo", yes=True)

        assert not wrapper.exists(), (
            f"{wrapper} survived profile delete; running it would re-create the deleted home"
        )

    def test_custom_alias_wrapper_is_removed(self, profile_env, monkeypatch):
        """A wrapper named for a CUSTOM alias is also removed — on both platforms.

        ``hermes profile alias demo --name bot`` writes ``bot`` (``.bat`` on Windows). The
        extensionless ``demo`` probe misses it on every platform, so this half of the class was
        broken everywhere, not just on Windows.
        """
        create_profile("demo", no_alias=True)
        wrapper = _make_wrapper("bot", "demo")
        assert find_alias_for_profile("demo") == "bot", "precondition: alias must resolve"

        _silence_delete_side_effects(monkeypatch)
        delete_profile("demo", yes=True)

        assert not wrapper.exists(), (
            f"{wrapper} survived profile delete; running it would re-create the deleted home"
        )

    def test_summary_advertises_the_wrapper_that_actually_exists(self, profile_env, monkeypatch, capsys):
        """The pre-delete confirmation names the real file, so consent covers what is removed.

        Before the fix the summary rendered the extensionless path — a file that does not exist
        on Windows — and stayed silent about a custom alias entirely.
        """
        create_profile("demo", no_alias=True)
        wrapper = create_wrapper_script("demo")
        monkeypatch.setattr(builtins, "input", lambda *a: "wrong-name")  # decline, not abort

        _silence_delete_side_effects(monkeypatch)
        delete_profile("demo", yes=False)

        out = capsys.readouterr().out
        assert str(wrapper) in out, f"summary should name the real wrapper {wrapper}, got:\n{out}"

    def test_a_wrapper_for_another_profile_survives(self, profile_env, monkeypatch):
        """Deleting one profile must not take a sibling profile's wrapper with it.

        The negative case: removal is name-scoped, so an over-broad fix (wiping the whole
        wrapper dir) would pass every test above while destroying unrelated profiles' commands.
        """
        create_profile("keep", no_alias=True)
        keeper = create_wrapper_script("keep")
        create_profile("demo", no_alias=True)
        doomed = create_wrapper_script("demo")
        assert keeper is not None and keeper.is_file()
        assert doomed is not None and doomed.is_file()

        _silence_delete_side_effects(monkeypatch)
        delete_profile("demo", yes=True)

        assert not doomed.exists()
        assert keeper.is_file(), f"{keeper} belongs to 'keep' and must survive deleting 'demo'"
