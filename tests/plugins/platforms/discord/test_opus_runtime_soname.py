"""Discord voice must find the opus that the docs tell Linux users to install.

``docs/user-guide/features/voice-mode.md`` says ``sudo apt install libopus0``. That package
installs ``libopus.so.0`` only; the linker name ``libopus.so`` comes from ``libopus-dev``.
CPython's ``ctypes.util.find_library("opus")`` resolves the *linker* name on Linux, so on a host
that followed the guide it returns ``None`` and voice playback is silently disabled::

    WARNING plugins.platforms.discord.adapter_opus:
        Opus codec not found - voice channel playback disabled

macOS already carries a versioned-soname fallback for the identical Homebrew blind spot
(``find_library`` misses Homebrew libs); Linux had no equivalent arm. These tests pin that parity.

The probe is exercised as behaviour: platform and ``find_library`` are stubbed and
``discord.opus.load_opus`` records what the probe actually tried, so the assertions fail if the
Linux arm is removed.
"""
from __future__ import annotations

import ctypes.util
import os
import re
from pathlib import Path

import pytest

from plugins.platforms.discord import adapter_opus

_SONAME = "libopus.so.0"


@pytest.fixture(autouse=True)
def _reset_probe_state():
    """The probe warns once per process; each test needs a fresh latch."""
    adapter_opus._probed = False
    yield
    adapter_opus._probed = False


def _stub(monkeypatch, *, platform="linux", find_library_result=None, existing=()):
    """Return the list that receives every candidate path the probe attempted to load."""
    import discord

    monkeypatch.setattr(adapter_opus.sys, "platform", platform)
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: find_library_result)
    monkeypatch.setattr(
        adapter_opus, "_find_discord_windows_bundled_opus", lambda _module: None
    )
    real_isfile = os.path.isfile
    monkeypatch.setattr(
        adapter_opus.os.path, "isfile", lambda p: p in existing or real_isfile(p)
    )
    monkeypatch.setattr(discord.opus, "is_loaded", lambda: False)
    attempted: list[str] = []
    monkeypatch.setattr(discord.opus, "load_opus", lambda path: attempted.append(path))
    return attempted


@pytest.mark.parametrize("platform_name", ["linux", "freebsd"])
def test_runtime_soname_probed_when_find_library_fails(monkeypatch, platform_name):
    """libopus0-style installs must still be found (issue: apt libopus0, no libopus-dev)."""
    staged = "/staged/usr/lib/x86_64-linux-gnu/libopus.so.0"
    monkeypatch.setattr(adapter_opus, "_LINUX_OPUS_SONAME_CANDIDATES", (staged,))
    attempted = _stub(monkeypatch, platform=platform_name, existing=(staged,))

    adapter_opus.load_opus_codec()

    assert attempted == [staged], (
        "the probe never reached a Linux soname candidate — find_library is the only "
        "Linux path, and it cannot resolve a runtime-only libopus install"
    )


def test_first_existing_candidate_wins(monkeypatch):
    """Candidate order matters: the earliest installed path is the one we load."""
    first = "/staged/usr/lib/x86_64-linux-gnu/libopus.so.0"
    second = "/staged/usr/lib64/libopus.so.0"
    monkeypatch.setattr(adapter_opus, "_LINUX_OPUS_SONAME_CANDIDATES", (first, second))
    attempted = _stub(monkeypatch, existing=(first, second))

    adapter_opus.load_opus_codec()

    assert attempted == [first]


def test_macos_homebrew_arm_still_probed(monkeypatch):
    """Regression guard: the new Linux else-arm must not shadow the macOS arm."""
    brew = "/opt/homebrew/lib/libopus.dylib"
    monkeypatch.setattr(adapter_opus, "_LINUX_OPUS_SONAME_CANDIDATES", ("/nope/libopus.so.0",))
    attempted = _stub(monkeypatch, platform="darwin", existing=(brew,))

    adapter_opus.load_opus_codec()

    assert attempted == [brew], "macOS must keep probing its Homebrew soname"
    assert not any(_SONAME in p for p in attempted), "Linux candidates must not be probed on macOS"


def test_find_library_result_is_preferred_over_sonames(monkeypatch):
    """When the linker name *does* resolve (libopus-dev installed), keep using it."""
    soname = "/staged/usr/lib/x86_64-linux-gnu/libopus.so.0"
    monkeypatch.setattr(adapter_opus, "_LINUX_OPUS_SONAME_CANDIDATES", (soname,))
    attempted = _stub(
        monkeypatch, find_library_result="/usr/lib/x86_64-linux-gnu/libopus.so",
        existing=(soname,),
    )

    adapter_opus.load_opus_codec()

    assert attempted[0].endswith("libopus.so") and not attempted[0].endswith(_SONAME)


def test_missing_opus_warns_exactly_once(monkeypatch, caplog):
    """No-opus hosts keep the single warning; the fix must not raise or spam."""
    monkeypatch.setattr(adapter_opus, "_LINUX_OPUS_SONAME_CANDIDATES", ("/nope/libopus.so.0",))
    with caplog.at_level("WARNING", logger="plugins.platforms.discord.adapter_opus"):
        attempted = _stub(monkeypatch)
        adapter_opus.load_opus_codec()
        adapter_opus.load_opus_codec()

    assert not attempted
    assert caplog.text.count("Opus codec not found") == 1


def _repo_script(name: str) -> Path | None:
    """Repo-root-relative script path, found by walking up from this file.

    Never build it from a fixed ``parents[N]`` count: under a scratch-path worktree a fixed
    ancestor index resolves *outside* the checkout, and ``tests/home_io_guard.py`` (correctly)
    refuses the read. Stopping at the first ancestor holding both markers cannot escape.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / name
        if candidate.is_file() and (ancestor / "pyproject.toml").is_file():
            return candidate
    return None


def test_linux_candidates_are_versioned_sonames():
    """The list must hold runtime sonames, not linker names (which -dev-only installs lack)."""
    assert adapter_opus._LINUX_OPUS_SONAME_CANDIDATES, "Linux fallback list must not be empty"
    assert all(
        c.endswith(".so.0") or c.endswith(".so") for c in adapter_opus._LINUX_OPUS_SONAME_CANDIDATES
    )
    assert all(c.startswith("/") for c in adapter_opus._LINUX_OPUS_SONAME_CANDIDATES)
    # The documented `apt install libopus0` target must be covered.
    assert "/usr/lib/x86_64-linux-gnu/libopus.so.0" in adapter_opus._LINUX_OPUS_SONAME_CANDIDATES
    # Must stay a superset of the Linux paths scripts/discord-voice-doctor.py probes, so the
    # runtime never looks in fewer places than the repo's own doctor does.
    doctor = _repo_script("scripts/discord-voice-doctor.py")
    if doctor is not None:
        probed = {
            p for p in re.findall(r'"(/usr/[^"]*libopus[^"]*)"', doctor.read_text())
            if p.endswith(".so") or p.endswith(".so.0")
        }
        assert probed, "doctor probe list changed shape; this guard needs updating"
        missing = probed - set(adapter_opus._LINUX_OPUS_SONAME_CANDIDATES)
        assert not missing, f"doctor probes paths the runtime does not: {sorted(missing)}"
