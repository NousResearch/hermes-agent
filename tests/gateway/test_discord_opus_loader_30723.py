"""Behavioural regression tests for ``ensure_discord_opus_loaded`` (#30723).

The pre-#30723 Discord adapter relied on ``ctypes.util.find_library("opus")``
plus a macOS-only Homebrew fallback. On NixOS that lookup returns
``None`` because libopus lives in a Nix store path that is not in the
linker cache, and the adapter would silently disable voice playback.

The shared loader in ``plugins.platforms.discord.opus_loader``:

  1. Honours ``DISCORD_OPUS_LIBRARY`` first — the recommended NixOS fix.
  2. Falls back through ``ctypes.util.find_library``.
  3. Then walks a per-platform candidate list (bare SONAMES for the
     dynamic linker plus distro absolute paths).
  4. Emits a single WARNING with the full attempt list pointing at
     ``DISCORD_OPUS_LIBRARY`` if every candidate failed.

These tests pin every branch with mocks so they run identically on
macOS / Linux / Windows CI runners.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from plugins.platforms.discord.opus_loader import (
    OPUS_LIBRARY_ENV_VAR,
    _build_unavailable_hint,
    _candidates_for_platform,
    ensure_discord_opus_loaded,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────
def _make_discord_stub(load_succeeds_for: List[str] | None = None) -> MagicMock:
    """Build a stub mimicking the surface of the ``discord`` module
    that the loader uses: ``discord.opus.is_loaded()`` and
    ``discord.opus.load_opus(name)``.

    ``load_succeeds_for`` is the list of candidate names that should
    "succeed" — calling ``load_opus(name)`` flips ``is_loaded`` to
    True. Every other name raises ``OSError`` (matching the real
    ``discord.py`` behaviour for a missing shared object).
    """
    state = {"loaded": False}
    success_set = set(load_succeeds_for or [])

    stub = MagicMock(name="discord")
    stub.opus = MagicMock(name="discord.opus")
    stub.opus.is_loaded = lambda: state["loaded"]

    def fake_load(name: str) -> None:
        if name in success_set:
            state["loaded"] = True
            return
        raise OSError(f"{name}: cannot open shared object file: No such file or directory")

    stub.opus.load_opus = MagicMock(side_effect=fake_load)
    return stub


@pytest.fixture
def fake_isfile():
    """Stub ``os.path.isfile`` — only paths in ``existing`` exist."""
    def factory(existing: set[str]):
        def _isfile(path: str) -> bool:
            return path in existing
        return _isfile
    return factory


# ──────────────────────────────────────────────────────────────────────
# 1. Short-circuit when already loaded
# ──────────────────────────────────────────────────────────────────────
class TestAlreadyLoadedShortCircuit:
    def test_returns_true_without_touching_env_or_find_library(self) -> None:
        discord_stub = MagicMock()
        discord_stub.opus.is_loaded.return_value = True

        find_lib = MagicMock(return_value=None)
        env: dict[str, str] = {OPUS_LIBRARY_ENV_VAR: "/should/not/be/read"}

        assert ensure_discord_opus_loaded(
            discord_module=discord_stub,
            platform="linux",
            env=env,
            isfile=lambda _: False,
            find_library=find_lib,
        ) is True

        discord_stub.opus.load_opus.assert_not_called()
        find_lib.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# 2. DISCORD_OPUS_LIBRARY override is tried FIRST
# ──────────────────────────────────────────────────────────────────────
class TestEnvVarOverrideHasPriority:
    """#30723: ``DISCORD_OPUS_LIBRARY`` must beat ctypes discovery so
    NixOS users who paste the Nix-store path always win."""

    def test_override_loaded_before_find_library_called(self) -> None:
        nix_path = "/nix/store/abc-libopus-1.4/lib/libopus.so"
        stub = _make_discord_stub(load_succeeds_for=[nix_path])
        find_lib = MagicMock(return_value="/usr/lib/libopus.so.0")

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={OPUS_LIBRARY_ENV_VAR: nix_path},
            isfile=lambda _: True,
            find_library=find_lib,
        )

        assert ok is True
        # Override is the FIRST load attempt — confirm by inspecting
        # the call list rather than just call count, because other
        # candidates would never get reached if the first one wins.
        first_attempt = stub.opus.load_opus.call_args_list[0][0][0]
        assert first_attempt == nix_path

    def test_override_used_even_when_find_library_succeeds(self) -> None:
        """The user setting ``DISCORD_OPUS_LIBRARY`` is an explicit
        statement of intent — Hermes must not silently prefer the
        ``find_library`` result over it."""
        nix_path = "/nix/store/abc/libopus.so"
        distro_path = "/usr/lib/x86_64-linux-gnu/libopus.so.0"
        stub = _make_discord_stub(load_succeeds_for=[nix_path, distro_path])

        ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={OPUS_LIBRARY_ENV_VAR: nix_path},
            isfile=lambda _: True,
            find_library=lambda _name: distro_path,
        )

        first_attempt = stub.opus.load_opus.call_args_list[0][0][0]
        assert first_attempt == nix_path

    def test_empty_override_falls_through_to_find_library(self) -> None:
        """An empty string env var must not be tried as a candidate —
        Linux ``CDLL("")`` would link the current process and is
        nonsense for libopus."""
        stub = _make_discord_stub(load_succeeds_for=["libopus.so.0"])

        ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={OPUS_LIBRARY_ENV_VAR: ""},
            isfile=lambda _: False,
            find_library=lambda _name: None,
        )

        attempted = [call[0][0] for call in stub.opus.load_opus.call_args_list]
        assert "" not in attempted


# ──────────────────────────────────────────────────────────────────────
# 3. NixOS scenario — find_library returns None, no DISCORD_OPUS_LIBRARY
# ──────────────────────────────────────────────────────────────────────
class TestNixOsScenarioWithoutOverride:
    """The exact failure mode reproduced in the #30723 issue body:
    ``ctypes.util.find_library("opus")`` returns ``None`` AND the user
    hasn't set ``DISCORD_OPUS_LIBRARY`` yet. The loader must still try
    the bare SONAMES via the dynamic linker so a non-Nix Linux user
    isn't punished for not setting the env var."""

    def test_bare_libopus_so_0_is_tried_when_find_library_returns_none(
        self,
    ) -> None:
        stub = _make_discord_stub(load_succeeds_for=["libopus.so.0"])

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=lambda _: False,
            find_library=lambda _name: None,
        )

        assert ok is True
        attempted = [call[0][0] for call in stub.opus.load_opus.call_args_list]
        assert "libopus.so.0" in attempted

    def test_fully_unloadable_returns_false_and_warns_with_env_hint(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        stub = _make_discord_stub(load_succeeds_for=[])

        caplog.set_level(logging.WARNING, logger="plugins.platforms.discord.opus_loader")
        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=lambda _: False,
            find_library=lambda _name: None,
        )

        assert ok is False
        # The warning must point the user at the env-var fix and cite
        # the issue so a future grep for "#30723" lands here.
        joined = " ".join(record.getMessage() for record in caplog.records)
        assert OPUS_LIBRARY_ENV_VAR in joined
        assert "#30723" in joined

    def test_attempt_list_in_warning_includes_failure_reasons(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Operators triaging a voice-failure should see exactly which
        names were tried and why each one failed."""
        stub = _make_discord_stub(load_succeeds_for=[])

        caplog.set_level(logging.WARNING, logger="plugins.platforms.discord.opus_loader")
        ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=lambda _: False,
            find_library=lambda _name: None,
        )

        joined = " ".join(record.getMessage() for record in caplog.records)
        assert "libopus.so.0" in joined
        assert "OSError" in joined or "cannot open shared object" in joined


# ──────────────────────────────────────────────────────────────────────
# 4. macOS Homebrew fallback preserved
# ──────────────────────────────────────────────────────────────────────
class TestMacosHomebrewFallbackPreserved:
    """Pre-#30723 behaviour: when ``ctypes.util.find_library`` fails on
    macOS, the loader still falls back to the Apple-Silicon / Intel
    Homebrew default prefixes. The refactor must not regress this."""

    def test_apple_silicon_homebrew_path_is_tried(
        self, fake_isfile,
    ) -> None:
        brew_path = "/opt/homebrew/lib/libopus.dylib"
        stub = _make_discord_stub(load_succeeds_for=[brew_path])

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="darwin",
            env={},
            isfile=fake_isfile({brew_path}),
            find_library=lambda _name: None,
        )

        assert ok is True
        first_attempt = stub.opus.load_opus.call_args_list[0][0][0]
        assert first_attempt == brew_path

    def test_intel_mac_path_used_when_apple_silicon_absent(
        self, fake_isfile,
    ) -> None:
        intel_path = "/usr/local/lib/libopus.dylib"
        stub = _make_discord_stub(load_succeeds_for=[intel_path])

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="darwin",
            env={},
            isfile=fake_isfile({intel_path}),
            find_library=lambda _name: None,
        )

        assert ok is True
        attempted = [call[0][0] for call in stub.opus.load_opus.call_args_list]
        assert intel_path in attempted
        assert "/opt/homebrew/lib/libopus.dylib" not in attempted, (
            "absolute paths that don't exist must not be tried"
        )

    def test_linux_paths_are_not_tried_on_darwin(self, fake_isfile) -> None:
        """A regression where the Linux candidate list leaked into the
        macOS branch would generate noisy ``OSError: cannot open shared
        object`` warnings on every Mac startup."""
        stub = _make_discord_stub(load_succeeds_for=[])
        # All darwin and linux candidates absent.
        ensure_discord_opus_loaded(
            discord_module=stub,
            platform="darwin",
            env={},
            isfile=fake_isfile(set()),
            find_library=lambda _name: None,
        )
        attempted = [call[0][0] for call in stub.opus.load_opus.call_args_list]
        assert "libopus.so.0" not in attempted
        assert "libopus.so" not in attempted


# ──────────────────────────────────────────────────────────────────────
# 5. Linux distro fallback list
# ──────────────────────────────────────────────────────────────────────
class TestLinuxDistroFallback:
    @pytest.mark.parametrize(
        "distro_path",
        [
            "/usr/lib/x86_64-linux-gnu/libopus.so.0",
            "/usr/lib/aarch64-linux-gnu/libopus.so.0",
            "/usr/lib/libopus.so",
            "/usr/lib64/libopus.so",
            "/usr/lib/libopus.so.0",
        ],
    )
    def test_distro_path_loaded_when_present(
        self, fake_isfile, distro_path: str,
    ) -> None:
        """For each distro convention, the loader must succeed when only
        that specific path exists. Drives the entire
        ``_LINUX_FALLBACK_PATHS`` tuple."""
        # No bare SONAMEs available on the dynamic linker either, so
        # the only thing that can succeed is the absolute path.
        stub = _make_discord_stub(load_succeeds_for=[distro_path])

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=fake_isfile({distro_path}),
            find_library=lambda _name: None,
        )

        assert ok is True
        attempted = [call[0][0] for call in stub.opus.load_opus.call_args_list]
        assert distro_path in attempted

    def test_absent_absolute_paths_are_not_attempted(
        self, fake_isfile,
    ) -> None:
        """Absolute paths must be gated by ``isfile`` so a load failure
        only fires for paths that actually exist — otherwise distro CI
        runners log a wall of misleading OSErrors on every startup."""
        stub = _make_discord_stub(load_succeeds_for=[])

        ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=fake_isfile(set()),
            find_library=lambda _name: None,
        )

        attempted = [call[0][0] for call in stub.opus.load_opus.call_args_list]
        # Bare SONAMEs are always tried (the dynamic linker resolves them).
        assert "libopus.so.0" in attempted
        assert "libopus.so" in attempted
        # Absolute paths that don't exist must NOT have been tried.
        for absolute in (
            "/usr/lib/x86_64-linux-gnu/libopus.so.0",
            "/usr/lib/aarch64-linux-gnu/libopus.so.0",
            "/usr/lib/libopus.so",
            "/usr/lib64/libopus.so",
            "/usr/lib/libopus.so.0",
        ):
            assert absolute not in attempted


# ──────────────────────────────────────────────────────────────────────
# 6. Robustness — find_library raising, dedup, etc.
# ──────────────────────────────────────────────────────────────────────
class TestRobustness:
    def test_find_library_raising_does_not_propagate(self) -> None:
        """Exotic ctypes shims (Termux, some embedded Linuxes) raise
        instead of returning ``None``. The loader must trap and
        continue to the platform fallbacks."""
        stub = _make_discord_stub(load_succeeds_for=["libopus.so.0"])

        def boom(_name: str) -> str:
            raise RuntimeError("find_library is unsupported on this platform")

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=lambda _: False,
            find_library=boom,
        )

        assert ok is True

    def test_duplicate_paths_only_attempted_once(self) -> None:
        """If ``DISCORD_OPUS_LIBRARY`` and ``find_library`` return the
        same path, the loader must not attempt it twice (would create
        confusing duplicate WARNING entries on failure)."""
        path = "/usr/lib/libopus.so.0"
        stub = _make_discord_stub(load_succeeds_for=[])

        ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={OPUS_LIBRARY_ENV_VAR: path},
            isfile=lambda p: p == path,
            find_library=lambda _name: path,
        )

        attempts_for_path = [
            call for call in stub.opus.load_opus.call_args_list
            if call[0][0] == path
        ]
        assert len(attempts_for_path) == 1

    def test_isfile_raising_on_one_path_does_not_abort_loop(
        self,
    ) -> None:
        """A racing ``stat()`` failure on one path must not block the
        remaining candidates from being tried."""
        bad_path = "/usr/lib/x86_64-linux-gnu/libopus.so.0"
        good_path = "/usr/lib64/libopus.so"

        def isfile(p: str) -> bool:
            if p == bad_path:
                raise PermissionError("EACCES")
            return p == good_path

        stub = _make_discord_stub(load_succeeds_for=[good_path])

        ok = ensure_discord_opus_loaded(
            discord_module=stub,
            platform="linux",
            env={},
            isfile=isfile,
            find_library=lambda _name: None,
        )

        assert ok is True

    def test_missing_discord_module_returns_false_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When discord.py is not installed and no stub is injected,
        the loader must fail closed with a clear log message, not
        ``ImportError``-up-the-stack."""
        import sys
        # Hide discord from the loader's lazy import.
        monkeypatch.setitem(sys.modules, "discord", None)
        ok = ensure_discord_opus_loaded(
            platform="linux",
            env={},
            isfile=lambda _: False,
            find_library=lambda _name: None,
        )
        assert ok is False


# ──────────────────────────────────────────────────────────────────────
# 7. _candidates_for_platform — direct unit coverage
# ──────────────────────────────────────────────────────────────────────
class TestCandidatesForPlatform:
    def test_darwin_returns_only_dylib_paths(self) -> None:
        candidates = _candidates_for_platform("darwin")
        assert all(c.endswith(".dylib") for c in candidates)
        assert "/opt/homebrew/lib/libopus.dylib" in candidates
        assert "/usr/local/lib/libopus.dylib" in candidates

    def test_linux_starts_with_bare_sonames(self) -> None:
        """The dynamic linker resolves bare SONAMEs against the runtime
        cache; trying them first means we don't false-fail on systems
        where the absolute path moved but ld.so.cache still resolves."""
        candidates = _candidates_for_platform("linux")
        assert candidates[0] == "libopus.so.0"
        assert candidates[1] == "libopus.so"

    def test_win32_returns_empty(self) -> None:
        """discord.py bundles libopus DLLs on Windows; nothing to add."""
        assert _candidates_for_platform("win32") == []

    def test_unknown_unix_platform_falls_through_to_linux_list(
        self,
    ) -> None:
        """FreeBSD / NixOS / Termux — none of these are ``darwin`` or
        ``win32``, so they should reuse the Linux candidate list."""
        assert _candidates_for_platform("freebsd13") == _candidates_for_platform("linux")


# ──────────────────────────────────────────────────────────────────────
# 8. _build_unavailable_hint — message shape
# ──────────────────────────────────────────────────────────────────────
class TestUnavailableHint:
    def test_no_attempts_message_is_actionable(self) -> None:
        msg = _build_unavailable_hint([])
        assert OPUS_LIBRARY_ENV_VAR in msg
        assert "#30723" in msg
        assert "No candidates" in msg

    def test_attempts_list_renders_candidate_and_reason(self) -> None:
        attempts: List[Tuple[str, str]] = [
            ("libopus.so.0", "OSError: cannot open shared object file"),
            ("/nix/store/abc/libopus.so", "OSError: missing"),
        ]
        msg = _build_unavailable_hint(attempts)
        assert "libopus.so.0" in msg
        assert "/nix/store/abc/libopus.so" in msg
        assert "cannot open shared object file" in msg

    def test_hint_mentions_both_linux_and_macos_artifacts(self) -> None:
        """The hint is the only thing the user sees, so it should
        cover both Linux (libopus.so) and macOS (libopus.dylib) so
        cross-platform users don't have to figure out the right
        extension."""
        msg = _build_unavailable_hint([])
        assert "libopus.so" in msg
        assert "libopus.dylib" in msg


# ──────────────────────────────────────────────────────────────────────
# 9. Public API surface
# ──────────────────────────────────────────────────────────────────────
class TestModuleSurface:
    def test_env_var_name_is_stable(self) -> None:
        """Operators wire ``DISCORD_OPUS_LIBRARY`` into their .env /
        Nix module — the constant name and value are part of the
        public contract, not internal trivia."""
        assert OPUS_LIBRARY_ENV_VAR == "DISCORD_OPUS_LIBRARY"

    def test_doctor_script_imports_the_same_helper(self) -> None:
        """``scripts/discord-voice-doctor.py`` and the adapter must
        share the loader so a NixOS user's diagnosis matches their
        runtime behaviour."""
        from pathlib import Path
        doctor = Path(__file__).resolve().parents[2] / "scripts" / "discord-voice-doctor.py"
        text = doctor.read_text()
        assert "ensure_discord_opus_loaded" in text
        assert "OPUS_LIBRARY_ENV_VAR" in text

    def test_loader_module_is_importable_without_discord_installed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Importing the loader module must not require discord.py —
        it's the discoverable-on-failure path; ``import opus_loader``
        itself can't trip the user before they even start the bot."""
        import sys
        monkeypatch.setitem(sys.modules, "discord", None)
        # Re-importing the module surface; the lazy ``import discord``
        # lives inside ensure_discord_opus_loaded(), not at module top.
        import importlib
        import plugins.platforms.discord.opus_loader as loader
        importlib.reload(loader)
        assert hasattr(loader, "ensure_discord_opus_loaded")
        assert hasattr(loader, "OPUS_LIBRARY_ENV_VAR")
