"""Tests for the session-scoped messaging write guard (agent.file_safety).

Messaging-platform sessions are denied writes to the execution-trusting
roots (cron/, scripts/) of the active home and every sibling profile.
Cross-platform logic runs
unmarked on every lane (Windows conventions via real ntpath + shim, darwin
seam via simulated probe); native behavior runs in windows_only /
macos_only lanes per tests-os.yml doctrine.
"""

from __future__ import annotations

import contextlib
import ntpath
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_constants
from agent import file_safety
from tools.approval import reset_current_session_key, set_current_session_key
from tools.file_tools import (
    _check_sensitive_path,
    _resolve_path_for_task,
    write_file_tool,
    patch_tool,
)

# Windows extended-length prefixes accepted by the OS (stripped by the
# guard's Windows normalization).
_WIN_EXTENDED_PREFIXES = ("\\\\?\\", "\\.\\", "\\??\\")


@contextlib.contextmanager
def _bind_key(key: str):
    token = set_current_session_key(key)
    try:
        yield
    finally:
        reset_current_session_key(token)


@contextlib.contextmanager
def _default_home():
    """Simulate the POSIX default home (~/.hermes) on every platform."""
    with patch.dict(os.environ, {"HERMES_HOME": ""}), \
            patch.object(
                hermes_constants, "_get_platform_default_hermes_home",
                return_value=Path.home() / ".hermes",
            ):
        yield


def _win_case_normalize(path: str) -> str:
    """Windows case/separator semantics, using the real ntpath primitives."""
    for prefix in _WIN_EXTENDED_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    return ntpath.normcase(path).replace("\\", "/")


@contextlib.contextmanager
def _win_semantics():
    """Run the guard with Windows normalization semantics on any host."""
    with patch("agent.file_safety._normalize_guard_path",
               side_effect=_win_case_normalize):
        yield


class _FakeSys:
    """Minimal sys stand-in forcing the Darwin branch of the seam."""

    def __init__(self):
        self.platform = "darwin"

    def __getattr__(self, name):
        raise AttributeError(name)


@contextlib.contextmanager
def _darwin_simulated(case_sensitive: bool):
    """Simulate the Darwin case-sensitivity seam on any host.

    Forces the darwin gate + a deterministic volume-probe answer, with the
    default home in a temp dir on a stable volume (the /var -> /private/var
    symlink would defeat the resolved/casefolded prefix match).
    """
    base = os.path.dirname(os.path.abspath(os.curdir))
    with tempfile.TemporaryDirectory(dir=base) as tmp, \
            patch.dict(os.environ, {"HERMES_HOME": tmp}), \
            patch.object(file_safety, "sys", _FakeSys()), \
            patch.object(file_safety, "_darwin_case_insensitive",
                         return_value=not case_sensitive), \
            patch("hermes_constants._get_platform_default_hermes_home",
                  return_value=Path(tmp)):
        file_safety._darwin_case_insensitive.cache_clear()
        os.makedirs(os.path.join(tmp, "scripts"), exist_ok=True)
        # On the simulated case-insensitive volume the case variant
        # resolves to the same dir; create it so the probe/test path is
        # plausible on every host (a real CI volume has the variant as the
        # same inode).
        if not case_sensitive:
            os.makedirs(os.path.join(tmp, "SCRIPTS"), exist_ok=True)
        yield tmp


class TestSessionPlatform(unittest.TestCase):
    def test_gateway_key_parses_platform(self):
        with _bind_key("agent:main:telegram:dm:123"):
            self.assertEqual(
                file_safety._messaging_platform_from_key(), "telegram"
            )

    def test_named_profile_gateway_key(self):
        with _bind_key("agent:example-profile:discord:g:9"):
            self.assertEqual(
                file_safety._messaging_platform_from_key(), "discord"
            )

    def test_unregistered_platform_token_is_none(self):
        with _bind_key("agent:main:not_a_real_platform:dm:1"):
            self.assertIsNone(file_safety._messaging_platform_from_key())

    def test_local_platform_is_none(self):
        with _bind_key("agent:main:local:dm:1"):
            self.assertIsNone(file_safety._messaging_platform_from_key())

    def test_api_server_platform_is_messaging_like(self):
        # API-server sessions bind api_server keys and are non-terminal,
        # non-interactive — the guard treats them like messaging surfaces
        # (they cannot approve or reason around the write).
        with _bind_key("agent:main:api_server:r:1"):
            self.assertEqual(
                file_safety._messaging_platform_from_key(), "api_server"
            )

    def test_malformed_key_is_none(self):
        with _bind_key("agent:main"):
            self.assertIsNone(file_safety._messaging_platform_from_key())

    def test_cli_key_has_no_platform(self):
        with _bind_key("default"):
            self.assertIsNone(file_safety._messaging_platform_from_key())
        with _bind_key("20260811_093018_2cb019f0"):
            self.assertIsNone(file_safety._messaging_platform_from_key())

    def test_no_key_binds_none(self):
        self.assertIsNone(file_safety._messaging_platform_from_key())


class TestMessagingWriteGuard(unittest.TestCase):
    def test_telegram_cron_write_blocked(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            err = file_safety.get_messaging_write_block_error(
                "~/.hermes/cron/jobs.json"
            )
            self.assertIsNotNone(err)
            self.assertIn("Access denied", err)
            self.assertIn("telegram", err)

    def test_telegram_scripts_write_blocked(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/scripts/evil.sh"
                )
            )

    def test_telegram_unrelated_paths_allowed(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "/tmp/note.md"
                )
            )
            # Non-execution hermes paths (logs, etc.) remain writable
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/logs/app.log"
                )
            )

    def test_cli_never_blocked(self):
        with _bind_key("default"), _default_home():
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/cron/jobs.json"
                )
            )
        with _bind_key("20260811_093018_2cb019f0"), _default_home():
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/scripts/x.sh"
                )
            )

    def test_unregistered_platform_not_blocked(self):
        with _bind_key("agent:main:not_a_real_platform:dm:1"), _default_home():
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/cron/jobs.json"
                )
            )

    def test_local_platform_not_blocked(self):
        with _bind_key("agent:main:local:dm:1"), _default_home():
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/cron/jobs.json"
                )
            )

    def test_api_server_blocked(self):
        # API-server sessions are non-terminal and non-interactive — the
        # guard fires for them like any messaging surface (they cannot
        # approve or reason around a write to an execution root).
        with _bind_key("agent:main:api_server:r:1"), _default_home():
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/cron/jobs.json"
                )
            )

    def test_other_gateway_platforms_blocked(self):
        with _bind_key("agent:main:slack:dm:9"), _default_home():
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/cron/jobs.json"
                )
            )
        with _bind_key("agent:main:whatsapp:dm:7"), _default_home():
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    "~/.hermes/scripts/x.sh"
                )
            )

    def test_ssh_out_of_scope_unchanged(self):
        # SSH key material is deliberately OUT of the messaging guard's
        # scope (backend write-deny covers it for all sessions; read side
        # keeps its own defence-in-depth list) — regression pin.
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    "~/.ssh/id_ed25519"
                )
            )

    def test_implicit_parent_creation_blocked(self):
        # Write to a not-yet-existing <home>/cron/ subdir must be caught by
        # the RAW normalized form even when no parent directory exists.
        with _bind_key("agent:main:telegram:dm:1"):
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, {"HERMES_HOME": tmp}):
                target = os.path.join(tmp, "cron", "sub", "jobs.json")
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(target)
                )


class TestExecutionTrustingRootsFollowActiveHome(unittest.TestCase):
    """Regression coverage for the live-home derivation (reviewer P1)."""

    def test_custom_hermes_home_roots_blocked(self):
        with _bind_key("agent:main:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HERMES_HOME": tmp}):
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "cron", "jobs.json")
                )
            )
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "scripts", "x.sh")
                )
            )
            # Unrelated paths under the same home remain writable
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "notes.md")
                )
            )

    def test_profile_style_home_roots_blocked(self):
        with _bind_key("agent:mypro:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as tmp:
            profile_home = os.path.join(tmp, "profiles", "mypro")
            with patch.dict(os.environ, {"HERMES_HOME": profile_home}):
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(profile_home, "scripts", "x.sh")
                    )
                )
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(profile_home, "cron", "jobs.json")
                    )
                )

    def test_custom_home_local_session_not_blocked(self):
        with _bind_key("default"), \
                tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HERMES_HOME": tmp}):
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "scripts", "x.sh")
                )
            )

    def test_prefixes_follow_active_home(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HERMES_HOME": tmp}):
            prefixes = file_safety._execution_trusting_prefixes()
            self.assertIn(os.path.join(tmp, "cron"), prefixes)
            self.assertIn(os.path.join(tmp, "scripts"), prefixes)
            # identical env/live home dedupes to a single root pair
            self.assertEqual(prefixes.count(os.path.join(tmp, "cron")), 1)

    def test_override_scoped_session_blocks_both_homes(self):
        with _bind_key("agent:main:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as env_home, \
                tempfile.TemporaryDirectory() as override_home, \
                patch.dict(os.environ, {"HERMES_HOME": env_home}):
            token = hermes_constants.set_hermes_home_override(override_home)
            try:
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(env_home, "scripts", "x.sh")
                    )
                )
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(override_home, "scripts", "x.sh")
                    )
                )
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(override_home, "cron", "jobs.json")
                    )
                )
            finally:
                hermes_constants.reset_hermes_home_override(token)

    def test_sibling_profile_roots_blocked(self):
        with _bind_key("agent:A:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as root:
            profile_a = os.path.join(root, "profiles", "A")
            profile_b = os.path.join(root, "profiles", "B")
            os.makedirs(profile_a)
            os.makedirs(profile_b)
            with patch.object(
                    hermes_constants, "get_default_hermes_root",
                    return_value=Path(root)), \
                    patch.dict(os.environ, {"HERMES_HOME": profile_a}):
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(profile_b, "scripts", "x.sh")
                    )
                )
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(profile_b, "cron", "jobs.json")
                    )
                )
                # Default profile's execution roots at the shared root
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(root, "cron", "jobs.json")
                    )
                )
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(root, "scripts", "x.sh")
                    )
                )
                # Own profile still blocked (regression)
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(profile_a, "scripts", "x.sh")
                    )
                )
                # Unrelated root paths stay writable
                self.assertIsNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(root, "notes.md")
                    )
                )

    def test_windows_conventions_roots_blocked(self):
        # Windows default home and conventions are covered by the
        # windows_only native lane; the unmarked lane keeps the generic
        # logic (custom-home + profile + sibling) that runs on every host.
        with _bind_key("agent:main:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HERMES_HOME": tmp}):
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "scripts", "x.ps1")
                )
            )
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "cron", "jobs.json")
                )
            )
            # Unrelated paths under the same home remain writable
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "logs", "app.log")
                )
            )

    def test_darwin_case_sensitivity_seam(self):
        # On a case-insensitive volume (default APFS) a case variant of the
        # home is the same directory and is blocked; on a case-sensitive
        # volume it is a genuinely different directory and stays writable.
        # Both arms run on every host via the simulated probe.
        with _bind_key("agent:main:telegram:dm:1"), \
                _darwin_simulated(case_sensitive=False) as tmp:
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "SCRIPTS", "x.sh")
                )
            )
        with _bind_key("agent:main:telegram:dm:1"), \
                _darwin_simulated(case_sensitive=True) as tmp:
            self.assertIsNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "SCRIPTS", "x.sh")
                )
            )

    def test_darwin_case_sensitivity_seam_unmarked_runs_everywhere(self):
        # Extreme pin: the unmarked seam tests above must EXECUTE their
        # darwin arm on every lane (no platform skip), so the
        # case-sensitive-volume answer cannot silently die on Linux CI.
        with _darwin_simulated(case_sensitive=False) as tmp:
            self.assertEqual(
                file_safety._normalize_guard_path(
                    os.path.join(tmp, "Scripts", "x.sh")
                ),
                os.path.join(tmp, "scripts", "x.sh").casefold(),
            )
        with _darwin_simulated(case_sensitive=True) as tmp:
            self.assertEqual(
                file_safety._normalize_guard_path(
                    os.path.join(tmp, "Scripts", "x.sh")
                ),
                os.path.join(tmp, "Scripts", "x.sh"),
            )

    def test_symlink_alias_into_scripts_dir_blocked(self):
        with _bind_key("agent:main:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as tmp:
            real = os.path.join(tmp, "scripts")
            os.makedirs(real)
            alias = os.path.join(tmp, "alias")
            try:
                os.symlink(real, alias)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable on this host: {exc}")
            with patch.dict(os.environ, {"HERMES_HOME": tmp}):
                self.assertIsNotNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(alias, "x.sh")
                    )
                )
                # Unrelated symlinked directories are not affected
                other = os.path.join(tmp, "other")
                os.makedirs(other)
                other_alias = os.path.join(tmp, "other_alias")
                os.symlink(other, other_alias)
                self.assertIsNone(
                    file_safety.get_messaging_write_block_error(
                        os.path.join(other_alias, "x.sh")
                    )
                )

    def test_relative_path_blocked_after_task_resolution(self):
        # A RELATIVE path that resolves (task-cwd aware) into a protected
        # root must be blocked — the file tools pre-resolve before calling
        # the classifier.
        with _bind_key("agent:main:telegram:dm:1"):
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "scripts"))
                with patch.dict(os.environ, {"HERMES_HOME": tmp}), \
                        patch("tools.file_tools.os.getcwd",
                              return_value=tmp):
                    resolved = str(
                        _resolve_path_for_task("scripts/x.sh", task_id="rel-test")
                    )
                    self.assertIn(tmp, resolved)
                    self.assertIsNotNone(
                        file_safety.get_messaging_write_block_error(
                            resolved
                        )
                    )


class TestToolWiring(unittest.TestCase):
    """End-to-end wiring of the guard through the actual file tools."""

    def _assert_write_blocked(self, path: str):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            out = write_file_tool(path, "payload", task_id="wiring-e2e")
        self.assertIsInstance(out, str)
        self.assertIn("execution-trusting", out)
        self.assertIn("Access denied", out)

    def test_write_file_tool_blocks_scripts(self):
        self._assert_write_blocked("~/.hermes/scripts/evil.sh")

    def test_write_file_tool_blocks_cron(self):
        self._assert_write_blocked("~/.hermes/cron/jobs.json")

    def test_write_file_tool_allows_unrelated(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            with tempfile.TemporaryDirectory() as tmp:
                target = os.path.join(tmp, "note.md")
                out = write_file_tool(target, "hello", task_id="wiring-e2e")
                self.assertIsInstance(out, str)
                self.assertNotIn("Access denied", out)
                self.assertNotIn("execution-trusting", out)

    def test_patch_tool_blocks_scripts(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            out = patch_tool(
                mode="replace",
                path="~/.hermes/scripts/evil.sh",
                old_string="x",
                new_string="y",
                task_id="wiring-e2e-patch",
            )
            self.assertIsInstance(out, str)
            self.assertIn("execution-trusting", out)
            self.assertIn("Access denied", out)

    def test_patch_tool_v4a_blocks_extracted_path(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            v4a = (
                "*** Begin Patch\n"
                "*** Update File: ~/.hermes/scripts/evil.sh\n"
                "@@\n"
                "-x\n"
                "+y\n"
                "*** End Patch\n"
            )
            out = patch_tool(
                mode="patch",
                path="/tmp/unrelated.md",
                patch=v4a,
                task_id="wiring-e2e-v4a",
            )
            self.assertIsInstance(out, str)
            self.assertIn("execution-trusting", out)


@contextlib.contextmanager
def _executor_boundary():
    """Run the guard through a ThreadPoolExecutor worker (the real dispatch).

    Subagent tool dispatch hops threads via propagate_context_to_thread
    (tools/thread_context.py); if context failed to propagate, the guard
    would see the default key and fail open — this test would catch it.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tools.thread_context import propagate_context_to_thread

    def _classify(path: str):
        return file_safety.get_messaging_write_block_error(path)

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            propagate_context_to_thread(_classify), "~/.hermes/scripts/evil.sh"
        )
        yield future.result(timeout=10)


class TestExecutorBoundary(unittest.TestCase):
    """The guard must survive the real executor dispatch hop (reviewer P1)."""

    def test_blocked_message_survives_executor_hop(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home(), \
                _executor_boundary() as err:
            self.assertIsNotNone(err)
            self.assertIn("telegram", err)

    def test_allowed_path_survives_executor_hop(self):
        with _bind_key("agent:main:telegram:dm:1"), _default_home():
            from concurrent.futures import ThreadPoolExecutor
            from tools.thread_context import propagate_context_to_thread

            def _classify(path: str):
                return file_safety.get_messaging_write_block_error(path)

            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    propagate_context_to_thread(_classify), "/tmp/note.md"
                )
                self.assertIsNone(fut.result(timeout=10))


@pytest.mark.windows_only
class TestWindowsNativeGuard(unittest.TestCase):
    """Native-Windows behaviour, run on windows-latest (tests-os.yml)."""

    def test_native_windows_roots_blocked(self):
        with _bind_key("agent:main:telegram:dm:1"), \
                patch.dict(os.environ, {
                    "HERMES_HOME": "C:/Users/t/AppData/Local/hermes"
                }):
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    "C:/Users/t/AppData/Local/hermes/scripts/x.ps1"
                )
            )
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    r"C:\Users\t\AppData\Local\hermes\scripts\x.ps1"
                )
            )
            # Case variants under the native normcase
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    "C:/Users/t/AppData/Local/HERMES/Scripts/x.ps1"
                )
            )
            # Extended-length prefix under the native OS path handling
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    r"\\?\C:\Users\t\AppData\Local\hermes\scripts\x.ps1"
                )
            )

    def test_native_windows_custom_home(self):
        with _bind_key("agent:main:telegram:dm:1"), \
                tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HERMES_HOME": tmp}):
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(tmp, "scripts", "x.ps1")
                )
            )


@pytest.mark.macos_only
class TestMacOSNativeGuard(unittest.TestCase):
    """Native-macOS behaviour, run on macos-latest (tests-os.yml).

    The macos CI host mounts case-insensitive APFS, so the case variant of
    the default home is the same directory and must be blocked through the
    REAL filesystem-backed probe.
    """

    def test_native_macos_case_variant_blocked(self):
        # conftest sandboxes HERMES_HOME into a per-test tempdir, which on
        # the case-insensitive CI APFS volume resolves case variants to the
        # same directory. Use it directly (no _default_home — that would
        # point back at the patched platform default) and clear the cached
        # darwin policy so it re-probes the sandbox home's parent.
        home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        file_safety._darwin_case_insensitive.cache_clear()
        with _bind_key("agent:main:telegram:dm:1"):
            variant = home[: -1] + home[-1].swapcase()
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(variant, "SCRIPTS", "evil.sh")
                )
            )
            self.assertIsNotNone(
                file_safety.get_messaging_write_block_error(
                    os.path.join(home, "Cron", "jobs.json")
                )
            )
