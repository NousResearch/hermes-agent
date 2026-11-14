"""Regression test: the WhatsApp bridge spawn must survive a no-breakaway job object.

A gateway relaunched by the updater/service runs inside a Windows job object that
does not set JOB_OBJECT_LIMIT_BREAKAWAY_OK. ``_spawn_bridge_process`` asked for
``CREATE_BREAKAWAY_FROM_JOB`` (part of ``windows_detach_flags()``), so every
spawn raised ``PermissionError: [WinError 5] Access is denied`` and WhatsApp
stayed ``disconnected``, retrying every 300 s indefinitely (44 consecutive
failures observed on a live gateway after an in-app update).

The fix: catch ``PermissionError`` and retry once with
``windows_detach_flags_without_breakaway()``, as the ``windows_detach_flags``
docstring already prescribes for its callers. Any other error must propagate
unchanged — a missing node binary or a bad argv must not be retried blindly.

Written with ``unittest`` so it also runs without pytest:
    python tests/gateway/test_whatsapp_spawn_fallback.py
"""

from __future__ import annotations

import unittest
from unittest import mock

from hermes_cli._subprocess_compat import (
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
)
from plugins.platforms.whatsapp import adapter


class _FakeProc:
    """Stand-in for the Popen object; only ``pid`` is read by the caller."""

    pid = 4242


class SpawnBridgeFallbackTest(unittest.TestCase):
    def _run(self, outcomes: list) -> list:
        """Drive _spawn_bridge_process; ``outcomes`` is consumed one call at a time."""
        flags: list = []

        def fake_spawn_owned(argv, **kwargs):
            flags.append(kwargs.get("creationflags"))
            outcome = outcomes[len(flags) - 1]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        with mock.patch.object(adapter, "spawn_owned", fake_spawn_owned):
            self.proc = adapter._spawn_bridge_process(
                ["node", "bridge.js"], None, {"PATH": ""}
            )
        return flags

    def test_falls_back_when_breakaway_is_denied(self) -> None:
        flags = self._run([PermissionError(5, "Access is denied"), _FakeProc()])
        self.assertEqual(self.proc.pid, 4242)
        self.assertEqual(flags, [windows_detach_flags(), windows_detach_flags_without_breakaway()])

    def test_fallback_is_used_only_once(self) -> None:
        with self.assertRaises(PermissionError):
            self._run([PermissionError(5, "Access is denied"),
                       PermissionError(5, "Access is denied")])

    def test_other_errors_are_not_swallowed(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run([RuntimeError("boom")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
