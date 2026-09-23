"""Regression tests: Hermes gateway running in Windows Session 0 must not
hard-fail when cua-driver refuses to spawn ``mcp`` in a non-interactive
session.

Issue #94756: ``cua-driver mcp`` rejects a Session-0 child with
"Cua Driver requires an interactive Windows user session: running in
Windows Session 0" before MCP initialization completes, so a long-lived
gateway installed as a Windows Scheduled Task (or service) cannot
initialize the built-in ``computer_use`` tool.

Hermes policy: when Session 0 is detected on Windows, the ``computer_use``
toolset must

  1. surface a clear, recoverable ``RuntimeError`` (instead of letting the
     underlying cua-driver message leak) — so the user can decide whether
     to switch transports, run the gateway in an interactive session, or
     accept the offline posture.
  2. honor the ``computer_use.session0_transport`` config knob:
       ``auto``  (default) → CLI transport for Session 0; MCP otherwise
       ``cli``             → always use the brokered CLI transport (skip MCP)
       ``mcp``             → force MCP (will fail closed in Session 0)
       ``off``             → refuse ``computer_use`` in Session 0 with an
                            actionable ``ComputerUseSession0UnavailableError``
  3. fail closed on the brokered CLI fallback path if the daemon is not
     reachable — never silently fall through to a Session-0 ``mcp`` retry.

These tests pin that policy. They are platform-independent: they patch
``sys.platform`` and the kernel32 seam so a Session-0 scenario can be
exercised on any CI host, the same way the existing ``test_cua_no_overlay``
suite tests macOS auto-detect off-host.
"""

from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import cua_backend
from tools.computer_use.cua_backend import (
    ComputerUseSession0UnavailableError,
    _cua_session0_transport,
    _is_windows_session_zero,
    _resolve_session0_transport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> cua_backend._CuaDriverSession:
    # call_tool() touches bridge state via self._bridge.run() and the CLI
    # fallback via self._call_tool_via_cli(); bypass __init__ and let each
    # test inject the surface it needs.
    return object.__new__(cua_backend._CuaDriverSession)


def _fake_completed_process(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


@pytest.fixture(autouse=True)
def _isolate_session0_cache(monkeypatch):
    """Per-test isolated state for the Session-0 policy helpers.

    The lru_cache on the Windows kernel32 probe plus the cached transport
    decision leak state across tests. Reset them around every test so the
    suite never couples one assertion to a previous patch.
    """
    monkeypatch.delenv("HERMES_CUA_SESSION0_TRANSPORT", raising=False)
    # The transport helper reads computer_use.session0_transport from the
    # config snapshot — clear it via the patched load_config() in each test.
    yield
    # lru_cache only attaches cache_clear() to the wrapped function after
    # the first call; some tests (e.g. non-Windows patching) never invoke
    # the cached path so the attribute is missing. Both states must work.
    for cached in (
        cua_backend._resolve_session0_transport,
        cua_backend._is_windows_session_zero,
    ):
        cache_clear = getattr(cached, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


# ---------------------------------------------------------------------------
# Session-0 detection
# ---------------------------------------------------------------------------


class TestWindowsSessionZeroDetection:
    """Pinned contract for the Session-0 detector.

    The detector must return:
      - ``True`` when running in Windows Session 0
      - ``False`` when running in an interactive Windows session
      - ``None`` on non-Windows or when the kernel32 seam is unavailable
    """

    def test_returns_none_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(cua_backend.sys, "platform", "linux")
        assert _is_windows_session_zero() is None

    def test_returns_none_when_kernel32_seam_unavailable(self, monkeypatch):
        # If the platform probe raises (no ctypes, no kernel32), the
        # detector must return None instead of crashing — the transport
        # selector then defaults to MCP, which is correct behavior on a
        # host we cannot introspect.
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")

        def _boom():
            raise OSError("kernel32 unavailable")

        monkeypatch.setattr(cua_backend, "_windows_session_id", _boom)
        assert _is_windows_session_zero() is None

    def test_returns_true_for_session_zero(self, monkeypatch):
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(cua_backend, "_windows_session_id", lambda: 0)
        assert _is_windows_session_zero() is True

    def test_returns_false_for_interactive_session(self, monkeypatch):
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(cua_backend, "_windows_session_id", lambda: 1)
        assert _is_windows_session_zero() is False

    def test_windows_session_id_is_cached(self, monkeypatch):
        """The kernel32 probe must be cached to avoid paying the syscall
        cost on every computer_use call. Verify the cache via repeated
        calls in a single test.
        """
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        counter = {"calls": 0}

        def _probe():
            counter["calls"] += 1
            return 0

        monkeypatch.setattr(cua_backend, "_windows_session_id", _probe)
        for _ in range(5):
            assert _is_windows_session_zero() is True
        # lru_cache(maxsize=1) collapses repeated calls to a single probe.
        assert counter["calls"] == 1


class TestWindowsSessionIdIsProcessScoped:
    """``_windows_session_id`` must name THIS process, never the console.

    ``WTSGetActiveConsoleSessionId`` takes no arguments, so its return value
    can only be a property of the machine — the session attached to the
    physical console — and cannot answer "which session is this process in".
    On the topology this policy exists for (gateway in Session 0, owner
    logged in interactively) it reports the *owner's* session, so
    ``_is_windows_session_zero()`` answers False, ``auto`` picks MCP, and
    cua-driver rejects the spawn with the exact Session-0 error being worked
    around here.

    The console session id is also mutable (RDP connect/disconnect, Fast
    User Switching) while a process's session id is fixed at creation, which
    is what makes it a valid ``lru_cache`` key.

    These tests drive the ctypes seam with a fake kernel32 so they run on any
    CI host, and fail loudly if the console API is reached for again.
    """

    @staticmethod
    def _fake_kernel32(session_id: int = 0, ok: bool = True):
        calls = {"pid": 0, "session": 0}

        class _Fn:
            def __init__(self, fn):
                self._fn = fn
                self.restype = None
                self.argtypes = None

            def __call__(self, *args):
                return self._fn(*args)

        class _Kernel32:
            def __init__(self):
                self.GetCurrentProcessId = _Fn(self._current_pid)
                self.ProcessIdToSessionId = _Fn(self._pid_to_session)

            @staticmethod
            def _current_pid():
                calls["pid"] += 1
                return 4242

            @staticmethod
            def _pid_to_session(pid, out):
                assert pid == 4242, pid
                calls["session"] += 1
                if not ok:
                    return 0
                out._obj.value = session_id
                return 1

            def __getattr__(self, name):
                # Regression guard: any console-scoped or unknown export
                # means the detector stopped being process-scoped.
                raise AssertionError(
                    f"Session-0 probe must not call kernel32.{name}"
                )

        return _Kernel32(), calls

    def _install(self, monkeypatch, **kwargs):
        fake, calls = self._fake_kernel32(**kwargs)
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(
            ctypes, "WinDLL", lambda *a, **kw: fake, raising=False
        )
        return calls

    def test_reports_session_zero_for_this_process(self, monkeypatch):
        calls = self._install(monkeypatch, session_id=0)
        assert cua_backend._windows_session_id() == 0
        assert calls["session"] == 1
        assert calls["pid"] == 1

    def test_reports_interactive_session_for_this_process(self, monkeypatch):
        calls = self._install(monkeypatch, session_id=1)
        assert cua_backend._windows_session_id() == 1
        assert calls["session"] == 1

    def test_returns_none_when_the_query_fails(self, monkeypatch):
        self._install(monkeypatch, session_id=0, ok=False)
        assert cua_backend._windows_session_id() is None


# ---------------------------------------------------------------------------
# Transport config knob
# ---------------------------------------------------------------------------


class TestSession0TransportConfig:
    """Pinned contract for ``computer_use.session0_transport`` resolution.

    Recognized values: ``auto`` (default), ``cli``, ``mcp``, ``off``.
    Unknown values fall closed to ``auto``.
    """

    def test_default_is_auto(self):
        # No config block, no env override → auto.
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _cua_session0_transport() == "auto"

    def test_explicit_cli(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"session0_transport": "cli"}},
        ):
            assert _cua_session0_transport() == "cli"

    def test_explicit_mcp(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"session0_transport": "mcp"}},
        ):
            assert _cua_session0_transport() == "mcp"

    def test_explicit_off(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"session0_transport": "off"}},
        ):
            assert _cua_session0_transport() == "off"

    def test_unknown_falls_closed_to_auto(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"session0_transport": "telemetry"}},
        ):
            assert _cua_session0_transport() == "auto"

    def test_unreadable_config_falls_closed_to_auto(self):
        # Config crashes (read failure) MUST NOT take down the Session-0
        # policy; default to ``auto`` so the host's session topology
        # decides the transport.
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert _cua_session0_transport() == "auto"

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv("HERMES_CUA_SESSION0_TRANSPORT", "cli")
        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"session0_transport": "off"}},
        ):
            # Env override always wins — same precedence model as
            # _CUA_DRIVER_CMD_ENV / HERMES_CUA_TELEMETRY.
            assert _cua_session0_transport() == "cli"


class TestResolveSession0Transport:
    """Pinned contract for the auto-detect verdict.

    ``_resolve_session0_transport`` collapses Session-0 detection +
    ``session0_transport`` config + env override into a single
    transport decision. It is lru_cached so it can be called repeatedly
    from hot paths without paying for re-detection.

      auto + non-Windows       → ``mcp``  (interactive Hermes is the norm)
      auto + Windows+Session1+ → ``mcp``
      auto + Windows+Session0+ → ``cli``  (broker via existing daemon)
      cli (any host)           → ``cli``
      mcp (any host)           → ``mcp``
      off  (any host)          → ``off``
    """

    def test_auto_on_non_windows_resolves_to_mcp(self, monkeypatch):
        monkeypatch.setattr(cua_backend.sys, "platform", "linux")
        monkeypatch.setattr(cua_backend, "_is_windows_session_zero", lambda: None)
        cua_backend._resolve_session0_transport.cache_clear()
        assert _resolve_session0_transport() == "mcp"

    def test_auto_in_windows_interactive_resolves_to_mcp(self, monkeypatch):
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(cua_backend, "_is_windows_session_zero", lambda: False)
        cua_backend._resolve_session0_transport.cache_clear()
        assert _resolve_session0_transport() == "mcp"

    def test_auto_in_windows_session0_resolves_to_cli(self, monkeypatch):
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(cua_backend, "_is_windows_session_zero", lambda: True)
        cua_backend._resolve_session0_transport.cache_clear()
        assert _resolve_session0_transport() == "cli"

    def test_off_is_honored_regardless_of_session(self, monkeypatch):
        monkeypatch.setenv("HERMES_CUA_SESSION0_TRANSPORT", "off")
        monkeypatch.setattr(cua_backend.sys, "platform", "linux")
        monkeypatch.setattr(cua_backend, "_is_windows_session_zero", lambda: None)
        cua_backend._resolve_session0_transport.cache_clear()
        assert _resolve_session0_transport() == "off"

    def test_cli_overrides_session0_even_on_interactive(self, monkeypatch):
        monkeypatch.setenv("HERMES_CUA_SESSION0_TRANSPORT", "cli")
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(cua_backend, "_is_windows_session_zero", lambda: False)
        cua_backend._resolve_session0_transport.cache_clear()
        assert _resolve_session0_transport() == "cli"


# ---------------------------------------------------------------------------
# Recoverable error surface
# ---------------------------------------------------------------------------


class TestSession0ErrorSurface:
    """The Session-0 error must be a distinct, catchable exception so the
    CLI / doctor can render an actionable message instead of the raw
    ``cua-driver mcp`` text."""

    def test_distinct_exception_type(self):
        # ComputerUseSession0UnavailableError must NOT inherit from a
        # generic Hermes-only base class — third-party callers may catch
        # RuntimeError directly, so the type must remain a RuntimeError
        # subclass AND be importable by name.
        err = ComputerUseSession0UnavailableError(
            "computer_use is unavailable in Windows Session 0 (interactive "
            "session 1 expected)."
        )
        assert isinstance(err, RuntimeError)

    def test_exception_carries_actionable_message(self):
        # Use the actual _session0_unavailable_error helper — the message
        # is the contract surface CLI / doctor / gateway health checks
        # surface to the user.
        from tools.computer_use.cua_backend import _session0_unavailable_error

        err = _session0_unavailable_error()
        message = str(err)
        # Every suggested fix from the issue (#94756) must show up in the
        # message — otherwise the user is left re-reading the same cryptic
        # cua-driver text.
        assert "Windows Session 0" in message, message
        assert "interactive" in message, message
        assert "session0_transport=cli" in message, message
        assert "session0_transport=off" in message, message
        assert "cua-driver" in message, message


# ---------------------------------------------------------------------------
# Call-tool routing in Session-0 CLI mode
# ---------------------------------------------------------------------------


class TestCallToolRoutesThroughCLIInSession0:
    """In Session-0 + ``auto`` (resolves to ``cli``) the MCP handshake is
    skipped entirely — every call_tool invocation must funnel through
    the existing brokered CLI transport, never spawn ``cua-driver mcp``.
    """

    def test_session_routes_to_cli_when_transport_is_cli(self, monkeypatch):
        monkeypatch.setattr(
            cua_backend, "_resolve_session0_transport", lambda: "cli"
        )
        session = _make_session()
        # The transport resolver + CLI path is the only expected route.
        # ``call_tool`` short-circuits to the CLI fallback before touching
        # the bridge, so the bridge must NEVER be invoked in Session-0
        # CLI mode — proxy through _call_tool_via_cli instead.
        captured = {}

        def fake_cli(name, args, timeout):
            captured["name"] = name
            captured["args"] = args
            return {"data": "ok", "images": [], "isError": False}

        monkeypatch.setattr(session, "_call_tool_via_cli", fake_cli)
        bridge_mock = MagicMock()
        session._bridge = bridge_mock  # direct attribute injection (no __slots__)

        out = session.call_tool("list_windows", {"session": "x"}, timeout=2.0)
        assert out == {"data": "ok", "images": [], "isError": False}
        assert captured["name"] == "list_windows"
        # Bridge MUST NOT have been touched — no MCP handshake in Session-0.
        assert bridge_mock.run.call_count == 0

    def test_session_raises_recoverable_error_when_transport_is_off(self, monkeypatch):
        monkeypatch.setattr(
            cua_backend, "_resolve_session0_transport", lambda: "off"
        )
        session = _make_session()

        with pytest.raises(ComputerUseSession0UnavailableError):
            session.call_tool("list_windows", {}, timeout=1.0)

    def test_session_uses_bridge_when_transport_is_mcp(self, monkeypatch):
        monkeypatch.setattr(
            cua_backend, "_resolve_session0_transport", lambda: "mcp"
        )
        session = _make_session()
        # _started must be set so call_tool's MCP path takes the bridge
        # instead of re-entering start().
        session._started = True
        session._timeout_suspect = False
        session._LIFECYCLE_CALLS = cua_backend._CuaDriverSession._LIFECYCLE_CALLS
        session._TRANSPORT_REPLAY_SAFE_TOOLS = (
            cua_backend._CuaDriverSession._TRANSPORT_REPLAY_SAFE_TOOLS
        )

        bridge_run = MagicMock(return_value={"data": "ok", "images": [], "isError": False})
        bridge_mock = MagicMock(run=bridge_run)
        session._bridge = bridge_mock
        monkeypatch.setattr(session, "_call_tool_via_cli", MagicMock())

        out = session.call_tool("list_windows", {}, timeout=2.0)
        assert out["data"] == "ok"
        # MCP path took the bridge.
        assert bridge_run.call_count == 1


# ---------------------------------------------------------------------------
# End-to-end: Session-0 + config + recovered routing
# ---------------------------------------------------------------------------


class TestSession0EndToEnd:
    """Smoke test that stitches detection + config + routing together.

    This is the regression: before the fix, the user saw the raw cua-driver
    rejection. After the fix, in ``auto`` (default) mode, the call funnels
    to the brokered CLI transport without ever spawning ``cua-driver mcp``.
    """

    def test_session0_auto_routes_via_cli_and_runs_subprocess(self, monkeypatch):
        # Windows + Session 0 + auto → CLI transport.
        monkeypatch.setattr(cua_backend.sys, "platform", "win32")
        monkeypatch.setattr(cua_backend, "_is_windows_session_zero", lambda: True)

        session = _make_session()
        captured_run = {}

        def fake_subprocess_run(cmd, **kwargs):
            captured_run["cmd"] = cmd
            return _fake_completed_process(
                stdout=json.dumps({"tree_markdown": "ax-root"})
            )

        # The CLI transport resolves the binary through cua_backend_driver
        # (main's decomposition moved resolve_cua_driver_cmd out of the facade),
        # so patch the seam where _call_tool_via_cli actually reads it.
        monkeypatch.setattr(
            "tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd",
            lambda: "cua-driver",
        )
        monkeypatch.setattr("subprocess.run", fake_subprocess_run)
        bridge_mock = MagicMock()
        session._bridge = bridge_mock

        out = session.call_tool("list_windows", {"session": "abc"}, timeout=5.0)

        # The CLI transport must have run, and the MCP bridge MUST NOT.
        assert captured_run["cmd"][0] == "cua-driver"
        assert captured_run["cmd"][1] == "call"
        assert captured_run["cmd"][2] == "list_windows"
        assert bridge_mock.run.call_count == 0
        # Structured response flowed back through the same shape the MCP
        # path would have produced — callers stay transport-agnostic.
        assert out["data"] == "ax-root"
        assert out["isError"] is False


# ---------------------------------------------------------------------------
# config_defaults.py wiring
# ---------------------------------------------------------------------------


class TestConfigDefaultsWiring:
    """The new ``computer_use.session0_transport`` key must appear in
    config_defaults.py with a documented default so users discover it via
    `hermes config edit`."""

    def test_session0_transport_default_present(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        cu = DEFAULT_CONFIG.get("computer_use", {})
        assert "session0_transport" in cu, (
            "config_defaults.py must document `session0_transport` so users "
            "discover it via `hermes config edit` (issue #94756)."
        )
        assert cu["session0_transport"] == "auto"