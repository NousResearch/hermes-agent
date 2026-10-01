"""Tests for hermes_bootstrap — Windows UTF-8 stdio shim.

The bootstrap module is imported at the top of every Hermes entry point
(hermes, hermes-agent, hermes-acp, gateway, batch_runner, cli.py).  It
fixes Python's Windows UTF-8 defaults so print("café") doesn't crash and
subprocess children inherit UTF-8 mode.

Key invariants covered by these tests:

  1. Windows: env vars get set, stdio reconfigured, non-ASCII print works
  2. POSIX: complete no-op (we don't touch LANG/LC_* or anything else)
  3. Idempotent: safe to call multiple times
  4. Respects user opt-out: if the user explicitly sets PYTHONUTF8=0 or
     PYTHONIOENCODING=something-else, we leave those alone
  5. Load order: every Hermes entry point imports hermes_bootstrap as its
     first non-docstring import (before anything that might do file I/O
     or print to stdout)
"""

from __future__ import annotations

import errno
import io
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# Import the module under test via an import-time side-effect check path.
# We need to be able to reset its state between tests, so we import it
# fresh in each test that manipulates _IS_WINDOWS.
def _fresh_import():
    """Return a freshly-imported hermes_bootstrap module.

    Drops any cached copy from sys.modules first so module-level code
    runs again and the platform check re-evaluates.
    """
    sys.modules.pop("hermes_bootstrap", None)
    import hermes_bootstrap  # noqa: WPS433
    return hermes_bootstrap


class TestWindowsBehavior:
    """Windows: the bootstrap does its job."""

    @pytest.mark.platforms("windows")
    def test_env_vars_set_on_windows(self, monkeypatch):
        # Clear any pre-existing values and re-run bootstrap.
        monkeypatch.delenv("PYTHONUTF8", raising=False)
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        hb = _fresh_import()
        # Module-level apply_windows_utf8_bootstrap() ran during import.
        assert os.environ.get("PYTHONUTF8") == "1"
        assert os.environ.get("PYTHONIOENCODING") == "utf-8"
        assert hb._bootstrap_applied is True

    @pytest.mark.platforms("windows")
    def test_stdout_reconfigured_to_utf8_on_windows(self):
        # The live process's stdout should now be UTF-8 (the Hermes CLI
        # runs on Windows with a pytest console that's cp1252 by default).
        # If reconfigure succeeded, sys.stdout.encoding is 'utf-8'.
        _fresh_import()
        # pytest may capture stdout, which makes encoding check flaky —
        # so instead verify the reconfigure call succeeded on the real
        # stream by attempting the failure case.
        out = sys.stdout
        reconfigure = getattr(out, "reconfigure", None)
        if reconfigure is None:
            pytest.skip("pytest replaced sys.stdout with a non-reconfigurable stream")
        # After bootstrap, encoding should be utf-8 (or the reconfigure
        # skipped because pytest's capture already set it to utf-8).
        assert out.encoding.lower() in {"utf-8", "utf8"}, (
            f"stdout encoding is {out.encoding!r} — bootstrap should have "
            "reconfigured it to UTF-8"
        )

    @pytest.mark.platforms("windows")
    def test_child_process_inherits_utf8_mode(self):
        """A subprocess spawned from this process should inherit
        PYTHONUTF8=1 and be able to print non-ASCII to stdout."""
        _fresh_import()
        # Non-ASCII chars that would crash under cp1252: arrow, emoji.
        script = textwrap.dedent("""
            import sys
            print("em-dash \\u2014 arrow \\u2192 emoji \\U0001f680")
            sys.exit(0)
        """).strip()
        # Don't pass env= — let the child inherit os.environ, which
        # now contains PYTHONUTF8=1 courtesy of the bootstrap.
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"Child crashed printing non-ASCII despite UTF-8 bootstrap:\n"
            f"  stdout: {result.stdout!r}\n"
            f"  stderr: {result.stderr!r}"
        )
        decoded = result.stdout.decode("utf-8")
        assert "\u2014" in decoded
        assert "\u2192" in decoded
        assert "\U0001f680" in decoded


class TestUserOptOut:
    """If the user has explicitly set PYTHONUTF8 / PYTHONIOENCODING in
    their environment, we respect that (setdefault, not overwrite)."""

    @pytest.mark.platforms("windows")
    def test_user_pythonutf8_zero_preserved(self, monkeypatch):
        monkeypatch.setenv("PYTHONUTF8", "0")
        _fresh_import()
        assert os.environ["PYTHONUTF8"] == "0", (
            "bootstrap must not overwrite an explicit user setting"
        )



class TestPosixNoOp:
    """POSIX: zero behavior change.  We don't touch LANG, LC_*, or any
    stdio.  The goal is that Linux/macOS behave identically before and
    after this module is imported."""

    @pytest.mark.platforms("linux")
    def test_noop_on_posix_host(self, monkeypatch):
        """Even when imported, the bootstrap function must return False
        and leave env untouched on a POSIX host (``_IS_WINDOWS`` is
        genuinely False here — nothing is faked)."""
        hb = _fresh_import()
        # Reset the idempotence latch so the call below is not a no-op for
        # the wrong reason.
        hb._bootstrap_applied = False
        monkeypatch.delenv("PYTHONUTF8", raising=False)
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)

        result = hb.apply_windows_utf8_bootstrap()

        assert result is False
        assert "PYTHONUTF8" not in os.environ
        assert "PYTHONIOENCODING" not in os.environ
        assert hb._bootstrap_applied is False






class TestStdioReconfigureErrorHandling:
    """If sys.stdout/stderr/stdin have been replaced with streams that
    don't support reconfigure (e.g. by a test harness), the bootstrap
    must degrade gracefully rather than crash."""

    @pytest.mark.platforms("windows")
    def test_non_reconfigurable_stream_does_not_crash(self, monkeypatch):
        """Replace sys.stdout with a BytesIO (no reconfigure method),
        then run the bootstrap and make sure it doesn't raise.

        ``platforms("windows")``: forcing ``_IS_WINDOWS = True`` on Linux was the only
        thing that made the reconfigure block reachable — off Windows the
        bootstrap returns before touching stdio, so the test proved nothing
        about the guard it names.
        """
        hb = _fresh_import()
        hb._bootstrap_applied = False

        fake = io.BytesIO()  # no .reconfigure attribute
        monkeypatch.setattr(sys, "stdout", fake)
        try:
            # Must not raise.
            hb.apply_windows_utf8_bootstrap()
        except Exception as exc:
            pytest.fail(f"bootstrap raised on non-reconfigurable stdout: {exc}")



@pytest.mark.parametrize("path", [
    "hermes_cli/main.py", "run_agent.py", "acp_adapter/entry.py",
    "gateway/run.py", "batch_runner.py", "cli.py",
])
def test_entrypoint_executes_bootstrap_before_application_imports(tmp_path, path):
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    entry = tmp_path / "startup.py"
    entry.write_bytes((root / path).read_bytes())
    # Stop at the first application import, after executing the REAL bootstrap.
    # pm repair is the supported stdlib-only startup, so no update/service runs.
    program = r"""
import builtins, os, runpy, sys
root, entry = sys.argv[1:]
sys.path.insert(0, root)
sys.argv = [entry, 'pm', 'repair']
real_import = builtins.__import__
class Boundary(BaseException): pass
seen = []
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if globals and globals.get('__file__') == entry:
        if name == '__future__':
            return real_import(name, globals, locals, fromlist, level)
        if not seen:
            assert name == 'hermes_bootstrap', name
            module = real_import(name, globals, locals, fromlist, level)
            assert module._pm_repair is True
            assert module._bootstrap_applied is (sys.platform == 'win32')
            seen.append(name)
            return module
        raise Boundary()
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
try:
    runpy.run_path(entry, run_name='__main__')
except Boundary:
    assert seen == ['hermes_bootstrap']
    print('bootstrap-before-app')
else:
    raise AssertionError('entrypoint never reached the application import boundary')
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(root), str(entry)],
                            cwd=tmp_path, env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bootstrap-before-app"


# "any": the OS lanes select only platforms-marked tests, and Windows is where hermes_cli's
# stdio repair fires on a cp1252 pipe.
@pytest.mark.platforms("any")
def test_library_imports_of_dual_use_entry_modules_stay_side_effect_free(tmp_path):
    # The codex hermes-tools MCP server and the compute host are ``python -m`` entry points
    # that agent/gateway code and tests also import; there the bootstrap exported TMPDIR and
    # HERMES_SCRATCH_DIR into every importer (gateway.relay's read-only routing included).
    code = textwrap.dedent("""
        import json, os, sys
        before = dict(os.environ)
        import agent.auxiliary_client, gateway.relay, tui_gateway.compute_host  # noqa: F401
        changed = sorted(k for k in before.keys() | os.environ.keys() if before.get(k) != os.environ.get(k))
        print(json.dumps({"bootstrapped": "hermes_bootstrap" in sys.modules, "changed": changed}))
    """)
    repo = Path(__file__).resolve().parents[1]
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR") if k in os.environ}
    env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "PYTHONPATH": str(repo),
                "HERMES_HOME": str(tmp_path / "home"), "PYTHONDONTWRITEBYTECODE": "1"})
    child = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines()[-1] == '{"bootstrapped": false, "changed": []}'


def test_pre_pm_editable_venv_reaches_pm_through_the_bootstrap(tmp_path):
    """A venv editable-installed from a pre-PM tree must still start the PM-era tree.

    setuptools' flat-layout editable finder maps only the top-level names it saw at
    install time (no ``pm``) and never puts the checkout on ``sys.path``. The console
    script imports ``hermes_cli`` first, then ``hermes_cli.main`` imports the bootstrap;
    both must load, and the bootstrap must reach ``pm``, or PM adoption never runs.
    """
    root = Path(__file__).resolve().parents[1]
    program = r"""
import importlib.util, os, sys
from importlib.abc import MetaPathFinder
root = sys.argv[1]
class PrePMEditableFinder(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'hermes_cli':
            pkg = os.path.join(root, 'hermes_cli')
            return importlib.util.spec_from_file_location(
                name, os.path.join(pkg, '__init__.py'), submodule_search_locations=[pkg])
        if name == 'hermes_bootstrap':
            return importlib.util.spec_from_file_location(name, os.path.join(root, 'hermes_bootstrap.py'))
        return None
sys.meta_path.append(PrePMEditableFinder())
sys.argv = ['hermes', 'pm', 'repair']
import hermes_cli
import hermes_bootstrap
assert hermes_bootstrap._pm_repair is True
print('reached-pm')
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(root)],
                            cwd=tmp_path, env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "reached-pm"


@pytest.mark.parametrize("path", [
    "hermes_cli/main.py", "run_agent.py", "acp_adapter/entry.py",
    "gateway/run.py", "batch_runner.py", "cli.py",
])
@pytest.mark.parametrize("bootstrap,expected", [
    (None, "proceeded"),
    ("import hermes_missing_dependency_probe\n", "raised hermes_missing_dependency_probe"),
])
def test_entrypoint_tolerates_only_an_absent_bootstrap(tmp_path, path, bootstrap, expected):
    """The entry-point guard covers a bootstrap a partial update left unregistered.

    A bootstrap that exists but cannot import its own dependencies must surface:
    swallowing that skipped PM activation silently and the tree ran on stale deps.
    """
    root = Path(__file__).resolve().parents[1]
    entry = tmp_path / "startup.py"
    entry.write_bytes((root / path).read_bytes())
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    if bootstrap is not None:
        (fake_root / "hermes_bootstrap.py").write_text(bootstrap)
    program = r"""
import builtins, runpy, sys
fake_root, entry = sys.argv[1:]
sys.path.insert(0, fake_root)
real_import = builtins.__import__
class Boundary(BaseException): pass
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if globals and globals.get('__file__') == entry and name not in ('__future__', 'hermes_bootstrap'):
        raise Boundary()
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
try:
    runpy.run_path(entry, run_name='__main__')
except Boundary:
    print('proceeded')
except ModuleNotFoundError as exc:
    print('raised', exc.name)
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(fake_root), str(entry)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


class TestHardenImportPath:
    """harden_import_path() must keep a same-named package in the launch
    directory from shadowing Hermes's own top-level modules — covering both
    the relative ('' / '.') and absolute-path forms the cwd can take on
    sys.path (issue #51286)."""

    def _run(self, hb, path_seed, env=None):
        original = sys.path[:]
        original_env = os.environ.get("HERMES_PYTHON_SRC_ROOT")
        try:
            sys.path[:] = path_seed
            if env is not None:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = env
            elif "HERMES_PYTHON_SRC_ROOT" in os.environ:
                del os.environ["HERMES_PYTHON_SRC_ROOT"]
            hb.harden_import_path(src_root="/opt/hermes")
            return sys.path[:]
        finally:
            sys.path[:] = original
            if original_env is None:
                os.environ.pop("HERMES_PYTHON_SRC_ROOT", None)
            else:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = original_env

    def test_relative_cwd_forms_removed(self):
        hb = _fresh_import()
        result = self._run(hb, ["", ".", "/opt/hermes", "/usr/lib/python"])
        assert "" not in result
        assert "." not in result

    def test_src_root_forced_to_front(self):
        hb = _fresh_import()
        result = self._run(hb, ["", "/opt/hermes", "/usr/lib/python"])
        assert result[0] == "/opt/hermes"

    def test_absolute_cwd_path_loses_to_src_root(self):
        # The real #51286 bug: the launch dir is present as its own absolute
        # path (venv activation / a project on PYTHONPATH), ahead of the
        # Hermes root.  The guard must relocate Hermes to the front.
        hb = _fresh_import()
        result = self._run(hb, ["/home/user/tg-ws-proxy", "/opt/hermes"])
        assert result[0] == "/opt/hermes"
        # The cwd absolute path may still appear (it can hold legit deps),
        # but only AFTER the Hermes root.
        assert result.index("/opt/hermes") < result.index("/home/user/tg-ws-proxy")


    def test_env_var_used_when_no_arg(self):
        hb = _fresh_import()
        original = sys.path[:]
        original_env = os.environ.get("HERMES_PYTHON_SRC_ROOT")
        try:
            sys.path[:] = ["", "/cwd/proj", "/usr/lib"]
            os.environ["HERMES_PYTHON_SRC_ROOT"] = "/env/hermes"
            hb.harden_import_path()
            assert sys.path[0] == "/env/hermes"
        finally:
            sys.path[:] = original
            if original_env is None:
                os.environ.pop("HERMES_PYTHON_SRC_ROOT", None)
            else:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = original_env



class TestEnableWindowsVt:
    """Hermes prints raw SGR codes; a conhost console renders them only with VT on."""

    @pytest.mark.platforms("windows")
    def test_turns_vt_on_for_a_console_that_has_it_off(self):
        # A fresh console via CREATE_NEW_CONSOLE is the installer's situation:
        # a real conhost whose output handle starts without VT processing.
        script = textwrap.dedent("""
            import ctypes, msvcrt, sys
            from ctypes import wintypes
            sys.path.insert(0, sys.argv[1])
            import hermes_bootstrap
            kernel32 = ctypes.WinDLL("kernel32")
            handle = msvcrt.get_osfhandle(sys.stdout.fileno())
            mode = wintypes.DWORD()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                sys.exit(4)
            kernel32.SetConsoleMode(handle, mode.value & ~0x0004)
            ok = hermes_bootstrap.enable_windows_vt()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            sys.exit(0 if ok and mode.value & 0x0004 else 3)
        """).strip()
        root = str(Path(__file__).resolve().parents[1])
        result = subprocess.run(
            [sys.executable, "-c", script, root],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            timeout=120,
        )
        if result.returncode == 4:
            pytest.skip("this session cannot create a console")
        assert result.returncode == 0

    @pytest.mark.platforms("windows")
    def test_leaves_non_console_handles_and_colour_alone(self, tmp_path, monkeypatch):
        # Redirected output must neither fail nor flip Hermes to NO_COLOR.
        monkeypatch.delenv("NO_COLOR", raising=False)
        import hermes_bootstrap

        with open(tmp_path / "out.txt", "w", encoding="utf-8") as stream:
            assert hermes_bootstrap.enable_windows_vt([stream]) is True
        assert "NO_COLOR" not in os.environ


class TestSuppressPlatformVerConsole:
    """suppress_platform_ver_console: stub applied on Windows, no-op on POSIX."""


    @pytest.mark.platforms("windows")
    def test_stub_applied_when_windows(self):
        # Faking _IS_WINDOWS on Linux asserted only that the stub was
        # installed; the reason it exists — ``platform.win32_ver()`` shelling
        # out ``cmd /c ver`` — has no counterpart off Windows.
        import platform
        hb = _fresh_import()
        original = getattr(platform, "_syscmd_ver", None)
        try:
            hb.suppress_platform_ver_console()
            stubbed = platform._syscmd_ver
            assert stubbed is not original
            # Stub returns its inputs — win32_ver()'s documented fallback path.
            assert stubbed("s", "r", "v") == ("s", "r", "v")
            # No-arg call (how Lib/platform.py invokes it in the fallback
            # probe) must not raise — the rejected PR #69522 wrapper
            # TypeError'd here.
            assert stubbed() == ("", "", "")
        finally:
            if original is not None:
                platform._syscmd_ver = original


class TestHappyEyeballsSocketConnect:
    """Importing the bootstrap races IPv6/IPv4 for every sync connect in the process (#114265)."""

    def test_import_routes_http_client_and_urllib3_connects_through_the_racer(self):
        import http.client

        import urllib3
        import urllib3.util.connection as urllib3_connection

        hb = _fresh_import()
        assert socket.create_connection.__module__ == hb.__name__
        # urllib3 keeps its own serial connect walker; it is patched once imported (lazily).
        assert getattr(urllib3_connection.create_connection, "_hermes_happy_eyeballs", False)
        # Re-importing the bootstrap (or importing it after urllib3) never wraps the racer twice.
        racer = socket.create_connection
        _fresh_import()
        assert socket.create_connection is racer
        # The bootstrap must not pay urllib3's import (~50 ms) on every process start: a fresh
        # interpreter gets the patch the moment urllib3 loads, not before.
        subprocess.run([sys.executable, "-c", textwrap.dedent("""
            import sys, hermes_bootstrap
            assert "urllib3" not in sys.modules, "bootstrap imported urllib3 eagerly"
            import urllib3.util.connection as c
            assert c.create_connection._hermes_happy_eyeballs
        """)], check=True, cwd=str(Path(hb.__file__).parent), timeout=60)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        http_conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        urllib3_conn = urllib3.connection.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            http_conn.connect()
            urllib3_conn.connect()  # exercises the socket_options kwarg of the urllib3 racer
            assert http_conn.sock.getpeername()[1] == port
            assert urllib3_conn.sock.getpeername()[1] == port
        finally:
            http_conn.close()
            urllib3_conn.close()
            listener.close()

    def test_installed_racer_wins_ipv4_while_ipv6_hangs(self, monkeypatch):
        hb = _fresh_import()
        clock = [0.0]
        sockets = []

        class FakeSocket:
            def __init__(self, family, socktype, proto):
                self.family = family
                self.closed = False
                self.timeout = None
                sockets.append(self)

            def setsockopt(self, *_args):
                pass

            def setblocking(self, _blocking):
                pass

            def settimeout(self, timeout):
                self.timeout = timeout

            def connect_ex(self, _address):
                return errno.EINPROGRESS if self.family == socket.AF_INET6 else 0

            def close(self):
                self.closed = True

        class FakeSelector:
            def register(self, *_args):
                pass

            def unregister(self, *_args):
                pass

            def select(self, timeout):
                clock[0] += timeout or 0.0  # the v6 attempt never completes
                return []

            def close(self):
                pass

        monkeypatch.setattr(hb.socket, "getaddrinfo", lambda *_a, **_k: [
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("2001:db8::1", 443, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.1", 443)),
        ])
        monkeypatch.setattr(hb.socket, "socket", FakeSocket)
        monkeypatch.setattr(hb.selectors, "DefaultSelector", FakeSelector)
        monkeypatch.setattr(hb.time, "monotonic", lambda: clock[0])

        # http.client passes the module timeout sentinel through positionally.
        winner = socket.create_connection(("example.com", 443), socket._GLOBAL_DEFAULT_TIMEOUT, None)

        assert winner.family == socket.AF_INET
        assert winner.timeout is None  # sentinel resolves to the process default, like stock
        assert clock[0] == hb._HAPPY_EYEBALLS_DELAY_SECONDS
        assert sockets[0].closed is True and sockets[1] is winner

    def test_racer_bug_raises_instead_of_falling_back_to_the_serial_walk(self, monkeypatch):
        """A non-OSError from the racer is a bug in the racer, not a network outcome: it must
        surface, never silently reroute the connect through the serial stock walker (which
        would reintroduce the exact stall the racer exists to remove). OSError still means
        "every candidate failed" and propagates unchanged."""
        import urllib3.util.connection as urllib3_connection

        _fresh_import()

        def boom(*_args, **_kwargs):
            raise RuntimeError("racer bug")

        for racer in (socket.create_connection, urllib3_connection.create_connection):
            assert getattr(racer, "_hermes_happy_eyeballs", False)
            monkeypatch.setitem(racer.__globals__, "_happy_eyeballs_create_connection", boom)
            with pytest.raises(RuntimeError, match="racer bug"):
                racer(("127.0.0.1", 1), 1.0)


@pytest.mark.skipif(
    sys.platform != "linux" or not (os.confstr("CS_GNU_LIBC_VERSION") or "").startswith("glibc"),
    reason="environ array lifetime is a glibc property")
class TestNeverFreeEnviron:
    """A new ``os.environ`` name must never free the ``environ`` array another thread may be
    walking in C ``getenv`` (getaddrinfo, OpenSSL) — glibc < 2.41 did, and the tui_gateway
    segfaulted when ``session.create`` added names during the picker-prewarm fetch."""

    def test_arrays_superseded_by_new_names_stay_intact(self):
        # Fresh interpreter: the test process's own environ history must not matter.
        subprocess.run([sys.executable, "-c", textwrap.dedent("""
            import ctypes, os, hermes_bootstrap
            environ = ctypes.c_void_p.in_dll(ctypes.CDLL(None), "environ")
            def snapshot(addr):
                array, out = ctypes.cast(addr, ctypes.POINTER(ctypes.c_void_p)), []
                while array[len(out)]:
                    out.append(array[len(out)])
                return out
            seen, junk = {}, []
            for i in range(80):
                seen.setdefault(environ.value, snapshot(environ.value))
                os.environ[f"HERMES_ENVIRON_PROBE_{i}"] = "1"
                junk.append(os.urandom(40))  # interleave heap chunks so a realloc must move
            assert len(seen) > 1, "environ never moved; the probe proves nothing"
            # Compare raw words only: dereferencing a freed slot would crash the probe itself.
            for addr, entries in seen.items():
                words = ctypes.cast(addr, ctypes.POINTER(ctypes.c_void_p))
                assert [words[i] for i in range(len(entries))] == entries, "a superseded environ array was freed"
            assert os.environ["HERMES_ENVIRON_PROBE_79"] == "1"
        """)], check=True, cwd=str(Path(__file__).resolve().parents[1]), timeout=60)

    def test_concurrent_writers_lose_no_name_in_the_c_environ(self):
        # Unserialized writers each copied the live array; the later publish dropped the others' names.
        subprocess.run([sys.executable, "-c", textwrap.dedent("""
            import ctypes, os, sys, threading, hermes_bootstrap
            getenv = ctypes.CDLL(None).getenv
            getenv.restype, getenv.argtypes = ctypes.c_char_p, [ctypes.c_char_p]
            sys.setswitchinterval(1e-6)
            batches = [[f"HERMES_ENVIRON_RACE_{t}_{i}" for i in range(200)] for t in range(8)]
            barrier = threading.Barrier(len(batches))
            def write(batch):
                barrier.wait()
                for name in batch:
                    os.environ[name] = name
            threads = [threading.Thread(target=write, args=(b,)) for b in batches]
            [t.start() for t in threads]; [t.join() for t in threads]
            lost = [n for b in batches for n in b if getenv(n.encode()) != n.encode()]
            assert not lost, f"{len(lost)}/1600 names in os.environ but missing from C getenv"
        """)], check=True, cwd=str(Path(__file__).resolve().parents[1]), timeout=120)

    def test_set_del_churn_of_the_same_names_keeps_memory_bounded(self):
        # Kanban ticks, the spinner pause and _restore_env set and pop the same names forever.
        subprocess.run([sys.executable, "-c", textwrap.dedent("""
            import os, tracemalloc, hermes_bootstrap
            def churn(cycles):
                for _ in range(cycles):
                    for k in range(4):
                        os.environ[f"HERMES_ENVIRON_CHURN_{k}"] = "1"
                    for k in range(4):
                        del os.environ[f"HERMES_ENVIRON_CHURN_{k}"]
            churn(50)
            tracemalloc.start()
            before = tracemalloc.get_traced_memory()[0]
            churn(5000)
            grown = tracemalloc.get_traced_memory()[0] - before
            assert grown < 64 * 1024, f"20k set/del of 4 names grew the heap by {grown} bytes"
        """)], check=True, cwd=str(Path(__file__).resolve().parents[1]), timeout=120)
