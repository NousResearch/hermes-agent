"""Shared fixtures for the hermes-agent test suite.

Hermetic-test invariants enforced here (see AGENTS.md for rationale):

1. **No credential env vars.** All provider/credential-shaped env vars
   (ending in _API_KEY, _TOKEN, _SECRET, _PASSWORD, _CREDENTIALS, etc.)
   are unset before every test. Local developer keys cannot leak in.
2. **Isolated Hermes homes.** HERMES_HOME and the platform-default root
   resolve inside a per-test tempdir. Profile/root resolution can inspect
   both without probing production state. HOME and Path.home() stay intact
   for subprocesses and non-Hermes paths. Explicit test overrides still win.
3. **Deterministic runtime.** TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0.
4. **No HERMES_SESSION_* inheritance** — the agent's current gateway
   session must not leak into tests.

These invariants make the local test run match CI closely. Gaps that
remain (CPU count, worker count) are addressed by the canonical
test runner at ``scripts/run_tests.sh``.
"""

import asyncio
import atexit
import importlib
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Sandbox HERMES_HOME before ANY test module is imported ──────────────────
# `hermes_cli/main.py` calls `setup_logging()` at MODULE level, which resolves
# `get_hermes_home()` and attaches rotating file handlers to the ROOT logger.
# So merely importing it - which many test modules do, directly or
# transitively - points the whole pytest session's logging at the operator's
# real `~/.hermes/logs/agent.log` and `errors.log`.
#
# The `_isolate_env` fixture below also sandboxes HERMES_HOME, but fixtures run
# AFTER collection imports test modules, by which point the handler already
# holds an absolute path to the real log. Measured on a live install: 126
# warnings in the operator's agent.log came from test runs, not the gateway -
# enough noise to make genuine warnings hard to find.
#
# conftest is imported before any test module, so setting it here closes that
# window. The per-test fixture still applies for everything after import.
#
# ORDER MATTERS: the kanban write guard's deny-list (further down) must know
# the REAL Hermes root — capture it BEFORE the sandbox rewires HERMES_HOME,
# otherwise the deny-list would point at the throwaway tempdir and the guard
# would silently stop protecting the operator's actual ~/.hermes (#69385).
_PRE_SANDBOX_KANBAN_OVERRIDE = os.environ.get("HERMES_KANBAN_HOME", "").strip()
_PRE_SANDBOX_HERMES_HOME = os.environ.get("HERMES_HOME", "")

# Capture before any test fixture can override Path.home()/LOCALAPPDATA.
from hermes_constants import _get_platform_default_hermes_home

_NATIVE_HERMES_PARENT = _get_platform_default_hermes_home().parent


def _hermes_home_points_at_production(value: str) -> bool:
    """True when a pre-set HERMES_HOME resolves to the real production root.

    Gateway-launched shells (and developer shells that ``export
    HERMES_HOME=~/.hermes``) hand pytest the PRODUCTION home. Historically
    the session sandbox below honored any pre-set value, so collection-time
    imports (logging handlers, ``hermes_state.DEFAULT_DB_PATH``) froze paths
    inside the real ``~/.hermes`` — the escape vector that landed pytest
    fixture rows (chat-1 / wx-chat sessions, /tmp/pytest-of-* routing
    scopes) in the live state.db and flipped its journal mode under the
    WAL-mode gateway writer. Only a genuinely custom (non-production)
    HERMES_HOME is honored now.
    """
    if not value:
        return True
    try:
        # The platform-default root, not a hardcoded ``~/.hermes``: Windows installs live under
        # ``%LOCALAPPDATA%\hermes``, and a dev shell exporting that path used to be honored as
        # "custom", pinning import-time paths (``tui_gateway.server._hermes_home``) to the live
        # install so the state.db guard tripped on every store-touching test (#112692).
        from hermes_state_guard import _real_platform_state_root

        resolved = Path(value).expanduser().resolve()
        real_root = _real_platform_state_root() or (Path.home() / ".hermes").resolve()
    except Exception:
        return True
    if resolved == real_root:
        return True
    # Profile home directly under the production root: <root>/profiles/<name>
    return resolved.parent.name == "profiles" and resolved.parent.parent == real_root


# ``import hermes_bootstrap`` (transitively: any entry-point module) runs
# ``export_scratch_tmp_env()``, which points TMPDIR/TMP/TEMP at
# ``<HERMES_HOME>/cache/scratch`` unless a temp var is already set — and a
# Hermes-launched shell (agent terminal, ``hermes`` child) arrives with that
# redirect already applied, tagged by HERMES_SCRATCH_DIR. Either way the tmp
# root ends up INSIDE a guarded real home (the operator's, or a custom one
# honored below), so the session sandbox, pytest's basetemp and every
# ``tempfile`` default in the code under test trip the real-home guard. Strip
# Hermes' own export (the marker tells it apart from a user-set var), and
# relocate even user-set temp directories inside a guarded home. Pin the
# system default so the import-time hook stays a no-op. The parallel runner
# exports its own disk-backed TMPDIR anyway.
from hermes_constants import SCRATCH_DIR_MARKER_ENV, SCRATCH_TMP_ENV_VARS

_HERMES_EXPORTED_TMP = os.environ.get(SCRATCH_DIR_MARKER_ENV, "")
if _HERMES_EXPORTED_TMP:
    for _key in SCRATCH_TMP_ENV_VARS:
        if os.environ.get(_key, "").strip() == _HERMES_EXPORTED_TMP:
            del os.environ[_key]
    del os.environ[SCRATCH_DIR_MARKER_ENV]

from hermes_state_guard import _real_platform_state_root

_real_test_root = _real_platform_state_root() or (Path.home() / ".hermes").resolve()
_guarded_tmp_roots = [_real_test_root]
_custom_test_home = os.environ.get("HERMES_HOME")
if _custom_test_home:
    _guarded_tmp_roots.append(Path(_custom_test_home).expanduser().resolve())
for _key in SCRATCH_TMP_ENV_VARS:
    _value = os.environ.get(_key)
    if _value:
        _path = Path(_value).expanduser().resolve()
        if any(_path.is_relative_to(_root) for _root in _guarded_tmp_roots):
            del os.environ[_key]
tempfile.tempdir = None  # re-resolve after stripping guarded temp directories
os.environ.setdefault("TMPDIR", tempfile.gettempdir())

if _hermes_home_points_at_production(os.environ.get("HERMES_HOME", "")):
    _SESSION_HERMES_HOME = tempfile.mkdtemp(prefix="hermes-test-home-")
    os.environ["HERMES_HOME"] = _SESSION_HERMES_HOME
    # Marker for re-imported conftest module bodies (xdist workers exec this
    # file more than once): the second import sees the already-redirected
    # sandbox in the env and must not register it as a guarded "real" root.
    os.environ["HERMES_TEST_SANDBOX_HOME"] = _SESSION_HERMES_HOME
    atexit.register(shutil.rmtree, _SESSION_HERMES_HOME, True)

# PYTHONPYCACHEPREFIX is a bytecode-mirror escape hatch: when set (the
# bundled desktop app exports it as %LOCALAPPDATA%\hermes\pycache),
# importlib/pytest write .pyc files to <prefix>/<absolute source path>
# instead of next to the sources. Un-scrubbed, that mirror lands under
# the REAL hermes home and trips the real-home tripwire on any module
# imported after sandboxing (test_find_shell was the first to bite).
# Clear it so bytecode goes back beside the (already sandboxed) sources.
os.environ.pop("PYTHONPYCACHEPREFIX", None)
try:
    sys.pycache_prefix = None
except AttributeError:
    pass

# Subprocess-surviving isolation marker (#82770). PYTEST_CURRENT_TEST /
# PYTEST_VERSION are pytest's own vars, and tests that spawn children
# routinely rebuild the child env and strip them ("the subprocess must look
# like a real CLI") — which used to disarm hermes_state's live-DB guard in
# the child at the same moment the child lost the HERMES_HOME redirect.
# HERMES_TEST_ISOLATION is OUR marker: exported here (before any test module
# imports), inherited by every child by default, and honored by
# hermes_state_guard._running_under_pytest() as a test-context signal. A child
# that carries it and still resolves the production state.db fails hard.
# Tests that legitimately need a child to look like a non-test process AND
# open a real DB must export HERMES_STATE_DB_GUARD_BYPASS=1 in that child's
# env instead of stripping markers.
os.environ["HERMES_TEST_ISOLATION"] = os.environ.get("HERMES_HOME", "") or "1"

# Lazy-install kill-switch, set before any test module is imported. The per-test
# fixture below sets it too, but collection runs first: agent/bedrock_adapter.py
# calls lazy_deps.ensure() at import time, so collecting a file that imports it
# ran a real `uv pip install boto3` into the shared venv while other files raced
# on whether botocore was importable yet.
os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"

#: HERMES_HOME as it stood when conftest was imported - i.e. before any test
#: module could import code that configures logging. Recorded so the guard in
#: tests/test_log_isolation.py can assert the sandbox existed AT THAT MOMENT.
#: Reading os.environ from inside a test is useless here: the per-test
#: `_isolate_env` fixture has sandboxed it by then, so the check would pass
#: even with this block removed.
HERMES_HOME_AT_CONFTEST_IMPORT = os.environ.get("HERMES_HOME", "")

# ── Host-rendezvous isolation ───────────────────────────────────────────────
# ``gateway/host_rendezvous.py`` publishes ONE record per role per OS USER, in
# ``$HERMES_GATEWAY_LOCK_DIR`` else ``$XDG_STATE_HOME/hermes/gateway-locks`` —
# deliberately outside HERMES_HOME, because the host singleton spans profiles.
# Under the per-file parallel runner that directory is shared by ~40 pytest
# subprocesses: one test that boots a real gateway publishes a record, and every
# other file's lifecycle code then correctly attaches to a gateway that has
# nothing to do with it. Give each pytest PROCESS its own rendezvous dir.
#
# A caller-supplied value always wins (both here and in the per-test fixture
# below) — otherwise the documented override is a silent no-op.
HOST_LOCK_DIR_AT_CONFTEST_IMPORT = os.environ.get("HERMES_GATEWAY_LOCK_DIR", "")
if not HOST_LOCK_DIR_AT_CONFTEST_IMPORT:
    # Deterministic per-PID name, not mkdtemp: the parallel runner SIGKILLs a worker on timeout,
    # which never runs atexit, so a random dir per run leaked one directory per killed worker.
    # A fixed name is reused by the next process with that PID, and dead siblings are swept here.
    _LOCK_DIR_PREFIX = "hermes-test-gateway-locks-"
    _LOCK_DIR_ROOT = Path(tempfile.gettempdir())
    for _stale in _LOCK_DIR_ROOT.glob(f"{_LOCK_DIR_PREFIX}*"):
        try:
            _stale_pid = int(_stale.name[len(_LOCK_DIR_PREFIX):])
        except ValueError:
            continue
        try:
            os.kill(_stale_pid, 0)
        except OSError:
            shutil.rmtree(_stale, ignore_errors=True)
    _SESSION_LOCK_DIR = str(_LOCK_DIR_ROOT / f"{_LOCK_DIR_PREFIX}{os.getpid()}")
    shutil.rmtree(_SESSION_LOCK_DIR, ignore_errors=True)
    os.environ["HERMES_GATEWAY_LOCK_DIR"] = _SESSION_LOCK_DIR
    atexit.register(shutil.rmtree, _SESSION_LOCK_DIR, True)


# ── File-level scheduling isolation ──────────────────────────────────────────
# Tests run via ``scripts/run_tests.sh``, which runs the per-file runner
# (``scripts/run_tests_parallel.py``) on every host: every file in its own
# freshly-spawned ``python -m pytest <file>`` subprocess — cross-file state
# leakage is impossible. Intra-file ordering is the test author's
# responsibility on every host — if test A in foo.py mutates state that
# test B in foo.py reads, that's a real bug to fix in the file (it would
# also bite anyone running ``pytest tests/foo.py`` directly).
#
# See ``scripts/run_tests.sh`` for the runner.


# Topic modules split out to keep this file under the size gate. They are
# imported rather than listed in ``pytest_plugins``: this is not the rootdir
# conftest (that is the repo root), and pytest fails a run that loads a
# non-root conftest carrying ``pytest_plugins`` after startup (e.g. ``pytest .``).
# Fixtures imported here register exactly as if they were defined here.
from tests._fixtures.env_filter import _HERMES_BEHAVIORAL_VARS, _looks_like_credential
from tests._fixtures.live_system_guard import (  # noqa: F401 — _live_system_guard registers here
    _GATEWAY_LOOKALIKE_MARK,
    _LIVE_SYSTEM_GUARD_BYPASS_MARK,
    _live_system_guard,
)
from tests._fixtures.platform_gating import _platforms_gate_reason, _reject_contradictory_platform_marks


@pytest.fixture(autouse=True)
def _hermetic_environment(tmp_path, monkeypatch):
    """Blank out all credential/behavioral env vars so local and CI match.

    Also redirects HOME and HERMES_HOME to per-test tempdirs so code that
    reads ``~/.hermes/*`` can't touch the real one, and pins TZ/LANG so
    datetime/locale-sensitive tests are deterministic.
    """
    # 1. Blank every credential-shaped env var that's currently set.
    for name in list(os.environ.keys()):
        if _looks_like_credential(name):
            monkeypatch.delenv(name, raising=False)

    # 2. Blank behavioral HERMES_* vars that could change test semantics.
    for name in _HERMES_BEHAVIORAL_VARS:
        monkeypatch.delenv(name, raising=False)

    # Honcho's fallback host/config resolution legitimately reads the user's
    # global ~/.honcho/config.json. Keep HOME stable (subprocess tests depend
    # on it), but pin the host so ordinary tests cannot inherit a developer's
    # defaultHost and silently select the wrong nested config block. Tests of
    # custom host resolution override/delete this explicitly.
    monkeypatch.setenv("HERMES_HONCHO_HOST", "hermes")

    # 3. Isolate both inputs to profile/root resolution. HERMES_HOME alone
    #    is insufficient: get_default_hermes_root() resolves the native root
    #    too, to distinguish standard profiles from custom deployments.
    #    Patch only the Hermes default, not HOME/Path.home(). Subprocesses need
    #    a stable HOME. Hardcoded real-home I/O must still trip the guard.
    import hermes_constants

    platform_default = hermes_constants._get_platform_default_hermes_home

    def isolated_platform_default() -> Path:
        root = platform_default()
        # Explicit Path.home()/LOCALAPPDATA overrides in individual tests
        # still select their own layout. Suffix changes retain their name.
        return tmp_path / root.name if root.parent == _NATIVE_HERMES_PARENT else root

    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", isolated_platform_default
    )
    fake_hermes_home = tmp_path / "hermes_test"
    fake_hermes_home.mkdir()
    (fake_hermes_home / "sessions").mkdir()
    (fake_hermes_home / "cron").mkdir()
    (fake_hermes_home / "memories").mkdir()
    (fake_hermes_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_hermes_home))
    # A test that pins the process home (hermes_constants.pin_process_hermes_home) must not
    # leak that module-global into the next test's routed-profile decisions.
    try:
        import hermes_constants as _hc
        monkeypatch.setattr(_hc, "_PINNED_PROCESS_HERMES_HOME", None, raising=False)
    except Exception:
        pass
    # Per-TEST host-rendezvous dir (see the session-level block at the top): the
    # host gateway/serve record is shared per OS user by design, so without this
    # one test's published owner makes the next test's lifecycle code attach to it.
    # HOME is deliberately NOT redirected above, so an unpinned run would read and
    # write the developer's live ~/.local/state/hermes/gateway-locks.
    # Skipped when the caller supplied the variable, so an explicit override still
    # works (tests of the resolution rule itself rely on that).
    if not HOST_LOCK_DIR_AT_CONFTEST_IMPORT:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    # Keep the subprocess-surviving isolation marker pointed at THIS test's
    # home (#82770): children spawned by the test inherit it by default, so
    # hermes_state's live-DB guard stays armed in them even when the test
    # strips pytest's own PYTEST_* vars from the child env.
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(fake_hermes_home))
    # And never let a developer-shell (or leaked child) bypass disarm the
    # guard for in-process code under test.
    monkeypatch.delenv("HERMES_STATE_DB_GUARD_BYPASS", raising=False)

    # 3b. hermes_state computes ``DEFAULT_DB_PATH = get_hermes_home() / "state.db"``
    #     at import time. When the module is first imported at collection (any
    #     test file with a top-level ``from hermes_state import ...``) that
    #     happens BEFORE this fixture ever runs, so every argless
    #     ``SessionDB()`` in every test opens the developer's REAL state.db —
    #     reading real sessions into assertions and writing test rows into the
    #     real profile. Re-pin the constant to this test's home. (Several test
    #     files already do this locally; this makes it an invariant.)
    # 3c. Multi-profile hosting is a process-global latch (``set_multiplex_active`` and the
    #     launch-env snapshot flip once and stay). A test that routes one RPC/request to a named
    #     profile would otherwise leave every later test in the file fail-closed (unscoped
    #     ``get_env_value`` in a test body raises). Reset the latch per test.
    secret_scope_mod = sys.modules.get("agent.secret_scope")
    if secret_scope_mod is not None and hasattr(secret_scope_mod, "_MULTIPLEX_ACTIVE"):
        monkeypatch.setattr(secret_scope_mod, "_MULTIPLEX_ACTIVE", False)
    if secret_scope_mod is not None and hasattr(secret_scope_mod, "_AUTO_PINNED_HOME"):
        monkeypatch.setattr(secret_scope_mod, "_AUTO_PINNED_HOME", None)
    launch_policy_mod = sys.modules.get("tui_gateway.launch_profile_policy")
    if launch_policy_mod is not None and hasattr(launch_policy_mod, "_snapshot"):
        monkeypatch.setattr(launch_policy_mod, "_snapshot", None)
    tui_server_mod = sys.modules.get("tui_gateway.server")
    if tui_server_mod is not None and hasattr(tui_server_mod, "_served_profile_homes"):
        monkeypatch.setattr(tui_server_mod, "_served_profile_homes", set())

    hermes_state_mod = sys.modules.get("hermes_state")
    if hermes_state_mod is not None and hasattr(hermes_state_mod, "DEFAULT_DB_PATH"):
        monkeypatch.setattr(
            hermes_state_mod, "DEFAULT_DB_PATH", fake_hermes_home / "state.db"
        )

    # 4. Deterministic locale / timezone / hashseed. CI runs in UTC with
    #    C.UTF-8 locale; local dev often doesn't. Pin everything.
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    # 4b. Disable AWS IMDS lookups. Without this, any test that ends up
    #     calling has_aws_credentials() / resolve_aws_auth_env_var()
    #     (e.g. provider auto-detect, status command, cron run_job) burns
    #     ~2s waiting for the metadata service at 169.254.169.254 to time
    #     out. Tests don't run on EC2 — IMDS is always unreachable here.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    # Tirith auto-installs from GitHub when enabled and missing. Unit tests
    # should never perform that implicit network/bootstrap path; Tirith-specific
    # tests opt back in by patching the security config directly.
    monkeypatch.setenv("TIRITH_ENABLED", "false")
    # On-demand extras (pm.sync_venv) install mid-test-run by design —
    # _allow_lazy_installs() fails open for users. Unit tests must never reach
    # pip/the network: with the SDK absent, any agent init whose tool checks
    # touch a lazy feature (e.g. check_tts_requirements →
    # ensure("tts.elevenlabs")) spawns a real pip install — which hangs to the
    # suite timeout under tests that set fake proxy env vars. The kill-switch
    # makes ensure() raise FeatureUnavailable immediately instead.
    # extras tests override this var in both directions.
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

    # 5. Reset plugin singleton so tests don't leak plugins from
    #    ~/.hermes/plugins/ (which, per step 3, is now empty — but the
    #    singleton might still be cached from a previous test).
    try:
        import hermes_cli.plugins as _plugins_mod
        monkeypatch.setattr(_plugins_mod, "_plugin_manager", None)
        # Also clear the keyed per-home manager cache (and any plugin
        # submodules it left in sys.modules) so a manager built for a
        # previous test's tmp_path HERMES_HOME can't leak forward. Paths
        # are unique per test, so collisions are unlikely, but a full
        # reset keeps this fixture the single source of plugin-state
        # hygiene rather than relying on path uniqueness.
        _plugins_mod._reset_plugin_managers_for_tests()
    except Exception:
        pass
    # Explicitly clear provider-specific base URL overrides that don't match
    # the generic credential-shaped env-var filter above.
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.delenv("GMI_BASE_URL", raising=False)


# Backward-compat alias — old tests reference this fixture name. Keep it
# as a no-op wrapper so imports don't break.
@pytest.fixture(autouse=True)
def _isolate_hermes_home(_hermetic_environment):
    """Alias preserved for any test that yields this name explicitly."""
    return None


@pytest.fixture(autouse=True)
def _reset_foreground_exit_fence():
    """A test that drives a hard-exit path raises the one-way foreground-spawn fence; lower it after."""
    yield
    if (base := sys.modules.get("tools.environments.base")) is not None:
        base._exit_fenced = False


@pytest.fixture(autouse=True)
def _neutralize_kanban_memory_guard(request, monkeypatch):
    """Pin the kanban dispatcher's memory guard to "no data" for every test.

    The dispatcher consults live system memory before spawning (OOF-30/
    OOF-77: memory-derived default cap + pressure-based spawn restriction).
    Left un-patched, dispatch tests would pass or fail based on how loaded
    the CI runner happens to be. Defaulting the sample to ``{}`` makes the
    derived cap ``None`` and the pressure level ``"unknown"`` — i.e. the
    pre-guard behaviour every existing test was written against. Tests that
    exercise the guard itself opt out with
    ``@pytest.mark.real_memory_guard`` or patch the seam directly.
    """
    if request.node.get_closest_marker("real_memory_guard"):
        return
    try:
        from hermes_cli import kanban_db_dispatch as _kbd_mod
    except Exception:
        return
    monkeypatch.setattr(_kbd_mod, "_system_memory_sample", lambda: {}, raising=False)


@pytest.fixture(autouse=True)
def _neutralize_git_safe_directory_read(request, monkeypatch):
    """Skip the ``git config --get-all safe.directory`` pre-read in ``noninteractive_git_env()``.

    Many tests fake ``subprocess.run``/``Popen`` with a fixed sequence of expected git calls;
    the pre-read is an extra spawn that would trip them. Tests of the carve-out itself opt in
    with ``@pytest.mark.real_safe_directory``.
    """
    if request.node.get_closest_marker("real_safe_directory"):
        return
    try:
        from hermes_cli import _subprocess_compat
    except Exception:
        return
    monkeypatch.setattr(_subprocess_compat, "_user_safe_directories", lambda base_env: [], raising=False)


@pytest.fixture(autouse=True)
def _close_leaked_session_dbs():
    """Close every SessionDB a test constructed but forgot to close.

    Root cause of OOM incident 20260816: ~40 files under tests/hermes_cli/
    build ``SessionDB(...)`` directly and never call ``close()``. Each open
    instance holds the writer connection (state.db + -wal fds), up to
    ``_READ_POOL_MAX`` pooled read connections, per-connection SQLite page
    caches, and — once token accounting has run — an ``atexit`` registration
    that pins the instance alive until interpreter exit. Under the sanctioned
    per-file-process runner this is invisible, but a raw single-process
    ``pytest tests/hermes_cli/`` accumulated 16-25 GB RSS and had to be
    OOM-killed three times in one day.

    Rather than editing every test file, ``SessionDB.__init__`` registers each
    instance in ``hermes_state_guard._test_instance_registry`` (a WeakSet,
    populated only when the ``HERMES_TEST_ISOLATION`` marker is set — i.e.
    only under this suite). This teardown closes whatever the test left open.
    ``close()`` is idempotent (``self._conn`` is None afterwards) and also
    unregisters the pinning atexit hook, so instances become collectable.

    Snapshotting the registry BEFORE the test and closing only NEW instances
    is deliberately avoided: closing pre-existing instances is harmless (they
    were leaked by an earlier test in the same process) and the simpler
    close-everything sweep is what actually bounds the process.

    Instances opened through ``hermes_state_registry.acquire()`` are skipped:
    on those ``close()`` releases a refcount rather than closing, so a sweep
    would silently retire a shared generation that a wider-scoped fixture
    still holds. The registry owns that lifecycle (``close_all()``).

    Before the sweep, the auto-title upgrade threads a turn spawned are joined
    (bounded): they hold the turn's SessionDB and write to it (and print to
    ``sys.stdout``) after the turn returns, so left running they race this
    close (``_reopen_after_close_locked`` on a daemon thread), the next test's
    capture, and interpreter finalization — the ``Fatal Python error`` /
    SIGSEGV shape of #113186, seen from ``tests/gateway/test_timestamp_sidecar_replay.py``.
    """
    yield
    # sys.modules lookup, not import: a file that never touched title_generator spawned
    # nothing. Tests that swap in a stub module (tui_gateway golden transcript) have no
    # real threads either, so a stub without the helper is the same "nothing to join" case.
    wait = getattr(sys.modules.get("agent.title_generator"), "wait_for_title_upgrades", None)
    if wait is not None:
        wait()
    try:
        from hermes_state_guard import _test_instance_registry as registry
    except Exception:
        return
    if not registry:
        return
    for db in list(registry):
        if getattr(db, "_shared_registry_owned", False):
            continue
        try:
            db.close()
        except Exception:
            # Teardown must never fail a passing test; a close that raises
            # (cross-thread ProgrammingError, already-closed) leaves at most
            # the one connection for the next sweep / process exit.
            pass


@pytest.fixture(autouse=True)
def _neutralize_webbrowser(monkeypatch):
    """Record browser-open attempts instead of opening real browser windows."""
    import webbrowser as _webbrowser

    opened: list[object] = []

    def _record(url=None, *_args, **_kwargs):
        opened.append(url)
        return True

    class _RecordingBrowser:
        def open(self, url, *_args, **_kwargs):
            return _record(url)

        def open_new(self, url, *_args, **_kwargs):
            return _record(url)

        def open_new_tab(self, url, *_args, **_kwargs):
            return _record(url)

    browser = _RecordingBrowser()

    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(_webbrowser, name, _record, raising=False)
    monkeypatch.setattr(_webbrowser, "get", lambda *_args, **_kwargs: browser)

    return opened


@pytest.fixture(autouse=True)
def _neutralize_macos_keychain_creds(request, monkeypatch):
    """Default Anthropic credential resolution away from the real macOS Keychain."""
    if request.node.get_closest_marker(_ALLOW_MACOS_KEYCHAIN_MARK):
        return None

    try:
        _mod = importlib.import_module("agent.anthropic_credentials")
    except Exception:
        return None
    monkeypatch.setattr(
        _mod,
        "_read_claude_code_credentials_from_keychain",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    # The #98334 refresh write also mirrors into the Keychain; keep that out of
    # the real store in any test that hasn't explicitly opted in.
    monkeypatch.setattr(
        _mod,
        "_mirror_claude_code_credentials_to_keychain",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    return None


# ── Kanban write guard (#69283) ─────────────────────────────────────────────
# When hermetic isolation is bypassed (stale checkout, wrong rootdir, direct
# invocation), kanban writes silently pollute the real ~/.hermes. This autouse
# fixture patches ``kanban_db_connect.connect`` to refuse writes whose resolved DB
# path lands under the REAL kanban root (captured at import time, before any
# fixture rewires the environment). A deny-list is used instead of an
# allow-list because test-level fixtures legitimately move HERMES_HOME to
# sibling directories — an allow-list captured at setup time would see the
# stale autouse-set value and falsely reject hermetic tests (#69385 review).


def _capture_real_kanban_root() -> Path:
    """Resolve the REAL kanban root from the pre-test environment.

    Uses the pre-sandbox environment snapshot taken at the very top of this
    file (before the session HERMES_HOME sandbox rewired the env), so the
    deny-list keeps pointing at the operator's actual root. Mirrors
    ``kanban_db.kanban_home()`` resolution order:
    1. ``HERMES_KANBAN_HOME`` env var when set and non-empty
    2. the real (pre-sandbox) Hermes root otherwise
    """
    if _PRE_SANDBOX_KANBAN_OVERRIDE:
        return Path(_PRE_SANDBOX_KANBAN_OVERRIDE).expanduser().resolve()
    if _PRE_SANDBOX_HERMES_HOME and not _hermes_home_points_at_production(
        _PRE_SANDBOX_HERMES_HOME
    ):
        # HERMES_HOME was genuinely set to a CUSTOM root before the sandbox
        # (production-pointing values are sandboxed away above, in which case
        # the env still holds the tempdir and the resolver would be wrong) —
        # honor it via the normal resolver (it may be a profile dir whose
        # root matters).
        from hermes_constants import get_default_hermes_root
        return get_default_hermes_root().resolve()
    # No pre-existing HERMES_HOME: the real root is the platform default,
    # NOT the sandbox tempdir now sitting in the env.
    return (Path.home() / ".hermes").resolve()


_REAL_KANBAN_ROOT = _capture_real_kanban_root()


@pytest.fixture(autouse=True)
def _kanban_write_guard(_hermetic_environment, monkeypatch):
    """Fail-closed guard: refuse kanban writes that target the REAL root.

    Uses a **deny-list**: only blocks writes where the resolved DB path
    (explicit ``db_path`` or ``kanban_db_path()``) lands under the real
    ``~/.hermes`` captured at import time. Hermetic tests that legitimately
    move HERMES_HOME to sibling tempdirs are unaffected.

    Only patches when ``hermes_cli.kanban_db_connect`` is *already imported*
    — a ``sys.modules`` probe, not an import — so the guard never drags the
    kanban module into unrelated test processes.

    Uses ``monkeypatch.setattr`` so pytest restores ``connect`` automatically
    after each test (no stacked wrappers or state leakage across tests).
    """
    _kdb = sys.modules.get("hermes_cli.kanban_db")
    _kdbc = sys.modules.get("hermes_cli.kanban_db_connect")
    if _kdb is None or _kdbc is None:
        return

    # The sys.modules probe can observe the module MID-IMPORT: a fixture
    # boundary firing while another test's lazy `import hermes_cli.kanban_db`
    # is still executing sees a partially initialized module whose `connect`
    # doesn't exist yet (AttributeError flake, caught in a full-suite run).
    # A half-imported module has no callers yet either — nothing to guard
    # this round; the next test's fixture will patch the completed module.
    _orig_connect = getattr(_kdbc, "connect", None)
    if _orig_connect is None or getattr(_kdb, "kanban_db_path", None) is None:
        return

    def _guarded_connect(db_path=None, *args, **kwargs):
        if db_path is not None:
            resolved = Path(db_path).expanduser().resolve()
        else:
            resolved = (
                _kdb.kanban_db_path(board=kwargs.get("board"))
                .expanduser()
                .resolve()
            )
        try:
            resolved.relative_to(_REAL_KANBAN_ROOT)
        except ValueError:
            # Resolved path is NOT under the real root — safe to write.
            return _orig_connect(db_path, *args, **kwargs)
        raise RuntimeError(
            f"kanban_write_guard: kanban DB path resolved to {resolved}, "
            f"which is under the REAL kanban root ({_REAL_KANBAN_ROOT}). "
            f"Hermetic isolation has been bypassed — refusing to write "
            f"to the real ~/.hermes. See #69283."
        )

    monkeypatch.setattr(_kdbc, "connect", _guarded_connect)


# ── Live state.db write guard ───────────────────────────────────────────────
# Companion to the kanban guard above, for the MAIN state database.
# ``hermes_state._ensure_test_isolation`` (the single choke point every
# ``SessionDB()`` construction goes through) refuses, under pytest, any DB
# path that resolves inside the REAL Hermes root. This fixture wires the
# test-side knobs:
#   • honors ``@pytest.mark.live_system_guard_bypass`` (the established
#     escape-hatch marker) by disabling the state-db guard for that test;
#   • injects the pre-sandbox CUSTOM production root (Docker/portable
#     installs where HERMES_HOME is not ~/.hermes) into the guard's
#     deny-list, mirroring the kanban deny-list capture above.
# The guard itself is env-activated (PYTEST_CURRENT_TEST / PYTEST_VERSION),
# so subprocess children that import hermes_state directly are covered even
# without this fixture.


@pytest.fixture(autouse=True)
def _state_db_write_guard(request, monkeypatch):
    _hs = sys.modules.get("hermes_state")
    if _hs is None or not hasattr(_hs, "_STATE_DB_GUARD_BYPASS"):
        yield
        return
    if request.node.get_closest_marker("live_system_guard_bypass") is not None:
        monkeypatch.setattr(_hs, "_STATE_DB_GUARD_BYPASS", True)
        yield
        return
    extra_roots = []
    if _PRE_SANDBOX_HERMES_HOME and not _hermes_home_points_at_production(
        _PRE_SANDBOX_HERMES_HOME
    ):
        extra_roots.append(
            Path(_PRE_SANDBOX_HERMES_HOME).expanduser().resolve()
        )
    monkeypatch.setattr(
        _hs, "_STATE_DB_GUARD_EXTRA_DENY_ROOTS", tuple(extra_roots)
    )
    yield


# ── Module-level state reset — replaced by per-file process isolation ───────
#
# ``scripts/run_tests_parallel.py`` runs each test FILE in its own freshly
# spawned pytest subprocess, so heavy co-scheduling pollution (module-level
# dicts / sets / ContextVars shared by many files) cannot cross file
# boundaries at all. Within a single file, ordering is the author's
# responsibility. If your tests in the same file share mutable state, either
# reset it explicitly in a fixture or split them across files.
#
# The skill ``test-suite-cascade-diagnosis`` documents the cascade patterns
# this replaces; the running example was ``test_command_guards`` failing
# 12/15 CI runs because ``tools.approval._session_approved`` carried
# approvals from one test's session into another's.


# ── tui_gateway.server shared-module state isolation ───────────────────────
#
# ``tui_gateway.server`` registers its RPC handlers in a module-level
# ``_methods`` dict at import time and keeps per-session state in module
# globals (sessions, child-run registry, config cache, DB handle). The
# canonical per-file process isolation above hides any leakage, but a direct
# multi-file invocation (``pytest tests/tui_gateway/ tests/tui_gateway/test_tui_gateway_server.py``,
# or plain ``pytest tests/``) shares one interpreter: a test that stubs
# ``_methods["slash.exec"]`` or leaves an active-session lease behind breaks
# unrelated tests in later files. This fixture snapshots the cheap-to-copy
# globals before each test and restores them after, so any file combination
# is order-independent. It is a near no-op (one sys.modules lookup) while
# the module has not been imported.
#
# The case this cannot cover — the module is first imported *during* a test
# that also mutates ``_methods`` — is handled by the importing files' own
# ``server`` fixtures (tests/tui_gateway/test_protocol.py and friends), which
# snapshot immediately after the import.

_TUI_SERVER_MODULE = "tui_gateway.server"


def _teardown_tui_server_sessions(mod) -> None:
    """Close leftover sessions through the production teardown boundary.

    Besides returning active-session leases, this finalizes the session,
    unregisters notification state, and closes its agent and slash worker.
    """
    sessions = getattr(mod, "_sessions", None)
    if not isinstance(sessions, dict):
        return
    for sid in list(sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")


@pytest.fixture(autouse=True)
def _reset_tui_gateway_server_state():
    mod = sys.modules.get(_TUI_SERVER_MODULE)
    snapshot = None
    if mod is not None:
        snapshot = {
            "methods": dict(mod._methods),
            "cfg": (mod._cfg_cache, mod._cfg_sig, mod._cfg_path),
            "db": (mod._db, mod._db_error),
            "real_stdout": mod._real_stdout,
        }

    yield

    mod = sys.modules.get(_TUI_SERVER_MODULE)
    if mod is None:
        return

    # This finalizer can run before the test's own monkeypatch undo, so a
    # global may still be replaced with a non-dict test double — skip those
    # (monkeypatch restores the real, pre-test object afterwards anyway).
    sessions = mod._sessions
    if isinstance(sessions, dict):
        _teardown_tui_server_sessions(mod)
    for name in (
        "_pending",
        "_pending_prompt_payloads",
        "_answers",
        "_child_mirrors",
        "_active_child_runs",
    ):
        obj = getattr(mod, name, None)
        if isinstance(obj, dict):
            obj.clear()

    if snapshot is not None:
        mod._methods.clear()
        mod._methods.update(snapshot["methods"])
        mod._cfg_cache, mod._cfg_sig, mod._cfg_path = snapshot["cfg"]
        mod._db, mod._db_error = snapshot["db"]
        mod._real_stdout = snapshot["real_stdout"]
    else:
        # First imported during this test — reset to import-time defaults
        # for the globals we could not snapshot (``_methods`` is left to
        # the importing file's fixture, see block comment above).
        mod._cfg_cache = None
        mod._cfg_sig = None
        mod._cfg_path = None
        mod._db = None
        mod._db_error = None

    # A leaked context-local Hermes home override redirects every later
    # ``get_hermes_home()`` call (active-session registry, config paths)
    # to a stale per-test tmpdir. Force the main-thread ContextVar back
    # to its default.
    try:
        from hermes_constants import get_hermes_home_override, set_hermes_home_override

        if get_hermes_home_override() is not None:
            set_hermes_home_override(None)
    except Exception:
        pass


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up automatically."""
    return tmp_path


@pytest.fixture()
def mock_config():
    """Return a minimal hermes config dict suitable for unit tests."""
    return {
        "model": "test/mock-model",
        "toolsets": ["terminal", "file"],
        "max_turns": 10,
        "terminal": {
            "backend": "local",
            "cwd": "/tmp",
            "timeout": 30,
        },
        "compression": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "command_allowlist": [],
    }


# ── Per-test timeout — handled by the isolation plugin ─────────────────────
#
# The subprocess-per-test plugin enforces the configured ``isolate_timeout``
# ini key by terminating the child if it overruns. The old SIGALRM-based
# fixture (POSIX-only, didn't work on Windows) is gone.


@pytest.fixture(autouse=True)
def _ensure_current_event_loop(request):
    """Provide a default event loop for sync tests that call get_event_loop().

    Python 3.11+ no longer guarantees a current loop for plain synchronous tests.
    A number of gateway tests still use asyncio.get_event_loop().run_until_complete(...).
    Ensure they always have a usable loop without interfering with pytest-asyncio's
    own loop management for @pytest.mark.asyncio tests.

    On Python 3.12+, ``asyncio.get_event_loop_policy().get_event_loop()`` with no
    *running* loop emits DeprecationWarning; skip that path and install a fresh
    loop via ``new_event_loop()`` instead.
    """
    if request.node.get_closest_marker("asyncio") is not None:
        yield
        return

    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop is None and sys.version_info < (3, 12):
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None

    created = loop is None or loop.is_closed()
    if created:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield
    finally:
        if created and loop is not None:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)


_REQUIRES_WAL_MARK = "requires_wal"


def _wal_is_usable() -> bool:
    """True when Hermes will actually put a database into WAL mode here.

    Hermes refuses journal_mode=WAL on SQLite builds carrying the upstream
    WAL-reset corruption bug (3.7.0–3.51.2, excluding backports 3.50.7 /
    3.44.6) and falls back to DELETE. On such a build NO ``-wal`` sidecar is
    ever created, so a test asserting on WAL frames, ``-wal`` file size, or
    checkpoint behaviour cannot pass — it is testing a mode the runtime
    declined to enable, not a regression.

    This matters because the interpreter running the tests and the interpreter
    running Hermes can link DIFFERENT SQLite versions: a repo ``.venv`` on
    3.50.4 (vulnerable → DELETE) alongside a Hermes managed runtime on 3.53.1
    (fixed → WAL). The same test then passes in one and fails in the other.

    IMPORTANT: this must NOT import ``hermes_state``. That module computes
    ``DEFAULT_DB_PATH`` from ``get_hermes_home()`` at import time, so importing
    it during collection — before the per-test ``_isolate_hermes_home`` fixture
    redirects ``HERMES_HOME`` — permanently caches the DEVELOPER'S REAL
    ``~/.hermes/state.db`` for the whole session. Tests then read live
    production sessions instead of a tempdir. The version predicate is
    duplicated from ``hermes_state._is_sqlite_wal_reset_vulnerable`` (upstream
    fixed ranges, stable) rather than imported, and
    ``test_conftest_wal_gate.py`` pins the two implementations in agreement.
    """
    info = sqlite3.sqlite_version_info
    if info < (3, 7, 0):
        return True  # pre-WAL library: cannot hit the race
    if info >= (3, 51, 3):
        return True  # fixed upstream
    if (3, 50, 7) <= info < (3, 51, 0):
        return True  # 3.50.x backport
    if (3, 44, 6) <= info < (3, 45, 0):
        return True  # 3.44.x backport
    return False


# ── Audio-playback guard ───────────────────────────────────────────────────
#
# Same class of incident as the live-system guard (``tests/_fixtures/live_system_guard.py``),
# different primitive:
# a test run spoke the string "partial answer complete" out of the developer's
# speakers. That string is a test fixture
# (``tests/tui_gateway/test_tui_gateway_server.py``'s fake ``final_response``), and the
# route it took is fully in-process — no leaked shell variable required:
#
#   1. ``test_voice_toggle_tts_branch_also_carries_record_key`` drives the
#      ``voice.toggle`` RPC with ``action="tts"``. The handler
#      (``tui_gateway/server.py``) flips the flag by writing the *real*
#      process environment: ``os.environ["HERMES_VOICE_TTS"] = "1"``. The
#      test's ``monkeypatch.delenv(..., raising=False)`` records no undo entry
#      (pytest only records an undo when the key was present), so the "1"
#      survives teardown and persists for the rest of the pytest process.
#   2. Any later test in that process that drives a turn to completion hits
#      the TTS dispatch in ``prompt.submit``, which checks
#      ``_voice_tts_enabled()`` — now true — and fires
#      ``hermes_cli.voice.speak_text(final_response)`` on a daemon thread.
#   3. ``speak_text`` needs no API key to be audible: ``tools/tts_tool.py``
#      defaults to the ``edge`` provider, which is keyless.
#
# Because the flag is set from *inside* the process, ``scripts/run_tests.sh``'s
# ``env -i`` does not help, and neither does env-blanking on its own — the
# hermetic fixture blanks at test setup, and step 1 re-sets it mid-test. So we
# also intercept the primitive that does the damage, exactly as the
# live-system guard intercepts ``os.kill`` rather than trusting every caller
# to mock it:
#
#  • ``hermes_cli.voice.speak_text`` — the synth+playback entry point both
#    gateway call sites late-import, so patching the module attribute catches
#    them wherever they import it from.
#  • ``hermes_cli.voice.play_audio_file`` — the module-level binding
#    ``speak_text`` actually plays through. Patching the binding inside
#    ``hermes_cli.voice`` (not ``tools.voice_mode``) keeps the real function
#    available to the tests that legitimately exercise it with a mocked
#    audio backend (``tests/tools/test_voice_mode.py``).
#
# Config cannot re-open this hole: the ``tts:`` section of ``config.yaml``
# only selects *which* provider speaks, never *whether* to speak — that gate
# is the env var alone.

_AUDIO_GUARD_BYPASS_MARK = "real_audio_playback"
_ALLOW_MACOS_KEYCHAIN_MARK = "allow_macos_keychain"


def _relocate_basetemp_outside_operator_home(config) -> None:
    """Move pytest's basetemp out of the operator's platform-native Hermes home.

    Every per-test sandbox is ``<basetemp>/.../hermes_test``. ``get_default_hermes_root()``
    prefers the platform-native home whenever ``HERMES_HOME`` sits *under* it, so a basetemp
    inside ``~/.hermes`` (or ``%LOCALAPPDATA%\\hermes``, where ``TEMP`` commonly lives on
    Windows) turns the sandbox back into the live install and ``get_profile_dir("default")``
    writes fixtures over the operator's config.yaml / .env / MEMORY.md (#111101).
    """
    from hermes_constants import _get_platform_default_hermes_home

    native = _get_platform_default_hermes_home().resolve()
    factory = config._tmp_path_factory
    given = factory._given_basetemp
    candidate = given if given is not None else Path(
        os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir()
    )
    if not candidate.resolve().is_relative_to(native):
        return
    # The system temp dir may itself be inside the home (Windows TEMP under the
    # Hermes home). The repo is no escape either: the default install checks it
    # out *inside* the home (~/.hermes/hermes-agent). The relocated basetemp goes
    # into ONE prunable root outside the home, never loose into the operator's
    # $HOME (123 ``hermes-pytest-basetemp-*`` dirs piled up there in a day, one per
    # test file the per-file runner spawned). It is removed when this pytest exits
    # and, for runs that were killed before that, swept once it is 24h idle.
    safe = Path(tempfile.mkdtemp(prefix="b-", dir=_pytest_disk_temp_root(native)))
    assert not safe.resolve().is_relative_to(native), (
        f"pytest basetemp {safe} still resolves inside the operator's Hermes home {native}; "
        "refusing to run the suite against the live install (pass --basetemp outside it)"
    )
    factory._given_basetemp = safe
    config.option.basetemp = str(safe)
    config._hermes_relocated_basetemp = safe


def _pytest_disk_temp_root(native: Path) -> Path:
    """The root for relocated basetemps: the disk-backed runner root when the host has
    one (``scripts/run_tests_parallel.py::_runner_scratch_root``), else a plain (not
    dot-prefixed — hidden-dir search tests would see every fixture as hidden) sibling of
    the native home. Entries idle for a day are swept on the way in."""
    from hermes_constants_scratch import prune_idle_entries

    if os.name != "nt" and os.path.isdir("/var/tmp"):  # no-tmp: ok — disk-backed FHS root
        root = Path("/var/tmp/hermes-pytest")  # no-tmp: ok — /var/tmp is disk-backed by FHS, never tmpfs
    else:
        root = native.parent / "hermes-pytest"
    root.mkdir(parents=True, exist_ok=True)
    prune_idle_entries(root, 24, frozenset())
    return root


def _remove_relocated_basetemp(config) -> None:
    safe = getattr(config, "_hermes_relocated_basetemp", None)
    if safe is not None:
        shutil.rmtree(safe, ignore_errors=True)


def _pinned_mcp_sdk_version() -> str:
    """The ``mcp==X`` pin carried by the ``[mcp]`` extra in pyproject.toml."""
    import tomllib

    with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as fh:
        extras = tomllib.load(fh)["project"]["optional-dependencies"]
    for req in extras["mcp"]:
        if req.startswith("mcp=="):
            return req.split("==", 1)[1].strip()
    raise RuntimeError("pyproject.toml [mcp] extra no longer pins mcp==X")


@pytest.fixture
def require_mcp_2_sdk():
    """Skip tests that pin mcp 2.0-only behaviour when an older SDK is installed.

    The runtime deliberately supports both SDK generations (the dual streamable-client probe in
    mcp_tool), so a stale ``mcp`` distribution imports fine and presence-only guards let these
    tests through — where they fail later with opaque SDK errors. Compare the installed
    distribution against the pin so the outcome is an explicit skip with an actionable reason.
    """
    from importlib.metadata import PackageNotFoundError, version as dist_version

    from packaging.version import Version

    pinned = _pinned_mcp_sdk_version()
    try:
        found = dist_version("mcp")
    except PackageNotFoundError:
        pytest.skip(f"requires mcp=={pinned} (not installed); install the [mcp] extra")
    if Version(found) < Version(pinned):
        pytest.skip(f"requires mcp=={pinned} (found {found}); install the [mcp] extra")


def pytest_unconfigure(config):  # noqa: D401 — pytest hook
    _remove_relocated_basetemp(config)


@pytest.hookimpl(trylast=True)  # after _pytest.tmpdir has built config._tmp_path_factory
def pytest_configure(config):  # noqa: D401 — pytest hook
    """Register markers used by hermetic conftest."""
    _relocate_basetemp_outside_operator_home(config)
    config.addinivalue_line(
        "markers",
        f"{_LIVE_SYSTEM_GUARD_BYPASS_MARK}: bypass the live-system guard "
        "(only for tests that genuinely need real os.kill / subprocess "
        "behaviour — e.g. PTY tests that signal their own child).",
    )
    config.addinivalue_line(
        "markers", "allow_real_home_io: explicitly bypass the test-only home I/O guard."
    )
    config.addinivalue_line(
        "markers", "real_release_channels: keep the real R2 channel reader (no local source-branch stub)."
    )
    config.addinivalue_line(
        "markers",
        f"{_GATEWAY_LOOKALIKE_MARK}: the test spawns and reaps its own stub "
        "child whose argv matches the gateway runtime matcher; only the "
        "real-gateway spawn check is lifted, os.kill stays guarded.",
    )
    config.addinivalue_line(
        "markers",
        "real_safe_directory: run the real `git config --get-all safe.directory` pre-read in "
        "noninteractive_git_env() (autouse fixture otherwise stubs it to no entries).",
    )
    config.addinivalue_line(
        "markers",
        f"{_REQUIRES_WAL_MARK}: test needs the runtime to actually enable "
        "SQLite WAL mode; skipped on builds where Hermes falls back to "
        "journal_mode=DELETE for the WAL-reset bug.",
    )
    config.addinivalue_line(
        "markers",
        f"{_AUDIO_GUARD_BYPASS_MARK}: bypass the audio-playback guard (only "
        "for tests that genuinely need real TTS synthesis and speaker "
        "playback — there are none in the default suite).",
    )
    config.addinivalue_line(
        "markers",
        "platforms(*specs, arch=None, arch_negate=False): run only on hosts "
        "matching at least one spec — linux/macos/windows/posix/any, "
        "'not X' negation, optional arch filter (e.g. arch='arm64')",
    )
    config.addinivalue_line(
        "markers",
        "platforms(*specs, arch=None, arch_negate=False): run only on hosts "
        "matching at least one spec — linux/macos/windows/posix/any, "
        "'not X' negation, optional arch filter (e.g. arch='arm64')",
    )
    config.addinivalue_line(
        "markers",
        f"{_ALLOW_MACOS_KEYCHAIN_MARK}: allow a test to exercise the macOS "
        "Keychain credential reader with its own subprocess/platform mocks.",
    )
    config.addinivalue_line(
        "markers",
        "require_symlinks: skip the test if symbolic links cannot be "
        "created in the current environment (needs admin/developer mode "
        "on Windows).",
    )
    config.addinivalue_line(
        "markers",
        "real_memory_guard: bypass the autouse fixture that pins the kanban "
        "dispatcher's memory guard to 'no data' — only for tests that "
        "exercise the guard itself with their own patched samples.",
    )
    # NOTE: platforms("linux") / platforms("macos") / platforms("windows") are declared in
    # pyproject.toml's ``markers`` list, not here — they are part of the
    # project's public marker vocabulary (``pytest --markers``, and the CI
    # lanes select on them), whereas the marks above are conftest-internal
    # guards. Declaring them in both places just meant two descriptions that
    # could drift apart.

    # The pyproject addopts pin ``--timeout-method=signal`` relies on
    # ``signal.SIGALRM``, which does not exist on Windows — pytest-timeout
    # raises AttributeError at timer setup and the whole run aborts before any
    # test executes. Fall back to the thread-based timer on Windows so the
    # suite runs natively there (POSIX keeps the more reliable signal method).
    if sys.platform == "win32" and getattr(config.option, "timeout_method", None) == "signal":
        config.option.timeout_method = "thread"


_symlink_supported_cache = None


def _check_symlink_support() -> bool:
    global _symlink_supported_cache
    if _symlink_supported_cache is not None:
        return _symlink_supported_cache

    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.touch()
            lnk = Path(d) / "lnk"
            lnk.symlink_to(src)
            _symlink_supported_cache = True
            return True
    except OSError:
        _symlink_supported_cache = False
        return False


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_call(item):
    """Join the turn's auto-title threads INSIDE capture, before pytest snaps it.

    A title thread that prints its failure warning while capture's
    ``readouterr`` swaps the fd crashed the interpreter (SIGSEGV in
    ``_pytest/capture.py::snap``). The teardown join in
    ``_close_leaked_session_dbs`` runs after that snap, too late for this race.
    """
    try:
        return (yield)
    finally:
        wait = getattr(sys.modules.get("agent.title_generator"), "wait_for_title_upgrades", None)
        if wait is not None:
            wait()


def pytest_runtest_setup(item):
    if item.get_closest_marker("require_symlinks"):
        if not _check_symlink_support():
            pytest.skip(
                "Environment does not support symbolic links "
                "(requires admin/developer mode on Windows)"
            )


def pytest_collection_modifyitems(config, items):  # noqa: D401 — pytest hook
    """Apply host-OS gating, then skip ``requires_wal`` where WAL is unusable.

    OS gating: a test marked ``platforms(...)`` runs only on hosts its
    specs match. See the block comment in ``tests/_fixtures/platform_gating.py``
    for why these tests are skipped rather than run against a patched ``sys.platform``.

    WAL gating is cheaper and more honest than each test hand-rolling a
    version check: the reason string names the actual linked version so the
    skip is diagnosable rather than mysterious.
    """
    _reject_contradictory_platform_marks(items)

    # platforms() gating: skip items whose specs exclude this host. The skip
    # markers (not -m expressions) are the authoritative host filter on
    # every lane, so a lane selects with plain ``-m platforms`` and lets the
    # specs decide per-test.
    for item in items:
        reason = _platforms_gate_reason(item)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))

    if _wal_is_usable():
        return

    reason = (
        f"SQLite {sqlite3.sqlite_version} has the WAL-reset bug — Hermes uses "
        "journal_mode=DELETE here, so no -wal sidecar exists to assert on"
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker(_REQUIRES_WAL_MARK) is not None:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True)
def _audio_playback_guard(request, monkeypatch):
    """Stub TTS synthesis + speaker playback for every test.

    See the block comment above for the incident this closes. Defence in
    depth behind ``_HERMES_BEHAVIORAL_VARS``: the env blanking stops the flag
    leaking *between* tests, this stops the speakers ever opening even when a
    test sets the flag *itself* (which the ``voice.toggle`` RPC handler does,
    by writing ``os.environ`` directly).

    Deliberately silent rather than raising: unlike a stray ``os.kill``, a
    stray ``speak_text`` is dispatched on a daemon thread whose exception
    nobody would ever see, so a hard failure would neither stop the test nor
    surface. Silence is the whole point. Tests that genuinely want real audio
    can opt out with ``@pytest.mark.real_audio_playback``.
    """
    if request.node.get_closest_marker(_AUDIO_GUARD_BYPASS_MARK):
        yield
        return

    try:
        import hermes_cli.voice as _voice
    except Exception:
        # Optional audio deps missing — nothing importable to speak with.
        yield
        return

    def _blocked_speak_text(text, *args, **kwargs):
        return None

    def _blocked_play_audio_file(path, *args, **kwargs):
        return False

    if hasattr(_voice, "speak_text"):
        monkeypatch.setattr(_voice, "speak_text", _blocked_speak_text)
    if hasattr(_voice, "play_audio_file"):
        monkeypatch.setattr(_voice, "play_audio_file", _blocked_play_audio_file)

    yield


@pytest.fixture(autouse=True)
def _isolate_computer_use_approval_state():
    """Reset the computer-use explicit approval callback after every test.

    ``tools.computer_use.tool._approval_callback`` is a module-global handed to
    the shared approval gate as its explicit callback, where it takes precedence
    over the per-thread terminal one. A test that installs it and does not
    reset it poisons every later computer-use test in the same process: a
    leaked callback that raises becomes a deny, a leaked one that blocks (the
    real CLI one waits on an answer queue) hangs the whole single-process run.
    Both symptoms are order-dependent. Teardown-only, so tests that install
    their own callback keep it for their own duration.
    """
    yield
    try:
        from tools.computer_use import tool as _cu_tool

        _cu_tool.set_approval_callback(None)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _moa_caches_isolated():
    """Clear module-level MoA cold-start caches before each test.

    ``agent.moa_loop`` caches the resolved preset and each slot's provider
    runtime at module level (keyed on config mtime / provider+model) so the
    tool loop doesn't re-resolve them serially on every iteration. Tests
    monkeypatch resolvers and config paths, so a cache entry leaked from one
    test would poison the next. Clear both around every test.
    """
    import agent.moa_loop as moa

    moa._preset_cache.clear()
    moa._runtime_cache.clear()
    yield
    moa._preset_cache.clear()
    moa._runtime_cache.clear()


# ── Real-home tripwire (universal read/write guard) ──────────────────────────
# The hermetic sandbox redirects get_hermes_home(), but TWO escape classes
# remain: (a) code hardcoding Path.home()/".hermes" (the exact restatement
# class AGENTS.md bans — the Path.home()/.hermes/profiles bug the 2026-09-03
# deployment review caught in pm/plugins_state.py), and (b) imports freezing
# real-home paths before fixtures run. The kanban guard (#69283) covers one
# subsystem; this covers EVERY file operation: any open()/mkdir/stat-family
# call resolving under the REAL hermes root fails the test immediately
# with a message naming the path — reads AND writes (a read of production
# state is as much a leak as a write: it drags fixture rows and real config
# into test assertions).
#
# The real root is captured at conftest import (pre-sandbox), honoring a
# genuinely-custom pre-set HERMES_HOME exactly like the kanban deny-list
# (_hermes_home_points_at_production governs which values count).
_REAL_HERMES_ROOT_CANDIDATES: list[Path] = []


def _capture_real_hermes_root() -> list[Path]:
    """The real root(s) to refuse: the default ~/.hermes plus a pre-sandbox
    custom HERMES_HOME when one was set. Both are guarded — the default
    because hardcoded restatements hit it; the custom one because
    deployment-shaped tests (Docker /opt/data) must not touch the operator's
    real custom root either."""
    import platform

    roots: list[Path] = []
    try:
        default_root = (Path.home() / ".hermes").resolve()
        roots.append(default_root)
    except Exception:
        pass
    # native-Windows default: %LOCALAPPDATA%\hermes (get_hermes_home's
    # platform-native path) — guard it too
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata:
        try:
            win_root = (Path(localappdata) / "hermes").resolve()
            if win_root not in roots:
                roots.append(win_root)
        except Exception:
            pass
    if _PRE_SANDBOX_HERMES_HOME and not _hermes_home_points_at_production(
        _PRE_SANDBOX_HERMES_HOME
    ):
        try:
            custom = Path(_PRE_SANDBOX_HERMES_HOME).expanduser().resolve()
            # The live session sandbox is test-owned, never a guarded root
            # (a re-imported conftest body sees it as _PRE_SANDBOX_HERMES_HOME).
            sandbox = os.environ.get("HERMES_TEST_SANDBOX_HOME", "")
            if sandbox and custom == Path(sandbox).expanduser().resolve():
                return roots
            if custom not in roots:
                roots.append(custom)
        except Exception:
            pass
    return roots


_REAL_HERMES_ROOT_CANDIDATES = _capture_real_hermes_root()


@pytest.fixture(autouse=True)
def _forbid_real_hermes_home_io(monkeypatch, request):
    """Guard Python file/metadata/deletion calls and SQLite against real state.

    Native libraries and subprocesses still need their own temporary-home
    contracts. The opt-out is for explicit guard tests, never implicit repair.
    """
    if request.node.get_closest_marker("allow_real_home_io"):
        return
    from tests.home_io_guard import HomeIOGuard

    HomeIOGuard(lambda: _REAL_HERMES_ROOT_CANDIDATES).install(monkeypatch)


@pytest.fixture
def real_bash() -> str:
    """A bash that runs shell scripts: on the Windows runners PATH resolves ``bash`` to
    System32's WSL launcher, which prints a UTF-16 "no installed distributions" notice and
    exits 1. Prefer Git for Windows' bash there; elsewhere the PATH one is real."""
    found = shutil.which("bash")
    if sys.platform == "win32" and (
            not found or any(marker in found.lower() for marker in ("system32", "windowsapps"))):
        for rel in (("Git", "bin", "bash.exe"), ("Git", "usr", "bin", "bash.exe")):
            candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")).joinpath(*rel)
            if candidate.exists():
                return str(candidate)
    return found or "bash"
