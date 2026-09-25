"""Shared fixtures for gateway tests.

The ``_ensure_telegram_mock`` helper guarantees that a minimal mock of
the ``telegram`` package is registered in :data:`sys.modules` **before**
any test file triggers ``from plugins.platforms.telegram.adapter import ...``.

Without this, pytest sessions that happen to collect
``test_telegram_caption_merge.py`` (bare top-level import, no per-file
mock) first will cache ``ChatType = None`` from the production
ImportError fallback, causing 30+ downstream test failures wherever
``ChatType.GROUP`` / ``ChatType.SUPERGROUP`` is accessed.

Individual test files may still call their own ``_ensure_telegram_mock``
— it short-circuits when the mock is already present.

Plugin-adapter anti-pattern guard
---------------------------------
Tests for platform plugins (``plugins/platforms/<name>/adapter.py``)
must load the adapter via
:func:`tests.gateway._plugin_adapter_loader.load_plugin_adapter`, not by
adding the plugin directory to ``sys.path`` and doing a bare
``from adapter import ...``. The guard at the bottom of this file
scans test module ASTs at collection time and fails collection with a
pointer to the helper if the anti-pattern is detected.

Rationale: every plugin ships its own ``adapter.py``, and two tests each
inserting their plugin dir on ``sys.path[0]`` race for
``sys.modules["adapter"]`` in the same pytest session. Whichever collects
first wins; the other fails with ``ImportError``, and the polluted
``sys.path`` cascades into unrelated tests. See PR #17764 for the
incident.
"""

import ast
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Live gateway-state isolation guard ───────────────────────────────────────
# ``gateway/status.py`` resolves its identity files (``gateway_state.json``,
# ``gateway.pid``, ``gateway.lock``, takeover markers) from the RAW ``HERMES_HOME``
# env, falling back to ``Path.home()/".hermes"`` (the LIVE home) when unset. The
# hermetic root conftest re-points ``HERMES_HOME`` at import, but that has escape
# hatches — an explicit ``HERMES_HOME=<live>``, a standalone run without the root
# conftest, or a subprocess child that rebuilds its env — any of which lets a
# gateway test write the LIVE ``~/.hermes/gateway_state.json`` (the bug that left a
# dead pytest pid as the live feishu writer). The production-side guard in
# ``gateway.status`` fails closed for the child case; this harness-side guard adds:
#   (a) refuse to start when ``HERMES_HOME`` resolves to the live home (unless
#       ``HERMES_TEST_ALLOW_LIVE=1``);
#   (b) otherwise force a scratch ``HERMES_HOME`` so every state write lands in a
#       throwaway dir;
#   (e) at session end, fail hard if the live gateway pid changed from the
#       session-start baseline; a sha-only drift is reported (which fields moved)
#       but not failed, because the live gateway's own event-driven writes make sha
#       equality spuriously brittle on a busy host;
#   (iii) force ephemeral listener ports (``API_SERVER_PORT`` / ``WEBHOOK_PORT`` = 0)
#       so a gateway test that binds a real api_server/webhook listener cannot squat
#       the live gateway's port.
# (a)/(b) run at conftest import time — before any gateway test module is collected
# — so import-time writes are covered too; (e) runs in ``pytest_sessionfinish``.
# ``--replace`` of the live gateway is subsumed by (b): takeover markers resolve
# through the same ``HERMES_HOME``, so they land in scratch, and (e) still catches a
# deliberate target at the live home.

_GATEWAY_LIVE_ALLOW_ENV = "HERMES_TEST_ALLOW_LIVE"
# Adapter listener ports forced to ephemeral under the guard: a gateway test that
# binds a real api_server/webhook listener must bind port 0, never the live gateway's.
_GATEWAY_EPHEMERAL_PORT_ENV = ("API_SERVER_PORT", "WEBHOOK_PORT")
# HERMES_HOME as seen when THIS conftest is imported (before our own redirect):
_GATEWAY_PRE_GUARD_HOME = os.environ.get("HERMES_HOME", "")
# (live_home, baseline_pid, baseline_sha, baseline_state) captured at session start;
# pid/sha/state are None when the live file was absent or unreadable at baseline:
_gateway_state_baseline = None


def _gateway_live_home():
    """The REAL production hermes home, robust against ``Path.home`` monkeypatching."""
    try:
        from hermes_state_guard import _real_platform_state_root
        home = _real_platform_state_root()
        if home is not None:
            return home
    except Exception:
        pass
    return Path.home() / ".hermes"


def _gateway_home_points_at_live(value):
    """True when *value* is the live home or a profile home directly under it."""
    if not value.strip():
        return False  # unset → force scratch below, not a refusal
    try:
        resolved = Path(value).expanduser().resolve()
    except Exception:
        return False
    live = _gateway_live_home().resolve()
    if resolved == live:
        return True
    return resolved.parent.name == "profiles" and resolved.parent.parent == live


def _gateway_read_live_state():
    """``(exists, pid, sha, state)`` of the live ``gateway_state.json``; ``pid``/``sha``/
    ``state`` are ``None`` when the file is absent or unreadable."""
    path = _gateway_live_home() / "gateway_state.json"
    try:
        raw = path.read_bytes()
    except OSError:
        return False, None, None, None
    state = None
    try:
        state = json.loads(raw.decode("utf-8"))
    except Exception:
        state = None
    pid = state.get("pid") if isinstance(state, dict) else None
    return True, pid, hashlib.sha256(raw).hexdigest(), state


def _gateway_moved_fields(before, after):
    """Short ``field: before -> after`` list for the top-level keys and per-platform
    writer pids that differ, to diagnose what a test (or the live gateway) changed."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return ["(unparseable state)"]
    moved = []
    for key in ("pid", "kind", "start_time", "gateway_state", "active_agents",
                "served_profiles", "session_store"):
        if before.get(key) != after.get(key):
            moved.append("%s: %r -> %r" % (key, before.get(key), after.get(key)))
    before_plat = before.get("platforms") if isinstance(before.get("platforms"), dict) else {}
    after_plat = after.get("platforms") if isinstance(after.get("platforms"), dict) else {}
    for key in sorted(set(before_plat) | set(after_plat)):
        bp = before_plat.get(key) if isinstance(before_plat.get(key), dict) else {}
        ap = after_plat.get(key) if isinstance(after_plat.get(key), dict) else {}
        if bp.get("writer_pid") != ap.get("writer_pid") or bp.get("state") != ap.get("state"):
            moved.append("platform %s: writer_pid %r -> %r, state %r -> %r" % (
                key, bp.get("writer_pid"), ap.get("writer_pid"),
                bp.get("state"), ap.get("state")))
    return moved or ["(sha changed but no tracked field moved)"]


# (a)/(b): redirect a live/unset HERMES_HOME to scratch at import time, before any
# gateway test module is imported, so collection can't write the live state.
# (iii): force ephemeral listener ports at import time too, so collection-time
# adapter construction can't default to the live gateway's bound port.
if os.environ.get(_GATEWAY_LIVE_ALLOW_ENV) != "1":
    if _gateway_home_points_at_live(_GATEWAY_PRE_GUARD_HOME) or not _GATEWAY_PRE_GUARD_HOME.strip():
        os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hermes-gw-test-home-")
    for _port_env in _GATEWAY_EPHEMERAL_PORT_ENV:
        os.environ[_port_env] = "0"


def pytest_sessionstart(session):
    global _gateway_state_baseline
    if os.environ.get(_GATEWAY_LIVE_ALLOW_ENV) != "1" and _gateway_home_points_at_live(_GATEWAY_PRE_GUARD_HOME):
        raise pytest.UsageError(
            "Refusing to run gateway tests: HERMES_HOME resolves to the live home "
            f"({_gateway_live_home()}). Gateway tests write gateway_state.json / "
            "gateway.pid and must never touch the live gateway. Point HERMES_HOME "
            "at a scratch dir, or set " + _GATEWAY_LIVE_ALLOW_ENV + "=1 to explicitly "
            "opt in."
        )
    exists, pid, sha, state = _gateway_read_live_state()
    _gateway_state_baseline = (_gateway_live_home(), pid if exists else None, sha, state)


def pytest_sessionfinish(session, exitstatus):
    global _gateway_state_baseline
    baseline = _gateway_state_baseline
    if baseline is None:
        return
    live, base_pid, base_sha, base_state = baseline
    exists, pid, sha, state = _gateway_read_live_state()
    problems = []
    if base_pid is not None:
        if not exists:
            problems.append("live gateway_state.json vanished during the run: %s" % (live / "gateway_state.json"))
        elif pid != base_pid:
            problems.append(
                "live gateway pid changed: was %s, now %s — a test replaced or "
                "corrupted the live gateway identity. Fields that moved: %s"
                % (base_pid, pid, "; ".join(_gateway_moved_fields(base_state, state)))
            )
        elif base_sha is not None and sha is not None and sha != base_sha:
            # The live gateway's own event-driven writes (platform transitions) drift
            # the sha on a busy host without touching the pid. Report, don't fail:
            # the pid is the unambiguous "a test took over" signal.
            logging.warning(
                "live gateway_state.json sha drifted during the run but pid is stable "
                "(%s) — the live gateway itself wrote between baseline and session end. "
                "Moved fields: %s",
                pid, "; ".join(_gateway_moved_fields(base_state, state)),
            )
    if problems:
        raise pytest.UsageError(
            "Gateway test isolation failure — the live gateway state changed during this "
            "test run:\n" + "\n".join("  - " + p for p in problems)
        )


@pytest.fixture(autouse=True)
def _gateway_ephemeral_ports(_hermetic_environment, monkeypatch):
    """Force adapter listener ports to 0 for the test, after ``_hermetic_environment``
    has blanked them. A gateway test that binds a real api_server/webhook listener must
    bind an ephemeral port, never the live gateway's configured port. Tests that need a
    specific port set it via adapter config (``extra={"port": N}``), which takes
    precedence over these env defaults."""
    for _port_env in _GATEWAY_EPHEMERAL_PORT_ENV:
        monkeypatch.setenv(_port_env, "0")


@pytest.fixture(scope="session", autouse=True)
def _bind_lark_sdk_globals_when_installed():
    """Bind the feishu adapter's lark SDK globals once per test session.

    The adapter defers ``import lark_oapi`` to first use
    (``_load_lark_oapi`` — called from connect()/probe_bot()/standalone
    send), so the request-builder globals (``CreateMessageRequestBody``
    etc.) stay ``None`` at module import time. Feishu tests across many
    files inject a mock ``_client`` and skip connect() entirely, then call
    send paths that reference those globals. Bind them eagerly when the
    SDK is installed; when it isn't, the affected tests already skip via
    their own ``skipUnless`` guards.
    """
    try:
        import lark_oapi  # noqa: F401
    except ImportError:
        yield
        return
    try:
        from plugins.platforms.feishu.adapter import _load_lark_oapi

        _load_lark_oapi()
    except Exception:
        pass  # adapter not importable in this environment — tests will skip
    yield


def make_async_session_db(sync_mock=None):
    """Wrap a sync mock SessionDB in AsyncSessionDB so gateway code that awaits
    the facade works in tests. Returns (facade, sync_mock); configure return
    values and assert calls on sync_mock."""
    from hermes_state import AsyncSessionDB
    sync_mock = sync_mock if sync_mock is not None else MagicMock()
    return AsyncSessionDB(sync_mock), sync_mock


class _FakeEnumMember(str):
    """A python-telegram-bot-faithful stand-in for a ``StrEnum`` member.

    PTB constants (``ParseMode``, ``ChatType``) are ``StrEnum`` members:
    ``str(x)`` and equality give the *value* (``"supergroup"``) while
    ``repr(x)`` shows the qualified *member name*
    (``<ChatType.SUPERGROUP>``). Test stubs that pick only one of those
    shapes break the other consumer: plain strings fail assertions like
    ``"MARKDOWN_V2" in repr(parse_mode)``, while auto-generated MagicMock
    attributes fail the adapter's ``str(chat.type)`` normalization
    (``adapter.py`` ``_build_message_event``). This class satisfies both,
    so every telegram test sees the same semantics regardless of which
    file's mock installed first.
    """

    _qualname: str

    def __new__(cls, enum_name: str, member_name: str, value: str):
        obj = str.__new__(cls, value)
        obj._qualname = f"{enum_name}.{member_name}"
        return obj

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self._qualname}: {str.__repr__(self)}>"


def _fake_str_enum(enum_name: str, **members: str):
    """Build a ``SimpleNamespace``-like enum of :class:`_FakeEnumMember`."""
    from types import SimpleNamespace

    return SimpleNamespace(
        **{name: _FakeEnumMember(enum_name, name, value) for name, value in members.items()}
    )


def _ensure_telegram_mock() -> None:
    """Install a comprehensive telegram mock in sys.modules.

    Idempotent — skips when the real library is already imported.
    Uses ``sys.modules[name] = mod`` (overwrite) instead of
    ``setdefault`` so it wins even if a partial/broken import
    already cached a module with ``ChatType = None``.
    """
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return  # Real library is installed — nothing to mock

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    # One shared PTB-faithful enum namespace per constant, attached to BOTH
    # access paths: ``sys.modules["telegram.constants"]`` is registered as
    # the root mock below, so ``from telegram.constants import ParseMode``
    # resolves ``mod.ParseMode`` — while config/docs-style access reads
    # ``telegram.constants.ParseMode``. Binding the same object to both
    # keeps every consumer comparing against identical members.
    _parse_mode = _fake_str_enum(
        "ParseMode", MARKDOWN="Markdown", MARKDOWN_V2="MarkdownV2", HTML="HTML"
    )
    _chat_type = _fake_str_enum(
        "ChatType",
        PRIVATE="private",
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
    )
    mod.ParseMode = _parse_mode
    mod.constants.ParseMode = _parse_mode
    mod.ChatType = _chat_type
    mod.constants.ChatType = _chat_type

    # Mirror PTB's exception hierarchy: BadRequest is a semantic API error,
    # but inherits from NetworkError in python-telegram-bot 22.x.
    mod.error.TelegramError = type("TelegramError", (Exception,), {})
    mod.error.NetworkError = type("NetworkError", (mod.error.TelegramError,), {})
    mod.error.TimedOut = type("TimedOut", (mod.error.NetworkError,), {})
    mod.error.BadRequest = type("BadRequest", (mod.error.NetworkError,), {})
    mod.error.Forbidden = type("Forbidden", (mod.error.TelegramError,), {})
    mod.error.InvalidToken = type("InvalidToken", (mod.error.TelegramError,), {})

    class RetryAfter(mod.error.TelegramError):
        def __init__(self, retry_after=1):
            self.retry_after = retry_after

    mod.error.RetryAfter = RetryAfter
    mod.error.Conflict = type("Conflict", (mod.error.TelegramError,), {})

    # Update.ALL_TYPES used in start_polling()
    mod.Update.ALL_TYPES = []

    for name in (
        "telegram",
        "telegram.ext",
        "telegram.constants",
        "telegram.request",
    ):
        sys.modules[name] = mod
    sys.modules["telegram.error"] = mod.error


def _ensure_discord_mock() -> None:
    """Install a comprehensive discord mock in sys.modules.

    Idempotent — skips when the real library is already imported.
    Uses ``sys.modules[name] = mod`` (overwrite) instead of
    ``setdefault`` so it wins even if a partial/broken import already
    cached the module.

    This mock is comprehensive — it includes **all** attributes needed by
    every gateway discord test file.  Individual test files should call
    this function (it short-circuits when already present) rather than
    maintaining their own mock setup.
    """
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return  # Real library is installed — nothing to mock

    from types import SimpleNamespace

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.Message = type("Message", (), {})

    # Embed: accept the kwargs production code / tests use
    # (title, description, color). MagicMock auto-attributes work too,
    # but some tests construct and inspect .title/.description directly.
    class _FakeEmbed:
        def __init__(self, *, title=None, description=None, color=None, **_):
            self.title = title
            self.description = description
            self.color = color
            self.fields = []
            self.footer = None
        def add_field(self, *, name=None, value=None, inline=False, **_):
            self.fields.append({"name": name, "value": value, "inline": inline})
            return self
        def set_footer(self, *, text=None, icon_url=None, **_):
            self.footer = {"text": text, "icon_url": icon_url}
            return self
    discord_mod.Embed = _FakeEmbed

    # ui.View / ui.Select / ui.Button: real classes (not MagicMock) so
    # tests that subclass ModelPickerView / iterate .children / clear
    # items work.
    class _FakeView:
        def __init__(self, timeout=None):
            self.timeout = timeout
            self.children = []
        def add_item(self, item):
            self.children.append(item)
        def clear_items(self):
            self.children.clear()

    class _FakeSelect:
        def __init__(self, *, placeholder=None, options=None, custom_id=None, **_):
            self.placeholder = placeholder
            self.options = options or []
            self.custom_id = custom_id
            self.callback = None
            self.disabled = False

    class _FakeButton:
        def __init__(self, *, label=None, style=None, custom_id=None, emoji=None,
                     url=None, disabled=False, row=None, sku_id=None, **_):
            self.label = label
            self.style = style
            self.custom_id = custom_id
            self.emoji = emoji
            self.url = url
            self.disabled = disabled
            self.row = row
            self.sku_id = sku_id
            self.callback = None

    class _FakeSelectOption:
        def __init__(self, *, label=None, value=None, description=None, **_):
            self.label = label
            self.value = value
            self.description = description
    discord_mod.SelectOption = _FakeSelectOption

    # AudioSource: real class so VoiceMixer(discord.AudioSource) can subclass
    # it cleanly in tests.  MagicMock auto-attributes would make is_opus()
    # return a Mock instead of False, breaking 9 TestVoiceMixerCore tests.
    class _FakeAudioSource:
        def is_opus(self):
            return False
        def read(self):
            return b"\x00" * 3840  # one silent stereo s16 frame
        def cleanup(self):
            pass
    discord_mod.AudioSource = _FakeAudioSource

    discord_mod.ui = SimpleNamespace(
        View=_FakeView,
        Select=_FakeSelect,
        Button=_FakeButton,
        button=lambda *a, **k: (lambda fn: fn),
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3,
        green=1, grey=2, blurple=2, red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3,
        red=lambda: 4, purple=lambda: 5, greyple=lambda: 6,
        gold=lambda: 7,
    )

    # app_commands — needed by _register_slash_commands auto-registration
    class _FakeGroup:
        def __init__(self, *, name, description, parent=None):
            self.name = name
            self.description = description
            self.parent = parent
            self._children: dict = {}
            if parent is not None:
                parent.add_command(self)

        def add_command(self, cmd):
            self._children[cmd.name] = cmd

    class _FakeCommand:
        def __init__(self, *, name, description, callback, parent=None):
            self.name = name
            self.description = description
            self.callback = callback
            self.parent = parent

    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        Group=_FakeGroup,
        Command=_FakeCommand,
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    for name in ("discord", "discord.ext", "discord.ext.commands"):
        sys.modules[name] = discord_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod


# Run at collection time — before any test file's module-level imports.
_ensure_telegram_mock()
_ensure_discord_mock()


# ---------------------------------------------------------------------------
# Plugin-adapter anti-pattern guard
# ---------------------------------------------------------------------------

_GATEWAY_DIR = Path(__file__).resolve().parent
_GUARD_HINT = (
    "Plugin adapter tests must use "
    "``from tests.gateway._plugin_adapter_loader import load_plugin_adapter`` "
    "and call ``load_plugin_adapter('<plugin_name>')`` instead of inserting "
    "``plugins/platforms/<name>/`` on sys.path and doing a bare ``import "
    "adapter`` / ``from adapter import ...``. See the 'Plugin-adapter "
    "anti-pattern guard' docstring in tests/gateway/conftest.py."
)


def _scan_for_plugin_adapter_antipattern(source: str) -> list[str]:
    """Return a list of offending-line descriptions, or [] if clean.

    Flags two things:
    1. ``sys.path.insert(..., <something mentioning 'plugins/platforms'>)``
    2. ``import adapter`` or ``from adapter import ...`` at module level.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # Let pytest surface the real syntax error.

    offenses: list[str] = []

    for node in ast.walk(tree):
        # sys.path.insert(0, ".../plugins/platforms/...")
        if isinstance(node, ast.Call):
            func = node.func
            target_name: str | None = None
            if isinstance(func, ast.Attribute):
                # sys.path.insert / sys.path.append
                if (
                    isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "sys"
                    and func.value.attr == "path"
                    and func.attr in {"insert", "append", "extend"}
                ):
                    target_name = f"sys.path.{func.attr}"

            if target_name is not None:
                call_src = ast.unparse(node)
                # Match both the string-literal form
                # ``.../plugins/platforms/...`` and the Path-operator form
                # ``Path(...) / 'plugins' / 'platforms' / ...`` that
                # plugin tests typically use.
                _src_no_ws = "".join(call_src.split())
                if (
                    "plugins/platforms" in call_src
                    or "plugins\\platforms" in call_src
                    or "'plugins'/'platforms'" in _src_no_ws
                    or '"plugins"/"platforms"' in _src_no_ws
                ):
                    offenses.append(
                        f"line {node.lineno}: {target_name}(...) points into "
                        f"plugins/platforms/"
                    )

    # Bare `import adapter` / `from adapter import ...` anywhere (module level
    # OR inside functions — both are symptoms of the same pattern).
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "adapter":
                    offenses.append(
                        f"line {node.lineno}: ``import adapter`` "
                        f"(bare — resolves to whichever plugin's adapter.py "
                        f"is first on sys.path)"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "adapter" and node.level == 0:
                offenses.append(
                    f"line {node.lineno}: ``from adapter import ...`` "
                    f"(bare — resolves to whichever plugin's adapter.py "
                    f"is first on sys.path)"
                )

    return offenses


def _fingerprint_gateway_tests() -> str:
    """Return a short fingerprint that changes when any gateway test file changes.

    Uses (mtime, size) pairs instead of content hashing — fast to compute
    (stat-only, no reads) and sufficient for cache invalidation across
    per-file subprocess runs.
    """
    import hashlib

    h = hashlib.sha256()
    for path in sorted(_GATEWAY_DIR.rglob("test_*.py")):
        try:
            st = path.stat()
            h.update(f"{path.name}:{st.st_mtime_ns}:{st.st_size}".encode())
        except OSError:
            h.update(f"{path.name}:missing".encode())
    return h.hexdigest()[:16]


def _run_adapter_antipattern_scan() -> list[str]:
    """Scan gateway test files for the plugin-adapter anti-pattern.

    Returns a list of violation strings (empty if clean).
    """
    violations: list[str] = []
    for path in _GATEWAY_DIR.rglob("test_*.py"):
        if path.name in {"_plugin_adapter_loader.py", "conftest.py"}:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Fast string pre-filter: skip files that can't possibly violate.
        # A violating file MUST contain both (a) an adapter/plugins/platforms
        # reference AND (b) either sys.path manipulation or a bare adapter import.
        if "adapter" not in source and "plugins/platforms" not in source:
            continue
        if not (
            "sys.path" in source
            or "import adapter" in source
            or "from adapter import" in source
        ):
            continue
        offenses = _scan_for_plugin_adapter_antipattern(source)
        if offenses:
            violations.append(
                f"  {path.relative_to(_GATEWAY_DIR.parent.parent)}:\n    "
                + "\n    ".join(offenses)
            )
    return violations


def pytest_configure(config):
    """Reject plugin-adapter tests that use the sys.path anti-pattern.

    Runs once per pytest session, before any test is collected. If any
    file under ``tests/gateway/`` matches the anti-pattern, we fail the
    whole session with a clear message — before a polluted ``sys.path``
    can cascade.

    **Performance**: in the per-file subprocess isolation model, every
    subprocess runs this hook — so the naive scan would run 257
    times, each costing ~1s of AST walking.  We avoid this with two
    strategies:

    1. **Tight string pre-filter**: a file can only violate if it contains
       *both* an adapter/plugins/platforms reference *and* a sys.path
       manipulation or bare ``import adapter``.  This drops ~95% of files
       from needing AST parsing.
    2. **File-locked cache**: the scan result is cached in
       ``.pytest-cache/gw-adapter-guard-<fingerprint>`` keyed on a
       fingerprint of the gateway test file mtimes/sizes.  Concurrent
       subprocesses acquire a lock; only the first performs the scan;
       the rest wait and read the cached result.
    """
    fp = _fingerprint_gateway_tests()
    cache_dir = Path.cwd() / ".pytest-cache"
    cache_file = cache_dir / f"gw-adapter-guard-{fp}"
    lock_file = cache_dir / f".gw-adapter-guard-{fp}.lock"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Evict stale cache entries from previous fingerprints (best-effort).
    # ``.tmp-gw-adapter-guard-*`` staging files (see
    # ``_write_guard_cache_atomic``) deliberately do NOT match these globs:
    # a sibling process evicting mid-publish must never unlink a temp file
    # between its write and its ``os.replace``.
    try:
        for old in cache_dir.glob("gw-adapter-guard-*"):
            if old.name != f"gw-adapter-guard-{fp}":
                old.unlink(missing_ok=True)
        for old in cache_dir.glob(".gw-adapter-guard-*.lock"):
            if old.name != f".gw-adapter-guard-{fp}.lock":
                old.unlink(missing_ok=True)
    except OSError:
        pass  # Non-critical; old files are harmless.

    # Use filelock to ensure only one process scans at a time.
    # Concurrent subprocesses all hit pytest_configure simultaneously;
    # without a lock they'd all find no cache and all run the scan.
    #
    # NOTE: filelock is NOT in CI's app and dev/test dependency closure,
    # so on CI the _NoLock fallback is
    # what actually runs. Correctness therefore cannot depend on the lock:
    # the cache write must be atomic and the read must tolerate a
    # not-yet-visible cache. Before the atomic-write fix, a reader could
    # observe a sibling's ``write_text`` mid-truncate, read "" and raise
    # ``UsageError("")`` — the empty ``ERROR:`` flake that hit
    # tests/gateway/relay/* files under the 96-way per-file runner.
    try:
        from filelock import FileLock
        lock = FileLock(str(lock_file), timeout=120)
    except ImportError:
        # Fallback: no locking (still correct, just slower under contention).

        class _NoLock:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        lock = _NoLock()

    with lock:
        cached = _read_guard_cache(cache_file)
        if cached == "clean":
            return
        if cached is not None:
            raise pytest.UsageError(cached)

        # Slow path: this process is the first to acquire the lock.
        violations = _run_adapter_antipattern_scan()

        if violations:
            msg = (
                "Plugin-adapter-import anti-pattern detected in gateway tests:\n"
                + "\n".join(violations)
                + "\n\n"
                + _GUARD_HINT
            )
            _write_guard_cache_atomic(cache_file, msg)
            raise pytest.UsageError(msg)
        else:
            _write_guard_cache_atomic(cache_file, "clean")


def _read_guard_cache(cache_file: Path) -> str | None:
    """Read the guard cache, treating unreadable/empty content as a miss.

    Returns the cached text, or ``None`` when the cache should be treated as
    absent (missing file, concurrent delete, or empty content). An empty or
    vanished file is NEVER a violation — before this guard existed, a reader
    racing a sibling's non-atomic write raised ``UsageError("")``, failing a
    perfectly healthy test file with a blank ``ERROR:`` line.
    """
    try:
        cached = cache_file.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    if not cached.strip():
        return None
    return cached


def _write_guard_cache_atomic(cache_file: Path, content: str) -> None:
    """Publish the guard cache atomically (write-temp + ``os.replace``).

    ``Path.write_text`` opens with truncation, so concurrent readers can see
    an empty or partial file. ``os.replace`` makes the new content appear
    all-at-once under the final name — readers see either the old state
    (miss → they scan too, harmless) or the complete verdict, never a
    torn write. This must hold WITHOUT the filelock, which CI doesn't
    install.

    The staging name starts with ``.tmp-`` so it can never match the stale
    eviction globs (``gw-adapter-guard-*`` / ``.gw-adapter-guard-*.lock``);
    otherwise a sibling's eviction pass could unlink the temp between the
    write and the ``os.replace``.
    """
    tmp = cache_file.with_name(f".tmp-{cache_file.name}.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, cache_file)
    except OSError:
        # Best-effort cache: losing it costs a rescan, never correctness.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

