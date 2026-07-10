"""Shared fixtures for gateway tests.

The ``_ensure_telegram_mock`` helper guarantees that a minimal mock of
the ``telegram`` package is registered in :data:`sys.modules` **before**
any test file triggers ``from plugins.platforms.telegram.adapter import ...``.

Without this, ``pytest-xdist`` workers that happen to collect
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
``sys.modules["adapter"]`` in the same xdist worker. Whichever collects
first wins; the other fails with ``ImportError``, and the polluted
``sys.path`` cascades into unrelated tests. See PR #17764 for the
incident.
"""

import ast
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _restore_os_environ_after_test(_hermetic_environment):
    """Snapshot ``os.environ`` at setup and diff-restore it at teardown.

    **Why:** ``load_gateway_config()`` (``gateway/config.py``) bridges
    ``config.yaml`` platform settings into the process environment via *raw
    assignment* — ``os.environ["TELEGRAM_ALLOWED_TOPICS"] = ...``,
    ``os.environ["SLACK_ALLOWED_CHANNELS"] = ...`` and ~40 sibling keys across 7
    platforms. A raw ``os.environ[...] =`` write is **NOT** reverted by
    ``monkeypatch`` (monkeypatch only undoes what *it* set), so any test body that
    calls the loader leaks those values into every later test in the same
    process. A later test that builds a real ``SlackAdapter``/``TelegramAdapter``
    then reads the leaked gating var and silently changes behavior (e.g. a leaked
    ``TELEGRAM_ALLOWED_TOPICS=8`` makes a general-topic guest message fail the
    allowed-topics gate; a leaked ``SLACK_ALLOWED_CHANNELS`` drops a mention
    before ``handle_message``). These only bite single-process / random-order
    runs (the per-file CI runner gets a fresh interpreter), so CI is green while
    a local ``-p randomly`` run fails.

    **The fix is by construction and immune to *how* the bridge codes the write
    (literal, loop, ``update``, ``setdefault``, computed key):** snapshot the
    whole environment at setup and re-establish it exactly at teardown. There is
    no enumerated var list anywhere — the source of truth is the live
    ``os.environ`` — so the fix can never drift as the bridge adds vars, and it
    fails *closed* (anything a test mutates is reverted).

    **Scope & ordering:** gateway-scoped (this conftest), so the blast radius
    equals the suite whose evidence we run. It explicitly depends on the root
    ``_hermetic_environment`` fixture (requested as a parameter) so it is
    guaranteed to snapshot the *already-clean* env (after the root fixture's
    HERMES_HOME redirect + credential/behavioral strip) and to finalize *before*
    the root fixture's monkeypatch undo — the dependency is explicit so a future
    autouse-ordering change can't silently invert it.

    **What this does NOT do (honest scope):** it reverts only *test-induced*
    mutations. A bridge-gating var inherited from the CI/dev shell at session
    start is faithfully *preserved* (not stripped) — ambient-env hygiene is the
    root ``_hermetic_environment`` strip list's job, not this fixture's.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        # Re-establish the exact snapshot: revert test-added, test-removed, and
        # test-changed keys in one shot. (clear()+update() over the live mapping
        # so the same os.environ object identity is preserved for any code
        # holding a reference.)
        if os.environ != snapshot:
            os.environ.clear()
            os.environ.update(snapshot)


# Module slots whose identity several gateway test files mutate at import (by
# installing a mock ``telegram``/``discord`` into ``sys.modules`` — often via
# ``setdefault`` so the FIRST file to run wins the slot — and by popping +
# reimporting the consumer). A file that does this without restoring leaks two
# things into every later single-process test:
#   1. the ``sys.modules`` slot itself (a foreign mock with different type
#      objects — e.g. ``discord.DMChannel`` is a different class than the one a
#      later test patches, so its ``isinstance`` checks silently mismatch);
#   2. the CONSUMER module's early-bound globals — ``gateway.platforms.telegram``
#      caches ``ParseMode``/``ChatType`` from whatever ``telegram.constants`` was
#      live at its import; ``plugins.platforms.discord.adapter`` caches the
#      ``discord`` module object — so a rebind during one test poisons the
#      identity every later test reads (the PR #89 ParseMode-plain-string leak,
#      reappearing under random order via a different leaker than the one #89
#      fixed; and the discord-mock-identity leak).
_GUARDED_SYS_MODULE_SLOTS = (
    "telegram",
    "telegram.ext",
    "telegram.constants",
    "telegram.request",
    "telegram.error",
    "discord",
    "discord.ext",
    "discord.ext.commands",
)

# Consumer modules + the attribute names they early-bind from the slots above.
# We snapshot/restore these attributes IN PLACE on the live module object (NOT by
# re-import — a fresh import makes a NEW module object, but consumers that did
# ``from gateway.platforms.telegram import X`` at their module top read X from the
# ORIGINAL module __dict__, so the live dict is what must be reverted).
_GUARDED_CONSUMER_BINDINGS = {
    "gateway.platforms.telegram": ("ParseMode",),
    "plugins.platforms.discord.adapter": ("discord",),
}


@pytest.fixture(autouse=True)
def _restore_mock_module_slots_after_test(_hermetic_environment):
    """Snapshot + restore the ``sys.modules`` mock slots and consumer bindings.

    By construction (no per-file fixture to forget): any test that mutates a
    guarded ``telegram``/``discord`` ``sys.modules`` slot, or rebinds a consumer
    module's early-bound ``ParseMode``/``ChatType``/``discord`` global, has that
    mutation reverted at teardown — so it cannot poison a later test in a
    single-process / random-order run. Mirrors the ``os.environ`` snapshot/restore
    fixture above, applied to ``sys.modules`` + the consumer bindings.

    Only reverts state that actually CHANGED during the test (identity compare),
    so the steady-state shared mocks the suite relies on are untouched on the
    overwhelming majority of tests that don't mutate these slots.

    **Setup-time normalization (the collection-time-poison case):** some leaker
    files install a plain-STRING ParseMode mock (``ParseMode.MARKDOWN_V2 =
    "MarkdownV2"``) at *module import / collection* time — BEFORE any test setup
    runs — via ``setdefault`` (so they win the slot only when it's empty). A
    snapshot-then-restore can't fix that: by this test's setup the consumer is
    ALREADY poisoned, so the snapshot captures the poison. So at setup we also
    *detect the poison signature* (consumer ``ParseMode.MARKDOWN_V2`` is a plain
    ``str`` whose repr lost the ``MARKDOWN_V2`` member name) and re-bind the
    consumer's ``ParseMode``/``ChatType`` to the canonical conftest telegram mock,
    so every test starts from a good binding regardless of collection order. (The
    canonical mock's members are MagicMock attributes whose repr DOES carry the
    member name — what the real ``StringEnum`` and the telegram tests rely on.)
    """
    _normalize_poisoned_telegram_consumer_binding()
    sentinel = object()
    slot_snapshot = {name: sys.modules.get(name, sentinel) for name in _GUARDED_SYS_MODULE_SLOTS}
    binding_snapshot = {}
    for mod_name, attrs in _GUARDED_CONSUMER_BINDINGS.items():
        mod = sys.modules.get(mod_name)
        if mod is not None:
            binding_snapshot[mod_name] = {a: getattr(mod, a, sentinel) for a in attrs}
    try:
        yield
    finally:
        # 1. Restore the sys.modules slots to their pre-test identity.
        for name, prior in slot_snapshot.items():
            current = sys.modules.get(name, sentinel)
            if current is prior:
                continue
            if prior is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
        # 2. Restore consumer early-bound globals IN PLACE on the live module.
        for mod_name, attrs in binding_snapshot.items():
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            for attr, prior in attrs.items():
                if getattr(mod, attr, sentinel) is prior:
                    continue
                if prior is sentinel:
                    if hasattr(mod, attr):
                        try:
                            delattr(mod, attr)
                        except AttributeError:
                            pass
                else:
                    setattr(mod, attr, prior)


def _parse_mode_is_poisoned(parse_mode) -> bool:
    """True when a ParseMode object is the leaked plain-string poison.

    The poison signature: ``MARKDOWN_V2`` is a plain ``str`` (``"MarkdownV2"``)
    whose repr LOST the member name. A real ``StringEnum`` member and the gateway
    test env's ``MagicMock`` member both keep ``MARKDOWN_V2`` in their repr — only
    the leaked plain string does not. (Asserting on repr, not ``==``: ParseMode is
    a ``str`` subclass so ``MARKDOWN_V2 == "MarkdownV2"`` is True for BOTH the real
    enum and the poison — a ``==`` check can't tell them apart.)
    """
    member = getattr(parse_mode, "MARKDOWN_V2", None)
    return isinstance(member, str) and "MARKDOWN_V2" not in repr(member)


def _normalize_poisoned_telegram_consumer_binding() -> None:
    """If the telegram consumer's ParseMode is the leaked plain string, re-bind it.

    Corrects a collection-time poison (a leaker file that installed a plain-string
    ParseMode mock via ``setdefault`` before any test setup ran) so each test
    starts from a good binding. No-op when the binding is already healthy or the
    real telegram library is installed.

    The healthy replacement is a ``MagicMock`` ParseMode whose members keep their
    name in ``repr`` (``<MagicMock name='...ParseMode.MARKDOWN_V2'>``) — matching
    what the real ``StringEnum`` member and the normal gateway-test mock provide,
    and what the telegram tests assert via ``"MARKDOWN_V2" in repr(parse_mode)``.
    """
    consumer = sys.modules.get("gateway.platforms.telegram")
    if consumer is None:
        return
    # If the real telegram library is installed, never touch the binding.
    tg = sys.modules.get("telegram")
    if tg is not None and hasattr(tg, "__file__"):
        return
    parse_mode = getattr(consumer, "ParseMode", None)
    if parse_mode is None or not _parse_mode_is_poisoned(parse_mode):
        return
    # Build a healthy ParseMode whose members keep their name in repr. ONLY
    # ParseMode is normalized — ChatType is NOT touched: ChatType carries
    # meaningful string values (PRIVATE="private", GROUP="group", …) that tests
    # compare against (chat_type == "group"), so replacing it with a generic
    # MagicMock whose members return arbitrary mocks would BREAK those tests. The
    # poison only ever affects ParseMode (the plain-string MARKDOWN_V2), so the
    # repair is scoped to it.
    healthy_parse_mode = MagicMock(name="ParseMode")
    consumer.ParseMode = healthy_parse_mode
    # Keep the sys.modules telegram.constants slot's ParseMode consistent so a
    # later consumer re-import also sees a healthy binding.
    constants = sys.modules.get("telegram.constants")
    if constants is not None:
        try:
            constants.ParseMode = healthy_parse_mode
        except Exception:
            pass



def make_async_session_db(sync_mock=None):
    """Wrap a sync mock SessionDB in AsyncSessionDB so gateway code that awaits
    the facade works in tests. Returns (facade, sync_mock); configure return
    values and assert calls on sync_mock."""
    from hermes_state import AsyncSessionDB
    sync_mock = sync_mock if sync_mock is not None else MagicMock()
    return AsyncSessionDB(sync_mock), sync_mock


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
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"

    # Real exception classes so ``except (NetworkError, ...)`` clauses
    # in production code don't blow up with TypeError.
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    mod.error.Forbidden = type("Forbidden", (Exception,), {})
    mod.error.InvalidToken = type("InvalidToken", (Exception,), {})
    mod.error.RetryAfter = type("RetryAfter", (Exception,), {"retry_after": 1})
    mod.error.Conflict = type("Conflict", (Exception,), {})

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


_ORIG_ARGV: list[str] | None = None


def _sanitize_pytest_randomly_argv(config) -> None:
    """Reduce ``sys.argv`` to just the program name (in place).

    Several gateway tests import ``hermes_cli.main`` inside a test body. That
    module runs ``_apply_profile_override()`` at import time, which scans the live
    ``sys.argv`` for a ``-p <profile>`` flag. Under pytest **any** ``-p <name>``
    plugin flag (``-p randomly``, ``-p no:cacheprovider``, …) is mis-read as a
    Hermes profile — ``-p randomly`` resolves to a nonexistent profile and the
    import ``sys.exit(1)``s, killing whichever test triggers the first
    ``hermes_cli.main`` import during the run. Because the collision is general to
    pytest's ``-p`` (not specific to pytest-randomly — review R6), we blanket-reduce
    argv to ``[argv0]`` rather than maintain an incomplete token allowlist. Gateway
    tests never read the process ``sys.argv`` (verified: zero ``sys.argv`` reads in
    ``tests/gateway/``), so this is side-effect-free for them. Captured ONCE and
    restored at session end (INV-6); re-entrancy-safe under xdist (review R5).
    """
    global _ORIG_ARGV
    # Capture-once: a 2nd pytest_configure (xdist controller+workers, or a re-run)
    # must NOT re-capture an already-reduced argv, or the restore would write garbage.
    if _ORIG_ARGV is not None:
        return
    _ORIG_ARGV = list(sys.argv)
    if sys.argv:
        sys.argv[:] = [sys.argv[0]]


def _restore_sanitized_argv() -> None:
    """Restore ``sys.argv`` to its pre-sanitize value (session teardown — INV-6)."""
    global _ORIG_ARGV
    if _ORIG_ARGV is not None:
        sys.argv[:] = _ORIG_ARGV
        _ORIG_ARGV = None


def pytest_unconfigure(config):  # noqa: D401 — pytest hook
    """Restore the argv we sanitized in ``pytest_configure`` (INV-6)."""
    _restore_sanitized_argv()


def pytest_configure(config):
    """Reject plugin-adapter tests that use the sys.path anti-pattern.

    Runs once per pytest session on the controller, BEFORE any xdist
    worker is spawned. If any file under ``tests/gateway/`` matches the
    anti-pattern, we fail the whole session with a clear message —
    before a polluted ``sys.path`` can cascade across workers.

    **Performance**: in the per-file subprocess isolation model (no xdist),
    every subprocess is a "controller" — so the naive scan would run 257
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
    # --- Random-order argv-collision guard (runs on EVERY process incl. workers) ---
    # Several gateway tests import ``hermes_cli.main`` inside a test body. That
    # module runs ``_apply_profile_override()`` at import time (main.py:508), which
    # scans the live ``sys.argv`` for a ``-p <profile>`` flag. Under pytest-randomly
    # the argv carries ``-p randomly`` / ``--randomly-seed=N``; ``_apply_profile_override``
    # parses ``randomly`` as a Hermes profile, fails to resolve it, and calls
    # ``sys.exit(1)`` — killing whichever test triggers the first ``hermes_cli.main``
    # import during a ``-p randomly`` run (a seed-dependent victim, a constant cause).
    # Because the collision is GENERAL to pytest's ``-p <plugin>`` flag (not specific to
    # randomly), we BLANKET-reduce argv to ``[argv0]`` rather than maintain an incomplete
    # token allowlist; gateway tests never read the process argv. Restored at session end
    # (see _restore_sanitized_argv / pytest_unconfigure).
    _sanitize_pytest_randomly_argv(config)

    # Only run on the xdist controller (or in non-xdist runs). Skip on
    # worker subprocesses so we don't scan the filesystem N times.
    if hasattr(config, "workerinput"):
        return

    fp = _fingerprint_gateway_tests()
    cache_dir = Path.cwd() / ".pytest-cache"
    cache_file = cache_dir / f"gw-adapter-guard-{fp}"
    lock_file = cache_dir / f".gw-adapter-guard-{fp}.lock"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Evict stale cache entries from previous fingerprints (best-effort).
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
        if cache_file.exists():
            cached = cache_file.read_text(encoding="utf-8")
            if cached == "clean":
                return
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
            cache_file.write_text(msg, encoding="utf-8")
            raise pytest.UsageError(msg)
        else:
            cache_file.write_text("clean", encoding="utf-8")

