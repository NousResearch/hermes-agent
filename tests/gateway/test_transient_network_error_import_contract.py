"""Import-contract tests for the shared transient-network-error classifier.

Background (#84210). ``TelegramAdapter._download_media_with_retry`` used to
decide "retry or surface?" via a **call-time import of a private symbol in
another module**::

    async def _download_media_with_retry(self, source, kind):
        from gateway.run import _is_transient_network_error   # ← call time, private
        ...

Two things were wrong with that, and this file pins the fix for both.

1. **Wrong coupling.** A plugin platform adapter reaching into
   ``gateway.run``'s privates is not a supported seam. The classifier now
   lives in ``gateway.platforms.helpers`` — the module plugin adapters
   already import shared gateway helpers from (``MessageDeduplicator``,
   ``strip_markdown``, ``compile_mention_patterns``, … consumed by 12+
   adapters) — and BOTH ``gateway.run`` and the Telegram adapter consume it
   from there.

2. **Silent degradation.** Because the import sat *inside* the method, a
   rename in ``gateway.run`` would not fail at startup or at import time. It
   would raise ``ImportError`` on the first media download — and every caller
   of ``_download_media_with_retry`` wraps the call in a broad
   ``except Exception``, which converts that ``ImportError`` into the same
   generic "media could not be downloaded" reply a real CDN failure produces.
   Retries would stop happening, permanently and invisibly, with the actual
   cause buried. (Verified empirically: ``ImportError`` IS an ``Exception``
   subclass, so the existing handlers do swallow it.)

   Hoisting the import to module scope converts that silent runtime
   degradation into a loud import-time failure — nothing can even construct
   the adapter. These tests assert the hoist, so nobody re-introduces the
   call-time form.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import gateway.platforms.helpers as helpers
import gateway.run as gateway_run

# The canonical, supported location. Renaming/moving this symbol without
# updating the consumers below breaks these tests.
CANONICAL_MODULE = "gateway.platforms.helpers"
CANONICAL_NAME = "is_transient_network_error"


# ---------------------------------------------------------------------------
# 1. The shared symbol exists at the canonical location and is importable.
# ---------------------------------------------------------------------------


def test_classifier_lives_at_the_canonical_shared_location():
    """The promoted classifier is importable from the shared helpers module."""
    fn = getattr(helpers, CANONICAL_NAME, None)
    assert callable(fn), (
        f"{CANONICAL_MODULE}.{CANONICAL_NAME} is missing. It is the single "
        "definition consumed by gateway/run.py AND the telegram adapter; if "
        "you moved it, update both consumers and this contract."
    )
    assert fn.__module__ == CANONICAL_MODULE


def test_shared_helpers_module_does_not_import_gateway_run():
    """The shared module must not depend on the god-file it was extracted from.

    ``gateway/run.py`` imports ``gateway.platforms.helpers`` at module scope,
    so a reverse edge would be a genuine import cycle rather than a style nit.
    """
    tree = ast.parse(Path(inspect.getfile(helpers)).read_text())
    offenders = [
        f"L{node.lineno}"
        for node in ast.walk(tree)
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gateway.run"))
        or (
            isinstance(node, ast.Import)
            and any(a.name.startswith("gateway.run") for a in node.names)
        )
    ]
    assert offenders == [], (
        f"{CANONICAL_MODULE} imports gateway.run at {offenders} — that is an "
        "import cycle; gateway/run.py imports this module at module scope."
    )


# ---------------------------------------------------------------------------
# 2. Both consumers reference the shared location, at MODULE scope.
# ---------------------------------------------------------------------------


def _shared_classifier_imports(module) -> list[tuple[int, bool]]:
    """Return ``(lineno, is_module_level)`` for each import of the classifier.

    Discovers every import site dynamically rather than checking a hardcoded
    line, so a moved import is still found and a *new* offending call-time
    import cannot hide.
    """
    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    module_level_nodes = {id(node) for node in tree.body}

    found: list[tuple[int, bool]] = []

    def _matches(node: ast.AST) -> bool:
        if isinstance(node, ast.ImportFrom):
            return (node.module or "") == CANONICAL_MODULE and any(
                alias.name == CANONICAL_NAME for alias in node.names
            )
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _matches(node):
            found.append((node.lineno, id(node) in module_level_nodes))
    return found


@pytest.mark.parametrize(
    "module, label",
    [
        (gateway_run, "gateway/run.py"),
        pytest.param(None, "plugins/platforms/telegram/adapter.py", id="telegram-adapter"),
    ],
)
def test_both_consumers_import_the_classifier_at_module_scope(module, label):
    """Both consumers import the SHARED classifier, and do so at module scope.

    A call-time (function-body) import is rejected explicitly: that is the
    exact shape whose failure mode is silent, since the callers' broad
    ``except Exception`` handlers swallow the resulting ``ImportError``.
    """
    if module is None:
        from plugins.platforms.telegram import adapter as module

    sites = _shared_classifier_imports(module)
    assert sites, (
        f"{label} does not import {CANONICAL_NAME} from {CANONICAL_MODULE}. "
        "Both gateway/run.py and the telegram adapter must consume the ONE "
        "shared classifier — do not re-derive or re-copy the class-name set."
    )
    call_time = [lineno for lineno, module_level in sites if not module_level]
    assert call_time == [], (
        f"{label} imports {CANONICAL_NAME} inside a function body at line(s) "
        f"{call_time}. Hoist it to module scope: a call-time import fails only "
        "on the first media download, where the callers' broad `except "
        "Exception` turns the ImportError into a generic 'could not be "
        "downloaded' reply and retries silently stop."
    )


def test_telegram_adapter_does_not_reach_into_gateway_run_privates():
    """No import of a ``_private`` symbol from ``gateway.run`` in the adapter.

    Scoped to the whole module (not just ``_download_media_with_retry``) so a
    future handler cannot re-open the same seam somewhere else in the file.
    """
    from plugins.platforms.telegram import adapter

    tree = ast.parse(Path(inspect.getfile(adapter)).read_text())
    offenders = [
        f"L{node.lineno}: from {node.module} import {alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gateway.run")
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert offenders == [], (
        "The telegram adapter imports private symbols from gateway.run: "
        f"{offenders}. Promote what you need to a shared module (e.g. "
        f"{CANONICAL_MODULE}) and import it from there."
    )


# ---------------------------------------------------------------------------
# 3. Single source of truth — no duplicated class-name set.
# ---------------------------------------------------------------------------


def test_classifier_class_name_set_is_defined_exactly_once():
    """Only the shared module may define the transient class-name set.

    Two copies would drift: the loop-level safety net and the media-retry
    ladder would then disagree about which errors are survivable.
    """
    assert hasattr(helpers, "TRANSIENT_NETWORK_ERROR_CLASS_NAMES")
    names = helpers.TRANSIENT_NETWORK_ERROR_CLASS_NAMES
    assert "TimedOut" in names and "ConnectError" in names
    assert "BadRequest" not in names, (
        "BadRequest is permanent (a wrong request), even though PTB 22.x "
        "makes it a NetworkError subclass. Retrying it wastes three attempts."
    )

    repo_root = Path(inspect.getfile(helpers)).resolve().parents[2]
    for rel in ("gateway/run.py", "plugins/platforms/telegram/adapter.py"):
        src = (repo_root / rel).read_text()
        assert "transient_class_names" not in src, (
            f"{rel} appears to define its own transient class-name set. There "
            f"must be exactly one, in {CANONICAL_MODULE}."
        )


def test_gateway_run_alias_is_a_thin_delegation_not_a_second_copy():
    """``gateway.run._is_transient_network_error`` must delegate, not re-implement.

    The alias is kept for existing importers, but it may not carry a second
    implementation — that is the drift this promotion exists to prevent.
    """
    body = inspect.getsource(gateway_run._is_transient_network_error)
    assert CANONICAL_NAME in body, (
        "gateway.run._is_transient_network_error no longer delegates to "
        f"{CANONICAL_MODULE}.{CANONICAL_NAME}."
    )
    # A delegating alias is short; a re-implementation walks a cause chain.
    assert "__cause__" not in body, (
        "gateway.run._is_transient_network_error looks re-implemented (it "
        "walks the cause chain itself). It must be a thin re-export."
    )
