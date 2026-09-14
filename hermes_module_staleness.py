"""Recover a long-lived Hermes process from a checkout updated in place beneath it.

``git pull`` / ``hermes update`` rewrites modules on disk while a running process keeps the
PREVIOUS code in ``sys.modules``. A later lazy ``from <module> import <new_symbol>`` then dies
with ``ImportError: cannot import name ...`` even though the source on disk HAS the symbol: the
import machinery returns the cached module object, which predates the symbol.

Field failure (2026-09-13, desktop ``hermes serve``): the checkout was pulled at 22:33, adding
``gateway.session.profile_from_session_key_namespace`` and ``gateway.run``'s import of it, while a
TUI gateway process cached ``gateway.session`` from before. ``_tui_compression_config_signature``'s
lazy ``from gateway.run import GatewayRunner`` then raised on every turn until the process died.

The updater already owns this class for its OWN process (``_purge_stale_hermes_modules`` in
``hermes_cli/update_cmd_maint.py``, which delegates here). This module additionally exposes
:func:`import_symbol` for any long-lived process that lazily imports Hermes code: the frozen
gateway/desktop backends, dashboards, and the TUI.

Stdlib-only and import-safe — importable from anywhere without circular-import risk.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Iterable, MutableMapping, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Package roots whose cached modules go stale when the checkout changes under a running
#: process. Purged (not reloaded) so any LATER import chain resolves against fresh source.
STALE_MODULE_PREFIXES: Tuple[str, ...] = ("hermes_cli", "gateway", "tools", "tui_gateway", "agent")

#: Modules EXECUTING the purge survive it: evicting a running module buys nothing (frames keep
#: the object) and reloading it mid-flight is the one genuinely unsafe move.
PURGE_PROTECTED: frozenset = frozenset({"hermes_cli", "hermes_cli.main", "hermes_cli.hermes_logging"})

#: The updater's own module family (``update_cmd*``, ``update_receipt``, ``update_inventory``,
#: ``update_lock``, ...) is protected as a prefix: these hold per-run state — the open receipt
#: singleton, the pre-update plan's ``RuntimeRecord`` class identity, the lock — and evicting one
#: swaps in a fresh module whose ``_current`` is None (receipt silently never written) or whose
#: dataclass fails every ``isinstance`` against the plan built before the purge.
PURGE_PROTECTED_PREFIX: str = "hermes_cli.update_"


def purge_stale_modules(
    prefixes: Sequence[str] = STALE_MODULE_PREFIXES,
    protect: Iterable[str] = PURGE_PROTECTED,
    protect_prefix: str = PURGE_PROTECTED_PREFIX,
    modules: MutableMapping | None = None,
) -> list:
    """Evict every cached Hermes module so later imports rebuild from the on-disk checkout.

    Dropping the ``sys.modules`` entry (unlike ``importlib.reload``) leaves running frames with
    their module objects, so evicting the caller's own package is safe; a later import of it just
    builds a second, fresh object. Returns the purged names. Never raises.
    """
    registry = sys.modules if modules is None else modules
    protect = frozenset(protect)
    purged: list = []
    try:
        importlib.invalidate_caches()
    except Exception:  # pragma: no cover - invalidate_caches is a no-op wrapper
        pass
    for name in list(registry):
        if name in protect or name.startswith(protect_prefix):
            continue
        # Root-segment check: startswith() alone also matches unrelated ``gateway_foo``.
        if name.split(".", 1)[0] in prefixes and registry.pop(name, None) is not None:
            purged.append(name)
    return purged


def _module_root(name: str) -> str:
    """Top-level package segment of a dotted module name."""
    return name.split(".", 1)[0]


def evict_modules(names: Iterable[str], modules: MutableMapping | None = None) -> list:
    """Evict exactly *names* (plus cached submodules of any that is a package). Returns them.

    The surgical twin of :func:`purge_stale_modules`: used when the *specific* stale modules are
    known, so only they — and not every Hermes module — are rebuilt from disk. Never raises.
    """
    registry = sys.modules if modules is None else modules
    purged: list = []
    try:
        importlib.invalidate_caches()
    except Exception:  # pragma: no cover - invalidate_caches is a no-op wrapper
        pass
    for name in names:
        victims = [cached for cached in list(registry) if cached == name or cached.startswith(name + ".")]
        for victim in victims:
            if registry.pop(victim, None) is not None:
                purged.append(victim)
    return purged


def implicated_modules(module_name: str, exc: BaseException, *, allowed_roots: Iterable[str] = ()) -> list:
    """Cached modules whose staleness produced *exc*, most-specific first.

    The stale module is rarely the one being imported: ``from gateway.run import GatewayRunner``
    fails because a module ``gateway.run`` imports (``e.name``, e.g. ``gateway.session``) predates
    a symbol the new source wants — or because a module *that* one imports does (a traceback frame,
    e.g. ``agent.session_activity``). Both are collected, filtered to Hermes roots (plus
    *allowed_roots*, so callers/tests can scope to their own packages) and de-duplicated.
    """
    allowed = {_module_root(name) for name in allowed_roots} | set(STALE_MODULE_PREFIXES)
    candidates: list = []
    name = getattr(exc, "name", None)
    if isinstance(name, str) and name:
        candidates.append(name)
    tb = exc.__traceback__
    while tb is not None:
        frame_module = tb.tb_frame.f_globals.get("__name__")
        if isinstance(frame_module, str):
            candidates.append(frame_module)
        tb = tb.tb_next
    candidates.append(module_name)
    out: list = []
    for candidate in candidates:
        if candidate not in out and _module_root(candidate) in allowed:
            out.append(candidate)
    return out


def import_symbol(
    module_name: str, symbol: str, *, retry_prefixes: Sequence[str] = ("gateway",),
    escalate_prefixes: Sequence[str] | None = None,
):
    """``from module_name import symbol``, healing a stale-module ImportError once.

    Only a failure inside a package named by *retry_prefixes* is treated as the stale-cache shape:
    a module cached before an in-place update no longer carries the symbol the (new) source needs.
    The *implicated* modules are rebuilt from disk and the import retried; if that was not enough
    (an intermediate sibling is stale too) the whole *escalate_prefixes* subtree is purged and the
    import retried once more. An unrelated ImportError (missing module, absent dependency) re-raises
    untouched rather than evicting the process's Hermes modules, and a symbol that is genuinely gone
    raises ImportError, matching ``from ... import``.
    """
    try:
        return getattr(importlib.import_module(module_name), symbol)
    except (ImportError, AttributeError) as failure:
        # ``from X import Y`` raises ImportError where ``getattr`` raises AttributeError: accept
        # both so a cached module simply missing the new symbol heals like a broken sibling import.
        if _module_root(module_name) not in tuple(retry_prefixes):
            raise
        # ``exc`` is unbound once the handler exits; keep the traceback for the eviction decision.
        exc = failure

    implicated = implicated_modules(module_name, exc, allowed_roots=retry_prefixes)
    purged = evict_modules(implicated)
    logger.warning(
        "Stale module cache blocked import of %s.%s (checkout updated in place?); "
        "evicted %d cached module(s) [%s] and retrying",
        module_name,
        symbol,
        len(purged),
        ", ".join(implicated) or "none",
    )
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        widen = tuple(escalate_prefixes if escalate_prefixes is not None else retry_prefixes)
        purged = purge_stale_modules(prefixes=widen)
        logger.warning(
            "Surgical eviction was not enough for %s.%s; purged %d module(s) under %s and retrying",
            module_name,
            symbol,
            len(purged),
            ", ".join(widen),
        )
        module = importlib.import_module(module_name)
    if not hasattr(module, symbol):
        raise ImportError(f"cannot import name {symbol!r} from {module_name!r}")
    return getattr(module, symbol)
