"""Plugin compatibility with the Sep 2026 decomposition: detect, warn, and (after the date) disable.

The decomposition (PR #102117) moved most of Hermes's internals into ``<stem>_<topic>`` sibling modules.
Old import paths keep resolving through ``PLUGIN-COMPAT`` blocks until :data:`COMPAT_REMOVAL_DATE`, when
the commit that added them is reverted. This module is the single source of truth for everything that
tells plugin authors and users about that:

* :func:`scan_plugin` — static AST scan of one plugin directory for imports of manifest names.
* :func:`compat_report` — ``{plugin_name: [Hit, ...]}`` across the user's ENABLED external plugins, cached.
* :func:`removal_in_effect` — True once today >= the removal date (or the layer is already gone).
* :func:`warn_once` — the per-name runtime warning emitted by the PLUGIN-COMPAT ``__getattr__`` blocks.

Surfaces that read from here: the CLI banner, ``hermes plugins compat``, ``hermes doctor``, the post-update
notices, the TUI/Desktop ``plugins.compat_report`` RPC, and ``PluginManager`` (which skips a hitting plugin
after the date unless ``plugins.allow_deprecated_imports: true``).

This module is part of the compat layer and is removed with it.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

COMPAT_REMOVAL_DATE = _dt.date(2026, 9, 14)
COMPAT_REMOVAL = COMPAT_REMOVAL_DATE.isoformat()
ALLOW_KEY = "allow_deprecated_imports"   # under plugins: in config.yaml
_MANIFEST_NAME = "compat_manifest.json"
_SKIP_DIRS = {"__pycache__", "node_modules", ".git", "tests", "test", ".venv", "venv"}


class HermesPluginCompatWarning(FutureWarning):
    """A plugin imported a name from its pre-decomposition module path."""


@dataclass(frozen=True)
class Hit:
    file: str          # path relative to the plugin dir
    line: int
    old: str           # "facade.name"
    new: str           # "target_module.name" (or the target module when the name is unchanged)


# ---------------------------------------------------------------------------------------------- manifest

_manifest_lock = threading.Lock()
_manifest_cache: Optional[Dict[str, Dict[str, str]]] = None   # facade -> {name: new_path}


def manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent / _MANIFEST_NAME


def load_manifest() -> Dict[str, Dict[str, str]]:
    """``{facade_module: {name: new_dotted_path}}``; ``{}`` when the compat layer is gone."""
    global _manifest_cache
    with _manifest_lock:
        if _manifest_cache is not None:
            return _manifest_cache
        out: Dict[str, Dict[str, str]] = {}
        p = manifest_path()
        if p.exists():
            try:
                for e in json.loads(p.read_text(encoding="utf-8"))["entries"]:
                    target = e.get("target") or ""
                    if target.startswith("("):            # restored-def etc.: no new home, just "gone later"
                        new = f"{e['facade']}.{e['name']} (removed; no replacement — vendor a copy)"
                    elif target.endswith("." + e["name"]):
                        new = target
                    else:
                        new = f"{target}.{e['name']}"
                    out.setdefault(e["facade"], {})[e["name"]] = new
            except Exception:
                out = {}
        _manifest_cache = out
        return out


def removal_in_effect(today: Optional[_dt.date] = None) -> bool:
    """True when hitting plugins must be disabled: the date has passed or the layer is already reverted."""
    if not manifest_path().exists():
        return True
    return (today or _dt.date.today()) >= COMPAT_REMOVAL_DATE


def days_until_removal(today: Optional[_dt.date] = None) -> int:
    return (COMPAT_REMOVAL_DATE - (today or _dt.date.today())).days


# ---------------------------------------------------------------------------------------------- scanner

def _iter_py(root: Path) -> Iterable[Path]:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in fns:
            if f.endswith(".py"):
                yield Path(dp) / f


_IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    """A handler that swallows the failed import. A bare ``raise`` body propagates it, so the plugin
    still dies on the cutoff and the import is not protected."""
    if len(handler.body) == 1 and isinstance(handler.body[0], ast.Raise) and handler.body[0].exc is None:
        return False
    t = handler.type
    if t is None:
        return True
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(n, ast.Name) and n.id in _IMPORT_ERROR_NAMES for n in names)


def _guarded_imports(tree: ast.AST) -> set:
    """Import nodes that never run against the removed layer, so they are not hits:

    * under ``if TYPE_CHECKING:`` — evaluated by type checkers only;
    * ``from F import n`` inside a ``try:`` whose handler catches ``ImportError`` — the documented
      "new path first, fall back to the old one" migration shape. The body is the fallback when the
      old path is in the handler, and the attempt when the old path is in the body; either way the
      plugin survives the removal, so neither counts. (The scanner cannot know which branch wins.)
      Plain ``import F`` is NOT protected by that try: the facade module keeps existing, only the name
      is gone, so ``F.n`` fails with AttributeError at call time, outside the try.
    """
    guarded: set = set()

    def _mark(body, kinds):
        for n in body:
            for sub in ast.walk(n):
                if isinstance(sub, kinds):
                    guarded.add(sub)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
               (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
                _mark(node.body, (ast.Import, ast.ImportFrom))
        elif isinstance(node, ast.Try) and any(_catches_import_error(h) for h in node.handlers):
            _mark(node.body, ast.ImportFrom)
            for h in node.handlers:
                if _catches_import_error(h):
                    _mark(h.body, ast.ImportFrom)
    return guarded


def _module_string(node: ast.AST) -> Optional[str]:
    """``importlib.import_module("F")`` / ``__import__("F")`` -> ``"F"``."""
    if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    f = node.func
    is_import_module = (isinstance(f, ast.Attribute) and f.attr == "import_module") or \
        (isinstance(f, ast.Name) and f.id in {"import_module", "__import__"})
    return node.args[0].value if is_import_module and isinstance(node.args[0].value, str) else None


def scan_source(src: str, rel: str, manifest: Dict[str, Dict[str, str]]) -> List[Hit]:
    """Hits in one file: ``from F import n``, ``from F import *`` (when F has a manifest name that is
    then used bare), ``import F`` + ``F.n``, ``import F as a`` + ``a.n``, ``import_module("F").n`` /
    ``getattr(import_module("F"), "n")`` and string targets ``"F.n"`` (``patch``/``import_module``).
    Imports under ``if TYPE_CHECKING:`` or in an ``ImportError``-guarded ``try`` are not hits."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    hits: List[Hit] = []
    guarded = _guarded_imports(tree)
    aliases: Dict[str, str] = {}                      # local alias -> facade module
    star_facades: List[str] = []                      # ``from F import *``: bare names may be F's
    local_names: set = set()                          # names the file binds itself (shadow star imports)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        if node in guarded:
            continue
        if isinstance(node, ast.ImportFrom) and node.module in manifest and node.level == 0:
            for a in node.names:
                if a.name == "*":
                    star_facades.append(node.module)
                elif a.name in manifest[node.module]:
                    hits.append(Hit(rel, node.lineno, f"{node.module}.{a.name}", manifest[node.module][a.name]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in manifest:
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and (fac := _module_string(node.value)) in manifest:
            aliases[node.targets[0].id] = fac                           # m = import_module("F")

    def _facade_of(base: ast.AST) -> Optional[str]:
        if isinstance(base, ast.Name):
            return aliases.get(base.id)
        fac = _module_string(base)
        return fac if fac in manifest else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (fac := _facade_of(node.value)):
            if node.attr in manifest[fac]:                              # F.n / a.n / import_module("F").n
                hits.append(Hit(rel, node.lineno, f"{fac}.{node.attr}", manifest[fac][node.attr]))
        elif isinstance(node, ast.Attribute):
            # dotted: pkg.sub.name  ->  resolve the full module chain
            parts: List[str] = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                parts.reverse()
                for i in range(1, len(parts)):
                    mod, name = ".".join(parts[:i]), parts[i]
                    if mod in manifest and name in manifest[mod]:
                        hits.append(Hit(rel, node.lineno, f"{mod}.{name}", manifest[mod][name]))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                and (fac := _facade_of(node.args[0])):
            name = node.args[1].value                                   # getattr(m, "n")
            if isinstance(name, str) and name in manifest[fac]:
                hits.append(Hit(rel, node.lineno, f"{fac}.{name}", manifest[fac][name]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "." in node.value:
            mod, _, name = node.value.rpartition(".")
            if mod in manifest and name in manifest[mod]:
                hits.append(Hit(rel, node.lineno, node.value, manifest[mod][name]))
        elif star_facades and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id not in local_names:
            for fac in star_facades:                                    # from F import *; n
                if node.id in manifest[fac]:
                    hits.append(Hit(rel, node.lineno, f"{fac}.{node.id}", manifest[fac][node.id]))
    # dedupe (the two walks can see the same Attribute)
    return sorted(set(hits), key=lambda h: (h.file, h.line, h.old))


def scan_plugin(plugin_dir: Optional[Path], manifest: Optional[Dict[str, Dict[str, str]]] = None) -> List[Hit]:
    manifest = load_manifest() if manifest is None else manifest
    if not manifest or not plugin_dir or not Path(plugin_dir).is_dir():
        return []
    hits: List[Hit] = []
    for p in _iter_py(Path(plugin_dir)):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits += scan_source(src, str(p.relative_to(plugin_dir)), manifest)
    return hits


# ---------------------------------------------------------------------------------------------- report

_report_lock = threading.Lock()
_report_cache: Dict[Tuple[str, ...], Dict[str, List[Hit]]] = {}


def _scan_root(manifest) -> Optional[Path]:
    """Directory to scan for ONE manifest, or None when there is nothing safe to scan.

    Directory plugins carry their own dir. Entry points carry ``module:attr``: resolve the module
    through import metadata to its installed package dir. Never fall back to a relative path — that
    made ``C:\\...`` and ``pkg:attr`` scan the CWD and attribute stray files to the plugin.
    """
    if getattr(manifest, "source", "") == "bundled" or not getattr(manifest, "path", None):
        return None
    raw = str(manifest.path)
    if getattr(manifest, "source", "") == "entrypoint":
        import importlib.util
        try:
            spec = importlib.util.find_spec(raw.partition(":")[0])
        except (ImportError, ValueError):
            spec = None
        origin = getattr(spec, "origin", None)
        if not origin or origin in ("built-in", "frozen"):
            return None
        p = Path(origin)
        return p.parent if p.name == "__init__.py" else None
    p = Path(raw)
    return p if p.is_dir() else None


def compat_report(manifests=None, *, force: bool = False) -> Dict[str, List[Hit]]:
    """``{plugin_name: hits}`` for every ENABLED external (non-bundled) plugin with at least one hit.

    ``manifests`` defaults to the current PluginManager's discovered manifests. Cached per manifest set.
    """
    if manifests is None:
        try:
            from hermes_cli.plugins import get_plugin_manager
            mgr = get_plugin_manager()
            mgr.discover_and_load()
            manifests = [lp.manifest for lp in mgr._plugins.values()]
        except Exception:
            return {}
    external = [m for m in manifests if getattr(m, "source", "") != "bundled" and getattr(m, "path", None)]
    key = tuple(sorted(f"{m.name}@{m.path}" for m in external))
    with _report_lock:
        if not force and key in _report_cache:
            return _report_cache[key]
    manifest = load_manifest()
    out: Dict[str, List[Hit]] = {}
    for m in external:
        hits = scan_plugin(_scan_root(m), manifest)
        if hits:
            out[m.name] = hits
    with _report_lock:
        _report_cache[key] = out
    _write_report_file(out)
    return out


REPORT_FILE = ".plugin-compat-report.json"


def report_file_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / REPORT_FILE


def _write_report_file(report: Dict[str, List[Hit]]) -> None:
    """Persist the latest report for surfaces without a Python runtime handy (the Desktop boot modal).

    Written on every scan so a fixed plugin clears the notice on the next start; removed outright when
    there is nothing to report so a stale file can never resurface a resolved warning.
    """
    try:
        p = report_file_path()
        if not report:
            if p.exists():
                p.unlink()
            return
        payload = {"removal_date": COMPAT_REMOVAL, "in_effect": removal_in_effect(),
                   "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   "plugins": {k: [h.__dict__ for h in v] for k, v in report.items()},
                   "lines": summary_lines(report)}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        pass


def plugin_hits(manifest) -> List[Hit]:
    """Hits for ONE manifest (used by the loader before importing it)."""
    return scan_plugin(_scan_root(manifest))


def allow_deprecated_imports(config: Optional[dict] = None) -> bool:
    """``plugins.allow_deprecated_imports: true`` keeps hitting plugins loading after the date."""
    try:
        if config is None:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        # Literal boolean only: YAML `"false"` / `"no"` must not open the post-removal bypass.
        return ((config or {}).get("plugins") or {}).get(ALLOW_KEY, False) is True
    except Exception:
        return False


def disable_reason(manifest, *, today: Optional[_dt.date] = None) -> Optional[str]:
    """Why the loader must skip this plugin now, or None. Only ever non-None after the removal date."""
    if not removal_in_effect(today) or allow_deprecated_imports():
        return None
    hits = plugin_hits(manifest)
    if not hits:
        return None
    return (f"uses {len(hits)} import path(s) removed on {COMPAT_REMOVAL}; run `hermes plugins compat` "
            f"for the list, update the plugin, or set plugins.{ALLOW_KEY}: true to force-load")


def summary_lines(report: Dict[str, List[Hit]], *, today: Optional[_dt.date] = None) -> List[str]:
    """Plain-text lines for banners/notices; empty when there is nothing to say."""
    if not report:
        return []
    n = len(report)
    names = ", ".join(f"{k} ({len(v)})" for k, v in sorted(report.items()))
    if removal_in_effect(today) and allow_deprecated_imports():
        head = (f"{n} plugin{'s' if n != 1 else ''} force-loaded via plugins.{ALLOW_KEY}: they import paths "
                f"removed on {COMPAT_REMOVAL}: {names}")
        tail = "Update the plugin(s); the old paths no longer exist. Details: hermes plugins compat"
    elif removal_in_effect(today):
        head = (f"{n} plugin{'s' if n != 1 else ''} DISABLED: they import paths removed on {COMPAT_REMOVAL}: {names}")
        tail = f"Update the plugin(s) or set plugins.{ALLOW_KEY}: true to force-load. Details: hermes plugins compat"
    else:
        d = days_until_removal(today)
        head = (f"{n} plugin{'s' if n != 1 else ''} use{'s' if n == 1 else ''} import paths that stop working on "
                f"{COMPAT_REMOVAL} ({d} day{'s' if d != 1 else ''}): {names}")
        tail = "Check for plugin updates or notify the author before then. Details: hermes plugins compat"
    return [head, tail]


# ---------------------------------------------------------------------------------------------- runtime warn

_seen: set = set()
_log = __import__("logging").getLogger(__name__)


def warn_once(facade: str, name: str, target_module: str, target_name: str) -> None:
    """Per-name record that a moved name was resolved through its old path: a ``HermesPluginCompatWarning``
    (so ``-W error`` catches it in tests and plugin authors' CI) plus a WARNING log line (agent.log /
    gateway.log). The interactive CLI hides the warning category from stderr (:func:`quiet_for_interactive`)
    because its banner carries the user-facing message with the plugin NAME, which this call site cannot know."""
    key = (facade, name)
    if key in _seen:
        return
    _seen.add(key)
    new = f"{target_module}.{target_name}" if target_name != name else f"{target_module}.{name}"
    msg = (f"hermes plugin compat: `{facade}.{name}` moved to `{new}`. The old path is kept only for external "
           f"plugins and is removed on {COMPAT_REMOVAL}; update your import.")
    _log.warning(msg)
    warnings.warn(msg, HermesPluginCompatWarning, stacklevel=3)


def quiet_for_interactive() -> None:
    """Called by the interactive CLI before plugin discovery: the banner notice replaces raw stderr warnings.
    Appends (does not override) so an explicit ``-W error::...HermesPluginCompatWarning`` still wins."""
    if not any(a == "error" and c is not None and issubclass(HermesPluginCompatWarning, c)
               for a, _m, c, _mod, _l in warnings.filters):
        warnings.filterwarnings("ignore", category=HermesPluginCompatWarning, append=True)
