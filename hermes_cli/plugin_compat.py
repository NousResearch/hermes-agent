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


def scan_source(src: str, rel: str, manifest: Dict[str, Dict[str, str]]) -> List[Hit]:
    """Hits in one file: ``from F import n``, ``import F`` + ``F.n``, ``import F as a`` + ``a.n``,
    and string targets ``"F.n"`` (``patch``/``import_module``)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    hits: List[Hit] = []
    aliases: Dict[str, str] = {}                      # local alias -> facade module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in manifest and node.level == 0:
            for a in node.names:
                if a.name in manifest[node.module]:
                    hits.append(Hit(rel, node.lineno, f"{node.module}.{a.name}", manifest[node.module][a.name]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in manifest:
                    aliases[a.asname or a.name] = a.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            fac = aliases[node.value.id]
            if node.attr in manifest[fac]:
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
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "." in node.value:
            mod, _, name = node.value.rpartition(".")
            if mod in manifest and name in manifest[mod]:
                hits.append(Hit(rel, node.lineno, node.value, manifest[mod][name]))
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
