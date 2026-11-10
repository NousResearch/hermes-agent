"""The hosted code fence: one process-wide enforcement point for in-process code loading.

A shared process serves many tenants, and four review rounds of per-loader provenance checks
kept finding loaders that skipped them (a dotted entry point's parent package importing from
the deployment home, a checked directory whose files inside it were not, discovery and warm-up
paths, the model-provider entry-point scan, schema and declaration loaders). So enforcement
moved here: an installed Python audit hook (``sys.addaudithook``; it cannot be removed
afterwards) refuses to load ANY code whose file realpath is outside the deployment image's
allowed roots. One choke point answers every loader alike, including loaders this repository
does not know about.

When the fence exists is itself enforced: the hosted start command carries the ``--hosted``
flag, and :func:`arm_from_start_argv` — called as the FIRST statement of the gateway's entry
modules, before their application imports — arms the fence from the flag alone (this module
imports only the stdlib, so arming needs no config read and no Hermes import). The boot's
preflight then confirms the flag against config and refuses a mismatch in either direction:
a hosted boot that did not arm before its application imports refuses, and so does a flagged
start on a deployment that is not hosted. :func:`install_hosted_code_fence` remains as the
verdict-gated install for hosts that boot hosted in-process.

The allowed roots are the interpreter's own stdlib and ``sysconfig`` purelib/platlib (where the
image's packages are installed) and the Hermes package's own tree (resolved from Hermes's code,
never from the deployment home, config or environment; the bundled plugins tree resolved the
same way is inside it). ``sys.path`` entries outside those roots are pruned when the fence arms
(:func:`prune_sys_path_outside_image` is the same pruning behind the frozen-verdict gate), so a
``.pth``-added home directory cannot even be searched afterwards.

Once installed the fence cannot be disarmed: the hook consults nothing mutable — no verdict
re-read, no flag to flip — and an audit hook cannot be removed. A later change to the boot's
recorded verdict (a config edit, a second ``record_boot_config``) leaves an armed fence armed;
``record_boot_config`` itself refuses a config that contradicts the frozen verdict.

Which audit events carry a code file's path was established empirically on this interpreter
(CPython 3.14) and is asserted by the tests:

* ``compile`` (source bytes, filename) — the authoritative signal for every source load:
  importlib, zipimport (the filename is ``<archive>/<member>``), ``runpy``, explicit
  ``spec_from_file_location`` loaders and plain ``compile()`` all raise it with the real path
  BEFORE the code object exists, so a refused module never executes.
* ``exec`` (code object) — ``co_filename`` checked as defence in depth. It is forgeable (a
  bytecode file's marshalled code names any path its builder chose), so it is never the only
  line: the two read-side checks below refuse the loads that would reach ``exec`` with a lie.
* ``open`` (path, mode, flags) — the read side of a sourceless load: ``io.open_code`` raises
  only this generic event on this build, so every open of a ``.pyc``/``.pyo`` whose realpath
  is outside the allowed roots is refused — a forged-bytecode file outside the image cannot
  even be read, whatever its header claims (verified: a forged-``co_filename`` ``.pyc``
  imported sourceless runs without this check). Data files are untouched.
* ``marshal.loads`` (bytes) — bytecode becomes code objects here, and the bytes carry no path,
  so the CALLER is what the fence judges: importlib's own frozen bootstrap readers
  (``importlib._bootstrap``, ``importlib._bootstrap_external``, ``zipimport`` — the only
  callers observed on this interpreter, and they only read bytecode files the ``open`` check
  already judged) are allowed; every other caller refuses. Nothing in this tree calls
  ``marshal.loads`` — untrusted bytes turning into code objects is exactly the deserialization
  the security-guidance plugin warns about.
* ``import`` — for dynamically loaded extension modules (``_imp.create_dynamic``) this event
  carries the ``.so`` file's path (verified empirically on this interpreter; the argument is
  ``None`` for ordinary imports, whose file the ``compile``/``exec`` pair already answers), so
  an extension module outside the image is refused before the dynamic library is opened.
  Loading through ``_imp.create_dynamic`` directly, like installing a custom meta-path finder,
  already requires trusted in-process code.
* ``io.open_code`` (path) — handled for forward compatibility: on a build where the distinct
  event fires it becomes an authoritative read-side check for every code file, not only
  bytecode.

Known boundary (documented, tested, unchanged from the reviewed design): a custom meta-path
finder can hand importlib an invented origin inside the allowed roots, and in-process trusted
code can build code objects with forged ``co_filename`` strings (from in-memory source, or
bytes it marshalled itself — though the ``marshal.loads`` caller check refuses the unmarshal).
Both require code ALREADY running in-process to build the lie; the fence refuses every file the
import machinery actually reads, and is not a Python sandbox against trusted code. ``ctypes``
loading of image system libraries is likewise out of scope: those are image code, loaded by
trusted callers.

Everything here is hosted-only: a local install never arms the fence (the start flag is absent,
the install is gated on the frozen hosted verdict) and never prunes ``sys.path`` — local
behaviour is byte-identical.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig  # imported at module level: the audit hook may run mid-import, where a
# lazy ``import sysconfig`` would re-enter the fence through a partially initialized module
from pathlib import Path
from typing import Any

__all__ = [
    "HostedCodeRefused",
    "allowed_inprocess_roots",
    "arm_from_start_argv",
    "armed_by_start_flag",
    "hosted_code_refusal",
    "hosted_fence_installed",
    "install_hosted_code_fence",
    "prune_sys_path_outside_image",
]


class HostedCodeRefused(ImportError):
    """A code load the hosted fence refused: the file it would execute resolves outside the
    deployment image, and the message names the path and the rule."""


# The Hermes package's own tree: every directory this checkout (or installed image) ships code
# from — the bundled plugins tree resolved from Hermes's own code lives inside it.
_TREE_ROOT = Path(__file__).resolve().parent.parent

# Which sysconfig path entries are image roots: the stdlib (and its platform extensions), and
# the interpreter's own package directories, where the image's installed packages live.
_SYSCONFIG_ROOT_KEYS = ("stdlib", "platstdlib", "purelib", "platlib")

_HOSTED_RULE = ("a hosted process loads in-process code only from what its deployment image "
                "ships (the plugins bundled with Hermes, or a package installed into the "
                "image's environment)")

# The audit events that carry a code file's path (see the module docstring for what each one
# covers on this interpreter), with how each one's args expose that path.
_CODE_EVENT_PATH: dict[str, Any] = {
    "io.open_code": lambda args: args[0] if args else None,  # the read-side check, where raised
    "compile": lambda args: args[1] if len(args) > 1 else None,  # the authoritative check here
    "exec": lambda args: getattr(args[0] if args else None, "co_filename", None),  # forgeable
    "import": lambda args: args[1] if len(args) > 1 else None,  # the .so path for ext loads
}

# The fence's own process-wide state. The scope-keyed/tenant_neutral declarations HX015 asks
# for live in hermes_scope_cache, a Hermes import this module must never make: the arm at
# process entry imports it before any Hermes code (the whole point — arming needs no Hermes
# state). The state is the deployment's, identical for every tenant by construction, and
# immutable in one direction: each flag flips once per process, at the first hosted arm.
_installed = False  # health: allow HX015 -- the fence is the deployment's own enforcement, tenant-identical; stdlib-only arming cannot import the scope cache
_armed_by_start_flag = False  # health: allow HX015 -- the start command's own flag, recorded once at process entry; tenant-identical deployment state
_roots_cache: tuple[tuple, tuple] = ((), ())  # health: allow HX015 -- the image's allowed roots, identical for every tenant of the one deployment


def _bundled_plugins_dir() -> Path | None:
    """The bundled plugins tree, from the already-imported facade (never imported here — this
    runs inside import machinery, where starting an import would re-enter the fence). When the
    facade is not imported yet the tree is the one inside the Hermes package, which the tree
    root already covers."""
    facade = sys.modules.get("hermes_cli.plugins")
    if facade is None:
        return None
    try:
        return Path(facade.get_bundled_plugins_dir())
    except Exception:  # health: allow BLE001 -- a facade mid-import or a patched seam that raises skips the root, never crashes a load
        return None


def allowed_inprocess_roots() -> tuple[Path, ...]:
    """The realpath roots whose code a deployment image ships — the only roots a hosted
    process may load in-process code from: this interpreter's stdlib and package directories,
    and the Hermes package's own tree (with its bundled plugins tree). Re-read from ``sysconfig``
    and the plugins facade each call and cached by raw value, so the fence follows the same
    resolution the loaders consult, never a stale snapshot."""
    global _roots_cache

    paths = sysconfig.get_paths()
    bundled = _bundled_plugins_dir()
    key = (tuple(str(paths.get(name) or "") for name in _SYSCONFIG_ROOT_KEYS),
           str(_TREE_ROOT), str(bundled) if bundled is not None else "")
    cached_key, cached_roots = _roots_cache
    if cached_key == key:
        return cached_roots
    roots: list[Path] = []
    for raw in (paths.get(name) for name in _SYSCONFIG_ROOT_KEYS):
        if not raw:
            continue
        try:
            real = Path(raw).resolve()
        except OSError:
            continue
        if real not in roots:
            roots.append(real)
    for real_root in (_TREE_ROOT, bundled):
        if real_root is None:
            continue
        try:
            real = Path(real_root).resolve()
        except OSError:
            continue
        if real not in roots:
            roots.append(real)
    _roots_cache = (key, tuple(roots))
    return tuple(roots)


def _realpath_inside(path, roots: tuple[Path, ...]) -> bool:
    """Whether *path*'s realpath stays inside one of *roots* (equal to it, or below it). A
    nonexistent tail (a zip member ``<archive>/<name>``, a ``.pyc`` header's source path) keeps
    the existing prefix's resolution, so an archive outside the roots is judged by the
    archive's own realpath."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _refusal_for(path) -> str:
    """The fence's answer for one candidate code path: ``""`` when it is image code, else the
    reason. Pseudo-filenames (``<string>``, ``<frozen importlib._bootstrap>``, …) name no file
    and are not a load from outside the image."""
    if path is None:
        return ""
    try:
        text = os.fspath(path)  # str and Path alike; anything else names no file
    except TypeError:
        return ""
    if not text or text.startswith("<"):
        return ""
    if _realpath_inside(text, allowed_inprocess_roots()):
        return ""
    return f"{_HOSTED_RULE}; {text} resolves outside that image, so its load is refused"


def hosted_code_refusal(code_path=None, *, module: str | None = None) -> str:
    """THE one shared predicate every hosted discovery pre-check consults: ``""`` when the code
    may load in-process, else the reason it may not. A file or directory path is judged by its
    realpath against the allowed roots (:func:`allowed_inprocess_roots`); an entry point's
    *module* is judged by the file ``importlib.util.find_spec`` resolves it to — never by its
    distribution's metadata, which a ``.pth`` file or an editable record can aim anywhere.

    Enforcement does NOT depend on callers of this predicate: the installed fence
    (:func:`install_hosted_code_fence`) refuses the load itself at the compile/exec boundary,
    whatever a loader forgot. A dotted module's parent import is refused there too, before the
    parent's module-level code can run. Dormant unless the boot froze a hosted verdict (a local
    install answers ``""`` for everything)."""
    from hermes_cli.hosted_mode import hosted_verdict_frozen

    if hosted_verdict_frozen() is not True:
        return ""
    if code_path is None:
        origin = None
        if module:
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, ValueError):  # unresolvable or ambiguous — or the parent import
                spec = None  # was just refused by the fence: no file to judge, so no trust
            origin = getattr(spec, "origin", None) if spec is not None else None
        # A pseudo-origin ("<unknown>", "<frozen …>") names no file to judge: fail closed.
        if origin and not origin.startswith("<") and _refusal_for(origin) == "":
            return ""
        return (f"{_HOSTED_RULE}; the entry point's module "
                + (f"{module!r} " if module else "")
                + "resolves outside that image, so its distribution's metadata is not "
                  "provenance and the entry point is refused")
    return _refusal_for(code_path)


def _refuse_outside_image(path) -> None:
    refusal = _refusal_for(path)
    if refusal:
        raise HostedCodeRefused(refusal)


# Bytecode files the import machinery loads sourceless: no source compile ever fires for them,
# so the read itself is the enforcement point. ``io.open_code`` raises only the generic ``open``
# event on this build, and a bytecode file's marshalled code can forge any ``co_filename`` — the
# file's own realpath is the one signal its builder cannot lie about.
_BYTECODE_SUFFIXES = (".pyc", ".pyo")

# The only callers allowed to unmarshal bytecode: importlib's own frozen bootstrap readers
# (``importlib._bootstrap``, ``importlib._bootstrap_external``, ``zipimport``) — the only
# callers observed on this interpreter (asserted by the tests), and they only read bytecode
# files the ``open`` check above already judged by realpath. Every other ``marshal.loads`` in a
# hosted process is bytes turning into code objects with no file the fence could judge — the
# same untrusted deserialization the security-guidance plugin warns about — and refuses.
_FROZEN_IMPORT_BOOTSTRAP = ("<frozen importlib._bootstrap", "<frozen zipimport")


def _refuse_bytecode_read_outside_image(path) -> None:
    """The read side of a sourceless load: a ``.pyc``/``.pyo`` whose realpath is outside the
    image never opens, so no loader can unmarshal it — whatever path its header claims. Data
    files are untouched: a hosted process reads its tenants' data constantly."""
    try:
        text = os.fspath(path)  # str and Path alike; anything else names no file
    except TypeError:
        return
    if not text or not text.endswith(_BYTECODE_SUFFIXES):
        return
    _refuse_outside_image(text)


def _hosted_audit_hook(event, args) -> None:
    """The fence itself: every code-loading audit event answers the one predicate before the
    code exists or runs. Non-code events pass untouched. The hook consults NOTHING mutable —
    no verdict re-read, no flag — so once installed it enforces for the rest of the process
    and nothing that later edits the boot's recorded state can disarm it."""
    if event == "open":
        _refuse_bytecode_read_outside_image(args[0] if args else None)
        return
    if event == "marshal.loads":
        # Frame 0 is this hook; the C-level marshal.loads adds no Python frame, so the
        # unmarshal's caller is exactly one frame up — inline, not in a helper, so the
        # depth cannot drift.
        try:
            caller = sys._getframe(1).f_code.co_filename
        except (ValueError, AttributeError):
            caller = ""  # no Python caller at all (pure C): not importlib's bootstrap either
        if not caller.startswith(_FROZEN_IMPORT_BOOTSTRAP):
            raise HostedCodeRefused(
                f"{_HOSTED_RULE}; marshal.loads would turn bytes no file answers for into "
                "code objects outside importlib's own bytecode readers, so the load is refused")
        return
    path_of = _CODE_EVENT_PATH.get(event)
    if path_of is None:
        return
    _refuse_outside_image(path_of(args))


def hosted_fence_installed() -> bool:
    """Whether this process already installed the fence (an audit hook cannot be removed, so
    this flips once per process, at the first hosted arm or install)."""
    return _installed


def armed_by_start_flag() -> bool:
    """Whether the start command's ``--hosted`` flag armed the fence at process entry — the
    confirmation the hosted boot preflight requires: a hosted boot whose start command did
    not arm before its application imports refuses."""
    return _armed_by_start_flag


def _install_hook() -> None:
    """The one install: warm the roots cache BEFORE the hook exists (the first
    ``sysconfig.get_paths()`` importfully loads the interpreter's own build data, and a hook
    firing in the middle of that would re-enter it through a partially initialized module),
    then add the audit hook. Called exactly once per process."""
    global _installed
    allowed_inprocess_roots()
    sys.addaudithook(_hosted_audit_hook)
    _installed = True


def install_hosted_code_fence() -> bool:
    """Install the process-wide code fence from a frozen hosted verdict: the in-process boot
    path. A no-op — with nothing installed — unless the boot verdict is already frozen hosted:
    a local install never installs any hook. Idempotent. The installed hook is immutable by
    construction: it re-reads nothing, and a later change to the recorded verdict leaves the
    fence enforcing."""
    from hermes_cli.hosted_mode import hosted_verdict_frozen

    if hosted_verdict_frozen() is not True or _installed:
        return False
    _install_hook()
    return True


def arm_from_start_argv() -> bool:
    """Arm the fence from the hosted start command's ``--hosted`` flag: the EARLIEST
    enforcement a hosted boot has. Called as the first statement of the gateway's entry
    modules, before their application imports — the flag is the deployment's own assertion
    that this process serves tenants, decided from argv alone, with no config read (this
    module imports only the stdlib; nothing in the arm consults Hermes). The boot's preflight
    later confirms the flag against config and refuses a mismatch in either direction.

    Arming also prunes every ``sys.path`` entry outside the image, so nothing outside it
    stays searchable once the application imports begin. Idempotent; a start command without
    the flag arms nothing and prunes nothing (a local install is untouched)."""
    if not any(arg == "--hosted" for arg in sys.argv[1:]):
        return False
    global _armed_by_start_flag
    if _installed:  # an earlier arm (or an in-process install) already added the hook
        _armed_by_start_flag = True  # the start command carried the flag either way
        return False
    _install_hook()
    _armed_by_start_flag = True
    _prune_sys_path_outside_image()
    return True


def _prune_sys_path_outside_image() -> list[str]:
    """The pruning itself: drop every ``sys.path`` entry whose realpath resolves outside the
    allowed roots and return what was removed. Called by the start-flag arm (an armed fence
    implies hosted) and behind the frozen-verdict gate for later callers."""
    roots = allowed_inprocess_roots()
    kept: list[str] = []
    pruned: list[str] = []
    for entry in sys.path:
        candidate = Path.cwd() if entry == "" else entry
        if not _realpath_inside(candidate, roots):
            pruned.append(entry)
            continue
        kept.append(entry)
    if pruned:
        sys.path[:] = kept
    return pruned


def prune_sys_path_outside_image() -> list[str]:
    """Remove every ``sys.path`` entry whose realpath resolves outside the allowed roots and
    return what was removed (the boot logs it): a ``.pth``-added home directory, an inherited
    ``PYTHONPATH`` residue — nothing outside the image stays searchable, so the import system
    cannot even find a tenant's module afterwards. Dormant unless the frozen hosted verdict is
    hosted; a local install's ``sys.path`` is never touched."""
    from hermes_cli.hosted_mode import hosted_verdict_frozen

    if hosted_verdict_frozen() is not True:
        return []
    return _prune_sys_path_outside_image()
