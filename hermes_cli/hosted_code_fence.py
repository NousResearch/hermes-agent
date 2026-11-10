"""The hosted code-provenance guarantee, in three parts.

(A) PRIMARY, the loader census: in hosted mode no Hermes code path loads Python from a
path without first asking the ONE predicate, :func:`hosted_code_refusal`, and refusing
when it says no — a plugin directory, a model or memory provider, a hook handler, a
dashboard api file, a config schema, an update checkout: every loader consults the same
answer before it loads. The census is structural and enforced by a test
(:mod:`tests.hermes_cli.test_hosted_loader_census`): an AST scan of the repository
finds every dynamic code-loading primitive (``runpy``, ``spec_from_file_location``,
the ``importlib.machinery`` file loaders, ``zipimport``, non-literal
``importlib.import_module``, ``exec``/``compile`` of file contents, ``sys.path``
insertions, and site-directory processing — ``site.addsitedir``/``site.addpackage``,
whose ``.pth`` files execute import-time code) and each hit must either sit in a
function that calls the predicate first
or carry a named exemption explaining why its input is image-only — a NEW unguarded
loader fails the census, so the guarantee does not depend on anyone noticing a new
import site. The predicate itself (:func:`hosted_code_refusal`) is the single
authority: a path is judged by its realpath against the allowed roots
(:func:`allowed_inprocess_roots`), an entry point by the file its module resolves to.

(B) INTERPRETER, isolated startup: the hosted start refuses to ARM — and the process
exits non-zero — unless the interpreter itself started isolated from user and data
site paths (:func:`startup_isolation_refusal`, consulted by
:func:`arm_from_start_argv` before anything installs): user-site loading disabled
(``sys.flags.no_user_site``, or ``site.ENABLE_USER_SITE`` false), no
``sitecustomize``/``usercustomize`` whose file lies outside the image roots, and no
``sys.meta_path``/``sys.path_hooks`` entry whose defining module's file lies outside
the roots at arm time — refused rather than trusted, because a site hook that ran
before the fence cannot be judged after the fact. This is a deployment prerequisite
the fence can only verify, never retrofit: the image entrypoint must run python with
``-I`` or ``-s`` plus ``PYTHONNOUSERSITE`` (an isolated venv with user site disabled
satisfies it), from site directories the image owns.

(C) DEFENCE IN DEPTH, the audit hook: the installed ``sys.addaudithook`` fence below
is the third layer, not the guarantee. It refuses — at ``compile``, at the bytecode
``open``, at ``import`` (extension paths and the import system's own inputs) — code
whose file resolves outside the allowed roots, whatever a loader forgot, and it
cannot be removed once installed. Its KNOWN LIMITS, deliberate and documented: zip
bytecode raises no ``compile`` or ``open`` event for the member it executes; bytecode
under a non-``.pyc``/``.pyo`` name slips the read-side suffix check; and in-process
calls to the standard-library loaders (``importlib.import_module`` of a module in a
zip added to ``sys.path``, ``zipimport.zipimporter(...).load_module``,
``runpy.run_path`` of bytecode or a zip, a direct ``SourcelessFileLoader``) raise no
event naming the real file at all. Those are exactly why (A) is primary: they are
raw standard-library calls, and the census answers the question that matters about
each of them — whether any HERMES loader can be driven to one with a
data-directory path. If it can, the loader is guarded by the predicate and the case
is a refusal test; if it cannot, in-process trusted code making the call is the
documented boundary below (code already running in the process could rewrite this
fence's own policy, so no in-process hook can defend against it).

When the fence exists is itself enforced: the hosted start command carries the ``--hosted``
flag, and :func:`arm_from_start_argv` — called as the FIRST statement of the gateway's entry
modules, before their application imports — arms the fence from the flag alone (this module
imports only the stdlib, so arming needs no config read and no Hermes import), after the
isolated-startup check (B). The boot's preflight then confirms the flag against config and
refuses a mismatch in either direction: a hosted boot that did not arm before its
application imports refuses, and so does a flagged start on a deployment that is not hosted.
:func:`install_hosted_code_fence` remains as the verdict-gated install for hosts that boot
hosted in-process.

The allowed roots are the interpreter's own stdlib and ``sysconfig`` purelib/platlib (where
the image's packages are installed) and the Hermes package's own tree (resolved from Hermes's code,
never from the deployment home, config or environment; the bundled plugins tree resolved the
same way is inside it). ``sys.path`` entries outside those roots are pruned when the fence arms
(:func:`prune_sys_path_outside_image` is the same pruning behind the frozen-verdict gate), so a
``.pth``-added home directory cannot even be searched afterwards — and the import event
re-answers every entry the pruning judgment might miss later in the process.

Once installed the fence is not disarmed by an edit to the boot's recorded state: the hook
consults no verdict re-read and no flag to flip, and an audit hook cannot be removed. A later
change to the boot's recorded verdict (a config edit, a second ``record_boot_config``) leaves
an armed fence armed; ``record_boot_config`` itself refuses a config that contradicts the
frozen verdict.

Which audit events carry a code file's path was established empirically on this interpreter
(CPython 3.14) and is asserted by the tests:

* ``compile`` (source bytes, filename) — the authoritative signal for every source load:
  importlib, zipimport (the filename is ``<archive>/<member>``), ``runpy``, explicit
  ``spec_from_file_location`` loaders and plain ``compile()`` all raise it with the real path
  BEFORE the code object exists, so a refused module never executes.
* ``open`` (path, mode, flags) — the read side of a sourceless load: ``io.open_code`` raises
  only this generic event on this build, so every open of a ``.pyc``/``.pyo`` whose realpath
  is outside the allowed roots is refused — a forged-bytecode file outside the image cannot
  even be read, whatever its header claims (verified: a forged-``co_filename`` ``.pyc``
  imported sourceless runs without this check). Data files are untouched. A bytecode file
  renamed to a non-``.pyc`` suffix is NOT covered — that is one of the documented limits in
  (C), answered by (A).
* ``import`` (module, path, sys.path, sys.meta_path, sys.path_hooks) — two checks. For
  dynamically loaded extension modules (``_imp.create_dynamic``) the event carries the ``.so``
  file's path (``None`` for ordinary imports), so an extension module outside the image is
  refused before the dynamic library is opened. And the event carries the import system's
  INPUTS — the live ``sys.path``, ``sys.meta_path`` and ``sys.path_hooks`` — which is where
  every loader must register to be consulted: an import is refused while any ``sys.path``
  entry's realpath (a data directory or a zip archive file alike) resolves outside the
  allowed roots, naming the entry, and likewise when ``sys.meta_path`` or ``sys.path_hooks``
  hold a finder or hook whose defining module's file is outside the roots — but only what
  the fence did not install over: the search entries and finders already present when the
  fence installed are the operator's interpreter site initialization. The isolated-startup
  check (B) is what refuses to arm at all when such an outside-defined finder is there at
  arm time.
* ``io.open_code`` (path) — handled for forward compatibility: on a build where the distinct
  event fires it becomes an authoritative read-side check for every code file, not only
  bytecode.

Known boundary (documented, tested, unchanged from the reviewed design): a custom meta-path
finder can hand importlib an invented origin inside the allowed roots, and in-process trusted
code can build code objects with forged ``co_filename`` strings from in-memory source or bytes
it marshalled itself. Both require code ALREADY running in-process to build the lie — which
is exactly the trusted image code the contract above scopes out — and the fence is not a
Python sandbox against it. ``ctypes`` loading, ``marshal`` deserialization and ``exec`` of
in-memory strings by such code are likewise out of scope: those are calls the image itself
makes, not a loader or an import the fence can see. (The stdlib's own consumers —
``modulefinder`` reading image bytecode, ordinary ``marshal`` data roundtrips, ``pstats`` —
must keep working in hosted mode, which is why ``marshal`` is not filtered by caller at all.)

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
    "HOSTED_BOOT_REFUSED_EXIT",
    "HostedCodeRefused",
    "allowed_inprocess_roots",
    "arm_from_start_argv",
    "armed_by_start_flag",
    "hosted_code_refusal",
    "hosted_fence_installed",
    "hosted_start_flag_requested",
    "install_hosted_code_fence",
    "prune_sys_path_outside_image",
    "startup_isolation_refusal",
]


# The exit a refused hosted start takes: the same fatal-config exit the boot's preflight
# refusals use, named here (not imported) because the arm path must import no Hermes code.
HOSTED_BOOT_REFUSED_EXIT = 78


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

# The import system's state at the moment the fence took over, captured once at install:
# every search entry, finder and path hook already present is the operator's interpreter
# site initialization (a ``.pth``-installed editable-install finder, a venv stack) — the
# trusted bootstrap the fence cannot and does not judge, exactly like the code that ran
# before it installed. What the snapshot buys is the line it draws: anything ADDED after
# the fence exists is judged at every import; a pre-existing entry is instead REMOVED by
# the arm's pruning (captured after the prune, so re-adding a pruned entry is refused anew).
_bootstrap_search_entries: tuple = ()  # health: allow HX015 -- the operator's bootstrap search path, deployment-identical, captured once per process
_bootstrap_finders: tuple = ()  # health: allow HX015 -- the operator's bootstrap finders, deployment-identical, captured once per process


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


def hosted_start_flag_requested() -> bool:
    """Whether the start command carries the ``--hosted`` flag: read from argv alone, with no
    arming, no install and no state — the earliest answer to "does this process serve tenants",
    so loaders that run before the arm (the bootstrap's dependency activation, which precedes
    the entry module's arm statement) can already refuse what a hosted process must not load.
    The boot's preflight later confirms the flag against config, refusing a flagged start on a
    deployment that is not hosted — a process that carries the flag never serves as a local
    install, so honouring it here only ever refuses sooner."""
    return any(arg == "--hosted" for arg in sys.argv[1:])


def hosted_code_refusal(code_path=None, *, module: str | None = None) -> str:
    """THE one shared predicate every hosted discovery pre-check consults: ``""`` when the code
    may load in-process, else the reason it may not. A file or directory path is judged by its
    realpath against the allowed roots (:func:`allowed_inprocess_roots`); an entry point's
    *module* is judged by the file ``importlib.util.find_spec`` resolves it to — never by its
    distribution's metadata, which a ``.pth`` file or an editable record can aim anywhere.

    Enforcement does NOT depend on callers of this predicate: the installed fence
    (:func:`install_hosted_code_fence`) refuses the load itself at the compile, read and
    import boundaries, whatever a loader forgot. A dotted module's parent import is refused
    there too, before the parent's module-level code can run. Dormant unless this process
    serves tenants — the start command's hosted flag (:func:`hosted_start_flag_requested`,
    honoured even before the boot records its verdict, so the bootstrap's own loaders are
    covered) or the boot's frozen hosted verdict. A local install answers ``""`` for
    everything."""
    if not hosted_start_flag_requested():
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


def _finder_defining_file(finder) -> str:
    """The file of the module that defines *finder* (its class, or its type), or ``""``
    when it names no file (builtins, frozen finders — nothing to judge)."""
    module_name = (getattr(finder, "__module__", None)
                   or getattr(type(finder), "__module__", None))
    module = sys.modules.get(module_name) if module_name else None
    file = getattr(module, "__file__", None)
    try:
        return os.fspath(file) if file else ""
    except TypeError:
        return ""


def startup_isolation_refusal() -> str:
    """``""`` when the interpreter started isolated from user and data site paths, else
    the reason a hosted start must refuse to arm. Three checks, all deployment
    prerequisites the fence can verify but never retrofit:

    * user-site loading must be disabled — ``sys.flags.no_user_site`` (a ``-I``/``-s``
      start) or ``site.ENABLE_USER_SITE`` false (an isolated venv): with user site on,
      a ``PYTHONUSERBASE`` in tenant-writable storage executes its ``.pth`` files and
      ``usercustomize`` before any code of ours runs.
    * no ``sitecustomize``/``usercustomize`` module whose file lies outside the image
      roots may be loaded in this process — site initialization code that already ran
      cannot be trusted after the fact, so the start refuses instead.
    * no ``sys.meta_path``/``sys.path_hooks`` entry whose defining module's file lies
      outside the roots may be present at arm time — refused rather than trusted, for
      the same reason: pruning removes search entries, but a finder or hook such a
      module installed keeps consulting whatever it wants.

    This module imports only the stdlib, so the check runs at the arm — before any
    application import — with ``site`` imported lazily (it is stdlib, but the fewer
    imports before the arm the better)."""
    import site

    if not sys.flags.no_user_site and site.ENABLE_USER_SITE:
        return ("this interpreter started with user-site loading enabled, so a "
                "PYTHONUSERBASE .pth or usercustomize outside the deployment image would "
                "run before the fence arms")
    roots = allowed_inprocess_roots()
    for name in ("sitecustomize", "usercustomize"):
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None) if module is not None else None
        if file and not _realpath_inside(file, roots):
            return (f"the {name} module this interpreter loaded is defined by {file}, "
                    "which resolves outside the deployment image")
    for finder in (*sys.meta_path, *sys.path_hooks):
        file = _finder_defining_file(finder)
        if file and not _realpath_inside(file, roots):
            return (f"the import system holds a finder or path hook defined by {file}, "
                    "which resolves outside the deployment image")
    return ""


def _refuse_outside_image(path) -> None:
    refusal = _refusal_for(path)
    if refusal:
        raise HostedCodeRefused(refusal)


# Bytecode files the import machinery loads sourceless: no source compile ever fires for them,
# so the read itself is the enforcement point. ``io.open_code`` raises only the generic ``open``
# event on this build, and a bytecode file's marshalled code can forge any ``co_filename`` — the
# file's own realpath is the one signal its builder cannot lie about.
_BYTECODE_SUFFIXES = (".pyc", ".pyo")

# Verdicts for the import system's INPUTS, memoized by raw value: every ``import`` audit event
# re-examines the live sys.path, sys.meta_path and sys.path_hooks, so each entry's realpath is
# resolved once and reused while its raw value stays on the path. (The empty-string entry means
# the current directory and is never cached, exactly because the directory can change.)
_search_entry_inside: dict[str, bool] = {}  # health: allow HX015 -- verdicts for the deployment's own search-path entries, identical for every tenant
_finder_file_inside: dict[str, bool] = {}  # health: allow HX015 -- verdicts for the deployment's own finders, identical for every tenant


def _entry_inside_image(entry) -> bool:
    """Whether one ``sys.path`` entry resolves inside the image — the judgment the boot-time
    pruning makes at arm time, answered again at every import for entries added afterwards
    (a data directory, a zip archive file: a loader that would read either must put it here).
    An entry that does not exist (the interpreter's own optimistic ``pythonNN.zip`` default
    is typically absent) can never supply code, so it is skipped — checked live each event,
    never cached, since the filesystem can change under a long-running process."""
    try:
        text = os.fsdecode(entry) if isinstance(entry, bytes) else os.fspath(entry)
    except TypeError:
        return False  # nothing the import system could search usefully; fail closed
    if text == "":
        return _realpath_inside(os.getcwd(), allowed_inprocess_roots())  # never cached
    if not os.path.lexists(text):
        return True  # absent entries load nothing; not cached, re-checked at every event
    cached = _search_entry_inside.get(text)
    if cached is not None:
        return cached
    inside = _realpath_inside(text, allowed_inprocess_roots())
    _search_entry_inside[text] = inside
    return inside


def _refuse_import_inputs_outside_image(args) -> None:
    """The import system's inputs at every import: the ``import`` audit event carries the live
    ``sys.path``, ``sys.meta_path`` and ``sys.path_hooks``, and a loader that wants the import
    system to read tenant-supplied code must surface in one of them. An out-of-image search
    entry (naming it, so the boot log says what to remove) or a finder or hook whose defining
    module's file is outside the roots refuses the import — before any loader, zipimport
    included, reads a single member. Only what the fence did NOT install over is judged:
    entries and finders present at install are the operator's trusted site initialization
    (see the snapshot above). Extension-load events carry ``None`` inputs and are answered
    by the ``.so`` path check alone."""
    if len(args) < 5 or args[2] is None or args[3] is None or args[4] is None:
        return
    for entry in args[2]:
        if any(entry == trusted for trusted in _bootstrap_search_entries):
            continue  # the operator's bootstrap search path; the boot's pruning removes it
        if not _entry_inside_image(entry):
            raise HostedCodeRefused(
                f"{_HOSTED_RULE}; the import search path holds {entry!r}, which resolves "
                "outside that image, so the import is refused")
    for finder in (*args[3], *args[4]):
        if any(finder is trusted for trusted in _bootstrap_finders):
            continue  # the operator's bootstrap finders (site initialization), trusted as such
        module_name = (getattr(finder, "__module__", None)
                       or getattr(type(finder), "__module__", None))
        module = sys.modules.get(module_name) if module_name else None
        file = getattr(module, "__file__", None)  # builtins/frozen finders name no file to judge
        if not file:
            continue
        try:
            text = os.fspath(file)
        except TypeError:
            continue
        cached = _finder_file_inside.get(text)
        if cached is None:
            cached = _realpath_inside(text, allowed_inprocess_roots())
            _finder_file_inside[text] = cached
        if not cached:
            raise HostedCodeRefused(
                f"{_HOSTED_RULE}; the import system's finders include one defined by {text}, "
                "which resolves outside that image, so the import is refused")


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
    if event == "import":
        if len(args) > 1 and args[1] is not None:
            _refuse_outside_image(args[1])  # an extension module's own .so file
        _refuse_import_inputs_outside_image(args)
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
    then capture the import system's bootstrap state, then add the audit hook. Called exactly
    once per process. The snapshot is the trust boundary: what is already present is the
    operator's site initialization and keeps working; everything added afterwards is judged
    by every import event."""
    global _installed, _bootstrap_search_entries, _bootstrap_finders
    allowed_inprocess_roots()
    _bootstrap_search_entries = tuple(sys.path)
    _bootstrap_finders = (*sys.meta_path, *sys.path_hooks)
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

    The arm first refuses — exiting the process non-zero (:data:`HOSTED_BOOT_REFUSED_EXIT`)
    — unless the interpreter itself started isolated from user and data site paths
    (:func:`startup_isolation_refusal`): a user-site ``.pth`` or ``usercustomize`` in
    tenant-writable storage, or a finder or path hook defined outside the image, cannot be
    trusted once it has already run, so a hosted start on such an interpreter refuses to
    serve at all. The deployment's image entrypoint must start python with ``-I`` or
    ``-s`` plus ``PYTHONNOUSERSITE``.

    Arming also prunes every ``sys.path`` entry outside the image — so nothing outside it
    stays searchable once the application imports begin — and records what it removed in
    the boot log. The prune runs BEFORE the fence installs, so the bootstrap snapshot the
    fence takes holds only what survived it: an entry the prune removed is judged as new if
    anything adds it back. Idempotent; a start command without the flag arms nothing and
    prunes nothing (a local install is untouched)."""
    if not hosted_start_flag_requested():
        return False
    global _armed_by_start_flag
    if _installed:  # an earlier arm (or an in-process install) already added the hook
        _armed_by_start_flag = True  # the start command carried the flag either way
        return False
    refusal = startup_isolation_refusal()
    if refusal:
        print(
            f"Hosted start refused: {refusal}. Start python with -I or -s plus "
            "PYTHONNOUSERSITE from the deployment image's own site directories.",
            file=sys.stderr, flush=True)
        raise SystemExit(HOSTED_BOOT_REFUSED_EXIT)
    pruned = _prune_sys_path_outside_image()
    _install_hook()
    _armed_by_start_flag = True
    if pruned:
        import logging  # stdlib: the arm imports no Hermes code

        logging.getLogger(__name__).info(
            "Hosted start removed sys.path entries outside the deployment image: %s",
            ", ".join(pruned))
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
