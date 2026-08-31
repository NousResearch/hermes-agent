"""Install and remove the Linux desktop entry (``hermes.desktop``).

``hermes desktop`` builds and launches the Electron app. On Linux, a
freshly-built app has no launcher presence: no menu item, no icon. This
module writes the XDG desktop entry that gives it one.
``hermes uninstall --gui`` removes the entry again.

Two values must be absolute for the entry to work:

  - ``Exec`` — the launcher runs without shell ``PATH`` customizations, so
    a bare ``hermes desktop`` fails when hermes lives in ``~/.local/bin``
    or a venv. Resolve the real binary and write its full path.
  - ``Icon`` — an unqualified icon name needs an indexed icon theme. The
    spec allows an absolute path instead, so point at the app icon in the
    checkout. Do not copy the icon: ``Exec`` already depends on that tree.

Cache refresh is best-effort and tool-gated: ``update-desktop-database``
for the freedesktop menu cache, and ``kbuildsycoca6``/``kbuildsycoca5``
for Plasma. Run each tool only when it exists. A missing tool is not an
error.

Import-light and side-effect-free at import time: the uninstaller uses
this without loading the full CLI.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

DESKTOP_ENTRY_NAME = "hermes.desktop"


def is_supported() -> bool:
    """XDG desktop entries exist only on Linux and BSD."""
    return sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd"))


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share"


def desktop_entry_path() -> Path:
    """Where the ``hermes.desktop`` entry lives."""
    return _xdg_data_home() / "applications" / DESKTOP_ENTRY_NAME


def icon_path(project_root: Path) -> Path:
    """The app icon shipped in the desktop workspace."""
    return project_root / "apps" / "desktop" / "assets" / "icon.png"


def _running_interpreter() -> str:
    """The venv-semantic interpreter path for the persisted ``Exec=`` line.
    ``sys.executable`` inside a venv is commonly a SYMLINK into a shared
    base-interpreter tree (uv, pyenv, conda). ``Path.resolve()`` follows it
    out of the venv, and CPython discovers ``pyvenv.cfg`` from the
    *lexical* argv[0] — so a dereferenced path boots without the venv's
    site-packages and dies on the first third-party import (#90292, one
    level up; identified in #80547's review and confirmed on real Zorin/uv
    hardware in this PR's review).

    Keep the lexical path only when it actually is venv-semantic (a
    ``pyvenv.cfg`` sits at or above it in the tree); otherwise the
    dereferenced absolute path is the more durable form (survives the
    symlink being re-pointed or its parent moving).

    Idea credit: the lexical-preservation rule was independently proposed
    in #92516/#94115/#94544 and by nosliwhtes' review of this PR; the
    pyvenv.cfg-detection refinement here keeps both properties.
    """
    lexical = os.path.abspath(sys.executable)
    path = Path(lexical)
    for base in (path.parent, *path.parent.parents):
        if (base / "pyvenv.cfg").is_file():
            return lexical
    return str(path.resolve())


_probe_cache: "dict[str, bool]" = {}


def _can_import_hermes_cli(interpreter: Path) -> bool:
    """Whether *interpreter* can import ``hermes_cli.main`` unaided.

    Runs the import in a subprocess under ``-I`` (isolated mode: no
    user site, no PYTHONPATH inheritance, no cwd on ``sys.path``) from
    a neutral cwd, so the answer matches what a cold desktop
    environment would get — a checkout cwd or an inherited
    ``PYTHONPATH`` cannot produce a false positive. Bounded by a
    timeout so a hung interpreter cannot stall entry generation.

    Result is cached per interpreter path for the process lifetime, so
    a desktop launch pays the subprocess cost at most once.

    Probe design per @nosliwhtes' isolated-mode capability check
    (#92122 lineage, commit 4150501f641).
    """
    key = str(interpreter)
    cached = _probe_cache.get(key)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [key, "-I", "-c", "import hermes_cli.main"],
            cwd=os.path.abspath(os.sep),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
        ok = result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        # Unprobeable (missing binary, spawn failure, timeout): do not
        # punish the entry on infra hiccups — assume capable and let the
        # existing fallback chain handle a genuinely broken interpreter.
        # This error-derived answer is deliberately NOT cached: one
        # transient hiccup must not freeze the "capable" assumption for
        # the whole session; the next install attempt re-probes.
        return True
    _probe_cache[key] = ok
    return ok


def _running_interpreter_fallback() -> str:
    """The interpreter to persist when the candidate fails the import probe.

    The RUNNING interpreter by definition has ``hermes_cli`` importable
    (this module is executing), so the module-form entry under it is the
    safe landing when every candidate path failed the capability check.
    """
    return os.path.abspath(sys.executable)


def resolve_exec_command(project_root: Optional[Path] = None) -> str:
    """Build the absolute ``Exec=`` command line for ``hermes desktop``.

    Prefer the real ``hermes`` executable (argv[0] or PATH). When Hermes
    runs as a module with no launcher installed, use the current
    interpreter, also absolute.

    The persisted entry must be launch-context independent: whatever
    process writes it, the next launch must read and rewrite the same
    bytes. ``resolve_hermes_bin()`` prefers ``sys.argv[0]``, which differs
    per launch path (wrapper, repo script, ``python -m``), so for this
    one caller an argv[0] that points inside the checkout is not a
    durable installed launcher — skip it and resolve from PATH instead.
    Otherwise a broken entry keeps regenerating itself (the repo-script
    form pins a mutable uv interpreter path; the ``python -m`` form
    persists a bare ``<python> desktop`` that no DE can run).

    ``project_root`` pins which checkout counts as "internal"; defaults to
    the running checkout.
    """
    from hermes_cli.relaunch import resolve_hermes_bin

    bin_path = _resolve_hermes_bin_for_desktop_entry(
        resolve_hermes_bin, checkout_root=project_root
    )
    interpreter = _running_interpreter()
    if not _can_import_hermes_cli(Path(interpreter)):
        # The candidate interpreter cannot actually import hermes_cli.main
        # (checked in isolated mode from a neutral cwd — so the probe can't
        # be fooled by a checkout cwd or an inherited PYTHONPATH). Persisting
        # it would write a dead entry: the DE spawns the Exec line in a cold
        # environment where exactly this import has to succeed. Fall back to
        # the module form under the RUNNING interpreter, which by definition
        # has the CLI importable. Probe design follows the isolated-mode
        # capability check proposed by @nosliwhtes (#92122 review lineage,
        # commit 4150501f641) — cached here per-process so a desktop launch
        # pays the subprocess cost at most once.
        interpreter = _running_interpreter_fallback()
    if bin_path:
        resolved = Path(bin_path).resolve()
        if _needs_interpreter(resolved):
            # The resolved launcher is a Python script whose shebang points at
            # a NON-venv interpreter (e.g. the repo's `hermes` script with
            # `#!/usr/bin/env python3` when argv[0] came from the shell
            # installer's bash wrapper). Launched from the .desktop entry that
            # shebang resolves to the SYSTEM python and dies on the first
            # third-party import (#90292) — silently, since Terminal=false.
            # sys.executable is the interpreter actually running Hermes (the
            # venv one), so prefix it explicitly.
            argv = [interpreter, str(resolved), "desktop"]
        else:
            argv = [str(resolved), "desktop"]
    else:
        argv = [
            interpreter,
            "-m",
            "hermes_cli.main",
            "desktop",
        ]
    return " ".join(_quote_exec_arg(a) for a in argv)


def _resolve_hermes_bin_for_desktop_entry(
    resolve_fn=None,
    checkout_root: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the launcher binary for the persisted ``.desktop`` entry.

    Wraps :func:`hermes_cli.relaunch.resolve_hermes_bin` with one
    desktop-entry-specific rule: an ``argv[0]`` that points inside this
    checkout is a launch-context artifact (the repo ``hermes`` script the
    wrapper execs with, or an interpreter binary surfaced by programmatic
    relaunch paths), not a durable installed launcher. Persisting it makes
    the entry a function of however the previous launch happened — the
    bootstrap loop behind #90492's incomplete fix. Skip argv[0]/relative
    candidates in that case and fall through to PATH, where the shell
    installer's wrapper lives.

    ``resolve_fn`` is injectable for tests.
    """
    if resolve_fn is None:
        from hermes_cli.relaunch import resolve_hermes_bin as resolve_fn

    if checkout_root is None:
        checkout_root = _project_root()
    # Keep the LEXICAL form: _inside_checkout resolves candidates for its
    # own comparison anyway, and _wrapper_targets_checkout needs the
    # lexical root because the installer writes $INSTALL_DIR lexically
    # into the shim text (symlinked homes would otherwise mismatch).
    # Production callers pass main.py's realpath'd PROJECT_ROOT; the
    # module-lexical root derived from __file__ is added alongside so a
    # symlinked home still matches the shim's lexically-written paths.
    checkout_root = Path(os.path.abspath(checkout_root))
    module_lexical_root = _project_root()
    original_argv0 = sys.argv[0]

    def _inside_checkout(candidate: str) -> bool:
        try:
            path = Path(candidate).resolve()
        except OSError:
            return False
        # The repo `hermes` script and anything else shipped in the tree is
        # checkout-internal. Compare against BOTH the lexical and resolved
        # roots (checkout_root is kept lexical; candidates resolve, so a
        # symlinked home needs the resolved comparison too).
        resolved_root = None
        try:
            resolved_root = checkout_root.resolve()
        except OSError:
            pass
        for root in {checkout_root, resolved_root}:
            if root is not None and (path == root or root in path.parents):
                return True
        # The `python -m hermes_cli.main` relaunch context surfaces the
        # invoking interpreter (or a non-executable main.py, which the
        # resolver already skips) as argv[0]; an interpreter is never a
        # durable, launchable entry target (it would persist a bare
        # `<python> desktop`). Compare against the *invoking* interpreter
        # (argv[0]'s own file), not sys.executable — under test harnesses
        # they differ.
        try:
            if path.samefile(original_argv0) and _is_interpreter(path):
                return True
        except OSError:
            pass
        return False

    def _is_interpreter(candidate: Path) -> bool:
        """A python interpreter binary (``bin/python*``), not a launcher.

        Strict basename match — accepts ``python``, ``python3``,
        ``python3.11``, ``python2.7``; rejects lookalikes such as
        ``python3-config``, ``pythonw``, and anything else merely
        *containing* "python". Regex approach proposed independently in
        #94051; kept here with the parent-dir guard so a script named
        ``python`` outside a bin/Scripts tree is not misclassified.
        """
        import re

        name = candidate.name.lower()
        if not re.fullmatch(r"python[23]?(\d+)?(\.\d+)?", name):
            return False
        return candidate.parent.name in {"bin", "scripts"}

    # Resolve the primary FIRST and only rerun the resolver with argv[0]
    # hidden when the primary could actually be checkout-internal: for an
    # already-external primary the comparison can never change the
    # outcome, so skipping the rerun saves a resolver call and shortens
    # the window in which a concurrent reader could see the mutated
    # sys.argv.
    primary = resolve_fn()

    # A primary that is NOT checkout-internal and not the invoking
    # interpreter is an external launcher (e.g. /opt/.../bin/hermes from
    # another install method, or a venv console script). It must be
    # evaluated BEFORE any known-location probing: probing first could
    # silently switch the entry to a different installation (#94443
    # review case 3).
    if primary and not _inside_checkout(primary):
        return primary

    # Only reroute when argv[0] actually drove the resolution: re-run the
    # resolver with argv[0] hidden and compare. If PATH yields nothing,
    # keep the resolver's original answer (its fallback chain stays
    # authoritative; #90492 semantics preserved).
    sys.argv[0] = ""
    try:
        rerouted = resolve_fn()
    finally:
        sys.argv[0] = original_argv0

    if primary and _inside_checkout(primary) and rerouted:
        return rerouted

    if rerouted is None and primary:
        # argv[0] was checkout-internal AND PATH had no `hermes` — common
        # in stripped systemd user sessions and autostart relaunches.
        # The installer's wrapper lives at known locations; probe them
        # directly before giving up, otherwise we'd silently persist the
        # checkout-internal form this fix exists to prevent. The probe
        # runs only after the primary was proven non-durable above, and
        # each candidate must itself target THIS checkout (a wrapper
        # from another install would make the entry stable-but-wrong —
        # same failure class the external-primary-first rule avoids).
        probe = _known_wrapper_candidates()
        for candidate in probe:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                if not _wrapper_shebang_safe(candidate):
                    # The wrapper targets this checkout but its own shebang
                    # would die in the DE context (e.g. `#!/usr/bin/env
                    # python3` resolving past the venv): skip it the same
                    # way a foreign-install wrapper is skipped. Idea
                    # credited to autumn8's #92122 rung-2 safety check;
                    # implemented on our ownership machinery.
                    continue
                if _wrapper_targets_checkout(
                    candidate, checkout_root
                ) or _wrapper_targets_checkout(candidate, module_lexical_root):
                    return str(candidate)
        # No durable wrapper for THIS checkout exists anywhere (PATH
        # miss, known locations miss or belong to another install).
        # Persisting the checkout-internal primary would produce an
        # entry that regenerates itself or dies on the venv escape;
        # dropping to None lets resolve_exec_command emit its runnable
        # module fallback.
        return None
    return primary


def _wrapper_shebang_safe(wrapper: Path) -> bool:
    """Whether an executable wrapper can actually run in the DE context.

    A wrapper whose own shebang escapes the venv (``#!/usr/bin/env
    python3`` or a bare interpreter name) would die exactly like the
    broken entry this module exists to fix — the checkout reference in
    its body does not save it. Native binaries and shell launchers are
    safe by construction (they exec the right interpreter themselves).
    A python-shebang wrapper is safe only when its interpreter resolves
    to the RUNNING venv's interpreter directory.
    """
    try:
        with open(wrapper, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if head[:4] == b"\x7fELF" or head.startswith(b"MZ"):
        return True
    if not head.startswith(b"#!"):
        # No shebang: the kernel cannot exec it directly either — but it
        # may be sourced or exec'd via `sh` by DE-specific glue. Fail
        # safe toward the module fallback.
        return False
    shebang = head.decode("utf-8", errors="replace").splitlines()[0]
    tokens = shebang[2:].strip().split()
    if not tokens:
        return False
    interp = Path(tokens[0])
    # `#!/usr/bin/env bash` (the installer's own launcher form): `env`
    # here is the standard trick to find bash on PATH, and the script
    # itself execs the right interpreter. Only python-flavored `env`
    # shebangs are the escape hazard.
    if interp.name == "env":
        # Skip env's own flags (-S, -u VAR, ...) and inspect the first
        # real token: `env -S bash` is still a shell launcher.
        target = next(
            (Path(t) for t in tokens[1:] if not t.startswith("-")),
            Path(""),
        )
        if target.name in ("bash", "sh", "dash", "zsh", "ksh"):
            return True
        return not _shebang_escapes_running_env(shebang)
    if interp.name in ("bash", "sh", "dash", "zsh", "ksh"):
        # A shell launcher execs the right interpreter itself.
        return True
    if "python" not in interp.name.lower():
        # Not a python interpreter either — fail safe toward the module
        # fallback rather than trusting an unknown interpreter.
        return False
    # Python wrapper: its shebang must stay inside the RUNNING venv.
    return not _shebang_escapes_running_env(shebang)


def _wrapper_targets_checkout(wrapper: Path, checkout_root: Path) -> bool:
    """Whether a candidate launcher script actually launches THIS checkout.

    Expects the LEXICAL checkout root (the caller keeps it un-resolved):
    the installer writes ``$INSTALL_DIR`` lexically into the shim, so on a
    symlinked home the shim text and the resolved root would never match.
    Both lexical and resolved forms of the root are tried regardless, to
    tolerate either caller convention.

    The installer's shim is a small bash script that execs
    ``<checkout>/venv/bin/python <checkout>/hermes``; a venv console
    script carries the venv interpreter in its shebang. Either way, a
    text launcher belonging to this installation references the
    checkout path (or its venv) somewhere in its first few KB. A
    binary launcher (PyInstaller & friends) cannot be inspected that
    way — accept it, since binary installs are self-contained and the
    external-primary-first rule has already had its say.
    """
    try:
        head = wrapper.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x7fELF" in head[:4] or head.startswith(b"MZ"):
        # Native binary: cannot verify, and cannot be another checkout's
        # bash shim either — accept.
        return True
    try:
        text = head.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - defensive decode
        return False
    # Boundary-aware matching: a bare substring test would also accept
    # sibling paths that EXTEND this checkout's path (an old install
    # renamed aside as `<checkout>-old` or `<checkout>.bak`), silently
    # pointing the entry at that other installation. Require the
    # reference to end the path (quote, whitespace, or end-of-line
    # right after the root) or continue INTO it.
    # Compare both the resolved root and its lexical form: the installer
    # writes $INSTALL_DIR lexically, so with a symlinked home
    # (/home/user -> /mnt/disk/home/user) the shim's text carries the
    # lexical path while checkout_root arrives resolved.
    roots = {str(checkout_root)}
    lexical_root = os.path.abspath(str(checkout_root))
    roots.add(lexical_root)
    try:
        resolved_lexical = str(Path(lexical_root).resolve())
        roots.add(resolved_lexical)
    except OSError:
        pass
    for root in roots:
        for terminator in ('"', "'", " ", "\n", "\t", "\r", "$", "\x00"):
            if root + terminator in text:
                return True
        if text.rstrip("\r\n").endswith(root):
            return True
        # The shim's exec line continues INTO the checkout (…/python
        # <root>/hermes …): a path-continuation boundary is also a match.
        if root + "/" in text:
            return True
    return False


def _known_wrapper_candidates():
    """Durable installed-launcher locations, most likely first.

    Mirrors the installer's ``get_command_link_dir()`` layouts: user
    (``~/.local/bin``), root FHS (``/usr/local/bin``), and Termux
    (``$PREFIX/bin``). The wrapper is always named ``hermes``.
    """
    candidates = []
    home = Path.home()
    prefix = os.environ.get("PREFIX")
    if prefix:
        candidates.append(Path(prefix) / "bin" / "hermes")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        candidates.append(Path("/usr/local/bin/hermes"))
    candidates.append(home / ".local" / "bin" / "hermes")
    return candidates


def _project_root() -> Path:
    """This file lives at ``<checkout>/hermes_cli/linux_desktop_entry.py``.

    Lexical (no .resolve()): callers feed this into shim-text matching
    where the installer's lexically-written $INSTALL_DIR must be able to
    match; symlinked homes would break a resolved comparison.
    """
    return Path(os.path.abspath(__file__)).parent.parent


def _needs_interpreter(bin_path: Path) -> bool:
    """Whether ``bin_path`` is a Python script that must run under
    ``sys.executable`` to see Hermes' venv (rather than its own shebang)."""
    try:
        with open(bin_path, "rb") as fh:
            head = fh.readline(256)
    except OSError:
        return False
    if not head.startswith(b"#!"):
        # Native binary (uv tool shim, PyInstaller, distro package) — its own
        # loader is self-sufficient.
        return False
    shebang = head.decode("utf-8", errors="replace").strip()
    if "python" not in shebang.lower():
        # A shell wrapper (e.g. the installer's bash launcher) execs the venv
        # python itself — leave it alone.
        return False
    return _shebang_escapes_running_env(shebang)


def _shebang_escapes_running_env(shebang: str) -> bool:
    """Whether a python shebang resolves OUTSIDE the running interpreter's env.

    Tokenizes the shebang (interpreter path plus any flags) and compares
    PATH COMPONENTS, never substrings: ``<venv>/bin-extra/python`` is not
    inside ``<venv>/bin`` even though it starts with it (sibling-directory
    confusion; independently surfaced in nosliwhtes' #92122 hardening
    ``b96427d0`` — reimplemented here with two extensions).

    Extensions over the parent-equality form:

    * ``env`` shebangs (``#!/usr/bin/env python3``) ALWAYS escape: ``env``
      resolves through PATH, which in the DE's cold environment is not the
      interactive PATH that installed the venv — the parent-equality form
      could be fooled when the resolved ``env`` binary happens to sit in
      the same directory tree.
    * Flags after the interpreter (``-S``, ``-E``...) are stripped before
      comparing, so a legitimate ``#!<venv>/bin/python -S`` is not
      misclassified by comparing against the flag token.

    The comparison uses the LEXICAL interpreter directory (abspath, not
    resolve()): on uv venvs the resolved parent is the base interpreter's
    dir, which makes a valid ``.venv/bin/python`` shebang look foreign
    (#94443 review case 1). Both sides use the SAME case operation
    (``.lower()``): interpreter paths legitimately carry uppercase (conda
    env names, usernames, uv's ephemeral build dirs) and an asymmetric
    compare would flag the venv's own console script as foreign.
    """
    tokens = shebang[2:].strip().split()
    if not tokens:
        # Bare "#!python" with no path: resolves via PATH — escapes.
        return True
    interp = Path(tokens[0])
    if interp.name in ("env", "env.exe"):
        # PATH-resolved interpreter: the DE environment's PATH decides,
        # not the installing shell's — treat as escaping. A real
        # ``env -S`` venv-absolute form (`env -S <abs>`), rare but valid,
        # still resolves the actual interpreter from the second token.
        rest = [t for t in tokens[1:] if not t.startswith("-")]
        if rest and Path(rest[0]).is_absolute():
            interp = Path(rest[0])
        else:
            return True
    running_dir = os.path.dirname(os.path.abspath(sys.executable)).lower()
    return str(interp.parent).lower() != running_dir


def _quote_exec_arg(arg: str) -> str:
    """Quote one ``Exec`` argument per the desktop entry spec.

    Reserved characters require double quotes. Inside the quotes, escape
    a backslash and a double quote with a backslash.
    """
    if not any(c in arg for c in " \t\n\"'\\><~|&;$*?#()`"):
        return arg
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_desktop_entry(exec_command: str, icon: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Hermes\n"
        "GenericName=Hermes Desktop\n"
        "Comment=Launch Hermes Desktop\n"
        f"Exec={exec_command}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Utility;\n"
        "StartupNotify=true\n"
        "StartupWMClass=Hermes\n"
    )


def refresh_desktop_databases(applications_dir: Path) -> "list[str]":
    """Reindex the menu caches. Run each tool only when it exists.

    Return the names of the tools that ran (for logging and tests).
    """
    ran: list[str] = []

    update_db = shutil.which("update-desktop-database")
    if update_db:
        if _run_quiet([update_db, str(applications_dir)]):
            ran.append("update-desktop-database")

    # Plasma 6 first, then Plasma 5. Only one of them is ever installed.
    for tool in ("kbuildsycoca6", "kbuildsycoca5"):
        resolved = shutil.which(tool)
        if not resolved:
            continue
        if _run_quiet([resolved, "--noincremental"]):
            ran.append(tool)
        break

    return ran


def _run_quiet(cmd: "list[str]") -> bool:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _install_icon_to_hicolor(icon: Path) -> bool:
    """Copy the app icon into the user's hicolor icon theme tree.

    The freedesktop icon lookup finds an installed ``apps/hermes.png``
    by the unqualified name ``hermes``, so the entry can reference the
    icon without an absolute checkout path. The size subdirectory must
    be one the theme actually indexes (hicolor's index.theme lists
    fixed sizes and ``scalable`` — an unindexed dir like ``1024x1024``
    would never be found), so the icon lands in ``scalable`` unless the
    source is exactly 256x256, which goes to the fixed-size dir.
    Idempotent via content-compare; OSError caught internally (False) —
    the caller then falls back to the absolute path.
    """
    try:
        raw = icon.read_bytes()
        is_256 = False
        if len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            width, height = struct.unpack(">II", raw[16:24])
            is_256 = (width, height) == (256, 256)
        subdir = "256x256" if is_256 else "scalable"
        dest = _xdg_data_home() / "icons" / "hicolor" / subdir / "apps" / "hermes.png"
        if dest.is_file() and dest.read_bytes() == raw:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(icon, dest)
        return True
    except OSError:
        return False


def install_desktop_entry(project_root: Path) -> Optional[Path]:
    """Write (or refresh) the Hermes desktop entry. Return its path.

    Return ``None`` on non-Linux platforms or when the write fails. This
    is a convenience, never a reason to fail a launch.
    """
    if not is_supported():
        return None

    entry_path = desktop_entry_path()
    icon = icon_path(project_root)
    # Prefer the themed name: the icon is COPIED into the user's hicolor
    # tree, so the entry outlives the checkout (moving/archiving the
    # checkout would break an absolute Icon= path — the same
    # durability class the Exec line was fixed for). Fall back to the
    # absolute path only when the copy is impossible (read-only tree),
    # and to the themed name when the checkout has no icon at all.
    icon_value = str(icon) if icon.is_file() else "hermes"
    if icon.is_file() and _install_icon_to_hicolor(icon):
        icon_value = "hermes"
    contents = render_desktop_entry(resolve_exec_command(project_root), icon_value)

    try:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        # When nothing changed, skip the rewrite. Then a launch does not
        # churn the menu caches.
        if entry_path.is_file() and entry_path.read_text(encoding="utf-8") == contents:
            return entry_path
        # Atomic replace: an interrupted plain write can leave a zero-byte
        # entry, which permanently breaks the taskbar pin (nothing later
        # rewrites a file that exists at the right path). The temp+rename
        # dance in utils.atomic_write_text is the codebase's shared
        # implementation — ported from #80547, which closed unmerged with
        # this piece unlanded.
        from utils import atomic_write_text

        atomic_write_text(entry_path, contents, create_mode=0o755)
        # Some launchers (and older Plasma) offer the entry only when it
        # is executable.
        entry_path.chmod(0o755)
    except OSError:
        return None

    refresh_desktop_databases(entry_path.parent)
    return entry_path
