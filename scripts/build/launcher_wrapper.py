"""Bundled-payload CLI entry wrapper — the distlib launcher's zip overlay.

scripts/build/launchers.py reads this file, substitutes the four
HERMES_* placeholders (see the constants below), and hands the result to
a distlib ScriptMaker as
the script text. On win32 the minted artifact is a real PE: the distlib
launcher stub + a shebang whose interpreter is the ``<launcher_dir>``
placeholder form (resolved at runtime relative to the launcher's own
directory) + this script packed as __main__.py in a zip overlay. The
The wrapper replaces everything the old rust shim + its sidecar did
(plan: pm-clean):

  * resolve the launcher's own directory from sys.argv[0] (the distlib
    launcher puts its exe path there — the shebang's ``<launcher_dir>``
    placeholder resolves the interpreter the same way),
  * put the payload repo snapshot FIRST and the venv site-packages second
    on sys.path (same order, same reason as the old shim: the repo's
    hermes_cli wins over anything stale in site-packages — the sealed
    payload's venv has no working editable install, its pointer names the
    BUILD machine). The site entry goes through ``site.addsitedir()`` —
    the only mechanism that runs ``.pth`` files — because pywin32.pth is
    what puts win32\\lib on sys.path and therefore what makes
    ``import pywintypes`` resolve on Windows bundles (portalocker's
    Win32Locker → concurrent-log-handler → hermes_logging.py's files),
  * drop inherited PYTHONPATH / PYTHONHOME so foreign installs can never
    shadow the bundle,
  * default sys.pycache_prefix to the user-level cache (%LOCALAPPDATA%
    \\hermes\\pycache on win32, ~/.cache/hermes-pycache elsewhere) when the
    user has not set one — the sealed payload must never see __pycache__
    writes (signature-breaking on mac, read-only mount on AppImage/MSIX).

The whole file is import-safe so tests can drive configure()/main()
directly (tests/scripts/test_desktop_cli_wrapper.py).
"""

import importlib
import os
import site
import sys

HERMES_ENTRY_MODULE = "__HERMES_ENTRY_MODULE__"
HERMES_ENTRY_FUNC = "__HERMES_ENTRY_FUNC__"
HERMES_REPO_REL = "__HERMES_REPO_REL__"
HERMES_SITE_REL = "__HERMES_SITE_REL__"


def _join_rel(here, rel):
    """Join a forward-slash relative path (the minted-in convention, same
    as the payload manifest) onto a host dir without os.sep assumptions."""
    return os.path.join(here, *rel.split("/"))


def launcher_dir(argv0):
    """The directory of the launcher itself, from sys.argv[0]. abspath
    because cwd is meaningless for a GUI/alias launch."""
    return os.path.dirname(os.path.abspath(argv0))


def payload_sys_paths(here):
    """Import roots for the sealed payload: repo first (wins), then the
    venv's dependency tree — mirroring the old shim's PYTHONPATH order."""
    return [_join_rel(here, HERMES_REPO_REL), _join_rel(here, HERMES_SITE_REL)]


def default_pycache_dir(environ):
    """User-level bytecode cache. The sealed payload is read-only at
    runtime (or signature-sealed, on mac), so __pycache__ writes must
    land outside it. Only consulted when the user has not set their own
    PYTHONPYCACHEPREFIX."""
    if sys.platform == "win32":
        base = environ.get("LOCALAPPDATA")
        return os.path.join(base, "hermes", "pycache") if base else None
    base = environ.get("HOME")
    return os.path.join(base, ".cache", "hermes-pycache") if base else None


def configure(here, environ=None):
    """Point THIS process (the minted launcher's python) at the payload.
    Returns the sys.path entries prepended, for tests."""
    environ = os.environ if environ is None else environ
    # Same hygiene as the old rust shim: an inherited PYTHONPATH could
    # shadow bundled modules with foreign ones; PYTHONHOME would repoint
    # the stdlib entirely.
    environ.pop("PYTHONPATH", None)
    environ.pop("PYTHONHOME", None)
    repo_entry, site_entry = payload_sys_paths(here)
    # Repo snapshot first — its hermes_cli wins over anything stale in
    # site-packages (the sealed payload has no working editable install).
    sys.path.insert(0, repo_entry)
    # addsitedir(), not a raw append: only it processes the venv's .pth
    # files. pywin32.pth is load-bearing on Windows — it puts win32\lib
    # on sys.path, which is what makes `import pywintypes` resolve, and
    # without that portalocker's Win32Locker dies and
    # concurrent-log-handler silently drops every file-log record (the
    # same trap gateway/run.py's MCP venv bootstrap works around). Keep
    # site-packages directly after the repo, ahead of .pth-added dirs.
    site.addsitedir(site_entry)
    if site_entry in sys.path:
        sys.path.remove(site_entry)
    sys.path.insert(1 if sys.path and sys.path[0] == repo_entry else 0, site_entry)
    if not environ.get("PYTHONPYCACHEPREFIX"):
        default = default_pycache_dir(environ)
        if default:
            environ["PYTHONPYCACHEPREFIX"] = default
            # The env var is only read at interpreter startup; we ARE at
            # startup, but setting it in os.environ cannot retro-activate
            # it — sys.pycache_prefix is the live switch.
            sys.pycache_prefix = default
    return [repo_entry, site_entry]


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    here = launcher_dir(argv[0] if argv else sys.argv[0])
    configure(here)
    importlib.import_module("hermes_bootstrap")
    module = importlib.import_module(HERMES_ENTRY_MODULE)
    target = getattr(module, HERMES_ENTRY_FUNC)
    # main() reads sys.argv; hand it the real argv (argv[0] stays the
    # launcher path, exactly like a console-script entry point).
    return target()


if __name__ == "__main__":
    sys.exit(main())
