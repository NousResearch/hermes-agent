"""Run code inside a venv on another interpreter: ``python -S _venv_entry.py SITE ARGS...``.

ARGS are what would follow the venv interpreter's own options: ``SCRIPT [ARG...]``,
``-c CODE [ARG...]`` or ``-m MODULE [ARG...]``. Built by ``pm.environments.venv_command``
for sealed payloads, whose venv redirectors must not run (see there). Stdlib only.
"""
import os
import runpy
import site
import sys
import types


def _usage() -> None:
    raise SystemExit("usage: _venv_entry.py SITE (SCRIPT | -c CODE | -m MODULE) [ARG...]")


def main() -> None:
    if len(sys.argv) < 3:
        _usage()
    site_dir, args = sys.argv[1], sys.argv[2:]
    here = os.path.dirname(os.path.abspath(__file__))
    # Python put this file's directory first; it is PM's package dir, not the caller's.
    sys.path[:] = [entry for entry in sys.path
                   if not entry or os.path.abspath(entry) != here]
    # A venv's site-packages follows the stdlib; addsitedir also runs its .pth files
    # (pywin32's puts win32\lib on sys.path).
    site.addsitedir(site_dir)
    safe_path = sys.flags.safe_path if hasattr(sys.flags, "safe_path") else sys.flags.isolated
    if args[0] == "-c":
        if len(args) < 2:
            _usage()
        sys.argv = ["-c", *args[2:]]
        if not safe_path:
            sys.path.insert(0, "")
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
        exec(compile(args[1], "<string>", "exec"), main_module.__dict__)
    elif args[0] == "-m":
        if len(args) < 2:
            _usage()
        sys.argv = [args[1], *args[2:]]
        if not safe_path:
            sys.path.insert(0, os.getcwd())
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    elif args[0].startswith("-"):
        _usage()
    else:
        sys.argv = list(args)
        if not safe_path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(args[0])))
        runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
