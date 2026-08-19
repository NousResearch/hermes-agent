"""``hermes desktop --cwd`` / ``--hermes-root`` accept Git Bash / MSYS paths.

Sibling of the ``--in`` fix (tests/hermes_cli/test_in_dir_msys_paths.py,
#85865): under Git Bash, ``hermes desktop --cwd ~`` reaches the native CLI
as ``/c/Users/<user>`` — the shell expands ``~`` to an MSYS POSIX path and
MSYS2's automatic argument conversion is disabled for native executables.
``cmd_gui`` passed ``--cwd``/``--hermes-root`` straight through
``Path(...).expanduser().resolve()`` with no MSYS translation, so
``HERMES_DESKTOP_CWD``/``HERMES_DESKTOP_HERMES_ROOT`` would carry a bogus
``<drive>:\\c\\Users\\...`` path on Windows (``Path(...).resolve()`` treats
an untranslated ``/c/Users/alice`` as drive-relative to the CLI's own CWD).

The translation itself (``_msys_to_windows_path``) has exhaustive unit
coverage in tests/tools/test_local_env_windows_msys.py; this pins the
``cmd_gui`` call sites actually applying it, using the same source-level
guard technique as the ``--in`` test (cmd_gui's downstream Electron spawn
makes a full behavioral invocation impractical to test safely).
"""

import inspect

import hermes_cli.main as main_mod


def test_cmd_gui_hermes_root_and_cwd_use_msys_translation():
    """Guard: --hermes-root/--cwd resolution in cmd_gui must route through
    _msys_to_windows_path, same as the --in fix. A plain expanduser/resolve
    does not survive Git Bash's ~ expansion to /c/Users/... form."""
    src = inspect.getsource(main_mod.cmd_gui)
    idx = src.find('if getattr(args, "hermes_root", None):')
    assert idx != -1, "cmd_gui's --hermes-root resolution block moved; update this test"
    block = src[idx : idx + 800]
    assert "_msys_to_windows_path" in block, (
        "cmd_gui no longer translates MSYS paths for --hermes-root/--cwd; "
        "Git Bash `hermes desktop --cwd ~` will resolve to a bogus "
        "<drive>:\\c\\Users\\... path"
    )
    assert '"hermes_root"' in block and '"cwd"' in block, (
        "expected both --hermes-root and --cwd handling in this block"
    )
