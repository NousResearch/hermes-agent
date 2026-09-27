"""The pm.shell module must compile without escape-sequence warnings.

The module docstring documents literal Windows paths (C:\\Windows\\System32\\
bash.exe, WindowsApps\\bash.exe). In a non-raw string those backslashes are
escape sequences: ``\\W`` is invalid and raises a SyntaxWarning at compile
time on Python 3.12+ (#125342), and ``\\b`` is a valid backspace escape that
silently corrupted the documented path. Guards the raw-string docstring.
"""
import ast
import pathlib
import warnings

import pm.shell

SOURCE = pathlib.Path(pm.shell.__file__).read_text(encoding="utf-8")


def test_module_compiles_without_escape_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        # compile() re-compiles from source, so any invalid escape sequence
        # in the file (docstring included) raises instead of warning.
        compile(SOURCE, pm.shell.__file__, "exec")


def test_docstring_documents_windows_paths_literally():
    doc = ast.get_docstring(ast.parse(SOURCE))
    assert doc is not None
    assert r"C:\Windows\System32\bash.exe" in doc
    # "\b" used to be swallowed as a backspace escape here.
    assert r"WindowsApps\bash.exe" in doc
    assert "\x08" not in doc
