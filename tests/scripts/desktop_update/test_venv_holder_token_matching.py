"""Token-exact venv-holder classification for the Windows update guard (rm-018).

``_detect_venv_python_processes`` nominates processes for ``taskkill /T /F`` when their
argv references the venv. The old decision matched SUBSTRINGS against the joined cmdline
(``venv_prefix in cmdline_low``), so an unrelated python process whose argv merely mentions
the venv path inside an argument (a file named after it, a ``-c`` snippet, a near-miss
module name) was nominated for a force-kill — the ``"serve" in cmdline`` bug class
(#87594/#90778). These tests pin the whole-token contract: a match requires an argv token
that IS a path under the venv (or the value of a ``--flag=path`` token), or a literal
``-m hermes_cli.main`` invocation anchored to the project root.

The predicate is pure string/token logic (platform arrives as data), so these run on every
host; the scan-level cases mock psutil exactly like tests/hermes_cli/test_update_venv_health.py.
"""

import sys
import types
from unittest.mock import MagicMock, patch

from hermes_cli import main as cli_main
from hermes_cli.update_cmd_windows import _cmdline_indicates_venv_holder, _detect_venv_python_processes

VENV_PREFIX = r"c:\proj\venv" + "\\"
ROOT_PREFIX = r"c:\proj" + "\\"


def holder(tokens, *, cwd=None, venv_prefix=VENV_PREFIX, root_prefix=ROOT_PREFIX):
    get_cwd = None
    if cwd is not None:
        get_cwd = lambda: cwd() if callable(cwd) else cwd  # noqa: E731
    return _cmdline_indicates_venv_holder(
        tokens, venv_prefix=venv_prefix, root_prefix=root_prefix, get_cwd=get_cwd
    ), get_cwd


# ---------------------------------------------------------------------------
# Decoys: argv that MENTIONS the venv / module but is not a holder
# ---------------------------------------------------------------------------


def test_venv_path_inside_a_dash_c_snippet_is_not_a_holder():
    # The whole -c payload is ONE argv token; it merely contains the venv path.
    tokens = ["python.exe", "-c", "print(open(r'C:\\proj\\venv\\Scripts\\config.py').read())"]
    matched, _ = holder(tokens, cwd=ROOT_PREFIX)
    assert not matched


def test_file_named_after_the_venv_directory_is_not_a_holder():
    # Sibling of the venv dir (venv-notes) and a decoy file whose name embeds the path.
    assert not holder(["python.exe", "analyze.py", r"C:\proj\venv-notes\total.txt"], cwd=ROOT_PREFIX)[0]
    assert not holder(["python.exe", "cat.py", "notes_about_c_proj_venv_Scripts.txt"], cwd=ROOT_PREFIX)[0]


def test_module_name_as_a_plain_argument_is_not_a_holder():
    # `analyze.py hermes_cli.main` passes the module NAME as data; no `-m` adjacency.
    assert not holder(["python.exe", "analyze.py", "hermes_cli.main"], cwd=ROOT_PREFIX)[0]


def test_near_miss_module_name_is_not_a_holder():
    assert not holder(["python.exe", "-m", "hermes_cli.mainx", "serve"], cwd=ROOT_PREFIX)[0]


def test_unanchored_module_invocation_is_not_a_holder():
    # Real `-m hermes_cli.main` but cwd outside the project and no root token.
    assert not holder(["python.exe", "-m", "hermes_cli.main", "serve"], cwd=r"c:\elsewhere" + "\\")[0]


def test_path_inside_a_larger_argument_is_not_a_holder():
    # venv path embedded mid-argument (old substring code matched this → force-kill).
    tokens = ["python.exe", "log.py", f"--source={VENV_PREFIX}Scripts", "--tail"]
    assert not holder(tokens, cwd=r"c:\elsewhere" + "\\")[0]


# ---------------------------------------------------------------------------
# Real holders: whole tokens (or --flag=value) that ARE venv/root paths
# ---------------------------------------------------------------------------


def test_trampoline_with_venv_python_as_whole_token_is_a_holder():
    tokens = ["python.exe", r"C:\proj\venv\Scripts\python.exe", "serve"]
    assert holder(tokens, cwd=r"c:\elsewhere" + "\\")[0]


def test_flag_value_pointing_into_the_venv_is_not_a_holder():
    # ``--python=...`` values are not whole tokens; nothing in this repo invokes uv that way,
    # and a ``--flag=<venv path>`` argument (log/source targets) is a bystander, not a loader.
    tokens = ["uv.exe", "run", r"--python=C:\proj\venv\Scripts\python.exe", "serve"]
    assert not holder(tokens)[0]
    assert not holder(["python.exe", "log.py", f"--source={VENV_PREFIX}Scripts"], cwd=r"c:\elsewhere" + "\\")[0]


def test_quoted_venv_path_token_is_a_holder():
    tokens = ['python.exe', '"C:\\proj\\venv\\Scripts\\python.exe"', "serve"]
    assert holder(tokens)[0]


def test_venv_directory_token_itself_is_a_holder():
    assert holder(["python.exe", "C:\\proj\\venv"], cwd=r"c:\elsewhere" + "\\")[0]


def test_module_invocation_anchored_by_cwd_is_a_holder():
    matched, get_cwd = holder(["python.exe", "-m", "hermes_cli.main", "serve"], cwd=ROOT_PREFIX)
    assert matched and get_cwd is not None


def test_module_invocation_anchored_by_root_token_never_reads_cwd():
    calls = []

    def cwd():
        calls.append(1)
        return r"c:\elsewhere" + "\\"

    tokens = ["python.exe", "-m", "hermes_cli.main", "serve", r"C:\proj\data\config.yaml"]
    matched, _ = holder(tokens, cwd=cwd)
    assert matched and calls == []


def test_venv_path_token_short_circuits_without_cwd_call():
    calls = []

    def cwd():
        calls.append(1)
        return ROOT_PREFIX

    matched, _ = holder(["python.exe", r"C:\proj\venv\Scripts\python.exe"], cwd=cwd)
    assert matched and calls == []


def test_exact_match_is_case_insensitive():
    assert holder(["PYTHON.EXE", r"C:\PROJ\VENV\Scripts\python.exe"], cwd=r"c:\elsewhere" + "\\")[0]


# ---------------------------------------------------------------------------
# Scan level: the rewire inside _detect_venv_python_processes
# ---------------------------------------------------------------------------


def _proc(pid, exe, name, cmdline=None, cwd=""):
    proc = MagicMock()
    proc.info = {"pid": pid, "exe": exe, "name": name}
    proc.cmdline.return_value = cmdline or []
    proc.cwd.return_value = cwd
    return proc


def _fake_psutil(procs):
    me = MagicMock()
    me.parents.return_value = []
    return types.SimpleNamespace(process_iter=lambda attrs: iter(procs), Process=lambda *a, **k: me)


def test_scan_matches_trampoline_whose_venv_token_contains_spaces(tmp_path):
    # Spaces in the path survive only because argv stays tokenized — a joined-and-split
    # string would break `venv/Scripts dir/python.exe` into pieces and lose the match.
    venv_py = str(tmp_path / "venv" / "Scripts dir" / "python.exe")
    tramp = _proc(201, r"C:\Python311\python.exe", "python.exe", ["python.exe", venv_py, "serve"])
    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.dict(sys.modules, {"psutil": _fake_psutil([tramp])}):
        matches = _detect_venv_python_processes()
    assert [pid for pid, _, _ in matches] == [201]
    assert matches[0][2] == f"python.exe {venv_py} serve"


def test_scan_rejects_decoy_mentioning_venv_in_an_argument(tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    decoy = _proc(
        202,
        r"C:\Python311\python.exe",
        "python.exe",
        ["python.exe", "cat.py", f"notes-{venv_py}-today.txt"],
        str(tmp_path),
    )
    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.dict(sys.modules, {"psutil": _fake_psutil([decoy])}):
        assert _detect_venv_python_processes() == []
