"""Windows .cmd fallback must not emit exec(base64.b64decode(...)) (#122463).

EDR heuristics flag the python -c "import base64;
exec(base64.b64decode('...'))" one-liner as commodity malware (Metasploit).
The fallback .cmd must therefore delegate to a sibling .py file that
carries the launcher script verbatim, instead of embedding an exec-of-decode
program. These tests are host-independent: they force the Windows branch and
never execute the .cmd itself.
"""

import py_compile
import sys
from pathlib import Path

import hermes_cli._launchers as launchers


def _mint_cmd(tmp_path, monkeypatch, name="hermes"):
    monkeypatch.setattr(launchers, "_is_windows", lambda: True)
    monkeypatch.setattr(launchers, "_load_script_maker", lambda: None)
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    target = launchers.mint_launcher(name, repo, out, Path(sys.executable), None)
    assert target is not None and target.suffix == ".cmd"
    return target


class TestCmdEmissionShape:
    def test_no_exec_b64decode_in_cmd_body(self, tmp_path, monkeypatch):
        body = _mint_cmd(tmp_path, monkeypatch).read_text(encoding="utf-8")
        assert "b64decode" not in body
        assert "exec(" not in body

    def test_sibling_script_carries_identical_payload(self, tmp_path, monkeypatch):
        target = _mint_cmd(tmp_path, monkeypatch)
        sibling = target.with_name("hermes.py")
        assert sibling.is_file()
        assert sibling.read_text(encoding="utf-8") == launchers._launcher_script(
            "hermes", tmp_path / "repo", None
        )

    def test_cmd_forwards_to_sibling_with_isolation_and_args(self, tmp_path, monkeypatch):
        body = _mint_cmd(tmp_path, monkeypatch).read_text(encoding="utf-8")
        assert "-I" in body  # isolated mode, as the old -c line
        assert "%*" in body  # CLI args still forwarded
        assert "hermes.py" in body

    def test_sibling_is_valid_python(self, tmp_path, monkeypatch):
        target = _mint_cmd(tmp_path, monkeypatch)
        assert py_compile.compile(str(target.with_name("hermes.py")), doraise=True)
