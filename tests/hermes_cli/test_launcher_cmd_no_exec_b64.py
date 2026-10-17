"""The Windows ``.cmd`` fallback must not carry the launcher script in its command line (#122463).

EDR static engines flag the ``python -c "import base64; exec(base64.b64decode('...'))"`` one-liner
as a Metasploit stager, and the inlined form had also grown past cmd.exe's 8191-character line
limit. The fallback therefore runs a sibling ``<name>-script.py`` holding the same bootstrap.

These tests force the Windows branch but never execute the ``.cmd``, so they run on every lane.
"""

import base64
import py_compile
import sys
from pathlib import Path

import hermes_cli._launchers as launchers

#: cmd.exe truncates a command line past this; the old inlined launcher had grown to ~9.5k.
CMD_LINE_LIMIT = 8191


def _mint_cmd(tmp_path, monkeypatch, name="hermes"):
    """Mint the Windows fallback: no distlib ScriptMaker, so it must fall through to the .cmd."""
    monkeypatch.setattr(launchers, "_is_windows", lambda: True)
    monkeypatch.setattr(launchers, "_load_script_maker", lambda: None)
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    target = launchers.mint_launcher(name, repo, out, Path(sys.executable), None)
    assert target is not None and target.suffix == ".cmd", target
    return target


def test_cmd_line_carries_no_exec_of_base64(tmp_path, monkeypatch):
    """The flagged shape must be gone from the emitted command line, not merely reworded."""
    body = _mint_cmd(tmp_path, monkeypatch).read_text(encoding="utf-8")
    assert "b64decode" not in body
    assert "exec(" not in body
    assert "base64" not in body


def test_cmd_line_stays_under_the_cmd_exe_limit(tmp_path, monkeypatch):
    """The bootstrap no longer rides the command line, so its length no longer decides if it runs."""
    body = _mint_cmd(tmp_path, monkeypatch).read_text(encoding="utf-8")
    line = next(entry for entry in body.splitlines()[1:] if entry.strip())
    assert len(line) < CMD_LINE_LIMIT, len(line)
    # The inlined script it replaced would not have fit.
    script = launchers._launcher_script("hermes", tmp_path / "repo", None)
    assert len(base64.b64encode(script.encode()).decode()) > CMD_LINE_LIMIT


def test_sibling_script_carries_the_bootstrap_verbatim(tmp_path, monkeypatch):
    target = _mint_cmd(tmp_path, monkeypatch)
    sibling = target.with_name("hermes-script.py")
    assert sibling.is_file()
    assert sibling.read_text(encoding="utf-8") == launchers._launcher_script(
        "hermes", tmp_path / "repo", None
    )


def test_cmd_delegates_to_the_sibling_with_isolation_and_args(tmp_path, monkeypatch):
    body = _mint_cmd(tmp_path, monkeypatch).read_text(encoding="utf-8")
    assert "-I" in body  # isolated mode, as the old -c line
    assert "%*" in body  # CLI args still forwarded
    # %~dp0, not a baked path: the launcher keeps working after the install moves.
    assert "%~dp0hermes-script.py" in body


def test_sibling_script_is_valid_python(tmp_path, monkeypatch):
    target = _mint_cmd(tmp_path, monkeypatch)
    assert py_compile.compile(str(target.with_name("hermes-script.py")), doraise=True)


def test_failed_cmd_write_leaves_no_orphan_script(tmp_path, monkeypatch):
    """No ``.cmd`` was ever published, so the script is a true orphan: nothing references it and
    the next mint would mistake it for current. (With a ``.cmd`` present it must STAY, since that
    ``.cmd`` runs it: see ``test_failed_cmd_replace_keeps_the_script_a_published_cmd_runs``.)"""
    real_write = launchers._write_atomic

    def fail_cmd(target, write):
        return None if target.suffix == ".cmd" else real_write(target, write)

    monkeypatch.setattr(launchers, "_write_atomic", fail_cmd)
    monkeypatch.setattr(launchers, "_is_windows", lambda: True)
    monkeypatch.setattr(launchers, "_load_script_maker", lambda: None)
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    assert launchers.mint_launcher("hermes", repo, out, Path(sys.executable), None) is None
    assert not (out / "hermes-script.py").exists()


def test_acp_launcher_gets_its_own_sibling(tmp_path, monkeypatch):
    """The sibling is named per command: hermes-acp must not run hermes' script."""
    target = _mint_cmd(tmp_path, monkeypatch, name="hermes-acp")
    assert target.with_name("hermes-acp-script.py").is_file()
    assert not target.with_name("hermes-script.py").exists()


def test_repin_guard_still_reads_the_interpreter_from_the_new_cmd(tmp_path, monkeypatch):
    """``_launcher_python`` backs the shared-launcher guard: it must see the interpreter through
    the delegation line, or a repin would treat every published .cmd as foreign."""
    target = _mint_cmd(tmp_path, monkeypatch)
    assert launchers._launcher_python(target) == Path(sys.executable)


def test_failed_cmd_replace_keeps_the_script_a_published_cmd_runs(tmp_path, monkeypatch):
    """A failed ``.cmd`` write must not delete the script the SURVIVING ``.cmd`` runs.

    ``_write_atomic`` stages and ``os.replace``s, so when the replace fails (an on-write scanner
    holding the freshly written file, a watcher, any handle without delete sharing) the previously
    published ``.cmd`` is left exactly as it was, and that ``.cmd`` delegates to
    ``%~dp0hermes-script.py``. Unlinking the script then turns a retryable mint miss into a command
    that exits 2, and the in-band repair route (``hermes update``) can no longer start through the
    launcher it would heal.
    """
    published = _mint_cmd(tmp_path, monkeypatch)
    script = published.with_name("hermes-script.py")
    assert script.is_file()

    real_write = launchers._write_atomic

    def fail_cmd(target, write):
        return None if target.suffix == ".cmd" else real_write(target, write)

    monkeypatch.setattr(launchers, "_write_atomic", fail_cmd)
    out = published.parent
    assert launchers.mint_launcher("hermes", tmp_path / "repo", out, Path(sys.executable), None) is None

    assert published.is_file()  # the replace failed, so the previous .cmd survives
    assert "%~dp0hermes-script.py" in published.read_text(encoding="utf-8")
    assert script.is_file(), "the surviving .cmd still runs this script"
    assert py_compile.compile(str(script), doraise=True)
