from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "apple" / "apple-notes"
SCRIPT_PATH = SKILL_DIR / "scripts" / "apple_notes.py"
SKILL_MD = SKILL_DIR / "SKILL.md"


def load_module():
    spec = importlib.util.spec_from_file_location("apple_notes_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frontmatter() -> str:
    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.search(r"^---\s*\n(.*?)\n---\s*$", text, re.MULTILINE | re.DOTALL)
    assert m, "SKILL.md frontmatter not found"
    return m.group(1)


# --- skill metadata (HARDLINE authoring standards) ---


def test_description_is_short_one_sentence_ending_with_period():
    desc = None
    for line in _frontmatter().splitlines():
        m = re.match(r"^description:\s*(.*)$", line)
        if m:
            desc = m.group(1).strip()
            break
    assert desc, "description field missing from frontmatter"
    assert len(desc) <= 60, f"description is {len(desc)} chars, must be <= 60: {desc!r}"
    assert desc.endswith("."), f"description must end with a period: {desc!r}"
    assert desc.count(".") == 1, f"description must be one sentence: {desc!r}"


def test_platforms_gated_to_macos_only():
    m = re.search(r"^platforms:\s*(.*)$", _frontmatter(), re.MULTILINE)
    assert m, "platforms field missing"
    platforms = m.group(1).lower()
    assert "macos" in platforms, "osascript skill must declare macos"
    assert "linux" not in platforms and "windows" not in platforms, (
        "osascript is macOS-only; no other platforms may be claimed"
    )


def test_author_credits_human_contributor_first():
    m = re.search(r"^author:\s*(.*)$", _frontmatter(), re.MULTILINE)
    assert m, "author field missing"
    author = m.group(1).strip()
    assert "lishix520" in author, f"contributor handle missing from author: {author!r}"
    handle_idx = author.find("lishix520")
    hermes_idx = author.lower().find("hermes agent")
    assert hermes_idx == -1 or handle_idx < hermes_idx, (
        f"human contributor must be credited before 'Hermes Agent': {author!r}"
    )


def test_skill_md_names_terminal_tool_and_helper_script():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`terminal`" in text, "prose must name the Hermes `terminal` tool"
    assert "scripts/apple_notes.py" in text, "SKILL.md must reference the helper script by path"
    assert "scripts/apple_notes.py" in text


# --- AppleScript string building (pure, no macOS needed) ---


def test_escape_applescript_string():
    mod = load_module()
    assert mod.escape_applescript_string("plain") == "plain"
    assert mod.escape_applescript_string('he said "hi"') == 'he said \\"hi\\"'
    assert mod.escape_applescript_string("back\\slash") == "back\\\\slash"


def test_text_to_notes_html_escapes_markup_and_breaks_lines():
    mod = load_module()
    out = mod.text_to_notes_html("<b>hi</b>\nline2")
    assert out == "&lt;b&gt;hi&lt;/b&gt;<br>line2"
    # user content must not survive as live markup
    assert "<b>" not in out
    malicious = "<script>alert(1)</script>"
    assert "<script>" not in mod.text_to_notes_html(malicious)


def test_build_create_note_script_escapes_title_and_scopes_to_folder():
    mod = load_module()
    script = mod.build_create_note_script('Title "X"', "<div>body</div>", folder="Notes")
    assert "make new note with properties" in script
    assert "tell first folder whose name is \"Notes\"" in script
    # the unescaped title (with a raw double quote) must NOT appear
    assert 'Title "X"' not in script
    # the escaped form must appear
    assert 'Title \\"X\\"' in script


def test_build_create_note_script_defaults_to_first_folder():
    mod = load_module()
    script = mod.build_create_note_script("T", "body")
    assert "tell first folder\n" in script
    assert "whose name is" not in script


def test_build_append_note_script_concatenates_existing_body():
    mod = load_module()
    script = mod.build_append_note_script("T", "<br>more", folder="F")
    assert "set n to first note whose name is" in script
    assert "set body of n to (body of n) &" in script
    assert "<br>more" in script


def test_build_append_note_script_without_folder_searches_all():
    mod = load_module()
    script = mod.build_append_note_script("T", "<br>more")
    assert "set n to first note whose name is" in script
    assert "tell first folder" not in script


def test_build_move_note_script_uses_dest_variable_and_src_scope():
    mod = load_module()
    script = mod.build_move_note_script("T", dest="Dest", src="Src")
    assert "set destFolder to first folder whose name is" in script
    assert "tell first folder whose name is \"Src\"" in script
    assert "move first note whose name is" in script
    assert "to destFolder" in script


def test_build_create_folder_script():
    mod = load_module()
    script = mod.build_create_folder_script("Project X")
    assert "make new folder with properties" in script
    assert "Project X" in script


def test_build_list_folders_and_search_scripts():
    mod = load_module()
    folders = mod.build_list_folders_script()
    assert folders.startswith('tell application "Notes"')
    assert "every folder" in folders
    assert "whose name contains" in mod.build_search_script("query")


def test_build_list_notes_script_scopes_to_named_folder():
    mod = load_module()
    script = mod.build_list_notes_script("My Folder")
    # must use a tell block so `whose` filters the folder, not the notes;
    # the one-liner `every note of first folder whose name is F` mis-parses
    # and silently returns nothing.
    assert "tell first folder whose name is \"My Folder\"" in script
    assert "get name of every note" in script
    assert "every note of first folder whose" not in script


# --- execution layer ---


def test_run_applescript_raises_on_nonzero_exit():
    mod = load_module()
    fake = MagicMock(returncode=1, stdout="", stderr="not authorized")
    with patch("subprocess.run", return_value=fake):
        with pytest.raises(RuntimeError, match="not authorized"):
            mod.run_applescript('tell application "Notes" to get name of every folder')


def test_run_applescript_returns_stdout_on_success():
    mod = load_module()
    fake = MagicMock(returncode=0, stdout="Notes\nPersonal\n", stderr="")
    with patch("subprocess.run", return_value=fake) as mock_run:
        out = mod.run_applescript("tell application \"Notes\" to ...")
    assert out == "Notes\nPersonal\n"
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ["osascript", "-"]


# --- CLI dispatch ---


def test_cli_create_invokes_run_applescript_with_built_script(monkeypatch):
    mod = load_module()
    captured = []

    def fake_run(script):
        captured.append(script)
        return "ok"

    monkeypatch.setattr(mod, "run_applescript", fake_run)
    rc = mod.main(["create", "--title", "T", "--body", "hello\nworld", "--folder", "Notes"])
    assert rc == 0
    assert captured, "run_applescript was not called"
    assert "make new note" in captured[0]
    assert "tell first folder whose name is \"Notes\"" in captured[0]
    # plain-text body was converted to Notes HTML
    assert "hello<br>world" in captured[0]


def test_cli_append_accepts_raw_body_html(monkeypatch):
    mod = load_module()
    captured = []
    monkeypatch.setattr(mod, "run_applescript", lambda s: captured.append(s) or "")
    mod.main(["append", "--title", "T", "--body-html", "<br>raw", "--folder", "F"])
    assert "set body of n to (body of n) &" in captured[0]
    assert "<br>raw" in captured[0]


def test_cli_create_folder_dispatches(monkeypatch):
    mod = load_module()
    captured = []
    monkeypatch.setattr(mod, "run_applescript", lambda s: captured.append(s) or "")
    mod.main(["create-folder", "--name", "Project X"])
    assert "make new folder with properties" in captured[0]
    assert "Project X" in captured[0]


def test_cli_move_dispatches_with_src_and_dest(monkeypatch):
    mod = load_module()
    captured = []
    monkeypatch.setattr(mod, "run_applescript", lambda s: captured.append(s) or "")
    mod.main(["move", "--title", "T", "--src", "Src", "--dest", "Dest"])
    assert "set destFolder to first folder whose name is \"Dest\"" in captured[0]
    assert "tell first folder whose name is \"Src\"" in captured[0]


def test_cli_body_and_body_html_are_mutually_exclusive():
    mod = load_module()
    with pytest.raises(SystemExit):
        mod.main(["create", "--title", "T", "--body", "a", "--body-html", "<b>b</b>", "--folder", "F"])
