"""Tests for _detect_file_drop — file path detection that prevents
dragged/pasted absolute paths from being mistaken for slash commands."""

import os
from pathlib import Path

import pytest

import cli
from cli import _detect_file_drop, _is_registered_slash_command


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_image(tmp_path):
    """Create a temporary .png file and return its path."""
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
    return img


@pytest.fixture()
def tmp_text(tmp_path):
    """Create a temporary .py file and return its path."""
    f = tmp_path / "main.py"
    f.write_text("print('hello')\n")
    return f


@pytest.fixture()
def tmp_image_with_spaces(tmp_path):
    """Create a file whose name contains spaces (like macOS screenshots)."""
    img = tmp_path / "Screenshot 2026-04-01 at 7.25.32 PM.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    return img


# ---------------------------------------------------------------------------
# Tests: returns None for non-file inputs
# ---------------------------------------------------------------------------

class TestNonFileInputs:
    def test_regular_slash_command(self):
        assert _detect_file_drop("/help") is None



    def test_empty_string(self):
        assert _detect_file_drop("") is None



    def test_nonexistent_path(self):
        assert _detect_file_drop("/nonexistent/path/to/file.png") is None


    def test_long_slash_command_does_not_raise(self):
        """Regression: long pasted slash commands like `/goal <long prose>`
        used to raise OSError(ENAMETOOLONG, errno 63 macOS / 36 Linux)
        from `Path.exists()` inside `_resolve_attachment_path`, which
        propagated up to `process_loop`'s catch-all and silently lost
        the user's input. The fix wraps the stat call in a try/except
        OSError and returns None, letting the slash-command dispatch
        path handle the input downstream.

        Reproducer: paste a `/goal` followed by ~430 chars of prose.
        Without the fix this triggers ENAMETOOLONG; with the fix it
        cleanly returns None (file-drop = no), so `_looks_like_slash_command`
        gets a chance to dispatch it.
        """
        # 430-char `/goal` payload — well above NAME_MAX (255 bytes) on
        # all common filesystems.
        long_goal = (
            "/goal " + ("Drive the board: triage triage-status items, "
                        "unblock spillover tasks where work is shipped, "
                        "advance P1 items by decomposing where needed. ") * 4
        )
        assert len(long_goal) > 255  # confirms it would have triggered ENAMETOOLONG
        assert _detect_file_drop(long_goal) is None



# ---------------------------------------------------------------------------
# Tests: image file detection
# ---------------------------------------------------------------------------

class TestImageFileDrop:
    def test_simple_image_path(self, tmp_image):
        result = _detect_file_drop(str(tmp_image))
        assert result is not None
        assert result["path"] == tmp_image
        assert result["is_image"] is True
        assert result["remainder"] == ""


    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".webp",
                                      ".bmp", ".tiff", ".tif", ".svg", ".ico"])
    def test_all_image_extensions(self, tmp_path, ext):
        img = tmp_path / f"test{ext}"
        img.write_bytes(b"fake")
        result = _detect_file_drop(str(img))
        assert result is not None
        assert result["is_image"] is True



# ---------------------------------------------------------------------------
# Tests: non-image file detection
# ---------------------------------------------------------------------------

class TestNonImageFileDrop:
    def test_python_file(self, tmp_text):
        result = _detect_file_drop(str(tmp_text))
        assert result is not None
        assert result["path"] == tmp_text
        assert result["is_image"] is False
        assert result["remainder"] == ""



# ---------------------------------------------------------------------------
# Tests: backslash-escaped spaces (macOS drag-and-drop)
# ---------------------------------------------------------------------------

class TestEscapedSpaces:
    def test_escaped_spaces_in_path(self, tmp_image_with_spaces):
        r"""macOS drags produce paths like /path/to/my\ file.png"""
        escaped = str(tmp_image_with_spaces).replace(' ', '\\ ')
        result = _detect_file_drop(escaped)
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["is_image"] is True


    def test_unquoted_spaces_in_path(self, tmp_image_with_spaces):
        result = _detect_file_drop(str(tmp_image_with_spaces))
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["is_image"] is True
        assert result["remainder"] == ""


    def test_mixed_escaped_and_literal_spaces_in_path(self, tmp_path):
        img = tmp_path / "Screenshot 2026-04-21 at 1.04.43 PM.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        mixed = str(img).replace("Screenshot ", "Screenshot\\ ").replace("2026-04-21 ", "2026-04-21\\ ").replace("at ", "at\\ ")
        result = _detect_file_drop(mixed)
        assert result is not None
        assert result["path"] == img
        assert result["is_image"] is True
        assert result["remainder"] == ""


    def test_tilde_prefixed_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        img = home / "storage" / "shared" / "Pictures" / "cat.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setenv("HOME", str(home))
        # ntpath.expanduser ignores HOME (Python 3.8+) — it wants USERPROFILE.
        monkeypatch.setenv("USERPROFILE", str(home))

        result = _detect_file_drop("~/storage/shared/Pictures/cat.png what is this?")

        assert result is not None
        assert result["path"] == img
        assert result["is_image"] is True
        assert result["remainder"] == "what is this?"


    @pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter URI contract")
    def test_windows_drive_letter_file_uri_drops_url_leading_slash(self, tmp_path):
        image = tmp_path / "drive-uri.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        uri = image.as_uri()
        assert uri.startswith("file:///") and ":/" in uri

        result = _detect_file_drop(uri)

        assert result is not None
        assert result["path"] == image


# ---------------------------------------------------------------------------
# Tests: registered slash command vs. existing file collision
# ---------------------------------------------------------------------------

class TestRegisteredCommandCollision:
    """A registered slash command must never be swallowed as a file drop,
    even when a real file with that name exists. WSL2 ships a binary at
    /init, so `/init <notes>` used to attach the WSL init daemon instead
    of dispatching the /init command added in v0.20 (see #79765)."""

    @pytest.fixture()
    def wsl_init_collision(self, monkeypatch):
        """Simulate a WSL2 host where `/init` exists as a real file."""
        real_resolve = cli._resolve_attachment_path
        fake_init = Path("/init")

        def fake_resolve(raw_path):
            token = str(raw_path or "").strip()
            if token in ("/init", '"/init"', "'/init'"):
                return fake_init
            return real_resolve(token)

        monkeypatch.setattr("cli._resolve_attachment_path", fake_resolve)
        return fake_init

    def test_bare_init_dispatches_command(self, wsl_init_collision):
        assert _detect_file_drop("/init") is None

    def test_init_with_notes_dispatches_command(self, wsl_init_collision):
        assert _detect_file_drop("/init focus on the test setup") is None

    def test_init_with_trailing_text_dispatches_command(self, wsl_init_collision):
        assert _detect_file_drop("/init 核实agent.md内容") is None

    def test_quoted_init_explicitly_attaches(self, wsl_init_collision):
        result = _detect_file_drop('"/init"')
        assert result is not None
        assert result["path"] == wsl_init_collision
        assert result["remainder"] == ""

    def test_other_registered_command_collision(self, monkeypatch):
        """The guard is generic: any registered command wins over a file
        that shares its name (e.g. /version), not just /init."""
        fake_version = Path("/version")
        monkeypatch.setattr(
            "cli._resolve_attachment_path",
            lambda p: fake_version if str(p).strip() == "/version" else None,
        )
        assert _detect_file_drop("/version") is None

    def test_multi_component_path_named_init_still_attaches(self, tmp_path):
        """A real multi-slash path ending in `init` is unaffected — it is
        not a bare command token, so it still attaches."""
        f = tmp_path / "init"
        f.write_text("not the wsl binary\n")
        result = _detect_file_drop(f"{f} 看看")
        assert result is not None
        assert result["path"] == f
        assert result["remainder"] == "看看"


class TestIsRegisteredSlashCommand:
    def test_registered_command(self):
        assert _is_registered_slash_command("/init") is True
        assert _is_registered_slash_command("/help") is True

    def test_real_path_and_quoted_tokens(self):
        assert _is_registered_slash_command("/home/me/init") is False
        assert _is_registered_slash_command('"/init"') is False
        assert _is_registered_slash_command("~/init") is False
        assert _is_registered_slash_command("") is False

    def test_unregistered_single_segment(self):
        assert _is_registered_slash_command("/nonexistentcmd") is False


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_path_with_no_extension(self, tmp_path):
        f = tmp_path / "Makefile"
        f.write_text("all:\n\techo hi\n")
        result = _detect_file_drop(str(f))
        assert result is not None
        assert result["is_image"] is False

    def test_path_that_looks_like_command_but_is_file(self, tmp_path):
        """A file literally named 'help' inside a directory starting with /."""
        f = tmp_path / "help"
        f.write_text("not a command\n")
        result = _detect_file_drop(str(f))
        assert result is not None
        assert result["is_image"] is False

