"""Tests for the file upload validation pipeline (file.attach JSON-RPC).

The pipeline is:
  1. _resolve_attachment_path(path)         # existing — path → Path
  2. _validate_upload(path, mime, size)     # NEW — whitelist + size
  3. _copy_to_sandbox(path, session_id)     # NEW — /tmp/<sid>/<hash>.<ext>
  4. _list_attached(session_id)             # NEW — list in sandbox

These tests cover step 2 first (pure, no FS side effects).
"""

import pytest

from cli import _validate_upload, _FILE_WHITELIST


class TestFileWhitelist:
    def test_includes_common_image_types(self):
        assert "image/png" in _FILE_WHITELIST
        assert "image/jpeg" in _FILE_WHITELIST
        assert "image/gif" in _FILE_WHITELIST
        assert "image/webp" in _FILE_WHITELIST

    def test_includes_pdf(self):
        assert "application/pdf" in _FILE_WHITELIST

    def test_includes_plain_text(self):
        assert "text/plain" in _FILE_WHITELIST

    def test_includes_json_yaml_toml(self):
        assert "application/json" in _FILE_WHITELIST
        assert "application/x-yaml" in _FILE_WHITELIST
        assert "application/toml" in _FILE_WHITELIST

    def test_excludes_executables(self):
        # Application/octet-stream is the generic binary catch-all — must be out.
        assert "application/octet-stream" not in _FILE_WHITELIST
        # Windows .exe / .dll MIME types are not in the whitelist.
        assert "application/x-msdownload" not in _FILE_WHITELIST
        assert "application/x-msdos-program" not in _FILE_WHITELIST

    def test_excludes_elf_and_mach_o(self):
        # Linux executables and macOS native binaries.
        assert "application/x-executable" not in _FILE_WHITELIST
        assert "application/x-mach-binary" not in _FILE_WHITELIST

    def test_is_a_frozenset(self):
        # Must be immutable so it can be defined at module scope safely.
        from cli import _FILE_WHITELIST as wl
        assert isinstance(wl, frozenset)


class TestValidateUpload:
    def test_accepts_png_under_limit(self, tmp_path):
        png = tmp_path / "test.png"
        # Minimal valid PNG (1x1 transparent).
        png.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfe\x02\xfe\xa3\x9b"
            b"\xe0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        # Should not raise.
        result = _validate_upload(png, mime="image/png", size_bytes=png.stat().st_size)
        assert result.mime_type == "image/png"
        assert result.allowed is True

    def test_rejects_executable_mime(self, tmp_path):
        f = tmp_path / "evil"
        f.write_bytes(b"MZ\x90\x00")  # PE/EXE header
        with pytest.raises(ValueError, match="MIME type not allowed"):
            _validate_upload(f, mime="application/x-msdownload", size_bytes=4)

    def test_rejects_file_over_size_limit(self, tmp_path, monkeypatch):
        # Patch the limit to something tiny for the test.
        import cli
        monkeypatch.setattr(cli, "MAX_UPLOAD_SIZE_BYTES", 100)
        f = tmp_path / "big.png"
        f.write_bytes(b"x" * 200)
        with pytest.raises(ValueError, match="exceeds size limit"):
            _validate_upload(f, mime="image/png", size_bytes=200)

import hashlib

from cli import _copy_to_sandbox, _list_attached, _cleanup_session_sandbox


class TestSandboxCopy:
    def test_copies_file_to_sandbox_dir(self, tmp_path, monkeypatch):
        # Redirect the sandbox root to tmp_path for the test.
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_bytes(b"hello world")

        attached = _copy_to_sandbox(src, session_id="sess-1")

        assert attached.stored_path.exists()
        assert attached.stored_path.read_bytes() == b"hello world"
        assert attached.session_id == "sess-1"
        assert attached.sha256 == hashlib.sha256(b"hello world").hexdigest()

    def test_uses_hash_in_filename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        src = tmp_path / "src.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nrest of png")

        attached = _copy_to_sandbox(src, session_id="sess-2")

        # Filename is <hash>.<ext>, not the original name.
        assert attached.stored_path.name.startswith(attached.sha256[:16])
        assert attached.stored_path.suffix == ".png"
        assert attached.original_path == src

    def test_chmod_600_on_copied_file(self, tmp_path, monkeypatch):
        import stat
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_bytes(b"x")

        attached = _copy_to_sandbox(src, session_id="sess-3")

        mode = attached.stored_path.stat().st_mode
        # Owner read+write only.
        assert mode & 0o777 == stat.S_IRUSR | stat.S_IWUSR

    def test_dedupes_within_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        src = tmp_path / "same.txt"
        src.write_bytes(b"same content")

        a1 = _copy_to_sandbox(src, session_id="sess-4")
        a2 = _copy_to_sandbox(src, session_id="sess-4")

        # Same hash → same stored path.
        assert a1.stored_path == a2.stored_path

    def test_different_sessions_different_sandboxes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        src = tmp_path / "shared.txt"
        src.write_bytes(b"x")

        a1 = _copy_to_sandbox(src, session_id="sess-A")
        a2 = _copy_to_sandbox(src, session_id="sess-B")

        assert a1.stored_path != a2.stored_path
        assert "sess-A" in str(a1.stored_path)
        assert "sess-B" in str(a2.stored_path)


class TestListAttached:
    def test_empty_session_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        result = _list_attached(session_id="empty")
        assert result == []

    def test_returns_all_files_in_session_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        from cli import _copy_to_sandbox
        src1 = tmp_path / "a.txt"; src1.write_bytes(b"alpha")
        src2 = tmp_path / "b.txt"; src2.write_bytes(b"beta")
        _copy_to_sandbox(src1, session_id="sess-X")
        _copy_to_sandbox(src2, session_id="sess-X")

        result = _list_attached(session_id="sess-X")

        assert len(result) == 2
        # In the sandbox, files are renamed to <hash>.<ext>. We verify the
        # count and the extensions, not the original names.
        suffixes = {a.stored_path.suffix for a in result}
        assert suffixes == {".txt"}
        # Each entry has a unique id and a valid sha256.
        ids = {a.id for a in result}
        assert len(ids) == 2
        for a in result:
            assert len(a.sha256) == 64  # full SHA-256 hex


class TestCleanupSessionSandbox:
    def test_removes_session_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        from cli import _copy_to_sandbox
        src = tmp_path / "gone.txt"; src.write_bytes(b"bye")
        attached = _copy_to_sandbox(src, session_id="doomed")
        assert attached.stored_path.exists()

        _cleanup_session_sandbox(session_id="doomed")

        assert not attached.stored_path.exists()
        # And the session directory itself is gone.
        assert not attached.stored_path.parent.exists()

    def test_missing_session_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SANDBOX_ROOT", str(tmp_path))
        # Should not raise even if the directory doesn't exist.
        _cleanup_session_sandbox(session_id="nonexistent")
