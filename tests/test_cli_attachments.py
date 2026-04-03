"""Tests for CLI local file attachment staging and preprocessing."""
from pathlib import Path

import pytest

from cli import AttachedFile, HermesCLI


@pytest.fixture
def cli_obj():
    cli = HermesCLI.__new__(HermesCLI)
    cli._attached_images = []
    cli._attached_files = []
    cli._image_counter = 0
    return cli


class TestAttachmentParsing:
    def test_windows_path_without_spaces_preserves_backslashes(self, cli_obj, monkeypatch):
        monkeypatch.setattr("cli.sys.platform", "win32")

        parts = cli_obj._split_attachment_arg_text(r"C:\Users\me\report.csv")

        assert parts == [r"C:\Users\me\report.csv"]

    def test_quoted_windows_path_with_spaces_is_single_argument(self, cli_obj, monkeypatch):
        monkeypatch.setattr("cli.sys.platform", "win32")

        parts = cli_obj._split_attachment_arg_text(r'"C:\Users\me\Screen Shot.png"')

        assert parts == [r"C:\Users\me\Screen Shot.png"]

    def test_unquoted_windows_path_with_spaces_is_single_argument(self, cli_obj, monkeypatch):
        monkeypatch.setattr("cli.sys.platform", "win32")

        parts = cli_obj._split_attachment_arg_text(r"C:\Users\me\Screen Shot.png")

        assert parts == [r"C:\Users\me\Screen Shot.png"]


class TestAttachCommand:
    def test_attach_command_stages_csv_file_in_hermes_cache(self, cli_obj, tmp_path, monkeypatch):
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("region,revenue\nwest,10\n", encoding="utf-8")
        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        cli_obj._handle_attach_command(f"/attach {csv_path}")

        assert len(cli_obj._attached_files) == 1
        attached = cli_obj._attached_files[0]
        assert isinstance(attached, AttachedFile)
        assert attached.path != csv_path
        assert attached.path.exists()
        assert str(attached.path).startswith(str(hermes_home))
        assert attached.path.name.endswith("sales.csv")
        assert attached.display_name == "sales.csv"
        assert cli_obj._attached_images == []

    def test_pasted_screenshot_path_is_staged_into_hermes_cache(self, cli_obj, tmp_path, monkeypatch):
        screenshot_path = tmp_path / "Screen Shot 2026-03-26 at 10.00.00 AM.png"
        screenshot_path.write_bytes(b"fake-image-bytes")
        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        pasted_path = str(screenshot_path).replace(" ", "\\ ")
        attached = cli_obj._maybe_attach_pasted_paths(pasted_path)

        assert attached is True
        assert len(cli_obj._attached_images) == 1
        staged = cli_obj._attached_images[0]
        assert staged != screenshot_path
        assert staged.exists()
        assert str(staged).startswith(str(hermes_home))

    def test_nonlocal_backend_rejects_unexpanded_pdf_attachment(self, cli_obj, tmp_path, monkeypatch):
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\n")
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        cli_obj._handle_attach_command(f"/attach {pdf_path}")

        assert cli_obj._attached_files == []


class TestPreprocessFileAttachments:
    def test_small_csv_is_inlined_with_original_display_name(self, cli_obj, tmp_path):
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("region,revenue\nwest,10\neast,20\n", encoding="utf-8")
        staged = tmp_path / "doc_123_sales.csv"
        staged.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

        result = cli_obj._preprocess_file_attachments(
            "Summarize this CSV",
            [AttachedFile(path=staged, display_name="sales.csv")],
        )

        assert "[Content of sales.csv]:" in result
        assert "region,revenue" in result
        assert "west,10" in result
        assert "Summarize this CSV" in result
        assert "doc_123_sales.csv" not in result

    def test_large_csv_is_previewed_and_marked_truncated(self, cli_obj, tmp_path):
        csv_path = tmp_path / "large.csv"
        rows = ["region,revenue,notes"] + [f"row{i},{i},{'x' * 5000}" for i in range(200)]
        csv_path.write_text("\n".join(rows), encoding="utf-8")

        result = cli_obj._preprocess_file_attachments(
            "Analyze the attached CSV",
            [AttachedFile(path=csv_path, display_name="large.csv")],
        )

        assert "[Preview of large.csv]:" in result
        assert "The preview is truncated" in result
        assert "region,revenue" in result
        assert "row0,0" in result
        assert "Analyze the attached CSV" in result
        assert "row199,199" not in result
        assert len(result) < 20000
