"""Tests for durable ephemeral MEDIA staging in chat responses."""

from __future__ import annotations

from pathlib import Path

from gateway.media_repair import (
    finalize_chat_media_paths,
    stage_ephemeral_chat_media_paths,
)


def test_stage_ephemeral_tmp_media_rewrites_to_hermes_cache(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import tempfile

    with tempfile.TemporaryDirectory(prefix="hermes-media-stage-") as td:
        src_file = Path(td) / "preview.png"
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

        response = f"Here is the mark.\nMEDIA:{src_file}\n"
        out = stage_ephemeral_chat_media_paths(response)

        assert "MEDIA:" in out
        assert str(src_file) not in out
        assert "chat-media" in out
        new_path = out.split("MEDIA:", 1)[1].strip().split()[0]
        assert Path(new_path).is_file()
        assert Path(new_path).read_bytes() == src_file.read_bytes()
        assert str(hermes_home.resolve()) in str(Path(new_path).resolve())


def test_stage_skips_non_ephemeral_project_paths(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # pytest's tmp_path lives under /tmp (ephemeral). Force a non-ephemeral
    # classification so we only assert the "leave project paths alone" branch.
    import gateway.media_repair as mr

    monkeypatch.setattr(mr, "_is_ephemeral_media_path", lambda path: False)

    project = tmp_path / "project"
    project.mkdir()
    img = project / "logo.png"
    img.write_bytes(b"PNGDATA" + b"1" * 32)

    response = f"MEDIA:{img}"
    out = stage_ephemeral_chat_media_paths(response)
    assert out == response


def test_stage_skips_code_fence_examples(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import tempfile

    with tempfile.TemporaryDirectory(prefix="hermes-media-stage-") as td:
        src_file = Path(td) / "real.png"
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"2" * 32)
        response = (
            "Use this form:\n"
            "```\n"
            "MEDIA:/tmp/example-only.png\n"
            "```\n"
            f"MEDIA:{src_file}\n"
        )
        out = stage_ephemeral_chat_media_paths(response)
        assert "MEDIA:/tmp/example-only.png" in out
        assert str(src_file) not in out
        assert "chat-media" in out


def test_finalize_combines_without_messages(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import tempfile

    with tempfile.TemporaryDirectory(prefix="hermes-media-stage-") as td:
        src_file = Path(td) / "x.png"
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"3" * 16)
        out = finalize_chat_media_paths(f"MEDIA:{src_file}")
        assert "chat-media" in out
