from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.oneshot import read_oneshot_file


def _private_prompt(tmp_path: Path, content: bytes = b"classify this") -> Path:
    path = tmp_path / "prompt.txt"
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def test_parser_accepts_oneshot_file_and_rejects_two_prompt_sources(tmp_path: Path) -> None:
    parser, _subparsers, _chat = build_top_level_parser()
    prompt = _private_prompt(tmp_path)

    args = parser.parse_args(["--oneshot-file", str(prompt)])
    assert args.oneshot is None
    assert args.oneshot_file == str(prompt)

    with pytest.raises(SystemExit):
        parser.parse_args(["-z", "inline", "--oneshot-file", str(prompt)])


def test_read_oneshot_file_accepts_private_utf8_regular_file(tmp_path: Path) -> None:
    prompt = _private_prompt(tmp_path, "mensagem privada".encode())
    assert read_oneshot_file(str(prompt)) == "mensagem privada"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits required")
def test_read_oneshot_file_rejects_group_readable_file(tmp_path: Path) -> None:
    prompt = _private_prompt(tmp_path)
    prompt.chmod(0o640)
    with pytest.raises(ValueError, match="group or other"):
        read_oneshot_file(str(prompt))


def test_read_oneshot_file_rejects_symlink_and_non_utf8(tmp_path: Path) -> None:
    prompt = _private_prompt(tmp_path)
    link = tmp_path / "link.txt"
    link.symlink_to(prompt)
    if hasattr(os, "O_NOFOLLOW"):
        with pytest.raises(ValueError, match="cannot open"):
            read_oneshot_file(str(link))

    invalid = _private_prompt(tmp_path, b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        read_oneshot_file(str(invalid))


def test_read_oneshot_file_rejects_oversize(tmp_path: Path) -> None:
    prompt = _private_prompt(tmp_path, b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        read_oneshot_file(str(prompt))
