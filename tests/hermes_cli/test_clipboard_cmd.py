"""``hermes clipboard image --json``: the reply shapes and exit codes clients script against.

The real clipboard is never read: ``save_clipboard_image`` is replaced by a fake, or the real
extractor runs against a stub ``xclip`` on PATH; the tool lookup is pointed at an executable that
exists (this interpreter) or one that cannot.
"""

import json
import os
import sys

import pytest

from hermes_cli import clipboard
from hermes_cli.main import _build_cli_parser, _plugin_cli_discovery_needed

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def _run(capsys, *argv):
    parser, _ = _build_cli_parser()
    args = parser.parse_args(["clipboard", "image", *argv])
    code = args.func(args)
    out, err = capsys.readouterr()
    return code, out, err


def _fake_save(data):
    def save(dest):
        if data is None:
            return False
        dest.write_bytes(data)
        return True
    return save


def test_image_written_reports_path_type_and_size(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clipboard, "save_clipboard_image", _fake_save(_PNG))
    dest = tmp_path / "new" / "paste.png"

    code, out, _ = _run(capsys, "--out", str(dest), "--json")

    assert code == 0
    assert json.loads(out) == {"ok": True, "path": str(dest), "mime": "image/png", "bytes": len(_PNG)}
    assert dest.read_bytes() == _PNG


@pytest.mark.parametrize("data, mime", [
    (b"\xff\xd8\xff\xe0" + b"\x00" * 12, "image/jpeg"), (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp")],
    ids=["jpeg", "webp"])
def test_a_copied_image_file_reports_its_own_type(tmp_path, monkeypatch, capsys, data, mime):
    """A Windows/WSL file drop is copied as it is, so the type comes from the written bytes."""
    monkeypatch.setattr(clipboard, "save_clipboard_image", _fake_save(data))

    code, out, _ = _run(capsys, "--out", str(tmp_path / "paste.png"), "--json")

    assert (code, json.loads(out)["mime"]) == (0, mime)


def test_no_image_is_a_stable_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clipboard, "save_clipboard_image", _fake_save(None))
    monkeypatch.setattr(clipboard, "clipboard_image_tools", lambda: (sys.executable,))

    code, out, _ = _run(capsys, "--out", str(tmp_path / "paste.png"), "--json")

    assert code == 1
    assert json.loads(out) == {"ok": False, "reason": "no_image", "message": "the clipboard holds no image"}


def test_missing_tool_names_what_to_install(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clipboard, "save_clipboard_image", _fake_save(None))
    monkeypatch.setattr(clipboard, "clipboard_image_tools", lambda: ("hermes-no-such-clipboard-tool",))

    code, out, _ = _run(capsys, "--out", str(tmp_path / "paste.png"), "--json")

    reply = json.loads(out)
    assert code == 3
    assert (reply["ok"], reply["reason"], reply["tools"]) == (False, "no_tool", ["hermes-no-such-clipboard-tool"])


def test_without_json_prints_the_path_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clipboard, "save_clipboard_image", _fake_save(_PNG))
    dest = tmp_path / "paste.png"

    assert _run(capsys, "--out", str(dest)) == (0, f"{dest}\n", "")


def test_runs_without_plugin_discovery(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hermes", "clipboard", "image", "--out", "paste.png", "--json"])
    assert _plugin_cli_discovery_needed() is False


@pytest.mark.platforms("linux")
def test_linux_tools_follow_the_extractor_fallback_order(monkeypatch):
    monkeypatch.setattr(clipboard, "_is_wsl", lambda: False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert clipboard.clipboard_image_tools() == ("wl-paste", "xclip")
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert clipboard.clipboard_image_tools() == ("xclip",)


def _stub_xclip(tmp_path, monkeypatch, *, output_ok):
    """A stub ``xclip`` that advertises image/png; ``-o`` of the image succeeds or fails."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    payload = tmp_path / "payload.png"
    payload.write_bytes(_PNG)
    image = f'cat "{payload}"' if output_ok else "exit 1"
    (bin_dir / "xclip").write_text(
        f'#!/bin/sh\ncase "$*" in *TARGETS*) echo image/png;; *) {image};; esac\n')
    (bin_dir / "xclip").chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(bin_dir), os.environ.get("PATH", ""))))
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(clipboard, "_is_wsl", lambda: False)


def _existing(tmp_path, mode=0o644):
    dest = tmp_path / "out" / "paste.png"
    dest.parent.mkdir()
    dest.write_bytes(b"keep me")
    dest.chmod(mode)
    return dest


@pytest.mark.platforms("linux")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
def test_read_only_destination_is_write_failed_and_kept(tmp_path, monkeypatch, capsys):
    _stub_xclip(tmp_path, monkeypatch, output_ok=True)
    dest = _existing(tmp_path, 0o444)

    code, out, _ = _run(capsys, "--out", str(dest), "--json")

    assert (code, json.loads(out)["reason"]) == (4, "write_failed")
    assert dest.read_bytes() == b"keep me"
    assert os.listdir(dest.parent) == ["paste.png"]


@pytest.mark.platforms("linux")
def test_failed_extraction_keeps_the_existing_destination(tmp_path, monkeypatch, capsys):
    _stub_xclip(tmp_path, monkeypatch, output_ok=False)
    dest = _existing(tmp_path)

    code, out, _ = _run(capsys, "--out", str(dest), "--json")

    assert (code, json.loads(out)["reason"]) == (1, "no_image")
    assert dest.read_bytes() == b"keep me"
    assert os.listdir(dest.parent) == ["paste.png"]


@pytest.mark.platforms("linux")
def test_real_extractor_replaces_a_writable_destination(tmp_path, monkeypatch, capsys):
    _stub_xclip(tmp_path, monkeypatch, output_ok=True)
    dest = _existing(tmp_path)

    code, out, _ = _run(capsys, "--out", str(dest), "--json")

    assert code == 0
    assert json.loads(out) == {"ok": True, "path": str(dest), "mime": "image/png", "bytes": len(_PNG)}
    assert dest.read_bytes() == _PNG
    assert os.listdir(dest.parent) == ["paste.png"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", [0o600, 0o640], ids=["mkstemp-0600", "shared-0640"])
def test_replacing_a_destination_keeps_its_permissions(tmp_path, monkeypatch, capsys, mode):
    monkeypatch.setattr(clipboard, "save_clipboard_image", _fake_save(_PNG))
    dest = _existing(tmp_path, mode)

    code, _, _ = _run(capsys, "--out", str(dest), "--json")

    assert (code, dest.read_bytes()) == (0, _PNG)
    assert dest.stat().st_mode & 0o777 == mode
