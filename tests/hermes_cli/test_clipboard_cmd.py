"""``hermes clipboard image --json``: the reply shapes and exit codes clients script against.

The real clipboard is never read: ``save_clipboard_image`` is replaced by a fake, and the tool
lookup is pointed at an executable that exists (this interpreter) or one that cannot.
"""

import json
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
