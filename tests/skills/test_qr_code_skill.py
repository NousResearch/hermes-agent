"""Tests for skills/media/qr-code/scripts/qr_code.py

No network, no real QR library required — all backends are monkeypatched so the
suite runs under the hermetic CI env (no qrencode/zbarimg/qrcode/pyzbar installed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills" / "media" / "qr-code" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import qr_code  # noqa: E402  (path injected above)


# --- helpers ---------------------------------------------------------------

def _run(argv: list[str]) -> int:
    return qr_code.main(argv)


def test_description_under_60_chars():
    """CONTRIBUTING.md hardline rule: description ≤ 60 chars, one sentence, period."""
    import re

    skill_md = (
        Path(__file__).resolve().parents[2]
        / "skills" / "media" / "qr-code" / "SKILL.md"
    )
    text = skill_md.read_text(encoding="utf-8")
    m = re.search(r'^description: (.*)$', text, re.MULTILINE)
    assert m, "description frontmatter line not found"
    desc = m.group(1).strip().strip('"').strip("'")
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"


def test_author_is_human_not_hermes():
    """CONTRIBUTING.md hardline rule: author credits the human, not 'Hermes Agent'."""
    import re

    skill_md = (
        Path(__file__).resolve().parents[2]
        / "skills" / "media" / "qr-code" / "SKILL.md"
    )
    text = skill_md.read_text(encoding="utf-8")
    m = re.search(r'^author: (.*)$', text, re.MULTILINE)
    assert m, "author frontmatter line not found"
    author = m.group(1).strip()
    assert author != "Hermes Agent", "author must be the human contributor, not 'Hermes Agent'"


# --- doctor ----------------------------------------------------------------

class TestDoctor:
    def test_doctor_always_exits_zero(self, capsys):
        rc = _run(["doctor"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Encoders:" in out
        assert "Decoders:" in out

    def test_doctor_reports_missing_backends(self, capsys, monkeypatch):
        monkeypatch.setattr(qr_code, "_have_qrencode", lambda: False)
        monkeypatch.setattr(qr_code, "_have_zbarimg", lambda: False)
        monkeypatch.setattr(qr_code, "_have_pyqrcode", lambda: False)
        monkeypatch.setattr(qr_code, "_have_pyzbar", lambda: False)
        _run(["doctor"])
        out = capsys.readouterr().out
        assert "NO ENCODER" in out
        assert "NO DECODER" in out


# --- argument validation ---------------------------------------------------

class TestArgValidation:
    def test_no_backend_for_encode_returns_2(self, monkeypatch, capsys):
        monkeypatch.setattr(qr_code, "_have_qrencode", lambda: False)
        monkeypatch.setattr(qr_code, "_have_pyqrcode", lambda: False)
        rc = _run(["encode", "hello"])
        assert rc == 2
        assert "no encoder found" in capsys.readouterr().err.lower()

    def test_empty_payload_returns_1(self, monkeypatch, capsys):
        rc = _run(["encode", ""])
        assert rc == 1
        assert "empty" in capsys.readouterr().err.lower()

    def test_decode_missing_file_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_zbarimg", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyzbar", lambda: False)
        rc = _run(["decode", str(tmp_path / "nope.png")])
        assert rc == 1
        assert "not found" in capsys.readouterr().err.lower()


# --- encoding via mocked CLI backend ---------------------------------------

class TestEncode:
    def test_encode_writes_png_via_qrencode(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_qrencode", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyqrcode", lambda: False)
        captured = {}

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = cmd

            class C:
                returncode = 0
                stdout = ""
                stderr = ""

            return C()

        monkeypatch.setattr(qr_code.subprocess, "run", fake_run)
        out = tmp_path / "x.png"
        rc = _run(["encode", "https://example.com", "-o", str(out)])
        assert rc == 0
        assert "-l" in captured["cmd"] and "M" in captured["cmd"]
        assert "https://example.com" in captured["cmd"]
        assert str(out) in captured["cmd"]
        assert f"Wrote {out}" in capsys.readouterr().out

    def test_encode_terminal_mode_uses_stdout(self, monkeypatch, capsys):
        monkeypatch.setattr(qr_code, "_have_qrencode", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyqrcode", lambda: False)

        def fake_run(cmd, *args, **kwargs):
            class C:
                returncode = 0
                stdout = "QQQQ\nQQQQ\n"
                stderr = ""

            return C()

        monkeypatch.setattr(qr_code.subprocess, "run", fake_run)
        rc = _run(["encode", "hi", "--terminal"])
        assert rc == 0
        assert "QQQQ" in capsys.readouterr().out

    def test_encode_cli_failure_returns_3(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_qrencode", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyqrcode", lambda: False)

        def fake_run(cmd, *args, **kwargs):
            class C:
                returncode = 1
                stdout = ""
                stderr = "boom"

            return C()

        monkeypatch.setattr(qr_code.subprocess, "run", fake_run)
        rc = _run(["encode", "hi", "-o", str(tmp_path / "x.png")])
        assert rc == 3


# --- special payloads ------------------------------------------------------

class TestPayloadBuilders:
    def test_wifi_payload_format(self, monkeypatch):
        captured = {}

        def fake_encode(payload, out, terminal, ec):
            captured["payload"] = payload
            return 0

        monkeypatch.setattr(qr_code, "encode", fake_encode)
        rc = _run(["wifi", "--ssid", "Cafe", "--password", "p;s"])
        assert rc == 0
        # password contains ; and must be escaped per WIFI: spec
        assert "S:Cafe;" in captured["payload"]
        assert "P:p\\;s;" in captured["payload"]
        assert captured["payload"].startswith("WIFI:T:WPA;")

    def test_wifi_requires_ssid(self, monkeypatch, capsys):
        rc = _run(["wifi", "--ssid", ""])
        assert rc == 1

    def test_wifi_hidden_flag(self, monkeypatch):
        captured = {}

        def fake_encode(payload, out, terminal, ec):
            captured["payload"] = payload
            return 0

        monkeypatch.setattr(qr_code, "encode", fake_encode)
        _run(["wifi", "--ssid", "H", "--password", "x", "--hidden"])
        assert "H:true" in captured["payload"]

    def test_vcard_payload_format(self, monkeypatch):
        captured = {}

        def fake_encode(payload, out, terminal, ec):
            captured["payload"] = payload
            return 0

        monkeypatch.setattr(qr_code, "encode", fake_encode)
        rc = _run(["vcard", "--name", "Ada", "--phone", "+1", "--email", "a@b.c"])
        assert rc == 0
        payload = captured["payload"]
        assert "BEGIN:VCARD" in payload
        assert "VERSION:2.1" in payload
        assert "N:Ada;;;;" in payload
        assert "FN:Ada" in payload
        assert "TEL;CELL:+1" in payload
        assert "EMAIL:a@b.c" in payload
        # RFC 2426: CRLF-delimited records with a trailing CRLF
        assert "\r\n" in payload
        assert payload.endswith("END:VCARD\r\n")

    def test_vcard_escapes_special_characters(self, monkeypatch):
        captured = {}

        def fake_encode(payload, out, terminal, ec):
            captured["payload"] = payload
            return 0

        monkeypatch.setattr(qr_code, "encode", fake_encode)
        rc = _run(["vcard", "--name", "Doe, John; Jr."])
        assert rc == 0
        payload = captured["payload"]
        # Commas and semicolons in the name are escaped per RFC 2426
        assert r"N:Doe\, John\; Jr.;;;;" in payload
        assert r"FN:Doe\, John\; Jr." in payload
        # No raw, unescaped special characters leaked into N/FN
        assert "N:Doe, " not in payload
        assert payload.endswith("END:VCARD\r\n")

    def test_vcard_requires_name(self, monkeypatch, capsys):
        rc = _run(["vcard", "--name", ""])
        assert rc == 1


# --- batch -----------------------------------------------------------------

class TestBatch:
    def test_batch_creates_one_png_per_line(self, monkeypatch, tmp_path):
        calls = []

        def fake_encode(payload, out, terminal, ec):
            calls.append((payload, out))
            # touch the file so it looks written
            Path(out).touch()
            return 0

        monkeypatch.setattr(qr_code, "encode", fake_encode)
        infile = tmp_path / "vals.txt"
        infile.write_text("first\nsecond\n\nthird\n", encoding="utf-8")
        outdir = tmp_path / "out"
        rc = _run(["batch", str(infile), "--outdir", str(outdir)])
        assert rc == 0
        # blank line skipped
        assert len(calls) == 3
        payloads = [c[0] for c in calls]
        assert payloads == ["first", "second", "third"]
        # unique filenames (no collision)
        outs = {str(c[1]) for c in calls}
        assert len(outs) == 3
        assert all(str(p).startswith(str(outdir)) for p in outs)

    def test_batch_missing_input_returns_1(self, monkeypatch, tmp_path, capsys):
        rc = _run(["batch", str(tmp_path / "missing.txt")])
        assert rc == 1

    def test_batch_empty_input_returns_1(self, monkeypatch, tmp_path, capsys):
        infile = tmp_path / "empty.txt"
        infile.write_text("\n\n", encoding="utf-8")
        rc = _run(["batch", str(infile)])
        assert rc == 1


# --- decoding --------------------------------------------------------------

class TestDecode:
    def test_decode_no_backend_returns_2(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_zbarimg", lambda: False)
        monkeypatch.setattr(qr_code, "_have_pyzbar", lambda: False)
        # need a real file so the not-found check passes
        img = tmp_path / "x.png"
        img.touch()
        rc = _run(["decode", str(img)])
        assert rc == 2
        assert "no decoder found" in capsys.readouterr().err.lower()

    def test_decode_via_zbarimg(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_zbarimg", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyzbar", lambda: False)

        def fake_run(cmd, *args, **kwargs):
            class C:
                returncode = 0
                stdout = "https://example.com\n"  # zbarimg --raw appends newline
                stderr = ""

            return C()

        monkeypatch.setattr(qr_code.subprocess, "run", fake_run)
        img = tmp_path / "x.png"
        img.touch()
        rc = _run(["decode", str(img)])
        assert rc == 0
        assert "https://example.com" in capsys.readouterr().out

    def test_decode_raw_strips_trailing_newline(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_zbarimg", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyzbar", lambda: False)

        def fake_run(cmd, *args, **kwargs):
            class C:
                returncode = 0
                stdout = "payload\n"
                stderr = ""

            return C()

        monkeypatch.setattr(qr_code.subprocess, "run", fake_run)
        img = tmp_path / "x.png"
        img.touch()
        _run(["decode", str(img), "--raw"])
        assert capsys.readouterr().out == "payload\n"

    def test_decode_no_qr_found_returns_3(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(qr_code, "_have_zbarimg", lambda: True)
        monkeypatch.setattr(qr_code, "_have_pyzbar", lambda: False)

        def fake_run(cmd, *args, **kwargs):
            class C:
                returncode = 1  # zbarimg returns 1 when no code found
                stdout = ""
                stderr = ""

            return C()

        monkeypatch.setattr(qr_code.subprocess, "run", fake_run)
        img = tmp_path / "x.png"
        img.touch()
        rc = _run(["decode", str(img)])
        assert rc == 3
        assert "no qr code found" in capsys.readouterr().err.lower()
