from __future__ import annotations

import base64
import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


def test_non_windows_reports_unavailable():
    from tools.computer_use.ocr import extract_local_ocr

    with patch("tools.computer_use.ocr.sys.platform", "linux"):
        result = extract_local_ocr(base64.b64encode(b"x").decode())

    assert result["available"] is False
    assert "Windows" in result["reason"]


def test_invalid_language_is_rejected_before_process_launch():
    from tools.computer_use.ocr import extract_local_ocr

    with patch("tools.computer_use.ocr.sys.platform", "win32"), \
         patch("tools.computer_use.ocr.subprocess.run") as run:
        result = extract_local_ocr(base64.b64encode(b"x").decode(), language="en-US;whoami")

    assert result["available"] is False
    run.assert_not_called()


def test_windows_ocr_result_is_bounded_and_temp_file_is_removed(tmp_path):
    from tools.computer_use.ocr import extract_local_ocr

    output = {
        "language": "en-US",
        "width": 640,
        "height": 480,
        "text": "Total 123.45",
        "lines": ["Total 123.45"],
        "words": [{"text": "123.45", "bounds": [10, 20, 40, 12]}],
    }
    captured_path = None

    def fake_run(command, **kwargs):
        nonlocal captured_path
        captured_path = Path(command[command.index("-Path") + 1])
        assert captured_path.exists()
        return CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    with patch("tools.computer_use.ocr.sys.platform", "win32"), \
         patch("tools.computer_use.ocr.shutil.which", return_value="powershell.exe"), \
         patch("tools.computer_use.ocr.subprocess.run", side_effect=fake_run):
        result = extract_local_ocr(base64.b64encode(b"image").decode())

    assert result["available"] is True
    assert result["text"] == "Total 123.45"
    assert result["words"][0]["bounds"] == [10, 20, 40, 12]
    assert captured_path is not None and not captured_path.exists()


def test_windows_powershell_control_characters_do_not_break_json():
    from tools.computer_use.ocr import extract_local_ocr

    raw_json = '{"text":"Total\u000b123","lines":[],"words":[]}'
    completed = CompletedProcess([], 0, stdout=raw_json, stderr="")
    with patch("tools.computer_use.ocr.sys.platform", "win32"), \
         patch("tools.computer_use.ocr.shutil.which", return_value="powershell.exe"), \
         patch("tools.computer_use.ocr.subprocess.run", return_value=completed):
        result = extract_local_ocr(base64.b64encode(b"image").decode())

    assert result["available"] is True
    assert result["text"] == "Total 123"
