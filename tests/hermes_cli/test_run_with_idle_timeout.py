"""Coverage for _run_with_idle_timeout — the streaming subprocess helper.

Kept in a dedicated test file because the tests spawn real ``subprocess.Popen``
instances; pytest-isolate runs each test file in its own worker process, so
isolating these here prevents real-Popen state from racing with the
``subprocess.run`` / ``_run_with_idle_timeout`` patches used by
``test_web_ui_build.py``.

Added for issue #33788: ``hermes update`` got stuck at "webui-build" because
``npm run build`` ran with ``capture_output=True`` and no timeout. The helper
fixes both halves — streams output AND idle-kills the process.
"""

import sys as _sys

from hermes_cli.main import _run_with_idle_timeout


def test_streams_output_and_returns_zero_on_success(tmp_path, capsys):
    script = tmp_path / "ok.py"
    script.write_text("print('line one'); print('line two')\n")
    result = _run_with_idle_timeout(
        [_sys.executable, str(script)], cwd=tmp_path, idle_timeout_seconds=10
    )
    assert result.returncode == 0
    assert "line one" in result.stdout
    assert "line two" in result.stdout
    # Default stream=True prints lines to the console.
    out = capsys.readouterr().out
    assert "line one" in out
    assert "line two" in out


def test_returns_127_when_binary_missing(tmp_path):
    result = _run_with_idle_timeout(
        ["/nonexistent/binary/does/not/exist"],
        cwd=tmp_path,
        idle_timeout_seconds=5,
    )
    assert result.returncode == 127


def test_stream_false_captures_without_printing(tmp_path, capsys):
    script = tmp_path / "quiet.py"
    script.write_text("print('hidden line')\n")
    result = _run_with_idle_timeout(
        [_sys.executable, str(script)],
        cwd=tmp_path,
        idle_timeout_seconds=10,
        stream=False,
    )
    assert result.returncode == 0
    assert "hidden line" in result.stdout
    out = capsys.readouterr().out
    assert "hidden line" not in out


def test_on_line_callback_fires_even_when_not_streaming(tmp_path):
    script = tmp_path / "cb.py"
    script.write_text("print('via callback')\n")
    seen: list[str] = []
    result = _run_with_idle_timeout(
        [_sys.executable, str(script)],
        cwd=tmp_path,
        idle_timeout_seconds=10,
        stream=False,
        on_line=seen.append,
    )
    assert result.returncode == 0
    assert any("via callback" in line for line in seen)
