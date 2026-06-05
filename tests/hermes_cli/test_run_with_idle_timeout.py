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
import time

from hermes_cli.main import _run_with_idle_timeout


def test_streams_output_and_returns_zero_on_success(tmp_path):
    script = tmp_path / "ok.py"
    script.write_text("print('line one'); print('line two')\n")
    result = _run_with_idle_timeout(
        [_sys.executable, str(script)], cwd=tmp_path, idle_timeout_seconds=10
    )
    assert result.returncode == 0
    assert "line one" in result.stdout
    assert "line two" in result.stdout


def test_propagates_nonzero_exit(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text("import sys; print('boom', file=sys.stderr); sys.exit(7)\n")
    result = _run_with_idle_timeout(
        [_sys.executable, str(script)], cwd=tmp_path, idle_timeout_seconds=10
    )
    assert result.returncode == 7
    # stderr is merged into stdout in the helper.
    assert "boom" in result.stdout


def test_kills_process_on_idle_timeout(tmp_path):
    # Sleeps without printing — exactly the failure mode users see when
    # `npm run build` stalls. Idle timeout must terminate it.
    script = tmp_path / "stall.py"
    script.write_text("import time; time.sleep(30)\n")

    start = time.monotonic()
    result = _run_with_idle_timeout(
        [_sys.executable, str(script)],
        cwd=tmp_path,
        idle_timeout_seconds=1,
    )
    elapsed = time.monotonic() - start
    # Should have died well before the 30s sleep completes.
    assert elapsed < 15
    assert result.returncode != 0
    assert "produced no output" in result.stdout


def test_returns_127_when_binary_missing(tmp_path):
    result = _run_with_idle_timeout(
        ["/nonexistent/binary/does/not/exist"],
        cwd=tmp_path,
        idle_timeout_seconds=5,
    )
    assert result.returncode == 127
