"""One-shot console silence must not suppress forensic file logs (#103056)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("response", ["", "final answer"], ids=["failure", "success"])
def test_oneshot_preserves_file_logs_without_console_noise(tmp_path, response):
    program = textwrap.dedent(
        """
        import logging
        import sys

        import hermes_cli.oneshot as oneshot
        from hermes_logging import flush_log_queue, setup_logging, setup_verbose_logging

        log_dir = setup_logging()
        setup_verbose_logging()
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler(sys.stdout))
        root.addHandler(logging.FileHandler(log_dir / "direct.log", encoding="utf-8"))
        response = sys.argv[1]

        def run_agent(*args, **kwargs):
            logger = logging.getLogger("run_agent")
            logger.info("one-shot diagnostic")
            logger.error("one-shot failure detail")
            logger.critical("one-shot critical detail")
            print("incidental stdout")
            print("incidental stderr", file=sys.stderr)
            return response, {"failed": not response}

        oneshot._run_agent = run_agent
        rc = oneshot.run_oneshot("hello")
        flush_log_queue()
        raise SystemExit(rc)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, response],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == (0 if response else 2), result.stderr
    assert result.stdout == (response + "\n" if response else "")
    assert result.stderr == ""
    logs = tmp_path / "logs"
    agent_log = (logs / "agent.log").read_text(encoding="utf-8")
    error_log = (logs / "errors.log").read_text(encoding="utf-8")
    direct_log = (logs / "direct.log").read_text(encoding="utf-8")
    for content in (agent_log, direct_log):
        assert "one-shot diagnostic" in content
    for content in (agent_log, error_log, direct_log):
        assert "one-shot failure detail" in content
        assert "one-shot critical detail" in content
    assert "one-shot diagnostic" not in error_log
