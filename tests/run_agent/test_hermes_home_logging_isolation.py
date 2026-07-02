"""Regression coverage for pytest log isolation (#57118)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_aiagent_logging_uses_current_home_after_run_agent_import(tmp_path):
    old_home = tmp_path / "old-home"
    new_home = tmp_path / "new-home"
    old_home.mkdir()
    new_home.mkdir()
    marker = "pytest-hermes-home-isolation-marker"
    script = textwrap.dedent(
        """
        import logging
        import os
        import sys
        from pathlib import Path

        old_home = Path(sys.argv[1])
        new_home = Path(sys.argv[2])
        marker = sys.argv[3]

        os.environ["HERMES_HOME"] = str(old_home)
        import run_agent
        assert Path(run_agent._hermes_home) == old_home

        os.environ["HERMES_HOME"] = str(new_home)
        agent = run_agent.AIAgent(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        logging.getLogger("run_agent").info(marker)

        old_log = old_home / "logs" / "agent.log"
        new_log = new_home / "logs" / "agent.log"
        assert new_log.exists(), new_log
        assert marker in new_log.read_text(encoding="utf-8")
        assert not old_log.exists() or marker not in old_log.read_text(encoding="utf-8")
        assert agent is not None
        """
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "OPENROUTER_API_KEY": "test-key",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(old_home), str(new_home), marker],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout
