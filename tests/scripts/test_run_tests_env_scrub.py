"""Regression: scripts/run_tests.sh must scrub HERMES_CRON_SESSION.

When the test runner is invoked from a Hermes cron job, HERMES_CRON_SESSION
leaks into pytest and changes approval behavior. The runner already scrubs
many HERMES_* behavioral vars; HERMES_CRON_SESSION must be among them.

See: NousResearch/hermes-agent#22400
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TESTS = REPO_ROOT / "scripts" / "run_tests.sh"


def _scrub_block() -> str:
    text = RUN_TESTS.read_text(encoding="utf-8")
    match = re.search(
        r"unset\s+HERMES_[\s\S]+?(?:2>/dev/null\s*\|\|\s*true)",
        text,
    )
    assert match, "could not locate HERMES_* unset block in run_tests.sh"
    return match.group(0)


def test_run_tests_unsets_hermes_cron_session():
    block = _scrub_block()
    assert "HERMES_CRON_SESSION" in block, (
        "scripts/run_tests.sh does not unset HERMES_CRON_SESSION; cron approval "
        "mode will leak into pytest. See issue #22400."
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_run_tests_scrub_block_actually_unsets_var(tmp_path: Path):
    block = _scrub_block()
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -u\n"
        + block
        + "\n"
        + 'echo "after=${HERMES_CRON_SESSION:-<unset>}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)

    env = {**os.environ, "HERMES_CRON_SESSION": "1"}
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "after=<unset>" in result.stdout, result.stdout
