"""Regression test for E741 (ambiguous variable name) in gateway/platforms/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestGatewayPlatformsE741:
    """Guard against E741 (ambiguous variable name) in gateway/platforms/."""

    def test_gateway_platforms_signal_py_has_zero_e741_violations(self) -> None:
        """gateway/platforms/signal.py must have zero E741 violations."""
        target = REPO_ROOT / "gateway" / "platforms" / "signal.py"
        assert target.exists(), f"Target file not found: {target}"

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select=E741",
             "--output-format=concise", str(target)],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, (
            f"gateway/platforms/signal.py has E741 violations:\n"
            f"{result.stdout}\n"
        )

    def test_gateway_platforms_feishu_comment_py_has_zero_e741_violations(self) -> None:
        """gateway/platforms/feishu_comment.py must have zero E741 violations."""
        target = REPO_ROOT / "gateway" / "platforms" / "feishu_comment.py"
        assert target.exists(), f"Target file not found: {target}"

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select=E741",
             "--output-format=concise", str(target)],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, (
            f"gateway/platforms/feishu_comment.py has E741 violations:\n"
            f"{result.stdout}\n"
        )
