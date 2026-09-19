"""
DeepEval metrics for evaluating the Hermes Agent itself.

Uses a mix of CustomMetric (deterministic checks, zero tokens)
and GEval (LLM-judged quality, uses configured eval model).
"""

import subprocess
import re
from typing import Optional
from pathlib import Path

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

HERMES_HOME = Path(__file__).parent.parent.parent
HERMES_BIN = HERMES_HOME / "venv" / "Scripts" / "hermes.exe"

# The hermes CLI needs HERMES_HOME to point to the config directory
# (which is ~/AppData/Local/hermes/, not the source tree)
import os as _os
HERMES_CONFIG_HOME = _os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))


def _run_hermes(prompt: str, timeout: int = 60) -> tuple[str, int]:
    """Run a one-shot hermes query and return (stdout, exit_code)."""
    result = subprocess.run(
        [str(HERMES_BIN), "chat", "-q", prompt],
        capture_output=True, text=True, timeout=timeout,
        input="n\n",  # Answer 'no' to any interactive prompts
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": "",
            "HERMES_HOME": HERMES_CONFIG_HOME,
        },
        cwd=str(HERMES_HOME),
    )
    return result.stdout.strip(), result.returncode


class AgentCLIHealthMetric(BaseMetric):
    """Smoke test: can hermes --help run successfully?"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.error = None

    def measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        import os
        result = subprocess.run(
            [str(HERMES_BIN), "--help"],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "PYTHONPATH": "", "HERMES_HOME": HERMES_CONFIG_HOME},
            cwd=str(HERMES_HOME),
        )
        details = []

        # Check 1: exit code 0
        if result.returncode == 0:
            details.append("✅ Exit code 0")
            points = 1
        else:
            details.append(f"❌ Exit code {result.returncode}")
            points = 0

        # Check 2: output contains expected subcommands
        stdout = result.stdout + result.stderr
        for keyword in ["chat", "setup", "config", "desktop"]:
            if keyword in stdout:
                details.append(f"✅ '{keyword}' in help output")
            else:
                details.append(f"❌ '{keyword}' missing from help output")

        self.score = 1.0 if result.returncode == 0 else 0.0
        self.success = self.score >= self.threshold
        self.reason = "; ".join(details)
        return self.score

    async def a_measure(self, test_case, _show_indicator=True):
        return self.measure(test_case, _show_indicator)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Agent CLI Health"


class AgentBasicResponseMetric(BaseMetric):
    """Runs a simple query and scores the response on objective criteria."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.error = None

    def measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        prompt = test_case.input
        stdout, exit_code = _run_hermes(prompt, timeout=90)

        if exit_code != 0:
            self.score = 0.0
            self.success = False
            self.reason = f"❌ Hermes exited with code {exit_code}: {stdout[:200]}"
            self.error = stdout
            return self.score

        points = 0
        total = 3
        details = []

        # 1. Non-empty response
        if len(stdout) > 10:
            points += 1
            details.append("✅ Response is substantive (>10 chars)")
        else:
            details.append(f"❌ Response too short: '{stdout[:100]}'")

        # 2. No error markers
        error_patterns = ["error", "Error", "Traceback", "exception", "failed"]
        has_errors = any(p in stdout for p in error_patterns)
        if not has_errors:
            points += 1
            details.append("✅ No error markers in response")
        else:
            details.append("❌ Response contains error markers")

        # 3. Response mentions key terms from the prompt (basic relevance)
        prompt_words = set(re.findall(r"\w+", prompt.lower())) - {
            "what", "is", "the", "a", "an", "in", "of", "to", "and", "or",
            "can", "you", "how", "does", "do", "be", "it", "on", "at", "by",
            "for", "with", "this", "that", "i"
        }
        response_lower = stdout.lower()
        matched = [w for w in prompt_words if w in response_lower]
        if prompt_words and len(matched) / len(prompt_words) >= 0.3:
            points += 1
            details.append(f"✅ Response matches {len(matched)}/{len(prompt_words)} prompt terms")
        elif prompt_words:
            details.append(f"❌ Only {len(matched)}/{len(prompt_words)} prompt terms matched")
        else:
            points += 1  # No content words to match = pass
            details.append("✅ N/A — no content terms to match")

        self.score = points / total
        self.success = self.score >= self.threshold
        self.reason = f"{points}/{total} criteria met ({self.score:.0%}). " + "; ".join(details)
        return self.score

    async def a_measure(self, test_case, _show_indicator=True):
        return self.measure(test_case, _show_indicator)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Agent Basic Response"


AGENT_EVAL_METRICS = [
    AgentCLIHealthMetric(threshold=0.5),
    AgentBasicResponseMetric(threshold=0.6),
]
