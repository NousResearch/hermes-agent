"""Project-checker repair loop for Hermes codegen.

Inspired by omp cleanse design (MIT) but implemented independently — no omp
source files vendored. Detects issues via VCS + manifests + linters, normalizes
diagnostics, and fixes root causes in parallel file-sticky workers.
"""

from agent.cleanse.detector import detect_issues
from agent.cleanse.loop import run_cleanse
from agent.cleanse.parser import Diagnostic, normalize_diagnostics

__all__ = [
    "detect_issues",
    "normalize_diagnostics",
    "run_cleanse",
    "Diagnostic",
]
