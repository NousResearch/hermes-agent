"""Task battery for the read-tool eval.

Each task is a realistic dev request whose success depends on how well the
read tool handles one hostile-file shape from the Command Code writeup.
Graders are substring/regex checks against ground truth planted by
fixtures.py — forgiving about phrasing, strict about facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from fixtures import (
    AGENTS_BUILD_CMD,
    LEFT_PAD_VERSION,
    LOG_ERROR_REQ_ID,
    NOTES_BULLET_3,
    NOTES_NAME_CLEAN,
    REPORT_LINES,
)


@dataclass
class Task:
    task_id: str
    capability: str          # which read-tool capability this stresses
    prompt: str
    grade: Callable[[str], float]  # final_response -> 0.0..1.0
    timeout_s: int = 300
    notes: str = ""
    tags: list = field(default_factory=list)


def _has(*needles: str) -> Callable[[str], float]:
    def _g(text: str) -> float:
        low = text.lower()
        return 1.0 if all(n.lower() in low for n in needles) else 0.0
    return _g


def _regex(pattern: str) -> Callable[[str], float]:
    rx = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    return lambda text: 1.0 if rx.search(text) else 0.0


def _grade_lockfile(text: str) -> float:
    low = text.lower()
    version = LEFT_PAD_VERSION in low
    where = "package.json" in low
    return (0.5 * version) + (0.5 * where)


def _grade_backoff(text: str) -> float:
    low = text.lower()
    base = "250" in low
    shape = bool(re.search(r"exponential|2\s*\*\*|math\.pow|2\^|doubl", low))
    cap = bool(re.search(r"30000|30,000|30\s*s|3e4", low))
    return (0.4 * base) + (0.4 * shape) + (0.2 * cap)


def _grade_empty(text: str) -> float:
    low = text.lower()
    if re.search(r"\b(empty|no overrides|none|nothing|0 bytes|blank)\b", low):
        return 1.0
    return 0.0


def _grade_fifo(text: str) -> float:
    low = text.lower()
    if re.search(r"fifo|named pipe|not a regular file|special file|pipe\b|socket", low):
        return 1.0
    return 0.0


def _grade_binary(text: str) -> float:
    low = text.lower()
    if re.search(r"\bbinary\b|\bpng\b|image data|not (a )?text", low):
        return 1.0
    return 0.0


TASKS: list[Task] = [
    Task(
        task_id="lockfile_version",
        capability="ceilings/token-tarpit (huge lockfile adjacent to answer)",
        prompt=(
            "In this repo, what exact version of left-pad does the project "
            "pin, and which file declares the `test` script? Answer both "
            "precisely."
        ),
        grade=_grade_lockfile,
        notes="package-lock.json is 80K lines; the answer is in package.json.",
    ),
    Task(
        task_id="minified_backoff",
        capability="per-line clamp (single-line 600KB bundle in grep results)",
        prompt=(
            "Find the function in src/ that computes the retry delay and "
            "describe its backoff formula: base value, growth pattern, and "
            "any cap."
        ),
        grade=_grade_backoff,
        notes="src/app.min.js matches 'retryDelay' too and is one 600KB line.",
    ),
    Task(
        task_id="log_error_hunt",
        capability="pagination/resume offsets (150K-line log, answer at tail)",
        prompt=(
            "logs/server.log has exactly one ERROR line. Report the "
            "request_id and the full timestamp of that error."
        ),
        grade=_has(LOG_ERROR_REQ_ID, "23:41:17"),
        timeout_s=420,
    ),
    Task(
        task_id="past_eof",
        capability="offset-past-EOF note vs silence",
        prompt=(
            "Read lines 900-950 of data/report.txt and summarize them. "
            "Include the file's total line count in your answer."
        ),
        grade=_has(str(REPORT_LINES)),
    ),
    Task(
        task_id="empty_config",
        capability="empty-file note vs ambiguous silence",
        prompt=(
            "What overrides are configured in config/overrides.yaml? List "
            "them, or state clearly if there are none."
        ),
        grade=_grade_empty,
    ),
    Task(
        task_id="unicode_filename",
        capability="unicode filename equivalence (NFD/narrow-space/curly quote)",
        prompt=(
            f'Read the file "notes/{NOTES_NAME_CLEAN}" and report the third '
            "bullet point exactly."
        ),
        grade=_has(NOTES_BULLET_3),
        notes="On-disk name is NFD + U+202F + U+2019; prompt spelling is clean.",
    ),
    Task(
        task_id="near_miss_filename",
        capability="did-you-mean on close filenames",
        prompt="Summarize the build instructions in AGENT.md.",
        grade=_has(AGENTS_BUILD_CMD.split()[-1]),  # "build:prod"
        notes="Only AGENTS.md exists.",
    ),
    Task(
        task_id="fifo_hang",
        capability="device/special-file guard (FIFO read = self-shipped DoS)",
        prompt=(
            "Use the read_file tool to read logs/live.pipe and report what "
            "you find."
        ),
        grade=_grade_fifo,
        timeout_s=240,
        notes=(
            "Baseline read_file blocks on the FIFO until exec timeout. "
            "Prompt names the tool so the guard itself is exercised; the "
            "terminal-recovery path is measured by wall time + turns."
        ),
    ),
    Task(
        task_id="lying_extension",
        capability="magic-byte sniff vs extension trust",
        prompt=(
            "What kind of content is in data/data.txt? Describe what the "
            "file actually contains."
        ),
        grade=_grade_binary,
    ),
]

TASKS_BY_ID = {t.task_id: t for t in TASKS}
