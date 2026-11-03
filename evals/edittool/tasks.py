"""Trap battery for comparing strict and Hermes file replacement semantics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class Task:
    task_id: str
    capability: str
    edits: tuple[Edit, ...]
    notes: str
    expected_files: tuple[tuple[str, str], ...]


TASKS = (
    Task(
        "anchored_replace",
        "exact unique replacement",
        (Edit("src/service.py", "return value.strip()", "return value.strip().lower()"),),
        "Both arms should apply an exact, unique replacement.",
        (("src/service.py", "def normalize(value):\n    return value.strip().lower()\n"),),
    ),
    Task(
        "indentation_drift",
        "anchored replacement versus whitespace recovery",
        (Edit("src/retry.py", "\n  return 250\n", "\n  return 500\n"),),
        "The request has two spaces while the source has four.",
        (("src/retry.py", "def retry_delay():\n    return 500\n"),),
    ),
    Task(
        "ambiguous_match",
        "unique-match enforcement",
        (Edit("config/settings.ini", "enabled = false", "enabled = true"),),
        "Two identical settings must not silently select one section.",
        (("config/settings.ini", "[api]\nenabled = false\n[worker]\nenabled = false\n"),),
    ),
    Task(
        "multi_hunk",
        "multi-hunk edit execution",
        (
            Edit(
                "src/handlers.py",
                "def create_user():\n    # TODO: validate input\n    pass",
                "def create_user():\n    validate_input()\n    pass",
            ),
            Edit("src/handlers.py", "def delete_user():", "def remove_user():"),
        ),
        "Two independent, anchored changes must both land.",
        (
            (
                "src/handlers.py",
                "def create_user():\n    validate_input()\n    pass\n\n"
                "def remove_user():\n    # TODO: validate input\n    pass\n",
            ),
        ),
    ),
    Task(
        "already_applied",
        "no-op / already-applied detection",
        (Edit("src/already.py", "STATUS = 'old'", "STATUS = 'new'"),),
        "The desired text is already present; the arm should say so loudly.",
        (("src/already.py", "STATUS = 'new'\n"),),
    ),
    Task(
        "missing_anchor",
        "failure loudness",
        (Edit("src/service.py", "return value.trim()", "return value"),),
        "A materially wrong anchor must be rejected instead of drifting.",
        (("src/service.py", "def normalize(value):\n    return value.strip()\n"),),
    ),
    Task(
        "partial_multi_hunk",
        "partial-write visibility",
        (
            Edit("src/partial.py", "FLAG = 'old'", "FLAG = 'new'"),
            Edit("src/partial.py", "MISSING = true", "MISSING = false"),
        ),
        "A later failed hunk must not hide an earlier write behind a rejected outcome.",
        (("src/partial.py", "FLAG = 'old'\n"),),
    ),
)

# Expectations are deliberately arm-specific. A task can document a useful
# fuzzy recovery (indentation_drift) or flag silent drift (missing_anchor).
EXPECTED_OUTCOMES = {
    "str_replace": {
        "anchored_replace": "applied",
        "indentation_drift": "rejected",
        "ambiguous_match": "rejected",
        "multi_hunk": "applied",
        "already_applied": "rejected",
        "missing_anchor": "rejected",
        "partial_multi_hunk": "rejected",
    },
    "hermes_patch": {
        "anchored_replace": "applied",
        "indentation_drift": "applied",
        "ambiguous_match": "rejected",
        "multi_hunk": "applied",
        "already_applied": "no_change",
        "missing_anchor": "rejected",
        "partial_multi_hunk": "rejected",
    },
}
