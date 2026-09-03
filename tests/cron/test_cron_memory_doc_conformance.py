"""Doc ⟷ code conformance: cron memory behavior.

``cron/scheduler.py`` runs cron jobs with ``skip_memory=False`` (see
``TestRunJobMemory`` in ``test_scheduler.py``) — memory loads for cron
sessions like any other agent run. Several docs used to claim the opposite
(``skip_memory=True``, "memory providers intentionally do not run during
cron"), which would lead a reader (human or agent) to reason from a false
premise about whether a cron job touched persistent memory. These tests pin
the docs to the code so the two can't silently diverge again.
"""

from pathlib import Path

# tests/cron/test_cron_memory_doc_conformance.py -> repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DOC_PATHS = [
    _REPO_ROOT / "AGENTS.md",
    _REPO_ROOT
    / "skills"
    / "autonomous-ai-agents"
    / "hermes-agent"
    / "references"
    / "background-systems.md",
    _REPO_ROOT / "website" / "docs" / "user-guide" / "features" / "spotify.md",
]


def test_cron_memory_docs_do_not_claim_skip_memory_true():
    for path in _DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "skip_memory=True" not in text, (
            f"{path} claims cron passes skip_memory=True, but cron/scheduler.py "
            f"passes skip_memory=False — memory loads for cron like any other "
            f"agent run. See test_run_job_memory_enabled_in_cron in "
            f"tests/cron/test_scheduler.py."
        )


def test_cron_memory_docs_state_skip_memory_false():
    for path in _DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "skip_memory=False" in text, (
            f"{path} should document that cron passes skip_memory=False, "
            f"matching cron/scheduler.py."
        )
