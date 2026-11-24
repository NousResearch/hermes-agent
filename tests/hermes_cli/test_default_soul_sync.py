"""The seeded SOUL, runtime fallback, and shipped copies are one identity.

The text is intentionally duplicated: the installer writes a real SOUL.md from
``DEFAULT_SOUL_MD``, while sessions without a SOUL.md fall back to
``DEFAULT_AGENT_IDENTITY``. The repository also ships literal copies for the
Docker image and source tree. If one copy changes alone, stock and Docker
installs silently speak with different default personas.
"""

from pathlib import Path

from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from hermes_cli.default_soul import DEFAULT_SOUL_MD

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_seeded_soul_matches_runtime_fallback():
    assert DEFAULT_SOUL_MD == DEFAULT_AGENT_IDENTITY


def test_shipped_soul_copies_match_seeded_default():
    for relative_path in ("SOUL.md", "docker/SOUL.md"):
        shipped = (REPO_ROOT / relative_path).read_text(encoding="utf-8").strip()
        assert shipped == DEFAULT_SOUL_MD, f"{relative_path} drifted from DEFAULT_SOUL_MD"
