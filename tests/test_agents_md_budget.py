from pathlib import Path


def test_root_agents_md_fits_main_profile_context_budget():
    """Root AGENTS.md is always-loaded project context; keep it below 8k."""
    text = Path("AGENTS.md").read_text(encoding="utf-8")

    assert len(text) <= 8000
    assert "website/docs/developer-guide/agents-full.md" in text
