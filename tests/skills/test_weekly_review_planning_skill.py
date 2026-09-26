"""Invariant: every skill a cron blueprint loads ships as a bundled skill."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _skill_dir_exists(name: str) -> bool:
    return bool(
        list(REPO_ROOT.glob(f"skills/*/{name}/SKILL.md"))
        + list(REPO_ROOT.glob(f"skills/*/*/{name}/SKILL.md"))
    )


def test_every_blueprint_skill_resolves_in_repo():
    """Invariant: any skill a blueprint loads must exist as a bundled skill."""
    from cron.blueprint_catalog import CATALOG

    for bp in CATALOG:
        for skill_name in bp.skills:
            assert _skill_dir_exists(skill_name), (
                f"blueprint {bp.key!r} loads nonexistent skill {skill_name!r}"
            )
