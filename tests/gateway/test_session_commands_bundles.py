"""A local (Desktop/Ink) session's slash dispatch resolves skill bundles, not only single skills.

`/bundle` from the Desktop composer reached `gateway/session_commands._skill_directive`, which only
consulted the skill table and refused with `unsupported_command`, so a bundle that the messaging
rewrite (`_hm_bundle_slash_rewrite`) and the classic CLI both expand never started a turn (#150311).
"""
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError


@pytest.fixture
def bundle_home(tmp_path, monkeypatch):
    """An isolated profile home: its own ``skills/`` and ``skill-bundles/`` trees."""
    home = tmp_path / "home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(home / "skill-bundles"))
    import agent.skill_bundles as bundles
    bundles._bundles_cache, bundles._bundles_cache_mtime = {}, None
    yield home, skills
    bundles._bundles_cache, bundles._bundles_cache_mtime = {}, None


def _write_skill(skills: Path, name: str, body: str) -> None:
    skill_dir = skills / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture skill {name}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )


def _write_bundle(home: Path, slug: str, members: list[str], instruction: str = "") -> None:
    bundles_dir = home / "skill-bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"name: {slug}", "skills:", *(f"  - {member}" for member in members)]
    if instruction:
        lines.append(f"instruction: {instruction}")
    (bundles_dir / f"{slug}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import agent.skill_bundles as bundles
    bundles.scan_bundles()


def test_local_slash_resolves_a_skill_bundle(bundle_home):
    """The reported repro: a bundle-only name loads its members instead of `unsupported_command`."""
    home, skills = bundle_home
    _write_skill(skills, "bundle-demo-skill", "BUNDLE_DEMO_BODY: state which guidance was loaded.")
    _write_bundle(home, "bundle-demo", ["bundle-demo-skill"],
                  instruction="Apply the included skill to the current request.")

    from gateway.session_commands import _skill_directive

    directive = _skill_directive("bundle-demo", "say which guidance was loaded", "local-route")

    assert directive["type"] == "skill"
    assert directive["name"] == "bundle-demo"
    assert "BUNDLE_DEMO_BODY" in directive["message"]
    assert "Apply the included skill to the current request." in directive["message"]
    # The client repaints `display` in the transcript: the invocation as typed, never the payload.
    assert directive["display"] == "/bundle-demo say which guidance was loaded"
    assert "BUNDLE_DEMO_BODY" not in directive["display"]


def test_a_bundle_wins_the_slug_it_shares_with_a_skill(bundle_home):
    """Bundle precedence, as the messaging rewrite and the CLI resolve it."""
    home, skills = bundle_home
    _write_skill(skills, "shared-demo", "SHARED_SKILL_BODY")
    _write_skill(skills, "shared-member", "SHARED_MEMBER_BODY")
    _write_bundle(home, "shared-demo", ["shared-member"])

    from gateway.session_commands import _skill_directive

    directive = _skill_directive("shared-demo", "", "local-route")

    assert "SHARED_MEMBER_BODY" in directive["message"]
    assert "SHARED_SKILL_BODY" not in directive["message"]


def test_a_bundle_with_no_available_member_is_a_named_refusal(bundle_home):
    """Every member uninstalled: ``command_unavailable``, never an empty turn."""
    home, _skills = bundle_home
    _write_bundle(home, "empty-demo", ["not-installed-member"])

    from gateway.session_commands import _skill_directive

    with pytest.raises(RuntimeStoreError) as refused:
        _skill_directive("empty-demo", "", "local-route")

    assert refused.value.reason == "command_unavailable"


def test_an_unknown_slash_still_refuses_unsupported(bundle_home):
    """The refusal the reported symptom names stays in place for names that are neither."""
    from gateway.session_commands import _skill_directive

    with pytest.raises(RuntimeStoreError) as refused:
        _skill_directive("no-such-thing", "", "local-route")

    assert refused.value.reason == "unsupported_command"


def test_a_single_skill_still_resolves_alongside_bundles(bundle_home):
    """The individual-skill path the same function carried is unchanged."""
    _home, skills = bundle_home
    _write_skill(skills, "probe-skill", "SINGLE_SKILL_BODY")

    from gateway.session_commands import _skill_directive

    directive = _skill_directive("probe-skill", "do the thing", "local-route")

    assert directive["type"] == "skill"
    assert directive["name"] == "probe-skill"
    assert "SINGLE_SKILL_BODY" in directive["message"]
    assert directive["display"] == "/probe-skill do the thing"
