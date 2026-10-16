"""Inline-shell expansion is trust-scoped (#63307): community hub installs never
auto-execute `` !`cmd` `` snippets, mirroring the hub scan's INSTALL_POLICY trust
gate — a ``--force`` or pre-scanner community install must not re-arm them."""
import json
from unittest.mock import patch

import pytest

from agent.skill_preprocessing import preprocess_skill_content

CFG = {"inline_shell": True, "inline_shell_timeout": 5}
SNIPPET = "Dynamic: !`printf SENTINEL`"


def _make_skill(skills_root, rel, body=SNIPPET):
    skill_dir = skills_root / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    name = rel.rsplit("/", 1)[-1]
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _record_hub_install(install_path: str, trust_level: str):
    """Write a hub lock entry the way ``install_from_quarantine`` does."""
    from tools.skills_hub import HubLockFile

    HubLockFile().record_install(
        name=install_path.rsplit("/", 1)[-1], source="github", identifier=f"o/r/{install_path}",
        trust_level=trust_level, scan_verdict="caution", skill_hash="sha256:0",
        install_path=install_path, files=["SKILL.md"])


def test_community_hub_install_does_not_expand(tmp_path):
    """A community-trust hub lock entry means the snippet must NOT run on view."""
    skill_dir = _make_skill(tmp_path, "mlops/hubskill")
    _record_hub_install("mlops/hubskill", "community")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        rendered = preprocess_skill_content(SNIPPET, skill_dir, skills_cfg=dict(CFG))
    assert rendered == SNIPPET  # raw DSL preserved — nothing executed


def test_trusted_hub_install_still_expands(tmp_path):
    """Trusted/builtin hub installs keep the operator's opt-in contract."""
    skill_dir = _make_skill(tmp_path, "mlops/hubskill")
    _record_hub_install("mlops/hubskill", "trusted")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        rendered = preprocess_skill_content(SNIPPET, skill_dir, skills_cfg=dict(CFG))
    assert rendered == "Dynamic: SENTINEL"


def test_local_skill_without_hub_entry_still_expands(tmp_path):
    """Bundled-synced and user-created skills carry no lock entry and keep expanding."""
    skill_dir = _make_skill(tmp_path, "mlops/my-own")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        rendered = preprocess_skill_content(SNIPPET, skill_dir, skills_cfg=dict(CFG))
    assert rendered == "Dynamic: SENTINEL"


def test_slash_invocation_leaves_community_snippet_raw(tmp_path):
    """The slash/bundle surface routes through the same gate: a community hub
    skill's message shows the raw snippet, never the executed output."""
    from agent.skill_commands import build_skill_invocation_message, scan_skill_commands

    _make_skill(tmp_path, "dyn-comm")
    _record_hub_install("dyn-comm", "community")
    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch("agent.skill_commands._load_skills_config",
              return_value={"template_vars": True, "inline_shell": True,
                            "inline_shell_timeout": 5}),
    ):
        scan_skill_commands()
        msg = build_skill_invocation_message("/dyn-comm")

    assert msg is not None
    # Executed output would be "Dynamic: SENTINEL"; the gate leaves the raw DSL.
    assert "Dynamic: SENTINEL" not in msg
    assert "!`printf SENTINEL`" in msg


def _community_bundle_home(home, lock_text=None):
    """A profile skills tree holding a --force community bundle with a nested SKILL.md,
    plus a user-created skill; *lock_text* overrides the lock (corrupt/wrong shape)."""
    skills = home / "skills"
    _make_skill(skills, "outer")
    _make_skill(skills, "outer/agents/inner")
    _make_skill(skills, "mine")
    lock = skills / ".hub" / "lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(lock_text if lock_text is not None else json.dumps({"version": 1, "installed": {
        "outer": {"trust_level": "community", "install_path": "outer"}}}), encoding="utf-8")
    return skills


@pytest.mark.parametrize("lock_text, viewer, rel, expands", [
    (None, "owner", "outer/agents/inner", False),  # nested SKILL.md inside a community bundle
    (None, "other", "outer", False),  # another profile reaching it through external_dirs
    (None, "other", "outer/agents/inner", False),
    ('{"version": 1, "inst', "owner", "outer", False),  # truncated lock fails closed
    ('{"version": 1, "installed": []}', "other", "outer", False),  # wrong shape fails closed
    (None, "owner", "mine", True),  # user-created skill beside the install keeps the opt-in
    (None, "other", "mine", True),
])
def test_community_provenance_follows_the_owning_lock(tmp_path, lock_text, viewer, rel, expands):
    owner = _community_bundle_home(tmp_path / "owner", lock_text)
    active = owner if viewer == "owner" else (tmp_path / "other" / "skills")
    active.mkdir(parents=True, exist_ok=True)
    with patch("tools.skills_tool.SKILLS_DIR", active):
        rendered = preprocess_skill_content(SNIPPET, owner / rel, skills_cfg=dict(CFG))
    assert rendered == ("Dynamic: SENTINEL" if expands else SNIPPET)
