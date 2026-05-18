"""Tests for the profile.yaml metadata layer (description + description_auto)
and the profile_describer LLM module.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import profiles as profiles_mod
from hermes_cli import profile_describer as describer


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Set up an isolated HERMES_HOME with a default profile dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_read_profile_meta_empty_when_missing(profile_env):
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta == {"description": "", "description_auto": False}


def test_write_and_read_profile_meta(profile_env):
    profiles_mod.write_profile_meta(
        profile_env,
        description="a useful researcher",
        description_auto=False,
    )
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta["description"] == "a useful researcher"
    assert meta["description_auto"] is False


def test_write_profile_meta_preserves_other_fields(profile_env):
    # First write sets description_auto=True; second write only updates
    # description and leaves description_auto unchanged.
    profiles_mod.write_profile_meta(
        profile_env,
        description="auto-gen",
        description_auto=True,
    )
    profiles_mod.write_profile_meta(profile_env, description="edited by hand")
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta["description"] == "edited by hand"
    assert meta["description_auto"] is True


def test_write_profile_meta_rejects_missing_dir(tmp_path):
    bogus = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        profiles_mod.write_profile_meta(bogus, description="x")


def test_read_profile_meta_tolerates_corrupt_yaml(profile_env):
    (profile_env / "profile.yaml").write_text("not: valid: yaml: [unclosed")
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta == {"description": "", "description_auto": False}


# ---------------------------------------------------------------------------
# profile_describer module
# ---------------------------------------------------------------------------


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(client, "test-model"),
    )


def test_describer_writes_description_with_auto_true(profile_env, monkeypatch):
    # Pretend "myprof" is a registered profile pointing at profile_env.
    monkeypatch.setattr(
        profiles_mod, "profile_exists", lambda n: n == "myprof",
    )
    monkeypatch.setattr(
        profiles_mod, "normalize_profile_name", lambda n: n,
    )
    monkeypatch.setattr(
        profiles_mod, "get_profile_dir", lambda n: profile_env,
    )

    payload = jsonlib.dumps({"description": "writes Python codebases"})
    with _patch_aux_client(payload), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok, outcome.reason
    assert outcome.description == "writes Python codebases"
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta["description"] == "writes Python codebases"
    assert meta["description_auto"] is True


def test_describer_refuses_to_overwrite_user_authored(profile_env, monkeypatch):
    profiles_mod.write_profile_meta(
        profile_env, description="curated", description_auto=False,
    )
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    outcome = describer.describe_profile("myprof")
    assert outcome.ok is False
    assert "already has a user-authored description" in outcome.reason
    # Description unchanged
    assert profiles_mod.read_profile_meta(profile_env)["description"] == "curated"


def test_describer_overwrite_flag_replaces_user_authored(profile_env, monkeypatch):
    profiles_mod.write_profile_meta(
        profile_env, description="curated", description_auto=False,
    )
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    payload = jsonlib.dumps({"description": "new auto-gen"})
    with _patch_aux_client(payload), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", overwrite=True)
    assert outcome.ok, outcome.reason
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta["description"] == "new auto-gen"
    assert meta["description_auto"] is True


def test_describer_handles_malformed_llm_response(profile_env, monkeypatch):
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    # Non-JSON: describer falls back to taking the first paragraph as the description.
    with _patch_aux_client("Plain text description that sneaks in"), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")
    assert outcome.ok
    assert "Plain text description" in (outcome.description or "")


def test_describer_returns_false_when_profile_missing(profile_env, monkeypatch):
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: False)
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    outcome = describer.describe_profile("ghost")
    assert outcome.ok is False
    assert "not found" in outcome.reason


# ---------------------------------------------------------------------------
# Built-in vs user-authored skill tagging + opt-in SOUL.md
# ---------------------------------------------------------------------------


def _capture_user_msg():
    """Helper that returns (patch_ctx, captured_messages_list).

    The patch_ctx is the unittest.mock context manager you `with`-enter.
    captured_messages_list is appended to on each LLM call so tests can
    assert on the prompt content the describer assembled.
    """
    captured: list = []
    client = MagicMock()

    def _fake_create(**kwargs):
        captured.append(kwargs.get("messages", []))
        return _fake_aux_response(jsonlib.dumps({"description": "ok"}))

    client.chat.completions.create = MagicMock(side_effect=_fake_create)
    return (
        patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(client, "test-model"),
        ),
        captured,
    )


def test_describer_does_not_tag_skills_by_default(profile_env, monkeypatch):
    """Default behavior: no [user]/[built-in] labels appear in the prompt."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    skills_dir = profile_env / "skills" / "trading"
    skills_dir.mkdir(parents=True)
    (skills_dir / "binance-trading-ops").mkdir()
    (skills_dir / "binance-trading-ops" / "SKILL.md").write_text(
        "---\nname: binance-trading-ops\n---\nbody"
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")  # no flags
    assert outcome.ok
    user_text = captured[0][1]["content"]
    # Skill name still appears, but no tags either way.
    assert "trading/binance-trading-ops" in user_text
    assert "[user]" not in user_text
    assert "[built-in]" not in user_text


def test_describer_tags_user_skill_when_not_in_builtin_set(profile_env, monkeypatch):
    """A skill that doesn't exist in the install-root catalogue is tagged [user]."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    # Force the describer to think no skills are built-in.
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    skills_dir = profile_env / "skills" / "trading"
    skills_dir.mkdir(parents=True)
    (skills_dir / "binance-trading-ops").mkdir()
    (skills_dir / "binance-trading-ops" / "SKILL.md").write_text(
        "---\nname: binance-trading-ops\n---\nbody"
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", tag_builtins=True)
    assert outcome.ok
    assert captured, "describer should have called the LLM"
    user_text = captured[0][1]["content"]
    assert "[user] trading/binance-trading-ops" in user_text
    assert "[built-in]" not in user_text


def test_describer_tags_builtin_skill_when_in_builtin_set(profile_env, monkeypatch):
    """A skill present in the install-root catalogue is tagged [built-in]."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(
        describer, "_builtin_skill_ids", lambda: {"github/codebase-inspection"}
    )

    skills_dir = profile_env / "skills" / "github"
    skills_dir.mkdir(parents=True)
    (skills_dir / "codebase-inspection").mkdir()
    (skills_dir / "codebase-inspection" / "SKILL.md").write_text(
        "---\nname: codebase-inspection\n---\nbody"
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", tag_builtins=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "[built-in] github/codebase-inspection" in user_text


def test_describer_does_not_include_soul_by_default(profile_env, monkeypatch):
    """SOUL.md content must not leak into the prompt when include_soul=False."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    (profile_env / "SOUL.md").write_text(
        "You are Hopper. Role: Fleet Performance Researcher."
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")  # include_soul defaults to False
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Hopper" not in user_text
    assert "SOUL.md" not in user_text


def test_describer_includes_soul_when_opt_in(profile_env, monkeypatch):
    """include_soul=True pulls SOUL.md content into the prompt."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    (profile_env / "SOUL.md").write_text(
        "You are Hopper. Role: Fleet Performance Researcher."
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", include_soul=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Hopper" in user_text
    assert "Fleet Performance Researcher" in user_text


def test_describer_include_soul_tolerates_missing_file(profile_env, monkeypatch):
    """include_soul=True must not error when SOUL.md doesn't exist."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", include_soul=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    # No SOUL block should appear.
    assert "SOUL.md" not in user_text


def test_describer_does_not_include_agents_by_default(profile_env, monkeypatch):
    """AGENTS.md content must not leak into the prompt when include_agents=False."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    (profile_env / "AGENTS.md").write_text(
        "# Newton — Delivery Engineer\nLane: All coding tasks for project repos."
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")  # no flags
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Newton" not in user_text
    assert "AGENTS.md" not in user_text


def test_describer_includes_agents_when_opt_in(profile_env, monkeypatch):
    """include_agents=True pulls AGENTS.md content into the prompt."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    (profile_env / "AGENTS.md").write_text(
        "# Newton — Delivery Engineer\nLane: All coding tasks for project repos."
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", include_agents=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Newton" in user_text
    assert "Delivery Engineer" in user_text
    assert "All coding tasks" in user_text


def test_describer_include_agents_falls_back_to_workspace(profile_env, monkeypatch):
    """When AGENTS.md isn't at profile root, fall back to workspace/AGENTS.md."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    workspace = profile_env / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text(
        "# Edison — Architect\nLane: Specs and architecture authoring."
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", include_agents=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Edison" in user_text
    assert "Architect" in user_text


def test_describer_include_agents_tolerates_missing_file(profile_env, monkeypatch):
    """include_agents=True must not error when no AGENTS.md exists."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", include_agents=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "AGENTS.md" not in user_text


def test_describer_caps_skill_list_and_keeps_user_skills(profile_env, monkeypatch):
    """When skill count exceeds MAX_SKILLS_FOR_PROMPT, user skills must all survive."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    # Set up a small built-in catalogue containing 80 skills, plus 5 user skills.
    builtin_ids = {f"category/builtin-{i:02d}" for i in range(80)}
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: builtin_ids)
    # Lower the cap so the test stays compact.
    monkeypatch.setattr(describer, "MAX_SKILLS_FOR_PROMPT", 20)

    skills_root = profile_env / "skills" / "category"
    skills_root.mkdir(parents=True)
    for i in range(80):
        sd = skills_root / f"builtin-{i:02d}"
        sd.mkdir()
        (sd / "SKILL.md").write_text("---\nname: x\n---\n")
    for i in range(5):
        sd = skills_root / f"user-{i:02d}"
        sd.mkdir()
        (sd / "SKILL.md").write_text("---\nname: x\n---\n")

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof", tag_builtins=True)
    assert outcome.ok
    user_text = captured[0][1]["content"]
    # All 5 user skills must appear despite the cap.
    for i in range(5):
        assert f"[user] category/user-{i:02d}" in user_text
    # Some built-ins should appear (sampled), but not all 80.
    builtin_count = user_text.count("[built-in]")
    assert builtin_count <= 20
    assert builtin_count > 0


def test_describer_does_not_emit_skill_descriptions_by_default(profile_env, monkeypatch):
    """Without the flag, a skill's frontmatter description must not appear."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    skills_dir = profile_env / "skills" / "trading"
    skills_dir.mkdir(parents=True)
    (skills_dir / "binance-trading-ops").mkdir()
    (skills_dir / "binance-trading-ops" / "SKILL.md").write_text(
        "---\nname: binance-trading-ops\n"
        "description: Execute Binance bucket strategy with risk monitoring\n---\nbody"
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")  # no flags
    assert outcome.ok
    user_text = captured[0][1]["content"]
    # Skill ID still appears, but not the description text.
    assert "binance-trading-ops" in user_text
    assert "Execute Binance bucket strategy" not in user_text


def test_describer_emits_skill_descriptions_when_opt_in(profile_env, monkeypatch):
    """with_skill_descriptions=True appends each skill's frontmatter description."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    skills_dir = profile_env / "skills" / "trading"
    skills_dir.mkdir(parents=True)
    (skills_dir / "binance-trading-ops").mkdir()
    (skills_dir / "binance-trading-ops" / "SKILL.md").write_text(
        "---\nname: binance-trading-ops\n"
        "description: Execute Binance bucket strategy with risk monitoring\n---\nbody"
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile(
            "myprof", with_skill_descriptions=True
        )
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Execute Binance bucket strategy with risk monitoring" in user_text


def test_describer_skill_descriptions_strip_quotes(profile_env, monkeypatch):
    """Quoted description values get stripped and surfaced cleanly."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    skills_dir = profile_env / "skills" / "trading"
    skills_dir.mkdir(parents=True)
    (skills_dir / "polymarket").mkdir()
    (skills_dir / "polymarket" / "SKILL.md").write_text(
        '---\nname: polymarket\n'
        'description: "Query Polymarket: markets, prices, orderbooks."\n---\nbody'
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile(
            "myprof", with_skill_descriptions=True
        )
    assert outcome.ok
    user_text = captured[0][1]["content"]
    assert "Query Polymarket: markets, prices, orderbooks." in user_text
    # The wrapping quotes should not appear in the rendered line.
    assert '"Query Polymarket' not in user_text


def test_describer_skill_descriptions_handles_missing_frontmatter(profile_env, monkeypatch):
    """A SKILL.md without frontmatter must not crash the run."""
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    monkeypatch.setattr(describer, "_builtin_skill_ids", lambda: set())

    skills_dir = profile_env / "skills" / "research"
    skills_dir.mkdir(parents=True)
    (skills_dir / "no-frontmatter").mkdir()
    (skills_dir / "no-frontmatter" / "SKILL.md").write_text(
        "# Just a body, no frontmatter at all."
    )

    patch_ctx, captured = _capture_user_msg()
    with patch_ctx, patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile(
            "myprof", with_skill_descriptions=True
        )
    assert outcome.ok
    user_text = captured[0][1]["content"]
    # Skill ID still appears even without a description to attach.
    assert "research/no-frontmatter" in user_text
