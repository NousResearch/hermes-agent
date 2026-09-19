"""Tests for the profile.yaml metadata layer (description + description_auto)
and the profile_describer LLM module.
"""

from __future__ import annotations

import json as jsonlib
import sys
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








# ---------------------------------------------------------------------------
# profile_describer module
# ---------------------------------------------------------------------------


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    # describe_profile now routes through call_llm (#35566) — mock it at the
    # source module.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
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


@pytest.fixture
def registered_profile(profile_env, monkeypatch):
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)
    return profile_env


@pytest.mark.parametrize("raw", [
    '{\n  "description": "Generalist agent that writes and debugs code, orchestrates autonomous sub-agents, and automates macOS/App',
    '{\n  "desc',
    '```json\n{\n  "description": "Generalist agent that writes and debugs cod',
    '```JSON\n{\n  "description": "Generalist agent that writes and debugs cod',
])
def test_describer_refuses_json_shaped_reply_that_does_not_parse(registered_profile, raw):
    """A reply that started as the requested JSON object but was cut off (#104067) is not prose:
    it must be refused and leave profile.yaml untouched -- including behind an uppercase fence."""
    profiles_mod.write_profile_meta(registered_profile, description="previous", description_auto=True)
    before = (registered_profile / "profile.yaml").read_bytes()
    with _patch_aux_client(raw), patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={}):
        outcome = describer.describe_profile("myprof", overwrite=True)
    assert outcome.ok is False
    assert (registered_profile / "profile.yaml").read_bytes() == before


def test_describer_still_accepts_plain_prose_fallback(registered_profile):
    """A reply that never looked like JSON keeps the lenient one-paragraph prose fallback."""
    with _patch_aux_client("Writes and debugs Python codebases.\n\nSecond paragraph is dropped."), \
         patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={}):
        outcome = describer.describe_profile("myprof")
    assert outcome.ok, outcome.reason
    assert outcome.description == "Writes and debugs Python codebases."
    assert profiles_mod.read_profile_meta(registered_profile)["description"] == outcome.description


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


# ---------------------------------------------------------------------------
# _collect_skills / _sample_skills — the pure prompt-building helpers
# ---------------------------------------------------------------------------


def test_collect_skills_returns_empty_without_skills_dir(tmp_path):
    assert describer._collect_skills(tmp_path / "no-such-profile") == []


def test_collect_skills_names_bare_and_categorised_skills_sorted(tmp_path):
    """``skills/alpha/SKILL.md`` -> ``alpha``; ``skills/tools/git/SKILL.md`` -> ``tools/git``."""
    profile = tmp_path / "prof"
    for rel in ("alpha/SKILL.md", "tools/git/SKILL.md", "zeta/SKILL.md"):
        p = profile / "skills" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\nname: x\n---\n", encoding="utf-8")

    assert describer._collect_skills(profile) == ["alpha", "tools/git", "zeta"]


def test_collect_skills_skips_excluded_paths(tmp_path, monkeypatch):
    profile = tmp_path / "prof"
    for rel in ("alpha/SKILL.md", "skipme/SKILL.md"):
        p = profile / "skills" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    monkeypatch.setattr(describer, "is_excluded_skill_path", lambda p: "skipme" in str(p))

    assert describer._collect_skills(profile) == ["alpha"]


def test_collect_skills_ignores_paths_outside_the_skills_dir(tmp_path, monkeypatch):
    """A ``SKILL.md`` that ``relative_to`` cannot resolve is skipped, not fatal."""
    profile = tmp_path / "prof"
    (profile / "skills").mkdir(parents=True)
    outside = tmp_path / "elsewhere" / "SKILL.md"
    monkeypatch.setattr(Path, "rglob", lambda self, _pattern: iter([outside]))

    assert describer._collect_skills(profile) == []


def test_sample_skills_is_identity_at_or_below_cap():
    names = [f"skill-{i:03d}" for i in range(describer.MAX_SKILLS_FOR_PROMPT)]
    assert describer._sample_skills(names) == names


def test_sample_skills_spreads_picks_across_the_whole_list_above_cap():
    names = [f"skill-{i:03d}" for i in range(600)]
    cap = describer.MAX_SKILLS_FOR_PROMPT
    got = describer._sample_skills(names)

    assert len(got) == cap
    assert got[0] == names[0]
    assert got[-1] == names[int((cap - 1) * (len(names) / cap))]
    # Alphabetical position is not importance: the sample must reach past the first `cap` names.
    assert got[cap // 2] > names[cap]


# ---------------------------------------------------------------------------
# describe_profile — expected failure modes return ok=False instead of raising
# ---------------------------------------------------------------------------


def test_describe_profile_unknown_profile_is_skipped(profile_env, monkeypatch):
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: False)

    outcome = describer.describe_profile("ghost")
    assert outcome.ok is False
    assert outcome.reason == "profile not found"
    assert outcome.description is None


def test_describe_profile_default_profile_resolves_hermes_home(profile_env, monkeypatch):
    """The virtual ``default`` profile has no profile dir record — it is the Hermes home."""
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: "default")
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "default")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: profile_env)

    with _patch_aux_client(jsonlib.dumps({"description": "the default profile"})):
        outcome = describer.describe_profile("default")

    assert outcome.ok, outcome.reason
    assert outcome.description == "the default profile"
    assert profiles_mod.read_profile_meta(profile_env)["description_auto"] is True


def test_describe_profile_reports_unresolvable_profile_dir(profile_env, monkeypatch):
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")

    def _boom(_name):
        raise RuntimeError("no such dir")

    monkeypatch.setattr(profiles_mod, "get_profile_dir", _boom)

    outcome = describer.describe_profile("myprof")
    assert outcome.ok is False
    assert outcome.reason.startswith("cannot resolve profile dir")
    assert "no such dir" in outcome.reason


def test_describe_profile_tolerates_unreadable_model_config(registered_profile, monkeypatch):
    def _boom(_dir):
        raise RuntimeError("unreadable config")

    monkeypatch.setattr(profiles_mod, "_read_config_model", _boom)

    with _patch_aux_client(jsonlib.dumps({"description": "still described"})):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok, outcome.reason
    assert outcome.description == "still described"


def test_describe_profile_reports_missing_aux_client(registered_profile, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)

    outcome = describer.describe_profile("myprof")
    assert outcome.ok is False
    assert outcome.reason == "auxiliary client unavailable"


def test_describe_profile_reports_llm_error_by_type(registered_profile):
    with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("429 slow down")):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    assert outcome.reason == "LLM error: RuntimeError"
    # The message is not leaked into the reason — it can carry provider payloads.
    assert "429" not in outcome.reason


def test_describe_profile_handles_response_without_choices(registered_profile):
    """A malformed response object yields an empty raw string -> refused, not a traceback."""
    broken = MagicMock()
    broken.choices = []

    with patch("agent.auxiliary_client.call_llm", return_value=broken):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    assert outcome.reason == "LLM returned an empty response"


def test_describe_profile_rejects_json_without_description_field(registered_profile):
    with _patch_aux_client(jsonlib.dumps({"summary": "wrong key"})):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    assert outcome.reason == "LLM response missing 'description' field"


def test_describe_profile_reports_write_failure(registered_profile, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(profiles_mod, "write_profile_meta", _boom)

    with _patch_aux_client(jsonlib.dumps({"description": "valid but unwritable"})):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    assert outcome.reason.startswith("failed to write profile.yaml")


# ---------------------------------------------------------------------------
# list_describable_profiles
# ---------------------------------------------------------------------------


def test_list_describable_profiles_keeps_only_missing_descriptions(monkeypatch):
    class _Row:
        def __init__(self, name, description, description_auto):
            self.name = name
            self.description = description
            self.description_auto = description_auto

    rows = [
        _Row("bare", "", True),
        _Row("auto", "generated by the describer", True),
        _Row("curated", "written by me", False),
    ]
    monkeypatch.setattr(profiles_mod, "list_profiles", lambda: rows)

    assert describer.list_describable_profiles() == ["bare", "auto"]
    assert describer.list_describable_profiles(missing_only=False) == ["bare", "auto", "curated"]


