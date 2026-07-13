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
