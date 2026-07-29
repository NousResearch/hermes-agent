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


