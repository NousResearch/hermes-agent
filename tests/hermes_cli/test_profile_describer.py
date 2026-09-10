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


