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


def test_describer_rejects_truncated_json_response(profile_env, monkeypatch):
    # Real-world shape (#104067): the aux model's response is cut off mid-object -- no
    # closing brace/quote -- so `_extract_json_blob` can't parse it. The old fallback
    # persisted this raw fragment verbatim as the description; it must now be refused.
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    truncated = '{\n  "description": "Generalist agent that writes and debugs code, orchestrates autonomous sub-agents, and automates macOS/App'
    with _patch_aux_client(truncated), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    assert "malformed" in outcome.reason.lower() or "truncated" in outcome.reason.lower()
    # Nothing was written: no profile.yaml, or if one exists it has no description key.
    meta = profiles_mod.read_profile_meta(profile_env)
    assert not meta.get("description")


def test_describer_rejects_json_shaped_response_with_no_closing_brace_at_all(profile_env, monkeypatch):
    # Even more truncated: cut off right after the opening brace, before any key.
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    with _patch_aux_client("{\n  \"desc"), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    meta = profiles_mod.read_profile_meta(profile_env)
    assert not meta.get("description")


def test_describer_rejects_fenced_json_shaped_truncated_response(profile_env, monkeypatch):
    # A ```json fence around a truncated object must still be recognized as JSON-shaped
    # after fence stripping, not fall through to the raw-text fallback.
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    fenced_truncated = '```json\n{\n  "description": "Generalist agent that writes and debugs cod'
    with _patch_aux_client(fenced_truncated), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    meta = profiles_mod.read_profile_meta(profile_env)
    assert not meta.get("description")


def test_describer_rejects_uppercase_fenced_json_shaped_truncated_response(profile_env, monkeypatch):
    # Regression for review feedback on #104075: an uppercase ```JSON fence must be
    # stripped the same as a lowercase one. Before, _FENCE_RE was case-sensitive, so
    # stripping this left "JSON\n{...}" -- which doesn't start with "{" -- and the
    # truncated fragment fell through to the raw-text-as-prose fallback and got persisted
    # anyway, defeating the fix for this fence variant.
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    fenced_truncated = '```JSON\n{\n  "description": "Generalist agent that writes and debugs cod'
    with _patch_aux_client(fenced_truncated), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok is False
    meta = profiles_mod.read_profile_meta(profile_env)
    assert not meta.get("description")


def test_describer_still_accepts_plain_prose_fallback(profile_env, monkeypatch):
    # Regression guard: a model that ignores the JSON instruction and just replies in
    # plain prose (never looked JSON-shaped) must still hit the existing lenient
    # raw-text fallback -- this fix must not make the describer stricter than before
    # for genuinely non-JSON replies.
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: n == "myprof")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: profile_env)

    with _patch_aux_client("Writes and debugs Python codebases end to end."), patch(
        "agent.auxiliary_client.get_auxiliary_extra_body", return_value={}
    ):
        outcome = describer.describe_profile("myprof")

    assert outcome.ok, outcome.reason
    assert outcome.description == "Writes and debugs Python codebases end to end."
    meta = profiles_mod.read_profile_meta(profile_env)
    assert meta["description"] == "Writes and debugs Python codebases end to end."
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


