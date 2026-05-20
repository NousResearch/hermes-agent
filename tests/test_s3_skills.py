"""
tests/test_s3_skills.py — Unit tests for hermes_storage.s3_skills.S3SkillSource.

All S3 calls are mocked with moto (if available) or unittest.mock.
No live AWS credentials required.

Tests cover:
  - S3SkillSource construction validation (empty tenant_slug, slashes, bad scope).
  - key_prefix() format is correct.
  - _skill_key() format is correct.
  - list_skills() returns correct skill names from paginated S3 response.
  - get_skill_file() downloads and returns UTF-8 content.
  - get_skill_file() raises KeyError when object missing.
  - push_skill() uploads SKILL.md and supporting files.
  - push_skill() raises ValueError when skill_dir missing or has no SKILL.md.
  - delete_skill() removes all objects for a skill.
  - download_to_local() downloads all files to local directory.
  - skill_source_from_identity() builds correct S3SkillSource from HermesIdentity.
  - skill_source_from_identity() raises ValueError when HERMES_MODE != saas.
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from hermes_identity import HermesIdentity
from hermes_storage.s3_skills import (
    S3SkillSource,
    skill_source_from_identity,
    DEFAULT_BUCKET,
    _KEY_PREFIX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_identity(
    platform: str = "slack",
    team_id: str = "TTEAM01",
    user_id: str = "UUSER01",
    channel_id: str = "CCHAN01",
) -> HermesIdentity:
    return HermesIdentity(
        platform=platform,
        team_id=team_id,
        user_id=user_id,
        channel_id=channel_id,
    )


@pytest.fixture
def mock_s3():
    """Provide a pre-configured mock boto3 S3 client."""
    client = MagicMock()
    # Attach exceptions as needed.
    client.exceptions = MagicMock()
    client.exceptions.NoSuchKey = KeyError  # simulate NoSuchKey
    return client


@pytest.fixture
def source(mock_s3) -> S3SkillSource:
    s = S3SkillSource(tenant_slug="slack_TTEAM01", bucket="test-bucket", scope="personal")
    s._s3 = mock_s3
    return s


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_empty_tenant_slug_raises():
    with pytest.raises(ValueError, match="tenant_slug must be non-empty"):
        S3SkillSource(tenant_slug="")


def test_tenant_slug_with_slash_raises():
    with pytest.raises(ValueError, match="must not contain slashes"):
        S3SkillSource(tenant_slug="slack/TTEAM01")


def test_invalid_scope_raises():
    with pytest.raises(ValueError, match="scope must be one of"):
        S3SkillSource(tenant_slug="slack_TTEAM01", scope="invalid")


def test_valid_scopes():
    for scope in ("personal", "team", "global"):
        s = S3SkillSource(tenant_slug="slack_TTEAM01", scope=scope)
        assert s._scope == scope


def test_bucket_defaults_to_env_var(monkeypatch):
    monkeypatch.setenv("S3_SKILLS_BUCKET", "custom-bucket-from-env")
    s = S3SkillSource(tenant_slug="slack_TTEAM01")
    assert s._bucket == "custom-bucket-from-env"


def test_bucket_defaults_to_default(monkeypatch):
    monkeypatch.delenv("S3_SKILLS_BUCKET", raising=False)
    s = S3SkillSource(tenant_slug="slack_TTEAM01")
    assert s._bucket == DEFAULT_BUCKET


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------

def test_key_prefix_no_skill():
    s = S3SkillSource(tenant_slug="slack_TTEAM01", scope="personal")
    prefix = s.key_prefix()
    assert prefix == f"{_KEY_PREFIX}/slack_TTEAM01/personal/"


def test_key_prefix_with_skill():
    s = S3SkillSource(tenant_slug="slack_TTEAM01", scope="team")
    prefix = s.key_prefix("my-skill")
    assert prefix == f"{_KEY_PREFIX}/slack_TTEAM01/team/my-skill/"


def test_skill_key_skill_md():
    s = S3SkillSource(tenant_slug="slack_TTEAM01", scope="personal")
    key = s._skill_key("my-skill", "SKILL.md")
    assert key == f"{_KEY_PREFIX}/slack_TTEAM01/personal/my-skill/SKILL.md"


def test_skill_key_nested():
    s = S3SkillSource(tenant_slug="slack_TTEAM01", scope="personal")
    key = s._skill_key("my-skill", "references/api.md")
    assert key == f"{_KEY_PREFIX}/slack_TTEAM01/personal/my-skill/references/api.md"


def test_skill_key_strips_leading_slash():
    s = S3SkillSource(tenant_slug="slack_TTEAM01", scope="personal")
    key = s._skill_key("my-skill", "/SKILL.md")
    assert not key.startswith("/")


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_skills_returns_names(source: S3SkillSource, mock_s3) -> None:
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": f"{_KEY_PREFIX}/slack_TTEAM01/personal/skill-a/"},
                {"Prefix": f"{_KEY_PREFIX}/slack_TTEAM01/personal/skill-b/"},
            ]
        }
    ]
    mock_s3.get_paginator.return_value = paginator

    skills = await source.list_skills()

    assert "skill-a" in skills
    assert "skill-b" in skills
    assert len(skills) == 2


@pytest.mark.asyncio
async def test_list_skills_empty(source: S3SkillSource, mock_s3) -> None:
    paginator = MagicMock()
    paginator.paginate.return_value = [{"CommonPrefixes": []}]
    mock_s3.get_paginator.return_value = paginator

    skills = await source.list_skills()
    assert skills == []


@pytest.mark.asyncio
async def test_list_skills_no_common_prefixes(source: S3SkillSource, mock_s3) -> None:
    """Page with no CommonPrefixes key (not just empty list)."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]  # No 'CommonPrefixes' key
    mock_s3.get_paginator.return_value = paginator

    skills = await source.list_skills()
    assert skills == []


# ---------------------------------------------------------------------------
# get_skill_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_skill_file_returns_content(source: S3SkillSource, mock_s3) -> None:
    content = "---\nname: my-skill\n---\n\n# My Skill"
    mock_s3.get_object.return_value = {
        "Body": io.BytesIO(content.encode("utf-8"))
    }

    result = await source.get_skill_file("my-skill", "SKILL.md")
    assert result == content


@pytest.mark.asyncio
async def test_get_skill_file_raises_key_error_on_missing(source: S3SkillSource, mock_s3) -> None:
    mock_s3.get_object.side_effect = KeyError("NoSuchKey")

    with pytest.raises(KeyError):
        await source.get_skill_file("nonexistent-skill", "SKILL.md")


@pytest.mark.asyncio
async def test_get_skill_file_default_is_skill_md(source: S3SkillSource, mock_s3) -> None:
    mock_s3.get_object.return_value = {"Body": io.BytesIO(b"content")}

    await source.get_skill_file("my-skill")

    call_kwargs = mock_s3.get_object.call_args[1]
    assert "SKILL.md" in call_kwargs.get("Key", "")


# ---------------------------------------------------------------------------
# push_skill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_skill_uploads_files(
    source: S3SkillSource, mock_s3, tmp_path: Path
) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n\n# My Skill")
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "api.md").write_text("# API Reference")

    count = await source.push_skill("my-skill", skill_dir)

    assert count == 2  # SKILL.md + references/api.md
    assert mock_s3.put_object.call_count == 2

    # Verify key format for SKILL.md.
    call_args_list = mock_s3.put_object.call_args_list
    keys_uploaded = [c[1]["Key"] for c in call_args_list]
    assert any("SKILL.md" in k for k in keys_uploaded)
    assert any("references/api.md" in k for k in keys_uploaded)


@pytest.mark.asyncio
async def test_push_skill_raises_if_dir_missing(source: S3SkillSource) -> None:
    with pytest.raises(ValueError, match="skill_dir not found"):
        await source.push_skill("my-skill", Path("/nonexistent/path"))


@pytest.mark.asyncio
async def test_push_skill_raises_if_no_skill_md(
    source: S3SkillSource, tmp_path: Path
) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    # No SKILL.md created.

    with pytest.raises(ValueError, match="SKILL.md missing"):
        await source.push_skill("my-skill", skill_dir)


@pytest.mark.asyncio
async def test_push_skill_skips_binary_files(
    source: S3SkillSource, mock_s3, tmp_path: Path
) -> None:
    """Binary files (.png, .jpg) are not uploaded."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---")
    (skill_dir / "logo.png").write_bytes(b"\x89PNG\r\n")  # binary file

    count = await source.push_skill("my-skill", skill_dir)

    assert count == 1  # Only SKILL.md.
    assert mock_s3.put_object.call_count == 1


# ---------------------------------------------------------------------------
# delete_skill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_skill_deletes_all_objects(source: S3SkillSource, mock_s3) -> None:
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": f"{_KEY_PREFIX}/slack_TTEAM01/personal/my-skill/SKILL.md"},
                {"Key": f"{_KEY_PREFIX}/slack_TTEAM01/personal/my-skill/references/api.md"},
            ]
        }
    ]
    mock_s3.get_paginator.return_value = paginator

    count = await source.delete_skill("my-skill")

    assert count == 2
    mock_s3.delete_objects.assert_called_once()
    delete_spec = mock_s3.delete_objects.call_args[1]["Delete"]
    assert len(delete_spec["Objects"]) == 2


@pytest.mark.asyncio
async def test_delete_skill_no_objects(source: S3SkillSource, mock_s3) -> None:
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": []}]
    mock_s3.get_paginator.return_value = paginator

    count = await source.delete_skill("nonexistent-skill")

    assert count == 0
    mock_s3.delete_objects.assert_not_called()


# ---------------------------------------------------------------------------
# download_to_local
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_to_local_writes_files(
    source: S3SkillSource, mock_s3, tmp_path: Path
) -> None:
    skill_content = b"---\nname: my-skill\n---\n\n# My Skill"
    ref_content = b"# API Reference"

    paginator = MagicMock()
    prefix = source.key_prefix("my-skill")
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": f"{prefix}SKILL.md"},
                {"Key": f"{prefix}references/api.md"},
            ]
        }
    ]
    mock_s3.get_paginator.return_value = paginator
    mock_s3.get_object.side_effect = [
        {"Body": io.BytesIO(skill_content)},
        {"Body": io.BytesIO(ref_content)},
    ]

    result_dir = await source.download_to_local("my-skill", tmp_path)

    assert result_dir == tmp_path / "my-skill"
    assert (result_dir / "SKILL.md").read_bytes() == skill_content
    assert (result_dir / "references" / "api.md").read_bytes() == ref_content


@pytest.mark.asyncio
async def test_download_to_local_creates_dir(
    source: S3SkillSource, mock_s3, tmp_path: Path
) -> None:
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": []}]
    mock_s3.get_paginator.return_value = paginator

    local_dir = tmp_path / "skills"
    result = await source.download_to_local("new-skill", local_dir)

    assert result.exists()


# ---------------------------------------------------------------------------
# skill_source_from_identity
# ---------------------------------------------------------------------------

def test_skill_source_from_identity(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_MODE", "saas")
    identity = make_identity(platform="slack", team_id="TTEAM01")
    source = skill_source_from_identity(identity, scope="personal")

    assert source._tenant_slug == "slack_TTEAM01"
    assert source._scope == "personal"


def test_skill_source_from_identity_team_scope(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_MODE", "saas")
    identity = make_identity()
    source = skill_source_from_identity(identity, scope="team")
    assert source._scope == "team"


def test_skill_source_from_identity_raises_if_not_saas(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_MODE", "local")
    identity = make_identity()
    with pytest.raises(ValueError, match="HERMES_MODE='local'"):
        skill_source_from_identity(identity)


def test_skill_source_from_identity_no_mode_raises(monkeypatch) -> None:
    env = {k: v for k, v in os.environ.items() if k != "HERMES_MODE"}
    with patch.dict(os.environ, env, clear=True):
        identity = make_identity()
        with pytest.raises(ValueError, match="HERMES_MODE="):
            skill_source_from_identity(identity)


# ---------------------------------------------------------------------------
# Tenant isolation: key prefixes don't overlap
# ---------------------------------------------------------------------------

def test_different_tenants_have_different_prefixes() -> None:
    s_a = S3SkillSource(tenant_slug="slack_TTEAM_A", scope="personal")
    s_b = S3SkillSource(tenant_slug="slack_TTEAM_B", scope="personal")

    assert s_a.key_prefix() != s_b.key_prefix()
    assert "TTEAM_A" in s_a.key_prefix()
    assert "TTEAM_B" in s_b.key_prefix()


def test_different_scopes_have_different_prefixes() -> None:
    s_p = S3SkillSource(tenant_slug="slack_TTEAM01", scope="personal")
    s_t = S3SkillSource(tenant_slug="slack_TTEAM01", scope="team")
    s_g = S3SkillSource(tenant_slug="slack_TTEAM01", scope="global")

    prefixes = {s_p.key_prefix(), s_t.key_prefix(), s_g.key_prefix()}
    assert len(prefixes) == 3
