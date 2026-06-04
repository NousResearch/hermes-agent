"""
tests/test_skills_scoped.py — Unit tests for tools/skills_scoped.py

All S3 calls are mocked with unittest.mock. No real AWS credentials or network
access required. Tests verify:
  - Scope resolution order: personal → team → global (first hit wins).
  - Global write is permanently hard-blocked with PermissionError.
  - Unknown scope raises ValueError.
  - list_skills() shadows same-named skills from less-specific scopes.
  - promote_skill_to_team() copies personal → team.
  - promote_skill_to_team() raises FileNotFoundError when personal skill is absent.
  - HERMES_MODE gate in skill_manage: saas → S3 path, local → filesystem path.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from hermes_identity import HermesIdentity


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_identity(
    platform: str = "slack",
    team_id: str = "TTEAM01",
    user_id: str = "UUSER01",
    channel_id: str = "CCHAN01",
    thread_id: Optional[str] = None,
) -> HermesIdentity:
    return HermesIdentity(
        platform=platform,
        team_id=team_id,
        user_id=user_id,
        channel_id=channel_id,
        thread_id=thread_id,
    )


ALICE = make_identity(user_id="UALICE")
BOB = make_identity(user_id="UBOB")  # same team, different user


SKILL_CONTENT = """---
name: my-skill
description: A test skill.
---

## Trigger
When running tests.

## Steps
1. Run pytest.
"""


def _make_s3_get_response(content: str) -> dict:
    """Build a minimal mock S3 get_object response dict."""
    return {"Body": BytesIO(content.encode("utf-8"))}


def _make_client_error(code: str) -> Exception:
    """Build a botocore ClientError-shaped exception with the given error code."""
    from botocore.exceptions import ClientError
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "Not Found"}},
        operation_name="GetObject",
    )


def _ckey(identity, scope: str, name: str) -> str:
    """Canonical S3 key for a skill file (matches the Skills Service layout)."""
    return f"hermes-skills/{identity.tenant_slug}/{scope}/{name}/SKILL.md"


def _cprefix(identity, scope: str) -> str:
    """Canonical S3 list prefix for a (tenant, scope) pair."""
    return f"hermes-skills/{identity.tenant_slug}/{scope}/"


# ---------------------------------------------------------------------------
# resolve_skill — scope resolution order
# ---------------------------------------------------------------------------

class TestResolveSkill:
    """resolve_skill() walks personal → team → global, returns first hit."""

    def test_personal_scope_hit_returns_content(self):
        """When the personal scope has the skill, it is returned immediately."""
        from tools.skills_scoped import resolve_skill

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = _make_s3_get_response(SKILL_CONTENT)

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = resolve_skill("my-skill", ALICE)

        assert result == SKILL_CONTENT
        # Should have stopped at personal — only one get_object call.
        assert mock_s3.get_object.call_count == 1
        called_key = mock_s3.get_object.call_args[1]["Key"]
        assert called_key == _ckey(ALICE, "personal", "my-skill")

    def test_team_scope_fallback_when_no_personal(self):
        """Missing personal scope falls back to team scope."""
        from tools.skills_scoped import resolve_skill

        personal_key = _ckey(ALICE, "personal", "my-skill")
        team_key = _ckey(ALICE, "team", "my-skill")

        def side_effect(Bucket, Key):
            if Key == personal_key:
                raise _make_client_error("NoSuchKey")
            if Key == team_key:
                return _make_s3_get_response("team version")
            raise AssertionError(f"Unexpected key: {Key}")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = side_effect

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = resolve_skill("my-skill", ALICE)

        assert result == "team version"
        assert mock_s3.get_object.call_count == 2

    def test_global_scope_fallback_when_no_personal_or_team(self):
        """Missing personal + team falls back to global scope."""
        from tools.skills_scoped import resolve_skill

        personal_key = _ckey(ALICE, "personal", "my-skill")
        team_key = _ckey(ALICE, "team", "my-skill")
        global_key = _ckey(ALICE, "global", "my-skill")

        def side_effect(Bucket, Key):
            if Key in (personal_key, team_key):
                raise _make_client_error("NoSuchKey")
            if Key == global_key:
                return _make_s3_get_response("global version")
            raise AssertionError(f"Unexpected key: {Key}")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = side_effect

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = resolve_skill("my-skill", ALICE)

        assert result == "global version"
        assert mock_s3.get_object.call_count == 3

    def test_returns_none_when_not_found_anywhere(self):
        """If no scope has the skill, resolve_skill returns None."""
        from tools.skills_scoped import resolve_skill

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _make_client_error("NoSuchKey")

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = resolve_skill("nonexistent", ALICE)

        assert result is None
        # All three scopes tried.
        assert mock_s3.get_object.call_count == 3

    def test_s3_error_other_than_not_found_propagates(self):
        """Non-404 S3 errors are not silenced."""
        from tools.skills_scoped import resolve_skill

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _make_client_error("AccessDenied")

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(Exception):  # ClientError
                resolve_skill("my-skill", ALICE)

    def test_personal_shadows_team_content(self):
        """Personal version is returned even when team also has the skill."""
        from tools.skills_scoped import resolve_skill

        mock_s3 = MagicMock()
        # Personal has "personal version"; team also has a copy (should not be hit).
        mock_s3.get_object.return_value = _make_s3_get_response("personal version")

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = resolve_skill("my-skill", ALICE)

        assert result == "personal version"
        assert mock_s3.get_object.call_count == 1


# ---------------------------------------------------------------------------
# write_skill — permission gates
# ---------------------------------------------------------------------------

class TestWriteSkill:
    def test_write_to_personal_succeeds(self):
        """Writing to personal scope calls put_object with correct key."""
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            write_skill("my-skill", SKILL_CONTENT, scope="personal", identity=ALICE)

        mock_s3.put_object.assert_called_once()
        kwargs = mock_s3.put_object.call_args[1]
        assert kwargs["Key"] == _ckey(ALICE, "personal", "my-skill")
        assert kwargs["Body"] == SKILL_CONTENT.encode("utf-8")

    def test_write_to_team_succeeds(self):
        """Writing to team scope calls put_object with team prefix."""
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            write_skill("my-skill", SKILL_CONTENT, scope="team", identity=ALICE)

        kwargs = mock_s3.put_object.call_args[1]
        assert kwargs["Key"] == _ckey(ALICE, "team", "my-skill")

    def test_write_to_global_raises_permission_error(self):
        """Global writes are hard-blocked — no S3 call made."""
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(PermissionError, match="global"):
                write_skill("my-skill", SKILL_CONTENT, scope="global", identity=ALICE)

        mock_s3.put_object.assert_not_called()

    def test_write_to_unknown_scope_raises_value_error(self):
        """Unrecognised scope names surface a ValueError, not a silent failure."""
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(ValueError, match="Unknown scope"):
                write_skill("my-skill", SKILL_CONTENT, scope="cosmic", identity=ALICE)

        mock_s3.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Canonical S3 key layout (Plan 039 A2.5)
#
# The agent's saas skill WRITE path must produce keys at the SAME layout the
# hermes-skills-service READS from:
#     hermes-skills/{tenant_slug}/{scope}/{name}/SKILL.md
# where tenant_slug = "{platform}_{team_id}" and scope ∈ {personal, team, global}.
# For this single-user deploy, "personal" is per-tenant (no user_id segment).
# ---------------------------------------------------------------------------

class TestCanonicalKeyLayout:
    """write/resolve/list/promote keys match the service's hermes-skills/ layout."""

    SLACK = make_identity(platform="slack", team_id="T0B16FV0KFF", user_id="U123")

    def test_write_personal_uses_canonical_key(self):
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            write_skill("uat-x", SKILL_CONTENT, scope="personal", identity=self.SLACK)

        key = mock_s3.put_object.call_args[1]["Key"]
        assert key == "hermes-skills/slack_T0B16FV0KFF/personal/uat-x/SKILL.md"

    def test_write_team_uses_canonical_key(self):
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            write_skill("uat-x", SKILL_CONTENT, scope="team", identity=self.SLACK)

        key = mock_s3.put_object.call_args[1]["Key"]
        assert key == "hermes-skills/slack_T0B16FV0KFF/team/uat-x/SKILL.md"

    def test_resolve_walks_canonical_keys(self):
        from tools.skills_scoped import resolve_skill

        tried = []

        def side_effect(Bucket, Key):
            tried.append(Key)
            raise _make_client_error("NoSuchKey")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = side_effect
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            resolve_skill("uat-x", self.SLACK)

        assert tried == [
            "hermes-skills/slack_T0B16FV0KFF/personal/uat-x/SKILL.md",
            "hermes-skills/slack_T0B16FV0KFF/team/uat-x/SKILL.md",
            "hermes-skills/slack_T0B16FV0KFF/global/uat-x/SKILL.md",
        ]

    def test_list_skills_uses_canonical_prefixes(self):
        from tools.skills_scoped import list_skills

        prefixes_listed = []

        def paginate_side_effect(Bucket, Prefix, Delimiter):
            prefixes_listed.append(Prefix)
            return [{"CommonPrefixes": []}]

        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = paginate_side_effect
        mock_s3 = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            list_skills(self.SLACK)

        assert prefixes_listed == [
            "hermes-skills/slack_T0B16FV0KFF/personal/",
            "hermes-skills/slack_T0B16FV0KFF/team/",
            "hermes-skills/slack_T0B16FV0KFF/global/",
        ]

    def test_promote_uses_canonical_keys(self):
        from tools.skills_scoped import promote_skill_to_team

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = _make_s3_get_response(SKILL_CONTENT)
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            promote_skill_to_team("uat-x", self.SLACK)

        get_key = mock_s3.get_object.call_args[1]["Key"]
        put_key = mock_s3.put_object.call_args[1]["Key"]
        assert get_key == "hermes-skills/slack_T0B16FV0KFF/personal/uat-x/SKILL.md"
        assert put_key == "hermes-skills/slack_T0B16FV0KFF/team/uat-x/SKILL.md"


# ---------------------------------------------------------------------------
# list_skills — scope annotation + shadowing
# ---------------------------------------------------------------------------

class TestListSkills:
    """list_skills() returns annotated skills with personal shadowing team/global."""

    def _make_paginator(self, common_prefixes: list[dict]) -> MagicMock:
        """Build a mock paginator that yields one page with the given CommonPrefixes."""
        page = {"CommonPrefixes": common_prefixes}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        return paginator

    def test_personal_skill_appears_as_personal_scope(self):
        """A skill only in personal scope is listed with scope='personal'."""
        from tools.skills_scoped import list_skills

        mock_s3 = MagicMock()
        personal_prefix = _cprefix(ALICE, "personal")

        def paginate_side_effect(Bucket, Prefix, Delimiter):
            if Prefix == personal_prefix:
                return [{"CommonPrefixes": [{"Prefix": f"{personal_prefix}my-skill/"}]}]
            return [{"CommonPrefixes": []}]

        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = paginate_side_effect
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = list_skills(ALICE)

        assert len(result) == 1
        assert result[0]["name"] == "my-skill"
        assert result[0]["scope"] == "personal"

    def test_personal_shadows_team_skill_of_same_name(self):
        """If personal and team both have 'foo', only the personal entry appears."""
        from tools.skills_scoped import list_skills

        mock_s3 = MagicMock()
        personal_prefix = _cprefix(ALICE, "personal")
        team_prefix = _cprefix(ALICE, "team")

        def paginate_side_effect(Bucket, Prefix, Delimiter):
            if Prefix == personal_prefix:
                return [{"CommonPrefixes": [{"Prefix": f"{personal_prefix}foo/"}]}]
            if Prefix == team_prefix:
                return [{"CommonPrefixes": [{"Prefix": f"{team_prefix}foo/"}]}]
            return [{"CommonPrefixes": []}]

        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = paginate_side_effect
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = list_skills(ALICE)

        # Only one entry for 'foo' — the personal one.
        foo_entries = [r for r in result if r["name"] == "foo"]
        assert len(foo_entries) == 1
        assert foo_entries[0]["scope"] == "personal"

    def test_team_skill_appears_when_no_personal_shadow(self):
        """Skills in team scope but not personal scope are listed as team-scoped."""
        from tools.skills_scoped import list_skills

        mock_s3 = MagicMock()
        team_prefix = _cprefix(ALICE, "team")

        def paginate_side_effect(Bucket, Prefix, Delimiter):
            if Prefix == team_prefix:
                return [{"CommonPrefixes": [{"Prefix": f"{team_prefix}bar/"}]}]
            return [{"CommonPrefixes": []}]

        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = paginate_side_effect
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = list_skills(ALICE)

        bar_entries = [r for r in result if r["name"] == "bar"]
        assert len(bar_entries) == 1
        assert bar_entries[0]["scope"] == "team"

    def test_global_skill_shadowed_by_team_skill(self):
        """Team 'baz' shadows global 'baz' — only team entry returned."""
        from tools.skills_scoped import list_skills

        mock_s3 = MagicMock()
        team_prefix = _cprefix(ALICE, "team")
        global_prefix = _cprefix(ALICE, "global")

        def paginate_side_effect(Bucket, Prefix, Delimiter):
            if Prefix == team_prefix:
                return [{"CommonPrefixes": [{"Prefix": f"{team_prefix}baz/"}]}]
            if Prefix == global_prefix:
                return [{"CommonPrefixes": [{"Prefix": f"{global_prefix}baz/"}]}]
            return [{"CommonPrefixes": []}]

        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = paginate_side_effect
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = list_skills(ALICE)

        baz_entries = [r for r in result if r["name"] == "baz"]
        assert len(baz_entries) == 1
        assert baz_entries[0]["scope"] == "team"


# ---------------------------------------------------------------------------
# promote_skill_to_team
# ---------------------------------------------------------------------------

class TestPromoteSkillToTeam:
    def test_promotes_personal_to_team(self):
        """promote_skill_to_team copies personal SKILL.md to team scope."""
        from tools.skills_scoped import promote_skill_to_team

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = _make_s3_get_response(SKILL_CONTENT)

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            promote_skill_to_team("my-skill", ALICE)

        # get_object called for personal key
        get_kwargs = mock_s3.get_object.call_args[1]
        assert get_kwargs["Key"] == _ckey(ALICE, "personal", "my-skill")

        # put_object called for team key with same content
        put_kwargs = mock_s3.put_object.call_args[1]
        assert put_kwargs["Key"] == _ckey(ALICE, "team", "my-skill")
        assert put_kwargs["Body"] == SKILL_CONTENT.encode("utf-8")

    def test_raises_file_not_found_when_personal_skill_absent(self):
        """promote_skill_to_team raises FileNotFoundError if personal skill missing."""
        from tools.skills_scoped import promote_skill_to_team

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _make_client_error("NoSuchKey")

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(FileNotFoundError, match="my-skill"):
                promote_skill_to_team("my-skill", ALICE)

        mock_s3.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Tenant isolation (regression guard)
#
# Plan 039 A2.5: the canonical key layout matches the Skills Service, which has
# NO per-user "personal" segment — "personal" is per-TENANT. So same-team users
# share personal scope (single-user-deploy semantics), but DIFFERENT tenants
# (different platform/team) remain fully isolated.
# ---------------------------------------------------------------------------

OTHER_TENANT = make_identity(team_id="TOTHER99", user_id="UELSE")


class TestTenantIsolation:
    """Skills in one tenant must never appear in another tenant's resolution."""

    def test_personal_is_per_tenant_not_per_user(self):
        """ALICE and BOB share a tenant_slug, so their personal keys are identical."""
        assert ALICE.tenant_slug == BOB.tenant_slug
        assert _ckey(ALICE, "personal", "x") == _ckey(BOB, "personal", "x")

    def test_other_tenant_personal_key_never_tried(self):
        """
        A skill in one tenant's personal scope is never reached by another
        tenant's resolution — the tenant_slug prefix differs.
        """
        from tools.skills_scoped import resolve_skill

        alice_personal_key = _ckey(ALICE, "personal", "private-skill")
        keys_checked = []

        def side_effect(Bucket, Key):
            keys_checked.append(Key)
            if Key == alice_personal_key:
                pytest.fail(f"Other tenant tried Alice's personal key: {Key}")
            raise _make_client_error("NoSuchKey")

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = side_effect

        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = resolve_skill("private-skill", OTHER_TENANT)

        assert result is None
        assert alice_personal_key not in keys_checked

    def test_alice_and_bob_share_team_scope(self):
        """A team-scoped skill is visible to both Alice and Bob (same tenant)."""
        from tools.skills_scoped import resolve_skill

        team_key = _ckey(ALICE, "team", "shared-skill")
        assert ALICE.team_scope == BOB.team_scope

        def make_side_effect(identity):
            def side_effect(Bucket, Key):
                if Key == _ckey(identity, "personal", "shared-skill"):
                    raise _make_client_error("NoSuchKey")
                if Key == team_key:
                    return _make_s3_get_response("shared content")
                raise _make_client_error("NoSuchKey")
            return side_effect

        for identity in (ALICE, BOB):
            mock_s3 = MagicMock()
            mock_s3.get_object.side_effect = make_side_effect(identity)
            with patch("tools.skills_scoped.boto3") as mock_boto3:
                mock_boto3.client.return_value = mock_s3
                result = resolve_skill("shared-skill", identity)
            assert result == "shared content", f"Failed for {identity.user_id}"


# ---------------------------------------------------------------------------
# HERMES_MODE gate in skill_manage
# ---------------------------------------------------------------------------

class TestSkillManageHermesModeGate:
    """Verify that HERMES_MODE=saas routes through S3; local stays on filesystem."""

    _VALID_CONTENT = SKILL_CONTENT

    def test_local_mode_does_not_call_s3(self):
        """With HERMES_MODE unset, skill_manage must not touch S3."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_MODE", None)
            with patch("tools.skill_manager_tool._create_skill") as mock_create:
                mock_create.return_value = {"success": True, "message": "ok"}
                result_json = skill_manage(
                    action="create",
                    name="local-skill",
                    content=self._VALID_CONTENT,
                    identity=None,
                )
        result = json.loads(result_json)
        assert result["success"] is True
        mock_create.assert_called_once()

    def test_saas_mode_without_identity_falls_through_to_local(self):
        """
        HERMES_MODE=saas but identity=None must not crash — it falls through
        to the local filesystem path gracefully.
        """
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_manager_tool._create_skill") as mock_create:
                mock_create.return_value = {"success": True, "message": "ok"}
                result_json = skill_manage(
                    action="create",
                    name="no-identity-skill",
                    content=self._VALID_CONTENT,
                    identity=None,
                )
        result = json.loads(result_json)
        assert result["success"] is True
        mock_create.assert_called_once()

    def test_saas_mode_with_identity_routes_to_s3(self):
        """HERMES_MODE=saas + valid identity → write_skill called, not _create_skill."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skills_scoped.write_skill") as mock_write, \
                 patch("tools.skill_manager_tool._create_skill") as mock_fs_create:
                # write_skill is imported inside the function body
                with patch("tools.skill_manager_tool.skill_manage.__module__"):
                    pass
                # Patch where it's actually used in skill_manager_tool
                with patch("tools.skill_manager_tool._create_skill") as mock_fs2:
                    # Patch the scoped module's write_skill via its import path
                    with patch("tools.skills_scoped.boto3") as mock_boto3:
                        mock_boto3.client.return_value = MagicMock()
                        result_json = skill_manage(
                            action="create",
                            name="saas-skill",
                            content=self._VALID_CONTENT,
                            identity=ALICE,
                        )
                    mock_fs2.assert_not_called()

        result = json.loads(result_json)
        assert result["success"] is True
        assert "S3" in result.get("message", "") or "personal" in result.get("scope", "")

    def test_saas_mode_create_returns_scope_info(self):
        """SaaS create response includes 'scope' and 's3_prefix' for observability."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skills_scoped.boto3") as mock_boto3:
                mock_boto3.client.return_value = MagicMock()
                result_json = skill_manage(
                    action="create",
                    name="saas-skill",
                    content=self._VALID_CONTENT,
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result.get("scope") == "personal"
        assert (
            f"hermes-skills/{ALICE.tenant_slug}/personal/saas-skill"
            in result.get("s3_prefix", "")
        )
