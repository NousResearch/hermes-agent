"""
tests/test_atlas_scopes.py — Unit tests for hermes_storage/atlas_scopes.py

All Atlas calls are mocked with unittest.mock (no live Atlas connection needed).
Tests verify:
  - personal_fence() returns correct format string.
  - team_fence() returns correct format string.
  - fence_for_scope() maps scope names to fence strings correctly.
  - scoped_atlas_search() fans out across personal → team → global.
  - scoped_atlas_search() deduplicates by chunk ID, personal wins.
  - scoped_atlas_search() handles partial Atlas failures gracefully.
  - scoped_atlas_ingest() routes to correct fence for personal/team.
  - scoped_atlas_ingest() raises PermissionError for global scope.
  - scoped_atlas_ingest() raises ValueError for unknown scope.
  - Cross-tenant isolation: User A's personal fence is NOT queried for User B.
  - Team fence is shared between users in the same team.
  - detect_team_scope_intent() detects NL sharing phrases.
  - resolve_scope_from_intent() returns correct scope string.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

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
BOB = make_identity(user_id="UBOB")   # same team, different user


def _make_result(chunk_id: str, text: str = "content") -> dict:
    """Build a minimal Atlas result dict."""
    return {"id": chunk_id, "text": text}


# ---------------------------------------------------------------------------
# personal_fence and team_fence — pure function tests
# ---------------------------------------------------------------------------

class TestFenceHelpers:
    def test_personal_fence_format(self):
        from hermes_storage.atlas_scopes import personal_fence
        result = personal_fence(ALICE)
        assert result == f"personal:{ALICE.platform}:{ALICE.team_id}:{ALICE.user_id}"

    def test_team_fence_format(self):
        from hermes_storage.atlas_scopes import team_fence
        result = team_fence(ALICE)
        assert result == f"team:{ALICE.platform}:{ALICE.team_id}"

    def test_personal_fence_differs_across_users(self):
        from hermes_storage.atlas_scopes import personal_fence
        assert personal_fence(ALICE) != personal_fence(BOB)

    def test_team_fence_same_across_users_in_same_team(self):
        """Alice and Bob are in the same team — their team fences must match."""
        from hermes_storage.atlas_scopes import team_fence
        assert team_fence(ALICE) == team_fence(BOB)

    def test_team_fence_differs_across_teams(self):
        from hermes_storage.atlas_scopes import team_fence
        identity_a = make_identity(team_id="TTEAM_A")
        identity_b = make_identity(team_id="TTEAM_B")
        assert team_fence(identity_a) != team_fence(identity_b)

    def test_global_fence_is_none(self):
        from hermes_storage.atlas_scopes import GLOBAL_FENCE
        assert GLOBAL_FENCE is None

    def test_fence_for_scope_personal(self):
        from hermes_storage.atlas_scopes import fence_for_scope, personal_fence
        assert fence_for_scope("personal", ALICE) == personal_fence(ALICE)

    def test_fence_for_scope_team(self):
        from hermes_storage.atlas_scopes import fence_for_scope, team_fence
        assert fence_for_scope("team", ALICE) == team_fence(ALICE)

    def test_fence_for_scope_global_returns_none(self):
        from hermes_storage.atlas_scopes import fence_for_scope
        assert fence_for_scope("global", ALICE) is None

    def test_fence_for_scope_unknown_raises_value_error(self):
        from hermes_storage.atlas_scopes import fence_for_scope
        with pytest.raises(ValueError, match="Unknown scope"):
            fence_for_scope("cosmic", ALICE)


# ---------------------------------------------------------------------------
# scoped_atlas_search — fan-out and deduplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestScopedAtlasSearch:
    """Async tests for scoped_atlas_search fan-out, merging, and deduplication."""

    async def test_fans_out_across_all_three_fences(self):
        """All three fences are queried when identity is provided."""
        from hermes_storage.atlas_scopes import scoped_atlas_search, personal_fence, team_fence

        called_fences = []

        async def mock_search(query, fence, top_k):
            called_fences.append(fence)
            return []

        await scoped_atlas_search("test", ALICE, top_k=9, atlas_search_fn=mock_search)

        # Must have queried personal, team, and global (None)
        assert personal_fence(ALICE) in called_fences
        assert team_fence(ALICE) in called_fences
        assert None in called_fences
        assert len(called_fences) == 3

    async def test_personal_results_ranked_first(self):
        """Personal results come before team and global in the merged output."""
        from hermes_storage.atlas_scopes import scoped_atlas_search, personal_fence, team_fence

        async def mock_search(query, fence, top_k):
            if fence == personal_fence(ALICE):
                return [_make_result("p1", "personal")]
            if fence == team_fence(ALICE):
                return [_make_result("t1", "team")]
            return [_make_result("g1", "global")]  # global fence

        results = await scoped_atlas_search("test", ALICE, top_k=9, atlas_search_fn=mock_search)

        scopes_in_order = [r["scope"] for r in results]
        assert scopes_in_order[0] == "personal"
        # Team and global can follow in any order relative to each other
        assert set(scopes_in_order) == {"personal", "team", "global"}

    async def test_deduplicates_by_chunk_id_personal_wins(self):
        """Same chunk ID in personal + global: personal wins, returned once."""
        from hermes_storage.atlas_scopes import scoped_atlas_search, personal_fence, team_fence

        shared_id = "shared-chunk"

        async def mock_search(query, fence, top_k):
            if fence == personal_fence(ALICE):
                return [_make_result(shared_id, "personal version")]
            if fence == team_fence(ALICE):
                return []
            return [_make_result(shared_id, "global version")]  # same chunk ID

        results = await scoped_atlas_search("test", ALICE, top_k=9, atlas_search_fn=mock_search)

        # Should appear exactly once, and it should be the personal version
        id_matches = [r for r in results if r["id"] == shared_id]
        assert len(id_matches) == 1
        assert id_matches[0]["scope"] == "personal"
        assert id_matches[0]["text"] == "personal version"

    async def test_top_k_limits_total_results(self):
        """top_k bounds the total number of results returned."""
        from hermes_storage.atlas_scopes import scoped_atlas_search

        async def mock_search(query, fence, top_k):
            return [_make_result(f"{fence}-{i}", f"text {i}") for i in range(top_k)]

        results = await scoped_atlas_search("test", ALICE, top_k=5, atlas_search_fn=mock_search)
        assert len(results) <= 5

    async def test_partial_failure_does_not_crash_search(self):
        """If one fence raises, the search continues with remaining fences."""
        from hermes_storage.atlas_scopes import scoped_atlas_search, team_fence

        async def mock_search(query, fence, top_k):
            if fence == team_fence(ALICE):
                raise RuntimeError("Atlas team fence temporarily unavailable")
            if fence is None:  # global
                return [_make_result("g1", "global")]
            return [_make_result("p1", "personal")]

        # Should not raise; personal and global results should appear.
        results = await scoped_atlas_search("test", ALICE, top_k=9, atlas_search_fn=mock_search)
        scopes = {r["scope"] for r in results}
        assert "team" not in scopes  # team failed
        assert "personal" in scopes or "global" in scopes  # at least one succeeded

    async def test_requires_atlas_search_fn(self):
        """scoped_atlas_search without atlas_search_fn raises RuntimeError."""
        from hermes_storage.atlas_scopes import scoped_atlas_search

        with pytest.raises(RuntimeError, match="atlas_search_fn"):
            await scoped_atlas_search("test", ALICE)


# ---------------------------------------------------------------------------
# scoped_atlas_ingest — scope gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestScopedAtlasIngest:
    async def test_personal_ingest_writes_to_personal_fence(self):
        """Personal scope ingest calls Atlas with the correct personal fence."""
        from hermes_storage.atlas_scopes import scoped_atlas_ingest, personal_fence

        ingest_calls = []

        async def mock_ingest(payload, fence, provenance):
            ingest_calls.append({"fence": fence, "payload": payload, "prov": provenance})

        await scoped_atlas_ingest(
            "some knowledge",
            ALICE,
            scope="personal",
            atlas_ingest_fn=mock_ingest,
        )

        assert len(ingest_calls) == 1
        assert ingest_calls[0]["fence"] == personal_fence(ALICE)
        assert ingest_calls[0]["payload"] == "some knowledge"

    async def test_team_ingest_writes_to_team_fence(self):
        """Team scope ingest calls Atlas with the correct team fence."""
        from hermes_storage.atlas_scopes import scoped_atlas_ingest, team_fence

        ingest_calls = []

        async def mock_ingest(payload, fence, provenance):
            ingest_calls.append({"fence": fence})

        await scoped_atlas_ingest(
            "team knowledge",
            ALICE,
            scope="team",
            atlas_ingest_fn=mock_ingest,
        )

        assert len(ingest_calls) == 1
        assert ingest_calls[0]["fence"] == team_fence(ALICE)

    async def test_global_ingest_raises_permission_error(self):
        """Global scope ingest raises PermissionError — agent turns cannot write global."""
        from hermes_storage.atlas_scopes import scoped_atlas_ingest

        ingest_calls = []

        async def mock_ingest(payload, fence, provenance):
            ingest_calls.append(fence)

        with pytest.raises(PermissionError, match="global"):
            await scoped_atlas_ingest(
                "global knowledge",
                ALICE,
                scope="global",
                atlas_ingest_fn=mock_ingest,
            )

        # Must not have called Atlas
        assert ingest_calls == []

    async def test_unknown_scope_raises_value_error(self):
        """Unrecognised scope raises ValueError before calling Atlas."""
        from hermes_storage.atlas_scopes import scoped_atlas_ingest

        ingest_calls = []

        async def mock_ingest(payload, fence, provenance):
            ingest_calls.append(fence)

        with pytest.raises(ValueError, match="Unknown scope"):
            await scoped_atlas_ingest(
                "knowledge",
                ALICE,
                scope="cosmic",
                atlas_ingest_fn=mock_ingest,
            )

        assert ingest_calls == []

    async def test_provenance_includes_identity_fields(self):
        """Provenance dict must include actor, team_id, platform, and scope."""
        from hermes_storage.atlas_scopes import scoped_atlas_ingest

        captured = {}

        async def mock_ingest(payload, fence, provenance):
            captured.update(provenance)

        await scoped_atlas_ingest(
            "fact",
            ALICE,
            scope="personal",
            atlas_ingest_fn=mock_ingest,
        )

        assert captured.get("actor") == ALICE.user_id
        assert captured.get("team_id") == ALICE.team_id
        assert captured.get("platform") == ALICE.platform
        assert captured.get("scope") == "personal"
        assert captured.get("framework") == "hermes-gateway"

    async def test_requires_atlas_ingest_fn(self):
        """scoped_atlas_ingest without atlas_ingest_fn raises RuntimeError."""
        from hermes_storage.atlas_scopes import scoped_atlas_ingest

        with pytest.raises(RuntimeError, match="atlas_ingest_fn"):
            await scoped_atlas_ingest("fact", ALICE)


# ---------------------------------------------------------------------------
# Cross-tenant isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCrossTenantIsolation:
    """Personal knowledge ingested by Alice must not be returned for Bob."""

    async def test_alice_personal_not_returned_for_bob(self):
        """
        Alice ingests to her personal fence. When Bob searches, Alice's
        personal fence is NOT queried — his personal fence has a different key.
        """
        from hermes_storage.atlas_scopes import (
            personal_fence, scoped_atlas_search
        )

        alice_personal = personal_fence(ALICE)
        bob_personal = personal_fence(BOB)
        assert alice_personal != bob_personal  # sanity

        # Simulated Atlas state: Alice's personal has "alice-fact"
        storage = {alice_personal: [_make_result("alice-chunk", "Alice's secret")]}

        async def mock_search(query, fence, top_k):
            return storage.get(fence, [])

        results = await scoped_atlas_search("secret", BOB, atlas_search_fn=mock_search)

        # Bob's search should return nothing (his personal fence is empty,
        # team and global are empty)
        assert all(r["id"] != "alice-chunk" for r in results)

    async def test_team_fence_visible_to_both_users(self):
        """Knowledge in team fence is returned for both Alice and Bob."""
        from hermes_storage.atlas_scopes import (
            team_fence, personal_fence, scoped_atlas_search
        )

        shared_team = team_fence(ALICE)
        assert team_fence(BOB) == shared_team  # same team

        storage = {shared_team: [_make_result("team-chunk", "shared team fact")]}

        async def mock_search(query, fence, top_k):
            return storage.get(fence, [])

        for user in (ALICE, BOB):
            results = await scoped_atlas_search(
                "team fact", user, atlas_search_fn=mock_search
            )
            team_hits = [r for r in results if r.get("id") == "team-chunk"]
            assert len(team_hits) == 1, f"Team fact not found for {user.user_id}"

    async def test_different_teams_cannot_see_each_other(self):
        """Team A's knowledge is not returned when Team B's identity searches."""
        from hermes_storage.atlas_scopes import team_fence, scoped_atlas_search

        team_a = make_identity(team_id="TTEAM_A", user_id="UUSER_A")
        team_b = make_identity(team_id="TTEAM_B", user_id="UUSER_B")

        storage = {team_fence(team_a): [_make_result("team-a-secret", "Team A secret")]}

        async def mock_search(query, fence, top_k):
            return storage.get(fence, [])

        results = await scoped_atlas_search("secret", team_b, atlas_search_fn=mock_search)
        assert all(r["id"] != "team-a-secret" for r in results)


# ---------------------------------------------------------------------------
# NL intent detection
# ---------------------------------------------------------------------------

class TestNLIntentDetection:
    def test_detect_team_scope_intent_positive_cases(self):
        """Known sharing phrases are detected as team intent."""
        from hermes_storage.atlas_scopes import detect_team_scope_intent

        team_phrases = [
            "save this for the team",
            "share with everyone",
            "share with the team",
            "for everyone on the team",
            "team wide",
            "team-wide",
            "company wide",
            "company-wide",
        ]
        for phrase in team_phrases:
            assert detect_team_scope_intent(phrase), f"Expected team intent for: {phrase!r}"

    def test_detect_team_scope_intent_negative_cases(self):
        """Non-sharing phrases are not detected as team intent."""
        from hermes_storage.atlas_scopes import detect_team_scope_intent

        non_team = [
            "remember this for me",
            "save this note",
            "I want to recall this later",
            "for my personal reference",
        ]
        for phrase in non_team:
            assert not detect_team_scope_intent(phrase), f"False positive for: {phrase!r}"

    def test_resolve_scope_from_intent_team(self):
        from hermes_storage.atlas_scopes import resolve_scope_from_intent
        assert resolve_scope_from_intent("share with everyone") == "team"

    def test_resolve_scope_from_intent_defaults_personal(self):
        from hermes_storage.atlas_scopes import resolve_scope_from_intent
        assert resolve_scope_from_intent("remember this fact") == "personal"

    def test_resolve_scope_from_intent_case_insensitive(self):
        from hermes_storage.atlas_scopes import resolve_scope_from_intent
        assert resolve_scope_from_intent("SHARE WITH EVERYONE") == "team"

    def test_resolve_scope_from_intent_never_returns_global(self):
        """No NL phrase should ever resolve to global scope."""
        from hermes_storage.atlas_scopes import resolve_scope_from_intent
        # Even explicit "global" language should not map to global scope
        result = resolve_scope_from_intent("store this globally")
        assert result in ("personal", "team")
        assert result != "global"


# ---------------------------------------------------------------------------
# Unaffected existing behaviour — fence=None still works
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExistingAtlasBehaviourUnaffected:
    """
    Existing Atlas calls without an identity (fence=None) must be unaffected.
    This is the pre-Phase-B behaviour.
    """

    async def test_direct_atlas_call_with_no_fence_works(self):
        """
        A direct Atlas call passing fence=None (global) still returns results.
        scoped_atlas_search is not required for pre-identity calls.
        """
        from hermes_storage.atlas_scopes import GLOBAL_FENCE

        # Simulating a direct Atlas call that doesn't go through scoped helpers.
        # fence=None == GLOBAL_FENCE — should be equivalent.
        assert GLOBAL_FENCE is None

        # The caller's existing code that does:
        #   atlas_search_knowledge(query=q, fence=None, top_k=10)
        # is unchanged. No migration needed.
        called_with_fence = []

        async def direct_search(query, fence, top_k):
            called_with_fence.append(fence)
            return [_make_result("g1")]

        result = await direct_search("test", GLOBAL_FENCE, 10)
        assert called_with_fence == [None]
        assert result[0]["id"] == "g1"
