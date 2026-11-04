"""Feishu ID-tier aliases in authorization.

Granting ``contact:user.employee_id:readonly`` makes Feishu start populating
``sender_id.user_id`` (the tenant-scoped employee ID). The adapter prefers that
tier as the primary identity, so a sender who was approved while only ``open_id``
existed suddenly presents an unknown ID and is bounced back to the pairing
handshake — the operator locks themselves out of their own bot.

The adapter therefore hands every tier it saw to ``SessionSource.user_id_aliases``
and authz accepts a grant keyed on ANY of them. Session keys stay on
``user_id_alt`` (union_id), so nothing re-keys when a scope is granted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_feishu_env(monkeypatch):
    for var in (
        "FEISHU_ALLOWED_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_runner(*approved: str):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda _platform, uid: uid in set(approved))
    return runner


def _make_source(*, aliases: tuple[str, ...]) -> SessionSource:
    """A Feishu DM whose primary tier is the tenant user_id (employee-id scope granted)."""
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="dm",
        user_id="55133",
        user_name="Zekai",
        user_id_alt="on_union",
        user_id_aliases=aliases,
    )


def test_resolve_sender_profile_exposes_every_tier():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = None  # name lookup short-circuits without an API client
    adapter._sender_name_cache = {}

    profile = asyncio.run(
        adapter._resolve_sender_profile(
            SimpleNamespace(open_id="ou_human", user_id="55133", union_id="on_union")
        )
    )

    assert profile["user_id"] == "55133"  # tenant tier stays primary
    assert profile["user_id_alt"] == "on_union"  # session keys unaffected
    assert set(profile["user_id_aliases"]) == {"ou_human", "55133", "on_union"}


def test_resolve_sender_profile_keeps_open_id_when_user_id_absent():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = None
    adapter._sender_name_cache = {}

    profile = asyncio.run(
        adapter._resolve_sender_profile(SimpleNamespace(open_id="ou_human", user_id=None, union_id=None))
    )

    assert profile["user_id"] == "ou_human"
    assert profile["user_id_aliases"] == ("ou_human",)


def test_pairing_grant_on_open_id_survives_the_tier_flip():
    runner = _make_runner("ou_human")  # approved before the employee-id scope existed

    assert runner._is_user_authorized(_make_source(aliases=("ou_human", "55133", "on_union"))) is True
    # Control: without the alias tiers the tenant user_id is an unknown stranger.
    assert runner._is_user_authorized(_make_source(aliases=())) is False


def test_env_allowlist_entry_on_open_id_matches_the_alias(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")
    runner = _make_runner()

    assert runner._is_user_authorized(_make_source(aliases=("ou_human", "55133", "on_union"))) is True
    assert runner._is_user_authorized(_make_source(aliases=())) is False


def test_aliases_round_trip_through_the_wire_dict():
    source = _make_source(aliases=("ou_human", "55133"))

    restored = SessionSource.from_dict(source.to_dict())

    assert restored.user_id_aliases == ("ou_human", "55133")
    assert restored.user_id == source.user_id
    assert restored.user_id_alt == source.user_id_alt
