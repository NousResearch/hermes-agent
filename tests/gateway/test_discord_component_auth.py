"""Security regression tests: Discord component views honor allowlists.

The interactive component views (ExecApprovalView, SlashConfirmView,
UpdatePromptView, ModelPickerView, ClarifyChoiceView) historically accepted only
``allowed_user_ids``. Deployments that configure DISCORD_ALLOWED_ROLES
without DISCORD_ALLOWED_USERS therefore had a wide-open component
surface: any guild member who could see the prompt could approve exec
commands, cancel slash confirmations, or switch the model -- even when
the same user would be rejected at the slash and on_message gates.

These tests pin user/role/global allowlist semantics, explicit allow-all
handling, and fail-closed behavior so the parity cannot regress.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Trigger the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ClarifyChoiceView,
    DiscordAdapter,
    ExecApprovalView,
    ModelPickerView,
    SlashConfirmView,
    UpdatePromptView,
    _component_check_auth,
)
from gateway.config import PlatformConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_component_auth_env(monkeypatch):
    for name in (
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Direct helper coverage -- the views all delegate to this helper, so
# pinning the helper's contract pins all call sites.
# ---------------------------------------------------------------------------


def _interaction(user_id, role_ids=None, *, drop_user=False, drop_roles=False):
    """Build a mock interaction with the requested user/role shape.

    drop_user simulates a payload whose .user attribute is None.
    drop_roles simulates a payload where .user has no .roles attribute
    at all (DM-context Member, raw User payload).
    """
    if drop_user:
        return SimpleNamespace(user=None)

    user_kwargs = {"id": user_id}
    if not drop_roles:
        user_kwargs["roles"] = [SimpleNamespace(id=r) for r in (role_ids or [])]
    return SimpleNamespace(user=SimpleNamespace(**user_kwargs))


def _make_adapter(*, allowed_users=None, allowed_roles=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    return adapter


# ── no policy configured -> deny unless allow-all is explicit ──────────────


def test_component_check_empty_allowlists_rejects_by_default(monkeypatch):
    """Button interactions must fail closed without an allowlist or allow-all."""
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    interaction = _interaction(11111)
    assert _component_check_auth(interaction, set(), set()) is False
    assert _component_check_auth(interaction, None, None) is False


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("DISCORD_ALLOW_ALL_USERS", "true"),
        ("GATEWAY_ALLOW_ALL_USERS", "yes"),
    ],
)
def test_component_check_explicit_allow_all_passes(monkeypatch, env_name, env_value):
    monkeypatch.setenv(env_name, env_value)
    interaction = _interaction(11111)
    assert _component_check_auth(interaction, set(), set()) is True


# ── user allowlist ─────────────────────────────────────────────────────────


def test_component_check_user_in_user_allowlist_passes():
    interaction = _interaction(11111)
    assert _component_check_auth(interaction, {"11111"}, set()) is True


def test_component_check_user_not_in_user_allowlist_rejected():
    interaction = _interaction(99999)
    assert _component_check_auth(interaction, {"11111"}, set()) is False


def test_component_check_user_in_global_allowlist_passes(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "11111,22222")
    interaction = _interaction(11111)
    assert _component_check_auth(interaction, set(), set()) is True


def test_component_check_global_allowlist_without_match_rejects(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "22222")
    interaction = _interaction(11111)
    assert _component_check_auth(interaction, set(), set()) is False


def test_component_check_wildcard_user_allowlist_passes():
    interaction = _interaction(99999)
    assert _component_check_auth(interaction, {"*"}, set()) is True


@pytest.mark.asyncio
async def test_send_exec_approval_allows_originating_requester_without_global_allowlist():
    adapter = _make_adapter()
    channel = MagicMock()
    sent_msg = MagicMock()
    sent_msg.id = 123
    channel.send = AsyncMock(return_value=sent_msg)
    adapter._client.get_channel = MagicMock(return_value=channel)

    result = await adapter.send_exec_approval(
        chat_id="9001",
        command="rm -rf /tmp/demo",
        session_key="sess-1",
        metadata={"requester_user_id": "42"},
    )

    assert result.success is True
    view = channel.send.call_args.kwargs["view"]
    assert isinstance(view, ExecApprovalView)
    assert view._check_auth(_interaction(42)) is True
    assert view._check_auth(_interaction(99)) is False


@pytest.mark.asyncio
async def test_send_slash_confirm_allows_originating_requester_without_global_allowlist():
    adapter = _make_adapter()
    channel = MagicMock()
    sent_msg = MagicMock()
    sent_msg.id = 456
    channel.send = AsyncMock(return_value=sent_msg)
    adapter._client.get_channel = MagicMock(return_value=channel)

    result = await adapter.send_slash_confirm(
        chat_id="9001",
        title="Confirm",
        message="Apply the change?",
        session_key="sess-2",
        confirm_id="confirm-1",
        metadata={"requester_user_id": "42"},
    )

    assert result.success is True
    view = channel.send.call_args.kwargs["view"]
    assert isinstance(view, SlashConfirmView)
    assert view._check_auth(_interaction(42)) is True
    assert view._check_auth(_interaction(99)) is False


# ── role allowlist OR semantics ────────────────────────────────────────────


def test_component_check_role_only_user_with_matching_role_passes():
    """Role-only deployment (DISCORD_ALLOWED_ROLES set, DISCORD_ALLOWED_USERS
    empty) where the user is not in the empty user list but DOES carry a
    matching role: must pass. This is the regression that prompted the
    fix -- previously _check_auth allowed everyone when the user set was
    empty, ignoring the role allowlist."""
    interaction = _interaction(99999, role_ids=[42])
    assert _component_check_auth(interaction, set(), {42}) is True


def test_component_check_accepts_role_allowlist_sequences():
    interaction = _interaction(99999, role_ids=[42])
    assert _component_check_auth(interaction, set(), [42]) is True


def test_component_check_role_only_user_without_matching_role_rejected():
    """Role-only deployment where the user has no matching role: reject.
    Previously this allowed everyone because allowed_user_ids was empty."""
    interaction = _interaction(99999, role_ids=[7, 8])
    assert _component_check_auth(interaction, set(), {42}) is False


def test_component_check_user_or_role_user_match():
    """Both allowlists set; user matches user allowlist: pass."""
    interaction = _interaction(11111, role_ids=[7])
    assert _component_check_auth(interaction, {"11111"}, {42}) is True


def test_component_check_user_or_role_role_match():
    """Both allowlists set; user not in user list but in role list: pass."""
    interaction = _interaction(99999, role_ids=[42])
    assert _component_check_auth(interaction, {"11111"}, {42}) is True


def test_component_check_user_or_role_neither_match():
    """Both allowlists set; user matches neither: reject."""
    interaction = _interaction(99999, role_ids=[7])
    assert _component_check_auth(interaction, {"11111"}, {42}) is False


# ── fail-closed on missing role data ───────────────────────────────────────


def test_component_check_role_policy_with_no_roles_attr_rejects():
    """Role allowlist configured but interaction.user has no .roles
    attribute (DM-context Member, raw User payload): must reject. A user
    without resolvable roles cannot satisfy a role allowlist."""
    interaction = _interaction(11111, drop_roles=True)
    assert _component_check_auth(interaction, set(), {42}) is False


def test_component_check_missing_user_with_allowlist_rejects():
    """interaction.user is None with any allowlist configured: fail
    closed without raising AttributeError."""
    interaction = _interaction(0, drop_user=True)
    assert _component_check_auth(interaction, {"11111"}, set()) is False
    assert _component_check_auth(interaction, set(), {42}) is False


# ---------------------------------------------------------------------------
# View construction: every view must accept allowed_role_ids and route
# through the shared helper. Default value preserves prior call-sites.
# ---------------------------------------------------------------------------


def test_exec_approval_view_accepts_role_allowlist():
    view = ExecApprovalView(
        session_key="sess-1",
        allowed_user_ids={"11111"},
        allowed_role_ids={42},
    )
    # Role-only user passes
    assert view._check_auth(_interaction(99999, role_ids=[42])) is True
    # Neither user nor role match: reject
    assert view._check_auth(_interaction(99999, role_ids=[7])) is False


def test_exec_approval_view_role_default_is_empty_set():
    """Existing call sites that pass only allowed_user_ids must continue
    working with the legacy semantics (no role gate)."""
    view = ExecApprovalView(session_key="sess-1", allowed_user_ids={"11111"})
    assert view.allowed_role_ids == set()
    assert view._check_auth(_interaction(11111)) is True
    assert view._check_auth(_interaction(99999)) is False


def test_slash_confirm_view_accepts_role_allowlist():
    view = SlashConfirmView(
        session_key="sess-1",
        confirm_id="c1",
        allowed_user_ids=set(),
        allowed_role_ids={42},
    )
    assert view._check_auth(_interaction(99999, role_ids=[42])) is True
    assert view._check_auth(_interaction(99999, role_ids=[7])) is False


def test_update_prompt_view_accepts_role_allowlist():
    view = UpdatePromptView(
        session_key="sess-1",
        allowed_user_ids=set(),
        allowed_role_ids={42},
    )
    assert view._check_auth(_interaction(99999, role_ids=[42])) is True
    assert view._check_auth(_interaction(99999, role_ids=[7])) is False


def test_model_picker_view_accepts_role_allowlist():
    async def _noop(*_a, **_k):
        return ""

    view = ModelPickerView(
        providers=[],
        current_model="m",
        current_provider="p",
        session_key="sess-1",
        on_model_selected=_noop,
        allowed_user_ids=set(),
        allowed_role_ids={42},
    )
    assert view._check_auth(_interaction(99999, role_ids=[42])) is True
    assert view._check_auth(_interaction(99999, role_ids=[7])) is False


def test_clarify_choice_view_accepts_role_allowlist():
    view = ClarifyChoiceView(
        choices=["one", "two"],
        clarify_id="clarify-1",
        allowed_user_ids=set(),
        allowed_role_ids={42},
    )
    assert view._check_auth(_interaction(99999, role_ids=[42])) is True
    assert view._check_auth(_interaction(99999, role_ids=[7])) is False


# ---------------------------------------------------------------------------
# Empty allowlists across views: fail closed unless allow-all is explicit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "view_factory",
    [
        lambda: ExecApprovalView(session_key="s", allowed_user_ids=set()),
        lambda: SlashConfirmView(session_key="s", confirm_id="c", allowed_user_ids=set()),
        lambda: UpdatePromptView(session_key="s", allowed_user_ids=set()),
        lambda: ClarifyChoiceView(
            choices=["one"],
            clarify_id="c",
            allowed_user_ids=set(),
        ),
    ],
)
def test_views_empty_allowlists_reject_by_default(view_factory, monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    view = view_factory()
    assert view._check_auth(_interaction(99999)) is False


def test_model_picker_view_empty_allowlists_reject_by_default(monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    async def _noop(*_a, **_k):
        return ""

    view = ModelPickerView(
        providers=[],
        current_model="m",
        current_provider="p",
        session_key="s",
        on_model_selected=_noop,
        allowed_user_ids=set(),
    )
    assert view.allowed_role_ids == set()
    assert view._check_auth(_interaction(99999)) is False


def test_view_empty_allowlists_allow_with_explicit_allow_all(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    view = ExecApprovalView(session_key="s", allowed_user_ids=set())
    assert view._check_auth(_interaction(99999)) is True
