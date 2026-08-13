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

import pytest

# Trigger the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ClarifyChoiceView,
    ExecApprovalView,
    ModelPickerView,
    SlashConfirmView,
    UpdatePromptView,
    _component_check_auth,
    _resolve_exec_approval_admin_gate,
)


@pytest.fixture(autouse=True)
def _clear_component_auth_env(monkeypatch):
    from unittest.mock import MagicMock, patch

    for name in (
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)

    # Default-mock PairingStore so tests don't hit the filesystem.
    # Pairing-specific tests override this with explicit mock values.
    mock_store = MagicMock()
    mock_store.is_approved.return_value = False
    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        yield


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


# ── no policy configured -> deny unless allow-all is explicit ──────────────


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


# ── role allowlist OR semantics ────────────────────────────────────────────


# ── fail-closed on missing role data ───────────────────────────────────────


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


# ---------------------------------------------------------------------------
# Pairing store: users approved via ``hermes pairing approve`` must be
# authorized even without DISCORD_ALLOWED_USERS / DISCORD_ALLOWED_ROLES.
# ---------------------------------------------------------------------------


def test_component_check_pairing_approved_user_passes(monkeypatch):
    """User approved in pairing store passes even without allowlists."""
    from unittest.mock import MagicMock, patch

    mock_store = MagicMock()
    mock_store.is_approved.return_value = True
    # Override the autouse fixture's mock with approved=True
    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        interaction = _interaction(11111)
        assert _component_check_auth(interaction, set(), set()) is True
    mock_store.is_approved.assert_called_once_with("discord", "11111")


# ---------------------------------------------------------------------------
# Opt-in admin gate for exec-approval buttons (feat/discord-admin-exec-approval).
# Default OFF: any admitted user can approve (the v0.16-restored behavior).
# When `require_admin_for_exec_approval` is true, the clicker must ALSO be in
# `allow_admin_from`. Fails closed (logged) when the toggle is on but no
# admins are configured. Only ExecApprovalView is gated — other views stay
# user-scope.
# ---------------------------------------------------------------------------


def test_admin_gate_resolver_on_parses_admins():
    """Toggle true -> gate enabled, admins coerced from allow_admin_from."""
    require_admin, admins = _resolve_exec_approval_admin_gate(
        {"require_admin_for_exec_approval": True, "allow_admin_from": "111, 222"}
    )
    assert require_admin is True
    assert admins == {"111", "222"}
    # list form normalizes identically
    _, admins_list = _resolve_exec_approval_admin_gate(
        {"require_admin_for_exec_approval": "true", "allow_admin_from": [111, 222]}
    )
    assert admins_list == {"111", "222"}


def test_exec_view_gate_on_non_admin_rejected():
    """Gate on: admitted user who is NOT an admin is rejected at the button."""
    view = ExecApprovalView(
        session_key="s",
        allowed_user_ids={"11111", "22222"},
        require_admin=True,
        admin_user_ids={"11111"},
    )
    # 22222 is admitted (in allowlist) but not an admin -> rejected.
    assert view._check_auth(_interaction(22222)) is False


def test_exec_view_gate_on_no_admins_fails_closed(caplog):
    """Gate on but no admins configured -> nobody approves, logged once."""
    import logging

    view = ExecApprovalView(
        session_key="s",
        allowed_user_ids={"11111"},
        require_admin=True,
        admin_user_ids=set(),
    )
    with caplog.at_level(logging.WARNING):
        assert view._check_auth(_interaction(11111)) is False
    assert any(
        "require_admin_for_exec_approval" in r.message for r in caplog.records
    )


def test_other_views_not_admin_gated():
    """Lower-stakes views never take the admin gate — they stay user-scope."""
    # SlashConfirmView/ModelPickerView/etc. construct without require_admin and
    # delegate straight to _component_check_auth.
    sc = SlashConfirmView(
        session_key="s", confirm_id="c", allowed_user_ids={"11111"}
    )
    assert sc._check_auth(_interaction(11111)) is True

