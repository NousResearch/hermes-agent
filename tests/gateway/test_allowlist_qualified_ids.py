"""Allowlist matching compares qualified user_ids, not bare '@' localparts.

The generic ``user_id.split("@")[0]`` alias in ``_principal_matches_allowlist``
was added for WhatsApp JIDs (``<phone>@s.whatsapp.net``) before WhatsApp got a
dedicated alias expansion. Left unconditional, it makes a bare allowlist entry
``alice`` admit ``alice@<any domain>`` on every platform whose user_id is
'@'-shaped (email, Google Chat, iMessage handles) — domains the sender controls.
"""

import pytest

from gateway.authz_mixin import _principal_matches_allowlist
from gateway.config import Platform
from gateway.session import SessionSource


def _source(platform: Platform, user_id: str, user_id_alt: str | None = None) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        user_id_alt=user_id_alt,
        chat_id=user_id,
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = None
    return runner


# The BlueBubbles adapter does no sender gate of its own, so with its platform
# allowlist unset GATEWAY_ALLOWED_USERS is the only enforcement on that path.
@pytest.mark.parametrize(
    ("platform", "env_var"),
    [(Platform.EMAIL, "EMAIL_ALLOWED_USERS"), (Platform.BLUEBUBBLES, "GATEWAY_ALLOWED_USERS")],
)
def test_bare_localpart_entry_admits_no_foreign_domain(monkeypatch, platform, env_var):
    # conftest's hermetic env blanks the EMAIL_/GATEWAY_ vars but not BlueBubbles' own.
    for key in ("BLUEBUBBLES_ALLOWED_USERS", "BLUEBUBBLES_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_var, "alice,bob@corp.example")
    runner = _make_runner()

    assert runner._is_user_authorized(_source(platform, "alice@evil.example")) is False
    assert runner._is_user_authorized(_source(platform, "bob@evil.example")) is False
    # Positive control: a qualified entry still admits its exact sender.
    assert runner._is_user_authorized(_source(platform, "bob@corp.example")) is True


@pytest.mark.parametrize(
    ("platform", "user_id"),
    [
        (Platform.WHATSAPP, "15550000001@s.whatsapp.net"),
        (Platform.WHATSAPP, "15550000001:47@s.whatsapp.net"),
        (Platform.WHATSAPP_CLOUD, "15550000001@s.whatsapp.net"),
    ],
)
def test_whatsapp_bare_phone_entry_still_matches_jid(platform, user_id):
    assert _principal_matches_allowlist(_source(platform, user_id), user_id, {"15550000001"}) is True


# --- Feishu/Lark: union_id is a valid allowlist anchor ----------------------
#
# ``open_id`` is app-scoped: the same person holds a *different* open_id under
# every Feishu app, and all of an app's open_ids change if it is recreated or
# migrated.  ``union_id`` is developer-scoped and stable, so an operator may
# write it in ``FEISHU_ALLOWED_USERS``; the adapter carries it as ``user_id_alt``.
# Without the alias the entry silently stops matching.  The expansion mirrors the
# existing WhatsApp / SimpleX / Buzz branches above.


def test_feishu_union_id_entry_matches_when_open_id_differs():
    """The allowlist holds the union_id; the event carries a different open_id."""
    source = _source(Platform.FEISHU, "ou_other_app", user_id_alt="on_owner")

    assert _principal_matches_allowlist(source, "ou_other_app", {"on_owner"}) is True


def test_feishu_open_id_entry_still_matches():
    """Positive control: the pre-existing open_id path is unchanged."""
    source = _source(Platform.FEISHU, "ou_owner", user_id_alt="on_owner")

    assert _principal_matches_allowlist(source, "ou_owner", {"ou_owner"}) is True


def test_feishu_without_union_id_does_not_match_unrelated_entry():
    """Control: with no union_id on the source nothing is widened."""
    source = _source(Platform.FEISHU, "ou_stranger")

    assert _principal_matches_allowlist(source, "ou_stranger", {"on_owner"}) is False


def test_union_id_alias_is_feishu_scoped():
    """Control: the alias must not leak to platforms that carry a different alt id.

    Telegram's ``user_id_alt`` is not a Feishu union_id, so an entry holding it
    must not admit a Telegram sender.
    """
    source = _source(Platform.TELEGRAM, "12345", user_id_alt="on_owner")

    assert _principal_matches_allowlist(source, "12345", {"on_owner"}) is False
