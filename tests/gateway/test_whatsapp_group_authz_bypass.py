"""WhatsApp group-chat authorization from config (``extra.group_allow_from``).

Regression: the authz fallback called ``self._adapter_for_source`` — a name that does
not exist (the intake resolver is ``_intake_adapter_for``), so the ``AttributeError``
was swallowed by the surrounding ``except Exception`` and the config-driven group
bypass never ran. Groups were therefore authorized only through
``WHATSAPP_GROUP_ALLOW_FROM``, and a group listed in ``config.yaml`` but absent from
that env var rejected every participant ("Unauthorized user").
"""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource

GROUP = "120363001234567890@g.us"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "WHATSAPP_GROUP_ALLOW_FROM",
        "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_GROUP_ALLOWED_CHATS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _source(user_id: str = "15550000001@s.whatsapp.net") -> SessionSource:
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=GROUP,
        chat_type="group",
        user_id=user_id,
        user_name="Participant",
    )


def _runner(group_allow_from):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        config=SimpleNamespace(extra={"group_allow_from": group_allow_from})
    )
    runner._intake_adapter_for = lambda source: adapter
    runner._delivery_adapter_for = lambda source: adapter
    runner._pairing_store_for = lambda source: None
    runner._adapter_flag = lambda *a, **k: False
    return runner


def test_config_group_allowlist_authorizes_any_participant():
    """A group listed in config authorizes the participant with no env set."""
    assert _runner([GROUP])._is_user_authorized(_source()) is True


def test_group_absent_from_config_stays_unauthorized():
    """Negative control: the bypass does not authorize a group outside the list."""
    assert _runner(["999999999999@g.us"])._is_user_authorized(_source()) is False
