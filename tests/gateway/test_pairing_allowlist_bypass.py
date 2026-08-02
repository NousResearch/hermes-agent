"""Pairing store <-> allowlist consolidation (#23778).

Design (union + option-i mirror):
  * A pairing-store entry is a first-class authorization grant. A paired user
    is authorized regardless of any configured allowlist (union), because
    ``approve_code`` is reachable only by the trusted operator (CLI/dashboard),
    never by an inbound sender.
  * When an allowlist IS already configured for the platform, approving a
    pairing code ALSO writes the user into that allowlist env var (and revoking
    removes them), so the two stay a single operator-visible source of truth.
  * On an open gateway (no allowlist configured) approval does NOT create an
    allowlist — that would silently lock an open gateway. The pairing store
    remains the grant record, honored by the authz union.
"""

import os
from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# authz union: a paired user is authorized regardless of the allowlist
# --------------------------------------------------------------------------

def _make_runner(*, paired: bool):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: paired)
    return runner


def _make_source(user_id: str = "pairme", chat_type: str = "dm"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type=chat_type,
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
    )


def test_paired_user_authorized_even_when_not_in_allowlist(monkeypatch):
    """Union semantics: pairing is a grant, honored alongside the allowlist."""
    runner = _make_runner(paired=True)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1,owner2")

    assert runner._is_user_authorized(_make_source("pairme")) is True


def test_unpaired_user_in_allowlist_still_authorized(monkeypatch):
    runner = _make_runner(paired=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1")

    assert runner._is_user_authorized(_make_source("owner1")) is True


# --------------------------------------------------------------------------
# B2 mirror: approval writes into the allowlist iff one is configured
# --------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real PairingStore backed by a temp pairing dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    import importlib

    import gateway.pairing as pairing_mod
    importlib.reload(pairing_mod)
    return pairing_mod.PairingStore()


def _approve_new_user(store, platform, user_id, user_name=""):
    code = store.generate_code(platform, user_id, user_name)
    assert code is not None
    return store.approve_code(platform, code)


def test_approval_adds_to_configured_allowlist(store, monkeypatch):
    """When an allowlist exists, approval appends the user to it (option i)."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1")
    # save_env_value writes to .env under HERMES_HOME; patch it to capture.
    captured = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "save_env_value",
                        lambda k, v: (captured.__setitem__(k, v),
                                      os.environ.__setitem__(k, v)))

    _approve_new_user(store, "telegram", "newuser99")

    assert captured.get("TELEGRAM_ALLOWED_USERS") == "owner1,newuser99"


def test_revoke_removes_from_allowlist(store, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1,newuser99")
    saved = {}
    removed = []
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "save_env_value",
                        lambda k, v: (saved.__setitem__(k, v),
                                      os.environ.__setitem__(k, v)))
    monkeypatch.setattr(cfg, "remove_env_value", lambda k: removed.append(k))
    # Seed the approved list directly so revoke has something to remove.
    store._approve_user("telegram", "newuser99", "")

    assert store.revoke("telegram", "newuser99") is True
    assert saved.get("TELEGRAM_ALLOWED_USERS") == "owner1"


def test_revoke_whatsapp_device_jid_removes_bare_allowlist_entry(store, monkeypatch):
    """Revoke with a device-suffix JID must clear the normalized phone allowlist entry.

    Approve persists/mirrors the bare phone; operators often revoke with the
    bridge's JID form. Exact-string allowlist remove left the user authorized.
    """
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "already,15551234567")
    saved = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg,
        "save_env_value",
        lambda k, v: (saved.__setitem__(k, v), os.environ.__setitem__(k, v)),
    )
    monkeypatch.setattr(cfg, "remove_env_value", lambda k: os.environ.pop(k, None))

    store._approve_user("whatsapp", "15551234567@s.whatsapp.net", "")
    assert store.is_approved("whatsapp", "15551234567@s.whatsapp.net") is True

    assert store.revoke("whatsapp", "15551234567:47@s.whatsapp.net") is True
    assert store.is_approved("whatsapp", "15551234567@s.whatsapp.net") is False
    assert saved.get("WHATSAPP_ALLOWED_USERS") == "already"
    assert os.environ.get("WHATSAPP_ALLOWED_USERS") == "already"


def test_revoke_whatsapp_removes_all_alias_forms_from_allowlist(store, monkeypatch):
    """Allowlist may hold both bare phone and JID; revoke must drop every alias."""
    monkeypatch.setenv(
        "WHATSAPP_ALLOWED_USERS",
        "keeper,15551234567,15551234567@s.whatsapp.net",
    )
    saved = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg,
        "save_env_value",
        lambda k, v: (saved.__setitem__(k, v), os.environ.__setitem__(k, v)),
    )
    store._approve_user("whatsapp", "15551234567", "")

    assert store.revoke("whatsapp", "15551234567@s.whatsapp.net") is True
    assert saved.get("WHATSAPP_ALLOWED_USERS") == "keeper"


def test_revoke_whatsapp_preserves_wildcard_allowlist_entry(store, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*,15551234567")
    saved = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg,
        "save_env_value",
        lambda k, v: (saved.__setitem__(k, v), os.environ.__setitem__(k, v)),
    )
    store._approve_user("whatsapp", "15551234567", "")

    assert store.revoke("whatsapp", "15551234567:47@s.whatsapp.net") is True
    assert saved.get("WHATSAPP_ALLOWED_USERS") == "*"


def test_revoke_whatsapp_sole_entry_denies_live_adapter_without_restart(
    store, monkeypatch,
):
    """Sole allowlist entry revoke must deny immediately on a live gateway.

    Persistence alone is not enough: WhatsAppAdapter snapshots ``_allow_from``
    at construction, and authz trusts ``dm_policy=allowlist`` when the env
    allowlist is gone. After revoke, intake and ``_is_user_authorized`` must
    both deny the device-suffix JID without restarting the gateway.
    """
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
    from gateway.run import GatewayRunner
    import gateway.run as gateway_run
    import hermes_cli.config as cfg

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15551234567")
    monkeypatch.setattr(
        cfg,
        "save_env_value",
        lambda k, v: os.environ.__setitem__(k, v),
    )
    monkeypatch.setattr(
        cfg,
        "remove_env_value",
        lambda k: (os.environ.pop(k, None), True)[1],
    )

    class LiveWhatsAppAdapter(WhatsAppBehaviorMixin):
        def __init__(self):
            self.config = SimpleNamespace(
                extra={
                    "dm_policy": "allowlist",
                    "allow_from": ["15551234567"],
                }
            )
            self.platform = Platform.WHATSAPP
            self._dm_policy = "allowlist"
            self._dm_allowlist_source = "config"
            self._allow_from = {"15551234567"}
            self._group_policy = "pairing"
            self._group_allow_from = set()

    adapter = LiveWhatsAppAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                extra={"dm_policy": "allowlist", "allow_from": ["15551234567"]},
            )
        }
    )
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = store
    runner.pairing_stores = {}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    store._approve_user("whatsapp", "15551234567@s.whatsapp.net", "")
    sender = "15551234567:47@s.whatsapp.net"
    assert adapter._is_dm_intake_allowed(sender) is True
    assert runner._is_user_authorized(
        SessionSource(
            platform=Platform.WHATSAPP,
            user_id=sender,
            chat_id=sender,
            user_name="revoked",
            chat_type="dm",
        )
    ) is True

    assert store.revoke("whatsapp", sender) is True
    assert store.is_approved("whatsapp", "15551234567@s.whatsapp.net") is False
    assert os.environ.get("WHATSAPP_ALLOWED_USERS") in (None, "")
    assert "15551234567" not in (adapter._allow_from or set())
    assert adapter._is_dm_intake_allowed(sender) is False
    assert adapter._is_dm_allowed(sender) is False
    assert runner._is_user_authorized(
        SessionSource(
            platform=Platform.WHATSAPP,
            user_id=sender,
            chat_id=sender,
            user_name="revoked",
            chat_type="dm",
        )
    ) is False


def test_whatsapp_live_allowlist_keeps_explicit_config_over_env(monkeypatch):
    """Explicit allow_from must stay authoritative when env differs.

    Live DM checks must not invert construction precedence: a stale or
    lower-precedence WHATSAPP_ALLOWED_USERS entry must not authorize (or
    replace) an explicit config allowlist.
    """
    from gateway.config import PlatformConfig
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000002")
    adapter = WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "dm_policy": "allowlist",
                "allow_from": ["15550000001"],
            },
        )
    )

    assert adapter._dm_allowlist_source == "config"
    assert adapter._is_dm_intake_allowed("15550000001") is True
    assert adapter._is_dm_allowed("15550000001") is True
    assert adapter._is_dm_intake_allowed("15550000002") is False
    assert adapter._is_dm_allowed("15550000002") is False

    # Pairing-style env mutation must not broaden a config-seeded allowlist.
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000002,15550000003")
    assert adapter._is_dm_intake_allowed("15550000001") is True
    assert adapter._is_dm_intake_allowed("15550000002") is False
    assert adapter._is_dm_intake_allowed("15550000003") is False


def test_whatsapp_live_allowlist_rereads_env_when_env_seeded(monkeypatch):
    """Env-seeded adapters still pick up pairing allowlist mutations live."""
    from gateway.config import PlatformConfig
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15551234567")
    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, extra={"dm_policy": "allowlist"})
    )

    assert adapter._dm_allowlist_source == "WHATSAPP_ALLOWED_USERS"
    assert adapter._is_dm_intake_allowed("15551234567") is True

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "")
    assert adapter._is_dm_intake_allowed("15551234567") is False
    assert adapter._is_dm_allowed("15551234567") is False
