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


def test_paired_user_authorized_with_no_allowlist(monkeypatch):
    runner = _make_runner(paired=True)

    assert runner._is_user_authorized(_make_source("pairme")) is True


def test_unpaired_user_in_allowlist_still_authorized(monkeypatch):
    runner = _make_runner(paired=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1")

    assert runner._is_user_authorized(_make_source("owner1")) is True


def test_unpaired_user_not_in_allowlist_denied(monkeypatch):
    runner = _make_runner(paired=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1")

    assert runner._is_user_authorized(_make_source("stranger")) is False


def test_unpaired_user_no_allowlist_denied_no_failopen(monkeypatch):
    runner = _make_runner(paired=False)

    assert runner._is_user_authorized(_make_source("stranger")) is False


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


def test_approval_no_allowlist_leaves_gateway_open(store, monkeypatch):
    """Open gateway: approval must NOT create an allowlist (option i)."""
    called = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "save_env_value",
                        lambda k, v: called.__setitem__(k, v))

    _approve_new_user(store, "telegram", "newuser99")

    assert "TELEGRAM_ALLOWED_USERS" not in called
    assert os.getenv("TELEGRAM_ALLOWED_USERS", "") == ""
    # The pairing store still records the grant (union honors it).
    assert store.is_approved("telegram", "newuser99") is True


def test_approval_idempotent_when_already_in_allowlist(store, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner1,newuser99")
    called = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "save_env_value",
                        lambda k, v: called.__setitem__(k, v))

    _approve_new_user(store, "telegram", "newuser99")

    # Already present — no rewrite.
    assert "TELEGRAM_ALLOWED_USERS" not in called


def test_approval_skips_wildcard_allowlist(store, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    called = {}
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "save_env_value",
                        lambda k, v: called.__setitem__(k, v))

    _approve_new_user(store, "telegram", "newuser99")

    assert "TELEGRAM_ALLOWED_USERS" not in called


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


def test_revoke_removes_env_var_when_list_empties(store, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "newuser99")
    removed = []
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "save_env_value",
                        lambda k, v: os.environ.__setitem__(k, v))
    monkeypatch.setattr(cfg, "remove_env_value", lambda k: removed.append(k))
    store._approve_user("telegram", "newuser99", "")
    # _approve_user's own add is a no-op (already present); reset for the revoke.
    os.environ["TELEGRAM_ALLOWED_USERS"] = "newuser99"

    assert store.revoke("telegram", "newuser99") is True
    assert "TELEGRAM_ALLOWED_USERS" in removed
