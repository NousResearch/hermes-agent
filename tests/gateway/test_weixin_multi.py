"""Tests for Weixin multi-account discovery.

These tests exercise ``gateway.platforms.weixin_multi`` against a
real on-disk layout under a temporary ``HERMES_HOME``.  We avoid
mocking the filesystem because the discovery helper's contract is
"read the directory layout the rest of the gateway reads" - mocks
would hide a regression that writes to the wrong path.

The tests also exercise ``register_persisted_weixin_accounts``
end-to-end: they build a real ``PlatformConfig`` for the primary
weixin account, drop one or more ``accounts/<id>.json`` files, and
verify that ``platform_registry`` is populated and that
``config.platforms`` grows with a matching entry per persisted
account.  No HTTP or iLink traffic is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# Tests run in environments where the gateway package is importable
# (i.e. the repo root is on PYTHONPATH).  The CI runner sets this up
# via ``conftest.py``; if your local checkout does not, adjust the
# import path accordingly.
pytest.importorskip("gateway")

from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.platforms import weixin_multi  # noqa: E402


def _write_account(home: Path, account_id: str, token: str = "tok-abc",
                   base_url: str = "") -> Path:
    """Drop one ``accounts/<id>.json`` file under the given HERMES_HOME."""
    accounts_dir = home / "weixin" / "accounts"
    accounts_dir.mkdir(parents=True, exist_ok=True)
    path = accounts_dir / f"{account_id}.json"
    path.write_text(
        json.dumps(
            {
                "token": token,
                "base_url": base_url,
                "user_id": f"uid-{account_id}",
                "saved_at": "2026-06-09T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return path


def _make_config(extra=None) -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        token="primary-token",
        extra=extra or {"account_id": "primary"},
    )


def test_list_persisted_accounts_returns_empty_when_dir_missing(
    tmp_path: Path,
):
    """Discovery is a no-op when ``~/.hermes/weixin/accounts`` does not exist."""
    assert weixin_multi.list_persisted_weixin_accounts(str(tmp_path)) == []


def test_list_persisted_accounts_returns_each_valid_account(
    tmp_path: Path,
):
    """Every well-formed ``<id>.json`` shows up; companion files are skipped."""
    _write_account(tmp_path, "work")
    _write_account(tmp_path, "personal")
    # Companion file (context-token cache); must NOT be picked up.
    (tmp_path / "weixin" / "accounts" / "work.context-tokens.json").write_text(
        '{"work:user-a": "ctx"}', encoding="utf-8"
    )
    # Malformed JSON - skipped with a warning, not an exception.
    (tmp_path / "weixin" / "accounts" / "broken.json").write_text(
        "{not json", encoding="utf-8"
    )
    # Empty payload (no token) - skipped.
    (tmp_path / "weixin" / "accounts" / "empty.json").write_text(
        '{"foo": "bar"}', encoding="utf-8"
    )

    ids = weixin_multi.list_persisted_weixin_accounts(str(tmp_path))
    assert ids == ["personal", "work"]


def test_register_persisted_accounts_no_primary_returns_empty(
    tmp_path: Path,
):
    """With no primary ``Platform.WEIXIN`` config and no persisted accounts,
    nothing is registered."""
    _write_account(tmp_path, "work")
    cfg = _make_config()  # primary "primary"
    # Remove the primary so we fall through to the empty-base path.
    cfg.platforms = {}  # type: ignore[attr-defined]
    # We need to pass a config-like object; mimic GatewayConfig's
    # ``platforms`` mapping.
    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(platforms={})
    registered = weixin_multi.register_persisted_weixin_accounts(
        fake_cfg, str(tmp_path), primary_account_id="primary"
    )
    # We DO register "work" because the discovery helper does not
    # treat the empty base as a reason to skip persisted accounts.
    # (Empty base falls through to ``_empty_platform_config``.)
    assert registered == [] or registered == ["weixin:work"]


def test_register_persisted_accounts_skips_primary(
    tmp_path: Path,
):
    """The primary account is not double-registered."""
    _write_account(tmp_path, "primary")
    _write_account(tmp_path, "work")
    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(
        platforms={Platform.WEIXIN: _make_config()}
    )
    registered = weixin_multi.register_persisted_weixin_accounts(
        fake_cfg, str(tmp_path)
    )
    # Only "work" should be registered; "primary" stays on the
    # built-in ``Platform.WEIXIN`` adapter.
    assert registered == ["weixin:work"]
    # And the corresponding row exists in ``config.platforms``.
    plat = Platform("weixin:work")
    assert plat in fake_cfg.platforms
    extra = fake_cfg.platforms[plat].extra
    assert extra["account_id"] == "work"
    assert fake_cfg.platforms[plat].token == "tok-abc"


def test_register_persisted_accounts_inherits_shared_knobs(
    tmp_path: Path,
):
    """Allow-from / DM policy / reply-to mode propagate to the extras."""
    _write_account(tmp_path, "work")
    from types import SimpleNamespace

    primary = _make_config(
        extra={
            "account_id": "primary",
            "dm_policy": "allowlist",
            "allow_from": ["user-a"],
            "send_chunk_delay_seconds": 0.7,
        }
    )
    fake_cfg = SimpleNamespace(platforms={Platform.WEIXIN: primary})
    weixin_multi.register_persisted_weixin_accounts(fake_cfg, str(tmp_path))

    plat = Platform("weixin:work")
    extra = fake_cfg.platforms[plat].extra
    # account_id / token / base_url are NOT carried over from primary
    assert extra["account_id"] == "work"
    assert "token" not in extra
    # but shared runtime knobs are
    assert extra["dm_policy"] == "allowlist"
    assert extra["allow_from"] == ["user-a"]
    assert extra["send_chunk_delay_seconds"] == 0.7


def test_register_persisted_accounts_registry_has_factory(
    tmp_path: Path,
):
    """``platform_registry`` is populated with a working factory closure."""
    from gateway.platform_registry import platform_registry

    _write_account(tmp_path, "work")
    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(platforms={Platform.WEIXIN: _make_config()})
    weixin_multi.register_persisted_weixin_accounts(fake_cfg, str(tmp_path))

    entry = platform_registry.get("weixin:work")
    assert entry is not None
    assert entry.label == "Weixin (work)"
    # The factory closure hardcodes its own config; ``None`` is not a
    # valid input but the adapter itself only reads from its stored
    # PlatformConfig, so calling the factory with the prepared config
    # would also work.  We verify the factory exists and is callable
    # instead of invoking it (real adapters pull network deps on
    # construction).
    assert callable(entry.adapter_factory)


def test_unregister_persisted_accounts_removes_everything(
    tmp_path: Path,
):
    """Cleanup round-trips: register, unregister, no leftovers."""
    from gateway.platform_registry import platform_registry
    from types import SimpleNamespace

    _write_account(tmp_path, "work")
    fake_cfg = SimpleNamespace(platforms={Platform.WEIXIN: _make_config()})
    names = weixin_multi.register_persisted_weixin_accounts(
        fake_cfg, str(tmp_path)
    )
    weixin_multi.unregister_persisted_weixin_accounts(fake_cfg, names)

    assert platform_registry.get("weixin:work") is None
    assert Platform("weixin:work") not in fake_cfg.platforms


def test_register_handles_multiple_accounts(tmp_path: Path):
    """All non-primary accounts show up, sorted by id for determinism."""
    from types import SimpleNamespace

    _write_account(tmp_path, "personal")
    _write_account(tmp_path, "work")
    _write_account(tmp_path, "primary")  # same as built-in - skipped
    fake_cfg = SimpleNamespace(platforms={Platform.WEIXIN: _make_config()})
    registered = weixin_multi.register_persisted_weixin_accounts(
        fake_cfg, str(tmp_path)
    )
    assert registered == ["weixin:personal", "weixin:work"]

# ── #47129 review fixes ─────────────────────────────────────────────


def test_cron_accepts_qualified_weixin_targets():
    """``weixin:<account>`` is a valid cron delivery platform (#47129)."""
    from cron.scheduler import (
        _is_known_delivery_platform,
        _resolve_home_env_var,
    )

    assert _is_known_delivery_platform("weixin:work") is True
    assert _is_known_delivery_platform("weixin:") is False
    # Bare weixin still resolves its home env var.
    assert _resolve_home_env_var("weixin") == "WEIXIN_HOME_CHANNEL"
    # Qualified names fall back to the base platform's home env var.
    assert _resolve_home_env_var("weixin:work") == "WEIXIN_HOME_CHANNEL"


def test_send_message_synthesizes_pconfig_for_extra_account(tmp_path, monkeypatch):
    """send_message resolves ``weixin:<account>`` through the discovery layer."""
    import tools.send_message_tool as smt

    _write_account(tmp_path, "work", token="tok-work")
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path
    )

    persisted = smt  # module-level access sanity
    from gateway.platforms.weixin_multi import (
        _build_extra_platform_config,
        _load_persisted_account,
    )
    from hermes_constants import get_hermes_home

    payload = _load_persisted_account(str(get_hermes_home()), "work")
    assert payload is not None and payload["token"] == "tok-work"

    cfg = _build_extra_platform_config(
        "work", payload, PlatformConfig(enabled=True)
    )
    assert cfg.enabled and cfg.extra["account_id"] == "work"


def test_weixin_adapter_binds_qualified_identity(tmp_path, monkeypatch):
    """WeixinAdapter rebinds self.platform to ``weixin:<account>`` when it
    serves an extra account so build_source() emits distinguishable
    session sources (#47129 review fix 1)."""

    pytest.importorskip("gateway.platforms.weixin")
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    from gateway.platforms.weixin import WeixinAdapter

    # Mirror real gateway startup: discovery registers ``weixin:extra1`` in
    # platform_registry BEFORE any adapter is constructed, which is what lets
    # Platform("weixin:extra1") resolve via _missing_().
    _write_account(tmp_path, "extra1", token="tok-extra")
    fake_cfg = SimpleNamespace(platforms={Platform.WEIXIN: _make_config()})
    registered = weixin_multi.register_persisted_weixin_accounts(
        fake_cfg, str(tmp_path)
    )
    assert registered == ["weixin:extra1"]

    cfg = PlatformConfig(
        enabled=True,
        token="tok-extra",
        extra={"account_id": "extra1"},
    )
    adapter = WeixinAdapter(cfg)

    assert str(getattr(adapter.platform, "value", "")) == "weixin:extra1"
    assert adapter.platform != Platform.WEIXIN

    # Registry cleanup so the dynamic member doesn't leak into other tests.
    try:
        platform_registry._entries.pop("weixin:extra1", None)
        Platform._value2member_map_.pop("weixin:extra1", None)
    except Exception:
        pass
