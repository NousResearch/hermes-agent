"""Regression: per-profile PairingStore creation in _start_secondary_profile_adapters.

``gateway/run.py`` referenced ``PairingStore`` at method scope in
``_start_secondary_profile_adapters`` while the class's only import was
method-local inside ``__init__`` — a ``NameError`` at runtime, silently
swallowed by the enclosing ``try/except``, so multiplexing gateways never
created per-profile pairing stores and authz pairing checks for secondary
profiles fell through to the global whitelist.

These tests drive the REAL method (bound onto a bare runner) with the
profile-enumeration and adapter-startup collaborators stubbed, and assert
the per-profile stores actually materialize.
"""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.run import GatewayRunner


def _bare_runner(multiplex: bool = True):
    runner = object.__new__(GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=multiplex)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()
    runner.pairing_stores = {}
    return runner


def test_secondary_profile_pairing_stores_created(tmp_path, monkeypatch):
    """The served-profiles loop must create a PairingStore per profile.

    Pre-fix this silently did nothing: the ``PairingStore(profile=name)``
    reference raised NameError inside the swallowed try/except.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
        ("coder", tmp_path / ".hermes" / "profiles" / "coder"),
    ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner._profile_adapters["coder"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    # Both the active profile and the served secondary get a store.
    assert "default" in runner.pairing_stores, (
        "active profile PairingStore missing — the NameError swallow is back"
    )
    assert runner.pairing_stores["default"] is runner.pairing_store
    assert "coder" in runner.pairing_stores, (
        "secondary profile PairingStore missing — the NameError swallow is back"
    )


def test_pairing_store_scoped_to_profile_dir(tmp_path, monkeypatch):
    """The created store must live under the profile's pairing directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
        ("ops", tmp_path / ".hermes" / "profiles" / "ops"),
    ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner._profile_adapters["ops"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    store = runner.pairing_stores["ops"]
    assert store.profile == "ops"
    assert "profiles/ops/platforms/pairing" in str(store._dir).replace("\\", "/"), (
        f"store not profile-scoped: {store._dir}"
    )


def test_routed_pairing_grant_mirror_stays_in_profile_scope(tmp_path, monkeypatch):
    """A /pair grant mirrored under a routed profile scope must update THAT
    profile's .env and installed scope, never the shared os.environ (#88441,
    #77490). Outside multiplex the legacy os.environ publish is unchanged."""
    import os

    from agent import secret_scope as ss
    from gateway.pairing import _sync_allowlist_add
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import save_env_value

    root = tmp_path / ".hermes"
    prof = root / "profiles" / "b"
    prof.mkdir(parents=True)
    (root / ".env").write_text("DISCORD_ALLOWED_USERS=default-admin\n")
    (prof / ".env").write_text("DISCORD_ALLOWED_USERS=b-admin\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "default-admin")

    was_active = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    try:
        with _profile_runtime_scope(prof):
            _sync_allowlist_add("discord", "111")
            assert ss.get_secret("DISCORD_ALLOWED_USERS") == "b-admin,111"
    finally:
        ss.set_multiplex_active(was_active)

    assert (prof / ".env").read_text().strip() == "DISCORD_ALLOWED_USERS=b-admin,111"
    assert (root / ".env").read_text().strip() == "DISCORD_ALLOWED_USERS=default-admin"
    assert os.environ["DISCORD_ALLOWED_USERS"] == "default-admin"

    # Single-profile: no multiplex -> save still publishes to the process env.
    save_env_value("DISCORD_ALLOWED_USERS", "default-admin,222")
    assert os.environ["DISCORD_ALLOWED_USERS"] == "default-admin,222"
