"""Proof-of-concept for #88441: multiplexer pairing updates must remain profile-scoped."""

import os

from agent import secret_scope as ss


def test_pairing_allowlist_updates_are_profile_scoped(tmp_path, monkeypatch):
    from gateway.pairing import PairingStore, _read_allowlist_env
    from gateway.run import _profile_runtime_scope

    default_profile_home = tmp_path / "default-profile"
    secondary_profile_home = tmp_path / "secondary-profile"
    default_profile_home.mkdir()
    secondary_profile_home.mkdir()

    default_env = default_profile_home / ".env"
    secondary_env = secondary_profile_home / ".env"
    default_env.write_text("TELEGRAM_ALLOWED_USERS=default-owner\n", encoding="utf-8")
    secondary_env.write_text(
        "TELEGRAM_ALLOWED_USERS=secondary-owner\n", encoding="utf-8"
    )

    user_id = "user-42"
    platform = "telegram"
    process_token = "process-owner"
    prior_multiplex = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    monkeypatch.setenv("HERMES_HOME", str(default_profile_home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", process_token)
    baseline_default = default_env.read_text(encoding="utf-8")

    try:
        with _profile_runtime_scope(secondary_profile_home):
            store = PairingStore()
            code = store.generate_code(platform, user_id, "Second User")
            assert code is not None
            approval = store.approve_code(platform, code)
            assert approval and approval["user_id"] == user_id
            assert _read_allowlist_env("TELEGRAM_ALLOWED_USERS") == (
                f"secondary-owner,{user_id}"
            )
            assert PairingStore().is_approved(platform, user_id) is True

            assert store.revoke(platform, user_id) is True
            assert _read_allowlist_env("TELEGRAM_ALLOWED_USERS") == "secondary-owner"

        assert os.environ.get("TELEGRAM_ALLOWED_USERS") == process_token
        with _profile_runtime_scope(default_profile_home):
            assert _read_allowlist_env("TELEGRAM_ALLOWED_USERS") == "default-owner"
            assert PairingStore().is_approved(platform, user_id) is False
    finally:
        ss.set_multiplex_active(prior_multiplex)

    assert default_env.read_text(encoding="utf-8") == baseline_default
    assert secondary_env.read_text(encoding="utf-8") == "TELEGRAM_ALLOWED_USERS=secondary-owner\n"
    assert os.environ.get("TELEGRAM_ALLOWED_USERS") == process_token
