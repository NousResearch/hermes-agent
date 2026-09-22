"""Conformance coverage for mutation-free read-only diagnostics (#116991)."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from tests.diagnostic_purity import assert_diagnostic_pure


def test_diagnostic_purity_oracle_names_exact_mutation(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    config = profile / "config.yaml"
    config.write_text("model: original\n", encoding="utf-8")

    with pytest.raises(AssertionError, match=r"path:profile/config\.yaml"):
        with assert_diagnostic_pure({"profile": profile}):
            config.write_text("model: changed\n", encoding="utf-8")


def test_status_and_doctor_are_pure_across_profiles_and_live_resources(monkeypatch, tmp_path):
    import agent.credential_pool as credential_pool
    import hermes_cli.auth as auth
    import hermes_cli.config as config
    import hermes_cli.doctor as doctor
    import hermes_cli.doctor_state as doctor_state
    import hermes_cli.status as status
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB

    homes = {name: tmp_path / name for name in ("profile-a", "profile-b")}
    holders: dict[str, sqlite3.Connection] = {}
    for name, home in homes.items():
        home.mkdir()
        for child in ("cron", "logs", "sessions", "skills"):
            (home / child).mkdir()
        (home / "config.yaml").write_text("model:\n  provider: auto\n", encoding="utf-8")
        (home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {"tokens": {"access_token": f"expired-{name}", "refresh_token": "unused"}},
                "xai-oauth": {"tokens": {"access_token": f"expired-{name}", "refresh_token": "unused"}},
            },
        }), encoding="utf-8")
        db = SessionDB(db_path=home / "state.db")
        db.create_session(f"session-{name}", source="cli")
        db.close()
        holder = sqlite3.connect(home / "state.db")
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("CREATE TABLE IF NOT EXISTS diagnostic_sentinel(value INTEGER)")
        holder.execute("INSERT INTO diagnostic_sentinel VALUES (1)")
        holder.commit()
        holders[name] = holder

    qwen = tmp_path / "qwen-oauth.json"
    qwen.write_text(json.dumps({
        "access_token": "expired-qwen", "refresh_token": "must-not-be-used", "expiry_date": 1,
    }), encoding="utf-8")
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: qwen)
    monkeypatch.setattr(credential_pool, "load_pool", lambda _provider: None)

    def forbidden_refresh(*_args, **_kwargs):
        raise AssertionError("diagnostic attempted a credential refresh")

    monkeypatch.setattr(
        config, "ensure_hermes_home",
        lambda: (_ for _ in ()).throw(AssertionError("diagnostic attempted profile bootstrap")),
    )
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", forbidden_refresh)
    monkeypatch.setattr(auth, "resolve_xai_oauth_runtime_credentials", forbidden_refresh)
    monkeypatch.setattr(auth, "_refresh_qwen_cli_tokens", forbidden_refresh)
    monkeypatch.setattr(
        status,
        "_SECTIONS",
        (status._render_auth_providers, status._render_sessions),
    )
    monkeypatch.setattr(
        doctor,
        "DOCTOR_CHECKS",
        (("Auth Providers", doctor._check_auth_providers), (None, doctor_state._check_state_db)),
    )

    live_resource = {"session_generation": 4}
    resources = {
        "mcp.session_generation": lambda: live_resource["session_generation"],
        **{
            f"{name}.session_rows": lambda holder=holder: holder.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            for name, holder in holders.items()
        },
    }

    try:
        with assert_diagnostic_pure(
            {**homes, "qwen-credentials": qwen},
            resources=resources,
        ):
            for name in ("profile-a", "profile-b", "profile-a"):
                home = homes[name]
                token = set_hermes_home_override(home)
                monkeypatch.setattr(doctor, "HERMES_HOME", home)
                monkeypatch.setattr(doctor, "_DHH", str(home))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        status.show_status(Namespace(deep=False, all=False))
                        doctor.run_doctor(Namespace(fix=False, live=False, ack=None))
                finally:
                    reset_hermes_home_override(token)
            assert live_resource["session_generation"] == 4
    finally:
        for holder in holders.values():
            holder.close()
