"""The remote secret store against a stub secrets plane on loopback (real HTTP, ``stub_secrets_plane``).

Covers: boot fails closed (outage with bounded retry, refusal without), the one-time import and
its 409 rule (the plane wins; local entries move to ``*.unimported``) and its crash idempotency,
writes from ``save_env_value`` / ``remove_env_value`` / the auth store landing on the plane and never
on disk, bootstrap entries never sent, and no value in any log line or error. Restarts on a change
made elsewhere: ``test_remote_restart_on_change.py``.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from plugins.config_backends.remote import credentials as cred_mod
from plugins.secret_store import remote as remote_pkg
from plugins.secret_store.remote import backend as backend_mod

from .stub_secrets_plane import StubSecretsPlane, remote_env

MARK = "remote-marker-5be2"


@pytest.fixture
def plane(tmp_path, monkeypatch):
    """A stub plane and a profile home (``<root>/profiles/default``) selected as HERMES_HOME.
    ``plane.restarts`` counts the graceful restarts the process was asked for."""
    root = tmp_path / "root"
    home = root / "profiles" / "default"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubSecretsPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        routing.reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        p.restarts = []
        routing.set_restart_handler(lambda: p.restarts.append("restart"))
        yield p
        routing.set_restart_handler(None)
        remote_pkg._reset_for_tests()
        routing.reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def _boot(home):
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home)


def _sent(plane) -> str:
    return json.dumps([r["body"] for r in plane.patches()])


def test_the_remote_store_is_selected_and_refuses_file_tooling(plane):
    provider = routing.routed_provider()
    assert provider is remote_pkg.get_provider() and not provider.supports_file_tooling()
    from hermes_cli.config_backend import require_profile_file_tooling
    with pytest.raises(routing.SecretStoreUnavailable):
        require_profile_file_tooling("`hermes backup`")


def test_boot_fails_closed_on_an_outage_after_bounded_retries(plane, caplog):
    plane.fail_status = 503
    (plane.home / ".env").write_text(f"OPENAI_API_KEY={MARK}\n", encoding="utf-8")
    with pytest.raises(routing.SecretStoreUnavailable) as refused:
        _boot(plane.home)
    assert "503" in str(refused.value.code) and MARK not in str(refused.value.code)
    assert len([r for r in plane.requests if r["method"] == "GET"]) == 3
    assert os.environ.get("OPENAI_API_KEY") != MARK and MARK not in caplog.text


def test_boot_does_not_retry_a_refusal(plane, monkeypatch):
    monkeypatch.setenv("HERMES_CONFIG_INSTANCE_ID", "inst-unknown")
    with pytest.raises(routing.SecretStoreUnavailable) as refused:
        _boot(plane.home)
    assert "403" in str(refused.value.code)
    assert len(plane.requests) == 1


def test_boot_requires_the_instance_id(plane, monkeypatch):
    monkeypatch.delenv("HERMES_CONFIG_INSTANCE_ID")
    with pytest.raises(routing.SecretStoreUnavailable, match="HERMES_CONFIG_INSTANCE_ID"):
        _boot(plane.home)
    assert plane.requests == []


def test_the_first_boot_imports_the_local_secrets_once_and_keeps_only_bootstrap_entries(plane, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    home = plane.home
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (home / ".env").write_text(
        "# my keys\nOPENAI_API_KEY=" + MARK + "\nTERMINAL_ENV=local\n"
        + "".join(f"{k}={v}\n" for k, v in remote_env(plane).items()), encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {
        "nous": {"access_token": "boot-nous"}, "openai-codex": {"tokens": {"access_token": MARK}}},
        "active_provider": "openai-codex"}), encoding="utf-8")
    _boot(home)
    prof = plane.profile("default")
    assert prof["env"]["OPENAI_API_KEY"][0] == MARK
    assert prof["auth"]["openai-codex"][0] == {"tokens": {"access_token": MARK}}
    assert prof["auth"][routing.STORE_META_RECORD][0] == {"active_provider": "openai-codex"}
    patch = plane.patches()[0]["body"]
    assert patch["expectedProfileVersion"] == 0
    assert not set(patch["set"]) & set(remote_env(plane)) and "TERMINAL_ENV" not in patch["set"]
    assert "nous" not in patch["auth"]
    env_text = (home / ".env").read_text(encoding="utf-8-sig")
    assert MARK not in env_text and "# my keys" in env_text and "GATEWAY_RELAY_IDP_CLIENT_ID=" in env_text
    assert "TERMINAL_ENV=local" in env_text
    on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8-sig"))
    assert on_disk["providers"] == {"nous": {"access_token": "boot-nous"}} and MARK not in json.dumps(on_disk)
    assert os.environ["OPENAI_API_KEY"] == MARK
    assert not (home / ".secrets-import.json").exists()
    assert MARK not in caplog.text
    # A second boot of a fresh process imports nothing again.
    remote_pkg._reset_for_tests()
    routing.reset_for_tests()
    _boot(home)
    assert len(plane.patches()) == 1


def test_when_the_plane_already_has_the_profile_it_wins_and_local_entries_are_set_aside(plane, monkeypatch):
    home = plane.home
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "from-the-plane"})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (home / ".env").write_text(f"OPENAI_API_KEY={MARK}\nOTHER_KEY=also-local\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"providers": {"openai-codex": {"t": MARK}}}), encoding="utf-8")
    _boot(home)
    assert plane.patches() == []
    assert plane.profile("default")["env"]["OPENAI_API_KEY"][0] == "from-the-plane"
    assert os.environ["OPENAI_API_KEY"] == "from-the-plane"
    unimported_env, unimported_auth = home / ".env.unimported", home / "auth.unimported.json"
    assert f"OPENAI_API_KEY={MARK}" in unimported_env.read_text(encoding="utf-8-sig")
    assert json.loads(unimported_auth.read_text(encoding="utf-8-sig"))["providers"]["openai-codex"] == {"t": MARK}
    for path in (unimported_env, unimported_auth):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert MARK not in (home / ".env").read_text(encoding="utf-8-sig")
    assert MARK not in (home / "auth.json").read_text(encoding="utf-8-sig")
    warnings = routing.routed_provider().doctor_warnings(home)
    assert warnings and ".env.unimported" in warnings[0][0] and MARK not in json.dumps(warnings)


def test_an_import_the_plane_answers_409_to_moves_the_entries_aside(plane, monkeypatch):
    home = plane.home
    (home / ".env").write_text(f"OPENAI_API_KEY={MARK}\n", encoding="utf-8")
    plane.conflict_next_patch = 1   # another writer creates the profile between our GET and PATCH
    _boot(home)
    assert len(plane.patches()) == 1
    assert "OPENAI_API_KEY" not in plane.profile("default")["env"]
    assert f"OPENAI_API_KEY={MARK}" in (home / ".env.unimported").read_text(encoding="utf-8-sig")


def test_an_import_interrupted_after_the_plane_took_it_finishes_without_setting_anything_aside(plane):
    home = plane.home
    (home / ".secrets-import.json").write_text(json.dumps({"env": {"OPENAI_API_KEY": MARK}, "auth": {}}),
                                               encoding="utf-8")
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": MARK})
    _boot(home)
    assert not (home / ".secrets-import.json").exists()
    assert not (home / ".env.unimported").exists()
    assert plane.patches() == []


def test_a_failed_poll_keeps_the_loaded_secrets(plane, caplog):
    plane.set_level(plane.tenant, env={"TENANT_KEY": MARK})
    _boot(plane.home)
    plane.fail_status = 503
    routing.routed_provider().poll_all()
    assert routing.read_profile_env(plane.home / ".env")["TENANT_KEY"] == MARK
    assert MARK not in caplog.text


def test_save_env_value_lands_on_the_plane_and_never_on_disk(plane, monkeypatch):
    _boot(plane.home)
    from hermes_cli.config import remove_env_value, save_env_value, save_env_value_secure
    save_env_value("OPENAI_API_KEY", MARK)
    save_env_value_secure("ANTHROPIC_API_KEY", MARK + "-2")
    prof = plane.profile("default")
    assert prof["env"]["OPENAI_API_KEY"][0] == MARK and prof["env"]["ANTHROPIC_API_KEY"][0] == MARK + "-2"
    assert not (plane.home / ".env").exists() or MARK not in (plane.home / ".env").read_text(encoding="utf-8-sig")
    assert os.environ["OPENAI_API_KEY"] == MARK
    assert remove_env_value("OPENAI_API_KEY") is True
    assert "OPENAI_API_KEY" not in plane.profile("default")["env"] and "OPENAI_API_KEY" not in os.environ
    assert all("expectedProfileVersion" in r["body"] for r in plane.patches())


def test_a_bootstrap_name_is_written_locally_and_never_sent(plane):
    _boot(plane.home)
    from hermes_cli.config import save_env_value
    save_env_value("GATEWAY_RELAY_IDP_SCOPE", "agents")
    save_env_value("HERMES_SECRETS_REMOTE_URL", plane.url)
    assert plane.patches() == []
    assert "GATEWAY_RELAY_IDP_SCOPE=agents" in (plane.home / ".env").read_text(encoding="utf-8-sig")


def test_an_auth_store_save_is_a_per_record_cas_write(plane):
    _boot(plane.home)
    from hermes_cli import auth
    with auth._auth_store_lock():
        st = auth._load_auth_store()
        st["providers"]["openai-codex"] = {"tokens": {"access_token": MARK}}
        st["providers"]["nous"] = {"access_token": "boot-nous"}
        auth._save_auth_store(st)
    body = plane.patches()[-1]["body"]
    assert "expectedProfileVersion" not in body and body["auth"]["openai-codex"]["expectedVersion"] == 0
    assert "nous" not in body["auth"]
    assert plane.profile("default")["auth"]["openai-codex"][0] == {"tokens": {"access_token": MARK}}
    assert MARK not in (plane.home / "auth.json").read_text(encoding="utf-8-sig")
    assert auth._load_auth_store()["providers"]["openai-codex"]["tokens"]["access_token"] == MARK


def test_bootstrap_entries_are_never_sent_anywhere(plane, monkeypatch):
    home = plane.home
    (home / ".env").write_text("".join(f"{k}={v}\n" for k, v in remote_env(plane).items())
                               + f"HERMES_SHARED_AUTH_DIR={home}\nOPENAI_API_KEY={MARK}\n", encoding="utf-8")
    _boot(home)
    from hermes_cli.config import save_env_value
    save_env_value("SOME_KEY", "v")
    provider = routing.routed_provider()
    sent = _sent(plane)
    for name in provider.bootstrap_env_names():
        assert f'"{name}"' not in sent
    assert '"nous"' not in sent and plane.url not in sent


def test_no_value_reaches_a_log_or_an_error_on_any_failure(plane, caplog):
    caplog.set_level(logging.DEBUG)
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": MARK})
    _boot(plane.home)
    plane.fail_status = 500
    from hermes_cli.config import save_env_value
    from hermes_secret_port import SecretWriteFailed
    with pytest.raises(SecretWriteFailed) as failed:
        save_env_value("NEW_KEY", MARK)
    routing.routed_provider().poll_all()
    assert MARK not in str(failed.value) and MARK not in repr(failed.value)
    assert MARK not in caplog.text
    describe = routing.routed_provider().describe(plane.home)
    assert MARK not in describe and "last poll failed" in describe


def test_a_routed_profile_boots_on_first_use_with_its_own_view(plane, monkeypatch):
    other = plane.home.parent / "work"
    other.mkdir()
    plane.set_level(plane.profile("work"), env={"WORK_KEY": MARK})
    _boot(plane.home)
    assert routing.read_profile_env(other / ".env") == {"WORK_KEY": MARK}
    assert "WORK_KEY" not in os.environ
    assert any(r["profile"] == "work" for r in plane.requests)


def test_a_home_that_is_not_a_profile_is_refused(plane, tmp_path):
    loose = tmp_path / "loose"
    loose.mkdir()
    with pytest.raises(routing.SecretStoreUnavailable, match="not a Hermes profile home"):
        routing.read_profile_env(loose / ".env")
